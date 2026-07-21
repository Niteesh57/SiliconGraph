const { execFile } = require("child_process");
const { promisify } = require("util");

const execFileAsync = promisify(execFile);
const adbPath = process.env.ADB_PATH || "adb";

async function runAdb(args, timeout = 12_000) {
  try {
    const { stdout, stderr } = await execFileAsync(adbPath, args, {
      windowsHide: true,
      timeout,
      maxBuffer: 1_024 * 1_024
    });
    if (stderr && !stdout) throw new Error(stderr.trim());
    return stdout.trim();
  } catch (error) {
    if (error.code === "ENOENT") {
      throw new Error("ADB was not found. Install Android Platform Tools or set ADB_PATH.");
    }
    throw new Error(error.stderr?.trim() || error.message);
  }
}

function parseDeviceList(output) {
  return output.split(/\r?\n/)
    .slice(1)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const [serial, state, ...attributes] = line.split(/\s+/);
      return {
        serial,
        state,
        attributes: Object.fromEntries(attributes
          .filter((item) => item.includes(":"))
          .map((item) => {
            const [key, ...value] = item.split(":");
            return [key, value.join(":")];
          }))
      };
    });
}

function parseGetProp(output) {
  const properties = {};
  for (const line of output.split(/\r?\n/)) {
    const match = line.match(/^\[([^\]]+)\]:\s*\[(.*)\]$/);
    if (match) properties[match[1]] = match[2];
  }
  return properties;
}

function parseBattery(output) {
  const level = output.match(/^\s*level:\s*(\d+)/m)?.[1];
  const status = output.match(/^\s*status:\s*(\d+)/m)?.[1];
  return {
    level: level ? Number(level) : null,
    status: status || null
  };
}

function parseMemTotal(output) {
  const totalKb = output.match(/^MemTotal:\s*(\d+)\s*kB/m)?.[1];
  return totalKb ? Math.round(Number(totalKb) / 1024) : null;
}

function profileFor(properties) {
  const soc = [
    properties["ro.soc.model"],
    properties["ro.board.platform"],
    properties["ro.hardware"],
    properties["ro.soc.manufacturer"]
  ].filter(Boolean).join(" ").toLowerCase();

  if (/(sm8650|snapdragon 8 gen 3)/.test(soc)) return "snapdragon_8_gen3";
  if (/(sm8750|snapdragon 8 elite)/.test(soc)) return "snapdragon_8_elite";
  if (/(mt6989|dimensity 9300)/.test(soc)) return "dimensity_9300";
  if (/(s5e9945|exynos 2400)/.test(soc)) return "exynos_2400";
  return "generic_arm64";
}

async function inspectDevice(entry) {
  const serialArgs = ["-s", entry.serial, "shell"];
  const [propertyOutput, coreOutput, memoryOutput, batteryOutput] = await Promise.all([
    runAdb([...serialArgs, "getprop"]),
    runAdb([...serialArgs, "nproc"]),
    runAdb([...serialArgs, "cat", "/proc/meminfo"]),
    runAdb([...serialArgs, "dumpsys", "battery"])
  ]);
  const properties = parseGetProp(propertyOutput);
  const manufacturer = properties["ro.product.manufacturer"] || "Unknown manufacturer";
  const model = properties["ro.product.model"] || entry.attributes.model || "Unknown device";

  return {
    serial: entry.serial,
    status: entry.state,
    name: `${manufacturer} ${model}`.trim(),
    manufacturer,
    model,
    androidVersion: properties["ro.build.version.release"] || "Unknown",
    apiLevel: properties["ro.build.version.sdk"] || "Unknown",
    abi: properties["ro.product.cpu.abi"] || "Unknown",
    socManufacturer: properties["ro.soc.manufacturer"] || "Unknown",
    socModel: properties["ro.soc.model"] || properties["ro.board.platform"] || "Unknown",
    boardPlatform: properties["ro.board.platform"] || "Unknown",
    coreCount: Number.parseInt(coreOutput, 10) || null,
    memoryMb: parseMemTotal(memoryOutput),
    battery: parseBattery(batteryOutput),
    suggestedProfile: profileFor(properties)
  };
}

async function listDevices() {
  const entries = parseDeviceList(await runAdb(["devices", "-l"]));
  const ready = entries.filter((entry) => entry.state === "device");
  const unavailable = entries.filter((entry) => entry.state !== "device").map((entry) => ({
    serial: entry.serial,
    status: entry.state,
    name: entry.attributes.model || entry.serial,
    unavailable: true
  }));
  const inspected = await Promise.all(ready.map(inspectDevice));
  return [...inspected, ...unavailable];
}

module.exports = { listDevices, parseDeviceList, profileFor };
