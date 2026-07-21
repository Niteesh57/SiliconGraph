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
  const temperature = output.match(/^\s*temperature:\s*(-?\d+)/m)?.[1];
  const voltage = output.match(/^\s*voltage:\s*(\d+)/m)?.[1];
  return {
    level: level ? Number(level) : null,
    status: status || null,
    temperatureC: temperature ? Number(temperature) / 10 : null,
    voltageMv: voltage ? Number(voltage) : null
  };
}

function parseMemInfo(output) {
  const valueMb = (name) => {
    const value = output.match(new RegExp(`^${name}:\\s*(\\d+)\\s*kB`, "m"))?.[1];
    return value ? Math.round(Number(value) / 1024) : null;
  };
  return {
    totalMb: valueMb("MemTotal"),
    availableMb: valueMb("MemAvailable"),
    swapTotalMb: valueMb("SwapTotal"),
    swapFreeMb: valueMb("SwapFree")
  };
}

function parseDataStorage(output) {
  const line = output.split(/\r?\n/).find((item) => /\s\/data(?:\/|$)/.test(item));
  if (!line) return { totalMb: null, availableMb: null };
  const columns = line.trim().split(/\s+/);
  return {
    totalMb: columns[1] ? Math.round(Number(columns[1]) / 1024) : null,
    availableMb: columns[3] ? Math.round(Number(columns[3]) / 1024) : null
  };
}

function parseThermals(output) {
  const status = output.match(/Thermal Status:\s*(\d+)/)?.[1];
  const currentTemperatures = output.match(/Current temperatures from HAL:\s*([\s\S]*?)(?=\n(?:Current cooling devices|Temperature static thresholds)|$)/)?.[1] || output;
  const temperatures = [];
  for (const match of currentTemperatures.matchAll(/Temperature\{mValue=([-\d.]+), mType=(\d+), mName=([^,]+), mStatus=(\d+)\}/g)) {
    temperatures.push({ valueC: Number(match[1]), type: Number(match[2]), name: match[3], status: Number(match[4]) });
  }
  const firstNamed = (pattern) => temperatures.find((item) => pattern.test(item.name))?.valueC ?? null;
  const cpuTemperatures = temperatures.filter((item) => /^cpu/i.test(item.name)).map((item) => item.valueC);
  return {
    status: status ? Number(status) : null,
    maxCpuC: cpuTemperatures.length ? Math.max(...cpuTemperatures) : null,
    gpuC: firstNamed(/^gpu/i),
    socC: firstNamed(/^soc/i),
    skinC: firstNamed(/^skin/i)
  };
}

function parseCpuParts(output) {
  const counts = new Map();
  for (const match of output.matchAll(/^CPU part\s*:\s*(\S+)/gm)) {
    counts.set(match[1], (counts.get(match[1]) || 0) + 1);
  }
  return [...counts.entries()].map(([part, cores]) => ({ part, cores }));
}

function expandCpuRange(value) {
  const cores = new Set();
  for (const part of String(value || "").trim().split(",")) {
    const match = part.match(/^(\d+)(?:-(\d+))?$/);
    if (!match) continue;
    const start = Number(match[1]);
    const end = Number(match[2] || match[1]);
    for (let core = start; core <= end; core += 1) cores.add(core);
  }
  return [...cores].sort((left, right) => left - right);
}

function parseCpuFrequencies(output) {
  return output.split(/\r?\n/).map((line) => {
    const match = line.match(/^(\d+):(\d+)$/);
    return match ? { core: Number(match[1]), mhz: Math.round(Number(match[2]) / 1000) } : null;
  }).filter(Boolean);
}

function parseCaches(output) {
  return output.split(/\r?\n/).map((line) => {
    const match = line.match(/^(\d+):([^:]+):(\d+)([KMG])$/i);
    if (!match) return null;
    const multiplier = { K: 1, M: 1024, G: 1024 * 1024 }[match[4].toUpperCase()];
    return { level: Number(match[1]), type: match[2], sizeKb: Number(match[3]) * multiplier };
  }).filter(Boolean);
}

function topologyFromFrequencies(physicalCores, frequencies) {
  const grouped = new Map();
  for (const frequency of frequencies) {
    const group = grouped.get(frequency.mhz) || [];
    group.push(frequency.core);
    grouped.set(frequency.mhz, group);
  }
  const clusters = [...grouped.entries()]
    .map(([mhz, cores]) => ({ mhz: Number(mhz), cores: cores.sort((left, right) => left - right) }))
    .sort((left, right) => left.mhz - right.mhz);
  const fallbackCores = physicalCores.length || frequencies.length || 1;
  const little = clusters[0] || { mhz: 0, cores: [] };
  const mid = clusters.length > 2 ? clusters[1] : { mhz: 0, cores: [] };
  const big = clusters.length > 1 ? clusters.at(-1) : { mhz: little.mhz, cores: little.cores };
  return {
    clusters,
    num_big_cores: big.cores.length || fallbackCores,
    num_mid_cores: mid.cores.length,
    num_little_cores: clusters.length > 1 ? little.cores.length : 0,
    big_core_freq_mhz: big.mhz,
    mid_core_freq_mhz: mid.mhz,
    little_core_freq_mhz: little.mhz
  };
}

function cpuFeatures(cpuInfo, abi) {
  const featureText = `${cpuInfo}\n${abi}`.toLowerCase();
  return {
    has_neon: /asimd|neon|arm64/.test(featureText),
    has_dotprod: /asimddp|dotprod/.test(featureText),
    has_i8mm: /i8mm/.test(featureText),
    has_sve: /\bsve\b/.test(featureText),
    has_sve2: /sve2/.test(featureText)
  };
}

function cacheSize(caches, level) {
  return Math.max(0, ...caches.filter((cache) => cache.level === level).map((cache) => cache.sizeKb));
}

function normalizedVendor(value) {
  const vendor = String(value || "").toLowerCase();
  if (/qualcomm|qcom/.test(vendor)) return "qualcomm";
  if (/mediatek|mtk/.test(vendor)) return "mediatek";
  if (/samsung|exynos/.test(vendor)) return "samsung";
  return vendor || "generic";
}

function buildDynamicProfile(device) {
  const topology = device.cpuTopology;
  const gpuRenderer = device.gpu?.renderer || "";
  const isQualcomm = normalizedVendor(device.socManufacturer) === "qualcomm" || /^sm\d+/i.test(device.socModel);
  return {
    schema_version: "1.0",
    profile_id: `detected_${device.serial.replace(/[^a-zA-Z0-9_-]+/g, "_")}`,
    source: "adb_live_probe",
    detected_at: new Date().toISOString(),
    // The C++ cost model uses this stable enum key while the remaining fields
    // describe this exact device and override the generic profile values.
    soc_id: "generic_arm64",
    name: `Detected ${device.name} (${device.socModel})`,
    vendor: normalizedVendor(device.socManufacturer),
    device: {
      manufacturer: device.manufacturer,
      model: device.model,
      android_version: device.androidVersion,
      api_level: device.apiLevel,
      abi: device.abi,
      soc_model: device.socModel,
      board_platform: device.boardPlatform,
      build_fingerprint: device.buildFingerprint,
      security_patch: device.securityPatch,
      kernel_version: device.kernelVersion
    },
    l1_cache_kb: cacheSize(device.caches, 1),
    l2_cache_kb: cacheSize(device.caches, 2),
    l3_cache_kb: cacheSize(device.caches, 3),
    // Android does not expose memory bandwidth through a stable unprivileged
    // API. The compiler uses this conservative fallback until the benchmark
    // harness writes a measured value for this profile.
    ram_bandwidth_gbps: 25,
    measurement_status: { ram_bandwidth_gbps: "estimated_conservative" },
    cpu: {
      ...topology,
      ...device.cpuFeatures,
      physical_core_count: device.physicalCoreCount,
      online_core_count: device.onlineCoreCount,
      adb_visible_core_count: device.adbVisibleCoreCount,
      cpu_parts: device.cpuParts
    },
    gpu: {
      present: Boolean(gpuRenderer || device.gpu?.hardware),
      name: gpuRenderer || device.gpu?.hardware || "Detected GPU",
      supports_vulkan: Boolean(device.gpu?.vulkanHardware),
      supports_opencl: false,
      supports_metal: false
    },
    // We report hardware discovery separately. `present` stays false until a
    // validated QNN/other delegate is installed, preventing unsafe NPU choice.
    npu: { present: false },
    detected_accelerators: {
      npu_hardware: isQualcomm ? "Qualcomm Hexagon / HTP (delegate not configured)" : "Unknown",
      gpu_renderer: gpuRenderer || "Unknown"
    },
    runtime_state: {
      memory_total_mb: device.memory?.totalMb ?? null,
      memory_available_mb: device.memory?.availableMb ?? null,
      storage_total_mb: device.storage?.totalMb ?? null,
      storage_available_mb: device.storage?.availableMb ?? null,
      battery_level_pct: device.battery?.level ?? null,
      thermal_status: device.thermals?.status ?? null
    },
    unavailable_fields: {
      cache_topology: device.caches.length ? null : "Blocked by this Android build's unprivileged sysfs permissions",
      memory_bandwidth: "Requires an on-device benchmark; Android does not expose a standard hardware value"
    },
    supported_dtypes: {
      cpu: ["f32", "f16", "i8", "u8"],
      gpu: [],
      npu: [],
      dsp: []
    },
    cost_table: []
  };
}

async function inspectDevice(entry) {
  const serialArgs = ["-s", entry.serial, "shell"];
  // `adb shell <command>` joins the remaining argument into one remote shell
  // command. Passing `sh -c` as separate arguments loses the script quoting on
  // Windows and makes feature probes silently return no data.
  const shellScript = (script) => runAdb([...serialArgs, script]).catch(() => "");
  const [propertyOutput, coreOutput, memoryOutput, batteryOutput, cpuInfo, cpuPresent, cpuOnline, frequencyOutput, cacheOutput, surfaceFlingerOutput, dataStorageOutput, securityPatch, kernelVersion, thermalOutput] = await Promise.all([
    runAdb([...serialArgs, "getprop"]),
    runAdb([...serialArgs, "nproc"]),
    runAdb([...serialArgs, "cat", "/proc/meminfo"]),
    runAdb([...serialArgs, "dumpsys", "battery"]),
    shellScript("cat /proc/cpuinfo"),
    shellScript("cat /sys/devices/system/cpu/present"),
    shellScript("cat /sys/devices/system/cpu/online"),
    shellScript("for d in /sys/devices/system/cpu/cpu[0-9]*; do f=$d/cpufreq/cpuinfo_max_freq; [ -r $f ] && echo ${d##*cpu}:$(cat $f); done"),
    shellScript("for d in /sys/devices/system/cpu/cpu0/cache/index*; do [ -r $d/level ] && [ -r $d/type ] && [ -r $d/size ] && echo $(cat $d/level):$(cat $d/type):$(cat $d/size); done"),
    shellScript("dumpsys SurfaceFlinger"),
    shellScript("df /data"),
    shellScript("getprop ro.build.version.security_patch"),
    shellScript("uname -r"),
    shellScript("dumpsys thermalservice")
  ]);
  const properties = parseGetProp(propertyOutput);
  const manufacturer = properties["ro.product.manufacturer"] || "Unknown manufacturer";
  const model = properties["ro.product.model"] || entry.attributes.model || "Unknown device";
  const abi = properties["ro.product.cpu.abi"] || "Unknown";
  const physicalCores = expandCpuRange(cpuPresent);
  const onlineCores = expandCpuRange(cpuOnline);
  const frequencies = parseCpuFrequencies(frequencyOutput);
  const renderer = surfaceFlingerOutput.match(/GLES:\s*(.+)/i)?.[1]?.trim() || "";

  const device = {
    serial: entry.serial,
    status: entry.state,
    name: `${manufacturer} ${model}`.trim(),
    manufacturer,
    model,
    androidVersion: properties["ro.build.version.release"] || "Unknown",
    apiLevel: properties["ro.build.version.sdk"] || "Unknown",
    abi,
    socManufacturer: properties["ro.soc.manufacturer"] || properties["ro.hardware"] || "Unknown",
    socModel: properties["ro.soc.model"] || properties["ro.board.platform"] || "Unknown",
    boardPlatform: properties["ro.board.platform"] || "Unknown",
    coreCount: physicalCores.length || Number.parseInt(coreOutput, 10) || null,
    physicalCoreCount: physicalCores.length || null,
    onlineCoreCount: onlineCores.length || null,
    adbVisibleCoreCount: Number.parseInt(coreOutput, 10) || null,
    memory: parseMemInfo(memoryOutput),
    storage: parseDataStorage(dataStorageOutput),
    battery: parseBattery(batteryOutput),
    thermals: parseThermals(thermalOutput),
    securityPatch: securityPatch || "Unknown",
    kernelVersion: kernelVersion || "Unknown",
    buildFingerprint: properties["ro.build.fingerprint"] || "Unknown",
    cpuTopology: topologyFromFrequencies(physicalCores, frequencies),
    cpuFeatures: cpuFeatures(cpuInfo, abi),
    cpuParts: parseCpuParts(cpuInfo),
    caches: parseCaches(cacheOutput),
    gpu: {
      renderer,
      hardware: properties["ro.hardware.egl"] || "",
      vulkanHardware: properties["ro.hardware.vulkan"] || ""
    }
  };

  return { ...device, dynamicProfile: buildDynamicProfile(device) };
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

module.exports = {
  listDevices,
  parseDeviceList,
  expandCpuRange,
  parseCpuFrequencies,
  parseThermals,
  buildDynamicProfile
};
