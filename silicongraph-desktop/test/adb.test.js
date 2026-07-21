const assert = require("assert/strict");
const test = require("node:test");
const {
  buildDynamicProfile,
  expandCpuRange,
  parseCpuFrequencies,
  parseDeviceList,
  parseThermals
} = require("../electron/adb");

test("parses an authorized adb device with its attributes", () => {
  const devices = parseDeviceList([
    "List of devices attached",
    "R5CT30ABCDEF device product:husky model:Pixel_8_Pro device:husky transport_id:1"
  ].join("\n"));

  assert.deepEqual(devices, [{
    serial: "R5CT30ABCDEF",
    state: "device",
    attributes: {
      product: "husky",
      model: "Pixel_8_Pro",
      device: "husky",
      transport_id: "1"
    }
  }]);
});

test("reads the physical CPU range instead of trusting adb nproc alone", () => {
  assert.deepEqual(expandCpuRange("0-3,4-7"), [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.deepEqual(parseCpuFrequencies("0:1785600\n7:2841600"), [
    { core: 0, mhz: 1786 },
    { core: 7, mhz: 2842 }
  ]);
});

test("uses current thermal HAL values rather than stale cached readings", () => {
  const thermals = parseThermals([
    "Thermal Status: 0",
    "Cached temperatures:",
    "  Temperature{mValue=74.0, mType=0, mName=CPU7, mStatus=0}",
    "Current temperatures from HAL:",
    "  Temperature{mValue=41.5, mType=0, mName=CPU7, mStatus=0}",
    "  Temperature{mValue=39.9, mType=1, mName=GPU0, mStatus=0}",
    "  Temperature{mValue=61.0, mType=8, mName=soc, mStatus=0}",
    "Current cooling devices from HAL:"
  ].join("\n"));

  assert.deepEqual(thermals, { status: 0, maxCpuC: 41.5, gpuC: 39.9, socC: 61, skinC: null });
});

test("builds a packageable profile from detected device facts", () => {
  const profile = buildDynamicProfile({
    serial: "8ee07cd0",
    name: "OnePlus GM1901",
    manufacturer: "OnePlus",
    model: "GM1901",
    androidVersion: "12",
    apiLevel: "31",
    abi: "arm64-v8a",
    socManufacturer: "Qualcomm",
    socModel: "SM8150",
    boardPlatform: "msmnile",
    physicalCoreCount: 8,
    onlineCoreCount: 8,
    adbVisibleCoreCount: 6,
    cpuTopology: {
      clusters: [{ mhz: 1786, cores: [0, 1, 2, 3] }, { mhz: 2420, cores: [4, 5, 6] }, { mhz: 2842, cores: [7] }],
      num_big_cores: 1,
      num_mid_cores: 3,
      num_little_cores: 4,
      big_core_freq_mhz: 2842,
      mid_core_freq_mhz: 2420,
      little_core_freq_mhz: 1786
    },
    cpuFeatures: { has_neon: true, has_dotprod: true, has_i8mm: false, has_sve: false, has_sve2: false },
    caches: [{ level: 1, sizeKb: 64 }, { level: 2, sizeKb: 512 }],
    gpu: { renderer: "Qualcomm Adreno (TM) 640", hardware: "qcom", vulkanHardware: "adreno" }
  });

  assert.equal(profile.soc_id, "generic_arm64");
  assert.equal(profile.cpu.physical_core_count, 8);
  assert.equal(profile.cpu.adb_visible_core_count, 6);
  assert.equal(profile.cpu.num_big_cores, 1);
  assert.equal(profile.npu.present, false);
  assert.match(profile.detected_accelerators.npu_hardware, /Hexagon/);
});
