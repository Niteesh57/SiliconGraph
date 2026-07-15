// ============================================================================
// Unit Tests: Cost Model
// ============================================================================
#include <gtest/gtest.h>
#include "analysis/cost_model.h"
#include "arm_ir/types.h"
#include <fstream>
#include <filesystem>

using namespace armcc;
using namespace armcc::analysis;
using namespace armcc::ir;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// DeviceProfile: JSON loading
// ---------------------------------------------------------------------------
static const char* k_test_profile_json = R"({
  "soc_id": "generic_arm64",
  "name": "Test Profile",
  "vendor": "test",
  "l1_cache_kb": 32,
  "l2_cache_kb": 512,
  "l3_cache_kb": 0,
  "ram_bandwidth_gbps": 25,
  "cpu": {
    "num_big_cores": 4,
    "num_mid_cores": 0,
    "num_little_cores": 4,
    "big_core_freq_mhz": 2000,
    "mid_core_freq_mhz": 0,
    "little_core_freq_mhz": 1400,
    "has_neon": true,
    "has_dotprod": false,
    "has_i8mm": false
  },
  "gpu":  { "present": false },
  "npu":  { "present": false },
  "dsp":  { "present": false },
  "supported_dtypes": {
    "cpu": ["f32", "f16", "i8"],
    "gpu": [],
    "npu": [],
    "dsp": []
  },
  "cost_table": [
    {
      "op": "MatMul", "input_dtype": "f16", "weight_dtype": "f16",
      "M": 1, "K": 4096, "N": 4096,
      "cpu_ms": 42.0, "gpu_ms": -1, "npu_ms": -1, "dsp_ms": -1,
      "cpu_mw": 600,  "gpu_mw": -1,  "npu_mw": -1, "dsp_mw": -1
    },
    {
      "op": "MatMul", "input_dtype": "i8", "weight_dtype": "i8",
      "M": 1, "K": 4096, "N": 4096,
      "cpu_ms": 28.0, "gpu_ms": -1, "npu_ms": -1, "dsp_ms": -1,
      "cpu_mw": 400,  "gpu_mw": -1,  "npu_mw": -1, "dsp_mw": -1
    }
  ]
})";

static DeviceProfile makeTestProfile() {
  auto j = nlohmann::json::parse(k_test_profile_json);
  return DeviceProfile::fromJSON(j);
}

// ---------------------------------------------------------------------------
// DeviceProfile
// ---------------------------------------------------------------------------
TEST(DeviceProfileTest, LoadsFromJSON) {
  auto p = makeTestProfile();
  EXPECT_EQ(p.soc_id, SoCID::Generic_ARM64);
  EXPECT_EQ(p.name, "Test Profile");
  EXPECT_TRUE(p.has_neon);
  EXPECT_FALSE(p.has_gpu);
  EXPECT_FALSE(p.has_npu);
  EXPECT_EQ(p.cost_table.size(), 2u);
}

TEST(DeviceProfileTest, CostTableEntries) {
  auto p = makeTestProfile();
  EXPECT_EQ(p.cost_table[0].op, OpCode::MatMul);
  EXPECT_EQ(p.cost_table[0].input_dtype, DType::F16);
  EXPECT_FLOAT_EQ(p.cost_table[0].cpu_ms, 42.0f);
  EXPECT_FLOAT_EQ(p.cost_table[0].gpu_ms, -1.0f);  // Unavailable
}

// ---------------------------------------------------------------------------
// CostModel: query
// ---------------------------------------------------------------------------
TEST(CostModelTest, QueryReturnsExactMatch) {
  auto p = makeTestProfile();

  CostModel cm;
  cm.addProfile(p);

  CostModelQuery q;
  q.op           = OpCode::MatMul;
  q.input_dtype  = DType::F16;
  q.weight_dtype = DType::F16;
  q.M = 1; q.K = 4096; q.N = 4096;

  auto result = cm.query(p, q);
  EXPECT_GT(result.cpu_ms, 0.0f);
  EXPECT_LT(result.cpu_ms, 200.0f);  // Should be around 42ms
}

TEST(CostModelTest, QueryInterpolatesForDifferentShape) {
  auto p = makeTestProfile();
  CostModel cm;
  cm.addProfile(p);

  CostModelQuery q;
  q.op = OpCode::MatMul;
  q.input_dtype  = DType::F16;
  q.weight_dtype = DType::F16;
  q.M = 1; q.K = 2048; q.N = 2048;  // Different from table entry (4096×4096)

  auto result = cm.query(p, q);
  // Should interpolate: 2048² / 4096² = 0.25 → ~42 * 0.25 = 10.5ms
  EXPECT_GT(result.cpu_ms, 0.0f);
  EXPECT_LT(result.cpu_ms, 42.0f);  // Should be scaled down
}

TEST(CostModelTest, QueryFallsBackToAnalytical) {
  auto p = makeTestProfile();
  CostModel cm;
  cm.addProfile(p);

  // Query for an op not in the table
  CostModelQuery q;
  q.op = OpCode::RMSNorm;  // Not in test table
  q.input_dtype = DType::F16;
  q.weight_dtype = DType::F16;
  q.M = 1; q.K = 1; q.N = 4096;

  auto result = cm.query(p, q);
  EXPECT_GT(result.cpu_ms, 0.0f);  // Should get analytical estimate
}

// ---------------------------------------------------------------------------
// CostModel: recommendUnit
// ---------------------------------------------------------------------------
TEST(CostModelTest, RecommendsOnlyCPUWhenNoPPUs) {
  auto p = makeTestProfile();
  CostModel cm;
  cm.addProfile(p);

  CostModelQuery q;
  q.op = OpCode::MatMul;
  q.input_dtype = DType::F16;
  q.weight_dtype = DType::F16;
  q.M = 1; q.K = 4096; q.N = 4096;

  auto unit = cm.recommendUnit(p, q, 90, ThermalState::Nominal);
  EXPECT_EQ(unit, ExecUnit::CPU);  // No GPU/NPU available
}

TEST(CostModelTest, PrefersNPUForAttentionWhenAvailable) {
  DeviceProfile p = makeTestProfile();
  p.has_npu   = true;
  p.npu_tops  = 73;
  // Add a row with NPU time
  CostEntry e;
  e.op = OpCode::GroupQueryAttention;
  e.input_dtype = DType::F16; e.weight_dtype = DType::F16;
  e.M = 1; e.K = 4096; e.N = 4096;
  e.cpu_ms = 6.2f; e.npu_ms = 0.8f; e.gpu_ms = -1.0f; e.dsp_ms = -1.0f;
  e.cpu_mw = 600;  e.npu_mw = 300;
  p.cost_table.push_back(e);

  CostModel cm;
  cm.addProfile(p);

  CostModelQuery q;
  q.op = OpCode::GroupQueryAttention;
  q.input_dtype = DType::F16; q.weight_dtype = DType::F16;
  q.M = 1; q.K = 4096; q.N = 4096;

  auto unit = cm.recommendUnit(p, q, 90, ThermalState::Nominal);
  EXPECT_EQ(unit, ExecUnit::NPU);
}

// ---------------------------------------------------------------------------
// CostModelResult helpers
// ---------------------------------------------------------------------------
TEST(CostModelResultTest, FastestUnit) {
  auto p = makeTestProfile();
  p.has_gpu = true;

  CostModelResult r;
  r.cpu_ms = 10.0f;
  r.gpu_ms = 2.5f;
  r.npu_ms = -1.0f;

  EXPECT_EQ(r.fastestUnit(p), ExecUnit::GPU);
}

TEST(CostModelResultTest, FallsBackToCPUWhenGPUUnavailable) {
  auto p = makeTestProfile();
  // p.has_gpu = false (default from test profile)

  CostModelResult r;
  r.cpu_ms = 10.0f;
  r.gpu_ms = 2.5f;  // GPU time exists but device has no GPU
  r.npu_ms = -1.0f;

  EXPECT_EQ(r.fastestUnit(p), ExecUnit::CPU);
}

// ---------------------------------------------------------------------------
// Profile loading from directory
// ---------------------------------------------------------------------------
TEST(CostModelTest, LoadProfileDirectory) {
  CostModel cm;
  // Load from real device_profiles/ if available
  std::string dir = "device_profiles";
  if (fs::exists(dir)) {
    cm.loadProfileDirectory(dir);
    EXPECT_GT(cm.numProfiles(), 0u);

    // Snapdragon profile should be there
    auto* sd = cm.getProfile(SoCID::Snapdragon_8_Gen3);
    if (sd) {
      EXPECT_FALSE(sd->name.empty());
      EXPECT_TRUE(sd->has_npu);
      EXPECT_GT(sd->npu_tops, 0);
    }
  } else {
    GTEST_SKIP() << "device_profiles/ not found — skipping directory load test";
  }
}
