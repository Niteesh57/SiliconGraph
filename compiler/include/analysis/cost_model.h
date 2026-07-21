// ============================================================================
// ARM AI Compiler — Analysis: Cost Model
//
// The Cost Model scores each operator on each execution unit for a given
// target device profile. This is the decision engine behind the hardware
// scheduler — it tells the compiler where each op should run.
//
// Data comes from:
//   1. Pre-measured latency tables (device_profiles/*.json)
//   2. Analytical models (FLOP-based estimates with bandwidth scaling)
//   3. Interpolation for shapes not in the lookup table
// ============================================================================
#pragma once

#include "arm_ir/types.h"
#include "arm_ir/ops.h"

#include <string>
#include <unordered_map>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>

namespace armcc {
namespace analysis {

// ---------------------------------------------------------------------------
// A single measurement row in the cost table
// ---------------------------------------------------------------------------
struct CostEntry {
  // Op descriptor
  ir::OpCode  op;
  ir::DType   input_dtype;
  ir::DType   weight_dtype;  // for ops with weights
  int64_t     M, K, N;       // shape dimensions (GEMM-style); -1 = any

  // Measured costs per execution unit (milliseconds, -1 = not supported)
  float  cpu_ms   = -1.0f;
  float  gpu_ms   = -1.0f;
  float  npu_ms   = -1.0f;
  float  dsp_ms   = -1.0f;
  float  ane_ms   = -1.0f;

  // Power (milliwatts) per unit
  float  cpu_mw   = -1.0f;
  float  gpu_mw   = -1.0f;
  float  npu_mw   = -1.0f;
  float  dsp_mw   = -1.0f;
  float  ane_mw   = -1.0f;
};

// ---------------------------------------------------------------------------
// Device profile — capabilities of one specific SoC
// ---------------------------------------------------------------------------
struct DeviceProfile {
  ir::SoCID   soc_id;
  std::string name;           // e.g. "Snapdragon 8 Gen 3"
  std::string vendor;         // "qualcomm" | "mediatek" | "samsung" | "apple"

  // Memory hierarchy
  uint32_t l1_cache_kb  = 0;
  uint32_t l2_cache_kb  = 0;
  uint32_t l3_cache_kb  = 0;
  uint32_t ram_bandwidth_gbps = 0;  // theoretical peak

  // CPU
  uint32_t num_big_cores    = 0;
  uint32_t num_mid_cores    = 0;
  uint32_t num_little_cores = 0;
  uint32_t cpu_freq_mhz     = 0;
  bool     has_sve          = false;
  bool     has_sve2         = false;
  bool     has_neon         = true;
  bool     has_dotprod      = false;
  bool     has_i8mm         = false;   // INT8 matrix multiply instructions

  // GPU
  bool     has_gpu          = false;
  uint32_t gpu_compute_units = 0;
  uint32_t gpu_freq_mhz     = 0;
  bool     supports_vulkan  = false;
  bool     supports_opencl  = false;
  bool     supports_metal   = false;

  // NPU / DSP
  bool     has_npu          = false;
  bool     has_dsp          = false;
  bool     has_hexagon      = false;   // Qualcomm Hexagon HTP
  bool     has_mediatek_apu = false;
  bool     has_apple_ane    = false;
  uint32_t npu_tops         = 0;       // INT8 TOPs

  // Supported dtypes per unit
  std::vector<ir::DType> cpu_dtypes;
  std::vector<ir::DType> gpu_dtypes;
  std::vector<ir::DType> npu_dtypes;

  // Cost table (loaded from JSON)
  std::vector<CostEntry>  cost_table;

  // Serialization
  static DeviceProfile fromJSON(const nlohmann::json& j);
  nlohmann::json       toJSON() const;
};

// ---------------------------------------------------------------------------
// CostModelQuery — input to the cost model
// ---------------------------------------------------------------------------
struct CostModelQuery {
  ir::OpCode  op;
  ir::DType   input_dtype;
  ir::DType   weight_dtype  = ir::DType::UNKNOWN;
  int64_t     M = 0, K = 0, N = 0;  // Primary dimensions
  int64_t     batch = 1;
  std::vector<int64_t> extra_dims;   // Additional shape info
};

// ---------------------------------------------------------------------------
// CostModelResult — per-unit cost estimates for one op
// ---------------------------------------------------------------------------
struct CostModelResult {
  float cpu_ms   = -1.0f;
  float gpu_ms   = -1.0f;
  float npu_ms   = -1.0f;
  float dsp_ms   = -1.0f;
  float ane_ms   = -1.0f;

  float cpu_mw   = -1.0f;
  float gpu_mw   = -1.0f;
  float npu_mw   = -1.0f;
  float dsp_mw   = -1.0f;
  float ane_mw   = -1.0f;

  // Returns the fastest supported unit and its latency
  ir::ExecUnit fastestUnit(const DeviceProfile& dev) const;
  float        fastestMs(const DeviceProfile& dev) const;

  // Returns the most efficient (FLOPS/W) unit
  ir::ExecUnit mostEfficientUnit(const DeviceProfile& dev) const;

  std::string toString() const;
};

// ---------------------------------------------------------------------------
// CostModel
//
// Usage:
//   CostModel cm;
//   cm.loadProfile("device_profiles/snapdragon_8_gen3.json");
//   auto result = cm.query(dev, {OpCode::MatMul, DType::I8, ..., M, K, N});
// ---------------------------------------------------------------------------
class CostModel {
public:
  CostModel() = default;

  // Load all profiles from a directory
  void loadProfileDirectory(const std::string& dir);

  // Load a single device profile
  void loadProfile(const std::string& jsonPath);

  // Get a loaded profile
  const DeviceProfile* getProfile(ir::SoCID id) const;
  const DeviceProfile* getProfile(const std::string& name) const;

  // Query cost for an op on a specific device
  CostModelResult query(const DeviceProfile& dev,
                        const CostModelQuery& q) const;

  // Analytic estimate when no measurement is available
  // Uses FLOP count + bandwidth model.
  CostModelResult analyticalEstimate(const DeviceProfile& dev,
                                     const CostModelQuery& q) const;

  // Recommend best execution unit for an op on this device,
  // given battery level (0-100) and thermal state.
  ir::ExecUnit recommendUnit(const DeviceProfile& dev,
                             const CostModelQuery& q,
                             int battery_pct,
                             ir::ThermalState thermal) const;

  size_t numProfiles() const { return profiles_.size(); }

private:
  std::unordered_map<uint16_t, DeviceProfile> profiles_;  // keyed by SoCID

  // Nearest-neighbour shape interpolation on the cost table
  const CostEntry* findNearest(const std::vector<CostEntry>& table,
                                const CostModelQuery& q) const;
};

} // namespace analysis
} // namespace armcc
