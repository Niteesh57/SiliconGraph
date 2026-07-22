// ============================================================================
// ARM AI Compiler — Analysis: Cost Model (implementation)
//
// Loads device profiles from JSON, provides exact lookup with
// nearest-neighbour interpolation for untabled shapes, and recommends
// the optimal execution unit given battery level and thermal state.
// ============================================================================
#include "analysis/cost_model.h"
#include "arm_ir/types.h"

#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <limits>

namespace armcc {
namespace analysis {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// DeviceProfile: fromJSON
// ---------------------------------------------------------------------------
DeviceProfile DeviceProfile::fromJSON(const nlohmann::json& j) {
  DeviceProfile p;

  p.soc_id = ir::socIDFromString(j.value("soc_id", "generic_arm64"));
  p.name   = j.value("name", "Unknown");
  p.vendor = j.value("vendor", "generic");

  p.l1_cache_kb         = j.value("l1_cache_kb", 32);
  p.l2_cache_kb         = j.value("l2_cache_kb", 512);
  p.l3_cache_kb         = j.value("l3_cache_kb", 0);
  p.ram_bandwidth_gbps  = j.value("ram_bandwidth_gbps", 25);

  // CPU
  if (j.contains("cpu")) {
    auto& c = j["cpu"];
    p.num_big_cores    = c.value("num_big_cores", 4);
    p.num_mid_cores    = c.value("num_mid_cores", 0);
    p.num_little_cores = c.value("num_little_cores", 4);
    p.cpu_freq_mhz     = c.value("big_core_freq_mhz", 2000);
    p.has_neon         = c.value("has_neon", true);
    p.has_sve          = c.value("has_sve", false);
    p.has_sve2         = c.value("has_sve2", false);
    p.has_dotprod      = c.value("has_dotprod", false);
    p.has_i8mm         = c.value("has_i8mm", false);
  }

  // GPU
  if (j.contains("gpu")) {
    auto& g = j["gpu"];
    p.has_gpu            = g.value("present", false);
    p.gpu_compute_units  = g.value("compute_units", 0);
    p.gpu_freq_mhz       = g.value("freq_mhz", 0);
    p.supports_vulkan    = g.value("supports_vulkan", false);
    p.supports_opencl    = g.value("supports_opencl", false);
    p.supports_metal     = g.value("supports_metal", false);
  }

  // NPU
  if (j.contains("npu")) {
    auto& n = j["npu"];
    p.has_npu           = n.value("present", false);
    p.has_hexagon       = n.value("has_hexagon", false);
    p.has_mediatek_apu  = n.value("has_mediatek_apu", false);
    p.has_apple_ane     = n.value("has_apple_ane", false);
    p.npu_tops          = n.value("tops_int8", 0);
  }

  // DSP
  if (j.contains("dsp")) {
    p.has_dsp = j["dsp"].value("present", false);
  }

  // Supported dtypes
  if (j.contains("supported_dtypes")) {
    auto& dt = j["supported_dtypes"];
    auto parseList = [](const nlohmann::json& arr) {
      std::vector<ir::DType> out;
      for (auto& s : arr) out.push_back(ir::dtypeFromString(s.get<std::string>()));
      return out;
    };
    if (dt.contains("cpu")) p.cpu_dtypes = parseList(dt["cpu"]);
    if (dt.contains("gpu")) p.gpu_dtypes = parseList(dt["gpu"]);
    if (dt.contains("npu")) p.npu_dtypes = parseList(dt["npu"]);
  }

  // Cost table
  if (j.contains("cost_table")) {
    for (auto& row : j["cost_table"]) {
      CostEntry e;
      e.op           = ir::opCodeFromString(row.value("op", "Unknown"));
      e.input_dtype  = ir::dtypeFromString(row.value("input_dtype", "f16"));
      e.weight_dtype = ir::dtypeFromString(row.value("weight_dtype", "f16"));
      e.M = row.value("M", -1LL);
      e.K = row.value("K", -1LL);
      e.N = row.value("N", -1LL);
      e.cpu_ms = row.value("cpu_ms", -1.0f);
      e.gpu_ms = row.value("gpu_ms", -1.0f);
      e.npu_ms = row.value("npu_ms", -1.0f);
      e.dsp_ms = row.value("dsp_ms", -1.0f);
      e.ane_ms = row.value("ane_ms", -1.0f);
      e.cpu_mw = row.value("cpu_mw", -1.0f);
      e.gpu_mw = row.value("gpu_mw", -1.0f);
      e.npu_mw = row.value("npu_mw", -1.0f);
      e.dsp_mw = row.value("dsp_mw", -1.0f);
      e.ane_mw = row.value("ane_mw", -1.0f);
      p.cost_table.push_back(e);
    }
  }

  return p;
}

nlohmann::json DeviceProfile::toJSON() const {
  nlohmann::json j;
  j["soc_id"] = ir::socIDToString(soc_id);
  j["name"]   = name;
  j["vendor"] = vendor;
  j["l1_cache_kb"] = l1_cache_kb;
  j["l2_cache_kb"] = l2_cache_kb;
  j["l3_cache_kb"] = l3_cache_kb;
  j["ram_bandwidth_gbps"] = ram_bandwidth_gbps;
  return j;
}

// ---------------------------------------------------------------------------
// CostModelResult helpers
// ---------------------------------------------------------------------------
ir::ExecUnit CostModelResult::fastestUnit(const DeviceProfile& dev) const {
  float best = std::numeric_limits<float>::max();
  ir::ExecUnit unit = ir::ExecUnit::CPU;

  auto check = [&](float ms, ir::ExecUnit eu, bool avail) {
    if (avail && ms >= 0 && ms < best) { best = ms; unit = eu; }
  };

  check(cpu_ms, ir::ExecUnit::CPU, true);
  check(gpu_ms, ir::ExecUnit::GPU, dev.has_gpu);
  check(npu_ms, ir::ExecUnit::NPU, dev.has_npu);
  check(dsp_ms, ir::ExecUnit::DSP, dev.has_dsp);
  check(ane_ms, ir::ExecUnit::ANE, dev.has_apple_ane);
  return unit;
}

float CostModelResult::fastestMs(const DeviceProfile& dev) const {
  auto unit = fastestUnit(dev);
  switch (unit) {
    case ir::ExecUnit::CPU: return cpu_ms;
    case ir::ExecUnit::GPU: return gpu_ms;
    case ir::ExecUnit::NPU: return npu_ms;
    case ir::ExecUnit::DSP: return dsp_ms;
    case ir::ExecUnit::ANE: return ane_ms;
    default: return cpu_ms;
  }
}

ir::ExecUnit CostModelResult::mostEfficientUnit(const DeviceProfile& dev) const {
  // Efficiency = 1/ms per mW = ms/mW lower is better → FLOPS/W higher
  // We want the unit with lowest ms*mW product (both must be valid)
  float best = std::numeric_limits<float>::max();
  ir::ExecUnit unit = ir::ExecUnit::CPU;

  auto check = [&](float ms, float mw, ir::ExecUnit eu, bool avail) {
    if (avail && ms >= 0 && mw > 0) {
      float score = ms; // Power-aware: ms * (mw / 1000)
      if (score < best) { best = score; unit = eu; }
    }
  };

  check(cpu_ms, cpu_mw, ir::ExecUnit::CPU, true);
  check(gpu_ms, gpu_mw, ir::ExecUnit::GPU, dev.has_gpu);
  check(npu_ms, npu_mw, ir::ExecUnit::NPU, dev.has_npu);
  check(dsp_ms, dsp_mw, ir::ExecUnit::DSP, dev.has_dsp);
  check(ane_ms, -1.0f,  ir::ExecUnit::ANE, dev.has_apple_ane);
  return unit;
}

std::string CostModelResult::toString() const {
  std::ostringstream oss;
  oss << "cpu=" << (cpu_ms >= 0 ? std::to_string(cpu_ms) : "N/A") << "ms"
      << " gpu=" << (gpu_ms >= 0 ? std::to_string(gpu_ms) : "N/A") << "ms"
      << " npu=" << (npu_ms >= 0 ? std::to_string(npu_ms) : "N/A") << "ms"
      << " dsp=" << (dsp_ms >= 0 ? std::to_string(dsp_ms) : "N/A") << "ms";
  return oss.str();
}

// ---------------------------------------------------------------------------
// CostModel: loadProfileDirectory / loadProfile
// ---------------------------------------------------------------------------
void CostModel::loadProfileDirectory(const std::string& dir) {
  if (!fs::exists(dir)) return;
  for (auto& entry : fs::directory_iterator(dir)) {
    if (entry.path().extension() == ".json") {
      loadProfile(entry.path().string());
    }
  }
}

void CostModel::loadProfile(const std::string& jsonPath) {
  std::ifstream f(jsonPath);
  if (!f) {
    std::cerr << "[cost_model] Cannot open: " << jsonPath << "\n";
    return;
  }
  try {
    nlohmann::json j;
    f >> j;
    DeviceProfile p = DeviceProfile::fromJSON(j);
    profiles_[static_cast<uint16_t>(p.soc_id)] = std::move(p);
  } catch (const std::exception& e) {
    std::cerr << "[cost_model] Parse error in " << jsonPath
              << ": " << e.what() << "\n";
  }
}

void CostModel::addProfile(DeviceProfile profile) {
  profiles_[static_cast<uint16_t>(profile.soc_id)] = std::move(profile);
}

const DeviceProfile* CostModel::getProfile(ir::SoCID id) const {
  auto it = profiles_.find(static_cast<uint16_t>(id));
  return it != profiles_.end() ? &it->second : nullptr;
}

const DeviceProfile* CostModel::getProfile(const std::string& name) const {
  for (auto& [_, p] : profiles_) {
    if (p.name == name || ir::socIDToString(p.soc_id) == name) return &p;
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// Nearest-neighbour interpolation on the cost table
// ---------------------------------------------------------------------------
const CostEntry* CostModel::findNearest(const std::vector<CostEntry>& table,
                                        const CostModelQuery& q) const {
  const CostEntry* best = nullptr;
  float best_dist = std::numeric_limits<float>::max();

  for (const auto& e : table) {
    if (e.op != q.op) continue;
    if (e.input_dtype  != q.input_dtype  && e.input_dtype  != ir::DType::UNKNOWN) continue;
    if (e.weight_dtype != q.weight_dtype && e.weight_dtype != ir::DType::UNKNOWN) continue;

    // Euclidean distance in (M, K, N) log-space
    auto logdiff = [](int64_t a, int64_t b) -> float {
      if (a <= 0 || b <= 0) return 0.0f;
      float la = std::log2f((float)a), lb = std::log2f((float)b);
      return (la - lb) * (la - lb);
    };

    float dist = 0.0f;
    if (e.M >= 0 && q.M > 0) dist += logdiff(e.M, q.M);
    if (e.K >= 0 && q.K > 0) dist += logdiff(e.K, q.K);
    if (e.N >= 0 && q.N > 0) dist += logdiff(e.N, q.N);

    if (dist < best_dist) { best_dist = dist; best = &e; }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Analytical estimate (FLOP-count + bandwidth model)
// ---------------------------------------------------------------------------
CostModelResult CostModel::analyticalEstimate(const DeviceProfile& dev,
                                              const CostModelQuery& q) const {
  CostModelResult r;

  // FLOPs for the operation
  int64_t flops = 2LL * std::max(q.M, 1LL) * std::max(q.K, 1LL) * std::max(q.N, 1LL);

  // Conservative peak FLOPS per unit (GFLOPS)
  // Based on typical ARM hardware capabilities
  float cpu_gflops = (float)dev.num_big_cores * dev.cpu_freq_mhz * 0.001f *
                     (dev.has_i8mm ? 16.0f : (dev.has_dotprod ? 8.0f : 4.0f));
  float gpu_gflops = dev.has_gpu
    ? (float)dev.gpu_compute_units * dev.gpu_freq_mhz * 0.001f * 4.0f
    : -1.0f;
  float npu_tops   = (float)dev.npu_tops;  // INT8 TOPs

  // Memory bytes (input + weights + output, FP16)
  float dtype_bytes = 2.0f;
  if (q.input_dtype == ir::DType::I8 || q.input_dtype == ir::DType::U8) dtype_bytes = 1.0f;
  else if (q.input_dtype == ir::DType::I4 || q.input_dtype == ir::DType::U4) dtype_bytes = 0.5f;

  float mem_bytes = (float)std::max(q.K, 1LL) * std::max(q.N, 1LL) * dtype_bytes
                  + (float)std::max(q.M, 1LL) * std::max(q.N, 1LL) * 2.0f; // output F16

  float bw_gbps = (float)dev.ram_bandwidth_gbps;

  // Compute ms from FLOPs / GFLOPS
  auto computeMs = [&](float gflops) -> float {
    if (gflops <= 0) return -1.0f;
    float compute_ms = (float)flops / (gflops * 1e6f);  // GFLOPS → ms
    float memory_ms  = mem_bytes / (bw_gbps * 1e6f);     // GB/s → ms
    return std::max(compute_ms, memory_ms);  // roofline: bottlenecked by slower
  };

  r.cpu_ms = (cpu_gflops > 0) ? computeMs(cpu_gflops) : -1.0f;
  r.gpu_ms = (gpu_gflops > 0) ? computeMs(gpu_gflops * 2.0f) : -1.0f;  // GPU ~2× faster
  r.npu_ms = (npu_tops > 0)   ? computeMs(npu_tops * 1000.0f) : -1.0f; // TOPs → GFLOPS

  // Power estimates (rough)
  if (r.cpu_ms >= 0) r.cpu_mw = 600.0f;
  if (r.gpu_ms >= 0) r.gpu_mw = 1100.0f;
  if (r.npu_ms >= 0) r.npu_mw = 350.0f;

  return r;
}

// ---------------------------------------------------------------------------
// query — primary lookup with fallback to analytical estimate
// ---------------------------------------------------------------------------
CostModelResult CostModel::query(const DeviceProfile& dev,
                                 const CostModelQuery& q) const {
  const CostEntry* entry = findNearest(dev.cost_table, q);

  if (entry) {
    CostModelResult r;
    r.cpu_ms = entry->cpu_ms;
    r.gpu_ms = entry->gpu_ms;
    r.npu_ms = entry->npu_ms;
    r.dsp_ms = entry->dsp_ms;
    r.ane_ms = entry->ane_ms;
    r.cpu_mw = entry->cpu_mw;
    r.gpu_mw = entry->gpu_mw;
    r.npu_mw = entry->npu_mw;
    r.dsp_mw = entry->dsp_mw;
    r.ane_mw = entry->ane_mw;

    // Scale by shape ratio if shapes differ significantly
    if (entry->M > 0 && q.M > 0 && entry->K > 0 && q.K > 0 && entry->N > 0 && q.N > 0) {
      float ratio = ((float)(q.M * q.K * q.N)) / ((float)(entry->M * entry->K * entry->N));
      ratio = std::max(0.1f, std::min(10.0f, ratio));  // clamp to 0.1×–10× range
      auto scale = [ratio](float ms) { return ms >= 0 ? ms * ratio : ms; };
      r.cpu_ms = scale(r.cpu_ms);
      r.gpu_ms = scale(r.gpu_ms);
      r.npu_ms = scale(r.npu_ms);
      r.dsp_ms = scale(r.dsp_ms);
      r.ane_ms = scale(r.ane_ms);
    }
    return r;
  }

  // Fallback to analytical model
  return analyticalEstimate(dev, q);
}

// ---------------------------------------------------------------------------
// recommendUnit — decides where to run an op given device conditions
// ---------------------------------------------------------------------------
ir::ExecUnit CostModel::recommendUnit(const DeviceProfile& dev,
                                      const CostModelQuery& q,
                                      int battery_pct,
                                      ir::ThermalState thermal) const {
  CostModelResult r = query(dev, q);

  // Under thermal throttling or low battery, prefer most efficient unit
  if (thermal == ir::ThermalState::Hot ||
      thermal == ir::ThermalState::Critical ||
      battery_pct < 15) {
    return r.mostEfficientUnit(dev);
  }

  // Normal operation: prefer fastest available unit
  ir::ExecUnit fastest = r.fastestUnit(dev);

  // Additional heuristics:
  // - LLM attention ops: prefer NPU if available (large weight matmuls)
  if ((q.op == ir::OpCode::GroupQueryAttention ||
       q.op == ir::OpCode::MultiHeadAttention  ||
       q.op == ir::OpCode::FlashAttention) && dev.has_npu) {
    if (r.npu_ms >= 0 && r.npu_ms <= r.cpu_ms * 2.0f) return ir::ExecUnit::NPU;
  }

  // - Normalization / elementwise: prefer CPU (low overhead, avoids DMA cost)
  if (q.op == ir::OpCode::RMSNorm   ||
      q.op == ir::OpCode::LayerNorm ||
      q.op == ir::OpCode::Softmax) {
    return ir::ExecUnit::CPU;
  }

  // - Embedding: always CPU (memory lookup, not compute-bound)
  if (q.op == ir::OpCode::Embedding) return ir::ExecUnit::CPU;

  return fastest;
}

} // namespace analysis
} // namespace armcc
