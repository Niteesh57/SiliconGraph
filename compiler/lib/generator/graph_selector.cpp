// ============================================================================
// ARM AI Compiler — Graph Selector (implementation)
//
// Runtime graph selection. Picks the best precompiled graph from the family
// based on current device state (RAM, battery, thermal, SoC, context).
// ============================================================================
#include "generator/graph_selector.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace armcc {
namespace generator {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
GraphSelector::GraphSelector(std::vector<GraphSelectorEntry> entries)
  : entries_(std::move(entries)) {}

// ---------------------------------------------------------------------------
// score — lower = better match for the given state
// ---------------------------------------------------------------------------
float GraphSelector::score(const GraphSelectorEntry& e,
                           const RuntimeDeviceState& state) const {
  float s = 0.0f;

  // Hard reject: wrong SoC (if SoC known)
  if (state.soc_id != ir::SoCID::Unknown &&
      e.conditions.soc_id != state.soc_id &&
      e.conditions.soc_id != ir::SoCID::Generic_ARM64) {
    return std::numeric_limits<float>::max();  // Hard reject
  }

  // Hard reject: insufficient memory
  if (e.peak_memory_bytes > 0) {
    uint64_t avail = (uint64_t)state.ram_available_mb * 1024 * 1024;
    if (e.peak_memory_bytes > avail * 95 / 100) {  // 5% headroom
      return std::numeric_limits<float>::max();  // Would OOM
    }
  }

  // Hard reject: requires GPU/NPU but unavailable
  for (auto unit : e.conditions.preferred_units) {
    if (unit == ir::ExecUnit::GPU && !state.gpu_available) s += 1000.0f;
    if ((unit == ir::ExecUnit::NPU || unit == ir::ExecUnit::APU) && !state.npu_available) s += 1000.0f;
  }

  // Latency mode mismatch (soft penalty)
  if (e.conditions.latency_mode != state.latency_target) s += 20.0f;

  // Thermal mismatch
  if (e.conditions.thermal != state.thermal) {
    s += 10.0f * std::abs((int)e.conditions.thermal - (int)state.thermal);
  }

  // Context length mismatch: prefer graph with context >= requested
  int64_t ctx_diff = (int64_t)e.conditions.context_length - (int64_t)state.context_length;
  if (ctx_diff < 0) {
    // Graph context too short: heavy penalty
    s += 500.0f + (float)(-ctx_diff) * 0.1f;
  } else {
    // Graph context larger than needed: small penalty (wastes memory)
    s += (float)ctx_diff * 0.001f;
  }

  // Memory margin: more available = better, but penalize wasteful graphs
  if (state.ram_available_mb > 0 && e.peak_memory_bytes > 0) {
    float usage_pct = (float)e.peak_memory_bytes /
                      ((float)state.ram_available_mb * 1024 * 1024);
    // Ideal is 60-80% RAM utilization
    if (usage_pct > 0.85f) s += 50.0f;       // Too tight
    else if (usage_pct < 0.3f) s += 10.0f;   // Wasteful (too conservative)
  }

  // Battery: under low battery, prefer lower precision (faster/more efficient)
  if (state.battery_pct < 20) {
    if (e.conditions.quant_dtype == ir::DType::I4) s -= 15.0f;  // Prefer INT4
    if (e.conditions.quant_dtype == ir::DType::I8) s -= 5.0f;
  }

  // Latency: add estimated TPOT as baseline score
  s += e.estimated_tpot_ms * 0.5f;

  return s;
}

// ---------------------------------------------------------------------------
// select
// ---------------------------------------------------------------------------
uint32_t GraphSelector::select(const RuntimeDeviceState& state) const {
  if (entries_.empty()) return 0;

  uint32_t best_idx = 0;
  float    best_score = std::numeric_limits<float>::max();

  for (size_t i = 0; i < entries_.size(); ++i) {
    float s = score(entries_[i], state);
    if (s < best_score) {
      best_score = s;
      best_idx   = entries_[i].graph_index;
    }
  }
  return best_idx;
}

uint32_t GraphSelector::selectWithFallback(const RuntimeDeviceState& state) const {
  if (entries_.empty()) return 0;

  // Try normal selection first
  uint32_t idx = select(state);

  // Validate it's not a hard-reject result
  if (idx < entries_.size()) {
    float s = score(entries_[idx], state);
    if (s < std::numeric_limits<float>::max() / 2) return idx;
  }

  // Fallback: find the entry with lowest peak memory (most conservative)
  uint32_t min_mem_idx = 0;
  uint64_t min_mem     = std::numeric_limits<uint64_t>::max();
  for (auto& e : entries_) {
    if (e.conditions.soc_id == ir::SoCID::Generic_ARM64 &&
        e.peak_memory_bytes < min_mem) {
      min_mem     = e.peak_memory_bytes;
      min_mem_idx = e.graph_index;
    }
  }
  return min_mem_idx;
}

// ---------------------------------------------------------------------------
// Serialization: toJSON / fromJSON
// ---------------------------------------------------------------------------
std::string GraphSelector::toJSON() const {
  nlohmann::json j = nlohmann::json::array();
  for (auto& e : entries_) {
    nlohmann::json je;
    je["graph_index"]        = e.graph_index;
    je["estimated_tpot_ms"]  = e.estimated_tpot_ms;
    je["peak_memory_bytes"]  = e.peak_memory_bytes;
    je["soc_id"]             = ir::socIDToString(e.conditions.soc_id);
    je["quant_dtype"]        = ir::dtypeToString(e.conditions.quant_dtype);
    je["mixed_precision"]    = e.conditions.mixed_precision;
    je["memory_mb"]          = e.conditions.memory_mb;
    je["context_length"]     = e.conditions.context_length;
    je["thermal"]            = (int)e.conditions.thermal;
    je["latency_mode"]       = (int)e.conditions.latency_mode;
    j.push_back(je);
  }
  return j.dump(2);
}

GraphSelector GraphSelector::fromJSON(const std::string& jsonStr) {
  std::vector<GraphSelectorEntry> entries;
  try {
    auto j = nlohmann::json::parse(jsonStr);
    for (auto& je : j) {
      GraphSelectorEntry e;
      e.graph_index       = je.value("graph_index", 0u);
      e.estimated_tpot_ms = je.value("estimated_tpot_ms", 0.0f);
      e.peak_memory_bytes = je.value("peak_memory_bytes", 0ULL);
      e.conditions.soc_id = ir::socIDFromString(je.value("soc_id", "generic_arm64"));
      e.conditions.quant_dtype = ir::dtypeFromString(je.value("quant_dtype", "i8"));
      e.conditions.mixed_precision = je.value("mixed_precision", false);
      e.conditions.memory_mb = je.value("memory_mb", 2048u);
      e.conditions.context_length = je.value("context_length", 2048u);
      e.conditions.thermal  = (ir::ThermalState)je.value("thermal", 0);
      e.conditions.latency_mode = (LatencyMode)je.value("latency_mode", 0);
      entries.push_back(e);
    }
  } catch (...) {}
  return GraphSelector(std::move(entries));
}

} // namespace generator
} // namespace armcc
