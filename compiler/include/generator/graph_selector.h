// ============================================================================
// ARM AI Compiler — Graph Selector
//
// Embedded at runtime, the GraphSelector chooses the best precompiled graph
// from the family based on current device conditions.
//
// This logic is also compiled to selector.wasm and included in .armpack,
// so it runs portably on any device without depending on libarmcc.
// ============================================================================
#pragma once

#include "arm_ir/types.h"
#include "generator/graph_family_generator.h"

#include <vector>
#include <string>
#include <optional>

namespace armcc {
namespace generator {

// ---------------------------------------------------------------------------
// Runtime device state — filled by the device at inference time
// ---------------------------------------------------------------------------
struct RuntimeDeviceState {
  ir::SoCID      soc_id          = ir::SoCID::Unknown;
  uint32_t       ram_available_mb = 0;   // Currently free RAM
  int            battery_pct      = 100;
  ir::ThermalState thermal        = ir::ThermalState::Nominal;
  LatencyMode    latency_target   = LatencyMode::Interactive;
  uint32_t       context_length   = 512;

  // The optional Vulkan candidate is considered only after its measured
  // end-to-end latency beats CPU. No NPU/DSP delegate is represented here.
  bool vulkan_benchmark_wins = false;

  // Media encoders are admitted only on full or medium precision graph
  // variants. INT4/FP8 remain an intentional text-only execution path.
  bool require_medium_or_higher_precision = false;
};

// ---------------------------------------------------------------------------
// GraphSelectorEntry — lightweight index stored in the package manifest
// ---------------------------------------------------------------------------
struct GraphSelectorEntry {
  uint32_t        graph_index;   // Index into the graphs[] array in .armpack
  GraphConditions conditions;
  float           estimated_tpot_ms;
  uint64_t        peak_memory_bytes;
};

// ---------------------------------------------------------------------------
// GraphSelector
//
// Loaded from the package's manifest.json at runtime.
// select() returns the graph_index of the best matching graph.
// ---------------------------------------------------------------------------
class GraphSelector {
public:
  explicit GraphSelector(std::vector<GraphSelectorEntry> entries);

  // Select the best graph for the given device state
  // Returns graph_index into the .armpack graphs array.
  uint32_t select(const RuntimeDeviceState& state) const;

  // Select with fallback (guaranteed to return something)
  uint32_t selectWithFallback(const RuntimeDeviceState& state) const;

  // Select only if a graph satisfies every hard resource constraint. Runtime
  // resource control must use this rather than the conservative fallback.
  std::optional<uint32_t> selectViable(const RuntimeDeviceState& state) const;

  // True when the graph is the optional Vulkan candidate rather than CPU.
  bool isVulkanGraph(uint32_t graph_index) const;

  // Precision of a selected graph, used for an auditable media admission
  // decision in the runtime.
  std::optional<ir::DType> quantDTypeForGraph(uint32_t graph_index) const;

  // Serialize the selector index to JSON (written into manifest.json)
  std::string toJSON() const;

  // Deserialize from JSON
  static GraphSelector fromJSON(const std::string& json);

  size_t numEntries() const { return entries_.size(); }

private:
  std::vector<GraphSelectorEntry> entries_;

  // Scoring function: lower = better match for state
  float score(const GraphSelectorEntry& entry,
              const RuntimeDeviceState& state) const;
};

// Build portable selector source for a compiled graph family. The generated
// source is stored in the package alongside the selector index.
std::string generateSelectorC(const std::vector<CompiledGraph>& family);

} // namespace generator
} // namespace armcc
