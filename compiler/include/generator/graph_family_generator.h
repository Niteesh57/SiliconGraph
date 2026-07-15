// ============================================================================
// ARM AI Compiler — Graph Family Generator
//
// Instead of one optimized graph, the compiler emits a Graph Family:
// many graphs optimized for different combinations of:
//   - Target hardware profile (SoC)
//   - Quantization precision (FP32/FP16/INT8/INT4/FP8/mixed)
//   - Memory budget (256MB / 512MB / 1GB / 2GB / 4GB+)
//   - Context length (512 / 2048 / 8192)
//   - Thermal state (nominal / warm / hot)
//   - Latency mode (interactive / batch / background)
//
// The runtime selects the best graph at inference time based on current
// device conditions — without re-compiling anything.
// ============================================================================
#pragma once

#include "arm_ir/graph.h"
#include "analysis/cost_model.h"

#include <string>
#include <vector>
#include <memory>

namespace armcc {
namespace generator {

// ---------------------------------------------------------------------------
// Graph conditions — what scenario is this graph optimized for?
// ---------------------------------------------------------------------------
enum class LatencyMode : uint8_t {
  Interactive = 0,  // User-facing, latency < 100ms per token
  Batch,            // Background batch inference
  Background,       // Low-priority, minimize power
};

struct GraphConditions {
  ir::SoCID      soc_id          = ir::SoCID::Generic_ARM64;
  ir::DType      quant_dtype     = ir::DType::INT8;
  bool           mixed_precision = false;
  uint32_t       memory_mb       = 2048;    // Available RAM in MB
  uint32_t       context_length  = 2048;
  ir::ThermalState thermal       = ir::ThermalState::Nominal;
  LatencyMode    latency_mode    = LatencyMode::Interactive;
  int            battery_pct     = 90;

  // Preferred execution units (empty = all available)
  std::vector<ir::ExecUnit> preferred_units;

  // Unique key for this condition set (used as filename component)
  std::string toKey() const;
  std::string toHumanString() const;
};

// ---------------------------------------------------------------------------
// CompiledGraph — one member of the graph family
// ---------------------------------------------------------------------------
struct CompiledGraph {
  GraphConditions          conditions;
  std::unique_ptr<ir::IRGraph> graph;

  // Compiled binary blob (ready to write into .armpack)
  std::vector<uint8_t>     serialized_graph;

  // Associated kernel object paths (relative to package root)
  std::vector<std::string> kernel_refs;

  // Estimated latency on the target (from cost model, ms/token)
  float estimated_ttft_ms   = 0.0f;   // Time-to-first-token
  float estimated_tpot_ms   = 0.0f;   // Time per output token

  // Peak memory usage
  uint64_t peak_memory_bytes = 0;
};

// ---------------------------------------------------------------------------
// GraphFamilySpec — user-supplied configuration for the family
// ---------------------------------------------------------------------------
struct GraphFamilySpec {
  // Target SoC profiles to compile for
  std::vector<ir::SoCID> target_socs;

  // Quantization dtypes to emit (one graph per dtype per soc)
  std::vector<ir::DType> quant_dtypes = {ir::DType::INT8, ir::DType::INT4};
  bool include_mixed_precision         = true;

  // Context lengths to optimize for
  std::vector<uint32_t> context_lengths = {512, 2048, 8192};

  // Memory budgets to target
  std::vector<uint32_t> memory_budgets_mb = {512, 1024, 2048, 4096};

  // Thermal scenarios
  std::vector<ir::ThermalState> thermal_states = {
    ir::ThermalState::Nominal,
    ir::ThermalState::Hot,
  };

  // Latency modes
  std::vector<LatencyMode> latency_modes = {
    LatencyMode::Interactive,
    LatencyMode::Background,
  };

  // Maximum number of graphs in the family (0 = unlimited)
  // The generator will prune redundant graphs if over this limit.
  uint32_t max_graphs = 0;

  // Prune graphs that aren't significantly better than another graph
  // in the family that also covers the same conditions
  bool enable_pruning = true;
  float pruning_threshold_pct = 5.0f;  // <5% improvement = prune
};

// ---------------------------------------------------------------------------
// GraphFamilyGenerator
// ---------------------------------------------------------------------------
class GraphFamilyGenerator {
public:
  GraphFamilyGenerator(const analysis::CostModel& costModel,
                       const analysis::DeviceProfile* profiles,
                       size_t numProfiles);

  // Generate the full graph family from an input IR graph
  // The input graph is the result of the analysis + optimization pipeline.
  std::vector<CompiledGraph> generate(
    const ir::IRGraph& sourceGraph,
    const GraphFamilySpec& spec
  );

  // Get the number of graphs that would be generated for a spec
  // (without actually generating them — for dry-run / preview)
  size_t estimateGraphCount(const GraphFamilySpec& spec) const;

  // Print a summary of a generated family
  static void printFamilySummary(const std::vector<CompiledGraph>& family);

private:
  const analysis::CostModel&     costModel_;
  const analysis::DeviceProfile* profiles_;
  size_t                         numProfiles_;

  // Compile one graph for a specific condition set
  CompiledGraph compileOneGraph(const ir::IRGraph& source,
                                const GraphConditions& conds);

  // Prune redundant graphs from the family
  void prunFamily(std::vector<CompiledGraph>& family,
                  float threshold_pct);
};

} // namespace generator
} // namespace armcc
