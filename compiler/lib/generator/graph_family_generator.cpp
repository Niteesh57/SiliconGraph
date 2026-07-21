// ============================================================================
// ARM AI Compiler — Graph Family Generator (implementation)
//
// Produces a family of optimized graphs from one source graph.
// Each graph is optimized for a specific combination of:
//   SoC × precision × memory budget × context length × thermal × latency mode
// ============================================================================
#include "generator/graph_family_generator.h"
#include "passes/pass_manager.h"
#include "analysis/graph_analyzer.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>

namespace armcc {
namespace generator {

// ---------------------------------------------------------------------------
// GraphConditions helpers
// ---------------------------------------------------------------------------
std::string GraphConditions::toKey() const {
  std::ostringstream oss;
  oss << ir::socIDToString(soc_id) << "_"
      << ir::dtypeToString(quant_dtype)
      << (mixed_precision ? "_mixed" : "") << "_"
      << memory_mb << "mb_"
      << context_length << "ctx_"
      << (thermal == ir::ThermalState::Nominal ? "nominal" :
          thermal == ir::ThermalState::Warm    ? "warm"    : "hot") << "_"
      << (latency_mode == LatencyMode::Interactive ? "interactive" :
          latency_mode == LatencyMode::Batch        ? "batch"       : "background");
  return oss.str();
}

std::string GraphConditions::toHumanString() const {
  std::ostringstream oss;
  oss << ir::socIDToString(soc_id)
      << " | " << ir::dtypeToString(quant_dtype)
      << (mixed_precision ? "+mixed" : "")
      << " | " << memory_mb << "MB"
      << " | ctx=" << context_length
      << " | " << (latency_mode == LatencyMode::Interactive ? "interactive" : "background");
  if (thermal != ir::ThermalState::Nominal) {
    oss << " | thermal=" << (thermal == ir::ThermalState::Warm ? "warm" : "hot");
  }
  return oss.str();
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
GraphFamilyGenerator::GraphFamilyGenerator(
    const analysis::CostModel& costModel,
    const analysis::DeviceProfile* profiles,
    size_t numProfiles)
  : costModel_(costModel)
  , profiles_(profiles)
  , numProfiles_(numProfiles)
{}

// ---------------------------------------------------------------------------
// estimateGraphCount
// ---------------------------------------------------------------------------
size_t GraphFamilyGenerator::estimateGraphCount(const GraphFamilySpec& spec) const {
  size_t count = 0;
  for (auto soc : spec.target_socs) {
    for (auto q : spec.quant_dtypes) {
      for (auto ctx : spec.context_lengths) {
        for (auto mem : spec.memory_budgets_mb) {
          for (auto th : spec.thermal_states) {
            for (auto lat : spec.latency_modes) {
              count++;
            }
          }
        }
      }
    }
    if (spec.include_mixed_precision) count += spec.context_lengths.size();
  }
  return count;
}

// ---------------------------------------------------------------------------
// generate — main compilation loop
// ---------------------------------------------------------------------------
std::vector<CompiledGraph> GraphFamilyGenerator::generate(
    const ir::IRGraph& sourceGraph,
    const GraphFamilySpec& spec)
{
  std::vector<CompiledGraph> family;
  uint32_t graphIndex = 0;

  auto t0 = std::chrono::steady_clock::now();
  size_t estimated = estimateGraphCount(spec);
  std::cout << "[generator] Graph family: ~" << estimated << " candidate graphs\n";

  // For each target SoC
  for (auto soc_id : spec.target_socs) {
    const analysis::DeviceProfile* dev = costModel_.getProfile(soc_id);
    if (!dev) {
      std::cout << "[generator]   [SKIP] No profile for "
                << ir::socIDToString(soc_id) << "\n";
      continue;
    }

    // For each quantization dtype
    for (auto quant_dtype : spec.quant_dtypes) {
      // Skip if device doesn't support this dtype in hardware
      bool dtype_ok = true;
      if (quant_dtype == ir::DType::F8_E4M3 && !dev->has_apple_ane && !dev->has_gpu)
        dtype_ok = false;  // FP8 only useful on ANE/GPU
      if (!dtype_ok) continue;

      for (auto ctx_len : spec.context_lengths) {
        for (auto mem_mb : spec.memory_budgets_mb) {
          // Skip memory budgets too small for the model
          // (simplified check: model needs at least half its weight bytes)
          uint64_t est_weight_bytes = sourceGraph.weight_bytes;
          if (est_weight_bytes > 0 && (uint64_t)mem_mb * 1024 * 1024 < est_weight_bytes / 2) {
            continue;
          }

          for (auto thermal : spec.thermal_states) {
            for (auto latency : spec.latency_modes) {
              GraphConditions conds;
              conds.soc_id          = soc_id;
              conds.quant_dtype     = quant_dtype;
              conds.mixed_precision = false;
              conds.memory_mb       = mem_mb;
              conds.context_length  = ctx_len;
              conds.thermal         = thermal;
              conds.latency_mode    = latency;
              conds.battery_pct     = (thermal == ir::ThermalState::Hot) ? 20 : 90;

              CompiledGraph cg = compileOneGraph(sourceGraph, conds);
              cg.conditions   = conds;
              family.push_back(std::move(cg));
              graphIndex++;
            }
          }
        }
      }

      // Mixed precision variant (one per SoC × quant_dtype if enabled)
      if (spec.include_mixed_precision &&
          (quant_dtype == ir::DType::I4 || quant_dtype == ir::DType::I8)) {
        for (auto ctx_len : spec.context_lengths) {
          GraphConditions conds;
          conds.soc_id          = soc_id;
          conds.quant_dtype     = quant_dtype;
          conds.mixed_precision = true;
          conds.memory_mb       = spec.memory_budgets_mb.back();
          conds.context_length  = ctx_len;
          conds.thermal         = ir::ThermalState::Nominal;
          conds.latency_mode    = LatencyMode::Interactive;
          conds.battery_pct     = 90;

          CompiledGraph cg = compileOneGraph(sourceGraph, conds);
          cg.conditions   = conds;
          family.push_back(std::move(cg));
          graphIndex++;
        }
      }
    }
  }

  // Pruning: remove redundant graphs
  if (spec.enable_pruning && spec.max_graphs > 0 && family.size() > spec.max_graphs) {
    prunFamily(family, spec.pruning_threshold_pct);
    if (family.size() > spec.max_graphs) {
      family.resize(spec.max_graphs);
    }
  }

  // Assign final indices
  for (size_t i = 0; i < family.size(); ++i) {
    // (index is implicit from position in vector)
  }

  auto t1 = std::chrono::steady_clock::now();
  auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  std::cout << "[generator] Generated " << family.size()
            << " graphs in " << ms << "ms\n";

  printFamilySummary(family);
  return family;
}

// ---------------------------------------------------------------------------
// compileOneGraph — compile one condition set
// ---------------------------------------------------------------------------
CompiledGraph GraphFamilyGenerator::compileOneGraph(
    const ir::IRGraph& source,
    const GraphConditions& conds)
{
  CompiledGraph cg;
  cg.conditions = conds;

  // Clone the source graph
  auto graph = source.clone();
  if (!graph) {
    std::cerr << "[generator] Clone failed\n";
    return cg;
  }

  // Get device profile
  const analysis::DeviceProfile* dev = costModel_.getProfile(conds.soc_id);

  // Build and run the optimization pipeline for this condition
  passes::PassManager pm;
  passes::PassOptions opts;
  opts.quant_group_size = 128;
  opts.context_lengths  = {conds.context_length};

  if (dev) {
    opts.device = dev;
    opts.memory_budget = (uint64_t)conds.memory_mb * 1024 * 1024;

    // Graph-level passes (always run)
    pm.addGraphPasses();

    // Context-length specific pass
    if (conds.context_length >= 4096) {
      pm.addPass(passes::makeStreamingAttentionPass(512));
    }

    // Quantization passes
    pm.addQuantizationPasses(conds.quant_dtype, conds.mixed_precision);

    // Memory passes
    pm.addMemoryPasses();

    // Hardware scheduling (depends on thermal/battery)
    pm.addSchedulingPasses(*dev, conds.battery_pct, conds.thermal);

    // Target-specific lowering
    pm.addLoweringPasses(*dev);
  } else {
    // Generic pipeline
    pm.addGraphPasses();
    pm.addQuantizationPasses(conds.quant_dtype, false);
    pm.addMemoryPasses();
  }

  auto results = pm.run(*graph, opts);

  // Count failed passes
  int failed = 0;
  for (auto& r : results) if (!r.success) failed++;
  if (failed > 0) {
    std::cerr << "[generator]   WARNING: " << failed << " passes failed for "
              << conds.toKey() << "\n";
  }

  // Estimate performance using cost model
  if (dev) {
    float total_ms = 0.0f;
    graph->forEachNode([&](ir::IRNode& n) {
      if (n.op == ir::OpCode::Constant || n.op == ir::OpCode::Input ||
          n.op == ir::OpCode::Output   || ir::opCodeIsHardwareHint(n.op)) return;

      float node_ms = 0.0f;
      switch (n.assigned_unit) {
        case ir::ExecUnit::NPU: case ir::ExecUnit::APU:
          node_ms = (n.cost_npu_ms >= 0) ? n.cost_npu_ms : n.cost_cpu_ms;
          break;
        case ir::ExecUnit::GPU:
          node_ms = (n.cost_gpu_ms >= 0) ? n.cost_gpu_ms : n.cost_cpu_ms;
          break;
        case ir::ExecUnit::DSP:
          node_ms = (n.cost_dsp_ms >= 0) ? n.cost_dsp_ms : n.cost_cpu_ms;
          break;
        case ir::ExecUnit::ANE:
          node_ms = (n.cost_ane_ms >= 0) ? n.cost_ane_ms : n.cost_cpu_ms;
          break;
        default:
          node_ms = (n.cost_cpu_ms >= 0) ? n.cost_cpu_ms : 1.0f;
          break;
      }
      if (node_ms >= 0) total_ms += node_ms;
    });

    // TTFT ≈ total forward pass × num_layers (prefill approximation)
    uint32_t layers = std::max(1u, source.num_layers);
    cg.estimated_ttft_ms = total_ms * layers;
    cg.estimated_tpot_ms = total_ms;  // decode = one token = one forward pass
  }

  // Peak memory
  cg.peak_memory_bytes = graph->activation_bytes + graph->weight_bytes;

  // Serialize the compiled graph
  cg.serialized_graph = graph->serialize();
  cg.graph = std::move(graph);

  return cg;
}

// ---------------------------------------------------------------------------
// prunFamily — remove redundant graphs
// ---------------------------------------------------------------------------
void GraphFamilyGenerator::prunFamily(std::vector<CompiledGraph>& family,
                                      float threshold_pct)
{
  if (family.empty()) return;

  std::vector<bool> keep(family.size(), true);
  float threshold = threshold_pct / 100.0f;

  for (size_t i = 0; i < family.size(); ++i) {
    if (!keep[i]) continue;
    for (size_t j = i + 1; j < family.size(); ++j) {
      if (!keep[j]) continue;

      auto& a = family[i];
      auto& b = family[j];

      // Same SoC and context length
      if (a.conditions.soc_id != b.conditions.soc_id ||
          a.conditions.context_length != b.conditions.context_length) continue;

      // If a and b have very similar latency → drop the slower one
      float faster = std::min(a.estimated_tpot_ms, b.estimated_tpot_ms);
      float slower = std::max(a.estimated_tpot_ms, b.estimated_tpot_ms);
      if (faster > 0 && (slower - faster) / faster < threshold) {
        // Keep the one with lower memory
        if (a.peak_memory_bytes <= b.peak_memory_bytes) {
          keep[j] = false;
        } else {
          keep[i] = false;
          break;
        }
      }
    }
  }

  std::vector<CompiledGraph> pruned;
  for (size_t i = 0; i < family.size(); ++i) {
    if (keep[i]) pruned.push_back(std::move(family[i]));
  }
  family = std::move(pruned);
  std::cout << "[generator] After pruning: " << family.size() << " graphs\n";
}

// ---------------------------------------------------------------------------
// printFamilySummary
// ---------------------------------------------------------------------------
void GraphFamilyGenerator::printFamilySummary(const std::vector<CompiledGraph>& family) {
  std::cout << "\n[generator] Graph Family Summary:\n";
  std::cout << std::left
            << std::setw(3)  << "#"
            << std::setw(45) << "Conditions"
            << std::setw(10) << "TTFT(ms)"
            << std::setw(10) << "TPOT(ms)"
            << std::setw(10) << "Peak(MB)"
            << "\n";
  std::cout << std::string(78, '-') << "\n";

  for (size_t i = 0; i < family.size(); ++i) {
    auto& cg = family[i];
    std::cout << std::left
              << std::setw(3)  << i
              << std::setw(45) << cg.conditions.toHumanString().substr(0, 44)
              << std::setw(10) << std::fixed << std::setprecision(1)
                               << cg.estimated_ttft_ms
              << std::setw(10) << cg.estimated_tpot_ms
              << std::setw(10) << cg.peak_memory_bytes / 1024 / 1024
              << "\n";
  }
  std::cout << "\n";
}

} // namespace generator
} // namespace armcc
