// ============================================================================
// ARM AI Compiler — Pass Manager (implementation)
// ============================================================================
#include "passes/pass_manager.h"
#include <iostream>
#include <iomanip>
#include <chrono>

namespace armcc {
namespace passes {

using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// PassResult::toString
// ---------------------------------------------------------------------------
std::string PassResult::toString() const {
  std::ostringstream oss;
  oss << std::left << std::setw(40) << pass_name
      << (success ? "OK  " : "FAIL")
      << "  " << std::setw(6) << elapsed.count() << "µs";
  if (changed) {
    oss << "  [-" << nodes_removed
        << " +" << nodes_added
        << " ~" << nodes_fused << " fused]";
  }
  if (!error_message.empty()) {
    oss << "  ERR: " << error_message;
  }
  return oss.str();
}

// ---------------------------------------------------------------------------
// PassManager: addPass
// ---------------------------------------------------------------------------
void PassManager::addPass(std::unique_ptr<Pass> pass) {
  passes_.push_back(std::move(pass));
}

void PassManager::addPass(std::unique_ptr<Pass> pass, size_t pos) {
  passes_.insert(passes_.begin() + pos, std::move(pass));
}

// ---------------------------------------------------------------------------
// Preset pipelines
// ---------------------------------------------------------------------------
void PassManager::addGraphPasses() {
  // Device-agnostic structural optimizations — run in this order
  addPass(makeShapeInferencePass());
  addPass(makeConstantFoldingPass());
  addPass(makeDeadCodeEliminationPass());
  addPass(makeOperatorFusionPass());
  addPass(makeKVCachePlanningPass());
}

void PassManager::addQuantizationPasses(ir::DType dtype, bool mixed) {
  switch (dtype) {
    case ir::DType::I8:
      addPass(makeINT8QuantizationPass());
      break;
    case ir::DType::I4:
      addPass(makeINT4QuantizationPass(128));
      if (mixed) addPass(makeMixedPrecisionPass({}));
      break;
    case ir::DType::F8_E4M3:
      addPass(makeFP8QuantizationPass());
      break;
    default:
      if (mixed) addPass(makeMixedPrecisionPass({}));
      break;
  }
}

void PassManager::addMemoryPasses() {
  addPass(makeTensorLifetimeAnalysisPass());
  addPass(makeMemoryAliasingPass());
  addPass(makeMemoryLayoutPass());
  addPass(makeZeroCopyPass());
  addPass(makePrefetchInsertionPass());
}

void PassManager::addLoweringPasses(const analysis::DeviceProfile& dev) {
  addPass(makeLayoutOptimizationPass(dev));
  addPass(makeKernelSelectionPass(dev));
  addPass(makeTilingPass(dev));
  addPass(makeVectorizationPass(dev));
}

void PassManager::addSchedulingPasses(const analysis::DeviceProfile& dev,
                                      int battery_pct,
                                      ir::ThermalState thermal) {
  addPass(makeHardwareFusionPass(dev));
  // HardwareFusionPass uses the cost model internally
  (void)battery_pct; (void)thermal;
}

void PassManager::addFullPipeline(const analysis::DeviceProfile& dev,
                                  ir::DType quant_dtype,
                                  int battery_pct,
                                  ir::ThermalState thermal) {
  addGraphPasses();
  addQuantizationPasses(quant_dtype, /*mixed=*/quant_dtype == ir::DType::I4);
  addMemoryPasses();
  addSchedulingPasses(dev, battery_pct, thermal);
  addLoweringPasses(dev);
}

// ---------------------------------------------------------------------------
// PassManager: run
// ---------------------------------------------------------------------------
std::vector<PassResult> PassManager::run(ir::IRGraph& graph,
                                         const PassOptions& opts) {
  std::vector<PassResult> results;
  results.reserve(passes_.size());

  for (auto& pass : passes_) {
    if (!pass->isApplicable(graph)) {
      PassResult r;
      r.pass_name = pass->name();
      r.success   = true;
      r.changed   = false;
      r.elapsed   = std::chrono::microseconds(0);
      results.push_back(r);
      continue;
    }

    if (opts.verbose) {
      std::cout << "[armcc] Running pass: " << pass->name() << " ...\n";
    }

    size_t nodesBefore = graph.numNodes();
    auto t0 = Clock::now();

    PassResult r = pass->run(graph, opts);

    auto t1 = Clock::now();
    r.elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    r.nodes_removed = (nodesBefore > graph.numNodes())
                      ? (nodesBefore - graph.numNodes()) : 0;

    graph.applied_passes.push_back(pass->name());
    results.push_back(r);

    if (opts.verbose) {
      std::cout << "  " << r.toString() << "\n";
    }
  }

  return results;
}

// ---------------------------------------------------------------------------
// PassManager: runUntilFixed
// ---------------------------------------------------------------------------
std::vector<PassResult> PassManager::runUntilFixed(ir::IRGraph& graph,
                                                    const PassOptions& opts,
                                                    int max_iterations) {
  std::vector<PassResult> all_results;
  for (int iter = 0; iter < max_iterations; ++iter) {
    bool any_changed = false;
    auto results = run(graph, opts);
    for (auto& r : results) {
      if (r.changed) any_changed = true;
      all_results.push_back(std::move(r));
    }
    if (!any_changed) break;
  }
  return all_results;
}

// ---------------------------------------------------------------------------
// PassManager: printPipeline
// ---------------------------------------------------------------------------
void PassManager::printPipeline() const {
  std::cout << "Pass pipeline (" << passes_.size() << " passes):\n";
  for (size_t i = 0; i < passes_.size(); ++i) {
    std::cout << "  [" << std::setw(2) << i << "] "
              << passes_[i]->name() << "\n";
  }
}

} // namespace passes
} // namespace armcc
