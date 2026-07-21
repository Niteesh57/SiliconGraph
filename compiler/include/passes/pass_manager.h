// ============================================================================
// ARM AI Compiler — Pass Manager
//
// The PassManager orchestrates all optimization passes over an IRGraph.
// Passes are independent, composable, and run in a defined order.
//
// Three pipeline levels:
//   1. GraphPipeline   — structural graph transformations
//   2. QuantPipeline   — quantization passes
//   3. LoweringPipeline — target-specific lowering (layout, kernels)
// ============================================================================
#pragma once

#include "arm_ir/graph.h"
#include "analysis/cost_model.h"

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>

namespace armcc {
namespace passes {

// ---------------------------------------------------------------------------
// Pass result
// ---------------------------------------------------------------------------
struct PassResult {
  std::string pass_name;
  bool        changed       = false;  // Did the pass modify the graph?
  uint32_t    nodes_removed = 0;
  uint32_t    nodes_added   = 0;
  uint32_t    nodes_fused   = 0;
  std::string error_message;          // empty = success
  bool        success       = true;
  std::chrono::microseconds elapsed;

  std::string toString() const;
};

// ---------------------------------------------------------------------------
// PassOptions — configuration shared across all passes
// ---------------------------------------------------------------------------
struct PassOptions {
  // Target device profile (may be null for device-agnostic passes)
  const analysis::DeviceProfile* device = nullptr;

  // Quantization settings
  ir::DType  default_quant_dtype  = ir::DType::I8;
  int32_t    quant_group_size     = 128;    // For INT4 grouped quantization
  float      quant_error_threshold = 0.01f; // Sensitivity threshold

  // Memory budget (bytes; 0 = unlimited)
  uint64_t   memory_budget = 0;

  // Context lengths to optimize for
  std::vector<uint32_t> context_lengths = {512, 2048};

  // Enable verbose pass logging
  bool verbose = false;

  // Allow experimental passes
  bool allow_experimental = false;
};

// ---------------------------------------------------------------------------
// Base pass interface
// ---------------------------------------------------------------------------
class Pass {
public:
  explicit Pass(std::string name) : name_(std::move(name)) {}
  virtual ~Pass() = default;

  const std::string& name() const { return name_; }

  // Run the pass over the graph.
  // Returns a PassResult describing what changed.
  virtual PassResult run(ir::IRGraph& graph, const PassOptions& opts) = 0;

  // Can this pass run on this graph? (quick pre-check)
  virtual bool isApplicable(const ir::IRGraph& graph) const { return true; }

  // Human-readable description
  virtual std::string description() const { return name_; }

protected:
  std::string name_;
};

// ---------------------------------------------------------------------------
// PassManager
//
// Usage:
//   PassManager pm;
//   pm.addGraphPasses();
//   pm.addQuantPasses(DType::I8);
//   pm.addLoweringPasses(deviceProfile);
//   auto results = pm.run(graph, opts);
// ---------------------------------------------------------------------------
class PassManager {
public:
  PassManager() = default;

  // ── Registration ──────────────────────────────────────────────────────────
  void addPass(std::unique_ptr<Pass> pass);
  void addPass(std::unique_ptr<Pass> pass, size_t position);

  // ── Preset pipelines ──────────────────────────────────────────────────────
  // Standard graph optimization passes (device-agnostic)
  void addGraphPasses();

  // Quantization passes (choose dtype: INT8, INT4, FP8, or MIXED)
  void addQuantizationPasses(ir::DType dtype, bool mixed = false);

  // Memory optimization passes
  void addMemoryPasses();

  // Device-specific lowering (layout, kernel selection, fusion)
  void addLoweringPasses(const analysis::DeviceProfile& dev);

  // Hardware scheduling passes (assigns ExecUnit per node)
  void addSchedulingPasses(const analysis::DeviceProfile& dev,
                           int battery_pct,
                           ir::ThermalState thermal);

  // Full default pipeline (graph → quant → memory → lowering → scheduling)
  void addFullPipeline(const analysis::DeviceProfile& dev,
                       ir::DType quant_dtype,
                       int battery_pct = 90,
                       ir::ThermalState thermal = ir::ThermalState::Nominal);

  // ── Execution ─────────────────────────────────────────────────────────────
  std::vector<PassResult> run(ir::IRGraph& graph, const PassOptions& opts);

  // Run until convergence (keep running passes until no more changes)
  std::vector<PassResult> runUntilFixed(ir::IRGraph& graph,
                                        const PassOptions& opts,
                                        int max_iterations = 10);

  // ── Inspection ────────────────────────────────────────────────────────────
  size_t numPasses() const { return passes_.size(); }
  void   printPipeline() const;

private:
  std::vector<std::unique_ptr<Pass>> passes_;
};

// ---------------------------------------------------------------------------
// Pass registration helper — used by pass implementations
// ---------------------------------------------------------------------------
// Concrete pass factories (defined in their respective .cpp files)
std::unique_ptr<Pass> makeConstantFoldingPass();
std::unique_ptr<Pass> makeDeadCodeEliminationPass();
std::unique_ptr<Pass> makeOperatorFusionPass();
std::unique_ptr<Pass> makeHardwareFusionPass(const analysis::DeviceProfile& dev);
std::unique_ptr<Pass> makeShapeInferencePass();
std::unique_ptr<Pass> makeLayoutOptimizationPass(const analysis::DeviceProfile& dev);
std::unique_ptr<Pass> makeKVCachePlanningPass();
std::unique_ptr<Pass> makeStreamingAttentionPass(uint32_t chunk_size = 512);

std::unique_ptr<Pass> makeINT8QuantizationPass();
std::unique_ptr<Pass> makeINT4QuantizationPass(int32_t group_size = 128);
std::unique_ptr<Pass> makeFP8QuantizationPass();
std::unique_ptr<Pass> makeMixedPrecisionPass(const std::vector<float>& sensitivity);

std::unique_ptr<Pass> makeTensorLifetimeAnalysisPass();
std::unique_ptr<Pass> makeMemoryAliasingPass();
std::unique_ptr<Pass> makeMemoryLayoutPass();
std::unique_ptr<Pass> makePrefetchInsertionPass();
std::unique_ptr<Pass> makeZeroCopyPass();

std::unique_ptr<Pass> makeKernelSelectionPass(const analysis::DeviceProfile& dev);
std::unique_ptr<Pass> makeTilingPass(const analysis::DeviceProfile& dev);
std::unique_ptr<Pass> makeVectorizationPass(const analysis::DeviceProfile& dev);

} // namespace passes
} // namespace armcc
