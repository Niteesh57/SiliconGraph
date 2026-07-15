// ============================================================================
// ARM AI Compiler — Analysis: Graph Analyzer
//
// The GraphAnalyzer runs over an IRGraph and computes:
//   - Total FLOPs
//   - Memory pressure per layer
//   - Op distribution (CPU-friendly vs GPU-friendly vs NPU-friendly)
//   - Bottleneck identification (compute-bound vs memory-bound ops)
//   - Model family classification (Llama-style, GPT2-style, etc.)
//   - Quantization sensitivity analysis
// ============================================================================
#pragma once

#include "arm_ir/graph.h"
#include "arm_ir/node.h"
#include "arm_ir/types.h"

#include <string>
#include <vector>
#include <unordered_map>

namespace armcc {
namespace analysis {

// ---------------------------------------------------------------------------
// Per-node analysis result
// ---------------------------------------------------------------------------
struct NodeAnalysis {
  uint32_t  nodeId;
  uint64_t  flops;           // Floating point operations
  uint64_t  memory_bytes;    // Input + output tensor memory traffic
  float     arithmetic_intensity;  // FLOPs / byte (roofline model)
  bool      compute_bound;
  bool      memory_bound;
  bool      npu_friendly;    // True if op is well-suited for NPU
  bool      gpu_friendly;
  bool      dsp_friendly;
};

// ---------------------------------------------------------------------------
// Graph-level analysis summary
// ---------------------------------------------------------------------------
struct GraphAnalysisSummary {
  // FLOPs breakdown
  uint64_t  total_flops         = 0;
  uint64_t  attention_flops     = 0;
  uint64_t  ffn_flops           = 0;
  uint64_t  embedding_flops     = 0;
  uint64_t  norm_flops          = 0;
  uint64_t  other_flops         = 0;

  // Memory
  uint64_t  weight_bytes        = 0;   // Static weights
  uint64_t  peak_activation_bytes = 0; // Peak live activation memory

  // Op counts
  uint32_t  num_matmuls         = 0;
  uint32_t  num_attention_ops   = 0;
  uint32_t  num_layernorm_ops   = 0;
  uint32_t  num_elementwise     = 0;
  uint32_t  total_nodes         = 0;

  // Roofline
  float     avg_arithmetic_intensity = 0.0f;
  bool      is_memory_bandwidth_bound = false;

  // Architecture
  ir::ModelFamily    model_family    = ir::ModelFamily::Unknown;
  uint32_t           num_layers      = 0;
  uint32_t           hidden_size     = 0;
  uint32_t           kv_head_ratio   = 1;   // num_heads / num_kv_heads (GQA factor)

  // Quantization sensitivity (layer index → sensitivity score 0.0-1.0)
  // Higher = more sensitive to quantization (keep in higher precision)
  std::vector<float> quant_sensitivity;

  std::string toString() const;
};

// ---------------------------------------------------------------------------
// GraphAnalyzer
// ---------------------------------------------------------------------------
class GraphAnalyzer {
public:
  explicit GraphAnalyzer(const ir::IRGraph& graph);

  // Run all analysis passes
  GraphAnalysisSummary analyze();

  // Individual analysis steps (can be called independently)
  void computeFLOPs();
  void computeMemoryPressure();
  void classifyModelFamily();
  void computeQuantizationSensitivity();
  void identifyBottlenecks();

  // Per-node results (available after analyze())
  const std::vector<NodeAnalysis>& nodeResults() const { return nodeResults_; }

  // Get the summary
  const GraphAnalysisSummary& summary() const { return summary_; }

private:
  const ir::IRGraph&           graph_;
  GraphAnalysisSummary         summary_;
  std::vector<NodeAnalysis>    nodeResults_;

  uint64_t computeNodeFLOPs(const ir::IRNode& node) const;
  uint64_t computeNodeMemory(const ir::IRNode& node) const;
  bool     isNPUFriendly(ir::OpCode op) const;
  bool     isGPUFriendly(ir::OpCode op) const;
  bool     isDSPFriendly(ir::OpCode op) const;
};

} // namespace analysis
} // namespace armcc
