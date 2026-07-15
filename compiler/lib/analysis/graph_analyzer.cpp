// ============================================================================
// ARM AI Compiler — Analysis: Graph Analyzer (implementation)
// ============================================================================
#include "analysis/graph_analyzer.h"
#include "arm_ir/ops.h"
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace armcc {
namespace analysis {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
GraphAnalyzer::GraphAnalyzer(const ir::IRGraph& graph) : graph_(graph) {}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
GraphAnalysisSummary GraphAnalyzer::analyze() {
  summary_ = {};
  summary_.total_nodes = (uint32_t)graph_.numNodes();

  classifyModelFamily();
  computeFLOPs();
  computeMemoryPressure();
  computeQuantizationSensitivity();
  identifyBottlenecks();

  return summary_;
}

// ---------------------------------------------------------------------------
// Model family classification
// ---------------------------------------------------------------------------
void GraphAnalyzer::classifyModelFamily() {
  // Use the graph's pre-set family if available
  if (graph_.model_family != ir::ModelFamily::Unknown) {
    summary_.model_family = graph_.model_family;
    return;
  }

  // Count op types to infer family
  int has_rmsnorm = 0, has_layernorm = 0, has_gqa = 0, has_mha = 0;
  int has_swiglu = 0, has_gelu = 0;

  for (auto& n : graph_.nodes) {
    switch (n->op) {
      case ir::OpCode::RMSNorm:
      case ir::OpCode::RMSNorm_Scale:       has_rmsnorm++;  break;
      case ir::OpCode::LayerNorm:            has_layernorm++; break;
      case ir::OpCode::GroupQueryAttention:  has_gqa++;      break;
      case ir::OpCode::MultiHeadAttention:   has_mha++;      break;
      case ir::OpCode::FFN_SwiGLU:
      case ir::OpCode::SiLU:                has_swiglu++;   break;
      case ir::OpCode::FFN_GeLU:
      case ir::OpCode::GeLU:
      case ir::OpCode::GELU_Approx:         has_gelu++;     break;
      default: break;
    }
  }

  if (has_rmsnorm > 0 && has_gqa > 0)      summary_.model_family = ir::ModelFamily::LlamaStyle;
  else if (has_rmsnorm > 0 && has_gelu > 0) summary_.model_family = ir::ModelFamily::GemmaStyle;
  else if (has_layernorm > 0)               summary_.model_family = ir::ModelFamily::GPT2Style;
  else                                       summary_.model_family = ir::ModelFamily::Unknown;

  // Propagate to layers
  summary_.num_layers  = graph_.num_layers;
  summary_.hidden_size = graph_.hidden_size;
  summary_.kv_head_ratio = (graph_.num_kv_heads > 0 && graph_.num_heads > 0)
                           ? graph_.num_heads / graph_.num_kv_heads : 1;
}

// ---------------------------------------------------------------------------
// FLOPs computation
// ---------------------------------------------------------------------------
void GraphAnalyzer::computeFLOPs() {
  summary_.total_flops = 0;
  nodeResults_.resize(graph_.numNodes());
  size_t idx = 0;

  const_cast<ir::IRGraph&>(graph_).forEachNode([&](ir::IRNode& node) {
    if (idx >= nodeResults_.size()) nodeResults_.resize(idx + 1);
    NodeAnalysis& na = nodeResults_[idx++];
    na.nodeId = node.id;
    na.flops  = computeNodeFLOPs(node);
    na.memory_bytes = computeNodeMemory(node);

    // Arithmetic intensity = FLOPs / bytes
    na.arithmetic_intensity = (na.memory_bytes > 0)
      ? (float)na.flops / (float)na.memory_bytes : 0.0f;
    na.compute_bound = na.arithmetic_intensity > 50.0f;  // >50 FLOPs/byte → compute-bound
    na.memory_bound  = !na.compute_bound;
    na.npu_friendly  = isNPUFriendly(node.op);
    na.gpu_friendly  = isGPUFriendly(node.op);
    na.dsp_friendly  = isDSPFriendly(node.op);

    // Accumulate
    summary_.total_flops += na.flops;
    switch (node.op) {
      case ir::OpCode::MultiHeadAttention:
      case ir::OpCode::GroupQueryAttention:
      case ir::OpCode::FlashAttention:
      case ir::OpCode::Attention_Masked:
        summary_.attention_flops += na.flops;
        summary_.num_attention_ops++;
        break;
      case ir::OpCode::FFN_SwiGLU:
      case ir::OpCode::FFN_GeLU:
        summary_.ffn_flops += na.flops;
        break;
      case ir::OpCode::Embedding:
        summary_.embedding_flops += na.flops;
        break;
      case ir::OpCode::LayerNorm:
      case ir::OpCode::RMSNorm:
      case ir::OpCode::RMSNorm_Scale:
        summary_.norm_flops += na.flops;
        summary_.num_layernorm_ops++;
        break;
      case ir::OpCode::MatMul:
      case ir::OpCode::GEMM:
      case ir::OpCode::BatchMatMul:
        summary_.ffn_flops += na.flops;  // Most non-attn matmuls are FFN
        summary_.num_matmuls++;
        break;
      case ir::OpCode::Add:
      case ir::OpCode::Mul:
      case ir::OpCode::Sub:
      case ir::OpCode::SiLU:
      case ir::OpCode::GeLU:
      case ir::OpCode::ReLU:
        summary_.other_flops += na.flops;
        summary_.num_elementwise++;
        break;
      default:
        summary_.other_flops += na.flops;
        break;
    }
  });
}

uint64_t GraphAnalyzer::computeNodeFLOPs(const ir::IRNode& node) const {
  switch (node.op) {
    case ir::OpCode::MatMul:
    case ir::OpCode::GEMM:
    case ir::OpCode::BatchMatMul: {
      // FLOPs = 2 * M * K * N (from output shape + input shape)
      if (node.outputs.empty() || node.inputs.empty()) return 0;
      const ir::IRTensor* out = const_cast<ir::IRGraph&>(graph_).getTensor(node.outputs[0]);
      const ir::IRTensor* inp = const_cast<ir::IRGraph&>(graph_).getTensor(node.inputs[0]);
      if (!out || !inp) return 0;
      int64_t rank = (int64_t)out->shape.rank();
      if (rank < 2 || inp->shape.rank() < 1) return 0;
      int64_t M = out->shape.dims[rank-2];
      int64_t N = out->shape.dims[rank-1];
      int64_t K = inp->shape.dims[inp->shape.rank()-1];
      if (M < 0 || N < 0 || K < 0) return 0;
      // Batch dimensions
      int64_t batch = 1;
      for (int64_t i = 0; i < rank - 2; ++i)
        if (out->shape.dims[i] > 0) batch *= out->shape.dims[i];
      return 2ULL * batch * M * K * N;
    }

    case ir::OpCode::GroupQueryAttention:
    case ir::OpCode::MultiHeadAttention:
    case ir::OpCode::FlashAttention:
    case ir::OpCode::Attention_Masked: {
      // Approximate: 4 * seq^2 * d_model per layer
      int64_t seq = 2048, d = graph_.hidden_size > 0 ? graph_.hidden_size : 4096;
      return 4ULL * seq * seq * d;
    }

    case ir::OpCode::FFN_SwiGLU:
    case ir::OpCode::FFN_GeLU: {
      // 2 * seq * d_model * intermediate_size * 2 (up + gate + down)
      int64_t seq = 1, d = graph_.hidden_size > 0 ? graph_.hidden_size : 4096;
      int64_t inter = graph_.intermediate_sz > 0 ? graph_.intermediate_sz : d * 4;
      return 6ULL * seq * d * inter;
    }

    case ir::OpCode::LayerNorm:
    case ir::OpCode::RMSNorm:
    case ir::OpCode::RMSNorm_Scale: {
      // Roughly 5 FLOPs per element (mean, var, scale, shift, normalize)
      if (node.inputs.empty()) return 0;
      const ir::IRTensor* t = const_cast<ir::IRGraph&>(graph_).getTensor(node.inputs[0]);
      if (!t) return 0;
      int64_t n = t->shape.numElements();
      return n > 0 ? 5ULL * n : 0;
    }

    case ir::OpCode::Softmax: {
      // 3 FLOPs/element (exp + sum + divide)
      if (node.inputs.empty()) return 0;
      const ir::IRTensor* t = const_cast<ir::IRGraph&>(graph_).getTensor(node.inputs[0]);
      if (!t) return 0;
      int64_t n = t->shape.numElements();
      return n > 0 ? 3ULL * n : 0;
    }

    default: {
      // Elementwise: 1 FLOP per element
      if (node.outputs.empty()) return 0;
      const ir::IRTensor* t = const_cast<ir::IRGraph&>(graph_).getTensor(node.outputs[0]);
      if (!t) return 0;
      int64_t n = t->shape.numElements();
      return n > 0 ? (uint64_t)n : 0;
    }
  }
}

uint64_t GraphAnalyzer::computeNodeMemory(const ir::IRNode& node) const {
  uint64_t total = 0;
  auto addTensor = [&](uint32_t tid) {
    const ir::IRTensor* t = const_cast<ir::IRGraph&>(graph_).getTensor(tid);
    if (!t) return;
    int64_t elems = t->shape.numElements();
    if (elems <= 0) return;
    size_t elemSize = ir::dtypeElementSize(t->dtype);
    if (elemSize == 0) elemSize = 1;  // Sub-byte: approximate as 1
    total += (uint64_t)elems * elemSize;
  };
  for (auto tid : node.inputs)  addTensor(tid);
  for (auto tid : node.outputs) addTensor(tid);
  return total;
}

// ---------------------------------------------------------------------------
// Memory pressure
// ---------------------------------------------------------------------------
void GraphAnalyzer::computeMemoryPressure() {
  // Weight bytes: sum of all constant tensors
  for (auto& t : graph_.tensors) {
    if (t->is_weight) {
      int64_t elems = t->shape.numElements();
      if (elems > 0) {
        size_t sz = ir::dtypeElementSize(t->dtype);
        if (sz == 0) sz = 1;
        summary_.weight_bytes += (uint64_t)elems * sz;
      }
    }
  }

  // Peak activation memory: simplified — estimate from hidden_size × 2 × seq × num_layers
  int64_t d = graph_.hidden_size > 0 ? graph_.hidden_size : 4096;
  summary_.peak_activation_bytes = 2ULL * d * 2048 * 4;  // F32, seq=2048

  // Roofline analysis
  if (!nodeResults_.empty()) {
    float sum_ai = 0;
    for (auto& na : nodeResults_) sum_ai += na.arithmetic_intensity;
    summary_.avg_arithmetic_intensity = sum_ai / nodeResults_.size();
  }

  // Memory bandwidth bound if avg AI < 50
  summary_.is_memory_bandwidth_bound = summary_.avg_arithmetic_intensity < 50.0f;
}

// ---------------------------------------------------------------------------
// Quantization sensitivity
// ---------------------------------------------------------------------------
void GraphAnalyzer::computeQuantizationSensitivity() {
  // Heuristic sensitivity scores by op type and position
  // Higher sensitivity → keep in higher precision
  summary_.quant_sensitivity.clear();

  uint32_t layer = 0;
  const_cast<ir::IRGraph&>(graph_).forEachNode([&](ir::IRNode& n) {
    float s = 0.3f;  // Default low sensitivity

    // Embedding layers are sensitive
    if (n.op == ir::OpCode::Embedding) s = 0.9f;

    // LM head (last MatMul before logits) is very sensitive
    bool is_last_matmul = (n.id == graph_.topoOrder.back());
    if ((n.op == ir::OpCode::MatMul || n.op == ir::OpCode::GEMM) && is_last_matmul)
      s = 0.95f;

    // First few layers: higher sensitivity
    if (graph_.num_layers > 0 && layer < graph_.num_layers / 8) s = 0.7f;

    // Last few layers: higher sensitivity
    if (graph_.num_layers > 0 && layer > graph_.num_layers * 7 / 8) s = 0.75f;

    // Attention Q projection: sensitive to quantization
    if (n.op == ir::OpCode::GroupQueryAttention ||
        n.op == ir::OpCode::MultiHeadAttention) s = 0.6f;

    // Norm layers: always sensitive (avoid quantizing them)
    if (n.op == ir::OpCode::RMSNorm || n.op == ir::OpCode::LayerNorm) s = 0.85f;

    summary_.quant_sensitivity.push_back(s);
    layer++;
  });
}

// ---------------------------------------------------------------------------
// Bottleneck identification
// ---------------------------------------------------------------------------
void GraphAnalyzer::identifyBottlenecks() {
  for (auto& na : nodeResults_) {
    na.npu_friendly = isNPUFriendly(
        const_cast<ir::IRGraph&>(graph_).getNode(na.nodeId)->op);
    na.gpu_friendly = isGPUFriendly(
        const_cast<ir::IRGraph&>(graph_).getNode(na.nodeId)->op);
  }
}

bool GraphAnalyzer::isNPUFriendly(ir::OpCode op) const {
  return op == ir::OpCode::MatMul         ||
         op == ir::OpCode::GEMM           ||
         op == ir::OpCode::BatchMatMul    ||
         op == ir::OpCode::GroupQueryAttention ||
         op == ir::OpCode::MultiHeadAttention  ||
         op == ir::OpCode::Conv1D         ||
         op == ir::OpCode::Conv2D         ||
         op == ir::OpCode::DepthwiseConv2D;
}

bool GraphAnalyzer::isGPUFriendly(ir::OpCode op) const {
  return op == ir::OpCode::MatMul         ||
         op == ir::OpCode::GEMM           ||
         op == ir::OpCode::BatchMatMul    ||
         op == ir::OpCode::Softmax        ||
         op == ir::OpCode::GeLU           ||
         op == ir::OpCode::SiLU           ||
         op == ir::OpCode::FFN_SwiGLU     ||
         op == ir::OpCode::FFN_GeLU       ||
         op == ir::OpCode::Add            ||
         op == ir::OpCode::Mul;
}

bool GraphAnalyzer::isDSPFriendly(ir::OpCode op) const {
  return op == ir::OpCode::Embedding      ||
         op == ir::OpCode::RopeEmbedding  ||
         op == ir::OpCode::Softmax        ||
         op == ir::OpCode::ReduceSum      ||
         op == ir::OpCode::ReduceMax;
}

// ---------------------------------------------------------------------------
// Summary toString
// ---------------------------------------------------------------------------
std::string GraphAnalysisSummary::toString() const {
  std::ostringstream oss;
  oss << "family=" << ir::modelFamilyToString(model_family)
      << " layers=" << num_layers
      << " flops=" << total_flops / 1e9 << "G"
      << " weights=" << weight_bytes / 1024 / 1024 << "MB"
      << " attn=" << attention_flops * 100 / std::max(total_flops, 1ULL) << "%"
      << " ffn=" << ffn_flops * 100 / std::max(total_flops, 1ULL) << "%"
      << (is_memory_bandwidth_bound ? " [memory-bound]" : " [compute-bound]");
  return oss.str();
}

} // namespace analysis
} // namespace armcc
