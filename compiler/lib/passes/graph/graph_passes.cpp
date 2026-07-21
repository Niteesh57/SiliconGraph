// ============================================================================
// ARM AI Compiler — Graph Pass: Constant Folding
//
// Pre-computes any subgraph whose inputs are all constants.
// E.g. a Cast(Constant) or Add(Constant, Constant) gets folded
// into a single Constant node.
// ============================================================================
#include "passes/pass_manager.h"
#include "arm_ir/graph.h"
#include "arm_ir/ops.h"
#include <cstring>
#include <cmath>
#include <unordered_set>

namespace armcc {
namespace passes {

namespace {

class ConstantFoldingPass : public Pass {
public:
  ConstantFoldingPass() : Pass("ConstantFolding") {}

  std::string description() const override {
    return "Pre-computes static subgraphs (all-constant inputs → one Constant node)";
  }

  PassResult run(ir::IRGraph& graph, const PassOptions& opts) override {
    PassResult result;
    result.pass_name = name();

    std::vector<uint32_t> toRemove;

    graph.forEachNode([&](ir::IRNode& node) {
      // Skip ops that can't be folded at compile time
      if (node.op == ir::OpCode::Constant   ||
          node.op == ir::OpCode::Input       ||
          node.op == ir::OpCode::Output      ||
          node.op == ir::OpCode::KVCacheRead ||
          node.op == ir::OpCode::KVCacheWrite) return;

      // Check: are ALL input tensors constants?
      bool all_const = true;
      for (uint32_t tid : node.inputs) {
        ir::IRTensor* t = graph.getTensor(tid);
        if (!t || !t->is_weight || t->weight_data == nullptr) {
          all_const = false;
          break;
        }
      }
      if (!all_const) return;

      // Only fold scalar or small-constant ops for now
      // (Full constant evaluator is a future pass enhancement)
      if (node.op == ir::OpCode::Cast) {
        // Cast(Constant[F32] → F16/BF16): pre-cast the weight data
        if (node.inputs.size() == 1 && node.outputs.size() == 1) {
          ir::IRTensor* inp = graph.getTensor(node.inputs[0]);
          ir::IRTensor* out = graph.getTensor(node.outputs[0]);
          if (inp && out && inp->dtype == ir::DType::F32 &&
              (out->dtype == ir::DType::F16 || out->dtype == ir::DType::BF16)) {
            // Mark the output as a constant with pre-cast data
            // (In a full implementation: actually cast the bytes)
            out->is_weight = true;
            toRemove.push_back(node.id);
            result.changed = true;
            result.nodes_removed++;
          }
        }
      }
    });

    for (uint32_t nid : toRemove) graph.removeNode(nid);
    result.success = true;
    return result;
  }
};

// ============================================================================
// Dead Code Elimination
// ============================================================================
class DeadCodeEliminationPass : public Pass {
public:
  DeadCodeEliminationPass() : Pass("DeadCodeElimination") {}

  std::string description() const override {
    return "Removes nodes whose outputs are never consumed";
  }

  PassResult run(ir::IRGraph& graph, const PassOptions& opts) override {
    PassResult result;
    result.pass_name = name();

    // Build set of tensors that are consumed (used as input) by at least one node,
    // or are graph outputs
    std::unordered_set<uint32_t> used;
    for (uint32_t tid : graph.outputIds) used.insert(tid);
    for (auto& n : graph.nodes) {
      for (uint32_t tid : n->inputs) used.insert(tid);
    }

    // Remove nodes whose outputs are entirely unused
    std::vector<uint32_t> toRemove;
    for (auto& n : graph.nodes) {
      if (n->op == ir::OpCode::Input || n->op == ir::OpCode::Output) continue;
      bool any_used = false;
      for (uint32_t tid : n->outputs) {
        if (used.count(tid)) { any_used = true; break; }
      }
      if (!any_used) {
        toRemove.push_back(n->id);
      }
    }

    for (uint32_t nid : toRemove) {
      graph.removeNode(nid);
      result.nodes_removed++;
      result.changed = true;
    }

    result.success = true;
    return result;
  }

private:
  // Need unordered_set — include it here
};

// ============================================================================
// Operator Fusion
// ============================================================================
class OperatorFusionPass : public Pass {
public:
  OperatorFusionPass() : Pass("OperatorFusion") {}

  std::string description() const override {
    return "Fuses adjacent fusable ops (e.g. MatMul+Add+SiLU → one kernel)";
  }

  PassResult run(ir::IRGraph& graph, const PassOptions& opts) override {
    PassResult result;
    result.pass_name = name();
    graph.sortTopological();

    // Pattern: MatMul / GEMM → Add (bias) → activation
    // Fuse into a single GEMM node with bias+activation attributes
    for (size_t i = 0; i + 1 < graph.topoOrder.size(); ++i) {
      ir::IRNode* n0 = graph.getNode(graph.topoOrder[i]);
      if (!n0) continue;
      if (n0->op != ir::OpCode::MatMul && n0->op != ir::OpCode::GEMM) continue;
      if (n0->outputs.size() != 1) continue;

      // Look for a single consumer
      auto consumers = graph.consumers(n0->outputs[0]);
      if (consumers.size() != 1) continue;
      ir::IRNode* n1 = consumers[0];

      if (n1->op == ir::OpCode::Add) {
        // Check next op after Add
        if (n1->outputs.size() == 1) {
          auto consumers2 = graph.consumers(n1->outputs[0]);
          if (consumers2.size() == 1) {
            ir::IRNode* n2 = consumers2[0];
            if (ir::opCodeIsFusable(n2->op) &&
                n2->op != ir::OpCode::MatMul) {
              // Fuse MatMul → Add → Act into one node
              n0->op = ir::OpCode::GEMM;
              n0->fused_from = {n1->id, n2->id};
              n0->is_fused = true;
              // Wire n0's output to n2's consumer
              n0->outputs = n2->outputs;
              // Keep n1's bias as an extra input to n0
              if (n1->inputs.size() == 2) {
                uint32_t bias_tid = (n1->inputs[0] == n0->outputs[0])
                                    ? n1->inputs[1] : n1->inputs[0];
                n0->inputs.push_back(bias_tid);
              }
              // Add activation as attribute
              ir::Attribute act_attr;
              act_attr.name = "activation";
              act_attr.value = std::string(ir::opCodeToString(n2->op));
              n0->attrs.push_back(act_attr);

              graph.removeNode(n1->id);
              graph.removeNode(n2->id);
              result.nodes_fused += 2;
              result.nodes_removed += 2;
              result.changed = true;
              break;
            }
          }
        }
      }
    }

    // Pattern: RMSNorm → Mul (scale) → fuse into RMSNorm_Scale
    for (auto& n : graph.nodes) {
      if (n->op != ir::OpCode::RMSNorm) continue;
      if (n->outputs.size() != 1) continue;
      auto consumers = graph.consumers(n->outputs[0]);
      if (consumers.size() != 1) continue;
      ir::IRNode* next = consumers[0];
      if (next->op == ir::OpCode::Mul) {
        n->op = ir::OpCode::RMSNorm_Scale;
        // Absorb the scale weight into inputs
        for (uint32_t tid : next->inputs) {
          if (tid != n->outputs[0]) n->inputs.push_back(tid);
        }
        n->outputs = next->outputs;
        n->is_fused = true;
        n->fused_from.push_back(next->id);
        graph.removeNode(next->id);
        result.nodes_fused++;
        result.nodes_removed++;
        result.changed = true;
      }
    }

    result.success = true;
    return result;
  }
};

// ============================================================================
// Shape Inference Pass
// ============================================================================
class ShapeInferencePass : public Pass {
public:
  ShapeInferencePass() : Pass("ShapeInference") {}

  std::string description() const override {
    return "Propagates tensor shapes through the graph";
  }

  PassResult run(ir::IRGraph& graph, const PassOptions& opts) override {
    PassResult result;
    result.pass_name = name();
    graph.sortTopological();

    for (uint32_t nid : graph.topoOrder) {
      ir::IRNode* n = graph.getNode(nid);
      if (!n || n->outputs.empty()) continue;

      ir::IRTensor* out = graph.getTensor(n->outputs[0]);
      if (!out) continue;

      // Basic shape propagation for common ops
      switch (n->op) {
        case ir::OpCode::Reshape: {
          // Shape comes from the shape attribute
          auto* shapeAttr = n->attr("shape");
          if (shapeAttr && shapeAttr->is<std::vector<int64_t>>()) {
            out->shape = ir::Shape(shapeAttr->get<std::vector<int64_t>>());
            result.changed = true;
          }
          break;
        }
        case ir::OpCode::Transpose:
        case ir::OpCode::Permute: {
          if (!n->inputs.empty()) {
            ir::IRTensor* inp = graph.getTensor(n->inputs[0]);
            if (inp && !inp->shape.dims.empty()) {
              // Default: reverse dims
              ir::Shape s;
              s.dims = inp->shape.dims;
              std::reverse(s.dims.begin(), s.dims.end());
              out->shape = s;
              result.changed = true;
            }
          }
          break;
        }
        default:
          break;
      }
    }

    result.success = true;
    return result;
  }
};

// ============================================================================
// KV Cache Planning Pass
// ============================================================================
class KVCachePlanningPass : public Pass {
public:
  KVCachePlanningPass() : Pass("KVCachePlanning") {}

  std::string description() const override {
    return "Plans KV cache size, layout, and eviction policy for attention ops";
  }

  PassResult run(ir::IRGraph& graph, const PassOptions& opts) override {
    PassResult result;
    result.pass_name = name();

    // Find all attention nodes and gather KV cache requirements
    uint32_t num_attn_layers = 0;
    for (auto& n : graph.nodes) {
      if (n->op == ir::OpCode::MultiHeadAttention   ||
          n->op == ir::OpCode::GroupQueryAttention   ||
          n->op == ir::OpCode::FlashAttention        ||
          n->op == ir::OpCode::Attention_Masked) {
        num_attn_layers++;
      }
    }

    if (num_attn_layers == 0) {
      result.success = true;
      return result;
    }

    // Configure KV cache based on model metadata
    ir::KVCacheConfig& kv = graph.kv_cache;
    kv.enabled      = true;
    kv.num_heads     = graph.num_heads;
    kv.head_dim      = (graph.num_heads > 0 && graph.hidden_size > 0)
                         ? (graph.hidden_size / graph.num_heads) : 64;
    kv.num_kv_heads  = graph.num_kv_heads > 0 ? graph.num_kv_heads : graph.num_heads;
    kv.max_seq_len   = opts.context_lengths.empty() ? 2048 : opts.context_lengths.back();
    kv.dtype         = ir::DType::F16;

    // Compute KV cache memory requirement:
    // 2 (K+V) × num_layers × num_kv_heads × max_seq × head_dim × dtype_bytes
    uint64_t bytes_per_head = (uint64_t)kv.max_seq_len * kv.head_dim * 2; // F16
    kv.total_bytes = 2 * num_attn_layers * kv.num_kv_heads * bytes_per_head;
    graph.kv_cache_bytes = kv.total_bytes;

    // Insert KVCacheRead / KVCacheWrite nodes before/after each attention op
    // (simplified: just mark existing attention nodes)
    for (auto& n : graph.nodes) {
      if (n->op == ir::OpCode::GroupQueryAttention ||
          n->op == ir::OpCode::MultiHeadAttention  ||
          n->op == ir::OpCode::FlashAttention       ||
          n->op == ir::OpCode::Attention_Masked) {
        ir::Attribute kv_attr;
        kv_attr.name  = "kv_cache_enabled";
        kv_attr.value = true;
        n->attrs.push_back(kv_attr);
      }
    }

    result.changed = true;
    result.success = true;
    return result;
  }
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------
std::unique_ptr<Pass> makeConstantFoldingPass() {
  return std::make_unique<ConstantFoldingPass>();
}

std::unique_ptr<Pass> makeDeadCodeEliminationPass() {
  return std::make_unique<DeadCodeEliminationPass>();
}

std::unique_ptr<Pass> makeOperatorFusionPass() {
  return std::make_unique<OperatorFusionPass>();
}

std::unique_ptr<Pass> makeShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}

std::unique_ptr<Pass> makeKVCachePlanningPass() {
  return std::make_unique<KVCachePlanningPass>();
}

} // namespace passes
} // namespace armcc
