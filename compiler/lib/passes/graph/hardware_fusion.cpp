// ============================================================================
// ARM AI Compiler — Hardware Fusion Pass
//
// The most novel pass in the compiler. Instead of only fusing:
//   MatMul + Add + ReLU → one kernel
//
// This pass fuses across hardware boundaries:
//   Embedding   → DSP
//   Attention   → NPU
//   LayerNorm   → CPU
//   MatMul      → GPU
//   Softmax     → DSP
//
// Uses the CostModel to assign each op to its optimal execution unit,
// then inserts HW_Boundary nodes at unit transition points so the
// runtime knows where to initiate DMA transfers.
// ============================================================================
#include "passes/pass_manager.h"
#include "analysis/cost_model.h"
#include "arm_ir/graph.h"
#include "arm_ir/ops.h"
#include <memory>

namespace armcc {
namespace passes {

namespace {

class HardwareFusionPass : public Pass {
public:
  explicit HardwareFusionPass(const analysis::DeviceProfile& dev)
    : Pass("HardwareFusion"), dev_(dev), costModel_() {}

  std::string description() const override {
    return "Assigns ops to optimal exec units (CPU/GPU/NPU/DSP) "
           "and inserts HW_Boundary transition markers";
  }

  PassResult run(ir::IRGraph& graph, const PassOptions& opts) override {
    PassResult result;
    result.pass_name = name();
    graph.sortTopological();

    int battery_pct = 90;  // Default; in the full impl, comes from opts
    auto thermal    = ir::ThermalState::Nominal;

    // Step 1: For each node, query the cost model and assign best unit
    for (uint32_t nid : graph.topoOrder) {
      ir::IRNode* n = graph.getNode(nid);
      if (!n || n->op == ir::OpCode::Constant ||
          n->op == ir::OpCode::Input  || n->op == ir::OpCode::Output) continue;

      analysis::CostModelQuery q;
      q.op           = n->op;
      q.input_dtype  = ir::DType::F16;  // Simplified; should read from tensor
      q.weight_dtype = n->is_quantized ? ir::DType::I8 : ir::DType::F16;

      // Extract shape info from output tensor
      if (!n->outputs.empty()) {
        ir::IRTensor* out = graph.getTensor(n->outputs[0]);
        if (out && out->shape.rank() >= 2) {
          q.M = out->shape.dims[out->shape.rank()-2];
          q.N = out->shape.dims[out->shape.rank()-1];
        }
      }
      if (!n->inputs.empty()) {
        ir::IRTensor* inp = graph.getTensor(n->inputs[0]);
        if (inp && inp->shape.rank() >= 1) {
          q.K = inp->shape.dims[inp->shape.rank()-1];
        }
      }

      auto recUnit = costModel_.recommendUnit(dev_, q, battery_pct, thermal);
      n->assigned_unit = recUnit;

      // Cache cost estimates on the node
      auto costResult = costModel_.query(dev_, q);
      n->cost_cpu_ms  = costResult.cpu_ms;
      n->cost_gpu_ms  = costResult.gpu_ms;
      n->cost_npu_ms  = costResult.npu_ms;
      n->cost_dsp_ms  = costResult.dsp_ms;
      n->cost_ane_ms  = costResult.ane_ms;

      // Mark NPU/GPU subgraph membership
      if (recUnit == ir::ExecUnit::NPU || recUnit == ir::ExecUnit::APU) {
        n->in_npu_subgraph = true;
      } else if (recUnit == ir::ExecUnit::GPU) {
        n->in_gpu_subgraph = true;
      } else if (recUnit == ir::ExecUnit::DSP) {
        n->in_dsp_subgraph = true;
      }

      result.changed = true;
    }

    // Step 2: Insert HW_Boundary nodes at execution unit transitions
    // Walk pairs of adjacent nodes; where the unit changes, insert a boundary.
    std::vector<std::pair<uint32_t, uint32_t>> boundaries; // (afterNode, beforeNode)
    for (size_t i = 0; i + 1 < graph.topoOrder.size(); ++i) {
      ir::IRNode* a = graph.getNode(graph.topoOrder[i]);
      ir::IRNode* b = graph.getNode(graph.topoOrder[i+1]);
      if (!a || !b) continue;
      if (a->assigned_unit != b->assigned_unit &&
          a->assigned_unit != ir::ExecUnit::Auto &&
          b->assigned_unit != ir::ExecUnit::Auto) {
        boundaries.push_back({a->id, b->id});
      }
    }

    for (auto [afterId, beforeId] : boundaries) {
      ir::IRNode* after  = graph.getNode(afterId);
      ir::IRNode* before = graph.getNode(beforeId);
      if (!after || !before || after->outputs.empty()) continue;

      // Find the shared tensor between after and before
      for (uint32_t out_tid : after->outputs) {
        for (uint32_t in_tid : before->inputs) {
          if (out_tid == in_tid) {
            // Insert a HW_Boundary node between them
            auto bNode = std::make_unique<ir::IRNode>();
            bNode->op   = ir::OpCode::HW_Boundary;
            bNode->name = "hw_boundary_" + std::to_string(afterId)
                        + "_" + std::to_string(beforeId);
            ir::Attribute fromAttr, toAttr;
            fromAttr.name = "from_unit";
            fromAttr.value = std::string(ir::execUnitToString(after->assigned_unit));
            toAttr.name   = "to_unit";
            toAttr.value  = std::string(ir::execUnitToString(before->assigned_unit));
            bNode->attrs.push_back(fromAttr);
            bNode->attrs.push_back(toAttr);
            bNode->inputs  = {out_tid};
            bNode->outputs = {out_tid};  // pass-through (DMA copy implied)
            graph.addNode(std::move(bNode));
            result.nodes_added++;
            break;
          }
        }
      }
    }

    result.success = true;
    return result;
  }

private:
  const analysis::DeviceProfile& dev_;
  analysis::CostModel             costModel_;
};

} // anonymous namespace

std::unique_ptr<Pass> makeHardwareFusionPass(const analysis::DeviceProfile& dev) {
  return std::make_unique<HardwareFusionPass>(dev);
}

// ---------------------------------------------------------------------------
// Layout Optimization Pass
// ---------------------------------------------------------------------------
namespace {

class LayoutOptimizationPass : public Pass {
public:
  explicit LayoutOptimizationPass(const analysis::DeviceProfile& dev)
    : Pass("LayoutOptimization"), dev_(dev) {}

  std::string description() const override {
    return "Converts tensor layouts to preferred format for target (NHWC for ARM)";
  }

  PassResult run(ir::IRGraph& graph, const PassOptions& opts) override {
    PassResult result;
    result.pass_name = name();

    for (auto& t : graph.tensors) {
      if (t->layout == ir::MemoryLayout::Unknown ||
          t->layout == ir::MemoryLayout::NCHW) {
        if (t->shape.rank() == 4) {
          // ARM prefers NHWC for Conv layers
          t->layout = ir::MemoryLayout::NHWC;
          result.changed = true;
        } else if (t->shape.rank() == 2) {
          // 2D tensors: row-major
          t->layout = ir::MemoryLayout::NC;
          result.changed = true;
        }
      }
    }

    result.success = true;
    return result;
  }

private:
  const analysis::DeviceProfile& dev_;
};

} // anonymous namespace

std::unique_ptr<Pass> makeLayoutOptimizationPass(const analysis::DeviceProfile& dev) {
  return std::make_unique<LayoutOptimizationPass>(dev);
}

// ---------------------------------------------------------------------------
// Streaming Attention Pass
// ---------------------------------------------------------------------------
namespace {

class StreamingAttentionPass : public Pass {
public:
  explicit StreamingAttentionPass(uint32_t chunk_size)
    : Pass("StreamingAttention"), chunk_size_(chunk_size) {}

  std::string description() const override {
    return "Rewrites attention for streaming/chunked inference (chunk_size="
           + std::to_string(chunk_size_) + ")";
  }

  PassResult run(ir::IRGraph& graph, const PassOptions& opts) override {
    PassResult result;
    result.pass_name = name();

    for (auto& n : graph.nodes) {
      if (n->op == ir::OpCode::MultiHeadAttention ||
          n->op == ir::OpCode::GroupQueryAttention ||
          n->op == ir::OpCode::Attention_Masked) {
        // Rewrite to FlashAttention with chunking
        if (n->op != ir::OpCode::FlashAttention) {
          n->op = ir::OpCode::FlashAttention;
          n->supports_streaming   = true;
          n->streaming_chunk_size = (int32_t)chunk_size_;
          ir::Attribute chunkAttr;
          chunkAttr.name  = "chunk_size";
          chunkAttr.value = (int64_t)chunk_size_;
          n->attrs.push_back(chunkAttr);
          result.changed = true;
        }
      }
    }

    result.success = true;
    return result;
  }

private:
  uint32_t chunk_size_;
};

} // anonymous namespace

std::unique_ptr<Pass> makeStreamingAttentionPass(uint32_t chunk_size) {
  return std::make_unique<StreamingAttentionPass>(chunk_size);
}

// ---------------------------------------------------------------------------
// Kernel Selection, Tiling, Vectorization (stubs)
// Full implementations emit target-specific metadata onto nodes.
// ---------------------------------------------------------------------------
namespace {

class KernelSelectionPass : public Pass {
public:
  explicit KernelSelectionPass(const analysis::DeviceProfile& dev)
    : Pass("KernelSelection"), dev_(dev) {}
  PassResult run(ir::IRGraph& graph, const PassOptions& opts) override {
    PassResult r; r.pass_name = name(); r.success = true; return r;
  }
private:
  const analysis::DeviceProfile& dev_;
};

class TilingPass : public Pass {
public:
  explicit TilingPass(const analysis::DeviceProfile& dev)
    : Pass("Tiling"), dev_(dev) {}
  PassResult run(ir::IRGraph& graph, const PassOptions& opts) override {
    PassResult r; r.pass_name = name(); r.success = true; return r;
  }
private:
  const analysis::DeviceProfile& dev_;
};

class VectorizationPass : public Pass {
public:
  explicit VectorizationPass(const analysis::DeviceProfile& dev)
    : Pass("Vectorization"), dev_(dev) {}
  PassResult run(ir::IRGraph& graph, const PassOptions& opts) override {
    PassResult r; r.pass_name = name(); r.success = true; return r;
  }
private:
  const analysis::DeviceProfile& dev_;
};

} // anonymous namespace

std::unique_ptr<Pass> makeKernelSelectionPass(const analysis::DeviceProfile& dev) {
  return std::make_unique<KernelSelectionPass>(dev);
}
std::unique_ptr<Pass> makeTilingPass(const analysis::DeviceProfile& dev) {
  return std::make_unique<TilingPass>(dev);
}
std::unique_ptr<Pass> makeVectorizationPass(const analysis::DeviceProfile& dev) {
  return std::make_unique<VectorizationPass>(dev);
}

} // namespace passes
} // namespace armcc
