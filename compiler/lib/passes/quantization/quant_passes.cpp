// ============================================================================
// ARM AI Compiler — Quantization Passes
//
// INT8 Post-Training Quantization (PTQ)
// INT4 Grouped Quantization (GPTQ-style)
// FP8 Quantization
// Mixed Precision (per-layer sensitivity-based)
// ============================================================================
#include "passes/pass_manager.h"
#include "arm_ir/graph.h"
#include "arm_ir/ops.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace armcc {
namespace passes {

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void insertQuantDequant(ir::IRGraph& graph,
                               ir::IRNode& weightNode,
                               ir::DType quant_dtype,
                               ir::QuantScheme scheme,
                               int group_size = -1)
{
  // For each weight tensor attached to a compute op, record quant params.
  // The actual quant/dequant nodes are inserted conceptually;
  // the weight_data bytes are reinterpreted as quantized.
  for (uint32_t tid : weightNode.inputs) {
    ir::IRTensor* t = graph.getTensor(tid);
    if (!t || !t->is_weight) continue;

    t->quant.scheme     = scheme;
    t->quant.stored_as  = quant_dtype;
    t->quant.group_size = (group_size > 0) ? group_size : 128;

    // Placeholder scale / zero-point (calibration pass fills these)
    int64_t nelems = t->shape.numElements();
    if (nelems <= 0) nelems = 1;

    if (scheme == ir::QuantScheme::PerTensor) {
      t->quant.scales     = {1.0f};
      t->quant.zero_points= {0};
    } else if (scheme == ir::QuantScheme::PerChannel) {
      int64_t num_ch = t->shape.dims.empty() ? 1 : t->shape.dims[0];
      t->quant.scales.assign(num_ch, 1.0f);
      t->quant.zero_points.assign(num_ch, 0);
    } else if (scheme == ir::QuantScheme::PerGroup) {
      int64_t num_groups = (nelems + group_size - 1) / group_size;
      t->quant.scales.assign(num_groups, 1.0f);
      t->quant.zero_points.assign(num_groups, 0);
    }

    t->dtype = quant_dtype;
    weightNode.is_quantized = true;
  }
}

// ---------------------------------------------------------------------------
// INT8 PTQ Pass
// ---------------------------------------------------------------------------
class INT8QuantizationPass : public Pass {
public:
  INT8QuantizationPass() : Pass("INT8Quantization") {}

  std::string description() const override {
    return "Post-training INT8 per-channel quantization of all weight tensors";
  }

  PassResult run(ir::IRGraph& graph, const PassOptions& opts) override {
    PassResult result;
    result.pass_name = name();

    int quantized = 0;
    graph.forEachNode([&](ir::IRNode& n) {
      if (n.op == ir::OpCode::MatMul   ||
          n.op == ir::OpCode::GEMM     ||
          n.op == ir::OpCode::Conv1D   ||
          n.op == ir::OpCode::Conv2D   ||
          n.op == ir::OpCode::BatchMatMul) {
        insertQuantDequant(graph, n, ir::DType::I8,
                           ir::QuantScheme::PerChannel);
        quantized++;
      }
    });

    result.changed = quantized > 0;
    result.success = true;
    return result;
  }
};

// ---------------------------------------------------------------------------
// INT4 Grouped Quantization Pass (GPTQ-style)
// ---------------------------------------------------------------------------
class INT4QuantizationPass : public Pass {
public:
  explicit INT4QuantizationPass(int group_size)
    : Pass("INT4Quantization"), group_size_(group_size) {}

  std::string description() const override {
    return "INT4 per-group weight quantization (GPTQ-style, group_size=" +
           std::to_string(group_size_) + ")";
  }

  PassResult run(ir::IRGraph& graph, const PassOptions& opts) override {
    PassResult result;
    result.pass_name = name();

    int gs = (opts.quant_group_size > 0) ? opts.quant_group_size : group_size_;
    int quantized = 0;

    graph.forEachNode([&](ir::IRNode& n) {
      if (n.op == ir::OpCode::MatMul   ||
          n.op == ir::OpCode::GEMM     ||
          n.op == ir::OpCode::BatchMatMul) {
        insertQuantDequant(graph, n, ir::DType::I4,
                           ir::QuantScheme::PerGroup, gs);
        quantized++;
      }
    });

    result.changed = quantized > 0;
    result.success = true;
    return result;
  }

private:
  int group_size_;
};

// ---------------------------------------------------------------------------
// FP8 Quantization Pass
// ---------------------------------------------------------------------------
class FP8QuantizationPass : public Pass {
public:
  FP8QuantizationPass() : Pass("FP8Quantization") {}

  std::string description() const override {
    return "FP8 (E4M3) per-tensor quantization of weight and activation tensors";
  }

  PassResult run(ir::IRGraph& graph, const PassOptions& opts) override {
    PassResult result;
    result.pass_name = name();
    int quantized = 0;

    graph.forEachNode([&](ir::IRNode& n) {
      if (n.op == ir::OpCode::MatMul   ||
          n.op == ir::OpCode::GEMM     ||
          n.op == ir::OpCode::BatchMatMul) {
        insertQuantDequant(graph, n, ir::DType::F8_E4M3,
                           ir::QuantScheme::PerTensor);
        quantized++;
      }
    });

    result.changed = quantized > 0;
    result.success = true;
    return result;
  }
};

// ---------------------------------------------------------------------------
// Mixed Precision Pass
//
// Uses per-layer sensitivity scores (from GraphAnalyzer) to decide:
//   - Sensitive layers (score > threshold) → FP16
//   - Insensitive layers → INT4
// ---------------------------------------------------------------------------
class MixedPrecisionPass : public Pass {
public:
  explicit MixedPrecisionPass(std::vector<float> sensitivity)
    : Pass("MixedPrecision"), sensitivity_(std::move(sensitivity)) {}

  std::string description() const override {
    return "Mixed precision: FP16 for sensitive layers, INT4 for rest";
  }

  bool isApplicable(const ir::IRGraph& graph) const override {
    return !sensitivity_.empty();
  }

  PassResult run(ir::IRGraph& graph, const PassOptions& opts) override {
    PassResult result;
    result.pass_name = name();

    float threshold = opts.quant_error_threshold;
    uint32_t layer_idx = 0;

    graph.forEachNode([&](ir::IRNode& n) {
      if (n.op != ir::OpCode::MatMul && n.op != ir::OpCode::GEMM &&
          n.op != ir::OpCode::BatchMatMul) return;

      float sensitivity = (layer_idx < sensitivity_.size())
                          ? sensitivity_[layer_idx] : 0.0f;

      if (sensitivity > threshold) {
        // High sensitivity → keep FP16
        for (uint32_t tid : n.inputs) {
          ir::IRTensor* t = graph.getTensor(tid);
          if (t && t->is_weight) {
            t->dtype = ir::DType::F16;
            t->quant.scheme = ir::QuantScheme::None;
          }
        }
      } else {
        // Low sensitivity → INT4
        insertQuantDequant(graph, n, ir::DType::I4,
                           ir::QuantScheme::PerGroup,
                           opts.quant_group_size);
      }

      layer_idx++;
      result.changed = true;
    });

    result.success = true;
    return result;
  }

private:
  std::vector<float> sensitivity_;
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------
std::unique_ptr<Pass> makeINT8QuantizationPass() {
  return std::make_unique<INT8QuantizationPass>();
}

std::unique_ptr<Pass> makeINT4QuantizationPass(int32_t group_size) {
  return std::make_unique<INT4QuantizationPass>(group_size);
}

std::unique_ptr<Pass> makeFP8QuantizationPass() {
  return std::make_unique<FP8QuantizationPass>();
}

std::unique_ptr<Pass> makeMixedPrecisionPass(const std::vector<float>& sensitivity) {
  return std::make_unique<MixedPrecisionPass>(sensitivity);
}

} // namespace passes
} // namespace armcc
