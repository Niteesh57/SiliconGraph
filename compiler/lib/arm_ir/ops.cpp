// ============================================================================
// ARM AI Compiler — ARM-IR: Ops (implementation)
// ============================================================================
#include "arm_ir/ops.h"

namespace armcc {
namespace ir {

const char* opCodeToString(OpCode op) {
  switch (op) {
    // Tensor creation
    case OpCode::Constant:              return "Constant";
    case OpCode::Input:                 return "Input";
    case OpCode::Output:                return "Output";
    // Linear algebra
    case OpCode::MatMul:                return "MatMul";
    case OpCode::BatchMatMul:           return "BatchMatMul";
    case OpCode::GEMM:                  return "GEMM";
    case OpCode::Conv1D:                return "Conv1D";
    case OpCode::Conv2D:                return "Conv2D";
    case OpCode::DepthwiseConv2D:       return "DepthwiseConv2D";
    case OpCode::GroupedConv2D:         return "GroupedConv2D";
    case OpCode::Deconv2D:              return "Deconv2D";
    // Attention
    case OpCode::MultiHeadAttention:    return "MultiHeadAttention";
    case OpCode::FlashAttention:        return "FlashAttention";
    case OpCode::GroupQueryAttention:   return "GroupQueryAttention";
    case OpCode::SlidingWindowAttention:return "SlidingWindowAttention";
    case OpCode::KVCacheRead:           return "KVCacheRead";
    case OpCode::KVCacheWrite:          return "KVCacheWrite";
    case OpCode::RopeEmbedding:         return "RopeEmbedding";
    case OpCode::ALiBiEmbedding:        return "ALiBiEmbedding";
    // Normalization
    case OpCode::LayerNorm:             return "LayerNorm";
    case OpCode::RMSNorm:               return "RMSNorm";
    case OpCode::BatchNorm:             return "BatchNorm";
    case OpCode::GroupNorm:             return "GroupNorm";
    case OpCode::InstanceNorm:          return "InstanceNorm";
    // Activations
    case OpCode::ReLU:                  return "ReLU";
    case OpCode::ReLU6:                 return "ReLU6";
    case OpCode::GeLU:                  return "GeLU";
    case OpCode::SiLU:                  return "SiLU";
    case OpCode::Swish:                 return "Swish";
    case OpCode::GELU_Approx:           return "GELU_Approx";
    case OpCode::Sigmoid:               return "Sigmoid";
    case OpCode::Tanh:                  return "Tanh";
    case OpCode::Softmax:               return "Softmax";
    case OpCode::LogSoftmax:            return "LogSoftmax";
    case OpCode::HardSigmoid:           return "HardSigmoid";
    case OpCode::HardSwish:             return "HardSwish";
    case OpCode::Mish:                  return "Mish";
    case OpCode::LeakyReLU:             return "LeakyReLU";
    case OpCode::ELU:                   return "ELU";
    case OpCode::PReLU:                 return "PReLU";
    // Binary
    case OpCode::Add:                   return "Add";
    case OpCode::Sub:                   return "Sub";
    case OpCode::Mul:                   return "Mul";
    case OpCode::Div:                   return "Div";
    case OpCode::Pow:                   return "Pow";
    case OpCode::Maximum:               return "Maximum";
    case OpCode::Minimum:               return "Minimum";
    case OpCode::Mod:                   return "Mod";
    // Unary
    case OpCode::Neg:                   return "Neg";
    case OpCode::Abs:                   return "Abs";
    case OpCode::Sqrt:                  return "Sqrt";
    case OpCode::Rsqrt:                 return "Rsqrt";
    case OpCode::Exp:                   return "Exp";
    case OpCode::Log:                   return "Log";
    case OpCode::Floor:                 return "Floor";
    case OpCode::Ceil:                  return "Ceil";
    case OpCode::Round:                 return "Round";
    case OpCode::Sign:                  return "Sign";
    case OpCode::Reciprocal:            return "Reciprocal";
    // Reduction
    case OpCode::ReduceSum:             return "ReduceSum";
    case OpCode::ReduceMean:            return "ReduceMean";
    case OpCode::ReduceMax:             return "ReduceMax";
    case OpCode::ReduceMin:             return "ReduceMin";
    case OpCode::ReduceProd:            return "ReduceProd";
    case OpCode::ArgMax:                return "ArgMax";
    case OpCode::ArgMin:                return "ArgMin";
    // Shape
    case OpCode::Reshape:               return "Reshape";
    case OpCode::Flatten:               return "Flatten";
    case OpCode::Squeeze:               return "Squeeze";
    case OpCode::Unsqueeze:             return "Unsqueeze";
    case OpCode::Transpose:             return "Transpose";
    case OpCode::Permute:               return "Permute";
    case OpCode::Slice:                 return "Slice";
    case OpCode::Gather:                return "Gather";
    case OpCode::GatherElements:        return "GatherElements";
    case OpCode::Scatter:               return "Scatter";
    case OpCode::ScatterElements:       return "ScatterElements";
    case OpCode::Concat:                return "Concat";
    case OpCode::Split:                 return "Split";
    case OpCode::Stack:                 return "Stack";
    case OpCode::Tile:                  return "Tile";
    case OpCode::Expand:                return "Expand";
    case OpCode::Pad:                   return "Pad";
    // Type / quant
    case OpCode::Cast:                  return "Cast";
    case OpCode::Quantize:              return "Quantize";
    case OpCode::Dequantize:            return "Dequantize";
    case OpCode::Requantize:            return "Requantize";
    // Pooling
    case OpCode::MaxPool2D:             return "MaxPool2D";
    case OpCode::AvgPool2D:             return "AvgPool2D";
    case OpCode::GlobalAvgPool:         return "GlobalAvgPool";
    case OpCode::GlobalMaxPool:         return "GlobalMaxPool";
    case OpCode::AdaptiveAvgPool:       return "AdaptiveAvgPool";
    // Embedding
    case OpCode::Embedding:             return "Embedding";
    case OpCode::EmbeddingBag:          return "EmbeddingBag";
    // LLM fused
    case OpCode::FFN_SwiGLU:            return "FFN_SwiGLU";
    case OpCode::FFN_GeLU:              return "FFN_GeLU";
    case OpCode::RMSNorm_Scale:         return "RMSNorm_Scale";
    case OpCode::Attention_Masked:      return "Attention_Masked";
    // Hardware hints
    case OpCode::HW_Boundary:           return "HW_Boundary";
    case OpCode::DMA_Prefetch:          return "DMA_Prefetch";
    case OpCode::NPU_SubgraphBegin:     return "NPU_SubgraphBegin";
    case OpCode::NPU_SubgraphEnd:       return "NPU_SubgraphEnd";
    // Control flow
    case OpCode::If:                    return "If";
    case OpCode::Loop:                  return "Loop";
    case OpCode::Return:                return "Return";
    default:                            return "Unknown";
  }
}

bool opCodeIsReduction(OpCode op) {
  return op == OpCode::ReduceSum   || op == OpCode::ReduceMean ||
         op == OpCode::ReduceMax   || op == OpCode::ReduceMin  ||
         op == OpCode::ReduceProd  || op == OpCode::ArgMax     ||
         op == OpCode::ArgMin      || op == OpCode::GlobalAvgPool ||
         op == OpCode::GlobalMaxPool;
}

bool opCodeIsFusable(OpCode op) {
  // Ops that are commonly fused with adjacent ops
  return op == OpCode::Add    || op == OpCode::ReLU   ||
         op == OpCode::GeLU   || op == OpCode::SiLU   ||
         op == OpCode::Swish  || op == OpCode::RMSNorm ||
         op == OpCode::LayerNorm || op == OpCode::Softmax;
}

bool opCodeIsLLMSpecific(OpCode op) {
  return op == OpCode::MultiHeadAttention   ||
         op == OpCode::FlashAttention       ||
         op == OpCode::GroupQueryAttention  ||
         op == OpCode::SlidingWindowAttention ||
         op == OpCode::KVCacheRead          ||
         op == OpCode::KVCacheWrite         ||
         op == OpCode::RopeEmbedding        ||
         op == OpCode::ALiBiEmbedding       ||
         op == OpCode::RMSNorm              ||
         op == OpCode::FFN_SwiGLU           ||
         op == OpCode::FFN_GeLU             ||
         op == OpCode::RMSNorm_Scale        ||
         op == OpCode::Attention_Masked;
}

bool opCodeIsHardwareHint(OpCode op) {
  return op == OpCode::HW_Boundary    ||
         op == OpCode::DMA_Prefetch   ||
         op == OpCode::NPU_SubgraphBegin ||
         op == OpCode::NPU_SubgraphEnd;
}

} // namespace ir
} // namespace armcc
