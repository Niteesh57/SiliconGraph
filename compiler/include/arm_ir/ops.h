// ============================================================================
// ARM AI Compiler — ARM Intermediate Representation: Operators
//
// The complete operator set of ARM-IR. Modelled after a union of
// PyTorch ATen ops, ONNX ops, and ARM-specific extensions.
// ============================================================================
#pragma once

#include <cstdint>
#include <string>

namespace armcc {
namespace ir {

// ---------------------------------------------------------------------------
// Op codes — every node in the graph has one of these
// ---------------------------------------------------------------------------
enum class OpCode : uint16_t {
  // ── Tensor creation / constants ──────────────────────────────────────────
  Constant = 0,
  Input,
  Output,

  // ── Linear algebra ───────────────────────────────────────────────────────
  MatMul,         // General matrix multiply: A @ B
  BatchMatMul,    // Batched matmul
  GEMM,           // General matrix multiply with bias: A @ B + C
  Conv1D,
  Conv2D,
  DepthwiseConv2D,
  GroupedConv2D,
  Deconv2D,       // Transposed convolution

  // ── Attention (LLM-specific) ─────────────────────────────────────────────
  MultiHeadAttention,     // Fused MHA
  FlashAttention,         // Memory-efficient attention
  GroupQueryAttention,    // GQA (Llama 3, Mistral)
  SlidingWindowAttention, // Mistral-style
  KVCacheRead,            // Read key/value from cache
  KVCacheWrite,           // Write key/value to cache
  RopeEmbedding,          // Rotary Position Embedding (RoPE)
  ALiBiEmbedding,         // ALiBi positional bias

  // ── Normalization ─────────────────────────────────────────────────────────
  LayerNorm,
  RMSNorm,        // Used by Llama, Qwen, etc.
  BatchNorm,
  GroupNorm,
  InstanceNorm,

  // ── Activation functions ──────────────────────────────────────────────────
  ReLU,
  ReLU6,
  GeLU,           // Gaussian Error Linear Unit
  SiLU,           // Sigmoid Linear Unit (used by Llama FFN)
  Swish,
  GELU_Approx,    // Fast GELU approximation
  Sigmoid,
  Tanh,
  Softmax,
  LogSoftmax,
  HardSigmoid,
  HardSwish,
  Mish,
  LeakyReLU,
  ELU,
  PReLU,

  // ── Element-wise binary ops ───────────────────────────────────────────────
  Add,
  Sub,
  Mul,
  Div,
  Pow,
  Maximum,
  Minimum,
  Mod,

  // ── Element-wise unary ops ────────────────────────────────────────────────
  Neg,
  Abs,
  Sqrt,
  Rsqrt,
  Exp,
  Log,
  Floor,
  Ceil,
  Round,
  Sign,
  Reciprocal,

  // ── Reduction ops ─────────────────────────────────────────────────────────
  ReduceSum,
  ReduceMean,
  ReduceMax,
  ReduceMin,
  ReduceProd,
  ArgMax,
  ArgMin,

  // ── Shape / layout ────────────────────────────────────────────────────────
  Reshape,
  Flatten,
  Squeeze,
  Unsqueeze,
  Transpose,
  Permute,
  Slice,
  Gather,
  GatherElements,
  Scatter,
  ScatterElements,
  Concat,
  Split,
  Stack,
  Tile,
  Expand,
  Pad,

  // ── Type / quantization ───────────────────────────────────────────────────
  Cast,
  Quantize,
  Dequantize,
  Requantize,

  // ── Pooling ───────────────────────────────────────────────────────────────
  MaxPool2D,
  AvgPool2D,
  GlobalAvgPool,
  GlobalMaxPool,
  AdaptiveAvgPool,

  // ── Embedding ─────────────────────────────────────────────────────────────
  Embedding,          // Token embedding lookup
  EmbeddingBag,

  // ── LLM-specific fused ops ────────────────────────────────────────────────
  FFN_SwiGLU,         // Fused SwiGLU FFN block (Llama)
  FFN_GeLU,           // Fused GeLU FFN block
  RMSNorm_Scale,      // RMSNorm + learned scale
  Attention_Masked,   // Causal masked attention

  // ── ARM hardware hints (inserted by hardware fusion pass) ─────────────────
  HW_Boundary,        // Marks a transition between execution units
  DMA_Prefetch,       // Hint: DMA prefetch next tensor
  NPU_SubgraphBegin,  // Begin of NPU-delegated subgraph
  NPU_SubgraphEnd,    // End of NPU-delegated subgraph

  // ── Control flow ──────────────────────────────────────────────────────────
  If,
  Loop,
  Return,

  // ── Unknown / placeholder ─────────────────────────────────────────────────
  Unknown = 0xFFFF,
};

const char*   opCodeToString(OpCode op);
OpCode        opCodeFromString(const std::string& s);
bool          opCodeIsReduction(OpCode op);
bool          opCodeIsFusable(OpCode op);
bool          opCodeIsLLMSpecific(OpCode op);
bool          opCodeIsHardwareHint(OpCode op);

} // namespace ir
} // namespace armcc
