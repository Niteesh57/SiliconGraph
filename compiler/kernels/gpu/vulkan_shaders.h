// ============================================================================
// ARM AI Compiler — Vulkan GLSL Compute Shaders for GPU kernels
//
// These GLSL compute shaders are compiled to SPIR-V by glslangValidator
// and packaged into the .armpack kernels/ directory.
//
// Supports:
//   - FP16 matrix multiplication (decode step, M=1)
//   - INT8 matmul with per-channel dequant
//   - Softmax (numerically stable)
//   - SiLU / GeLU activation
// ============================================================================

// ── GEMM FP16 (M=1) Shader ──────────────────────────────────────────────────
// File: gemm_fp16_m1.comp
// Dispatch: 1D, ceil(N/64) workgroups × 1 thread = 64 output elements/group

static const char* k_gemm_fp16_m1_glsl = R"GLSL(
#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 64) in;

// Push constants
layout(push_constant) uniform PushConsts {
  int K;
  int N;
  float scale_A;
} pc;

// Buffers
layout(set = 0, binding = 0, std430) readonly  buffer BufA { float16_t A[]; };
layout(set = 0, binding = 1, std430) readonly  buffer BufB { float16_t B[]; };
layout(set = 0, binding = 2, std430) writeonly buffer BufC { float16_t C[]; };
layout(set = 0, binding = 3, std430) readonly  buffer BufBias { float16_t Bias[]; };

void main() {
  uint n = gl_GlobalInvocationID.x;
  if (n >= uint(pc.N)) return;

  float acc = 0.0;
  for (int k = 0; k < pc.K; ++k) {
    acc += float(A[k]) * float(B[k * pc.N + n]);
  }
  // Add bias if provided
  acc += float(Bias[n]);
  C[n] = float16_t(acc);
}
)GLSL";

// ── INT8 GEMM with per-channel dequant ───────────────────────────────────────
static const char* k_gemm_int8_glsl = R"GLSL(
#version 450
#extension GL_EXT_shader_8bit_storage : require

layout(local_size_x = 64) in;

layout(push_constant) uniform PushConsts {
  int K;
  int N;
  float scale_A;
} pc;

layout(set = 0, binding = 0, std430) readonly  buffer BufA    { int8_t  A[];    };
layout(set = 0, binding = 1, std430) readonly  buffer BufB    { int8_t  B[];    };
layout(set = 0, binding = 2, std430) writeonly buffer BufC    { float   C[];    };
layout(set = 0, binding = 3, std430) readonly  buffer BufScale { float  Scales[]; };

void main() {
  uint n = gl_GlobalInvocationID.x;
  if (n >= uint(pc.N)) return;

  int acc = 0;
  for (int k = 0; k < pc.K; ++k) {
    acc += int(A[k]) * int(B[k * pc.N + n]);
  }

  // Dequantize: output = acc * scale_A * scale_B[n]
  C[n] = float(acc) * pc.scale_A * Scales[n];
}
)GLSL";

// ── Softmax shader ────────────────────────────────────────────────────────────
static const char* k_softmax_fp16_glsl = R"GLSL(
#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 256) in;

layout(push_constant) uniform PushConsts {
  int N;    // Number of logits
  int batch; // Batch index
} pc;

layout(set = 0, binding = 0, std430) buffer BufX { float16_t X[]; };

shared float smax;
shared float ssum;

void main() {
  uint tid = gl_LocalInvocationID.x;
  uint gid = gl_WorkGroupID.x;
  uint base = gid * uint(pc.N);

  // Phase 1: Find maximum (tree reduction)
  float local_max = -1e30;
  for (uint i = tid; i < uint(pc.N); i += gl_WorkGroupSize.x) {
    local_max = max(local_max, float(X[base + i]));
  }

  // Shared memory reduction for max
  shared float sdata[256];
  sdata[tid] = local_max;
  barrier();
  for (uint s = gl_WorkGroupSize.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
    barrier();
  }
  if (tid == 0) smax = sdata[0];
  barrier();

  // Phase 2: Compute exp(x - max) and sum
  float local_sum = 0.0;
  for (uint i = tid; i < uint(pc.N); i += gl_WorkGroupSize.x) {
    float ev = exp(float(X[base + i]) - smax);
    X[base + i] = float16_t(ev);
    local_sum += ev;
  }

  sdata[tid] = local_sum;
  barrier();
  for (uint s = gl_WorkGroupSize.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    barrier();
  }
  if (tid == 0) ssum = sdata[0];
  barrier();

  // Phase 3: Normalize
  float inv_sum = 1.0 / ssum;
  for (uint i = tid; i < uint(pc.N); i += gl_WorkGroupSize.x) {
    X[base + i] = float16_t(float(X[base + i]) * inv_sum);
  }
}
)GLSL";

// ── SiLU activation shader ────────────────────────────────────────────────────
static const char* k_silu_fp16_glsl = R"GLSL(
#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 256) in;
layout(push_constant) uniform PushConsts { int N; } pc;
layout(set = 0, binding = 0, std430) readonly  buffer BufX { float16_t X[]; };
layout(set = 0, binding = 1, std430) writeonly buffer BufY { float16_t Y[]; };

void main() {
  uint i = gl_GlobalInvocationID.x;
  if (i >= uint(pc.N)) return;
  float x = float(X[i]);
  // SiLU: x * sigmoid(x)
  Y[i] = float16_t(x / (1.0 + exp(-x)));
}
)GLSL";

// ── GeLU activation shader ────────────────────────────────────────────────────
static const char* k_gelu_fp16_glsl = R"GLSL(
#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 256) in;
layout(push_constant) uniform PushConsts { int N; } pc;
layout(set = 0, binding = 0, std430) readonly  buffer BufX { float16_t X[]; };
layout(set = 0, binding = 1, std430) writeonly buffer BufY { float16_t Y[]; };

void main() {
  uint i = gl_GlobalInvocationID.x;
  if (i >= uint(pc.N)) return;
  float x = float(X[i]);
  // GeLU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
  const float c = 0.7978845608f;  // sqrt(2/pi)
  float v = c * (x + 0.044715f * x * x * x);
  Y[i] = float16_t(0.5f * x * (1.0f + tanh(v)));
}
)GLSL";

// ── Layer Norm shader ─────────────────────────────────────────────────────────
static const char* k_layernorm_fp16_glsl = R"GLSL(
#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 256) in;

layout(push_constant) uniform PushConsts {
  int N;     // Hidden size
  float eps; // Epsilon
} pc;

layout(set = 0, binding = 0, std430) readonly  buffer BufX { float16_t X[]; };  // [B, N]
layout(set = 0, binding = 1, std430) readonly  buffer BufW { float16_t W[]; };  // [N] scale
layout(set = 0, binding = 2, std430) readonly  buffer BufB { float16_t Bias[]; }; // [N] bias
layout(set = 0, binding = 3, std430) writeonly buffer BufY { float16_t Y[]; };  // [B, N]

shared float smean;
shared float svar;

void main() {
  uint tid  = gl_LocalInvocationID.x;
  uint base = gl_WorkGroupID.x * uint(pc.N);
  shared float sdata[256];

  // Mean
  float local_sum = 0.0;
  for (uint i = tid; i < uint(pc.N); i += gl_WorkGroupSize.x) {
    local_sum += float(X[base + i]);
  }
  sdata[tid] = local_sum;
  barrier();
  for (uint s = gl_WorkGroupSize.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    barrier();
  }
  if (tid == 0) smean = sdata[0] / float(pc.N);
  barrier();

  // Variance
  float local_var = 0.0;
  for (uint i = tid; i < uint(pc.N); i += gl_WorkGroupSize.x) {
    float d = float(X[base + i]) - smean;
    local_var += d * d;
  }
  sdata[tid] = local_var;
  barrier();
  for (uint s = gl_WorkGroupSize.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    barrier();
  }
  if (tid == 0) svar = sdata[0] / float(pc.N);
  barrier();

  float inv_std = inversesqrt(svar + pc.eps);
  for (uint i = tid; i < uint(pc.N); i += gl_WorkGroupSize.x) {
    float norm = (float(X[base + i]) - smean) * inv_std;
    Y[base + i] = float16_t(norm * float(W[i]) + float(Bias[i]));
  }
}
)GLSL";
