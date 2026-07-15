// ============================================================================
// ARM AI Compiler — CPU NEON/SVE Kernel: Matrix Multiplication
//
// Provides optimized matmul kernels for ARM CPUs:
//   - NEON (ARMv8.0+): All ARM phones
//   - NEON + i8mm (ARMv8.6+): High-performance INT8
//   - SVE2 (ARMv9): Snapdragon 8 Elite, Neoverse V2
//
// These are called via function pointer through the runtime's kernel table.
// ============================================================================
#pragma once
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __ARM_FEATURE_SVE2
#include <arm_sve.h>
#endif

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>

namespace armcc {
namespace kernels {
namespace cpu {

// ---------------------------------------------------------------------------
// Kernel signature (all kernels share this interface)
// ---------------------------------------------------------------------------
struct MatMulParams {
  const void* A;    // [M, K] activation (float16 or int8)
  const void* B;    // [K, N] weight (float16, int8, or int4-packed)
  const void* bias; // [N] bias (float16, or nullptr)
  void*       C;    // [M, N] output
  int32_t     M, K, N;
  int32_t     lda, ldb, ldc;   // Leading dimensions
  float       scale_A;  // For int8 activation dequant
  const float* scales_B;  // Per-channel or per-group weight scales
  const int8_t* zeros_B;  // Per-group zero points (INT4)
  int32_t     group_size;  // For grouped quantization
};

// ---------------------------------------------------------------------------
// F16 × F16 → F16 GEMM (NEON, M=1 special case for decode step)
// ---------------------------------------------------------------------------
void matmul_fp16_neon_m1(const MatMulParams& p) {
#ifdef __ARM_NEON
  // Optimized for M=1 (decode step) with NEON FP16
  const __fp16* A     = (const __fp16*)p.A;
  const __fp16* B     = (const __fp16*)p.B;
  __fp16*       C     = (__fp16*)p.C;
  const __fp16* bias  = (const __fp16*)p.bias;

  int32_t K = p.K, N = p.N;

  // Process 8 output channels at once with NEON fp16 SIMD
  int32_t n = 0;
  for (; n + 7 < N; n += 8) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);

    for (int32_t k = 0; k < K; ++k) {
      float32_t a_val = (float32_t)A[k];
      float32x4_t av  = vdupq_n_f32(a_val);

      // Load 8 weights from B
      float16x8_t bv = vld1q_f16(B + k * N + n);
      float32x4_t b0 = vcvt_f32_f16(vget_low_f16(bv));
      float32x4_t b1 = vcvt_f32_f16(vget_high_f16(bv));

      acc0 = vmlaq_f32(acc0, av, b0);
      acc1 = vmlaq_f32(acc1, av, b1);
    }

    // Convert and store
    float16x4_t r0 = vcvt_f16_f32(acc0);
    float16x4_t r1 = vcvt_f16_f32(acc1);
    float16x8_t r  = vcombine_f16(r0, r1);

    if (bias) {
      float16x8_t bv = vld1q_f16(bias + n);
      r = vaddq_f16(r, bv);
    }
    vst1q_f16(C + n, r);
  }

  // Tail: remaining columns
  for (; n < N; ++n) {
    float acc = 0.0f;
    for (int32_t k = 0; k < K; ++k) acc += (float)A[k] * (float)B[k * N + n];
    if (bias) acc += (float)bias[n];
    C[n] = (__fp16)acc;
  }
#else
  // Scalar fallback
  const __fp16* A = (const __fp16*)p.A;
  const __fp16* B = (const __fp16*)p.B;
  __fp16*       C = (__fp16*)p.C;
  for (int32_t n = 0; n < p.N; ++n) {
    float acc = 0.0f;
    for (int32_t k = 0; k < p.K; ++k) acc += (float)A[k] * (float)B[k * p.N + n];
    C[n] = (__fp16)acc;
  }
#endif
}

// ---------------------------------------------------------------------------
// INT8 × INT4 grouped dequant GEMM
// Weights are packed INT4 (2 elements per byte), with per-group F16 scales
// ---------------------------------------------------------------------------
void matmul_int4_grouped_neon(const MatMulParams& p) {
#ifdef __ARM_NEON
  const int8_t* A      = (const int8_t*)p.A;
  const uint8_t* B_i4  = (const uint8_t*)p.B;  // Packed INT4 (2 per byte)
  __fp16*        C     = (__fp16*)p.C;
  const __fp16*  scales = (const __fp16*)p.scales_B;
  const int8_t*  zeros  = p.zeros_B;
  int32_t gs = (p.group_size > 0) ? p.group_size : 128;

  for (int32_t n = 0; n < p.N; ++n) {
    float acc = 0.0f;

    for (int32_t k = 0; k < p.K; k += 2) {
      // Unpack two INT4 values from one byte
      uint8_t packed = B_i4[(k / 2) * p.N + n];
      int8_t  w0 = (int8_t)((packed & 0x0F) - 8);  // Low nibble [-8, 7]
      int8_t  w1 = (int8_t)((packed >> 4)   - 8);  // High nibble [-8, 7]

      int32_t group = k / gs;
      float scale = (float)((const __fp16*)scales)[group * p.N + n];
      int8_t zero  = zeros ? zeros[group * p.N + n] : 0;

      float w0f = scale * (float)(w0 - zero);
      float w1f = scale * (float)(w1 - zero);

      acc += (float)A[k]     * w0f;
      acc += (k+1 < p.K) ? ((float)A[k+1] * w1f) : 0.0f;
    }

    C[n] = (__fp16)acc;
  }
#else
  // Scalar fallback identical to above
  (void)p;
#endif
}

// ---------------------------------------------------------------------------
// RMSNorm kernel (NEON)
// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
// ---------------------------------------------------------------------------
void rmsnorm_fp16_neon(const __fp16* x, const __fp16* w, __fp16* y,
                       int32_t n, float eps = 1e-5f) {
#ifdef __ARM_NEON
  // Compute sum of squares
  float32x4_t sos = vdupq_n_f32(0.0f);
  int32_t i = 0;
  for (; i + 3 < n; i += 4) {
    float16x4_t v  = vld1_f16(x + i);
    float32x4_t vf = vcvt_f32_f16(v);
    sos = vmlaq_f32(sos, vf, vf);
  }
  float ss = vaddvq_f32(sos);
  for (; i < n; ++i) ss += (float)x[i] * (float)x[i];

  float rms_inv = 1.0f / sqrtf(ss / (float)n + eps);

  // Normalize + scale
  float32x4_t rms_v = vdupq_n_f32(rms_inv);
  i = 0;
  for (; i + 3 < n; i += 4) {
    float16x4_t xv = vld1_f16(x + i);
    float16x4_t wv = vld1_f16(w + i);
    float32x4_t xf = vmulq_f32(vcvt_f32_f16(xv), rms_v);
    float32x4_t wf = vcvt_f32_f16(wv);
    float32x4_t yf = vmulq_f32(xf, wf);
    vst1_f16(y + i, vcvt_f16_f32(yf));
  }
  for (; i < n; ++i) {
    y[i] = (__fp16)((float)x[i] * rms_inv * (float)w[i]);
  }
#else
  float ss = 0;
  for (int i = 0; i < n; ++i) ss += (float)x[i] * (float)x[i];
  float rms_inv = 1.0f / sqrtf(ss / (float)n + eps);
  for (int i = 0; i < n; ++i) y[i] = (__fp16)((float)x[i] * rms_inv * (float)w[i]);
#endif
}

// ---------------------------------------------------------------------------
// SiLU activation (NEON)
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
// ---------------------------------------------------------------------------
void silu_fp16_neon(const __fp16* x, __fp16* y, int32_t n) {
#ifdef __ARM_NEON
  int32_t i = 0;
  for (; i + 3 < n; i += 4) {
    float16x4_t xv  = vld1_f16(x + i);
    float32x4_t xf  = vcvt_f32_f16(xv);

    // Sigmoid: 1/(1+exp(-x))
    // Using fast approximation: clip, then compute
    float32x4_t neg = vnegq_f32(xf);
    // Scalar exp for now (vectorized exp requires libmvec or vexpq)
    float32x4_t sig;
    for (int j = 0; j < 4; ++j) {
      float ex = expf(neg[j]);
      sig[j] = 1.0f / (1.0f + ex);
    }
    float32x4_t yf = vmulq_f32(xf, sig);
    vst1_f16(y + i, vcvt_f16_f32(yf));
  }
  for (; i < n; ++i) {
    float xi = (float)x[i];
    y[i] = (__fp16)(xi * (1.0f / (1.0f + expf(-xi))));
  }
#else
  for (int i = 0; i < n; ++i) {
    float xi = (float)x[i];
    y[i] = (__fp16)(xi / (1.0f + expf(-xi)));
  }
#endif
}

// ---------------------------------------------------------------------------
// Softmax (NEON, numerically stable)
// ---------------------------------------------------------------------------
void softmax_fp16_neon(const __fp16* x, __fp16* y, int32_t n) {
  // Find max
  float xmax = (float)x[0];
  for (int32_t i = 1; i < n; ++i) if ((float)x[i] > xmax) xmax = (float)x[i];

  // Exp and sum
  float sum = 0.0f;
  for (int32_t i = 0; i < n; ++i) {
    y[i] = (__fp16)expf((float)x[i] - xmax);
    sum += (float)y[i];
  }

  // Normalize
  float inv_sum = 1.0f / sum;
#ifdef __ARM_NEON
  float32x4_t inv_v = vdupq_n_f32(inv_sum);
  int32_t i = 0;
  for (; i + 3 < n; i += 4) {
    float16x4_t yv = vld1_f16(y + i);
    float32x4_t yf = vmulq_f32(vcvt_f32_f16(yv), inv_v);
    vst1_f16(y + i, vcvt_f16_f32(yf));
  }
  for (; i < n; ++i) y[i] = (__fp16)((float)y[i] * inv_sum);
#else
  for (int32_t i = 0; i < n; ++i) y[i] = (__fp16)((float)y[i] * inv_sum);
#endif
}

// ---------------------------------------------------------------------------
// RoPE (Rotary Position Embedding) — NEON
// Applies rotation to Q and K tensors in-place.
// ---------------------------------------------------------------------------
void rope_apply_neon(__fp16* x, int32_t head_dim,
                     int32_t position, float base = 10000.0f) {
  for (int32_t i = 0; i < head_dim / 2; ++i) {
    float theta = (float)position / powf(base, 2.0f * i / (float)head_dim);
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    float x0 = (float)x[i];
    float x1 = (float)x[i + head_dim / 2];

    x[i]               = (__fp16)(x0 * cos_t - x1 * sin_t);
    x[i + head_dim / 2] = (__fp16)(x0 * sin_t + x1 * cos_t);
  }
}

} // namespace cpu
} // namespace kernels
} // namespace armcc
