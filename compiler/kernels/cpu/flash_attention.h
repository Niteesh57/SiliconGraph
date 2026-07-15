// ============================================================================
// ARM AI Compiler — Flash Attention kernel (NEON/scalar)
//
// Implements the Flash Attention algorithm for memory-efficient attention:
//   - O(seq²) compute but O(seq) memory (no N×N attention matrix materialized)
//   - Chunked softmax (online softmax for numerical stability)
//   - Causal masking support
//   - Grouped Query Attention (num_kv_heads ≤ num_heads)
// ============================================================================
#pragma once
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

namespace armcc {
namespace kernels {
namespace cpu {

struct FlashAttentionParams {
  const void* Q;   // [B, num_heads, S, head_dim] FP16
  const void* K;   // [B, num_kv_heads, S, head_dim] FP16
  const void* V;   // [B, num_kv_heads, S, head_dim] FP16
  void*       O;   // [B, num_heads, S, head_dim] FP16 output

  const void* mask;  // Optional [S, S] causal mask (nullptr = autoregressive)

  int32_t B;           // Batch size
  int32_t S;           // Sequence length
  int32_t num_heads;
  int32_t num_kv_heads;
  int32_t head_dim;
  float   scale;       // 1/sqrt(head_dim)
  bool    causal;      // Apply causal mask
  bool    streaming;   // Streaming (chunked) mode
  int32_t chunk_size;  // Tokens per chunk (streaming)
};

// ---------------------------------------------------------------------------
// Flash Attention: compute one head with online softmax
// Input: q [head_dim], k [S, head_dim], v [S, head_dim]
// Output: out [head_dim]
// ---------------------------------------------------------------------------
static void flash_attn_head(
    const float* q, const float* k, const float* v,
    float* out, float* tmp_scores,
    int32_t S, int32_t head_dim, float scale, bool causal, int32_t query_pos)
{
  // Online softmax with running max and sum
  float m = -std::numeric_limits<float>::infinity();  // Running max
  float l = 0.0f;                                       // Running normalizer
  std::fill(out, out + head_dim, 0.0f);

  for (int32_t j = 0; j < S; ++j) {
    if (causal && j > query_pos) break;  // Causal mask

    // QK dot product
    float s = 0.0f;
    for (int32_t d = 0; d < head_dim; ++d) {
      s += q[d] * k[j * head_dim + d];
    }
    s *= scale;

    // Online softmax update
    float m_new = std::max(m, s);
    float exp_s = expf(s - m_new);
    float scale_old = expf(m - m_new);

    // Update running normalizer
    l = l * scale_old + exp_s;

    // Update output: rescale previous + add new
    for (int32_t d = 0; d < head_dim; ++d) {
      out[d] = out[d] * scale_old + exp_s * v[j * head_dim + d];
    }

    m = m_new;
  }

  // Normalize output
  if (l > 0) {
    float inv_l = 1.0f / l;
    for (int32_t d = 0; d < head_dim; ++d) out[d] *= inv_l;
  }
}

// ---------------------------------------------------------------------------
// Full Flash Attention (FP16 I/O, FP32 accumulation)
// ---------------------------------------------------------------------------
void flash_attention_fp16(const FlashAttentionParams& p) {
  const __fp16* Q = (const __fp16*)p.Q;
  const __fp16* K = (const __fp16*)p.K;
  const __fp16* V = (const __fp16*)p.V;
  __fp16*       O = (__fp16*)p.O;

  float scale = (p.scale > 0) ? p.scale : 1.0f / sqrtf((float)p.head_dim);
  int32_t kv_group = p.num_heads / std::max(1, p.num_kv_heads);

  // Temporary buffers
  std::vector<float> q_f32(p.head_dim);
  std::vector<float> k_f32(p.S * p.head_dim);
  std::vector<float> v_f32(p.S * p.head_dim);
  std::vector<float> out_f32(p.head_dim);
  std::vector<float> scores(p.S);

  for (int32_t b = 0; b < p.B; ++b) {
    for (int32_t h = 0; h < p.num_heads; ++h) {
      int32_t kv_head = h / kv_group;

      // Convert Q for this head to FP32
      const __fp16* qh = Q + (b * p.num_heads + h) * p.S * p.head_dim;
      for (int32_t s = 0; s < p.S; ++s) {
        // Process query token s
        for (int32_t d = 0; d < p.head_dim; ++d) {
          q_f32[d] = (float)qh[s * p.head_dim + d];
        }

        // Convert K, V for this kv_head to FP32
        const __fp16* kh = K + (b * p.num_kv_heads + kv_head) * p.S * p.head_dim;
        const __fp16* vh = V + (b * p.num_kv_heads + kv_head) * p.S * p.head_dim;
        for (int32_t j = 0; j < p.S * p.head_dim; ++j) {
          k_f32[j] = (float)kh[j];
          v_f32[j] = (float)vh[j];
        }

        std::fill(out_f32.begin(), out_f32.end(), 0.0f);
        flash_attn_head(q_f32.data(), k_f32.data(), v_f32.data(),
                        out_f32.data(), scores.data(),
                        p.S, p.head_dim, scale, p.causal, s);

        // Write result
        __fp16* oh = O + (b * p.num_heads + h) * p.S * p.head_dim;
        for (int32_t d = 0; d < p.head_dim; ++d) {
          oh[s * p.head_dim + d] = (__fp16)out_f32[d];
        }
      }
    }
  }
}

} // namespace cpu
} // namespace kernels
} // namespace armcc
