// ============================================================================
// ARM AI Compiler — OpenCL Kernel Strings for Mali/Adreno GPU
//
// Used as fallback when Vulkan is unavailable (older Mali GPUs).
// ============================================================================
#pragma once

namespace armcc {
namespace kernels {
namespace gpu {

// ── FP16 GEMM decode (M=1) for OpenCL ───────────────────────────────────────
static const char* k_gemm_fp16_opencl = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void gemm_fp16_m1(
    __global const half* A,   // [K]
    __global const half* B,   // [K, N]
    __global const half* bias,// [N]
    __global half*       C,   // [N]
    int K, int N, float scale_A)
{
  int n = get_global_id(0);
  if (n >= N) return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += (float)A[k] * (float)B[k * N + n];
  }
  acc += (float)bias[n];
  C[n] = (half)acc;
}
)CL";

// ── INT8 GEMM with per-channel dequant (OpenCL) ──────────────────────────────
static const char* k_gemm_int8_opencl = R"CL(
__kernel void gemm_int8_m1(
    __global const char*  A,
    __global const char*  B,
    __global const float* scales_B,
    __global float*       C,
    int K, int N, float scale_A)
{
  int n = get_global_id(0);
  if (n >= N) return;
  int acc = 0;
  for (int k = 0; k < K; ++k) {
    acc += (int)A[k] * (int)B[k * N + n];
  }
  C[n] = (float)acc * scale_A * scales_B[n];
}
)CL";

// ── Softmax (OpenCL, single workgroup) ───────────────────────────────────────
static const char* k_softmax_opencl = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void softmax_fp16(
    __global half* X,
    int N,
    __local float* sdata)
{
  int tid  = get_local_id(0);
  int wgs  = get_local_size(0);
  int base = get_group_id(0) * N;

  // Find max
  float lmax = -1e30f;
  for (int i = tid; i < N; i += wgs) lmax = fmax(lmax, (float)X[base + i]);
  sdata[tid] = lmax;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = wgs/2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid+s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float gmax = sdata[0];
  barrier(CLK_LOCAL_MEM_FENCE);

  // Exp and sum
  float lsum = 0.0f;
  for (int i = tid; i < N; i += wgs) {
    float ev = exp((float)X[base+i] - gmax);
    X[base+i] = (half)ev;
    lsum += ev;
  }
  sdata[tid] = lsum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = wgs/2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid+s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float inv_sum = 1.0f / sdata[0];
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = tid; i < N; i += wgs) {
    X[base+i] = (half)((float)X[base+i] * inv_sum);
  }
}
)CL";

// ── SiLU (OpenCL) ────────────────────────────────────────────────────────────
static const char* k_silu_opencl = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void silu_fp16(
    __global const half* X,
    __global half* Y,
    int N)
{
  int i = get_global_id(0);
  if (i >= N) return;
  float x = (float)X[i];
  Y[i] = (half)(x / (1.0f + exp(-x)));
}
)CL";

} // namespace gpu
} // namespace kernels
} // namespace armcc
