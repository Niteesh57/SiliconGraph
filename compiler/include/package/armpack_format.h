// ============================================================================
// ARM AI Compiler — Package Format: .armpack
//
// .armpack is a ZIP-based portable archive containing the full compiled
// output of the ARM AI Compiler for one model. It is the compiler's only
// output artifact and is designed to be consumed by any ARM runtime.
//
// A third-party runtime only needs to:
//   1. Open the ZIP
//   2. Read manifest.json to discover graphs and kernels
//   3. Run selector.wasm to pick the best graph
//   4. mmap the weight file and run inference
//
// No libarmcc dependency at runtime.
// ============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

namespace armcc {
namespace pkg {

// ---------------------------------------------------------------------------
// Magic number + version for binary headers
// ---------------------------------------------------------------------------
constexpr uint32_t ARMPACK_MAGIC   = 0x41524D43;  // 'ARMC'
constexpr uint32_t ARMPACK_VERSION = 0x00010000;   // 1.0.0

// ---------------------------------------------------------------------------
// Kernel target enum (for manifest)
// ---------------------------------------------------------------------------
enum class KernelTarget : uint8_t {
  CPU_NEON    = 0,
  CPU_SVE,
  CPU_SVE2,
  GPU_Vulkan,
  GPU_OpenCL,
  GPU_Metal,
  NPU_HTP,         // Qualcomm Hexagon HTP
  NPU_NeuroPilot,  // MediaTek NeuroPilot
  NPU_ANE,         // Apple Neural Engine (CoreML)
  NPU_Xclipse,     // Samsung Xclipse
};

const char* kernelTargetToString(KernelTarget t);

// ---------------------------------------------------------------------------
// KernelEntry — one precompiled kernel object in the package
// ---------------------------------------------------------------------------
struct KernelEntry {
  std::string    id;           // e.g. "matmul_int8_neon_128x128"
  KernelTarget   target;
  std::string    path;         // relative path inside .armpack
  uint64_t       size_bytes;
  std::string    op_pattern;   // the op(s) this kernel implements
  bool           is_fused;     // does this kernel implement a fused op?
};

// ---------------------------------------------------------------------------
// GraphEntry — one compiled graph in the package
// ---------------------------------------------------------------------------
struct GraphEntry {
  uint32_t       index;           // 0-based index (used by selector)
  std::string    id;              // e.g. "snapdragon_8_gen3_int8_2gb_interactive"
  std::string    path;            // relative path inside .armpack

  // Conditions this graph was optimized for
  std::string    soc_name;
  std::string    quant_dtype;
  uint32_t       memory_budget_mb;
  uint32_t       context_length;
  std::string    thermal_state;
  std::string    latency_mode;

  // Performance estimates
  float          estimated_ttft_ms;   // Time-to-first-token
  float          estimated_tpot_ms;   // Time per output token

  // Memory
  uint64_t       peak_memory_bytes;
  uint64_t       weight_bytes;

  // Kernel references (IDs of KernelEntry)
  std::vector<std::string> kernel_ids;
};

// ---------------------------------------------------------------------------
// WeightEntry — one quantized weight file
// ---------------------------------------------------------------------------
struct WeightEntry {
  std::string    id;            // e.g. "weights_int4_g128"
  std::string    path;          // relative path inside .armpack
  std::string    dtype;         // "int4", "int8", "fp16", etc.
  uint64_t       size_bytes;
  bool           mmap_safe;     // Can be mmap'd directly?
  uint32_t       page_alignment;  // Bytes; typically 4096
};

// ---------------------------------------------------------------------------
// Package manifest — the top-level JSON index of the entire .armpack
// ---------------------------------------------------------------------------
struct PackageManifest {
  // Package identity
  std::string    armpack_version = "1.0";
  std::string    compiler_version;
  std::string    model_id;          // HuggingFace model ID
  std::string    model_family;      // "llama_style", "gpt2_style", etc.
  std::string    created_at;        // ISO 8601 timestamp

  // Model architecture
  uint32_t       num_layers;
  uint32_t       hidden_size;
  uint32_t       num_heads;
  uint32_t       num_kv_heads;
  uint32_t       vocab_size;
  uint32_t       max_position_embeddings;

  // Contents
  std::vector<GraphEntry>   graphs;
  std::vector<KernelEntry>  kernels;
  std::vector<WeightEntry>  weights;

  // Paths to special files
  std::string    tokenizer_path    = "tokenizer/";
  std::string    selector_wasm_path = "selector.wasm";
  std::string    runtime_config_path = "runtime_config.json";
  std::string    memory_maps_path  = "memory_maps/";
  std::string    device_profile_path;

  // Serialization
  std::string toJSON(bool pretty = true) const;
  static PackageManifest fromJSON(const std::string& json);
};

// ---------------------------------------------------------------------------
// RuntimeConfig — generation parameters embedded in the package
// ---------------------------------------------------------------------------
struct RuntimeConfig {
  // Sampling defaults
  float    temperature       = 1.0f;
  float    top_p             = 0.9f;
  int32_t  top_k             = 50;
  float    repetition_penalty = 1.0f;

  // Generation
  int32_t  max_new_tokens    = 512;
  int32_t  min_new_tokens    = 0;

  // Special token IDs
  int32_t  bos_token_id      = 1;
  int32_t  eos_token_id      = 2;
  std::vector<int32_t> eos_token_ids;
  int32_t  pad_token_id      = 0;

  // Chat template
  std::string chat_template;

  std::string toJSON(bool pretty = true) const;
  static RuntimeConfig fromJSON(const std::string& json);
};

} // namespace pkg
} // namespace armcc
