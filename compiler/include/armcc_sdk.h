// ============================================================================
// ARM AI Compiler SDK — C API
//
// A thin C wrapper over the compiler's C++ internals.
// This is the integration surface for runtimes (Cactus, llama.cpp, custom).
//
// Usage example:
//
//   ArmPackage* pkg = armpack_load("llama-3.2-1b.armpack");
//
//   ArmDeviceState state = {
//     .soc_id        = ARMCC_SOC_SNAPDRAGON_8_GEN3,
//     .ram_mb        = 1500,
//     .battery_pct   = 85,
//     .thermal_state = ARMCC_THERMAL_NOMINAL,
//     .latency_mode  = ARMCC_LATENCY_INTERACTIVE,
//   };
//
//   ArmGraph* graph = armpack_select_graph(pkg, &state);
//   ArmSession* sess = armgraph_create_session(graph, NULL);
//
//   int32_t tokens[] = {1, 2, 3};
//   ArmInferenceResult result = armsess_run(sess, tokens, 3);
//   printf("Next token: %d\n", result.next_token_id);
//
//   armsess_destroy(sess);
//   armpack_destroy(pkg);
// ============================================================================
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Version ─────────────────────────────────────────────────────────────────
#define ARMCC_SDK_VERSION_MAJOR 0
#define ARMCC_SDK_VERSION_MINOR 1
#define ARMCC_SDK_VERSION_PATCH 0

// ── Enumerations ────────────────────────────────────────────────────────────

typedef enum {
  ARMCC_SOC_UNKNOWN                 = 0,
  ARMCC_SOC_SNAPDRAGON_8_GEN1       = 1,
  ARMCC_SOC_SNAPDRAGON_8_GEN2       = 2,
  ARMCC_SOC_SNAPDRAGON_8_GEN3       = 3,
  ARMCC_SOC_SNAPDRAGON_8_ELITE      = 4,
  ARMCC_SOC_SNAPDRAGON_7_GEN3       = 5,
  ARMCC_SOC_DIMENSITY_9300          = 10,
  ARMCC_SOC_DIMENSITY_9200          = 11,
  ARMCC_SOC_DIMENSITY_8300          = 12,
  ARMCC_SOC_EXYNOS_2400             = 20,
  ARMCC_SOC_EXYNOS_2200             = 21,
  ARMCC_SOC_APPLE_A17               = 30,
  ARMCC_SOC_APPLE_A18               = 31,
  ARMCC_SOC_APPLE_M3                = 32,
  ARMCC_SOC_APPLE_M4                = 33,
  ARMCC_SOC_BCM2712                 = 40,
  ARMCC_SOC_JETSON_ORIN_NX          = 50,
  ARMCC_SOC_GENERIC_ARM64           = 0xFF,
} ArmSoCID;

typedef enum {
  ARMCC_THERMAL_NOMINAL  = 0,
  ARMCC_THERMAL_WARM     = 1,
  ARMCC_THERMAL_HOT      = 2,
  ARMCC_THERMAL_CRITICAL = 3,
} ArmThermalState;

typedef enum {
  ARMCC_LATENCY_INTERACTIVE = 0,
  ARMCC_LATENCY_BATCH       = 1,
  ARMCC_LATENCY_BACKGROUND  = 2,
} ArmLatencyMode;

typedef enum {
  ARMCC_OK                  =  0,
  ARMCC_ERR_NOT_FOUND       = -1,
  ARMCC_ERR_INVALID_PACKAGE = -2,
  ARMCC_ERR_OOM             = -3,
  ARMCC_ERR_UNSUPPORTED     = -4,
  ARMCC_ERR_RUNTIME         = -5,
} ArmStatus;

// ── Opaque handles ───────────────────────────────────────────────────────────
typedef struct ArmPackage_  ArmPackage;
typedef struct ArmGraph_    ArmGraph;
typedef struct ArmSession_  ArmSession;

// ── Device state ─────────────────────────────────────────────────────────────
typedef struct {
  ArmSoCID        soc_id;
  uint32_t        ram_mb;           // Currently available RAM in MB
  int32_t         battery_pct;      // 0–100
  ArmThermalState thermal_state;
  ArmLatencyMode  latency_mode;
  uint32_t        context_length;   // Expected prompt + output length
  bool            gpu_available;
  bool            npu_available;
} ArmDeviceState;

// ── Session options ───────────────────────────────────────────────────────────
typedef struct {
  int32_t  num_threads;    // CPU threads (0 = auto)
  bool     use_gpu;        // Enable GPU execution units
  bool     use_npu;        // Enable NPU execution units
  uint32_t seed;           // Random seed (for sampling)
  bool     verbose;
} ArmSessionOptions;

// ── Inference result ─────────────────────────────────────────────────────────
typedef struct {
  int32_t   next_token_id;
  float     logprob;            // Log-probability of the chosen token
  float     ttft_ms;            // Time-to-first-token (ms)
  float     tpot_ms;            // Time-per-output-token (ms)
  uint32_t  tokens_generated;
  ArmStatus status;
} ArmInferenceResult;

// ── Package API ──────────────────────────────────────────────────────────────

// Load an .armpack file. Returns NULL on failure.
ArmPackage* armpack_load(const char* path);

// Release all resources
void        armpack_destroy(ArmPackage* pkg);

// Get the last error message
const char* armpack_last_error(void);

// Select the best graph for the given device state
// Returns NULL if no suitable graph found.
ArmGraph*   armpack_select_graph(ArmPackage* pkg,
                                 const ArmDeviceState* state);

// Get total number of graphs in the package
uint32_t    armpack_num_graphs(const ArmPackage* pkg);

// Get graph by explicit index (0-based)
ArmGraph*   armpack_get_graph(ArmPackage* pkg, uint32_t index);

// Human-readable description of a graph
const char* armgraph_description(const ArmGraph* graph);

// Estimated latency (ms/token) on the target device
float       armgraph_estimated_tpot_ms(const ArmGraph* graph);

// Peak memory usage in bytes
uint64_t    armgraph_peak_memory_bytes(const ArmGraph* graph);

// ── Session API ──────────────────────────────────────────────────────────────

// Create an inference session for a graph.
// opts may be NULL (uses defaults).
ArmSession* armgraph_create_session(ArmGraph* graph,
                                    const ArmSessionOptions* opts);

void        armsess_destroy(ArmSession* sess);

// Reset KV cache (start a new conversation)
void        armsess_reset(ArmSession* sess);

// Prefill prompt tokens (returns after prefill completes)
ArmStatus   armsess_prefill(ArmSession*    sess,
                            const int32_t* tokens,
                            uint32_t       n_tokens);

// Decode next token (call repeatedly for autoregressive generation)
ArmInferenceResult armsess_decode(ArmSession* sess);

// Convenience: prefill + decode n_tokens in one call
// Calls output_cb for each generated token.
ArmStatus   armsess_generate(ArmSession*    sess,
                             const int32_t* prompt_tokens,
                             uint32_t       prompt_len,
                             uint32_t       max_new_tokens,
                             void (*output_cb)(int32_t token_id, void* ctx),
                             void*          ctx);

// ── Package inspection ───────────────────────────────────────────────────────
typedef struct {
  char     model_id[256];
  char     model_family[64];
  char     compiler_version[32];
  uint32_t num_graphs;
  uint32_t num_kernels;
  uint32_t num_weight_files;
  uint32_t num_layers;
  uint32_t hidden_size;
  uint32_t vocab_size;
} ArmPackageInfo;

ArmStatus armpack_get_info(const ArmPackage* pkg, ArmPackageInfo* info);

// Print a human-readable summary of the package contents
void      armpack_print_summary(const ArmPackage* pkg);

// ── Tokenizer API ────────────────────────────────────────────────────────────
typedef struct ArmTokenizer_ ArmTokenizer;

ArmTokenizer* armpack_get_tokenizer(ArmPackage* pkg);
int32_t*      armtok_encode(ArmTokenizer* tok,
                            const char*   text,
                            uint32_t*     out_len);
char*         armtok_decode(ArmTokenizer* tok,
                            const int32_t* tokens,
                            uint32_t       n_tokens);
void          armtok_free_text(char* text);
void          armtok_free_tokens(int32_t* tokens);

#ifdef __cplusplus
}  // extern "C"
#endif
