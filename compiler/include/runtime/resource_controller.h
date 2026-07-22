// ============================================================================
// SiliconGraph Runtime Resource Controller
//
// Selects an already-compiled graph from live Android resource signals. The
// controller is deliberately stateless so an Android runtime can call it
// before prefill, after memory-trim callbacks, and between decode batches.
// ============================================================================
#pragma once

#include "generator/graph_selector.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace armcc::runtime {

struct ResourceControllerConfig {
  uint32_t min_system_reserve_mb = 512;
  uint32_t low_memory_extra_reserve_mb = 256;
  uint32_t full_max_new_tokens = 512;
  uint32_t direct_max_new_tokens = 128;
  uint32_t full_prefill_chunk_tokens = 256;
  uint32_t constrained_prefill_chunk_tokens = 64;
  uint32_t direct_answer_battery_pct = 50;
  uint32_t hot_thermal_status = 2;
  float hot_cpu_c = 55.0f;
  std::vector<std::string> supported_input_modalities{"text"};
  uint32_t media_min_safe_ram_mb = 768;
  uint32_t default_image_working_set_mb = 96;
  uint32_t default_audio_working_set_mb = 64;
  uint32_t default_video_working_set_mb = 256;
  std::string direct_answer_instruction =
      "Answer directly and concisely. Do not produce a long explanation unless explicitly requested.";
  std::vector<uint32_t> context_buckets = {512, 2048};

  static std::optional<ResourceControllerConfig> fromJSON(
      const std::string& json, std::string& error);
};

// These values come from the Android application process, not the Windows
// desktop profiler. `available_ram_mb`, `low_memory`, and `memory_threshold_mb`
// map directly to ActivityManager.MemoryInfo. Vulkan values are populated from
// the app's allocator and VK_EXT_memory_budget when that extension is present.
struct RuntimeResourceState {
  uint32_t available_ram_mb = 0;
  uint32_t memory_threshold_mb = 0;
  bool low_memory = false;

  uint32_t selected_weight_mb = 0;
  uint32_t activation_mb = 0;
  uint32_t scratch_mb = 0;
  uint64_t kv_bytes_per_token = 0;

  // Populated after cheap media metadata inspection and before full decoding.
  // An adapter may provide an exact working-set estimate; otherwise the policy
  // reserves conservative image/audio/video defaults.
  bool image_input_requested = false;
  bool audio_input_requested = false;
  bool video_input_requested = false;
  uint32_t media_working_set_mb = 0;

  int battery_pct = 100;
  int thermal_status = 0;
  float cpu_temperature_c = 0.0f;

  bool vulkan_runtime_available = false;
  bool vulkan_benchmark_wins = false;
  uint32_t vulkan_budget_mb = 0;  // 0 means the driver did not expose it.
  uint32_t vulkan_usage_mb = 0;
  uint32_t required_vulkan_mb = 0;
};

struct ResourceDecision {
  bool can_run = false;
  uint32_t safe_ram_mb = 0;
  uint32_t kv_cache_budget_mb = 0;
  uint32_t context_length = 0;
  uint32_t prefill_chunk_tokens = 0;
  uint32_t max_new_tokens = 0;
  bool direct_answer_mode = false;
  bool media_inputs_accepted = false;
  bool preprocess_image_and_audio_in_parallel = false;
  uint32_t media_working_set_mb = 0;
  std::string selected_quant_dtype;
  std::string generation_instruction;
  bool use_vulkan = false;
  std::optional<uint32_t> graph_index;
  std::string reason;
};

class RuntimeResourceController {
public:
  explicit RuntimeResourceController(ResourceControllerConfig config);

  ResourceDecision decide(const RuntimeResourceState& state,
                          const generator::GraphSelector& selector) const;

private:
  ResourceControllerConfig config_;

  uint32_t safeRamMB(const RuntimeResourceState& state) const;
  std::optional<uint32_t> contextThatFits(const RuntimeResourceState& state,
                                          uint32_t safe_ram_mb,
                                          uint32_t& kv_budget_mb) const;
  bool directAnswerMode(const RuntimeResourceState& state) const;
  bool thermalConstraintApplies(const RuntimeResourceState& state) const;
  bool vulkanMemoryFits(const RuntimeResourceState& state) const;
  bool mediaInputRequested(const RuntimeResourceState& state) const;
  bool mediaRequestSupported(const RuntimeResourceState& state,
                             std::string& error) const;
  uint32_t mediaWorkingSetMB(const RuntimeResourceState& state) const;
};

} // namespace armcc::runtime
