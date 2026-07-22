// ============================================================================
// SiliconGraph Runtime Resource Controller (implementation)
// ============================================================================
#include "runtime/resource_controller.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <limits>

namespace armcc::runtime {
namespace {

constexpr uint64_t MiB = 1024ULL * 1024ULL;

uint32_t positiveUInt(const nlohmann::json& object, const char* key,
                      uint32_t fallback) {
  if (!object.contains(key) || !object[key].is_number_unsigned()) return fallback;
  const auto value = object[key].get<uint64_t>();
  return value > std::numeric_limits<uint32_t>::max()
      ? fallback : static_cast<uint32_t>(value);
}

std::vector<std::string> stringArray(const nlohmann::json& object,
                                     const char* key,
                                     std::vector<std::string> fallback) {
  if (!object.contains(key) || !object[key].is_array()) return fallback;
  std::vector<std::string> values;
  for (const auto& value : object[key]) {
    if (value.is_string()) values.push_back(value.get<std::string>());
  }
  return values.empty() ? fallback : values;
}

} // namespace

std::optional<ResourceControllerConfig> ResourceControllerConfig::fromJSON(
    const std::string& text, std::string& error) {
  try {
    const auto policy = nlohmann::json::parse(text);
    ResourceControllerConfig config;
    const auto memory = policy.value("memory", nlohmann::json::object());
    const auto normal = policy.value("normal", nlohmann::json::object());
    const auto battery = policy.value("battery_aware", nlohmann::json::object());
    const auto when = battery.value("when", nlohmann::json::object());
    const auto admission = policy.value("input_admission", nlohmann::json::object());
    const auto defaults = admission.value(
        "default_media_working_set_mb", nlohmann::json::object());

    config.min_system_reserve_mb = positiveUInt(
        memory, "min_system_reserve_mb", config.min_system_reserve_mb);
    config.low_memory_extra_reserve_mb = positiveUInt(
        memory, "low_memory_extra_reserve_mb", config.low_memory_extra_reserve_mb);
    config.full_max_new_tokens = positiveUInt(
        normal, "max_new_tokens", config.full_max_new_tokens);
    config.direct_max_new_tokens = positiveUInt(
        battery, "max_new_tokens", config.direct_max_new_tokens);
    config.direct_answer_battery_pct = positiveUInt(
        when, "battery_pct_below", config.direct_answer_battery_pct);
    config.direct_answer_instruction = battery.value(
        "instruction", config.direct_answer_instruction);
    config.supported_input_modalities = stringArray(
        admission, "supported_inputs", config.supported_input_modalities);
    config.media_min_safe_ram_mb = positiveUInt(
        admission, "minimum_safe_ram_mb_for_media", config.media_min_safe_ram_mb);
    config.default_image_working_set_mb = positiveUInt(
        defaults, "image", config.default_image_working_set_mb);
    config.default_audio_working_set_mb = positiveUInt(
        defaults, "audio", config.default_audio_working_set_mb);
    config.default_video_working_set_mb = positiveUInt(
        defaults, "video", config.default_video_working_set_mb);

    if (policy.contains("context_buckets") && policy["context_buckets"].is_array()) {
      config.context_buckets.clear();
      for (const auto& value : policy["context_buckets"]) {
        if (value.is_number_unsigned() && value.get<uint32_t>() > 0) {
          config.context_buckets.push_back(value.get<uint32_t>());
        }
      }
      std::sort(config.context_buckets.begin(), config.context_buckets.end());
      config.context_buckets.erase(
          std::unique(config.context_buckets.begin(), config.context_buckets.end()),
          config.context_buckets.end());
    }
    if (config.context_buckets.empty()) {
      error = "execution policy has no valid context buckets";
      return std::nullopt;
    }
    return config;
  } catch (const std::exception& exception) {
    error = std::string("invalid execution policy: ") + exception.what();
    return std::nullopt;
  }
}

RuntimeResourceController::RuntimeResourceController(ResourceControllerConfig config)
    : config_(std::move(config)) {}

uint32_t RuntimeResourceController::safeRamMB(
    const RuntimeResourceState& state) const {
  uint64_t reserve = std::max(
      static_cast<uint64_t>(config_.min_system_reserve_mb),
      static_cast<uint64_t>(state.memory_threshold_mb));
  if (state.low_memory) reserve += config_.low_memory_extra_reserve_mb;
  return state.available_ram_mb > reserve
      ? static_cast<uint32_t>(state.available_ram_mb - reserve) : 0;
}

std::optional<uint32_t> RuntimeResourceController::contextThatFits(
    const RuntimeResourceState& state, uint32_t safe_ram_mb,
    uint32_t& kv_budget_mb) const {
  const uint64_t fixed_bytes = (
      static_cast<uint64_t>(state.selected_weight_mb) + state.activation_mb +
      state.scratch_mb + mediaWorkingSetMB(state)) * MiB;
  const uint64_t safe_bytes = static_cast<uint64_t>(safe_ram_mb) * MiB;
  if (safe_bytes <= fixed_bytes) {
    kv_budget_mb = 0;
    return std::nullopt;
  }

  const uint64_t kv_budget_bytes = safe_bytes - fixed_bytes;
  kv_budget_mb = static_cast<uint32_t>(kv_budget_bytes / MiB);
  if (state.kv_bytes_per_token == 0) return config_.context_buckets.back();

  const uint64_t max_context = kv_budget_bytes / state.kv_bytes_per_token;
  std::optional<uint32_t> selected;
  for (const auto bucket : config_.context_buckets) {
    if (bucket <= max_context) selected = bucket;
  }
  return selected;
}

bool RuntimeResourceController::directAnswerMode(
    const RuntimeResourceState& state) const {
  return state.battery_pct >= 0 &&
      static_cast<uint32_t>(state.battery_pct) < config_.direct_answer_battery_pct;
}

bool RuntimeResourceController::thermalConstraintApplies(
    const RuntimeResourceState& state) const {
  if (!directAnswerMode(state)) return false;
  return state.thermal_status >= static_cast<int>(config_.hot_thermal_status) ||
      state.cpu_temperature_c >= config_.hot_cpu_c;
}

bool RuntimeResourceController::vulkanMemoryFits(
    const RuntimeResourceState& state) const {
  if (!state.vulkan_runtime_available || !state.vulkan_benchmark_wins) return false;
  if (state.vulkan_budget_mb == 0) return true;
  const uint64_t needed = static_cast<uint64_t>(state.vulkan_usage_mb) +
      state.required_vulkan_mb;
  return needed <= state.vulkan_budget_mb;
}

bool RuntimeResourceController::mediaInputRequested(
    const RuntimeResourceState& state) const {
  return state.image_input_requested || state.audio_input_requested ||
      state.video_input_requested;
}

bool RuntimeResourceController::mediaRequestSupported(
    const RuntimeResourceState& state, std::string& error) const {
  const auto supports = [this](const char* modality) {
    return std::find(config_.supported_input_modalities.begin(),
                     config_.supported_input_modalities.end(), modality) !=
        config_.supported_input_modalities.end();
  };
  if (state.image_input_requested && !supports("image")) {
    error = "model_does_not_support_image_input";
    return false;
  }
  if (state.audio_input_requested && !supports("audio")) {
    error = "model_does_not_support_audio_input";
    return false;
  }
  if (state.video_input_requested && !supports("video")) {
    error = "model_does_not_support_video_input";
    return false;
  }
  return true;
}

uint32_t RuntimeResourceController::mediaWorkingSetMB(
    const RuntimeResourceState& state) const {
  if (!mediaInputRequested(state)) return 0;
  if (state.media_working_set_mb > 0) return state.media_working_set_mb;
  uint64_t total = 0;
  if (state.image_input_requested) total += config_.default_image_working_set_mb;
  if (state.audio_input_requested) total += config_.default_audio_working_set_mb;
  if (state.video_input_requested) total += config_.default_video_working_set_mb;
  return total > std::numeric_limits<uint32_t>::max()
      ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(total);
}

ResourceDecision RuntimeResourceController::decide(
    const RuntimeResourceState& state,
    const generator::GraphSelector& selector) const {
  ResourceDecision decision;
  decision.safe_ram_mb = safeRamMB(state);
  const bool has_media = mediaInputRequested(state);
  decision.media_working_set_mb = mediaWorkingSetMB(state);
  decision.preprocess_image_and_audio_in_parallel =
      state.image_input_requested && state.audio_input_requested;
  if (has_media) {
    if (!mediaRequestSupported(state, decision.reason)) return decision;
    if (decision.safe_ram_mb < config_.media_min_safe_ram_mb ||
        decision.safe_ram_mb < decision.media_working_set_mb) {
      decision.reason = "insufficient_current_free_ram_for_media";
      return decision;
    }
  }
  const auto context = contextThatFits(
      state, decision.safe_ram_mb, decision.kv_cache_budget_mb);
  if (!context) {
    decision.reason = "insufficient_current_free_ram";
    return decision;
  }

  decision.context_length = *context;
  decision.direct_answer_mode = directAnswerMode(state);
  decision.max_new_tokens = decision.direct_answer_mode
      ? config_.direct_max_new_tokens : config_.full_max_new_tokens;
  decision.generation_instruction = decision.direct_answer_mode
      ? config_.direct_answer_instruction : "";
  decision.prefill_chunk_tokens = *context == config_.context_buckets.back()
      ? config_.full_prefill_chunk_tokens : config_.constrained_prefill_chunk_tokens;

  generator::RuntimeDeviceState graph_state;
  // The selected graph must fit after media decode/feature tensors have been
  // reserved; otherwise a graph that fits by itself could still OOM when an
  // image or audio request arrives.
  graph_state.ram_available_mb = decision.safe_ram_mb > decision.media_working_set_mb
      ? decision.safe_ram_mb - decision.media_working_set_mb : 0;
  graph_state.context_length = *context;
  graph_state.thermal = thermalConstraintApplies(state)
      ? ir::ThermalState::Hot : ir::ThermalState::Nominal;
  graph_state.vulkan_benchmark_wins = vulkanMemoryFits(state) &&
      !thermalConstraintApplies(state);
  graph_state.require_medium_or_higher_precision = has_media;

  decision.graph_index = selector.selectViable(graph_state);
  if (!decision.graph_index) {
    decision.reason = has_media
        ? "no_compiled_full_or_medium_precision_graph_for_media"
        : "no_compiled_graph_fits_current_safe_ram";
    return decision;
  }

  decision.can_run = true;
  decision.media_inputs_accepted = has_media;
  if (const auto dtype = selector.quantDTypeForGraph(*decision.graph_index)) {
    decision.selected_quant_dtype = ir::dtypeToString(*dtype);
  }
  decision.use_vulkan = selector.isVulkanGraph(*decision.graph_index);
  if (thermalConstraintApplies(state)) {
    decision.reason = "battery_below_50_and_thermal_constraint";
  } else if (decision.direct_answer_mode) {
    decision.reason = "battery_below_50_direct_answer";
  } else if (decision.use_vulkan) {
    decision.reason = "vulkan_benchmark_and_memory_winner";
  } else {
    decision.reason = "cpu_memory_safe_fallback";
  }
  return decision;
}

} // namespace armcc::runtime
