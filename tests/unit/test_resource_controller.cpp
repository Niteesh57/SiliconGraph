#include "runtime/resource_controller.h"

#include <gtest/gtest.h>

namespace {

constexpr uint64_t MiB = 1024ULL * 1024ULL;

armcc::generator::GraphSelector selector() {
  armcc::generator::GraphConditions cpu;
  cpu.soc_id = armcc::ir::SoCID::Generic_ARM64;
  cpu.context_length = 512;
  cpu.execution_backend = armcc::ir::ExecUnit::CPU;
  cpu.preferred_units = {armcc::ir::ExecUnit::CPU};

  auto vulkan = cpu;
  vulkan.execution_backend = armcc::ir::ExecUnit::GPU;
  vulkan.preferred_units = {armcc::ir::ExecUnit::GPU};

  return armcc::generator::GraphSelector({
      {0, cpu, 4.0f, 700 * MiB},
      {1, vulkan, 2.0f, 700 * MiB},
  });
}

armcc::runtime::RuntimeResourceState baseState() {
  armcc::runtime::RuntimeResourceState state;
  state.available_ram_mb = 2048;
  state.selected_weight_mb = 512;
  state.activation_mb = 128;
  state.scratch_mb = 64;
  state.kv_bytes_per_token = MiB;
  state.vulkan_runtime_available = true;
  state.vulkan_benchmark_wins = true;
  state.vulkan_budget_mb = 512;
  state.vulkan_usage_mb = 64;
  state.required_vulkan_mb = 128;
  return state;
}

TEST(RuntimeResourceController, UsesOnlyTheCurrentSafeRamAndChoosesVulkan) {
  armcc::runtime::ResourceControllerConfig config;
  config.context_buckets = {128, 512, 2048};
  armcc::runtime::RuntimeResourceController controller(config);

  const auto decision = controller.decide(baseState(), selector());

  EXPECT_TRUE(decision.can_run);
  EXPECT_EQ(decision.safe_ram_mb, 1536u);
  EXPECT_EQ(decision.kv_cache_budget_mb, 832u);
  EXPECT_EQ(decision.context_length, 512u);
  EXPECT_TRUE(decision.use_vulkan);
  ASSERT_TRUE(decision.graph_index.has_value());
  EXPECT_EQ(*decision.graph_index, 1u);
}

TEST(RuntimeResourceController, OnlyMakesAnswersDirectBelowFiftyPercent) {
  armcc::runtime::ResourceControllerConfig config;
  config.context_buckets = {128, 512, 2048};
  armcc::runtime::RuntimeResourceController controller(config);

  auto full_battery = baseState();
  full_battery.battery_pct = 50;
  const auto full = controller.decide(full_battery, selector());
  EXPECT_FALSE(full.direct_answer_mode);
  EXPECT_EQ(full.max_new_tokens, 512u);

  auto low_battery_hot = baseState();
  low_battery_hot.battery_pct = 49;
  low_battery_hot.thermal_status = 2;
  const auto direct = controller.decide(low_battery_hot, selector());
  EXPECT_TRUE(direct.direct_answer_mode);
  EXPECT_EQ(direct.max_new_tokens, 128u);
  EXPECT_FALSE(direct.use_vulkan);
}

TEST(RuntimeResourceController, RefusesBeforeAllocatingWhenMemoryIsInsufficient) {
  armcc::runtime::ResourceControllerConfig config;
  config.context_buckets = {128, 512};
  armcc::runtime::RuntimeResourceController controller(config);

  auto state = baseState();
  state.available_ram_mb = 1000;
  state.low_memory = true;
  const auto decision = controller.decide(state, selector());

  EXPECT_FALSE(decision.can_run);
  EXPECT_EQ(decision.reason, "insufficient_current_free_ram");
}

TEST(RuntimeResourceController, AdmitsImageAndAudioTogetherOnMediumOrFullPrecision) {
  armcc::runtime::ResourceControllerConfig config;
  config.context_buckets = {128, 512, 2048};
  config.supported_input_modalities = {"text", "image", "audio"};
  armcc::runtime::RuntimeResourceController controller(config);

  auto state = baseState();
  state.image_input_requested = true;
  state.audio_input_requested = true;
  state.media_working_set_mb = 160;
  const auto decision = controller.decide(state, selector());

  EXPECT_TRUE(decision.can_run);
  EXPECT_TRUE(decision.media_inputs_accepted);
  EXPECT_TRUE(decision.preprocess_image_and_audio_in_parallel);
  EXPECT_EQ(decision.media_working_set_mb, 160u);
  EXPECT_EQ(decision.selected_quant_dtype, "i8");
}

TEST(RuntimeResourceController, RejectsMediaWhenOnlyLowPrecisionGraphExists) {
  armcc::generator::GraphConditions int4;
  int4.soc_id = armcc::ir::SoCID::Generic_ARM64;
  int4.context_length = 512;
  int4.quant_dtype = armcc::ir::DType::I4;
  int4.execution_backend = armcc::ir::ExecUnit::CPU;
  int4.preferred_units = {armcc::ir::ExecUnit::CPU};
  armcc::generator::GraphSelector int4_selector({{0, int4, 2.0f, 700 * MiB}});

  armcc::runtime::ResourceControllerConfig config;
  config.context_buckets = {128, 512};
  config.supported_input_modalities = {"text", "image"};
  armcc::runtime::RuntimeResourceController controller(config);

  auto state = baseState();
  state.image_input_requested = true;
  const auto decision = controller.decide(state, int4_selector);

  EXPECT_FALSE(decision.can_run);
  EXPECT_EQ(decision.reason, "no_compiled_full_or_medium_precision_graph_for_media");
}

TEST(RuntimeResourceController, ReservesMediaMemoryBeforeSelectingAGraph) {
  armcc::generator::GraphConditions cpu;
  cpu.soc_id = armcc::ir::SoCID::Generic_ARM64;
  cpu.context_length = 512;
  cpu.quant_dtype = armcc::ir::DType::I8;
  cpu.execution_backend = armcc::ir::ExecUnit::CPU;
  cpu.preferred_units = {armcc::ir::ExecUnit::CPU};
  armcc::generator::GraphSelector selector_with_large_candidate({
      {0, cpu, 1.0f, 1400 * MiB},
      {1, cpu, 2.0f, 700 * MiB},
  });

  armcc::runtime::ResourceControllerConfig config;
  config.context_buckets = {128, 512};
  config.supported_input_modalities = {"text", "image"};
  armcc::runtime::RuntimeResourceController controller(config);

  auto state = baseState();
  state.image_input_requested = true;
  state.media_working_set_mb = 160;
  const auto decision = controller.decide(state, selector_with_large_candidate);

  ASSERT_TRUE(decision.can_run);
  ASSERT_TRUE(decision.graph_index.has_value());
  EXPECT_EQ(*decision.graph_index, 1u);
}

} // namespace
