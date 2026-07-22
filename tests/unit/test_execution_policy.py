from armcc.execution_policy import build_execution_policy, select_execution_plan
from armcc.modalities import ModalityCapabilities


def profile(vulkan=True, memory_total_mb=6144):
    return {
        "gpu": {"present": vulkan, "supports_vulkan": vulkan},
        "runtime_state": {"memory_total_mb": memory_total_mb},
    }


def test_vulkan_is_only_selected_after_a_faster_measurement():
    policy = build_execution_policy(profile(), "512,2048")

    assert select_execution_plan(policy, {}, {})["backend"] == "cpu"
    assert select_execution_plan(policy, {"cpu_decode_ms": 10, "vulkan_decode_ms": 8}, {})["backend"] == "vulkan"
    assert select_execution_plan(policy, {"cpu_decode_ms": 8, "vulkan_decode_ms": 10}, {})["backend"] == "cpu"


def test_battery_below_fifty_uses_direct_answers_without_precision_change():
    policy = build_execution_policy(profile(), [512, 2048])
    measured_gpu = {"cpu_decode_ms": 10, "vulkan_decode_ms": 5}

    hot = select_execution_plan(policy, measured_gpu, {"cpu_temperature_c": 58})
    low_battery = select_execution_plan(policy, measured_gpu, {"battery_pct": 15})
    empty_battery = select_execution_plan(policy, measured_gpu, {"battery_pct": 0})

    assert hot["response_mode"] == "full"
    assert hot["backend"] == "vulkan"
    assert low_battery["response_mode"] == "direct"
    assert low_battery["precision"] == "quality_validated_static_variant"
    assert low_battery["max_new_tokens"] == 128
    assert empty_battery["response_mode"] == "direct"


def test_current_free_ram_limits_context_and_refuses_before_oom():
    policy = build_execution_policy(profile(), [128, 512, 2048])
    base_state = {
        "available_ram_mb": 2048,
        "selected_weight_mb": 512,
        "activation_mb": 128,
        "scratch_mb": 64,
        "kv_bytes_per_token": 1024 * 1024,
    }

    constrained = select_execution_plan(policy, {}, base_state)
    assert constrained["can_run"] is True
    assert constrained["context_length"] == 512
    assert constrained["kv_cache_budget_mb"] == 832

    no_room = select_execution_plan(policy, {}, {**base_state, "available_ram_mb": 1000, "low_memory": True})
    assert no_room["can_run"] is False
    assert no_room["backend"] == "cpu"
    assert no_room["context_length"] == 0
    assert no_room["safe_memory_mb"] == 232
    assert no_room["reason"] == "insufficient_current_free_ram"


def test_image_and_audio_are_admitted_together_only_above_low_precision():
    capabilities = ModalityCapabilities(
        task="image_audio_text_to_text", inputs=("text", "image", "audio"),
        outputs=("text",), primary_input="text", primary_output="text",
        native_status="experimental",
    )
    state = {
        "available_ram_mb": 2048,
        "selected_weight_mb": 512,
        "activation_mb": 128,
        "scratch_mb": 64,
        "kv_bytes_per_token": 1024 * 1024,
    }
    medium = build_execution_policy(profile(), [128, 512], capabilities, "mixed")

    accepted = select_execution_plan(medium, {}, state, ["text", "image", "audio"])
    assert accepted["can_run"] is True
    assert accepted["input_admission"]["accepted"] is True
    assert accepted["input_admission"]["parallel_preprocessing"] is True
    assert accepted["context_length"] == 512

    low = build_execution_policy(profile(), [128, 512], capabilities, "int4")
    blocked = select_execution_plan(low, {}, state, ["text", "image"])
    assert blocked["can_run"] is False
    assert blocked["reason"] == "media_requires_full_or_medium_precision"

    text_only = select_execution_plan(low, {}, state, ["text"])
    assert text_only["can_run"] is True


def test_media_is_refused_early_when_current_free_ram_is_too_low():
    capabilities = ModalityCapabilities(
        task="image_text_to_text", inputs=("text", "image"), outputs=("text",),
        primary_input="image", primary_output="text", native_status="experimental",
    )
    policy = build_execution_policy(profile(), [128, 512], capabilities, "int8")

    plan = select_execution_plan(
        policy, {},
        {"available_ram_mb": 1000, "selected_weight_mb": 512},
        ["text", "image"],
    )

    assert plan["can_run"] is False
    assert plan["reason"] == "insufficient_current_free_ram_for_media"


def test_cpu_only_policy_cannot_select_vulkan_without_live_vulkan_support():
    policy = build_execution_policy(profile(vulkan=False), [512, 2048])

    assert policy["backends"]["vulkan"]["enabled"] is False
    assert select_execution_plan(policy, {"cpu_decode_ms": 10, "vulkan_decode_ms": 1}, {})["backend"] == "cpu"


def test_six_gb_profile_generates_capacity_limited_graph_budgets():
    policy = build_execution_policy(profile(memory_total_mb=6144), [512, 2048])

    assert policy["device_capacity"]["compile_reserve_mb"] == 922
    assert policy["device_capacity"]["graph_memory_budgets_mb"] == [512, 1024, 2048, 4096, 5120]
