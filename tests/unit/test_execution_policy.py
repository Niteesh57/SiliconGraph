from armcc.execution_policy import build_execution_policy, select_execution_plan


def profile(vulkan=True):
    return {"gpu": {"present": vulkan, "supports_vulkan": vulkan}}


def test_vulkan_is_only_selected_after_a_faster_measurement():
    policy = build_execution_policy(profile(), "512,2048")

    assert select_execution_plan(policy, {}, {})["backend"] == "cpu"
    assert select_execution_plan(policy, {"cpu_decode_ms": 10, "vulkan_decode_ms": 8}, {})["backend"] == "vulkan"
    assert select_execution_plan(policy, {"cpu_decode_ms": 8, "vulkan_decode_ms": 10}, {})["backend"] == "cpu"


def test_thermal_or_low_battery_forces_small_cpu_plan():
    policy = build_execution_policy(profile(), [512, 2048])
    measured_gpu = {"cpu_decode_ms": 10, "vulkan_decode_ms": 5}

    hot = select_execution_plan(policy, measured_gpu, {"cpu_temperature_c": 58})
    low_battery = select_execution_plan(policy, measured_gpu, {"battery_pct": 15})

    assert hot == {"backend": "cpu", "quantization": "i8", "context_length": 512, "reason": "thermal_or_battery_constraint"}
    assert low_battery == {"backend": "cpu", "quantization": "i4", "context_length": 512, "reason": "thermal_or_battery_constraint"}
