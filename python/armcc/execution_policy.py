"""Open CPU/Vulkan execution policy for Android Arm64 packages.

The policy deliberately has no NPU backend. Vulkan is an optional candidate and
may only be selected after an on-device end-to-end benchmark beats the CPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from armcc.modalities import ModalityCapabilities


DIRECT_ANSWER_BATTERY_PCT = 50
HOT_THERMAL_STATUS = 2
HOT_CPU_C = 55.0
MIN_SYSTEM_RESERVE_MB = 512
EMERGENCY_MEMORY_RESERVE_MB = 256
FULL_MAX_NEW_TOKENS = 512
DIRECT_MAX_NEW_TOKENS = 128
MEDIA_MIN_SAFE_RAM_MB = 768
DEFAULT_IMAGE_WORKING_SET_MB = 96
DEFAULT_AUDIO_WORKING_SET_MB = 64
DEFAULT_VIDEO_WORKING_SET_MB = 256


def _precision_tier(quantization: str | None) -> str:
    """Map a package precision policy to an input-admission quality tier.

    Media encoders are much more sensitive to aggressive precision reduction
    than text tokenization.  INT4 and FP8 are therefore intentionally a
    text-only runtime path.  A ``mixed`` graph family still contains an INT8
    candidate, which the native selector requires whenever media is present.
    """
    value = str(quantization or "mixed").lower()
    if value in {"fp32", "fp16"}:
        return "full"
    if value in {"int8", "mixed"}:
        return "medium"
    return "low"


def _input_admission_policy(
    capabilities: ModalityCapabilities | Mapping[str, Any] | None,
    quantization: str | None,
) -> dict[str, Any]:
    if isinstance(capabilities, ModalityCapabilities):
        supported_inputs = list(capabilities.inputs)
    elif isinstance(capabilities, Mapping):
        supported_inputs = [str(value) for value in capabilities.get("inputs", ["text"])]
    else:
        supported_inputs = ["text"]

    media_inputs = [
        modality for modality in supported_inputs
        if modality in {"image", "audio", "video"}
    ]
    return {
        "schema_version": 1,
        "supported_inputs": supported_inputs,
        "media_inputs": media_inputs,
        "media_requires_precision_tiers": ["full", "medium"],
        "low_precision_behavior": "accept_text_only",
        "package_precision_tier": _precision_tier(quantization),
        "minimum_safe_ram_mb_for_media": MEDIA_MIN_SAFE_RAM_MB,
        "default_media_working_set_mb": {
            "image": DEFAULT_IMAGE_WORKING_SET_MB,
            "audio": DEFAULT_AUDIO_WORKING_SET_MB,
            "video": DEFAULT_VIDEO_WORKING_SET_MB,
        },
        "preprocessing": {
            "strategy": "preprocess_independent_inputs_concurrently_then_fuse",
            "parallelizable_inputs": ["text", "image", "audio", "video"],
            "synchronization": "all_requested_input_tensors_ready_before_fused_forward",
        },
    }


def _context_buckets(values: Sequence[int] | str) -> list[int]:
    if isinstance(values, str):
        values = values.split(",")
    buckets = sorted({int(value) for value in values})
    if not buckets or buckets[0] <= 0:
        raise ValueError("context lengths must contain positive integers")
    return buckets


def build_execution_policy(
    profile: Mapping[str, Any], context_lengths: Sequence[int] | str,
    capabilities: ModalityCapabilities | Mapping[str, Any] | None = None,
    quantization: str | None = None,
) -> dict[str, Any]:
    """Create a capacity-aware CPU/Vulkan policy for one Android device.

    The installed RAM determines the graph family compiled into the package.
    Current free RAM determines which member can run at inference time. Battery
    and thermal state never lower numerical precision; below 50% they only
    switch the generation budget to concise/direct answers.
    """
    contexts = _context_buckets(context_lengths)
    gpu = profile.get("gpu") or {}
    runtime = profile.get("runtime_state") or {}
    device_total_mb = _positive_int(runtime.get("memory_total_mb"))
    reserve_mb = _compile_reserve_mb(device_total_mb)
    graph_budgets = _graph_memory_budgets(device_total_mb, reserve_mb)
    supports_vulkan = bool(gpu.get("present") and gpu.get("supports_vulkan"))
    return {
        "schema_version": "1.1",
        "device_capacity": {
            "total_ram_mb": device_total_mb,
            "compile_reserve_mb": reserve_mb,
            "graph_memory_budgets_mb": graph_budgets,
        },
        "memory": {
            "always_enforced": True,
            "min_system_reserve_mb": MIN_SYSTEM_RESERVE_MB,
            "low_memory_extra_reserve_mb": EMERGENCY_MEMORY_RESERVE_MB,
            "allocation": "load_only_selected_variant_and_use_paged_kv_cache",
            "context_selection": "largest_bucket_that_fits_current_safe_ram",
            "on_insufficient_memory": "refuse_before_allocation",
        },
        "input_admission": _input_admission_policy(capabilities, quantization),
        "backends": {
            "cpu": {"enabled": True, "required": True},
            "vulkan": {
                "enabled": supports_vulkan,
                "required": False,
                "selection": "only_if_end_to_end_latency_beats_cpu",
            },
        },
        "benchmark": {
            "metric": "median_end_to_end_decode_ms",
            "warmup_runs": 3,
            "measured_runs": 10,
            "cpu_backend": "cpu",
            "vulkan_backend": "vulkan",
        },
        "normal": {
            "backend": "benchmark_winner",
            "precision": "quality_validated_static_variant",
            "max_context_length": contexts[-1],
            "response_mode": "full",
            "max_new_tokens": FULL_MAX_NEW_TOKENS,
        },
        "battery_aware": {
            "when": {
                "battery_pct_below": DIRECT_ANSWER_BATTERY_PCT,
                "thermal_applies_only_below_battery_pct": DIRECT_ANSWER_BATTERY_PCT,
            },
            "precision": "unchanged_quality_validated_variant",
            "response_mode": "direct",
            "max_new_tokens": DIRECT_MAX_NEW_TOKENS,
            "instruction": "Answer directly and concisely. Do not produce a long explanation unless explicitly requested.",
            "thermal_backend": "prefer_cpu_only_when_hot",
        },
        "context_buckets": contexts,
    }


def select_execution_plan(
    policy: Mapping[str, Any],
    measurements: Mapping[str, Any] | None,
    runtime_state: Mapping[str, Any] | None,
    requested_inputs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Resolve a memory-safe graph/runtime plan from live device state."""
    measurements = measurements or {}
    state = runtime_state or {}
    contexts = policy["context_buckets"]
    memory = policy.get("memory", {})
    safe_memory_mb = _safe_memory_mb(state, memory)
    admission = _admit_inputs(policy, state, requested_inputs, safe_memory_mb)
    if not admission["accepted"]:
        return {
            "can_run": False,
            "backend": "cpu",
            "context_length": 0,
            "safe_memory_mb": safe_memory_mb,
            "reason": admission["reason"],
            "input_admission": admission,
        }
    context_length, kv_cache_budget_mb = _context_that_fits(
        contexts, safe_memory_mb, state, admission["media_working_set_mb"],
    )
    if context_length is None:
        return {
            "can_run": False,
            "backend": "cpu",
            "context_length": 0,
            "safe_memory_mb": safe_memory_mb,
            "reason": "insufficient_current_free_ram",
            "input_admission": admission,
        }

    vulkan_enabled = bool(policy.get("backends", {}).get("vulkan", {}).get("enabled"))
    cpu_ms = _positive_number(measurements.get("cpu_decode_ms"))
    vulkan_ms = _positive_number(measurements.get("vulkan_decode_ms"))
    battery_pct = _nonnegative_number(state.get("battery_pct"))
    below_fifty = battery_pct is not None and battery_pct < DIRECT_ANSWER_BATTERY_PCT
    hot = (
        _number_at_or_above(state.get("thermal_status"), HOT_THERMAL_STATUS)
        or _number_at_or_above(state.get("cpu_temperature_c"), HOT_CPU_C)
    )
    gpu_memory_ok = not state.get("vulkan_memory_ok") is False
    backend = "vulkan" if vulkan_enabled and gpu_memory_ok and cpu_ms and vulkan_ms and vulkan_ms < cpu_ms else "cpu"
    if below_fifty and hot:
        backend = "cpu"
    return {
        "can_run": True,
        "backend": backend,
        "precision": "quality_validated_static_variant",
        "context_length": context_length,
        "kv_cache_budget_mb": kv_cache_budget_mb,
        "safe_memory_mb": safe_memory_mb,
        "prefill_chunk_tokens": 256 if context_length == contexts[-1] else 64,
        "response_mode": "direct" if below_fifty else "full",
        "max_new_tokens": DIRECT_MAX_NEW_TOKENS if below_fifty else FULL_MAX_NEW_TOKENS,
        "input_admission": admission,
        "reason": (
            "battery_below_50_and_thermal_constraint" if below_fifty and hot
            else "battery_below_50_direct_answer" if below_fifty
            else "vulkan_benchmark_winner" if backend == "vulkan"
            else "cpu_fallback_or_benchmark_winner"
        ),
    }


def write_execution_policy(path: str | Path, policy: Mapping[str, Any]) -> str:
    output = Path(path)
    output.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    return str(output)


def _positive_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _positive_int(value: Any) -> int:
    number = _positive_number(value)
    return int(number) if number is not None else 0


def _compile_reserve_mb(device_total_mb: int) -> int:
    if device_total_mb <= 0:
        return MIN_SYSTEM_RESERVE_MB
    return min(1024, max(MIN_SYSTEM_RESERVE_MB, (device_total_mb * 15 + 99) // 100))


def _graph_memory_budgets(device_total_mb: int, reserve_mb: int) -> list[int]:
    if device_total_mb <= 0:
        return [512, 1024, 2048, 4096]
    ceiling = max(512, device_total_mb - reserve_mb)
    budgets = [budget for budget in (512, 1024, 2048, 4096, 6144, 8192) if budget <= ceiling]
    rounded_ceiling = ceiling // 512 * 512
    if rounded_ceiling >= 512 and rounded_ceiling not in budgets:
        budgets.append(rounded_ceiling)
    return sorted(set(budgets))


def _safe_memory_mb(state: Mapping[str, Any], memory: Mapping[str, Any]) -> int:
    available_mb = _positive_int(state.get("available_ram_mb"))
    if available_mb == 0:
        available_mb = _positive_int(state.get("memory_available_mb"))
    threshold_mb = _positive_int(state.get("memory_threshold_mb"))
    reserve_mb = max(_positive_int(memory.get("min_system_reserve_mb")), threshold_mb)
    if state.get("low_memory"):
        reserve_mb += _positive_int(memory.get("low_memory_extra_reserve_mb"))
    return max(0, available_mb - reserve_mb)


def _context_that_fits(
    contexts: Sequence[int], safe_memory_mb: int, state: Mapping[str, Any],
    media_working_set_mb: int = 0,
) -> tuple[int | None, int]:
    fixed_mb = (
        _positive_int(state.get("selected_weight_mb"))
        + _positive_int(state.get("activation_mb"))
        + _positive_int(state.get("scratch_mb"))
        + media_working_set_mb
    )
    kv_budget_mb = max(0, safe_memory_mb - fixed_mb)
    kv_bytes_per_token = _positive_number(state.get("kv_bytes_per_token"))
    if kv_bytes_per_token is None:
        return contexts[-1], kv_budget_mb
    max_context = int(kv_budget_mb * 1024 * 1024 // kv_bytes_per_token)
    fitting = [context for context in contexts if context <= max_context]
    return (fitting[-1] if fitting else None), kv_budget_mb


def _admit_inputs(
    policy: Mapping[str, Any], state: Mapping[str, Any],
    requested_inputs: Sequence[str] | None, safe_memory_mb: int,
) -> dict[str, Any]:
    """Validate a request before allocating image/audio/video buffers.

    This is intentionally a preflight gate.  The Android runtime should run it
    after inspecting media metadata but before decoding full-resolution images
    or long audio streams into memory.
    """
    config = policy.get("input_admission", {})
    requested = list(dict.fromkeys(str(value).lower() for value in (requested_inputs or ("text",))))
    supported = {str(value).lower() for value in config.get("supported_inputs", ["text"])}
    unsupported = [value for value in requested if value not in supported]
    media = [value for value in requested if value in {"image", "audio", "video"}]
    defaults = config.get("default_media_working_set_mb", {})
    media_working_set_mb = _positive_int(state.get("media_working_set_mb"))
    if media_working_set_mb == 0:
        media_working_set_mb = sum(_positive_int(defaults.get(value)) for value in media)

    decision: dict[str, Any] = {
        "accepted": False,
        "requested_inputs": requested,
        "media_inputs": media,
        "media_working_set_mb": media_working_set_mb,
        "parallel_preprocessing": "image" in media and "audio" in media,
        "preprocessing_strategy": config.get("preprocessing", {}).get(
            "strategy", "preprocess_independent_inputs_concurrently_then_fuse",
        ),
    }
    if unsupported:
        decision["reason"] = "model_does_not_support_" + "_and_".join(unsupported)
        return decision
    if not media:
        decision.update(accepted=True, reason="text_only_input")
        return decision

    tier = str(state.get("selected_precision_tier") or config.get("package_precision_tier", "low"))
    allowed_tiers = set(config.get("media_requires_precision_tiers", ["full", "medium"]))
    if tier not in allowed_tiers:
        decision["reason"] = "media_requires_full_or_medium_precision"
        return decision
    min_safe_ram_mb = _positive_int(config.get("minimum_safe_ram_mb_for_media"))
    if safe_memory_mb < min_safe_ram_mb or safe_memory_mb < media_working_set_mb:
        decision["reason"] = "insufficient_current_free_ram_for_media"
        return decision
    decision.update(accepted=True, reason="media_input_admitted")
    return decision


def _number_at_or_below(value: Any, limit: float) -> bool:
    number = _nonnegative_number(value)
    return number is not None and number <= limit


def _number_at_or_above(value: Any, limit: float) -> bool:
    number = _nonnegative_number(value)
    return number is not None and number >= limit


def _nonnegative_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None
