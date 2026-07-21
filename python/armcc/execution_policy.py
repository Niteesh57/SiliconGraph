"""Open CPU/Vulkan execution policy for Android Arm64 packages.

The policy deliberately has no NPU backend. Vulkan is an optional candidate and
may only be selected after an on-device end-to-end benchmark beats the CPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


LOW_BATTERY_PCT = 20
HOT_THERMAL_STATUS = 2
HOT_CPU_C = 55.0


def _context_buckets(values: Sequence[int] | str) -> list[int]:
    if isinstance(values, str):
        values = values.split(",")
    buckets = sorted({int(value) for value in values})
    if not buckets or buckets[0] <= 0:
        raise ValueError("context lengths must contain positive integers")
    return buckets


def build_execution_policy(
    profile: Mapping[str, Any], context_lengths: Sequence[int] | str,
) -> dict[str, Any]:
    """Create a portable, runtime-selectable CPU/Vulkan policy."""
    contexts = _context_buckets(context_lengths)
    gpu = profile.get("gpu") or {}
    supports_vulkan = bool(gpu.get("present") and gpu.get("supports_vulkan"))
    return {
        "schema_version": "1.0",
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
            "quantization_preference": ["i8", "i4"],
            "max_context_length": contexts[-1],
        },
        "constrained": {
            "when": {
                "battery_pct_at_or_below": LOW_BATTERY_PCT,
                "thermal_status_at_or_above": HOT_THERMAL_STATUS,
                "cpu_temperature_c_at_or_above": HOT_CPU_C,
            },
            "backend": "cpu",
            "quantization_preference": ["i8", "i4"],
            "max_context_length": contexts[0],
            "kv_cache": "smallest_context_bucket",
        },
        "context_buckets": contexts,
    }


def select_execution_plan(
    policy: Mapping[str, Any],
    measurements: Mapping[str, Any] | None,
    runtime_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve the backend, precision and KV-cache bucket at runtime.

    A missing or tied benchmark always selects CPU. This makes CPU the safe
    baseline and prevents an unbenchmarked GPU from being used by assumption.
    """
    measurements = measurements or {}
    state = runtime_state or {}
    constrained = (
        _number_at_or_below(state.get("battery_pct"), LOW_BATTERY_PCT)
        or _number_at_or_above(state.get("thermal_status"), HOT_THERMAL_STATUS)
        or _number_at_or_above(state.get("cpu_temperature_c"), HOT_CPU_C)
    )
    contexts = policy["context_buckets"]
    if constrained:
        return {
            "backend": "cpu",
            "quantization": "i4" if _number_at_or_below(state.get("battery_pct"), LOW_BATTERY_PCT) else "i8",
            "context_length": contexts[0],
            "reason": "thermal_or_battery_constraint",
        }

    vulkan_enabled = bool(policy.get("backends", {}).get("vulkan", {}).get("enabled"))
    cpu_ms = _positive_number(measurements.get("cpu_decode_ms"))
    vulkan_ms = _positive_number(measurements.get("vulkan_decode_ms"))
    backend = "vulkan" if vulkan_enabled and cpu_ms and vulkan_ms and vulkan_ms < cpu_ms else "cpu"
    return {
        "backend": backend,
        "quantization": "i8",
        "context_length": contexts[-1],
        "reason": "vulkan_benchmark_winner" if backend == "vulkan" else "cpu_fallback_or_benchmark_winner",
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


def _number_at_or_below(value: Any, limit: float) -> bool:
    number = _positive_number(value)
    return number is not None and number <= limit


def _number_at_or_above(value: Any, limit: float) -> bool:
    number = _positive_number(value)
    return number is not None and number >= limit
