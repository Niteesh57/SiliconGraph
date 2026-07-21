# SiliconGraph Implementation Plan

## Product goal

SiliconGraph compiles one HuggingFace/PyTorch model for one verified mobile device profile.

Its output is a tested deployment package: executable model, exact weights, KV-cache/memory plan, profiling evidence, and a safe selector.

`model + exact device profile + quality/context requirement -> tested .armpack`

Version 1 supports one Android Arm64 reference phone and one decoder-only LLM. It may contain Normal, Low-Memory, and Hot modes for that same phone.

## Technology stack

| Area | Technology | Role |
| --- | --- | --- |
| Model import | HuggingFace Transformers and PyTorch | Load checkpoint, tokenizer, configuration, and prompts. |
| Graph capture | `torch.export()` | Capture prefill and decode graphs. |
| Runtime | ExecuTorch `.pte` | Execute the compiled program on Android. |
| CPU backend | XNNPACK | Lower supported subgraphs to mobile CPU operators. |
| Arm kernels | KleidiAI through XNNPACK | Run optimized low-bit matrix multiplication. |
| SiliconGraph | Python orchestration plus C++ core | Profile, choose policies, plan memory, package, and select. |
| Device tooling | Android Kotlin/C++ app and ADB | Probe the phone and run repeatable benchmarks. |

ExecuTorch/XNNPACK/KleidiAI are the runtime foundation. SiliconGraph owns device-specific decisions, evidence, packaging, and fallback behavior.

## System architecture

```text
HuggingFace model -> model metadata -> torch.export -> ExecuTorch graph
                                                |
device profile + calibration + quality rules --> SiliconGraph policy engine
                                                |
                  quantize -> lower -> benchmark on reference device
                                                |
                .armpack containing tested .pte variants

phone capabilities + available RAM + thermal + battery + context request
                                                |
                                   SiliconGraph selector
                                                |
                          matching .pte, weights, and KV plan
```

Each program is inseparable from its packed weights, kernel layout, backend requirements, and memory plan.

## Version 1 scope

- Android `arm64-v8a`, one physical reference device, and one versioned profile.
- One known decoder-only model family such as SmolLM2 or Llama-style.
- Separate prefill and decode entry points.
- CPU baseline through ExecuTorch, XNNPACK, and KleidiAI.
- FP16 reference, INT8 fallback, then supported W4A8/per-block INT4 optimization.
- Static KV-cache context buckets: 512, 2048, and 8192 tokens.
- Safe modes: Normal, Low-Memory, Hot, plus CPU fallback.

Not in version 1: model training, a generic NPU compiler, unverified multi-device support, unsupported custom weight formats, or unsafe mid-chat graph switching.

## Device-profile contract

The Android probe must record ABI, CPU features, Android API level, total/available memory, installed delegates, driver/runtime versions, battery state, and thermal state.

```json
{
  "id": "reference_android_arm64",
  "os": {"min_api": 31, "abi": "arm64-v8a"},
  "cpu": {"features": ["neon", "dotprod", "i8mm"], "threads": 8},
  "memory": {"safe_headroom_mb": 768},
  "gpu": {"api": "vulkan", "available": true},
  "npu": {"delegate": null, "available": false},
  "policy": {"low_battery_pct": 20, "thermal_hot": "severe"}
}
```

The profile also stores measured benchmark results. Processor marketing names alone are never sufficient evidence of compatibility.

## Compilation workflow

1. Load the model, tokenizer, generation configuration, target profile, and requested quality/context limits.
2. Validate that the model architecture is supported; record source hash and tensor shapes.
3. Export prefill and decode graphs using `torch.export()`.
4. Run calibration and quality evaluation on representative prompts.
5. Produce candidate precision, context, KV-cache, thread, and backend policies for the target profile.
6. Lower valid candidates through ExecuTorch and `XnnpackPartitioner` to `.pte` programs.
7. Preserve ExecuTorch CPU fallback for unsupported operators.
8. Benchmark every candidate on the reference phone.
9. Keep only variants passing quality, memory, latency, sustained-thermal, and recovery gates.
10. Build `.armpack` from retained variants plus reports and package metadata.

An export failure blocks release packaging. The current skeletal graph fallback is development-only and must never represent a shipping model.

## Quantization plan

| Variant | Purpose | Order |
| --- | --- | --- |
| FP16 | Correctness and quality reference | First |
| INT8 | Low-risk fallback | Second |
| W4A8 / per-block INT4 weights | Primary optimized CPU variant | Third |
| Mixed INT4/INT8 | Quality-preserving optimized mode | Fourth |
| INT2 | Experimental low-memory mode only | Later |

For every candidate, validate group size, tensor shapes, alignment, weight packing, activation scaling, operator coverage, and CPU feature requirements against the actual runtime kernel contract.

Mixed precision is sensitivity-driven: begin with INT4 where supported, measure quality impact per tensor/layer, and promote the most sensitive tensors to INT8/FP16 until quality passes. “Reasoning layers” are product language, not a reliable compiler rule.

QAT-ready checkpoints are supported by preserving their quantization metadata. SiliconGraph does not perform model training.

## KV-cache and memory plan

Every variant calculates and measures: `weights + KV cache + activations + backend workspace + runtime overhead + safety headroom`.

Implementation order:

1. Static bounded KV arenas for 512, 2048, and 8192 token contexts.
2. Correct grouped-query attention metadata through `num_kv_heads`.
3. Explicit cache precision per variant: FP16 first, then evaluated INT8.
4. Paged KV allocation for long sessions.
5. Sliding-window/eviction for requests beyond the chosen context bucket.

Available memory is a hard admission constraint. A graph is rejected if its measured peak memory plus headroom exceeds available RAM.

## Selector and fallback

1. Reject incompatible ABI, CPU features, OS, delegate, and driver versions.
2. Reject variants over the available-memory budget.
3. Reject variants invalid for battery/thermal policy.
4. Rank remaining variants using measured quality, latency, and energy evidence.
5. Load the matching `.pte`, weights, and KV/memory plan.
6. On failure, log the reason and use the next safe fallback, ending with CPU.

Use hysteresis for heat/low-memory changes. Version 1 changes variants between requests; only cache-compatible variants may change during a conversation.

## Profiling system

The Android benchmark harness must warm up, run fixed prefill/decode workloads, collect JSON evidence, and tag every result with package/profile/device/runtime revisions.

Required metrics:

- TTFT and prefill tokens/second for 32, 128, 512, and 2048-token prompts.
- Decode tokens/second and p50/p95 token latency.
- Peak resident memory, KV-cache size, and backend workspace.
- Quality relative to FP16 using fixed prompts, perplexity, and task checks.
- Sustained thermal behavior, battery/power where exposed, and fallback events.

Release gates are profile/model specific: allowed quality regression, memory headroom, target latency, no crash under stress, and verified fallback if the preferred backend is unavailable.

## Package format

```text
model-device.armpack
├── manifest.json
├── device_profile.json
├── selector_policy.json
├── variants/normal_w4a8/model.pte
├── variants/normal_w4a8/memory_plan.json
├── variants/normal_w4a8/kv_cache_plan.json
├── variants/normal_w4a8/quantization_report.json
├── variants/normal_w4a8/benchmark_summary.json
├── variants/low_memory_int8/
├── tokenizer/
├── runtime_config.json
└── reports/validation_report.json
```

The `.pte` owns its executable representation and packed weights. SiliconGraph metadata selects it; it does not repack or modify it on the phone.

## Workstreams

| Workstream | Deliverables |
| --- | --- |
| Exporter | ExecuTorch export/lowering driver replacing release JSON handoff. |
| Quantization | Calibration, policy search, quality report, INT8/W4A8 variants. |
| C++ core | Profile parser, memory/KV planner, package reader/writer, selector. |
| Android app | Capability probe, package loader, ExecuTorch host, telemetry. |
| Benchmarking | ADB automation, workload suite, JSON reports, regressions. |
| Testing | Unit, integration, package compatibility, hardware-in-the-loop tests. |
| Accelerators | One validated delegate only after CPU baseline succeeds. |

## Delivery phases

### Phase 0 — Reproducible foundation

Pin PyTorch, ExecuTorch, Android NDK, and toolchain versions. Create the profile schema and Android probe.

Exit: the reference device produces a versioned, verified profile JSON.

### Phase 1 — FP16 end-to-end proof

Export one model to `.pte`, build Android ExecuTorch/XNNPACK runtime, run prefill/decode, and package the program/tokenizer/profile/report.

Exit: one `.armpack` reliably generates correct text on the reference phone.

### Phase 2 — Evidence and package integrity

Build benchmark automation, real package verification, real tokenizer/runtime assets, and C++ inspection of `.armpack`.

Exit: every package has reproducible correctness, memory, and performance evidence.

### Phase 3 — Quantized CPU optimization

Add calibration, INT8 fallback, W4A8 variant, KleidiAI/XNNPACK delegated-coverage checks, and FP16 quality comparisons.

Exit: the selected CPU variant is measurably faster/smaller than FP16 and passes quality gates.

### Phase 4 — State-aware operation

Add per-variant memory/KV plans, live RAM/battery/thermal probes, selector constraints, hysteresis, and fallback stress tests.

Exit: low-memory and hot-device tests trigger controlled modes instead of crashes.

### Phase 5 — Mixed precision and paged KV

Implement sensitivity promotion, evaluated INT8 KV cache, paged cache, and Pareto pruning of redundant variants.

Exit: every shipped variant represents a measured quality/performance trade-off.

### Phase 6 — One accelerator backend

Integrate one device-specific GPU or NPU delegate with partitioning, transfer-cost accounting, profile checks, and CPU fallback.

Exit: it improves a defined workload without reducing reliability.

### Phase 7 — More devices

Repeat the complete validation pipeline for each new device profile, then enable multi-device packages.

## Main risks and controls

| Risk | Control |
| --- | --- |
| Unsupported export/lowering | Start with one known model and preserve CPU fallback. |
| Quality loss | Use calibration, evaluation gates, and sensitivity promotion. |
| OOM on long chats | Use hard admission checks, context buckets, headroom, and paged KV. |
| Thermal throttling | Measure sustained runs and ship a hot mode with hysteresis. |
| Fragile NPU path | Keep CPU baseline; add one delegate at a time. |
| Package explosion | Pareto prune and avoid full Cartesian products. |

## Definition of done for the first release

A user selects the supported model and reference phone profile, runs one compile command, installs the package, and gets generated text with a published quality/performance report. The package has a verified CPU fallback, bounded memory behavior, and safe low-memory/hot-device handling.

---

## In simple words: what we are achieving

Normally, an AI model is put on a phone and expected to run the same way everywhere. SiliconGraph prepares the model specifically for one phone.

It knows that phone’s processor, memory, heat, battery, and available accelerators. It organizes the model so the phone can answer quickly, use less RAM, keep important parts accurate, and use a safer lighter mode when the phone gets hot or low on memory.

In short: **we are not merely putting an AI model on a phone; we are preparing it to run efficiently, safely, and intelligently on one particular phone.**
