# Device-Aware Compilation Roadmap

## Product definition

SiliconGraph should compile one source model into a **device-aware graph family**,
not merely convert it into one portable model file. Every family member is a
complete deployment plan for a declared operating envelope:

```text
model + device profile + runtime state + quality target
                    -> selected graph, memory plan, kernels, and weights
```

The runtime selects a compatible graph using live RAM, thermal, battery, and
accelerator state. Model transformation is offline; runtime selection is cheap.

## Target architecture

```text
HuggingFace model -> canonical ARM IR -> analysis + calibration
      -> generate candidates -> benchmark + Pareto prune -> .armpack family
                                                         |
device probe + live RAM / battery / thermal state -> constrained selector
                                                         |
                                           graph + exact weights + kernels
```

The essential invariant: a graph must be packaged with the exact quantized
weights, kernel ABI, execution placement, and memory map it requires. Never
combine a graph from one variant with another variant's weights or layout.

## Graph-key dimensions

| Dimension | Examples | Purpose |
| --- | --- | --- |
| Hardware | CPU features, GPU API, NPU delegate/version | Legal kernels and layouts differ. |
| Precision | FP16, INT8, W4A8, INT4, future INT2 | Controls quality, bandwidth, and memory. |
| Context | 512, 2048, 8192 | Controls KV-cache and attention plan. |
| Available-memory class | 512 MB, 1 GB, 2 GB | Prevents unsafe selection. |
| Thermal/power | nominal, warm, hot; low battery | Models sustainable performance. |
| Latency intent | interactive, batch, power saver | Makes trade-offs explicit. |

Generate broad candidates only initially. Retain the measured Pareto frontier:
variants for which no other compatible option is both faster, smaller, and at
least as accurate.

## Quantization strategy

Start with W4A8 per-block quantization: packed INT4 weights grouped by output
channel/block, dynamic INT8 activations per row/token, and scale/zero-point
metadata stored in the exact layout expected by the chosen kernel. Store group
size per tensor rather than globally.

Mixed precision must be sensitivity-driven, not based on fixed names such as
"reasoning layers" and "token-generation layers":

1. Measure tensor/layer sensitivity using calibration and evaluation data.
2. Keep sensitive tensors in INT8 or FP16.
3. Quantize tolerant linear weights to INT4.
4. Gate every policy on quality, latency, and memory measurements.

Treat INT2 as an experimental low-memory tier only when quality passes a gate,
a real target kernel supports its packing, and metadata does not erase its
memory advantage. Always include INT4/INT8 fallback variants. Accept QAT-ready
models by preserving their learned quantization parameters; training itself
does not need to be part of this compiler.

## KV-cache strategy

Plan KV cache per graph variant because layer count, KV-head count, head size,
context limit, precision, and paging all change the allocation. Deliver this in
order:

1. Bounded static KV arenas for each context bucket.
2. Correct grouped-query-attention metadata (`num_kv_heads`).
3. Paged KV allocation for long sessions.
4. FP16 and INT8 cache variants, with lower precision only after quality tests.
5. Sliding-window/eviction policy for requests beyond the graph's context cap.

The selector must reject a graph when available RAM is below its peak-memory
requirement plus a safety headroom; a memory budget is a hard constraint.

## Kernel and placement strategy

- Keep a portable CPU baseline with NEON, then runtime-dispatch dot-product,
  i8mm, BF16, SVE2, and SME/SME2 kernels only when the CPU advertises them.
- Adopt KleidiAI-compatible low-bit matrix layouts first; integration through
  XNNPACK or ExecuTorch is a practical early runtime path.
- Define GPU/NPU backend contracts for supported ops, data layouts, workspace,
  latency/cost estimates, and fallback behavior.
- Partition contiguous regions and include transfer costs. Per-op offloading can
  lose to CPU execution when device-copy overhead dominates.

Use function multiversioning for CPU ISA kernels, and use graph-family selection
for larger hardware, memory, and thermal decisions.

## Runtime selection policy

1. Probe immutable capabilities: SoC/CPU features, GPU API, NPU delegate and
   driver version, total RAM.
2. Read mutable state: available RAM, thermal state, battery, charging, and
   requested context/latency mode.
3. Filter incompatible or unsafe variants.
4. Choose the lowest predicted latency (or energy) among the remaining quality
   tier.
5. Fall back after execution failure or material state change, ending with a
   portable CPU graph.

Apply hysteresis so small temperature/battery fluctuations do not trigger graph
switches. Preserve KV cache only between compatible variants; otherwise rebuild
it safely.

## Delivery order for this codebase

### 1. Make packages executable and consistent

- Pass real weights, calibration, tokenizer assets, and runtime config from
  Python to C++.
- Replace weight, kernel, and selector placeholders with usable artifacts.
- Create memory plans per compiled graph, and reference them per graph entry.
- Make C++ inspection and IR-dump read `.armpack` files.

### 2. Establish the CPU reference path

- Implement W4A8 per-block quantization and sensitivity-driven mixed precision.
- Add kernel-layout descriptors to ARM IR and package metadata.
- Integrate a tested Arm CPU low-bit kernel path and feature-dispatch table.
- Measure quality, TTFT, decode tokens/s, peak RSS, and energy on real devices.

### 3. Make state and memory first class

- Implement paged/static KV variants, RAM admission checks, selector
  constraints, hysteresis, and safe fallback.
- Cap family size through benchmark-based Pareto pruning.

### 4. Add accelerator backends

- Add backend-specific partitions and delegate/kernel artifacts.
- Validate transfer cost, thermal behavior, fallback, and driver compatibility
  on each claimed device profile.

## Release criteria

Publish quality relative to FP16/BF16, TTFT/prefill throughput, decode tokens/s,
peak memory including KV/workspace, energy per token, sustained thermal behavior,
and fallback behavior for every released device profile. This turns the promise
into a measurable one: a graph is proven runnable for a stated device and
operating envelope, not just labelled "optimized for Arm".
