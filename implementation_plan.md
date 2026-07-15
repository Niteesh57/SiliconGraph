# ARM AI Compiler — Universal AI Model Optimizer

## Vision

Build the **LLVM of AI deployment**: a universal AI compiler that ingests open-source models from Hugging Face and emits a portable, multi-graph optimized package targeting every ARM hardware profile — from smartwatches to Snapdragon flagship SoCs — without requiring any additional on-device optimization.

Existing tools (llama.cpp, ONNX Runtime, Cactus) solve ~30-40% of the problem. They compile one graph, optimize kernels, and assume a fixed device profile. This framework targets the full pipeline.

---

## Architecture Overview

```
HF Model (PyTorch / Safetensors)
         ↓
  [Model Ingestion Layer]       ← Python
         ↓
  [ARM IR Generator]            ← C++
         ↓
  [Analysis & Cost Model]       ← C++
         ↓
  [Optimization Pass Pipeline]  ← C++  (MLIR-based)
         ↓
  [Graph Family Generator]      ← C++
         ↓
  [Kernel Generator]            ← C++ + LLVM backends
         ↓
  [Memory Planner]              ← C++
         ↓
  [Package Generator]           ← C++
         ↓
  [Adaptive Runtime]            ← C++ (future phase)
```

---

## Phase 1 — Core Compiler (This Implementation)

### Stage 1 — Model Ingestion (Python Layer)

**Purpose**: Bridge between HuggingFace ecosystem and the C++ compiler core.

- `model_loader.py` — loads any HF model (Llama, Qwen, Gemma, Phi, SmolLM) via `transformers` + `safetensors`
- `graph_exporter.py` — uses `torch.export` / `torch.fx` to extract a clean computation graph
- `calibration.py` — runs calibration datasets (for INT8/INT4 quantization statistics)
- `hf_metadata.py` — extracts tokenizer, config, generation params, model card metadata
- Python CLI entry point: `armcc compile --model <hf_id> --targets <profiles>`

### Stage 2 — ARM Intermediate Representation (ARM-IR)

A custom IR designed specifically for AI deployment, richer than ONNX, built on top of MLIR dialects.

**ARM-IR node metadata includes**:
| Field | Purpose |
|---|---|
| `tensor_lifetime` | When tensor is live (for memory planning) |
| `memory_affinity` | Preferred memory type (SRAM, DRAM, shared) |
| `preferred_exec_unit` | CPU / GPU / NPU / DSP hint |
| `quant_metadata` | Scale, zero-point, calibration stats |
| `cache_policy` | prefetch, evict, pin hints |
| `thermal_cost` | Estimated thermal budget |
| `scheduling_hint` | parallelism, latency sensitivity |
| `streaming_support` | Can this op be chunked/streamed? |
| `kv_cache_affinity` | KV cache layout preferences |

**Files**:
- `ir/arm_ir.h` — IR node, graph, and tensor definitions
- `ir/arm_ir_dialect.td` — MLIR TableGen dialect definition
- `ir/ir_builder.cpp` — constructs ARM-IR from the exported torch graph
- `ir/ir_printer.cpp` — human-readable IR dump (like LLVM IR text format)
- `ir/ir_verifier.cpp` — validates IR correctness

### Stage 3 — Analysis & Cost Model

The **Cost Model** is the biggest missing piece in current AI compilers. Every operator gets profiled per execution unit.

**Cost Model structure**:
```
MatMul [M=512, K=4096, N=4096]
  ├── Cortex-A76 CPU    →  14.2ms, 2.1 GFLOPS/W
  ├── Adreno 750 GPU    →   3.8ms, 7.4 GFLOPS/W
  ├── Hexagon NPU       →   1.1ms, 18.2 GFLOPS/W
  ├── MediaTek APU      →   1.9ms, 12.1 GFLOPS/W
  └── Apple ANE         →   0.7ms, 31.4 GFLOPS/W
```

**Files**:
- `analysis/cost_model.h / .cpp` — operator cost tables, lookup, interpolation
- `analysis/graph_analyzer.cpp` — analyzes op shapes, data flow, memory pressure
- `analysis/profiling_db.json` — pre-measured latency/power profiles per SoC
- `analysis/device_profiles/` — one JSON per target (snapdragon_8_gen3.json, mediatek_dimensity_9300.json, etc.)

### Stage 4 — Optimization Pass Pipeline

Inspired by TVM, XLA, TensorRT. Each pass is independent and composable.

**Passes (in order)**:

#### Graph-Level Passes
| Pass | Description |
|---|---|
| `ConstantFoldingPass` | Pre-compute static subgraphs |
| `DeadCodeEliminationPass` | Remove unused ops |
| `OperatorFusionPass` | Fuse adjacent ops (MatMul+Add+ReLU → 1 kernel) |
| `HardwareFusionPass` | **Novel**: fuse ops across hardware boundaries (Attn→NPU, LayerNorm→CPU, MatMul→GPU) |
| `ShapeInferencePass` | Propagate tensor shapes throughout graph |
| `LayoutOptimizationPass` | NCHW ↔ NHWC ↔ custom layout per target |
| `KVCachePlanningPass` | Plan KV cache size, layout, eviction policy |
| `StreamingAttentionPass` | Rewrite attention for streaming/chunked inference |

#### Quantization Passes
| Pass | Description |
|---|---|
| `INT8QuantizationPass` | Post-training INT8 with calibration |
| `INT4QuantizationPass` | Weight-only INT4 (GPTQ-style groupwise) |
| `FP8QuantizationPass` | FP8 for targets that support it (Apple Silicon) |
| `MixedPrecisionPass` | Assign different precision per layer based on sensitivity |
| `QuantizationAwarePass` | Fine-tunes quant params using calibration data |

#### Memory Passes
| Pass | Description |
|---|---|
| `TensorLifetimeAnalysisPass` | Compute live ranges for every tensor |
| `MemoryAliasingPass` | Allow tensors to share memory when safe |
| `MemoryLayoutPass` | Optimize tensor layout for cache lines, NUMA, DMA |
| `PrefetchInsertionPass` | Insert prefetch hints before compute-heavy ops |
| `ZeroCopyPass` | Eliminate copies where input/output memory can overlap |

#### Kernel-Level Passes
| Pass | Description |
|---|---|
| `KernelSelectionPass` | Select best kernel variant per op per target |
| `TilingPass` | Choose tile sizes for GEMM/Conv per cache hierarchy |
| `VectorizationPass` | Map to NEON/SVE/AMX SIMD intrinsics |
| `LoopUnrollingPass` | Unroll inner loops based on register count |

**Files**:
- `passes/pass_manager.h / .cpp` — pipeline orchestration
- `passes/graph/` — one file per graph-level pass
- `passes/quantization/` — quantization passes
- `passes/memory/` — memory optimization passes
- `passes/kernel/` — kernel-level passes

### Stage 5 — Graph Family Generator

Instead of one optimized graph, the compiler emits a **Graph Family** — multiple optimized graphs for different device conditions.

**Graph dimensions**:
- **Hardware profile**: CPU-only, GPU, NPU, DSP, hybrid
- **Precision**: FP32, FP16, INT8, INT4, mixed
- **Memory budget**: 512MB, 1GB, 2GB, 4GB+
- **Context length**: short (512), medium (2048), long (8192+)
- **Latency target**: interactive (<100ms), batch, background
- **Thermal state**: nominal, warm, hot (throttled)

**Example graph family for Llama-3.2-1B**:
```
graphs/
  ├── graph_snapdragon_npu_int4_2gb_interactive.armgraph
  ├── graph_snapdragon_gpu_int8_2gb_interactive.armgraph
  ├── graph_snapdragon_cpu_int8_1gb_background.armgraph
  ├── graph_mediatek_apu_int4_2gb_interactive.armgraph
  ├── graph_apple_ane_fp16_4gb_interactive.armgraph
  ├── graph_generic_arm_cpu_int8_512mb_background.armgraph
  └── graph_generic_arm_cpu_int4_256mb_minimal.armgraph
```

**Files**:
- `generator/graph_family_generator.h / .cpp`
- `generator/graph_selector.h / .cpp` — runtime graph selection logic (embedded into package)
- `generator/condition_matrix.cpp` — maps device conditions → graph index

### Stage 6 — Kernel Generator

Generates hardware-specific kernel code for each target.

**Targets**:

| Target | Backend | Key ops |
|---|---|---|
| Snapdragon Hexagon NPU | HTP delegate / QNN | GEMM, Attention, Conv |
| Adreno GPU | OpenCL / Vulkan compute | GEMM, Softmax, Norm |
| MediaTek APU | NeuroPilot / NNAPI | Attention, GEMM |
| Mali GPU | OpenCL / Vulkan | GEMM, elementwise |
| Apple ANE | CoreML | Attention, GEMM |
| Apple GPU | Metal | GEMM, Softmax |
| Generic ARM CPU | NEON / SVE intrinsics | All ops |
| ARM SVE2 | SVE2 intrinsics | GEMM, Attention |

**Files**:
- `kernels/cpu/` — NEON/SVE kernel templates
- `kernels/gpu/opencl/` — OpenCL kernel sources
- `kernels/gpu/vulkan/` — Vulkan compute shaders (GLSL)
- `kernels/gpu/metal/` — Metal shaders (MSL)
- `kernels/npu/hexagon/` — Hexagon HTP delegate wrappers
- `kernels/npu/mediatek/` — NeuroPilot wrappers
- `kernels/npu/apple_ane/` — CoreML op patterns
- `kernel_generator.cpp` — drives kernel emission per target

### Stage 7 — Memory Planner

Treats memory like a database query optimizer treats a query plan.

- Static memory map for weights (mmap-able, page-aligned)
- Dynamic memory arena for activations (slab allocator)
- KV cache memory plan (ring buffer, paged attention)
- DMA transfer schedule for NPU/DSP workloads
- Zero-copy paths where input/output tensors can alias

**Files**:
- `memory/memory_planner.h / .cpp`
- `memory/tensor_allocator.cpp`
- `memory/kv_cache_layout.cpp`
- `memory/mmap_layout.cpp`

### Stage 8 — Package Generator

Produces the final `.armpack` portable package — a structured archive.

**Package structure**:
```
model.armpack (ZIP-based archive)
├── manifest.json          — package metadata, graph index, version
├── tokenizer/             — tokenizer assets (vocab, merges, config)
├── graphs/                — compiled graph family (binary)
│   ├── graph_000.armgraph
│   └── graph_001.armgraph
├── kernels/               — precompiled kernel objects per target
│   ├── snapdragon_htp/
│   ├── adreno_vulkan/
│   ├── mali_opencl/
│   └── arm_neon/
├── weights/               — quantized weights (mmap layout)
│   ├── weights_int4.bin
│   └── weights_int8.bin
├── memory_maps/           — static memory layout descriptors
├── runtime_config.json    — generation params, sampling config
└── selector.wasm          — graph selector logic (portable, runs on device)
```

**Files**:
- `package/package_generator.h / .cpp`
- `package/armpack_format.h` — binary format spec
- `package/manifest_writer.cpp`

---

## Project File Structure

```
compile-for-arm/
├── CMakeLists.txt
├── README.md
│
├── python/                        # Python ingestion layer
│   ├── armcc/
│   │   ├── __init__.py
│   │   ├── cli.py                 # CLI entry point
│   │   ├── model_loader.py        # HF model loading
│   │   ├── graph_exporter.py      # torch.export → protobuf/flatbuf
│   │   ├── calibration.py         # calibration dataset runner
│   │   └── hf_metadata.py         # tokenizer + config extraction
│   ├── pyproject.toml
│   └── requirements.txt
│
├── compiler/                      # C++ compiler core
│   ├── include/
│   │   ├── arm_ir/
│   │   ├── analysis/
│   │   ├── passes/
│   │   ├── generator/
│   │   ├── kernels/
│   │   ├── memory/
│   │   └── package/
│   │
│   ├── lib/
│   │   ├── arm_ir/
│   │   ├── analysis/
│   │   ├── passes/
│   │   │   ├── graph/
│   │   │   ├── quantization/
│   │   │   ├── memory/
│   │   │   └── kernel/
│   │   ├── generator/
│   │   ├── kernels/
│   │   │   ├── cpu/
│   │   │   ├── gpu/
│   │   │   │   ├── opencl/
│   │   │   │   ├── vulkan/
│   │   │   │   └── metal/
│   │   │   └── npu/
│   │   │       ├── hexagon/
│   │   │       ├── mediatek/
│   │   │       └── apple_ane/
│   │   ├── memory/
│   │   └── package/
│   │
│   └── tools/
│       ├── armcc/                 # C++ CLI driver
│       └── arm-ir-dump/           # IR printer tool
│
├── device_profiles/               # SoC capability descriptors
│   ├── snapdragon_8_gen3.json
│   ├── snapdragon_8_elite.json
│   ├── dimensity_9300.json
│   ├── exynos_2400.json
│   ├── apple_m4.json
│   ├── apple_a18.json
│   └── generic_arm_cpu.json
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
│
└── docs/
    ├── arm_ir_spec.md
    ├── package_format.md
    └── adding_a_target.md
```

---

## Target Hardware Profiles (Phase 1)

| Profile | NPU | GPU | CPU | Special |
|---|---|---|---|---|
| `snapdragon_8_gen3` | Hexagon HTP | Adreno 750 | Cortex-X4 | CDSP |
| `snapdragon_8_elite` | Hexagon HTP | Adreno 830 | Oryon | CDSP |
| `dimensity_9300` | APU 790 | Immortalis-G720 | Cortex-X4 | — |
| `exynos_2400` | NPU | Xclipse 940 | Cortex-X4 | — |
| `apple_m4` | ANE | Apple GPU | Performance+Efficiency | AMX |
| `apple_a18` | ANE | Apple GPU | A18 cores | AMX |
| `generic_arm64` | — | — | Any ARM64 | NEON/SVE |

---

## Quantization Strategy

| Scheme | Target Use | Method |
|---|---|---|
| FP32 | Debug/reference | None |
| FP16 | GPU (Adreno, Metal) | Direct cast |
| INT8 | CPU, NPU | PTQ with calibration |
| INT4 | Memory-constrained | GPTQ groupwise (g128) |
| FP8 (E4M3) | Apple ANE, server | Direct cast + calibration |
| Mixed | Sensitive layers FP16, rest INT4 | Per-layer sensitivity analysis |

---

## CLI Design

```bash
# Compile a model to all default targets
armcc compile \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --targets snapdragon_8_gen3,dimensity_9300,apple_a18,generic_arm64 \
  --quantization mixed \
  --context-lengths 512,2048,8192 \
  --output ./llama-3.2-1b.armpack

# Inspect a compiled package
armcc inspect ./llama-3.2-1b.armpack

# Dump ARM-IR for debugging
arm-ir-dump ./llama-3.2-1b.armpack --graph snapdragon_npu_int4

# Benchmark a graph against a device profile
armcc benchmark ./llama-3.2-1b.armpack --profile snapdragon_8_gen3
```

---

## SDK Design (C API)

```c
// Load package
ArmPackage* pkg = armpack_load("llama-3.2-1b.armpack");

// Describe device state (runtime fills this from sensors)
ArmDeviceState state = {
  .soc_id        = ARMCC_SOC_SNAPDRAGON_8_GEN3,
  .ram_available = 1500,   // MB
  .battery_pct   = 85,
  .thermal_state = ARMCC_THERMAL_NOMINAL,
  .latency_target = ARMCC_LATENCY_INTERACTIVE,
};

// Compiler selects best graph automatically
ArmGraph* graph = armpack_select_graph(pkg, &state);

// Run inference
ArmInferenceResult result = armgraph_run(graph, input_tokens, n_tokens);
```

---

## Build System

- **Build**: CMake + Ninja
- **C++ Standard**: C++20
- **Dependencies**:
  - LLVM/MLIR (compiler backbone, via vcpkg or system install)
  - FlatBuffers (ARM-IR binary serialization)
  - libzip (`.armpack` archive)
  - nlohmann/json (device profiles, manifests)
  - Python 3.11+ with: `transformers`, `torch`, `safetensors`, `datasets`

---

## Open Questions

> [!IMPORTANT]
> **Q1: MLIR vs custom IR?**
> Using MLIR gives access to existing dialects (linalg, tosa, arith) and infrastructure (pass manager, pattern rewriting). A custom IR gives full control. Recommendation: build on MLIR but define a custom `arm_ir` dialect on top.

> [!IMPORTANT]
> **Q2: How to handle NPU delegation (Hexagon HTP, MediaTek NeuroPilot)?**
> These NPUs require vendor SDKs (Qualcomm AI Engine Direct SDK, NeuroPilot SDK). The compiler should generate the delegate call boundaries and serialized subgraph for the NPU, but the actual NPU execution happens through the vendor runtime. This requires those SDKs to be present during compilation.

> [!IMPORTANT]
> **Q3: Should Phase 1 include the Adaptive Runtime or only the compiler?**
> The runtime (graph selector, on-device dispatch) is architecturally separate. Phase 1 should implement the full compiler pipeline and the `selector.wasm` embedded in the package. A thin C reference runtime can be included for integration testing, with a full adaptive runtime as Phase 2.

> [!WARNING]
> **Q4: `.armpack` format licensing**
> If this is intended to be open-source, we need to define the package format spec clearly and publish it independently so third-party runtimes (including llama.cpp, Cactus) can consume `.armpack` files — making this truly the LLVM of AI deployment.

---

## Implementation Order (Phase 1)

1. **Week 1**: Project scaffolding, CMake, Python ingestion layer, `torch.export` → proto/flatbuf graph dump
2. **Week 2**: ARM-IR definition (MLIR dialect), IR builder from exported graph, IR printer
3. **Week 3**: Cost model + device profile database, graph analyzer
4. **Week 4**: Core optimization passes (constant folding, DCE, operator fusion, shape inference)
5. **Week 5**: Quantization passes (INT8 PTQ, INT4 GPTQ-style, mixed precision)
6. **Week 6**: Memory planner (lifetime analysis, memory map, KV cache layout)
7. **Week 7**: Graph family generator (multi-target, multi-condition)
8. **Week 8**: Kernel templates (CPU NEON, basic Vulkan/OpenCL, NPU delegate boundaries)
9. **Week 9**: Package generator (`.armpack` format), CLI, C SDK
10. **Week 10**: Integration testing, benchmark harness, documentation

---

## Verification Plan

### Automated Tests
```bash
# Unit tests
cmake --build build && ctest --test-dir build -R unit

# IR roundtrip test
arm-ir-dump test_model.armpack | arm-ir-parse | arm-ir-dump  # must be identical

# Quantization accuracy test
python -m armcc.test.accuracy --model Qwen/Qwen2.5-0.5B --quant int8 --dataset wikitext

# Package integrity test
armcc inspect output.armpack --verify
```

### Manual Verification
- Compile `SmolLM2-135M` end-to-end and inspect the generated `.armpack`
- Verify graph family has expected number of graphs per target
- Validate ARM-IR dump is human-readable and structurally correct
- Run C SDK reference runtime on at least one graph on ARM64 Linux
