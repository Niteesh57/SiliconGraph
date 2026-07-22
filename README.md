# ARM AI Compiler

> The LLVM of AI deployment — a universal AI model optimizer targeting ARM devices.

## What this is

ARM AI Compiler ingests open-source models from HuggingFace (Llama, Qwen, Gemma, Phi, SmolLM) and emits a portable `.armpack` file containing **multiple optimized graphs** — each tuned for a specific ARM hardware profile (Snapdragon, MediaTek, Exynos, Apple Silicon, generic ARM CPU).

Unlike llama.cpp, ONNX Runtime, or Cactus — which compile one graph — this compiler generates a **Graph Family**: many optimized graphs varying by hardware, precision (FP32/FP16/INT8/INT4/FP8), memory budget, context length, and thermal state. The runtime simply picks the best graph at deployment time.

## Multimodal model contracts

Packages can declare text, image, audio, video, embedding, image-generation,
and audio-generation I/O. Image/audio/video packages include the model's
Hugging Face processor assets alongside tokenizer assets, so the on-device
runtime can apply the model's own preprocessing rules. See
[the multimodal support guide](docs/multimodal_support.md) for native-runtime
status and rollout order.

## Architecture

```
HF Model (PyTorch / Safetensors)
         ↓
  [Model Ingestion Layer]       ← Python (HF integration)
         ↓
  [ARM IR Generator]            ← C++ / MLIR
         ↓
  [Analysis & Cost Model]       ← C++
         ↓
  [Optimization Pass Pipeline]  ← C++ (20+ passes)
         ↓
  [Graph Family Generator]      ← C++
         ↓
  [Kernel Generator]            ← C++ + LLVM backends
         ↓
  [Memory Planner]              ← C++
         ↓
  [Package Generator]           ← C++
         ↓
  model.armpack  ←  Portable multi-graph package
```

## Quick Start

### Compile a model

```bash
armcc compile \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --targets snapdragon_8_gen3,dimensity_9300,apple_a18,generic_arm64 \
  --quantization mixed \
  --context-lengths 512,2048,8192 \
  --output ./llama-3.2-1b.armpack
```

### Inspect a package

```bash
armcc inspect ./llama-3.2-1b.armpack
arm-ir-dump ./llama-3.2-1b.armpack --graph snapdragon_npu_int4
```

### Use the C SDK

```c
ArmPackage* pkg   = armpack_load("llama-3.2-1b.armpack");
ArmDeviceState st = { .soc_id = ARMCC_SOC_SNAPDRAGON_8_GEN3,
                      .ram_available = 1500, .battery_pct = 85,
                      .thermal_state = ARMCC_THERMAL_NOMINAL };
ArmGraph* g = armpack_select_graph(pkg, &st);
ArmInferenceResult r = armgraph_run(g, tokens, n_tokens);
```

## Supported Targets

| Profile | NPU | GPU | CPU |
|---|---|---|---|
| `snapdragon_8_gen3` | Hexagon HTP | Adreno 750 | Cortex-X4 |
| `snapdragon_8_elite` | Hexagon HTP | Adreno 830 | Oryon |
| `dimensity_9300` | APU 790 | Immortalis-G720 | Cortex-X4 |
| `exynos_2400` | NPU | Xclipse 940 | Cortex-X4 |
| `apple_m4` | ANE | Apple GPU | M4 |
| `apple_a18` | ANE | Apple GPU | A18 |
| `generic_arm64` | — | — | NEON/SVE |

## Building

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DARMCC_ENABLE_MLIR=ON
cmake --build build
```

### Arm CPU micro-kernels (Android/Arm builds)

Enable KleidiAI for an Arm target to compile the project with Arm's optimized
CPU micro-kernels available to downstream runtime integrations:

```bash
cmake -B build-arm -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DARMCC_ENABLE_KLEIDIAI=ON
cmake --build build-arm
```

KleidiAI is Arm-only, so this option intentionally remains off for x86 host
development builds. The build fetches the pinned `v1.28.0` release by default;
set `ARMCC_KLEIDIAI_SOURCE_DIR` to use an audited local checkout instead.

**Dependencies** (see `vcpkg.json`):
- LLVM/MLIR ≥ 18
- FlatBuffers ≥ 24.3
- nlohmann-json ≥ 3.11
- libzip ≥ 1.10

**Python** (see `python/requirements.txt`):
- transformers ≥ 4.45
- torch ≥ 2.4
- safetensors ≥ 0.4
- datasets ≥ 2.20

## Project Layout

```
compile-for-arm/
├── compiler/          # C++ compiler core
│   ├── include/       # Public headers
│   ├── lib/           # Implementation
│   └── tools/         # CLI tools (armcc, arm-ir-dump)
├── python/            # Python ingestion + calibration
│   └── armcc/
├── device_profiles/   # SoC capability descriptors (JSON)
├── tests/             # Unit + integration tests
└── docs/              # Specs and guides
```

## License

Apache 2.0 — the `.armpack` format spec is an open standard.
