# .armpack Format Specification v1.0

> The portable output format of the ARM AI Compiler

## Overview

`.armpack` is a **ZIP-based portable archive** that contains the complete compiled output for one AI model. It is the only artifact produced by the ARM AI Compiler and can be consumed by any compliant ARM runtime — no dependency on `libarmcc` required.

## Archive Structure

```
model.armpack  (ZIP archive, standard ZIP64)
│
├── manifest.json                # Top-level index (required)
├── selector.wasm                # Portable graph selector (WASM, required)
├── runtime_config.json          # Sampling config (required)
│
├── graphs/                      # Compiled graph family
│   ├── graph_000.armgraph       # Graph 0: Snapdragon 8 Gen 3, INT4, NPU
│   ├── graph_001.armgraph       # Graph 1: Snapdragon 8 Gen 3, INT8, GPU
│   ├── graph_002.armgraph       # Graph 2: Dimensity 9300, INT4, APU
│   └── graph_003.armgraph       # Graph 3: Generic ARM64, INT8, CPU
│
├── kernels/                     # Precompiled hardware kernels
│   ├── snapdragon_htp/
│   │   └── attention_int4_htp.bin
│   ├── adreno_vulkan/
│   │   └── gemm_int8_vulkan.spv
│   ├── mali_opencl/
│   │   └── gemm_fp16_cl.bin
│   └── arm_neon/
│       └── matmul_int8_neon.o
│
├── weights/                     # Quantized weight blobs
│   ├── weights_int4_g128.bin    # INT4 grouped weights (mmap-safe, page-aligned)
│   └── weights_int8.bin         # INT8 per-channel weights
│
├── memory_maps/                 # Static memory layout descriptors
│   ├── weights_layout.json      # Weight tensor offsets and sizes
│   └── kv_cache_layout.json     # KV cache layout for each context length
│
└── tokenizer/                   # Standard HuggingFace tokenizer assets
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── vocab.json  (or merges.txt for BPE)
```

## manifest.json Schema

```json
{
  "armpack_version": "1.0",
  "compiler_version": "0.1.0",
  "model_id": "meta-llama/Llama-3.2-1B-Instruct",
  "model_family": "llama_style",
  "created_at": "2025-07-09T12:00:00Z",

  "arch": {
    "num_layers": 16,
    "hidden_size": 2048,
    "num_heads": 32,
    "num_kv_heads": 8,
    "vocab_size": 128256,
    "max_position_embeddings": 131072
  },

  "graphs": [
    {
      "index": 0,
      "id": "snapdragon_8_gen3_int4_2048mb_interactive",
      "path": "graphs/graph_000.armgraph",
      "soc_name": "snapdragon_8_gen3",
      "quant_dtype": "int4",
      "memory_budget_mb": 2048,
      "context_length": 2048,
      "thermal_state": "nominal",
      "latency_mode": "interactive",
      "estimated_ttft_ms": 45.2,
      "estimated_tpot_ms": 12.4,
      "peak_memory_bytes": 1073741824,
      "weight_bytes": 671088640,
      "kernel_ids": ["attn_int4_htp", "gemm_int4_htp"]
    }
  ],

  "kernels": [
    {
      "id": "attn_int4_htp",
      "target": "npu_htp",
      "path": "kernels/snapdragon_htp/attention_int4_htp.bin",
      "size_bytes": 524288,
      "op_pattern": "GroupQueryAttention",
      "is_fused": false
    }
  ],

  "weights": [
    {
      "id": "weights_int4_g128",
      "path": "weights/weights_int4_g128.bin",
      "dtype": "int4",
      "size_bytes": 671088640,
      "mmap_safe": true,
      "page_alignment": 4096
    }
  ],

  "tokenizer_path": "tokenizer/",
  "selector_wasm_path": "selector.wasm",
  "runtime_config_path": "runtime_config.json",
  "memory_maps_path": "memory_maps/"
}
```

## selector.wasm

The `selector.wasm` is a portable WebAssembly module that implements graph selection. It exports:

```typescript
// Select the best graph index for a device state
export function selectGraph(
  soc_id: i32,
  ram_mb: i32,
  battery_pct: i32,
  thermal_state: i32,
  latency_mode: i32,
  context_length: i32,
  gpu_available: i32,
  npu_available: i32
): i32;  // Returns graph index (0-based)
```

By compiling the selector to WASM, any runtime on any platform can run it without depending on the C++ compiler library.

## Weight Format

Weights are stored in a flat binary format optimized for `mmap`:

- **Page-aligned**: Each weight tensor starts at a multiple of 4096 bytes
- **Contiguous**: All tensors concatenated with padding for alignment
- **Zero-copy ready**: The layout matches the ARM-IR memory map exactly
- **INT4 packing**: Two INT4 values per byte, little-endian

The `memory_maps/weights_layout.json` file describes the exact offset and size of every tensor within the weight blob.

## .armgraph Format

Each `.armgraph` file is a **FlatBuffers** binary containing the serialized `IRGraph` for that condition set. The schema is defined in `compiler/include/arm_ir/arm_ir.fbs` (FlatBuffers schema file).

The binary format includes:
- Op graph (nodes, tensors, edges)
- Per-node execution unit assignments
- Per-node cost model cache
- Quantization parameters
- Memory offsets

## Runtime Integration

A minimal runtime integration requires:

1. **Open the ZIP**: Standard ZIP library (libzip, minizip, etc.)
2. **Parse manifest.json**: Any JSON parser
3. **Run selector.wasm**: Any WASM runtime (wasmtime, wasm3, etc.)
4. **mmap the weight file**: OS-native mmap (no copy)
5. **Dispatch ops**: Using kernel objects from `kernels/`

No other dependency on the ARM AI Compiler is required.
