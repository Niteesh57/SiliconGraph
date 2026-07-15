# ARM-IR Specification v1.0

> The Intermediate Representation of the ARM AI Compiler

## Overview

ARM-IR is the core data structure of the ARM AI Compiler. It is a **rich, metadata-annotated computation graph** designed specifically for AI model deployment on ARM hardware.

Unlike ONNX (which targets portability) or TOSA (which targets hardware abstraction), ARM-IR is designed for **compiler-driven optimization** — every node and tensor carries metadata that optimization passes read and write.

## IR Structure

```
IRGraph
├── metadata: model_id, arch_family, num_layers, hidden_size, ...
├── nodes: IRNode[]
│   ├── id, name, op (OpCode)
│   ├── inputs: tensor_id[]
│   ├── outputs: tensor_id[]
│   ├── attrs: Attribute[]
│   └── ARM metadata:
│       ├── preferred_unit / assigned_unit  (ExecUnit)
│       ├── cost_cpu_ms / cost_gpu_ms / cost_npu_ms / cost_dsp_ms
│       ├── thermal_cost_mJ
│       ├── scheduling_hint (parallelizable, latency_sensitive, ...)
│       ├── supports_streaming / streaming_chunk_size
│       └── in_npu_subgraph / in_gpu_subgraph / in_dsp_subgraph
│
└── tensors: IRTensor[]
    ├── id, name, dtype (DType), shape (Shape)
    ├── quant (QuantParams: scheme, scales, zero_points, group_size)
    ├── layout (MemoryLayout: NCHW, NHWC, Tiled_8x8, Blocked_16, ...)
    ├── is_weight / is_input / is_output / is_kv_cache
    ├── preferred_unit, cache_policy
    └── memory planning: live_start, live_end, mem_offset, mem_arena
```

## Text Format

ARM-IR has a human-readable text format (produced by `arm-ir-dump`):

```
// ARM AI Compiler — ARM-IR Dump
// Model: meta-llama/Llama-3.2-1B-Instruct
// Family: llama_style
// Nodes: 420  Tensors: 844

graph @llama_3.2_1b {
  // Inputs
  input %input_ids : i32[1, ?]

  %embed_0 = Embedding(%input_ids)  // exec=cpu cpu=0.1ms
  %norm_0 = RMSNorm_Scale(%embed_0, %norm_weight)  // exec=cpu cpu=0.3ms [fused]
  %q_proj = MatMul(%norm_0, %q_weight)  // exec=npu npu=0.4ms
  %k_proj = MatMul(%norm_0, %k_weight)  // exec=npu npu=0.4ms
  %v_proj = MatMul(%norm_0, %v_weight)  // exec=npu npu=0.4ms
  %hw0 = HW_Boundary(%norm_0)           // exec=auto from=cpu to=npu
  %attn_0 = GroupQueryAttention(%q_proj, %k_proj, %v_proj)  // exec=npu npu=0.8ms
  %hw1 = HW_Boundary(%attn_0)           // exec=auto from=npu to=gpu
  %ffn_0 = FFN_SwiGLU(%attn_0, ...)     // exec=gpu gpu=1.9ms [fused]
  ...

  // Outputs
  return %logits
}
```

## Operator Set

See [ops.h](../compiler/include/arm_ir/ops.h) for the complete OpCode list.

**LLM-specific operators** (not in ONNX or TOSA):

| OpCode | Description |
|---|---|
| `GroupQueryAttention` | GQA (Llama 3, Mistral) |
| `FlashAttention` | Memory-efficient attention (chunked) |
| `RopeEmbedding` | Rotary Position Embedding |
| `ALiBiEmbedding` | ALiBi positional bias |
| `RMSNorm` | Root Mean Square Layer Norm |
| `RMSNorm_Scale` | Fused RMSNorm + learned scale |
| `FFN_SwiGLU` | Fused SwiGLU FFN block |
| `KVCacheRead/Write` | KV cache access ops |
| `Attention_Masked` | Causal masked attention |

**ARM hardware hint operators** (inserted by HardwareFusionPass):

| OpCode | Description |
|---|---|
| `HW_Boundary` | Marks a CPU↔GPU↔NPU↔DSP transition |
| `DMA_Prefetch` | Hint: initiate DMA prefetch |
| `NPU_SubgraphBegin/End` | Marks delegated NPU subgraph boundaries |

## Quantization Metadata (on IRTensor)

```cpp
struct QuantParams {
  QuantScheme scheme;      // PerTensor / PerChannel / PerGroup / Dynamic
  DType       stored_as;   // I8 / I4 / F8_E4M3 / ...
  int32_t     group_size;  // For PerGroup (e.g. 128)
  int32_t     channel_dim; // For PerChannel
  vector<float>   scales;
  vector<int32_t> zero_points;
};
```

## Memory Metadata (on IRTensor)

| Field | Set by | Purpose |
|---|---|---|
| `live_start` | TensorLifetimeAnalysisPass | Op index where tensor is produced |
| `live_end` | TensorLifetimeAnalysisPass | Last op index to use tensor |
| `mem_offset` | MemoryAliasingPass | Byte offset in arena |
| `mem_arena` | MemoryAliasingPass | 0=weights, 1=activations, 2=kv_cache |
| `cache_policy` | PrefetchInsertionPass | Default/Prefetch/Pin/Evict/Bypass |

## Serialization

ARM-IR is serialized to FlatBuffers binary format for efficient loading by the C++ compiler. The text format is for debugging only.

Binary files use the `.armgraph` extension inside `.armpack` archives.
