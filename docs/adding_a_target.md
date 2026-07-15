# Adding a New Target to the ARM AI Compiler

This guide explains how to add support for a new ARM SoC target.

## Step 1: Create a Device Profile JSON

Create a new file in `device_profiles/<soc_id>.json`:

```json
{
  "soc_id": "my_soc_v1",
  "name": "My SoC V1",
  "vendor": "my_vendor",

  "l1_cache_kb": 64,
  "l2_cache_kb": 2048,
  "l3_cache_kb": 8192,
  "ram_bandwidth_gbps": 60,

  "cpu": {
    "num_big_cores": 4,
    "num_little_cores": 4,
    "big_core_freq_mhz": 3000,
    "little_core_freq_mhz": 2000,
    "has_neon": true,
    "has_dotprod": true,
    "has_i8mm": true,
    "has_sve": false
  },

  "gpu": {
    "present": true,
    "name": "My GPU",
    "compute_units": 8,
    "freq_mhz": 900,
    "supports_vulkan": true,
    "supports_opencl": true,
    "fp16_tflops": 3.0
  },

  "npu": {
    "present": true,
    "name": "My NPU",
    "tops_int8": 40,
    "tops_int4": 80,
    "supported_dtypes": ["i8", "i4", "f16"],
    "delegate": "my_npu_sdk"
  },

  "supported_dtypes": {
    "cpu": ["f32", "f16", "bf16", "i8", "i4"],
    "gpu": ["f32", "f16", "i8"],
    "npu": ["i8", "i4", "f16"],
    "dsp": []
  },

  "cost_table": [
    {
      "op": "MatMul",
      "input_dtype": "f16", "weight_dtype": "f16",
      "M": 1, "K": 4096, "N": 4096,
      "cpu_ms": 15.0, "gpu_ms": 4.0, "npu_ms": 1.5, "dsp_ms": -1,
      "cpu_mw": 700,  "gpu_mw": 1100, "npu_mw": 400, "dsp_mw": -1
    }
  ]
}
```

> [!TIP]
> Measure the `cost_table` entries with real hardware for best results. Use the `armcc benchmark` command or run microbenchmarks on-device. Inaccurate cost tables will result in sub-optimal op placement.

## Step 2: Register the SoC ID

In `compiler/include/arm_ir/types.h`, add your SoC to the `SoCID` enum:

```diff
 enum class SoCID : uint16_t {
   Unknown         = 0x0000,
   Generic_ARM64   = 0x00FF,
   Snapdragon_8_Gen3 = 0x0101,
   ...
+  My_SoC_V1      = 0x0500,   // Add your SoC
 };
```

Then update the `socIDFromString` / `socIDToString` functions in `types.cpp`:

```diff
+  if (s == "my_soc_v1")  return SoCID::My_SoC_V1;
```

## Step 3: Add NPU Delegate (Optional)

If your NPU uses a custom SDK, add a delegate stub in `compiler/kernels/npu/npu_delegates.h`:

```cpp
struct MyNPUDelegate {
  static const char* name() { return "my_npu_sdk"; }

  static bool compile(NPUSubgraph& sg, const uint8_t* data, size_t size) {
    // Call your SDK's graph compilation API
    // e.g., myNPU_compile(data, size, &sg.handle);
    sg.backend = NPUSubgraph::Backend::None;  // Add enum value if needed
    return true;
  }

  static bool execute(NPUSubgraph& sg) {
    // Call your SDK's inference API
    // e.g., myNPU_execute(sg.handle, sg.buffers...);
    return true;
  }

  static void destroy(NPUSubgraph& sg) {
    // Free SDK resources
    sg.handle = nullptr;
  }

  static bool isAvailable() {
    // Check if the SDK library is present at runtime
    return dlopen("libmynpu.so", RTLD_LAZY) != nullptr;
  }
};
```

Then register it in `DelegateRegistry::detectAvailable()`.

## Step 4: Test with a Model

Run the compiler targeting your new SoC:

```bash
armcc compile \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --targets my_soc_v1 \
  --quantization int8 \
  --output smollm2-135m-my-soc.armpack
```

Inspect the output:
```bash
armcc inspect smollm2-135m-my-soc.armpack
```

## Step 5: Add Unit Tests

Add a test entry in `tests/unit/test_cost_model.cpp`:

```cpp
TEST(CostModelTest, MySOCProfile) {
  CostModel cm;
  cm.loadProfile("device_profiles/my_soc_v1.json");
  auto* p = cm.getProfile(SoCID::My_SoC_V1);
  ASSERT_NE(p, nullptr);
  EXPECT_TRUE(p->has_npu);
  EXPECT_GT(p->npu_tops, 0);
}
```

## Cost Table Measurement Guide

To measure accurate cost table entries on real hardware:

| Op | How to measure |
|---|---|
| `MatMul` | Run `torch.mm(torch.randn(1, K), torch.randn(K, N).half())` 100×, median time |
| `GroupQueryAttention` | Run `scaled_dot_product_attention` 100×, measure time per call |
| `RMSNorm` | Run `torch.nn.functional.rms_norm` 1000×, median time |
| `Softmax` | `torch.nn.functional.softmax(torch.randn(1, N).half())` 1000× |
| `Embedding` | `F.embedding(torch.randint(1000, (1,)), torch.randn(vocab, d))` |

Use `torch.cuda.synchronize()` or appropriate sync for GPU/NPU measurements.

Alternatively, use the `armcc benchmark` command (Phase 2) which will automatically run all microbenchmarks on the target device.

## File Checklist

| File | What to change |
|---|---|
| `device_profiles/<soc_id>.json` | **Create**: Full device profile + cost table |
| `compiler/include/arm_ir/types.h` | **Edit**: Add `SoCID::My_SoC_V1` enum value |
| `compiler/lib/arm_ir/types.cpp` | **Edit**: Add `socIDFromString/toStr` entries |
| `compiler/kernels/npu/npu_delegates.h` | **Edit**: Add NPU delegate (if NPU present) |
| `tests/unit/test_cost_model.cpp` | **Edit**: Add profile load test |
| `docs/arm_ir_spec.md` | **Update**: Mention new target in hardware table |
