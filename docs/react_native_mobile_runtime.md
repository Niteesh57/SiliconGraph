# React Native mobile runtime

`silicongraph-mobile` is the Android React Native client for `.armpack` files.
Its UI is deliberately separate from model execution:

```text
React Native UI
  -> Kotlin bridge: file staging, package inspection, live Android telemetry
  -> arm64 JNI runtime: package loader -> graph selector -> tensor executor
       -> CPU (KleidiAI / NEON) or Vulkan (only after an end-to-end benchmark)
```

The bridge holds archive paths, not weight buffers. Document-provider URIs are
staged to private storage because a generic `content://` stream cannot reliably
be memory mapped or randomly accessed by a native inference engine.

## Runtime metrics contract

During actual generation, native code must emit `SiliconGraphTelemetry` at
prefill start, first decoded token, every decode interval, and completion. The
event carries values measured at that moment:

- available system RAM and Java/native heap;
- memory immediately before generation, at first token, and application peak;
- TTFT, generated-token count, decode tokens/sec, selected graph/backend, and
  KV-cache bytes;
- battery, thermal, CPU, ABI and Vulkan capability.

The UI must leave a metric in its `Waiting` state until native inference emits
it. Estimated compiler latency in `manifest.json` is package-planning metadata,
not a mobile benchmark, and must not be displayed as measured speed.

## Current boundary

The `NativeRuntime` JNI library is linked for `arm64-v8a`, but the compiler
repository still needs an implementation for the C SDK package/session APIs and
the kernel dispatcher. Until then `generate()` returns
`ARMCC_RUNTIME_NOT_LINKED` after the *real* package has been inspected and a
safe graph variant has been selected. This is intentional safety behavior.
