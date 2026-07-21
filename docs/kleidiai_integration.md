# KleidiAI Integration

SiliconGraph uses KleidiAI as an optional Arm CPU micro-kernel dependency. It
does not replace the compiler or the runtime: KleidiAI supplies highly tuned
building blocks such as packing and low-bit matrix multiplication, while
SiliconGraph decides which graph, precision, memory plan, and fallback policy
are safe for the target phone.

## Build configuration

`ARMCC_ENABLE_KLEIDIAI=ON` enables the dependency only for an Arm target
(`aarch64`, `arm64`, or `armv*`). It is deliberately rejected for x86 hosts
because KleidiAI micro-kernels contain Arm-specific instructions.

By default CMake fetches the pinned `v1.28.0` release. For a controlled build,
provide an already-audited checkout:

```bash
cmake -B build-arm -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DARMCC_ENABLE_KLEIDIAI=ON \
  -DARMCC_FETCH_KLEIDIAI=OFF \
  -DARMCC_KLEIDIAI_SOURCE_DIR=/path/to/kleidiai
```

The root project exposes the `armcc_kleidiai` interface target. Consumers that
link `armcc_all` therefore receive `KleidiAI::kleidiai` and the
`ARMCC_HAS_KLEIDIAI` compile definition when this option is enabled.

## Runtime strategy

The first runtime path is ExecuTorch with XNNPACK. XNNPACK handles graph
partitioning, memory management, and thread scheduling; its Arm low-bit path
can use KleidiAI kernels. SiliconGraph must emit only quantization layouts that
are compatible with the selected runtime/kernel contract.

Direct KleidiAI calls are reserved for a later native CPU backend. That backend
must provide the surrounding operator implementation, packing, tiling,
threading, workspace planning, and fallback logic; KleidiAI intentionally does
not supply those runtime services.
