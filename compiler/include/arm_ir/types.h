// ============================================================================
// ARM AI Compiler — ARM Intermediate Representation: Types
//
// Defines the fundamental data types used throughout the ARM-IR.
// These mirror the type system of the models we ingest (PyTorch dtypes)
// but are extended with quantization and hardware-affinity metadata.
// ============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <optional>

namespace armcc {
namespace ir {

// ---------------------------------------------------------------------------
// Element type
// ---------------------------------------------------------------------------
enum class DType : uint8_t {
  // Floating point
  F32 = 0,
  F16,
  BF16,
  F8_E4M3,   // FP8: 4-bit exponent, 3-bit mantissa (NaN-aware)
  F8_E5M2,   // FP8: 5-bit exponent, 2-bit mantissa

  // Integer (signed)
  I32,
  I16,
  I8,
  I4,   // packed, 2 per byte
  I2,   // packed, 4 per byte

  // Integer (unsigned / quantized)
  U8,
  U4,

  // Boolean
  BOOL,

  UNKNOWN,
};

const char* dtypeToString(DType dtype);
DType       dtypeFromString(const std::string& s);
size_t      dtypeElementSize(DType dtype);  // bytes, 0.5 for 4-bit
bool        dtypeIsFloat(DType dtype);
bool        dtypeIsQuantized(DType dtype);

// ---------------------------------------------------------------------------
// Tensor shape
// ---------------------------------------------------------------------------
struct Shape {
  std::vector<int64_t> dims;   // -1 = dynamic dimension

  Shape() = default;
  explicit Shape(std::initializer_list<int64_t> d) : dims(d) {}
  explicit Shape(std::vector<int64_t> d) : dims(std::move(d)) {}

  size_t rank() const { return dims.size(); }
  int64_t numElements() const;         // -1 if any dynamic dim
  bool isDynamic() const;
  bool operator==(const Shape& o) const { return dims == o.dims; }
  std::string toString() const;
};

// ---------------------------------------------------------------------------
// Quantization parameters
// ---------------------------------------------------------------------------
enum class QuantScheme : uint8_t {
  None = 0,
  PerTensor,           // one scale + zero_point
  PerChannel,          // one scale + zp per output channel
  PerGroup,            // GPTQ-style group quantization (e.g. g128)
  Dynamic,             // dynamic per-token quantization
};

struct QuantParams {
  QuantScheme   scheme      = QuantScheme::None;
  DType         stored_as   = DType::F32;  // after dequant → original dtype
  int32_t       group_size  = 128;          // for PerGroup
  int32_t       channel_dim = 0;            // for PerChannel

  // For PerTensor / PerChannel / PerGroup:
  // scales and zero_points stored as flat vectors.
  // Length = 1 (PerTensor), num_channels (PerChannel), or
  //          (num_elements / group_size) (PerGroup)
  std::vector<float>   scales;
  std::vector<int32_t> zero_points;

  bool isQuantized() const { return scheme != QuantScheme::None; }
};

// ---------------------------------------------------------------------------
// Memory layout / format
// ---------------------------------------------------------------------------
enum class MemoryLayout : uint8_t {
  // Standard row-major layouts
  NCHW = 0,  // batch, channels, height, width
  NHWC,      // batch, height, width, channels (preferred by ARM)
  NC,        // batch, features (2D)
  NxKx,      // general matrix (N × K)

  // Tiled / packed formats for NPU/DSP
  Tiled_4x4,
  Tiled_8x8,
  Tiled_16x16,

  // Blocked for SIMD (e.g. 8 elements per block for NEON)
  Blocked_8,
  Blocked_16,

  // Planar vs interleaved
  Planar,
  Interleaved,

  Unknown,
};

// ---------------------------------------------------------------------------
// Execution unit affinity
// Used as hints to the scheduler. The compiler may override at will.
// ---------------------------------------------------------------------------
enum class ExecUnit : uint8_t {
  Auto = 0,   // let the cost model decide
  CPU,
  GPU,
  NPU,
  DSP,
  ANE,        // Apple Neural Engine
  APU,        // MediaTek AI Processing Unit
};

const char* execUnitToString(ExecUnit eu);

// ---------------------------------------------------------------------------
// Thermal / power state (used in graph family selection)
// ---------------------------------------------------------------------------
enum class ThermalState : uint8_t {
  Nominal = 0,
  Warm,
  Hot,      // throttling may occur
  Critical, // emergency throttle
};

// ---------------------------------------------------------------------------
// SoC identifier (used by cost model + graph selector)
// ---------------------------------------------------------------------------
enum class SoCID : uint16_t {
  Unknown = 0,

  // Qualcomm Snapdragon
  Snapdragon_8_Gen1,
  Snapdragon_8_Gen2,
  Snapdragon_8_Gen3,
  Snapdragon_8_Elite,
  Snapdragon_7_Gen3,

  // MediaTek Dimensity
  Dimensity_9300,
  Dimensity_9200,
  Dimensity_8300,

  // Samsung Exynos
  Exynos_2400,
  Exynos_2200,

  // Apple (A-series + M-series)
  Apple_A17,
  Apple_A18,
  Apple_M3,
  Apple_M4,

  // Broadcom (Raspberry Pi)
  BCM2712,   // Pi 5

  // NVIDIA Jetson
  Jetson_Orin_NX,
  Jetson_Orin_Nano,

  // Generic ARM64 (no specific NPU)
  Generic_ARM64,
};

const char* socIDToString(SoCID id);
SoCID       socIDFromString(const std::string& s);

} // namespace ir
} // namespace armcc
