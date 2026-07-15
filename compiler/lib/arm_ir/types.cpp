// ============================================================================
// ARM AI Compiler — ARM-IR: Types (implementation)
// ============================================================================
#include "arm_ir/types.h"
#include <stdexcept>
#include <cassert>

namespace armcc {
namespace ir {

// ---------------------------------------------------------------------------
// DType helpers
// ---------------------------------------------------------------------------
const char* dtypeToString(DType dtype) {
  switch (dtype) {
    case DType::F32:     return "f32";
    case DType::F16:     return "f16";
    case DType::BF16:    return "bf16";
    case DType::F8_E4M3: return "f8_e4m3";
    case DType::F8_E5M2: return "f8_e5m2";
    case DType::I32:     return "i32";
    case DType::I16:     return "i16";
    case DType::I8:      return "i8";
    case DType::I4:      return "i4";
    case DType::I2:      return "i2";
    case DType::U8:      return "u8";
    case DType::U4:      return "u4";
    case DType::BOOL:    return "bool";
    default:             return "unknown";
  }
}

DType dtypeFromString(const std::string& s) {
  if (s == "f32")     return DType::F32;
  if (s == "f16")     return DType::F16;
  if (s == "bf16")    return DType::BF16;
  if (s == "f8_e4m3") return DType::F8_E4M3;
  if (s == "f8_e5m2") return DType::F8_E5M2;
  if (s == "i32")     return DType::I32;
  if (s == "i16")     return DType::I16;
  if (s == "i8")      return DType::I8;
  if (s == "i4")      return DType::I4;
  if (s == "i2")      return DType::I2;
  if (s == "u8")      return DType::U8;
  if (s == "u4")      return DType::U4;
  if (s == "bool")    return DType::BOOL;
  return DType::UNKNOWN;
}

size_t dtypeElementSize(DType dtype) {
  switch (dtype) {
    case DType::F32:  case DType::I32: return 4;
    case DType::F16:  case DType::BF16: case DType::I16: return 2;
    case DType::F8_E4M3: case DType::F8_E5M2:
    case DType::I8:   case DType::U8:  case DType::BOOL: return 1;
    case DType::I4:   case DType::U4:  return 0; // 0.5 bytes; caller uses /2
    case DType::I2:   return 0;  // 0.25 bytes; caller uses /4
    default: return 0;
  }
}

bool dtypeIsFloat(DType dtype) {
  return dtype == DType::F32  || dtype == DType::F16 ||
         dtype == DType::BF16 || dtype == DType::F8_E4M3 ||
         dtype == DType::F8_E5M2;
}

bool dtypeIsQuantized(DType dtype) {
  return dtype == DType::I8  || dtype == DType::U8 ||
         dtype == DType::I4  || dtype == DType::U4 ||
         dtype == DType::I2;
}

// ---------------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------------
int64_t Shape::numElements() const {
  if (dims.empty()) return 1;
  int64_t n = 1;
  for (auto d : dims) {
    if (d < 0) return -1;   // dynamic
    n *= d;
  }
  return n;
}

bool Shape::isDynamic() const {
  for (auto d : dims) if (d < 0) return true;
  return false;
}

std::string Shape::toString() const {
  std::string s = "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i > 0) s += ", ";
    s += (dims[i] < 0 ? "?" : std::to_string(dims[i]));
  }
  return s + "]";
}

// ---------------------------------------------------------------------------
// ExecUnit helpers
// ---------------------------------------------------------------------------
const char* execUnitToString(ExecUnit eu) {
  switch (eu) {
    case ExecUnit::Auto: return "auto";
    case ExecUnit::CPU:  return "cpu";
    case ExecUnit::GPU:  return "gpu";
    case ExecUnit::NPU:  return "npu";
    case ExecUnit::DSP:  return "dsp";
    case ExecUnit::ANE:  return "ane";
    case ExecUnit::APU:  return "apu";
    default:             return "unknown";
  }
}

// ---------------------------------------------------------------------------
// SoCID helpers
// ---------------------------------------------------------------------------
const char* socIDToString(SoCID id) {
  switch (id) {
    case SoCID::Snapdragon_8_Gen1:  return "snapdragon_8_gen1";
    case SoCID::Snapdragon_8_Gen2:  return "snapdragon_8_gen2";
    case SoCID::Snapdragon_8_Gen3:  return "snapdragon_8_gen3";
    case SoCID::Snapdragon_8_Elite: return "snapdragon_8_elite";
    case SoCID::Snapdragon_7_Gen3:  return "snapdragon_7_gen3";
    case SoCID::Dimensity_9300:     return "dimensity_9300";
    case SoCID::Dimensity_9200:     return "dimensity_9200";
    case SoCID::Dimensity_8300:     return "dimensity_8300";
    case SoCID::Exynos_2400:        return "exynos_2400";
    case SoCID::Exynos_2200:        return "exynos_2200";
    case SoCID::Apple_A17:          return "apple_a17";
    case SoCID::Apple_A18:          return "apple_a18";
    case SoCID::Apple_M3:           return "apple_m3";
    case SoCID::Apple_M4:           return "apple_m4";
    case SoCID::BCM2712:            return "bcm2712";
    case SoCID::Jetson_Orin_NX:     return "jetson_orin_nx";
    case SoCID::Jetson_Orin_Nano:   return "jetson_orin_nano";
    case SoCID::Generic_ARM64:      return "generic_arm64";
    default:                        return "unknown";
  }
}

SoCID socIDFromString(const std::string& s) {
  if (s == "snapdragon_8_gen1")  return SoCID::Snapdragon_8_Gen1;
  if (s == "snapdragon_8_gen2")  return SoCID::Snapdragon_8_Gen2;
  if (s == "snapdragon_8_gen3")  return SoCID::Snapdragon_8_Gen3;
  if (s == "snapdragon_8_elite") return SoCID::Snapdragon_8_Elite;
  if (s == "snapdragon_7_gen3")  return SoCID::Snapdragon_7_Gen3;
  if (s == "dimensity_9300")     return SoCID::Dimensity_9300;
  if (s == "dimensity_9200")     return SoCID::Dimensity_9200;
  if (s == "dimensity_8300")     return SoCID::Dimensity_8300;
  if (s == "exynos_2400")        return SoCID::Exynos_2400;
  if (s == "exynos_2200")        return SoCID::Exynos_2200;
  if (s == "apple_a17")          return SoCID::Apple_A17;
  if (s == "apple_a18")          return SoCID::Apple_A18;
  if (s == "apple_m3")           return SoCID::Apple_M3;
  if (s == "apple_m4")           return SoCID::Apple_M4;
  if (s == "bcm2712")            return SoCID::BCM2712;
  if (s == "jetson_orin_nx")     return SoCID::Jetson_Orin_NX;
  if (s == "jetson_orin_nano")   return SoCID::Jetson_Orin_Nano;
  return SoCID::Generic_ARM64;
}

} // namespace ir
} // namespace armcc
