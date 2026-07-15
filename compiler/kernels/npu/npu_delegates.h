// ============================================================================
// ARM AI Compiler — NPU Delegate Stubs
//
// Provides boundary stubs for NPU subgraph delegation:
//   - Qualcomm QNN/HTP (Hexagon)
//   - MediaTek NeuroPilot (APU)
//   - Apple CoreML / ANE
//   - Samsung Xclipse NPU
//
// These are compile-time stubs. At runtime, the actual NPU SDKs are
// linked dynamically. The stub ABI is stable and matches the runtime SDK.
// ============================================================================
#pragma once
#include <cstdint>
#include <vector>
#include <string>

namespace armcc {
namespace kernels {
namespace npu {

// ---------------------------------------------------------------------------
// Unified NPU subgraph handle
// ---------------------------------------------------------------------------
struct NPUSubgraph {
  enum class Backend { QNN_HTP, MediaTek_APU, Apple_ANE, Samsung_Xclipse, None };

  Backend        backend  = Backend::None;
  void*          handle   = nullptr;   // SDK-specific compiled subgraph
  uint32_t       first_op = 0;         // First op ID in source graph
  uint32_t       last_op  = 0;         // Last op ID in source graph
  std::vector<uint32_t> input_tids;   // Input tensor IDs
  std::vector<uint32_t> output_tids;  // Output tensor IDs

  // Buffer descriptors for zero-copy (all pre-registered with NPU SDK)
  struct IOBuffer {
    void*    ptr    = nullptr;
    uint64_t bytes  = 0;
    bool     is_input;
  };
  std::vector<IOBuffer> buffers;
};

// ---------------------------------------------------------------------------
// QNN HTP (Qualcomm) delegate
// ---------------------------------------------------------------------------
struct QNNHTPDelegate {
  static const char* name() { return "qnn_htp"; }

  // Compile a serialized subgraph for HTP
  // In real impl: calls Qnn_GraphCreate + Qnn_BackendCreate
  static bool compile(NPUSubgraph& sg, const uint8_t* graph_data, size_t size) {
    (void)graph_data; (void)size;
    sg.backend = NPUSubgraph::Backend::QNN_HTP;
    return true;  // Stub: always succeeds
  }

  // Execute a compiled subgraph
  // In real impl: calls Qnn_GraphExecute with pre-allocated IOBuffers
  static bool execute(NPUSubgraph& sg) {
    (void)sg;
    return true;  // Stub
  }

  static void destroy(NPUSubgraph& sg) {
    sg.handle = nullptr;
  }

  // Check if HTP is available on this device
  static bool isAvailable() {
#if defined(__ANDROID__) && defined(__aarch64__)
    // Check for libQnnHtp.so
    return true;  // Stub: in real impl, dlopen("libQnnHtp.so")
#else
    return false;
#endif
  }
};

// ---------------------------------------------------------------------------
// MediaTek NeuroPilot APU delegate
// ---------------------------------------------------------------------------
struct NeuroPilotDelegate {
  static const char* name() { return "neuripilot"; }

  static bool compile(NPUSubgraph& sg, const uint8_t* data, size_t size) {
    (void)data; (void)size;
    sg.backend = NPUSubgraph::Backend::MediaTek_APU;
    return true;
  }

  static bool execute(NPUSubgraph& sg) { (void)sg; return true; }
  static void destroy(NPUSubgraph& sg) { sg.handle = nullptr; }

  static bool isAvailable() {
#if defined(__ANDROID__) && defined(__aarch64__)
    return false;  // Stub: dlopen("libneuropilot.so")
#else
    return false;
#endif
  }
};

// ---------------------------------------------------------------------------
// Apple CoreML / ANE delegate
// ---------------------------------------------------------------------------
struct CoreMLDelegate {
  static const char* name() { return "coreml"; }

  static bool compile(NPUSubgraph& sg, const uint8_t* data, size_t size) {
    (void)data; (void)size;
    sg.backend = NPUSubgraph::Backend::Apple_ANE;
    return true;
  }

  static bool execute(NPUSubgraph& sg) { (void)sg; return true; }
  static void destroy(NPUSubgraph& sg) { sg.handle = nullptr; }

  static bool isAvailable() {
#if defined(__APPLE__)
    return true;  // Stub: CoreML always available on Apple
#else
    return false;
#endif
  }
};

// ---------------------------------------------------------------------------
// Samsung Xclipse NPU delegate
// ---------------------------------------------------------------------------
struct XclipseDelegate {
  static const char* name() { return "xclipse"; }

  static bool compile(NPUSubgraph& sg, const uint8_t* data, size_t size) {
    (void)data; (void)size;
    sg.backend = NPUSubgraph::Backend::Samsung_Xclipse;
    return true;
  }

  static bool execute(NPUSubgraph& sg) { (void)sg; return true; }
  static void destroy(NPUSubgraph& sg) { sg.handle = nullptr; }
  static bool isAvailable() { return false; }
};

// ---------------------------------------------------------------------------
// DelegateRegistry: auto-select the appropriate NPU delegate at runtime
// ---------------------------------------------------------------------------
class DelegateRegistry {
public:
  enum class DelegateType { None, QNN_HTP, NeuroPilot, CoreML, Xclipse };

  static DelegateType detectAvailable() {
    if (QNNHTPDelegate::isAvailable())    return DelegateType::QNN_HTP;
    if (NeuroPilotDelegate::isAvailable()) return DelegateType::NeuroPilot;
    if (CoreMLDelegate::isAvailable())     return DelegateType::CoreML;
    if (XclipseDelegate::isAvailable())    return DelegateType::Xclipse;
    return DelegateType::None;
  }

  static std::string delegateName(DelegateType t) {
    switch (t) {
      case DelegateType::QNN_HTP:     return "qnn_htp";
      case DelegateType::NeuroPilot:  return "neuripilot";
      case DelegateType::CoreML:      return "coreml";
      case DelegateType::Xclipse:     return "xclipse";
      default:                        return "none";
    }
  }

  static bool compileSubgraph(NPUSubgraph& sg,
                               DelegateType type,
                               const uint8_t* data, size_t size)
  {
    switch (type) {
      case DelegateType::QNN_HTP:     return QNNHTPDelegate::compile(sg, data, size);
      case DelegateType::NeuroPilot:  return NeuroPilotDelegate::compile(sg, data, size);
      case DelegateType::CoreML:      return CoreMLDelegate::compile(sg, data, size);
      case DelegateType::Xclipse:     return XclipseDelegate::compile(sg, data, size);
      default: return false;
    }
  }

  static bool executeSubgraph(NPUSubgraph& sg, DelegateType type) {
    switch (type) {
      case DelegateType::QNN_HTP:     return QNNHTPDelegate::execute(sg);
      case DelegateType::NeuroPilot:  return NeuroPilotDelegate::execute(sg);
      case DelegateType::CoreML:      return CoreMLDelegate::execute(sg);
      case DelegateType::Xclipse:     return XclipseDelegate::execute(sg);
      default: return false;
    }
  }
};

} // namespace npu
} // namespace kernels
} // namespace armcc
