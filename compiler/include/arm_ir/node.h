// ============================================================================
// ARM AI Compiler — ARM Intermediate Representation: Node & Tensor
//
// An IRNode represents one operation in the computation graph.
// An IRTensor represents the data flowing between nodes.
//
// Key innovation: nodes carry rich ARM-specific metadata that the
// optimization passes read and write. This metadata — thermal cost,
// execution unit affinity, cache policy, scheduling hints — is what
// distinguishes ARM-IR from ONNX or TOSA.
// ============================================================================
#pragma once

#include "arm_ir/types.h"
#include "arm_ir/ops.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <variant>
#include <optional>

namespace armcc {
namespace ir {

class IRNode;
class IRGraph;

// ---------------------------------------------------------------------------
// Attribute — carries per-node hyperparameters (kernel size, groups, etc.)
// ---------------------------------------------------------------------------
using AttrValue = std::variant<
  int64_t,
  float,
  bool,
  std::string,
  std::vector<int64_t>,
  std::vector<float>
>;

struct Attribute {
  std::string name;
  AttrValue   value;

  template<typename T>
  const T& get() const { return std::get<T>(value); }

  template<typename T>
  bool is() const { return std::holds_alternative<T>(value); }
};

// ---------------------------------------------------------------------------
// Cache policy hint
// ---------------------------------------------------------------------------
enum class CachePolicy : uint8_t {
  Default = 0,
  Prefetch,    // Pre-load into L2/L3 before use
  Pin,         // Keep in cache (e.g. small weight matrices)
  Evict,       // Aggressively evict after use
  Bypass,      // Skip cache entirely (streaming DMA path)
};

// ---------------------------------------------------------------------------
// IRTensor — typed, shaped data flowing between nodes
// ---------------------------------------------------------------------------
struct IRTensor {
  uint32_t        id;
  std::string     name;
  DType           dtype;
  Shape           shape;
  QuantParams     quant;
  MemoryLayout    layout   = MemoryLayout::Unknown;

  // Lifetime tracking (filled by TensorLifetimeAnalysisPass)
  int32_t         live_start = -1;   // op index where this tensor is produced
  int32_t         live_end   = -1;   // last op index that consumes it

  // Memory planning (filled by MemoryPlanner)
  uint64_t        mem_offset = 0;    // byte offset in memory arena
  uint32_t        mem_arena  = 0;    // 0=weights, 1=activations, 2=kv_cache

  // Hardware affinity (hints, can be overridden by compiler)
  ExecUnit        preferred_unit = ExecUnit::Auto;
  CachePolicy     cache_policy   = CachePolicy::Default;

  // Is this tensor a model weight (constant) or a runtime activation?
  bool            is_weight   = false;
  bool            is_input    = false;
  bool            is_output   = false;
  bool            is_kv_cache = false;

  // Raw weight data (for constant tensors only; nullptr otherwise)
  // Stored as the native dtype bytes; ownership is shared with IRGraph.
  const uint8_t*  weight_data = nullptr;
  size_t          weight_size = 0;   // bytes

  std::string toString() const;
};

// ---------------------------------------------------------------------------
// SchedulingHint — advice to the heterogeneous scheduler
// ---------------------------------------------------------------------------
struct SchedulingHint {
  bool can_parallelize   = false;  // Can be split across multiple cores
  bool latency_sensitive = true;   // On the critical path
  bool can_stream        = false;  // Supports chunked streaming execution
  bool is_memory_bound   = false;  // Memory-bandwidth limited (vs compute)
  int  preferred_threads = 0;      // 0 = auto
};

// ---------------------------------------------------------------------------
// IRNode — one operation in the computation graph
// ---------------------------------------------------------------------------
struct IRNode {
  uint32_t                     id;
  std::string                  name;
  OpCode                       op;

  // Data flow edges (tensor IDs)
  std::vector<uint32_t>        inputs;
  std::vector<uint32_t>        outputs;

  // Operation hyperparameters (e.g. kernel size, groups, axis)
  std::vector<Attribute>       attrs;

  // ── ARM-IR rich metadata ───────────────────────────────────────────────────

  // Hardware scheduling
  ExecUnit            preferred_unit  = ExecUnit::Auto;
  ExecUnit            assigned_unit   = ExecUnit::Auto;  // set by scheduler
  SchedulingHint      sched_hint;

  // Cost model cache (populated by CostModelPass)
  float               cost_cpu_ms     = -1.0f;
  float               cost_gpu_ms     = -1.0f;
  float               cost_npu_ms     = -1.0f;
  float               cost_dsp_ms     = -1.0f;
  float               cost_ane_ms     = -1.0f;

  // Thermal / power
  float               thermal_cost_mJ = 0.0f;   // millijoules per inference

  // Quantization
  // (quant metadata lives on the output tensors; this is just a flag)
  bool                is_quantized    = false;

  // Streaming / chunking
  bool                supports_streaming = false;
  int32_t             streaming_chunk_size = 0;  // 0 = no chunking

  // Graph region markers
  bool                in_npu_subgraph = false;
  bool                in_gpu_subgraph = false;
  bool                in_dsp_subgraph = false;

  // Fusion
  bool                is_fused        = false;   // Fused into adjacent node
  std::vector<uint32_t> fused_from;              // Node IDs that were fused in

  // Utility
  const Attribute*    attr(const std::string& name) const;
  std::string         toString() const;
};

} // namespace ir
} // namespace armcc
