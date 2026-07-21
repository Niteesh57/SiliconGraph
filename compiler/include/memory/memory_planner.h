// ============================================================================
// ARM AI Compiler — Memory Planner
//
// Treats memory like a database query optimizer treats a query plan.
// Instead of allocating tensors naively, the MemoryPlanner:
//
//   1. Runs tensor lifetime analysis to know exactly when each tensor
//      is "live" (produced → last consumed)
//   2. Assigns tensors to memory arenas:
//        Arena 0: Weights (static, mmap-able, page-aligned)
//        Arena 1: Activations (dynamic slab, reused across layers)
//        Arena 2: KV cache (ring buffer or paged)
//   3. Applies memory aliasing (two non-overlapping tensors share memory)
//   4. Optimizes layout for cache lines, NUMA, DMA alignment
//   5. Inserts prefetch hints before compute-heavy ops
//   6. Identifies zero-copy opportunities
// ============================================================================
#pragma once

#include "arm_ir/graph.h"
#include "analysis/cost_model.h"

#include <vector>
#include <string>
#include <unordered_map>

namespace armcc {
namespace memory {

// ---------------------------------------------------------------------------
// Memory arena types
// ---------------------------------------------------------------------------
enum class ArenaType : uint8_t {
  Weights    = 0,   // Static, mmap-able, page-aligned (4KB)
  Activations = 1,  // Dynamic slab, allocated once, reused per layer
  KVCache    = 2,   // KV cache (ring buffer or paged attention)
  Scratch    = 3,   // Temporary scratch space (for NPU/DSP transfers)
};

// ---------------------------------------------------------------------------
// TensorLifetime — [first_use, last_use] op indices
// ---------------------------------------------------------------------------
struct TensorLifetime {
  uint32_t tensor_id;
  int32_t  first_op = -1;   // Op index where this tensor is produced
  int32_t  last_op  = -1;   // Last op index that consumes this tensor
  uint64_t size_bytes = 0;

  bool overlapsWith(const TensorLifetime& other) const;
};

// ---------------------------------------------------------------------------
// MemoryBlock — a contiguous chunk in an arena
// ---------------------------------------------------------------------------
struct MemoryBlock {
  uint64_t offset;          // Bytes from arena start
  uint64_t size;            // Bytes
  uint32_t alignment;       // Required alignment (e.g. 64 bytes for SIMD)
  ArenaType arena;
  std::vector<uint32_t> tenant_tensor_ids;  // May be shared via aliasing
};

// ---------------------------------------------------------------------------
// KV Cache layout
// ---------------------------------------------------------------------------
struct KVCacheLayout {
  bool      paged            = false;
  uint32_t  num_layers       = 0;
  uint32_t  num_kv_heads     = 0;
  uint32_t  head_dim         = 0;
  uint32_t  max_seq_len      = 0;
  uint32_t  page_size_tokens = 16;    // For paged attention
  ir::DType dtype            = ir::DType::F16;

  uint64_t  key_cache_bytes  = 0;
  uint64_t  val_cache_bytes  = 0;
  uint64_t  total_bytes      = 0;

  // Flat layout: [layer][head][seq][dim] packed as a 4D tensor
  std::vector<uint64_t> layer_offsets_key;  // per-layer byte offsets
  std::vector<uint64_t> layer_offsets_val;

  std::string toJSON() const;
};

// ---------------------------------------------------------------------------
// Memory plan summary
// ---------------------------------------------------------------------------
struct MemoryPlan {
  // Arena sizes
  uint64_t weights_arena_bytes     = 0;
  uint64_t activations_arena_bytes = 0;   // Peak usage
  uint64_t kv_cache_bytes          = 0;
  uint64_t scratch_bytes           = 0;
  uint64_t total_bytes             = 0;

  // All memory blocks
  std::vector<MemoryBlock> blocks;

  // KV cache layout
  KVCacheLayout kv_layout;

  // Aliasing groups (tensor IDs that share a block)
  std::vector<std::vector<uint32_t>> aliasing_groups;

  // Zero-copy pairs (tensor_id → aliases tensor_id without a copy)
  std::vector<std::pair<uint32_t, uint32_t>> zero_copy_pairs;

  // Savings summary
  uint64_t bytes_saved_aliasing   = 0;
  uint64_t bytes_saved_zero_copy  = 0;

  std::string toString() const;
};

// ---------------------------------------------------------------------------
// MemoryPlanner
// ---------------------------------------------------------------------------
class MemoryPlanner {
public:
  explicit MemoryPlanner(const analysis::DeviceProfile* device = nullptr);

  // Run the full memory planning pipeline
  MemoryPlan plan(ir::IRGraph& graph);

  // Individual stages (called by plan(), can be run independently)
  std::vector<TensorLifetime> computeLifetimes(const ir::IRGraph& graph);
  void assignArenas(ir::IRGraph& graph,
                    const std::vector<TensorLifetime>& lifetimes);
  void applyAliasing(ir::IRGraph& graph,
                     const std::vector<TensorLifetime>& lifetimes);
  void planKVCache(ir::IRGraph& graph, KVCacheLayout& layout);
  void insertPrefetchHints(ir::IRGraph& graph);
  void identifyZeroCopy(ir::IRGraph& graph);

  // Alignment requirement for a tensor (dtype + target specific)
  uint32_t requiredAlignment(const ir::IRTensor& t) const;

private:
  const analysis::DeviceProfile* device_;

  // Simple interval-graph coloring for arena allocation
  uint64_t assignOffsetsGreedy(
    const std::vector<TensorLifetime>& lifetimes,
    ir::IRGraph& graph,
    ArenaType arena
  );
};

} // namespace memory
} // namespace armcc
