// ============================================================================
// ARM AI Compiler — Memory Planner (implementation)
//
// Full memory planning pipeline:
//  1. Tensor lifetime analysis
//  2. Arena assignment (weights / activations / kv-cache / scratch)
//  3. Interval-graph coloring for activation aliasing
//  4. KV cache layout planning
//  5. Prefetch hint insertion
//  6. Zero-copy opportunity identification
// ============================================================================
#include "memory/memory_planner.h"
#include "arm_ir/ops.h"
#include <algorithm>
#include <sstream>
#include <cstring>
#include <limits>
#include <unordered_set>
#include <numeric>

namespace armcc {
namespace memory {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
MemoryPlanner::MemoryPlanner(const analysis::DeviceProfile* device)
  : device_(device) {}

// ---------------------------------------------------------------------------
// plan — run all stages
// ---------------------------------------------------------------------------
MemoryPlan MemoryPlanner::plan(ir::IRGraph& graph) {
  MemoryPlan result;

  // Stage 1: Lifetime analysis
  auto lifetimes = computeLifetimes(graph);

  // Stage 2: Assign arenas
  assignArenas(graph, lifetimes);

  // Stage 3: Aliasing (activation memory reuse)
  applyAliasing(graph, lifetimes);

  // Stage 4: KV cache layout
  planKVCache(graph, result.kv_layout);

  // Stage 5: Prefetch hints
  insertPrefetchHints(graph);

  // Stage 6: Zero-copy
  identifyZeroCopy(graph);

  // Build summary
  result.weights_arena_bytes     = graph.weight_bytes;
  result.activations_arena_bytes = graph.activation_bytes;
  result.kv_cache_bytes          = graph.kv_cache_bytes;
  result.total_bytes             = result.weights_arena_bytes
                                 + result.activations_arena_bytes
                                 + result.kv_cache_bytes
                                 + result.scratch_bytes;

  // Build block list
  for (auto& t : graph.tensors) {
    if (t->mem_offset == 0 && !t->is_weight && !t->is_input) continue;
    MemoryBlock b;
    b.offset  = t->mem_offset;
    b.size    = std::max(1ULL, (uint64_t)std::max(0LL, t->shape.numElements()));
    b.arena   = (ArenaType)t->mem_arena;
    b.alignment = requiredAlignment(*t);
    b.tenant_tensor_ids = {t->id};
    result.blocks.push_back(b);
  }

  return result;
}

// ---------------------------------------------------------------------------
// Stage 1: Tensor Lifetime Analysis
// ---------------------------------------------------------------------------
std::vector<TensorLifetime> MemoryPlanner::computeLifetimes(const ir::IRGraph& graph) {
  const_cast<ir::IRGraph&>(graph).sortTopological();
  const auto& order = graph.topoOrder;

  // Build: tensor_id → op index where produced
  std::unordered_map<uint32_t, int32_t> producedAt;
  std::unordered_map<uint32_t, int32_t> lastUsedAt;

  for (int32_t i = 0; i < (int32_t)order.size(); ++i) {
    auto* n = const_cast<ir::IRGraph&>(graph).getNode(order[i]);
    if (!n) continue;
    for (uint32_t tid : n->outputs) producedAt[tid] = i;
    for (uint32_t tid : n->inputs)  lastUsedAt[tid] = i;
  }

  std::vector<TensorLifetime> lifetimes;
  for (auto& t : graph.tensors) {
    TensorLifetime lt;
    lt.tensor_id = t->id;

    auto ip = producedAt.find(t->id);
    auto iu = lastUsedAt.find(t->id);
    lt.first_op = (ip != producedAt.end()) ? ip->second : 0;
    lt.last_op  = (iu != lastUsedAt.end()) ? iu->second : lt.first_op;

    // Compute size
    int64_t elems = t->shape.numElements();
    if (elems > 0) {
      size_t esz = ir::dtypeElementSize(t->dtype);
      if (esz == 0) esz = 1;
      lt.size_bytes = (uint64_t)elems * esz;
    } else {
      lt.size_bytes = 64; // Minimum allocation
    }

    // Annotate the tensor
    t->live_start = lt.first_op;
    t->live_end   = lt.last_op;
    lifetimes.push_back(lt);
  }

  return lifetimes;
}

bool TensorLifetime::overlapsWith(const TensorLifetime& other) const {
  return !(last_op < other.first_op || other.last_op < first_op);
}

// ---------------------------------------------------------------------------
// Stage 2: Arena Assignment
// ---------------------------------------------------------------------------
void MemoryPlanner::assignArenas(ir::IRGraph& graph,
                                  const std::vector<TensorLifetime>& lifetimes) {
  // Weights → arena 0
  for (auto& t : graph.tensors) {
    if (t->is_weight) {
      t->mem_arena  = 0;  // Weights arena
      t->is_weight  = true;
    } else if (t->is_kv_cache) {
      t->mem_arena = 2;   // KV cache arena
    } else {
      t->mem_arena = 1;   // Activations arena
    }
  }

  // Assign sequential offsets for weights (page-aligned for mmap)
  uint64_t weight_offset = 0;
  for (auto& t : graph.tensors) {
    if (!t->is_weight) continue;
    uint32_t align = requiredAlignment(*t);
    weight_offset = (weight_offset + align - 1) & ~(uint64_t)(align - 1);
    t->mem_offset = weight_offset;
    uint64_t sz = t->live_end == -1 ? 0 : 0; // Size computed in aliasing pass
    int64_t elems = t->shape.numElements();
    sz = (elems > 0)
      ? (static_cast<uint64_t>(elems) * std::max<size_t>(1, ir::dtypeElementSize(t->dtype)))
      : 64;
    weight_offset += sz;
  }
  graph.weight_bytes = weight_offset;
}

// ---------------------------------------------------------------------------
// Stage 3: Activation Aliasing (Greedy Interval-Graph Coloring)
// ---------------------------------------------------------------------------
void MemoryPlanner::applyAliasing(ir::IRGraph& graph,
                                   const std::vector<TensorLifetime>& lifetimes) {
  uint64_t arena_peak = assignOffsetsGreedy(lifetimes, graph, ArenaType::Activations);
  graph.activation_bytes = arena_peak;
}

uint64_t MemoryPlanner::assignOffsetsGreedy(
    const std::vector<TensorLifetime>& lifetimes,
    ir::IRGraph& graph,
    ArenaType arena)
{
  // Filter to activation tensors only
  std::vector<const TensorLifetime*> acts;
  for (auto& lt : lifetimes) {
    ir::IRTensor* t = graph.getTensor(lt.tensor_id);
    if (!t || t->mem_arena != (uint32_t)arena) continue;
    if (lt.size_bytes == 0) continue;
    acts.push_back(&lt);
  }

  // Sort by live_start (earliest first)
  std::sort(acts.begin(), acts.end(), [](auto* a, auto* b) {
    return a->first_op < b->first_op;
  });

  // Greedy interval-graph coloring with free list
  struct FreeSlot {
    uint64_t offset;
    uint64_t size;
    int32_t  freed_at; // op index when this slot is free again
  };

  std::vector<FreeSlot> freeList;
  uint64_t arena_top = 0;

  for (const TensorLifetime* lt : acts) {
    ir::IRTensor* t = graph.getTensor(lt->tensor_id);
    if (!t) continue;

    uint32_t align  = requiredAlignment(*t);
    uint64_t needed = (lt->size_bytes + align - 1) & ~(uint64_t)(align - 1);

    // Find a free slot that's big enough and already freed
    FreeSlot* best = nullptr;
    for (auto& s : freeList) {
      if (s.freed_at <= lt->first_op && s.size >= needed) {
        if (!best || s.size < best->size) best = &s;  // Best-fit
      }
    }

    if (best) {
      t->mem_offset  = best->offset;
      best->freed_at = lt->last_op + 1;
      best->size     = std::max(best->size, needed); // May grow slot
    } else {
      // Align arena_top
      arena_top = (arena_top + align - 1) & ~(uint64_t)(align - 1);
      t->mem_offset  = arena_top;
      FreeSlot ns;
      ns.offset   = arena_top;
      ns.size     = needed;
      ns.freed_at = lt->last_op + 1;
      freeList.push_back(ns);
      arena_top  += needed;
    }
  }

  return arena_top;
}

// ---------------------------------------------------------------------------
// Stage 4: KV Cache Layout
// ---------------------------------------------------------------------------
void MemoryPlanner::planKVCache(ir::IRGraph& graph, KVCacheLayout& layout) {
  if (!graph.kv_cache.enabled) return;

  layout.paged        = false;  // Simple flat layout by default
  layout.num_layers   = graph.num_layers;
  layout.num_kv_heads = graph.kv_cache.num_kv_heads;
  layout.head_dim     = graph.kv_cache.head_dim;
  layout.max_seq_len  = graph.kv_cache.max_seq_len;
  layout.dtype        = graph.kv_cache.dtype;

  // Compute bytes per layer
  // Layout: [num_kv_heads, max_seq, head_dim] per key and value
  uint64_t dtype_bytes = (layout.dtype == ir::DType::F16) ? 2 : 4;
  uint64_t per_layer_kv = (uint64_t)layout.num_kv_heads
                        * layout.max_seq_len
                        * layout.head_dim
                        * dtype_bytes;

  layout.key_cache_bytes = per_layer_kv * layout.num_layers;
  layout.val_cache_bytes = per_layer_kv * layout.num_layers;
  layout.total_bytes     = layout.key_cache_bytes + layout.val_cache_bytes;

  // Compute per-layer offsets
  layout.layer_offsets_key.resize(layout.num_layers);
  layout.layer_offsets_val.resize(layout.num_layers);
  for (uint32_t i = 0; i < layout.num_layers; ++i) {
    layout.layer_offsets_key[i] = i * per_layer_kv;
    layout.layer_offsets_val[i] = layout.key_cache_bytes + i * per_layer_kv;
  }

  graph.kv_cache_bytes = layout.total_bytes;

  // For paged attention (large context lengths): use page-based layout
  if (layout.max_seq_len > 8192) {
    layout.paged = true;
    layout.page_size_tokens = 16;
  }
}

// ---------------------------------------------------------------------------
// Stage 5: Prefetch Insertion
// ---------------------------------------------------------------------------
void MemoryPlanner::insertPrefetchHints(ir::IRGraph& graph) {
  for (auto& n : graph.nodes) {
    // Mark weight tensors of heavy compute ops for prefetch
    if (n->op == ir::OpCode::MatMul   ||
        n->op == ir::OpCode::GEMM     ||
        n->op == ir::OpCode::BatchMatMul ||
        n->op == ir::OpCode::GroupQueryAttention ||
        n->op == ir::OpCode::FlashAttention) {
      for (uint32_t tid : n->inputs) {
        ir::IRTensor* t = graph.getTensor(tid);
        if (t && t->is_weight) {
          t->cache_policy = ir::CachePolicy::Prefetch;
        }
      }
    }
    // Pin small matrices that are reused across layers (e.g. RoPE tables)
    if (n->op == ir::OpCode::RopeEmbedding) {
      for (uint32_t tid : n->inputs) {
        ir::IRTensor* t = graph.getTensor(tid);
        if (t && t->is_weight) {
          t->cache_policy = ir::CachePolicy::Pin;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Stage 6: Zero-Copy Identification
// ---------------------------------------------------------------------------
void MemoryPlanner::identifyZeroCopy(ir::IRGraph& graph) {
  // Zero-copy: if a Reshape/Permute/Transpose has a single consumer
  // and its output's memory layout is compatible → alias the tensors.
  for (auto& n : graph.nodes) {
    if (n->op != ir::OpCode::Reshape &&
        n->op != ir::OpCode::Squeeze &&
        n->op != ir::OpCode::Unsqueeze) continue;
    if (n->inputs.size() != 1 || n->outputs.size() != 1) continue;

    ir::IRTensor* inp = graph.getTensor(n->inputs[0]);
    ir::IRTensor* out = graph.getTensor(n->outputs[0]);
    if (!inp || !out) continue;
    if (inp->is_weight) continue;  // Can't alias weight tensors

    // If same number of elements → the reshape is a view → zero-copy safe
    int64_t in_elems  = inp->shape.numElements();
    int64_t out_elems = out->shape.numElements();
    if (in_elems > 0 && out_elems > 0 && in_elems == out_elems) {
      // Alias: point output to same memory as input
      out->mem_offset = inp->mem_offset;
      out->mem_arena  = inp->mem_arena;
    }
  }
}

// ---------------------------------------------------------------------------
// requiredAlignment
// ---------------------------------------------------------------------------
uint32_t MemoryPlanner::requiredAlignment(const ir::IRTensor& t) const {
  // Weights: 4KB page alignment for mmap
  if (t.is_weight) return 4096;

  // SIMD: 64-byte alignment for NEON/SVE
  if (device_ && (device_->has_neon || device_->has_sve)) return 64;

  // GPU/NPU: 256-byte alignment for DMA
  if (device_ && (device_->has_gpu || device_->has_npu)) return 256;

  return 16;  // Minimum ARM alignment
}

// ---------------------------------------------------------------------------
// MemoryPlan::toString
// ---------------------------------------------------------------------------
std::string MemoryPlan::toString() const {
  std::ostringstream oss;
  oss << "MemoryPlan{"
      << " weights="     << weights_arena_bytes / 1024 / 1024 << "MB"
      << " activations=" << activations_arena_bytes / 1024 << "KB"
      << " kv_cache="    << kv_cache_bytes / 1024 / 1024 << "MB"
      << " total="       << total_bytes / 1024 / 1024 << "MB"
      << " blocks="      << blocks.size()
      << "}";
  return oss.str();
}

} // namespace memory
} // namespace armcc
