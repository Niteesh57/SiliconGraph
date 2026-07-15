// ============================================================================
// ARM AI Compiler — KV Cache Layout Planner
//
// Plans the memory layout for the Key-Value cache used in LLM inference.
// Supports both flat layout (small context) and paged layout (large context).
// ============================================================================
#include "memory/memory_planner.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <sstream>

namespace armcc {
namespace memory {

// ---------------------------------------------------------------------------
// Flat KV cache layout:
//   [num_layers][2 (K/V)][num_kv_heads][max_seq_len][head_dim]
// ---------------------------------------------------------------------------
KVCacheLayout planFlatLayout(
    uint32_t num_layers,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t max_seq_len,
    ir::DType dtype)
{
  KVCacheLayout layout;
  layout.paged         = false;
  layout.num_layers    = num_layers;
  layout.num_kv_heads  = num_kv_heads;
  layout.head_dim      = head_dim;
  layout.max_seq_len   = max_seq_len;
  layout.dtype         = dtype;

  uint64_t dtype_bytes = (dtype == ir::DType::F16) ? 2 :
                         (dtype == ir::DType::F32) ? 4 : 1;

  // Per-layer: num_kv_heads × max_seq_len × head_dim × dtype_bytes
  uint64_t per_layer = (uint64_t)num_kv_heads * max_seq_len * head_dim * dtype_bytes;

  layout.key_cache_bytes = per_layer * num_layers;
  layout.val_cache_bytes = per_layer * num_layers;
  layout.total_bytes     = layout.key_cache_bytes + layout.val_cache_bytes;

  layout.layer_offsets_key.resize(num_layers);
  layout.layer_offsets_val.resize(num_layers);

  for (uint32_t i = 0; i < num_layers; ++i) {
    layout.layer_offsets_key[i] = i * per_layer;
    layout.layer_offsets_val[i] = layout.key_cache_bytes + i * per_layer;
  }

  return layout;
}

// ---------------------------------------------------------------------------
// Paged KV cache layout:
//   Tokens are grouped into fixed-size pages.
//   Enables dynamic context extension beyond max_seq_len.
//
//   Physical layout:
//     page_pool[total_pages][2][num_kv_heads][page_size][head_dim]
//
//   Page table: maps (layer, logical_page) → physical page index
// ---------------------------------------------------------------------------
KVCacheLayout planPagedLayout(
    uint32_t num_layers,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t max_seq_len,
    ir::DType dtype,
    uint32_t page_size_tokens)
{
  KVCacheLayout layout = planFlatLayout(
      num_layers, num_kv_heads, head_dim, max_seq_len, dtype);

  layout.paged            = true;
  layout.page_size_tokens = page_size_tokens;

  // Number of pages per layer
  uint32_t pages_per_layer = (max_seq_len + page_size_tokens - 1) / page_size_tokens;
  uint64_t dtype_bytes     = (dtype == ir::DType::F16) ? 2 :
                             (dtype == ir::DType::F32) ? 4 : 1;

  // Physical page size = 2 (K+V) × num_kv_heads × page_size_tokens × head_dim
  uint64_t page_bytes = 2ULL * num_kv_heads * page_size_tokens * head_dim * dtype_bytes;

  // Page pool: enough pages for max_seq_len × num_layers
  uint32_t total_pages = pages_per_layer * num_layers;
  layout.total_bytes   = page_bytes * total_pages;

  // Rebuild layer offsets for paged layout
  layout.layer_offsets_key.resize(num_layers);
  layout.layer_offsets_val.resize(num_layers);
  for (uint32_t i = 0; i < num_layers; ++i) {
    // For paged layout, offset is the start of this layer's page table entry
    layout.layer_offsets_key[i] = i * pages_per_layer * (page_bytes / 2);
    layout.layer_offsets_val[i] = layout.total_bytes / 2 +
                                   i * pages_per_layer * (page_bytes / 2);
  }

  return layout;
}

// ---------------------------------------------------------------------------
// recommendLayout — picks flat vs paged based on context length
// ---------------------------------------------------------------------------
KVCacheLayout recommendKVCacheLayout(
    const ir::IRGraph& graph,
    uint32_t max_seq_len,
    const analysis::DeviceProfile* device)
{
  uint32_t num_layers   = std::max(1u, graph.num_layers);
  uint32_t num_kv_heads = (graph.kv_cache.num_kv_heads > 0)
                          ? graph.kv_cache.num_kv_heads
                          : std::max(1u, graph.num_heads);
  uint32_t head_dim     = (graph.kv_cache.head_dim > 0)
                          ? graph.kv_cache.head_dim
                          : (graph.hidden_size > 0 && graph.num_heads > 0
                             ? graph.hidden_size / graph.num_heads : 64);
  ir::DType dtype = ir::DType::F16;

  // Use paged layout for long contexts (>4096 tokens)
  if (max_seq_len > 4096) {
    uint32_t page_size = 16;  // 16 tokens/page is standard
    return planPagedLayout(num_layers, num_kv_heads, head_dim,
                           max_seq_len, dtype, page_size);
  }

  return planFlatLayout(num_layers, num_kv_heads, head_dim, max_seq_len, dtype);
}

// ---------------------------------------------------------------------------
// KVCacheLayout: toJSON
// ---------------------------------------------------------------------------
std::string KVCacheLayout::toJSON() const {
  nlohmann::json j;
  j["paged"]             = paged;
  j["num_layers"]        = num_layers;
  j["num_kv_heads"]      = num_kv_heads;
  j["head_dim"]          = head_dim;
  j["max_seq_len"]       = max_seq_len;
  j["dtype"]             = ir::dtypeToString(dtype);
  j["page_size_tokens"]  = page_size_tokens;
  j["key_cache_bytes"]   = key_cache_bytes;
  j["val_cache_bytes"]   = val_cache_bytes;
  j["total_bytes"]       = total_bytes;
  j["layer_offsets_key"] = layer_offsets_key;
  j["layer_offsets_val"] = layer_offsets_val;
  return j.dump(2);
}

} // namespace memory
} // namespace armcc
