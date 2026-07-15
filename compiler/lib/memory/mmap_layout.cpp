// ============================================================================
// ARM AI Compiler — Memory-Mapped Layout (mmap-safe weight file layout)
//
// Generates a weight file layout where every tensor is:
//   1. Page-aligned (4096 bytes) for zero-copy mmap()
//   2. Described by a byte-offset map for fast random access
//   3. Packed contiguously with no gaps beyond alignment padding
// ============================================================================
#include "memory/memory_planner.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <sstream>

namespace armcc {
namespace memory {

// ---------------------------------------------------------------------------
// MmapLayoutEntry: Offset + size for one tensor in the mmap file
// ---------------------------------------------------------------------------
struct MmapLayoutEntry {
  uint32_t tensor_id;
  std::string tensor_name;
  uint64_t offset;      // Byte offset in the weight file
  uint64_t size_bytes;  // Byte length
  std::string dtype;
  std::vector<int64_t> shape;
  bool is_quantized;
  std::string quant_dtype;
  int32_t quant_group_size;
};

// ---------------------------------------------------------------------------
// MmapLayout: Full layout description for the weight blob
// ---------------------------------------------------------------------------
struct MmapLayout {
  uint64_t total_bytes;
  uint32_t page_size;
  std::vector<MmapLayoutEntry> entries;

  std::string toJSON() const {
    nlohmann::json j;
    j["total_bytes"] = total_bytes;
    j["page_size"]   = page_size;
    auto arr = nlohmann::json::array();
    for (auto& e : entries) {
      nlohmann::json je;
      je["tensor_id"]     = e.tensor_id;
      je["name"]          = e.tensor_name;
      je["offset"]        = e.offset;
      je["size_bytes"]    = e.size_bytes;
      je["dtype"]         = e.dtype;
      je["shape"]         = e.shape;
      je["is_quantized"]  = e.is_quantized;
      je["quant_dtype"]   = e.quant_dtype;
      je["quant_group"]   = e.quant_group_size;
      arr.push_back(je);
    }
    j["tensors"] = arr;
    return j.dump(2);
  }
};

// ---------------------------------------------------------------------------
// computeMmapLayout — compute the page-aligned layout for all weight tensors
// ---------------------------------------------------------------------------
MmapLayout computeMmapLayout(const ir::IRGraph& graph,
                              uint32_t page_size = 4096)
{
  MmapLayout layout;
  layout.page_size = page_size;
  uint64_t offset  = 0;

  // Sort weight tensors by size (largest first for better alignment efficiency)
  std::vector<ir::IRTensor*> weights;
  for (auto& t : graph.tensors) {
    if (t->is_weight) weights.push_back(t.get());
  }
  std::sort(weights.begin(), weights.end(), [](auto* a, auto* b) {
    return a->shape.numElements() > b->shape.numElements();
  });

  for (auto* t : weights) {
    // Align to page boundary
    offset = (offset + page_size - 1) & ~(uint64_t)(page_size - 1);

    // Compute tensor byte size
    int64_t elems = t->shape.numElements();
    uint64_t sz   = 0;
    if (elems > 0) {
      if (t->quant.scheme != ir::QuantScheme::None) {
        // Quantized: INT4 = 0.5 byte/elem, INT8 = 1 byte/elem
        if (t->quant.stored_as == ir::DType::I4) {
          sz = ((uint64_t)elems + 1) / 2;  // 2 elements per byte
        } else {
          size_t esz = ir::dtypeElementSize(t->quant.stored_as);
          sz = (uint64_t)elems * (esz > 0 ? esz : 1);
        }
        // Add scale/zero-point overhead: (elems / group_size) × 6 bytes (F16 scale + i8 zp)
        int gs = t->quant.group_size > 0 ? t->quant.group_size : 128;
        uint64_t num_groups = ((uint64_t)elems + gs - 1) / gs;
        sz += num_groups * 6;  // F16 scale (2B) + i8 zp (1B) + padding (3B)
      } else {
        size_t esz = ir::dtypeElementSize(t->dtype);
        sz = (uint64_t)elems * (esz > 0 ? esz : 2);  // Default F16
      }
    }
    if (sz == 0) sz = page_size;  // Minimum 1 page

    MmapLayoutEntry e;
    e.tensor_id       = t->id;
    e.tensor_name     = t->name;
    e.offset          = offset;
    e.size_bytes      = sz;
    e.dtype           = ir::dtypeToString(t->dtype);
    e.shape           = t->shape.dims;
    e.is_quantized    = t->quant.scheme != ir::QuantScheme::None;
    e.quant_dtype     = ir::dtypeToString(t->quant.stored_as);
    e.quant_group_size = t->quant.group_size;

    // Write back to tensor
    t->mem_offset = offset;

    layout.entries.push_back(e);
    offset += sz;
  }

  // Final size aligned to page
  layout.total_bytes = (offset + page_size - 1) & ~(uint64_t)(page_size - 1);
  return layout;
}

} // namespace memory
} // namespace armcc
