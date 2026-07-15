// ============================================================================
// ARM AI Compiler — Memory Passes (Lifetime + Aliasing + Layout + etc.)
// ============================================================================
#include "passes/pass_manager.h"
#include "memory/memory_planner.h"
#include "arm_ir/graph.h"
#include <algorithm>
#include <unordered_map>

namespace armcc {
namespace passes {

namespace {

// ---------------------------------------------------------------------------
// Tensor Lifetime Analysis Pass
// ---------------------------------------------------------------------------
class TensorLifetimeAnalysisPass : public Pass {
public:
  TensorLifetimeAnalysisPass() : Pass("TensorLifetimeAnalysis") {}

  PassResult run(ir::IRGraph& graph, const PassOptions& opts) override {
    PassResult result;
    result.pass_name = name();
    graph.sortTopological();

    // For each tensor, find the op index where it's produced and last consumed
    // topoOrder[i] gives us op indices 0..N-1
    std::unordered_map<uint32_t, int32_t> producedAt;  // tensor_id → op index
    std::unordered_map<uint32_t, int32_t> lastUsedAt;  // tensor_id → op index

    for (int32_t i = 0; i < (int32_t)graph.topoOrder.size(); ++i) {
      uint32_t nid = graph.topoOrder[i];
      ir::IRNode* n = graph.getNode(nid);
      if (!n) continue;

      for (uint32_t tid : n->outputs) {
        producedAt[tid] = i;
      }
      for (uint32_t tid : n->inputs) {
        lastUsedAt[tid] = i;  // overwritten with later uses
      }
    }

    // Apply to tensors
    for (auto& t : graph.tensors) {
      auto it_p = producedAt.find(t->id);
      auto it_l = lastUsedAt.find(t->id);
      if (it_p != producedAt.end()) t->live_start = it_p->second;
      if (it_l != lastUsedAt.end()) t->live_end   = it_l->second;
      result.changed = true;
    }

    result.success = true;
    return result;
  }
};

// ---------------------------------------------------------------------------
// Memory Aliasing Pass
// (Interval-graph coloring: tensors with non-overlapping lifetimes share memory)
// ---------------------------------------------------------------------------
class MemoryAliasingPass : public Pass {
public:
  MemoryAliasingPass() : Pass("MemoryAliasing") {}

  PassResult run(ir::IRGraph& graph, const PassOptions& opts) override {
    PassResult result;
    result.pass_name = name();

    // Collect activation tensors (not weights, not KV cache)
    std::vector<ir::IRTensor*> acts;
    for (auto& t : graph.tensors) {
      if (!t->is_weight && !t->is_kv_cache &&
          t->live_start >= 0 && t->live_end >= 0) {
        acts.push_back(t.get());
      }
    }

    // Greedy interval-graph coloring
    // Assign each tensor to the earliest freed "color" (memory slot)
    struct Slot {
      uint64_t size     = 0;
      int32_t  free_at  = -1;  // op index when this slot becomes free
      uint64_t offset   = 0;
    };

    std::vector<Slot> slots;
    uint64_t arena_top = 0;

    // Sort by live_start
    std::sort(acts.begin(), acts.end(), [](auto* a, auto* b) {
      return a->live_start < b->live_start;
    });

    for (ir::IRTensor* t : acts) {
      int64_t nelems = t->shape.numElements();
      uint64_t sz = (nelems > 0) ? ((uint64_t)nelems * 2) : 64; // assume F16
      if (ir::dtypeElementSize(t->dtype) > 0)
        sz = (nelems > 0) ? ((uint64_t)nelems * ir::dtypeElementSize(t->dtype)) : 64;

      // Find a freed slot big enough
      Slot* best = nullptr;
      for (auto& s : slots) {
        if (s.free_at <= t->live_start && s.size >= sz) {
          best = &s;
          break;
        }
      }

      if (best) {
        t->mem_offset = best->offset;
        t->mem_arena  = 1; // activations
        best->free_at = t->live_end;
        result.changed = true;
      } else {
        // Allocate new slot
        Slot ns;
        // Align to 64 bytes
        ns.offset  = (arena_top + 63) & ~uint64_t(63);
        ns.size    = sz;
        ns.free_at = t->live_end;
        t->mem_offset = ns.offset;
        t->mem_arena  = 1;
        arena_top = ns.offset + sz;
        slots.push_back(ns);
        result.changed = true;
      }
    }

    graph.activation_bytes = arena_top;
    result.success = true;
    return result;
  }
};

// ---------------------------------------------------------------------------
// Memory Layout Pass
// ---------------------------------------------------------------------------
class MemoryLayoutPass : public Pass {
public:
  MemoryLayoutPass() : Pass("MemoryLayout") {}
  PassResult run(ir::IRGraph& graph, const PassOptions& opts) override {
    PassResult r; r.pass_name = name(); r.success = true; return r;
  }
};

// ---------------------------------------------------------------------------
// Prefetch Insertion Pass
// ---------------------------------------------------------------------------
class PrefetchInsertionPass : public Pass {
public:
  PrefetchInsertionPass() : Pass("PrefetchInsertion") {}
  PassResult run(ir::IRGraph& graph, const PassOptions& opts) override {
    PassResult result;
    result.pass_name = name();
    // Insert DMA_Prefetch nodes before heavy MatMul / Attention ops
    for (auto& n : graph.nodes) {
      if (n->op == ir::OpCode::MatMul || n->op == ir::OpCode::GEMM ||
          n->op == ir::OpCode::FlashAttention) {
        for (uint32_t tid : n->inputs) {
          ir::IRTensor* t = graph.getTensor(tid);
          if (t) t->cache_policy = ir::CachePolicy::Prefetch;
        }
        result.changed = true;
      }
    }
    result.success = true;
    return result;
  }
};

// ---------------------------------------------------------------------------
// Zero Copy Pass
// ---------------------------------------------------------------------------
class ZeroCopyPass : public Pass {
public:
  ZeroCopyPass() : Pass("ZeroCopy") {}
  PassResult run(ir::IRGraph& graph, const PassOptions& opts) override {
    PassResult r; r.pass_name = name(); r.success = true; return r;
  }
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------
std::unique_ptr<Pass> makeTensorLifetimeAnalysisPass() {
  return std::make_unique<TensorLifetimeAnalysisPass>();
}
std::unique_ptr<Pass> makeMemoryAliasingPass() {
  return std::make_unique<MemoryAliasingPass>();
}
std::unique_ptr<Pass> makeMemoryLayoutPass() {
  return std::make_unique<MemoryLayoutPass>();
}
std::unique_ptr<Pass> makePrefetchInsertionPass() {
  return std::make_unique<PrefetchInsertionPass>();
}
std::unique_ptr<Pass> makeZeroCopyPass() {
  return std::make_unique<ZeroCopyPass>();
}

} // namespace passes
} // namespace armcc
