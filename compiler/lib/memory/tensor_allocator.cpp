// ============================================================================
// ARM AI Compiler — Tensor Allocator
// Low-level arena-based allocator for activation tensors.
// ============================================================================
#include "memory/memory_planner.h"
#include <cstring>
#include <stdexcept>

namespace armcc {
namespace memory {

// ---------------------------------------------------------------------------
// TensorArena: Fixed-size arena
// ---------------------------------------------------------------------------
class TensorArena {
public:
  explicit TensorArena(size_t capacity, ArenaType type)
    : capacity_(capacity), type_(type) {
    data_.resize(capacity, 0);
  }

  uint64_t allocate(size_t size, uint32_t alignment) {
    // Align current offset
    uint64_t aligned = (current_ + alignment - 1) & ~(uint64_t)(alignment - 1);
    if (aligned + size > capacity_) {
      throw std::bad_alloc();
    }
    current_ = aligned + size;
    return aligned;
  }

  void reset() { current_ = 0; }
  size_t used()     const { return current_; }
  size_t remaining() const { return capacity_ - current_; }
  ArenaType type()   const { return type_; }

  void* at(uint64_t offset) { return data_.data() + offset; }

private:
  size_t    capacity_;
  uint64_t  current_ = 0;
  ArenaType type_;
  std::vector<uint8_t> data_;
};

// ---------------------------------------------------------------------------
// TensorAllocator: Higher-level allocator that manages multiple arenas
// ---------------------------------------------------------------------------
class TensorAllocator {
public:
  TensorAllocator(size_t weight_capacity,
                  size_t activation_capacity,
                  size_t kv_cache_capacity)
    : weights_(weight_capacity, ArenaType::Weights)
    , activations_(activation_capacity, ArenaType::Activations)
    , kv_cache_(kv_cache_capacity, ArenaType::KVCache)
  {}

  uint64_t allocateWeight(size_t size, uint32_t align = 4096) {
    return weights_.allocate(size, align);
  }

  uint64_t allocateActivation(size_t size, uint32_t align = 64) {
    return activations_.allocate(size, align);
  }

  uint64_t allocateKVCache(size_t size, uint32_t align = 64) {
    return kv_cache_.allocate(size, align);
  }

  void resetActivations() { activations_.reset(); }

  size_t weightBytes()     const { return weights_.used(); }
  size_t activationBytes() const { return activations_.used(); }
  size_t kvCacheBytes()    const { return kv_cache_.used(); }
  size_t totalBytes()      const {
    return weightBytes() + activationBytes() + kvCacheBytes();
  }

  void* weightData()     { return weights_.at(0); }
  void* activationData() { return activations_.at(0); }
  void* kvCacheData()    { return kv_cache_.at(0); }

private:
  TensorArena weights_;
  TensorArena activations_;
  TensorArena kv_cache_;
};

} // namespace memory
} // namespace armcc
