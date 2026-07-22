// ============================================================================
// ARM AI Compiler — ARM Intermediate Representation: Graph
//
// IRGraph is the top-level container for the computation graph.
// It owns all nodes and tensors, and carries model-level metadata.
// ============================================================================
#pragma once

#include "arm_ir/node.h"
#include "arm_ir/types.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>

namespace armcc {
namespace ir {

// ---------------------------------------------------------------------------
// Model architecture family — used to select appropriate optimization passes
// ---------------------------------------------------------------------------
enum class ModelFamily : uint8_t {
  Unknown = 0,
  LlamaStyle,   // Llama 2/3, Mistral, Phi-3, Qwen2: RMSNorm + SwiGLU + GQA
  GemmaStyle,   // Gemma: RMSNorm + GeLU gating
  GPT2Style,    // GPT-2, SmolLM: LayerNorm + GeLU + MHA
  T5Style,      // Encoder-decoder
  MoE,          // Mixture of Experts (Mixtral)
  VisionLM,     // Multimodal (image + text)
  AudioLM,      // Audio + text generation or speech-to-text
  AudioGeneration, // Text-to-audio / speech synthesis
  ImageGeneration, // Text/image-to-image generation
  Embedding,    // Text/image/audio encoder producing vectors
  Custom,
};

const char* modelFamilyToString(ModelFamily f);

// ---------------------------------------------------------------------------
// KV Cache configuration (filled by KVCachePlanningPass)
// ---------------------------------------------------------------------------
struct KVCacheConfig {
  bool     enabled        = false;
  uint32_t num_heads      = 0;
  uint32_t head_dim       = 0;
  uint32_t num_kv_heads   = 0;  // For GQA: < num_heads
  uint32_t max_seq_len    = 0;
  DType    dtype          = DType::F16;

  // Memory layout chosen by MemoryPlanner
  uint64_t  total_bytes   = 0;
  bool      paged         = false;   // Paged attention KV cache
  uint32_t  page_size     = 0;       // tokens per page
};

// ---------------------------------------------------------------------------
// IRGraph — the main graph structure
// ---------------------------------------------------------------------------
class IRGraph {
public:
  // ── Identity ──────────────────────────────────────────────────────────────
  std::string   name;
  std::string   model_id;       // HuggingFace model ID, e.g. "meta-llama/..."
  ModelFamily   model_family = ModelFamily::Unknown;
  std::string   source_framework = "pytorch";

  // ── Topology ──────────────────────────────────────────────────────────────
  std::vector<std::unique_ptr<IRNode>>   nodes;
  std::vector<std::unique_ptr<IRTensor>> tensors;

  // Fast lookup
  std::unordered_map<uint32_t, IRNode*>   nodeById;
  std::unordered_map<uint32_t, IRTensor*> tensorById;
  std::unordered_map<std::string, uint32_t> tensorByName;

  // Input / output tensor IDs
  std::vector<uint32_t> inputIds;
  std::vector<uint32_t> outputIds;

  // Topological order of node IDs (computed by sortTopological)
  std::vector<uint32_t> topoOrder;

  // ── Model metadata ────────────────────────────────────────────────────────
  uint32_t  num_layers      = 0;
  uint32_t  hidden_size     = 0;
  uint32_t  num_heads       = 0;
  uint32_t  num_kv_heads    = 0;
  uint32_t  intermediate_sz = 0;
  uint32_t  vocab_size      = 0;
  uint32_t  max_position_embeddings = 0;

  // ── KV cache ──────────────────────────────────────────────────────────────
  KVCacheConfig kv_cache;

  // ── Memory summary (filled by MemoryPlanner) ──────────────────────────────
  uint64_t  weight_bytes      = 0;   // Total static weight size
  uint64_t  activation_bytes  = 0;   // Peak activation memory needed
  uint64_t  kv_cache_bytes    = 0;   // KV cache memory

  // ── Compiler pass audit trail ─────────────────────────────────────────────
  std::vector<std::string> applied_passes;

  // ── Construction helpers ──────────────────────────────────────────────────
  IRNode*   addNode(std::unique_ptr<IRNode> n);
  IRTensor* addTensor(std::unique_ptr<IRTensor> t);
  IRNode*   getNode(uint32_t id);
  IRTensor* getTensor(uint32_t id);
  IRTensor* getTensorByName(const std::string& name);

  // Remove a node and re-wire its consumers to its inputs (DCE helper)
  bool      removeNode(uint32_t nodeId);

  // Replace one tensor with another across all edges
  void      replaceTensor(uint32_t oldId, uint32_t newId);

  // ── Graph analysis helpers ────────────────────────────────────────────────
  void sortTopological();
  std::vector<IRNode*> consumers(uint32_t tensorId) const;
  IRNode*              producer(uint32_t tensorId) const;
  bool                 isDynamic() const;   // any dynamic shape?

  // ── Traversal ─────────────────────────────────────────────────────────────
  // Visit nodes in topological order
  void forEachNode(const std::function<void(IRNode&)>& fn);
  void forEachNodeMutable(const std::function<bool(IRNode&)>& fn); // return false = remove

  // ── Clone / copy ─────────────────────────────────────────────────────────
  std::unique_ptr<IRGraph> clone() const;

  // ── Statistics ────────────────────────────────────────────────────────────
  size_t   numNodes()   const { return nodes.size(); }
  size_t   numTensors() const { return tensors.size(); }
  uint64_t totalFLOPs() const;

  // ── Serialization (to/from FlatBuffers binary) ────────────────────────────
  std::vector<uint8_t> serialize() const;
  static std::unique_ptr<IRGraph> deserialize(const uint8_t* data, size_t size);

  // ── Debug ─────────────────────────────────────────────────────────────────
  std::string dump() const;         // human-readable IR text
  void        dumpToFile(const std::string& path) const;

private:
  uint32_t nextNodeId_   = 1;
  uint32_t nextTensorId_ = 1;
};

} // namespace ir
} // namespace armcc
