// ============================================================================
// ARM AI Compiler — ARM-IR: Graph (implementation)
// ============================================================================
#include "arm_ir/graph.h"
#include <algorithm>
#include <queue>
#include <sstream>
#include <fstream>
#include <stdexcept>

namespace armcc {
namespace ir {

// ---------------------------------------------------------------------------
// ModelFamily helpers
// ---------------------------------------------------------------------------
const char* modelFamilyToString(ModelFamily f) {
  switch (f) {
    case ModelFamily::LlamaStyle: return "llama_style";
    case ModelFamily::GemmaStyle: return "gemma_style";
    case ModelFamily::GPT2Style:  return "gpt2_style";
    case ModelFamily::T5Style:    return "t5_style";
    case ModelFamily::MoE:        return "moe";
    case ModelFamily::VisionLM:   return "vision_lm";
    case ModelFamily::Custom:     return "custom";
    default:                      return "unknown";
  }
}

// ---------------------------------------------------------------------------
// IRGraph: addNode / addTensor
// ---------------------------------------------------------------------------
IRNode* IRGraph::addNode(std::unique_ptr<IRNode> n) {
  n->id = nextNodeId_++;
  IRNode* ptr = n.get();
  nodeById[ptr->id] = ptr;
  nodes.push_back(std::move(n));
  return ptr;
}

IRTensor* IRGraph::addTensor(std::unique_ptr<IRTensor> t) {
  t->id = nextTensorId_++;
  IRTensor* ptr = t.get();
  tensorById[ptr->id] = ptr;
  if (!ptr->name.empty()) tensorByName[ptr->name] = ptr->id;
  tensors.push_back(std::move(t));
  return ptr;
}

IRNode* IRGraph::getNode(uint32_t id) {
  auto it = nodeById.find(id);
  return it != nodeById.end() ? it->second : nullptr;
}

IRTensor* IRGraph::getTensor(uint32_t id) {
  auto it = tensorById.find(id);
  return it != tensorById.end() ? it->second : nullptr;
}

IRTensor* IRGraph::getTensorByName(const std::string& name) {
  auto it = tensorByName.find(name);
  if (it == tensorByName.end()) return nullptr;
  return getTensor(it->second);
}

// ---------------------------------------------------------------------------
// Topological sort (Kahn's algorithm)
// ---------------------------------------------------------------------------
void IRGraph::sortTopological() {
  // Build in-degree map keyed by node id
  std::unordered_map<uint32_t, int> indegree;
  // For each node, find predecessors via shared tensors
  // Build a producer map: tensor_id → producing node id
  std::unordered_map<uint32_t, uint32_t> tensorProducer;
  for (auto& n : nodes) {
    indegree[n->id] = 0;
    for (auto tid : n->outputs) tensorProducer[tid] = n->id;
  }

  // Count in-degrees
  for (auto& n : nodes) {
    for (auto tid : n->inputs) {
      auto it = tensorProducer.find(tid);
      if (it != tensorProducer.end()) indegree[n->id]++;
    }
  }

  std::queue<uint32_t> q;
  for (auto& [id, deg] : indegree) {
    if (deg == 0) q.push(id);
  }

  topoOrder.clear();
  while (!q.empty()) {
    uint32_t nid = q.front(); q.pop();
    topoOrder.push_back(nid);
    IRNode* n = getNode(nid);
    if (!n) continue;
    // For each consumer of n's outputs, decrement their in-degree
    for (auto tid : n->outputs) {
      for (auto& m : nodes) {
        for (auto mtid : m->inputs) {
          if (mtid == tid) {
            if (--indegree[m->id] == 0) q.push(m->id);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Consumer / producer lookup
// ---------------------------------------------------------------------------
IRNode* IRGraph::producer(uint32_t tensorId) const {
  for (auto& n : nodes) {
    for (auto tid : n->outputs) {
      if (tid == tensorId) return n.get();
    }
  }
  return nullptr;
}

std::vector<IRNode*> IRGraph::consumers(uint32_t tensorId) const {
  std::vector<IRNode*> result;
  for (auto& n : nodes) {
    for (auto tid : n->inputs) {
      if (tid == tensorId) {
        result.push_back(n.get());
        break;
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Remove a node (DCE helper)
// ---------------------------------------------------------------------------
bool IRGraph::removeNode(uint32_t nodeId) {
  auto it = std::find_if(nodes.begin(), nodes.end(),
    [nodeId](const auto& n) { return n->id == nodeId; });
  if (it == nodes.end()) return false;
  nodeById.erase(nodeId);
  nodes.erase(it);
  return true;
}

// ---------------------------------------------------------------------------
// Replace tensor ID across all edges
// ---------------------------------------------------------------------------
void IRGraph::replaceTensor(uint32_t oldId, uint32_t newId) {
  for (auto& n : nodes) {
    for (auto& tid : n->inputs)  if (tid == oldId) tid = newId;
    for (auto& tid : n->outputs) if (tid == oldId) tid = newId;
  }
  // Update graph I/O
  for (auto& tid : inputIds)  if (tid == oldId) tid = newId;
  for (auto& tid : outputIds) if (tid == oldId) tid = newId;
}

// ---------------------------------------------------------------------------
// Traversal helpers
// ---------------------------------------------------------------------------
void IRGraph::forEachNode(const std::function<void(IRNode&)>& fn) {
  sortTopological();
  for (uint32_t nid : topoOrder) {
    IRNode* n = getNode(nid);
    if (n) fn(*n);
  }
}

bool IRGraph::isDynamic() const {
  for (auto& t : tensors) {
    if (t->shape.isDynamic()) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Total FLOPs estimate
// ---------------------------------------------------------------------------
uint64_t IRGraph::totalFLOPs() const {
  uint64_t total = 0;
  for (auto& n : nodes) {
    // Approximate FLOP counts per op type
    switch (n->op) {
      case OpCode::MatMul:
      case OpCode::GEMM:
      case OpCode::BatchMatMul: {
        // For A[M,K] @ B[K,N] → 2*M*K*N FLOPs
        // We look at the output tensor shape
        if (!n->outputs.empty()) {
          const IRTensor* out = const_cast<IRGraph*>(this)->getTensor(n->outputs[0]);
          if (out && !out->shape.isDynamic() && out->shape.rank() >= 2) {
            int64_t M = out->shape.dims[out->shape.rank()-2];
            int64_t N = out->shape.dims[out->shape.rank()-1];
            // K from input
            if (!n->inputs.empty()) {
              const IRTensor* inp = const_cast<IRGraph*>(this)->getTensor(n->inputs[0]);
              if (inp && inp->shape.rank() >= 2) {
                int64_t K = inp->shape.dims[inp->shape.rank()-1];
                total += 2ULL * (uint64_t)M * (uint64_t)K * (uint64_t)N;
              }
            }
          }
        }
        break;
      }
      default:
        // Elementwise: roughly count output elements
        if (!n->outputs.empty()) {
          const IRTensor* out = const_cast<IRGraph*>(this)->getTensor(n->outputs[0]);
          if (out) {
            int64_t elems = out->shape.numElements();
            if (elems > 0) total += (uint64_t)elems;
          }
        }
        break;
    }
  }
  return total;
}

// ---------------------------------------------------------------------------
// Human-readable IR dump
// ---------------------------------------------------------------------------
std::string IRGraph::dump() const {
  std::ostringstream oss;
  oss << "// ARM AI Compiler — ARM-IR Dump\n";
  oss << "// Model: " << model_id << "\n";
  oss << "// Family: " << modelFamilyToString(model_family) << "\n";
  oss << "// Nodes: " << nodes.size()
      << "  Tensors: " << tensors.size() << "\n\n";

  oss << "graph @" << name << " {\n";

  // Inputs
  oss << "  // Inputs\n";
  for (uint32_t id : inputIds) {
    const IRTensor* t = const_cast<IRGraph*>(this)->getTensor(id);
    if (t) {
      oss << "  input %" << t->name << " : "
          << dtypeToString(t->dtype)
          << t->shape.toString() << "\n";
    }
  }
  oss << "\n";

  // Nodes (in topological order if available)
  const auto& order = topoOrder.empty() ? [&](){
    std::vector<uint32_t> v;
    for (auto& n : nodes) v.push_back(n->id);
    return v;
  }() : topoOrder;

  for (uint32_t nid : order) {
    const IRNode* n = const_cast<IRGraph*>(this)->getNode(nid);
    if (!n) continue;

    // Output tensors
    oss << "  ";
    for (size_t i = 0; i < n->outputs.size(); ++i) {
      if (i > 0) oss << ", ";
      const IRTensor* t = const_cast<IRGraph*>(this)->getTensor(n->outputs[i]);
      oss << "%" << (t ? t->name : std::to_string(n->outputs[i]));
    }
    oss << " = " << opCodeToString(n->op) << "(";

    // Input tensors
    for (size_t i = 0; i < n->inputs.size(); ++i) {
      if (i > 0) oss << ", ";
      const IRTensor* t = const_cast<IRGraph*>(this)->getTensor(n->inputs[i]);
      oss << "%" << (t ? t->name : std::to_string(n->inputs[i]));
    }
    oss << ")";

    // Metadata annotations
    oss << "  // exec=" << execUnitToString(n->assigned_unit);
    if (n->cost_cpu_ms >= 0) oss << " cpu=" << n->cost_cpu_ms << "ms";
    if (n->cost_npu_ms >= 0) oss << " npu=" << n->cost_npu_ms << "ms";
    if (n->is_fused) oss << " [fused]";
    oss << "\n";
  }

  // Outputs
  oss << "\n  // Outputs\n";
  for (uint32_t id : outputIds) {
    const IRTensor* t = const_cast<IRGraph*>(this)->getTensor(id);
    if (t) oss << "  return %" << t->name << "\n";
  }

  oss << "}\n";
  return oss.str();
}

void IRGraph::dumpToFile(const std::string& path) const {
  std::ofstream f(path);
  f << dump();
}

} // namespace ir
} // namespace armcc
