// ============================================================================
// ARM AI Compiler — arm-ir-dump tool
// Reads an exported_graph.json and prints a human-readable ARM-IR dump.
// ============================================================================
#include "arm_ir/graph.h"
#include "arm_ir/types.h"
#include "arm_ir/ops.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <string>

using json = nlohmann::json;

// Re-use the graphFromJSON logic (extracted to a shared header in practice)
static std::unique_ptr<armcc::ir::IRGraph> loadGraphJSON(const std::string& path) {
  std::ifstream f(path);
  if (!f) { std::cerr << "Cannot open: " << path << "\n"; return nullptr; }
  json j; f >> j;

  auto graph = std::make_unique<armcc::ir::IRGraph>();
  graph->model_id = j.value("model_id", "unknown");

  if (j.contains("arch")) {
    auto& a = j["arch"];
    graph->num_layers   = a.value("num_layers", 0);
    graph->hidden_size  = a.value("hidden_size", 0);
    graph->num_heads    = a.value("num_heads",   0);
    graph->num_kv_heads = a.value("num_kv_heads", 0);
    graph->vocab_size   = a.value("vocab_size", 0);
  }

  std::string fam = j.value("arch_family", "unknown");
  if (fam == "llama_style") graph->model_family = armcc::ir::ModelFamily::LlamaStyle;
  if (fam == "gpt2_style")  graph->model_family = armcc::ir::ModelFamily::GPT2Style;

  for (auto& jt : j.value("tensors", json::array())) {
    auto t = std::make_unique<armcc::ir::IRTensor>();
    t->id       = jt.value("id", 0);
    t->name     = jt.value("name", "");
    t->dtype    = armcc::ir::dtypeFromString(jt.value("dtype", "f32"));
    t->is_weight= jt.value("is_weight", false);
    t->is_input = jt.value("is_input", false);
    for (auto& d : jt.value("shape", json::array()))
      t->shape.dims.push_back(d.get<int64_t>());
    graph->addTensor(std::move(t));
  }

  for (auto& jn : j.value("nodes", json::array())) {
    auto n = std::make_unique<armcc::ir::IRNode>();
    n->id   = jn.value("id", 0);
    n->name = jn.value("name", "");
    n->op   = armcc::ir::opCodeFromString(jn.value("op", "Unknown"));
    for (auto& tid : jn.value("inputs",  json::array())) n->inputs.push_back(tid.get<uint32_t>());
    for (auto& tid : jn.value("outputs", json::array())) n->outputs.push_back(tid.get<uint32_t>());
    graph->addNode(std::move(n));
  }
  for (auto& tid : j.value("input_ids",  json::array())) graph->inputIds.push_back(tid.get<uint32_t>());
  for (auto& tid : j.value("output_ids", json::array())) graph->outputIds.push_back(tid.get<uint32_t>());
  graph->sortTopological();
  return graph;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "arm-ir-dump: Print ARM-IR for a compiled graph\n"
              << "Usage: arm-ir-dump <exported_graph.json> [--stats]\n";
    return 1;
  }

  auto graph = loadGraphJSON(argv[1]);
  if (!graph) return 1;

  bool show_stats = (argc > 2 && std::string(argv[2]) == "--stats");

  std::cout << graph->dump();

  if (show_stats) {
    std::cout << "\n// Statistics\n"
              << "// Nodes:   " << graph->numNodes()   << "\n"
              << "// Tensors: " << graph->numTensors() << "\n"
              << "// FLOPs:   " << graph->totalFLOPs() / 1e9 << " GFLOPs\n"
              << "// Weights: " << graph->weight_bytes / 1024 / 1024 << " MB\n";
  }

  return 0;
}
