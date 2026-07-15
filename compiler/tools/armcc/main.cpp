// ============================================================================
// ARM AI Compiler — CLI Tool (C++ driver)
//
// This is the C++ side of the compiler. The Python CLI calls this binary
// with the exported graph JSON and orchestrates the optimization pipeline.
//
// Usage:
//   armcc compile --graph exported_graph.json
//                 --targets snapdragon_8_gen3,generic_arm64
//                 --quant int8
//                 --calibration calibration.json
//                 --output model.armpack
//
//   armcc inspect model.armpack
//   armcc ir-dump model.armpack --graph snapdragon_npu_int8
// ============================================================================
#include "arm_ir/graph.h"
#include "arm_ir/types.h"
#include "arm_ir/ops.h"
#include "analysis/cost_model.h"
#include "analysis/graph_analyzer.h"
#include "passes/pass_manager.h"
#include "generator/graph_family_generator.h"
#include "memory/memory_planner.h"
#include "package/package_generator.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <algorithm>

namespace fs = std::filesystem;
using json   = nlohmann::json;

// ---------------------------------------------------------------------------
// Helper: split a comma-separated string
// ---------------------------------------------------------------------------
static std::vector<std::string> splitCSV(const std::string& s, char delim = ',') {
  std::vector<std::string> result;
  std::istringstream ss(s);
  std::string token;
  while (std::getline(ss, token, delim)) {
    if (!token.empty()) result.push_back(token);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Helper: parse --key value from argv
// ---------------------------------------------------------------------------
static std::string argValue(int argc, char** argv,
                             const std::string& key,
                             const std::string& defval = "") {
  for (int i = 1; i < argc - 1; ++i) {
    if (std::string(argv[i]) == key) return argv[i+1];
  }
  return defval;
}

static bool argFlag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == key) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Build an IRGraph from the exported JSON produced by the Python layer
// ---------------------------------------------------------------------------
static std::unique_ptr<armcc::ir::IRGraph> graphFromJSON(const std::string& path) {
  std::ifstream f(path);
  if (!f) {
    std::cerr << "[armcc] ERROR: Cannot open graph JSON: " << path << "\n";
    return nullptr;
  }

  json j;
  try {
    f >> j;
  } catch (const std::exception& e) {
    std::cerr << "[armcc] ERROR: JSON parse error: " << e.what() << "\n";
    return nullptr;
  }

  auto graph = std::make_unique<armcc::ir::IRGraph>();
  graph->model_id = j.value("model_id", "unknown");

  // Architecture metadata
  if (j.contains("arch")) {
    auto& arch = j["arch"];
    graph->num_layers      = arch.value("num_layers", 0);
    graph->hidden_size     = arch.value("hidden_size", 0);
    graph->num_heads       = arch.value("num_heads", 0);
    graph->num_kv_heads    = arch.value("num_kv_heads", 0);
    graph->intermediate_sz = arch.value("intermediate_size", 0);
    graph->vocab_size      = arch.value("vocab_size", 0);
    graph->max_position_embeddings = arch.value("max_position_embeddings", 2048);
  }

  // Detect model family
  std::string family = j.value("arch_family", "unknown");
  if      (family == "llama_style") graph->model_family = armcc::ir::ModelFamily::LlamaStyle;
  else if (family == "gemma_style") graph->model_family = armcc::ir::ModelFamily::GemmaStyle;
  else if (family == "gpt2_style")  graph->model_family = armcc::ir::ModelFamily::GPT2Style;

  // Tensors
  for (auto& jt : j.value("tensors", json::array())) {
    auto t = std::make_unique<armcc::ir::IRTensor>();
    t->id       = jt.value("id", 0);
    t->name     = jt.value("name", "");
    t->dtype    = armcc::ir::dtypeFromString(jt.value("dtype", "f32"));
    t->is_weight= jt.value("is_weight", false);
    t->is_input = jt.value("is_input", false);

    // Shape
    if (jt.contains("shape") && jt["shape"].is_array()) {
      for (auto& d : jt["shape"]) {
        t->shape.dims.push_back(d.get<int64_t>());
      }
    }

    graph->addTensor(std::move(t));
  }

  // Nodes
  for (auto& jn : j.value("nodes", json::array())) {
    auto n = std::make_unique<armcc::ir::IRNode>();
    n->id   = jn.value("id", 0);
    n->name = jn.value("name", "");
    n->op   = armcc::ir::opCodeFromString(jn.value("op", "Unknown"));

    for (auto& tid : jn.value("inputs", json::array()))
      n->inputs.push_back(tid.get<uint32_t>());
    for (auto& tid : jn.value("outputs", json::array()))
      n->outputs.push_back(tid.get<uint32_t>());

    graph->addNode(std::move(n));
  }

  // Graph I/O
  for (auto& tid : j.value("input_ids", json::array()))
    graph->inputIds.push_back(tid.get<uint32_t>());
  for (auto& tid : j.value("output_ids", json::array()))
    graph->outputIds.push_back(tid.get<uint32_t>());

  graph->sortTopological();
  return graph;
}

// ---------------------------------------------------------------------------
// Compile command
// ---------------------------------------------------------------------------
static int cmdCompile(int argc, char** argv) {
  std::string graph_json  = argValue(argc, argv, "--graph");
  std::string targets_csv = argValue(argc, argv, "--targets", "generic_arm64");
  std::string quant_str   = argValue(argc, argv, "--quant", "int8");
  std::string calib_json  = argValue(argc, argv, "--calibration");
  std::string output_path = argValue(argc, argv, "--output", "model.armpack");
  bool        verbose     = argFlag(argc, argv, "--verbose");

  if (graph_json.empty()) {
    std::cerr << "[armcc] ERROR: --graph is required\n";
    return 1;
  }

  auto t0 = std::chrono::steady_clock::now();

  // ── Load graph ──────────────────────────────────────────────────────────
  std::cout << "[armcc] Loading graph from: " << graph_json << "\n";
  auto source_graph = graphFromJSON(graph_json);
  if (!source_graph) return 1;

  std::cout << "[armcc] Model: " << source_graph->model_id
            << "  Nodes: " << source_graph->numNodes()
            << "  Family: " << armcc::ir::modelFamilyToString(source_graph->model_family)
            << "\n";

  // ── Load device profiles ────────────────────────────────────────────────
  armcc::analysis::CostModel cost_model;
  std::string profiles_dir = "device_profiles";
  if (fs::exists(profiles_dir)) {
    cost_model.loadProfileDirectory(profiles_dir);
    std::cout << "[armcc] Loaded " << cost_model.numProfiles()
              << " device profiles\n";
  } else {
    std::cout << "[armcc] WARNING: device_profiles/ not found — "
                 "using analytical cost estimates\n";
  }

  // ── Graph analysis ──────────────────────────────────────────────────────
  std::cout << "[armcc] Running graph analysis ...\n";
  armcc::analysis::GraphAnalyzer analyzer(*source_graph);
  auto analysis = analyzer.analyze();
  std::cout << "[armcc]   " << analysis.toString() << "\n";

  // ── Determine target SoCs ───────────────────────────────────────────────
  auto target_names = splitCSV(targets_csv);
  std::vector<armcc::ir::SoCID> target_socs;
  for (auto& name : target_names) {
    target_socs.push_back(armcc::ir::socIDFromString(name));
  }

  // ── Build graph family spec ─────────────────────────────────────────────
  armcc::generator::GraphFamilySpec spec;
  spec.target_socs = target_socs;

  // Quantization dtypes
  armcc::ir::DType main_quant = armcc::ir::DType::INT8;
  if      (quant_str == "fp32")  main_quant = armcc::ir::DType::F32;
  else if (quant_str == "fp16")  main_quant = armcc::ir::DType::F16;
  else if (quant_str == "int8")  main_quant = armcc::ir::DType::I8;
  else if (quant_str == "int4")  main_quant = armcc::ir::DType::I4;
  else if (quant_str == "fp8")   main_quant = armcc::ir::DType::F8_E4M3;
  else if (quant_str == "mixed") {
    spec.quant_dtypes = {armcc::ir::DType::I8, armcc::ir::DType::I4};
    spec.include_mixed_precision = true;
  }
  if (quant_str != "mixed") spec.quant_dtypes = {main_quant};

  // ── Run optimization passes for each target ─────────────────────────────
  std::cout << "[armcc] Running optimization pipeline ...\n";
  for (auto soc_id : target_socs) {
    const auto* dev = cost_model.getProfile(soc_id);
    if (!dev) {
      std::cout << "[armcc]   [SKIP] No profile for: "
                << armcc::ir::socIDToString(soc_id) << "\n";
      continue;
    }

    std::cout << "[armcc]   Target: " << dev->name << "\n";
    auto graph_copy = source_graph->clone();

    armcc::passes::PassManager pm;
    armcc::passes::PassOptions opts;
    opts.device  = dev;
    opts.verbose = verbose;

    pm.addFullPipeline(*dev, main_quant);
    auto results = pm.run(*graph_copy, opts);

    if (verbose) {
      for (auto& r : results) std::cout << "    " << r.toString() << "\n";
    }
    std::cout << "[armcc]     Passes: " << results.size()
              << "  Nodes after: " << graph_copy->numNodes() << "\n";
  }

  // ── Memory planning ─────────────────────────────────────────────────────
  std::cout << "[armcc] Memory planning ...\n";
  armcc::memory::MemoryPlanner mem_planner;
  auto mem_plan = mem_planner.plan(*source_graph);
  std::cout << "[armcc]   Weights: " << mem_plan.weights_arena_bytes/1024/1024 << "MB"
            << "  Activations: " << mem_plan.activations_arena_bytes/1024 << "KB"
            << "  KV cache: "    << mem_plan.kv_cache_bytes/1024/1024 << "MB\n";

  // ── Graph family generation ─────────────────────────────────────────────
  std::cout << "[armcc] Generating graph family ...\n";
  armcc::generator::GraphFamilyGenerator gen(cost_model, nullptr, 0);
  auto family = gen.generate(*source_graph, spec);
  std::cout << "[armcc]   Generated " << family.size() << " graph(s)\n";

  // ── Package generation ──────────────────────────────────────────────────
  std::cout << "[armcc] Writing " << output_path << " ...\n";
  armcc::pkg::PackageGeneratorOptions pkg_opts;
  pkg_opts.output_path = output_path;
  pkg_opts.overwrite   = true;

  armcc::pkg::RuntimeConfig rt_cfg;
  armcc::pkg::PackageGenerator pkg_gen(pkg_opts);

  bool ok = pkg_gen.generate(
    source_graph->model_id,
    *source_graph,
    family,
    mem_plan,
    rt_cfg
  );

  if (!ok) {
    std::cerr << "[armcc] ERROR: " << pkg_gen.lastError() << "\n";
    return 1;
  }

  auto t1 = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);

  std::cout << "\n[armcc] ✓ Done in " << elapsed.count() << "ms\n";
  std::cout << "[armcc] → " << output_path << "\n\n";
  return 0;
}

// ---------------------------------------------------------------------------
// IR dump command
// ---------------------------------------------------------------------------
static int cmdIRDump(int argc, char** argv) {
  std::string pack_or_json = (argc > 2) ? argv[2] : "";
  if (pack_or_json.empty()) {
    std::cerr << "Usage: armcc ir-dump <exported_graph.json>\n";
    return 1;
  }

  auto graph = graphFromJSON(pack_or_json);
  if (!graph) return 1;

  std::cout << graph->dump();
  return 0;
}

// ---------------------------------------------------------------------------
// Inspect command
// ---------------------------------------------------------------------------
static int cmdInspect(int argc, char** argv) {
  std::string path = (argc > 2) ? argv[2] : "";
  if (path.empty()) {
    std::cerr << "Usage: armcc inspect <model.armpack>\n";
    return 1;
  }
  // For now, delegate to armpack_format reader
  std::cout << "Inspecting: " << path << "\n";
  std::cout << "(Full inspect requires libzip integration — see package_generator.h)\n";
  return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "ARM AI Compiler v0.1.0\n\n"
              << "Usage:\n"
              << "  armcc compile  --graph <json> --targets <csv> --quant <dtype> --output <pack>\n"
              << "  armcc inspect  <model.armpack>\n"
              << "  armcc ir-dump  <exported_graph.json>\n"
              << "\nRun 'armcc <command> --help' for more details.\n";
    return 1;
  }

  std::string cmd = argv[1];
  if (cmd == "compile")  return cmdCompile(argc, argv);
  if (cmd == "inspect")  return cmdInspect(argc, argv);
  if (cmd == "ir-dump")  return cmdIRDump(argc, argv);

  std::cerr << "[armcc] Unknown command: " << cmd << "\n";
  return 1;
}
