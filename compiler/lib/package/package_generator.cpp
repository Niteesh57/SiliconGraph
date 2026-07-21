// ============================================================================
// ARM AI Compiler — Package Generator (implementation)
//
// Assembles the complete .armpack ZIP archive:
//   manifest.json, graphs/, kernels/, weights/, tokenizer/,
//   memory_maps/, runtime_config.json, selector.wasm
// ============================================================================
#include "package/package_generator.h"
#include "generator/condition_matrix.h"
#include <zip.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace armcc {
namespace pkg {

// ---------------------------------------------------------------------------
// Helper: ISO 8601 timestamp
// ---------------------------------------------------------------------------
static std::string isoTimestamp() {
  std::time_t t = std::time(nullptr);
  char buf[64];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
  return buf;
}

// ---------------------------------------------------------------------------
// Helper: add bytes to a zip archive
// ---------------------------------------------------------------------------
static bool zipAddBuffer(zip_t* za,
                         const std::string& entryName,
                         const void* data,
                         size_t size)
{
  zip_source_t* src = zip_source_buffer(za, data, size, 0);
  if (!src) return false;
  if (zip_file_add(za, entryName.c_str(), src, ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8) < 0) {
    zip_source_free(src);
    return false;
  }
  return true;
}

static bool zipAddString(zip_t* za, const std::string& entryName,
                         const std::string& str) {
  // Make a heap copy — libzip takes ownership when we use ZIP_SOURCE_BUFFER
  char* buf = new char[str.size()];
  std::memcpy(buf, str.data(), str.size());
  zip_source_t* src = zip_source_buffer(za, buf, str.size(), 1 /* freep */);
  if (!src) { delete[] buf; return false; }
  if (zip_file_add(za, entryName.c_str(), src, ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8) < 0) {
    zip_source_free(src);
    return false;
  }
  return true;
}

static bool zipAddFile(zip_t* za, const std::string& entryName,
                       const std::string& filePath) {
  std::ifstream f(filePath, std::ios::binary | std::ios::ate);
  if (!f) return false;
  size_t sz = f.tellg();
  f.seekg(0);
  std::string data(sz, '\0');
  f.read(data.data(), sz);
  return zipAddString(za, entryName, data);
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
PackageGenerator::PackageGenerator(PackageGeneratorOptions opts)
  : opts_(std::move(opts)) {}

// ---------------------------------------------------------------------------
// generate
// ---------------------------------------------------------------------------
bool PackageGenerator::generate(
    const std::string&                          model_id,
    const ir::IRGraph&                          source_graph,
    const std::vector<generator::CompiledGraph>& graph_family,
    const memory::MemoryPlan&                   mem_plan,
    const RuntimeConfig&                        runtime_cfg,
    std::function<void(int, const std::string&)> progress_cb)
{
  if (opts_.output_path.empty()) {
    lastError_ = "No output path specified";
    return false;
  }

  // Remove existing file if overwrite enabled
  if (opts_.overwrite && fs::exists(opts_.output_path)) {
    fs::remove(opts_.output_path);
  }

  int zip_error = 0;
  zip_t* za = zip_open(opts_.output_path.c_str(),
                        ZIP_CREATE | ZIP_TRUNCATE, &zip_error);
  if (!za) {
    zip_error_t ze;
    zip_error_init_with_code(&ze, zip_error);
    lastError_ = std::string("Cannot create ZIP: ") + zip_error_strerror(&ze);
    zip_error_fini(&ze);
    return false;
  }

  // Build manifest
  PackageManifest manifest;
  manifest.compiler_version = "0.1.0";
  manifest.model_id         = model_id;
  manifest.model_family     = ir::modelFamilyToString(source_graph.model_family);
  manifest.created_at       = isoTimestamp();
  manifest.num_layers       = source_graph.num_layers;
  manifest.hidden_size      = source_graph.hidden_size;
  manifest.num_heads        = source_graph.num_heads;
  manifest.num_kv_heads     = source_graph.num_kv_heads;
  manifest.vocab_size       = source_graph.vocab_size;
  manifest.max_position_embeddings = source_graph.max_position_embeddings;

  int progress = 0;
  auto report = [&](const std::string& step) {
    ++progress;
    std::cout << "[pkg]  [" << progress << "] " << step << "\n";
    if (progress_cb) progress_cb(progress, step);
  };

  // 1. Write graphs
  report("Writing compiled graphs ...");
  if (!writeGraphs(za, graph_family, manifest)) goto fail;

  // 2. Write weights
  report("Writing quantized weights ...");
  if (!writeWeights(za, source_graph, mem_plan, manifest)) goto fail;

  // 3. Write kernels (stubs in Phase 1)
  report("Writing kernel objects ...");
  if (!writeKernels(za, manifest)) goto fail;

  // 4. Write tokenizer
  report("Writing tokenizer assets ...");
  if (!writeTokenizer(za, manifest)) goto fail;

  // 5. Write runtime config
  report("Writing runtime config ...");
  if (!writeRuntimeConfig(za, runtime_cfg)) goto fail;

  // 6. Write memory maps
  report("Writing memory maps ...");
  if (!writeMemoryMaps(za, mem_plan)) goto fail;

  // 7. Build and write selector (selector.wasm stub + selector JSON)
  report("Writing graph selector ...");
  if (!writeSelectorWasm(za, graph_family)) goto fail;

  // 8. Write manifest (last — references everything above)
  report("Writing manifest.json ...");
  if (!writeManifest(za, manifest)) goto fail;

  zip_close(za);
  std::cout << "[pkg] ✓ Package written: " << opts_.output_path
            << " (" << graph_family.size() << " graphs)\n";
  return true;

fail:
  zip_discard(za);
  return false;
}

// ---------------------------------------------------------------------------
// writeManifest
// ---------------------------------------------------------------------------
bool PackageGenerator::writeManifest(void* za, const PackageManifest& manifest) {
  return zipAddString((zip_t*)za, "manifest.json", manifest.toJSON());
}

// ---------------------------------------------------------------------------
// writeGraphs
// ---------------------------------------------------------------------------
bool PackageGenerator::writeGraphs(void* za,
    const std::vector<generator::CompiledGraph>& family,
    PackageManifest& manifest)
{
  for (uint32_t i = 0; i < (uint32_t)family.size(); ++i) {
    const auto& cg = family[i];
    std::string graphPath = "graphs/graph_" + std::to_string(i) + ".armgraph";

    GraphEntry ge;
    ge.index               = i;
    ge.id                  = cg.conditions.toKey();
    ge.path                = graphPath;
    ge.soc_name            = ir::socIDToString(cg.conditions.soc_id);
    ge.quant_dtype         = ir::dtypeToString(cg.conditions.quant_dtype);
    ge.memory_budget_mb    = cg.conditions.memory_mb;
    ge.context_length      = cg.conditions.context_length;
    ge.thermal_state       = (cg.conditions.thermal == ir::ThermalState::Nominal) ? "nominal" :
                             (cg.conditions.thermal == ir::ThermalState::Warm)    ? "warm"    : "hot";
    ge.latency_mode        = (cg.conditions.latency_mode == generator::LatencyMode::Interactive) ?
                             "interactive" : "background";
    ge.estimated_ttft_ms   = cg.estimated_ttft_ms;
    ge.estimated_tpot_ms   = cg.estimated_tpot_ms;
    ge.peak_memory_bytes   = cg.peak_memory_bytes;
    ge.weight_bytes        = (cg.graph ? cg.graph->weight_bytes : 0);
    manifest.graphs.push_back(ge);

    // Write serialized graph bytes
    if (!cg.serialized_graph.empty()) {
      if (!zipAddBuffer((zip_t*)za, graphPath,
                        cg.serialized_graph.data(),
                        cg.serialized_graph.size())) return false;
    } else {
      // Write a placeholder header if serialization isn't complete yet
      std::string placeholder = "ARMGRAPH_V1\n" + cg.conditions.toKey() + "\n";
      if (!zipAddString((zip_t*)za, graphPath, placeholder)) return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// writeWeights
// ---------------------------------------------------------------------------
bool PackageGenerator::writeWeights(void* za,
    const ir::IRGraph& graph,
    const memory::MemoryPlan& plan,
    PackageManifest& manifest)
{
  // Collect weight tensors and write a flat binary blob
  std::vector<uint8_t> weight_blob;
  weight_blob.reserve(graph.weight_bytes > 0 ? graph.weight_bytes : 1024);

  for (auto& t : graph.tensors) {
    if (!t->is_weight) continue;
    if (t->weight_data && t->weight_size > 0) {
      size_t old_size = weight_blob.size();
      // Pad to alignment
      uint32_t align = plan.blocks.empty() ? 64 : 4096;
      while (weight_blob.size() % align != 0) weight_blob.push_back(0);
      weight_blob.insert(weight_blob.end(),
                         t->weight_data,
                         t->weight_data + t->weight_size);
    }
  }

  // If no weight data (common in Phase 1 without actual model loading):
  // write a placeholder
  std::string dtype_tag = "int8";
  if (!graph.nodes.empty()) {
    for (auto& n : graph.nodes) {
      if (n->is_quantized) {
        for (uint32_t tid : n->inputs) {
          const ir::IRTensor* t = const_cast<ir::IRGraph&>(graph).getTensor(tid);
          if (t && t->is_weight) {
            dtype_tag = ir::dtypeToString(t->dtype);
            break;
          }
        }
        break;
      }
    }
  }

  std::string weightPath = "weights/weights_" + dtype_tag + ".bin";

  WeightEntry we;
  we.id            = "weights_" + dtype_tag;
  we.path          = weightPath;
  we.dtype         = dtype_tag;
  we.size_bytes    = weight_blob.empty() ? 64 : weight_blob.size();
  we.mmap_safe     = true;
  we.page_alignment = 4096;
  manifest.weights.push_back(we);

  if (!weight_blob.empty()) {
    return zipAddBuffer((zip_t*)za, weightPath,
                        weight_blob.data(), weight_blob.size());
  } else {
    // Placeholder
    std::string ph = "ARMWEIGHTS_V1 dtype=" + dtype_tag + "\n";
    return zipAddString((zip_t*)za, weightPath, ph);
  }
}

// ---------------------------------------------------------------------------
// writeKernels — Phase 1: stubs
// ---------------------------------------------------------------------------
bool PackageGenerator::writeKernels(void* za, PackageManifest& manifest) {
  // In Phase 1, write kernel descriptor stubs.
  // Phase 2 will have actual compiled kernel objects.
  struct KernelStub {
    std::string id, target_str, path, op_pattern;
    KernelTarget target;
  };

  std::vector<KernelStub> stubs = {
    {"matmul_int8_neon",  "cpu_neon",    "kernels/arm_neon/matmul_int8.o",    "MatMul",  KernelTarget::CPU_NEON},
    {"attn_fp16_neon",    "cpu_neon",    "kernels/arm_neon/attention_fp16.o", "GroupQueryAttention", KernelTarget::CPU_NEON},
    {"gemm_fp16_vulkan",  "gpu_vulkan",  "kernels/gpu_vulkan/gemm_fp16.spv",  "GEMM",    KernelTarget::GPU_Vulkan},
    {"softmax_vulkan",    "gpu_vulkan",  "kernels/gpu_vulkan/softmax.spv",    "Softmax", KernelTarget::GPU_Vulkan},
    {"matmul_int4_htp",   "npu_htp",    "kernels/snapdragon_htp/matmul_int4.bin","MatMul",KernelTarget::NPU_HTP},
    {"attn_int8_neuripilot","npu_np",   "kernels/mediatek_npu/attn_int8.bin","GroupQueryAttention",KernelTarget::NPU_NeuroPilot},
  };

  for (auto& stub : stubs) {
    std::string content = "ARMKERNEL_STUB target=" + stub.target_str + " op=" + stub.op_pattern;
    zipAddString((zip_t*)za, stub.path, content);

    KernelEntry ke;
    ke.id         = stub.id;
    ke.target     = stub.target;
    ke.path       = stub.path;
    ke.size_bytes = content.size();
    ke.op_pattern = stub.op_pattern;
    ke.is_fused   = false;
    manifest.kernels.push_back(ke);
  }
  return true;
}

// ---------------------------------------------------------------------------
// writeTokenizer
// ---------------------------------------------------------------------------
bool PackageGenerator::writeTokenizer(void* za, PackageManifest& manifest) {
  if (!opts_.tokenizer_dir.empty() && fs::exists(opts_.tokenizer_dir)) {
    for (auto& entry : fs::directory_iterator(opts_.tokenizer_dir)) {
      if (!entry.is_regular_file()) continue;
      std::string rel = "tokenizer/" + entry.path().filename().string();
      zipAddFile((zip_t*)za, rel, entry.path().string());
    }
    manifest.tokenizer_path = "tokenizer/";
    return true;
  }

  // Write minimal tokenizer placeholder
  std::string tok_stub = R"({"version":"placeholder","model_type":"bpe"})";
  zipAddString((zip_t*)za, "tokenizer/tokenizer.json", tok_stub);
  manifest.tokenizer_path = "tokenizer/";
  return true;
}

// ---------------------------------------------------------------------------
// writeRuntimeConfig
// ---------------------------------------------------------------------------
bool PackageGenerator::writeRuntimeConfig(void* za, const RuntimeConfig& cfg) {
  return zipAddString((zip_t*)za, "runtime_config.json", cfg.toJSON());
}

// ---------------------------------------------------------------------------
// writeMemoryMaps
// ---------------------------------------------------------------------------
bool PackageGenerator::writeMemoryMaps(void* za, const memory::MemoryPlan& plan) {
  // Write weights_layout.json
  json wl;
  wl["weights_bytes"]     = plan.weights_arena_bytes;
  wl["activations_bytes"] = plan.activations_arena_bytes;
  wl["kv_cache_bytes"]    = plan.kv_cache_bytes;
  wl["total_bytes"]       = plan.total_bytes;
  zipAddString((zip_t*)za, "memory_maps/weights_layout.json", wl.dump(2));

  // Write kv_cache_layout.json
  json kv;
  kv["enabled"]            = plan.kv_layout.num_layers > 0;
  kv["num_layers"]         = plan.kv_layout.num_layers;
  kv["num_kv_heads"]       = plan.kv_layout.num_kv_heads;
  kv["head_dim"]           = plan.kv_layout.head_dim;
  kv["max_seq_len"]        = plan.kv_layout.max_seq_len;
  kv["dtype"]              = ir::dtypeToString(plan.kv_layout.dtype);
  kv["paged"]              = plan.kv_layout.paged;
  kv["page_size_tokens"]   = plan.kv_layout.page_size_tokens;
  kv["key_cache_bytes"]    = plan.kv_layout.key_cache_bytes;
  kv["val_cache_bytes"]    = plan.kv_layout.val_cache_bytes;
  kv["total_bytes"]        = plan.kv_layout.total_bytes;
  kv["layer_offsets_key"]  = plan.kv_layout.layer_offsets_key;
  kv["layer_offsets_val"]  = plan.kv_layout.layer_offsets_val;
  zipAddString((zip_t*)za, "memory_maps/kv_cache_layout.json", kv.dump(2));
  return true;
}

// ---------------------------------------------------------------------------
// writeSelectorWasm — writes selector JSON + C source stub
// ---------------------------------------------------------------------------
bool PackageGenerator::writeSelectorWasm(void* za,
    const std::vector<generator::CompiledGraph>& family)
{
  // Build selector entries
  std::vector<generator::GraphSelectorEntry> entries;
  for (uint32_t i = 0; i < (uint32_t)family.size(); ++i) {
    generator::GraphSelectorEntry e;
    e.graph_index       = i;
    e.conditions        = family[i].conditions;
    e.estimated_tpot_ms = family[i].estimated_tpot_ms;
    e.peak_memory_bytes = family[i].peak_memory_bytes;
    entries.push_back(e);
  }

  generator::GraphSelector selector(entries);

  // Write selector index JSON
  zipAddString((zip_t*)za, "selector_index.json", selector.toJSON());

  // Write the C source for selector.wasm (compile with clang --target=wasm32)
  std::string csrc = generator::generateSelectorC(family);
  zipAddString((zip_t*)za, "selector_src.c", csrc);

  // Write a placeholder selector.wasm (real WASM compilation done by build script)
  std::string wasm_stub = "ARMPACK_SELECTOR_WASM_STUB\n"
                          "Compile selector_src.c with:\n"
                          "  clang --target=wasm32 -O2 -o selector.wasm selector_src.c\n";
  zipAddString((zip_t*)za, "selector.wasm", wasm_stub);
  return true;
}

// ---------------------------------------------------------------------------
// verify
// ---------------------------------------------------------------------------
bool PackageGenerator::verify(const std::string& path, std::string& error_out) {
  int err = 0;
  zip_t* za = zip_open(path.c_str(), ZIP_RDONLY, &err);
  if (!za) {
    error_out = "Cannot open package";
    return false;
  }

  // Check required entries
  std::vector<std::string> required = {"manifest.json", "selector.wasm",
                                        "runtime_config.json"};
  for (auto& entry : required) {
    if (zip_name_locate(za, entry.c_str(), 0) < 0) {
      error_out = "Missing required entry: " + entry;
      zip_close(za);
      return false;
    }
  }

  zip_close(za);
  return true;
}

// ---------------------------------------------------------------------------
// inspect
// ---------------------------------------------------------------------------
void PackageGenerator::inspect(const std::string& path) {
  int err = 0;
  zip_t* za = zip_open(path.c_str(), ZIP_RDONLY, &err);
  if (!za) { std::cout << "Cannot open: " << path << "\n"; return; }

  zip_int64_t n = zip_get_num_entries(za, 0);
  std::cout << "\n.armpack: " << path << "\n";
  std::cout << std::string(50, '─') << "\n";
  std::cout << "  Entries: " << n << "\n";

  // Read and print manifest
  zip_int64_t mi = zip_name_locate(za, "manifest.json", 0);
  if (mi >= 0) {
    zip_stat_t st;
    zip_stat_index(za, mi, 0, &st);
    std::string data(st.size, '\0');
    zip_file_t* f = zip_fopen_index(za, mi, 0);
    if (f) {
      zip_fread(f, data.data(), st.size);
      zip_fclose(f);
      try {
        auto j = json::parse(data);
        std::cout << "  Model:    " << j.value("model_id", "?") << "\n";
        std::cout << "  Family:   " << j.value("model_family", "?") << "\n";
        std::cout << "  Compiler: " << j.value("compiler_version", "?") << "\n";
        std::cout << "  Created:  " << j.value("created_at", "?") << "\n";
        if (j.contains("graphs")) {
          std::cout << "  Graphs:   " << j["graphs"].size() << "\n";
          for (auto& g : j["graphs"]) {
            std::cout << "    [" << g.value("index", 0) << "] "
                      << g.value("id", "?") << "\n"
                      << "         TTFT=" << g.value("estimated_ttft_ms", 0.0f) << "ms"
                      << " TPOT=" << g.value("estimated_tpot_ms", 0.0f) << "ms"
                      << " Mem=" << g.value("peak_memory_bytes", 0ULL)/1024/1024 << "MB\n";
          }
        }
      } catch (...) {}
    }
  }
  zip_close(za);
}

// ---------------------------------------------------------------------------
// PackageManifest: toJSON / fromJSON
// ---------------------------------------------------------------------------
std::string PackageManifest::toJSON(bool pretty) const {
  json j;
  j["armpack_version"]   = armpack_version;
  j["compiler_version"]  = compiler_version;
  j["model_id"]          = model_id;
  j["model_family"]      = model_family;
  j["created_at"]        = created_at;

  json arch;
  arch["num_layers"]              = num_layers;
  arch["hidden_size"]             = hidden_size;
  arch["num_heads"]               = num_heads;
  arch["num_kv_heads"]            = num_kv_heads;
  arch["vocab_size"]              = vocab_size;
  arch["max_position_embeddings"] = max_position_embeddings;
  j["arch"] = arch;

  json garr = json::array();
  for (auto& g : graphs) {
    json jg;
    jg["index"]               = g.index;
    jg["id"]                  = g.id;
    jg["path"]                = g.path;
    jg["soc_name"]            = g.soc_name;
    jg["quant_dtype"]         = g.quant_dtype;
    jg["memory_budget_mb"]    = g.memory_budget_mb;
    jg["context_length"]      = g.context_length;
    jg["thermal_state"]       = g.thermal_state;
    jg["latency_mode"]        = g.latency_mode;
    jg["estimated_ttft_ms"]   = g.estimated_ttft_ms;
    jg["estimated_tpot_ms"]   = g.estimated_tpot_ms;
    jg["peak_memory_bytes"]   = g.peak_memory_bytes;
    jg["weight_bytes"]        = g.weight_bytes;
    jg["kernel_ids"]          = g.kernel_ids;
    garr.push_back(jg);
  }
  j["graphs"] = garr;

  json karr = json::array();
  for (auto& k : kernels) {
    json jk;
    jk["id"]         = k.id;
    jk["target"]     = kernelTargetToString(k.target);
    jk["path"]       = k.path;
    jk["size_bytes"] = k.size_bytes;
    jk["op_pattern"] = k.op_pattern;
    jk["is_fused"]   = k.is_fused;
    karr.push_back(jk);
  }
  j["kernels"] = karr;

  json warr = json::array();
  for (auto& w : weights) {
    json jw;
    jw["id"]             = w.id;
    jw["path"]           = w.path;
    jw["dtype"]          = w.dtype;
    jw["size_bytes"]     = w.size_bytes;
    jw["mmap_safe"]      = w.mmap_safe;
    jw["page_alignment"] = w.page_alignment;
    warr.push_back(jw);
  }
  j["weights"] = warr;

  j["tokenizer_path"]    = tokenizer_path;
  j["selector_wasm_path"] = selector_wasm_path;
  j["runtime_config_path"] = runtime_config_path;
  j["memory_maps_path"]  = memory_maps_path;

  return pretty ? j.dump(2) : j.dump();
}

// ---------------------------------------------------------------------------
// RuntimeConfig: toJSON / fromJSON
// ---------------------------------------------------------------------------
const char* kernelTargetToString(KernelTarget t) {
  switch (t) {
    case KernelTarget::CPU_NEON:        return "cpu_neon";
    case KernelTarget::CPU_SVE:         return "cpu_sve";
    case KernelTarget::CPU_SVE2:        return "cpu_sve2";
    case KernelTarget::GPU_Vulkan:      return "gpu_vulkan";
    case KernelTarget::GPU_OpenCL:      return "gpu_opencl";
    case KernelTarget::GPU_Metal:       return "gpu_metal";
    case KernelTarget::NPU_HTP:         return "npu_htp";
    case KernelTarget::NPU_NeuroPilot:  return "npu_neuripilot";
    case KernelTarget::NPU_ANE:         return "npu_ane";
    case KernelTarget::NPU_Xclipse:     return "npu_xclipse";
    default: return "unknown";
  }
}

std::string RuntimeConfig::toJSON(bool pretty) const {
  json j;
  j["temperature"]        = temperature;
  j["top_p"]              = top_p;
  j["top_k"]              = top_k;
  j["repetition_penalty"] = repetition_penalty;
  j["max_new_tokens"]     = max_new_tokens;
  j["min_new_tokens"]     = min_new_tokens;
  j["bos_token_id"]       = bos_token_id;
  j["eos_token_id"]       = eos_token_id;
  j["pad_token_id"]       = pad_token_id;
  j["chat_template"]      = chat_template;
  return pretty ? j.dump(2) : j.dump();
}

RuntimeConfig RuntimeConfig::fromJSON(const std::string& s) {
  RuntimeConfig cfg;
  try {
    auto j = json::parse(s);
    cfg.temperature        = j.value("temperature", 1.0f);
    cfg.top_p              = j.value("top_p", 0.9f);
    cfg.top_k              = j.value("top_k", 50);
    cfg.repetition_penalty = j.value("repetition_penalty", 1.0f);
    cfg.max_new_tokens     = j.value("max_new_tokens", 512);
    cfg.min_new_tokens     = j.value("min_new_tokens", 0);
    cfg.bos_token_id       = j.value("bos_token_id", 1);
    cfg.eos_token_id       = j.value("eos_token_id", 2);
    cfg.pad_token_id       = j.value("pad_token_id", 0);
    cfg.chat_template      = j.value("chat_template", std::string(""));
  } catch (...) {}
  return cfg;
}

} // namespace pkg
} // namespace armcc
