// ============================================================================
// ARM AI Compiler — Package Generator
//
// Takes the compiled graph family + kernels + memory plan and writes
// a complete .armpack file (ZIP archive).
// ============================================================================
#pragma once

#include "package/armpack_format.h"
#include "generator/graph_family_generator.h"
#include "memory/memory_planner.h"
#include "arm_ir/graph.h"

#include <string>
#include <vector>
#include <functional>

namespace armcc {
namespace pkg {

// ---------------------------------------------------------------------------
// PackageGeneratorOptions
// ---------------------------------------------------------------------------
struct PackageGeneratorOptions {
  std::string output_path;           // path to output .armpack file
  bool        compress_graphs  = true;  // zlib compress .armgraph files
  bool        compress_kernels = false; // kernels are already binary
  bool        include_debug_ir = false; // embed human-readable IR dump
  int         compression_level = 6;    // zlib level (1-9)

  // Tokenizer directory (if present, gets copied into package)
  std::string tokenizer_dir;

  // Overwrite existing file?
  bool        overwrite        = true;
};

// ---------------------------------------------------------------------------
// PackageGenerator
// ---------------------------------------------------------------------------
class PackageGenerator {
public:
  explicit PackageGenerator(PackageGeneratorOptions opts);

  // Write a complete .armpack from a compiled graph family
  bool generate(
    const std::string&                       model_id,
    const ir::IRGraph&                       source_graph,
    const std::vector<generator::CompiledGraph>& graph_family,
    const memory::MemoryPlan&                mem_plan,
    const RuntimeConfig&                     runtime_cfg,
    std::function<void(int, const std::string&)> progress_cb = nullptr
  );

  // Verify an existing .armpack (checks manifest consistency, checksums)
  static bool verify(const std::string& armpack_path,
                     std::string& error_out);

  // Print a summary of the contents of an .armpack
  static void inspect(const std::string& armpack_path);

  const std::string& lastError() const { return lastError_; }

private:
  PackageGeneratorOptions opts_;
  std::string             lastError_;

  bool writeManifest(void* zip, const PackageManifest& manifest);
  bool writeGraphs(void* zip,
                   const std::vector<generator::CompiledGraph>& family,
                   PackageManifest& manifest);
  bool writeWeights(void* zip,
                    const ir::IRGraph& graph,
                    const memory::MemoryPlan& plan,
                    PackageManifest& manifest);
  bool writeKernels(void* zip, PackageManifest& manifest);
  bool writeTokenizer(void* zip, PackageManifest& manifest);
  bool writeRuntimeConfig(void* zip, const RuntimeConfig& cfg);
  bool writeMemoryMaps(void* zip, const memory::MemoryPlan& plan);
  bool writeSelectorWasm(void* zip,
                         const std::vector<generator::CompiledGraph>& family);
};

} // namespace pkg
} // namespace armcc
