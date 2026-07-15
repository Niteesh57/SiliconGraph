// ============================================================================
// ARM AI Compiler — Manifest Writer (standalone writer utility)
// Provides formatted manifest writing, diff, and validation.
// ============================================================================
#include "package/package_generator.h"
#include <nlohmann/json.hpp>
#include <sstream>
#include <algorithm>

namespace armcc {
namespace pkg {
namespace manifest {

// ---------------------------------------------------------------------------
// Validate: Check manifest has required fields
// ---------------------------------------------------------------------------
std::vector<std::string> validate(const PackageManifest& m) {
  std::vector<std::string> errors;

  if (m.model_id.empty())
    errors.push_back("model_id is empty");
  if (m.compiler_version.empty())
    errors.push_back("compiler_version is empty");
  if (m.graphs.empty())
    errors.push_back("No graphs compiled (graphs[] is empty)");
  if (m.weights.empty())
    errors.push_back("No weight entries (weights[] is empty)");

  for (size_t i = 0; i < m.graphs.size(); ++i) {
    auto& g = m.graphs[i];
    if (g.id.empty())   errors.push_back("Graph " + std::to_string(i) + ": empty id");
    if (g.path.empty()) errors.push_back("Graph " + std::to_string(i) + ": empty path");
    if (g.estimated_tpot_ms < 0)
      errors.push_back("Graph " + std::to_string(i) + ": negative TPOT");
  }

  return errors;
}

// ---------------------------------------------------------------------------
// Diff: Show differences between two manifests (for version comparison)
// ---------------------------------------------------------------------------
std::string diff(const PackageManifest& a, const PackageManifest& b) {
  std::ostringstream oss;
  oss << "--- " << a.model_id << " v" << a.compiler_version << "\n";
  oss << "+++ " << b.model_id << " v" << b.compiler_version << "\n";

  if (a.graphs.size() != b.graphs.size()) {
    oss << "  graphs: " << a.graphs.size() << " → " << b.graphs.size() << "\n";
  }

  float avg_tpot_a = 0, avg_tpot_b = 0;
  for (auto& g : a.graphs) avg_tpot_a += g.estimated_tpot_ms;
  for (auto& g : b.graphs) avg_tpot_b += g.estimated_tpot_ms;
  if (!a.graphs.empty()) avg_tpot_a /= a.graphs.size();
  if (!b.graphs.empty()) avg_tpot_b /= b.graphs.size();

  oss << "  avg TPOT: " << avg_tpot_a << "ms → " << avg_tpot_b << "ms"
      << " (" << (avg_tpot_b < avg_tpot_a ? "-" : "+")
      << std::abs(avg_tpot_b - avg_tpot_a) << "ms)\n";

  return oss.str();
}

// ---------------------------------------------------------------------------
// Summary: Print a human-readable summary table
// ---------------------------------------------------------------------------
std::string summary(const PackageManifest& m) {
  std::ostringstream oss;
  oss << ".armpack Summary\n";
  oss << "  Model:    " << m.model_id << "\n";
  oss << "  Family:   " << m.model_family << "\n";
  oss << "  Compiler: " << m.compiler_version << "\n";
  oss << "  Created:  " << m.created_at << "\n";
  oss << "  Arch: "
      << m.num_layers << "L × "
      << m.hidden_size << "h × "
      << m.num_heads << " heads\n";
  oss << "\n";
  oss << "  Graphs (" << m.graphs.size() << "):\n";
  for (auto& g : m.graphs) {
    oss << "    [" << g.index << "] " << g.id << "\n";
    oss << "         " << g.soc_name << " " << g.quant_dtype
        << " ctx=" << g.context_length
        << " TTFT=" << g.estimated_ttft_ms << "ms"
        << " TPOT=" << g.estimated_tpot_ms << "ms"
        << " Peak=" << g.peak_memory_bytes/1024/1024 << "MB\n";
  }
  oss << "\n  Kernels: " << m.kernels.size() << "\n";
  oss << "  Weights: " << m.weights.size() << "\n";
  if (!m.weights.empty()) {
    uint64_t total_w = 0;
    for (auto& w : m.weights) total_w += w.size_bytes;
    oss << "  Total weight bytes: " << total_w/1024/1024 << "MB\n";
  }
  return oss.str();
}

} // namespace manifest
} // namespace pkg
} // namespace armcc
