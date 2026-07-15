// ============================================================================
// ARM AI Compiler — Condition Matrix (implementation)
//
// Builds the condition matrix that maps device state dimensions to
// graph indices. Used to generate the selector.wasm logic at compile time.
// ============================================================================
#include "generator/graph_family_generator.h"
#include "generator/graph_selector.h"
#include <sstream>
#include <algorithm>

namespace armcc {
namespace generator {

// ---------------------------------------------------------------------------
// Build a GraphSelector from a compiled family
// ---------------------------------------------------------------------------
GraphSelector buildSelectorFromFamily(const std::vector<CompiledGraph>& family) {
  std::vector<GraphSelectorEntry> entries;
  entries.reserve(family.size());

  for (uint32_t i = 0; i < (uint32_t)family.size(); ++i) {
    const CompiledGraph& cg = family[i];
    GraphSelectorEntry e;
    e.graph_index        = i;
    e.conditions         = cg.conditions;
    e.estimated_tpot_ms  = cg.estimated_tpot_ms;
    e.peak_memory_bytes  = cg.peak_memory_bytes;
    entries.push_back(e);
  }

  return GraphSelector(std::move(entries));
}

// ---------------------------------------------------------------------------
// Generate a compact C decision table (for embedding in selector.wasm)
// ---------------------------------------------------------------------------
std::string generateSelectorC(const std::vector<CompiledGraph>& family) {
  std::ostringstream oss;
  oss << "// ARM AI Compiler — Auto-generated Graph Selector\n"
      << "// DO NOT EDIT — regenerate with armcc compile\n\n"
      << "#include <stdint.h>\n\n"
      << "typedef struct {\n"
      << "  uint16_t soc_id;\n"
      << "  uint8_t  quant_dtype;\n"
      << "  uint16_t memory_mb;\n"
      << "  uint16_t context_length;\n"
      << "  uint8_t  thermal;\n"
      << "  uint8_t  latency_mode;\n"
      << "  uint32_t graph_index;\n"
      << "  float    tpot_ms;\n"
      << "  uint64_t peak_mem_bytes;\n"
      << "} GraphEntry;\n\n";

  oss << "static const GraphEntry GRAPH_TABLE[] = {\n";
  for (uint32_t i = 0; i < (uint32_t)family.size(); ++i) {
    auto& cg = family[i];
    oss << "  { "
        << (uint16_t)cg.conditions.soc_id << ", "
        << (uint8_t)cg.conditions.quant_dtype << ", "
        << cg.conditions.memory_mb << ", "
        << cg.conditions.context_length << ", "
        << (uint8_t)cg.conditions.thermal << ", "
        << (uint8_t)cg.conditions.latency_mode << ", "
        << i << ", "
        << cg.estimated_tpot_ms << "f, "
        << cg.peak_memory_bytes << "ULL"
        << " },  // " << cg.conditions.toKey() << "\n";
  }
  oss << "};\n\n";
  oss << "static const uint32_t GRAPH_TABLE_SIZE = "
      << family.size() << ";\n\n";

  // Generate the selector function
  oss << "uint32_t select_graph(\n"
      << "  uint16_t soc_id, uint32_t ram_mb, int battery_pct,\n"
      << "  uint8_t thermal, uint8_t latency_mode, uint32_t context_length,\n"
      << "  int gpu_available, int npu_available)\n"
      << "{\n"
      << "  uint32_t best_idx = 0;\n"
      << "  float    best_score = 1e30f;\n"
      << "\n"
      << "  for (uint32_t i = 0; i < GRAPH_TABLE_SIZE; i++) {\n"
      << "    const GraphEntry* e = &GRAPH_TABLE[i];\n"
      << "    float score = 0.0f;\n"
      << "\n"
      << "    // Hard reject: OOM\n"
      << "    if (e->peak_mem_bytes > (uint64_t)ram_mb * 1024 * 1024 * 95 / 100)\n"
      << "      continue;\n"
      << "\n"
      << "    // SoC mismatch penalty\n"
      << "    if (e->soc_id != soc_id && e->soc_id != 0xFF) score += 200.0f;\n"
      << "\n"
      << "    // Context length penalty\n"
      << "    if (e->context_length < context_length) score += 500.0f;\n"
      << "    else score += (float)(e->context_length - context_length) * 0.001f;\n"
      << "\n"
      << "    // Thermal / latency\n"
      << "    if (e->thermal != thermal) score += 10.0f;\n"
      << "    if (e->latency_mode != latency_mode) score += 20.0f;\n"
      << "\n"
      << "    // Battery: prefer smaller quant when low battery\n"
      << "    if (battery_pct < 20 && e->quant_dtype == 8) score -= 15.0f;\n"
      << "\n"
      << "    score += e->tpot_ms * 0.5f;\n"
      << "\n"
      << "    if (score < best_score) {\n"
      << "      best_score = score;\n"
      << "      best_idx   = e->graph_index;\n"
      << "    }\n"
      << "  }\n"
      << "  return best_idx;\n"
      << "}\n";

  return oss.str();
}

} // namespace generator
} // namespace armcc
