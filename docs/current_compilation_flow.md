# Current Compilation Flow

This document describes the flow implemented in the repository today. It is
intended as an implementation map, not a target architecture. In particular,
some package contents are deliberately Phase 1 placeholders (called out below).

## Entry points

There are two command-line layers:

- `python/armcc/cli.py` exposes the user-facing `armcc` command. Its `compile`
  subcommand accepts a HuggingFace model ID or local model directory.
- `compiler/tools/armcc/main.cpp` implements the C++ compiler driver. The
  Python CLI invokes it as `armcc-compiler compile --graph <json> ...` after
  preparing inputs.

The Python `inspect` implementation reads `manifest.json` directly from a ZIP
archive. The C++ `inspect` command is currently only a stub. `arm-ir-dump`
uses exported graph JSON as its input; despite the README example, the C++
driver does not currently load a graph from an `.armpack` file.

## End-to-end compile path

```text
HuggingFace model ID / local model
        |
        v
Python CLI (`armcc compile`)
  1. Load model, config, and tokenizer with Transformers
  2. Extract model/runtime metadata and save tokenizer assets to a temp folder
  3. Export a Torch graph to `exported_graph.json`; save F16 weights separately
  4. Optionally collect calibration statistics into `calibration.json`
  5. Start the C++ compiler driver
        |
        v
C++ compiler (`armcc-compiler compile`)
  6. Parse exported JSON into `IRGraph`
  7. Load JSON device profiles and analyze the source graph
  8. Generate graph variants for the requested SoCs and precision choices
  9. Produce a memory plan and write a ZIP-based `.armpack`
```

The temporary work directory is created by Python with an `armcc_` prefix. It
contains the graph JSON, calibration file when enabled, saved tokenizer
artifacts, and `weights_f16.bin`.

## Python preparation layer

`ModelLoader` loads `AutoConfig`, `AutoTokenizer`, and
`AutoModelForCausalLM` in evaluation mode. It derives the model family and
dimensions from the HuggingFace configuration.

`HFMetadataExtractor` writes tokenizer/runtime metadata. `GraphExporter`
attempts `torch.export` and converts its FX nodes to the repository's JSON
representation. The JSON contains model architecture fields, tensor records,
node records, graph inputs/outputs, and the path to the separately saved
weight file. If export fails, it emits a minimal skeletal `GEMM` graph so the
remaining compiler flow can still execute.

For INT8, INT4, and `mixed` requests, `CalibrationRunner` can gather tensor
statistics. The JSON path is passed to the C++ driver as `--calibration`.

## C++ compiler flow

`graphFromJSON()` converts the exported tensors and nodes into `IRGraph`, maps
the recognized architecture family, and computes topological order. The driver
then loads profiles from `device_profiles/` through `CostModel` and runs
`GraphAnalyzer` on the source graph.

The driver creates a `GraphFamilySpec` from `--targets` and `--quant`. It first
runs the default pass pipeline once for each target as a reporting/preflight
step. It then calls `GraphFamilyGenerator`, which clones the source graph and
runs the condition-specific pipeline for every generated variant.

By default, `GraphFamilySpec` expands each selected SoC across:

- precision (the requested type, or INT8 and INT4 for `mixed`)
- context lengths: 512, 2048, and 8192
- memory budgets: 512, 1024, 2048, and 4096 MB
- thermal states: nominal and hot
- latency modes: interactive and background

Mixed-precision variants are additionally created for INT4 and INT8. The
generator skips unsupported profile/dtype combinations and budgets judged too
small, and only prunes when `max_graphs` is set above zero.

## Optimization pipeline

For each graph variant, the pass manager applies passes in this order:

1. Shape inference, constant folding, dead-code elimination, operator fusion,
   and KV-cache planning.
2. Streaming attention for context lengths of 4096 or more.
3. Quantization: INT8, INT4 (with optional mixed precision), or FP8.
4. Tensor lifetime analysis, memory aliasing, memory layout, zero-copy, and
   prefetch insertion.
5. Hardware fusion/scheduling.
6. Device-specific layout optimization, kernel selection, tiling, and
   vectorization.

Each compiled graph stores its condition key, serialized IR, estimated TTFT /
TPOT, and peak-memory estimate. The cost model estimates per-node execution
cost using the assigned execution unit.

## Memory planning and package output

The top-level driver computes one `MemoryPlan` from the original source graph.
`PackageGenerator` writes the output ZIP in this order:

1. `graphs/graph_<n>.armgraph` and graph manifest entries
2. `weights/weights_<dtype>.bin`
3. kernel entries
4. tokenizer assets
5. `runtime_config.json`
6. `memory_maps/weights_layout.json` and `memory_maps/kv_cache_layout.json`
7. `selector_index.json`, `selector_src.c`, and `selector.wasm`
8. `manifest.json`

The manifest is the package index. It associates every graph with its SoC,
precision, memory budget, context length, thermal state, latency estimate, and
package path.

## Current limitations and hand-off gaps

These are important when using or extending the flow:

- The Python CLI exports only the first value supplied to `--context-lengths`.
  It does not forward the requested list to the C++ graph-family spec, which
  instead uses its built-in context-length defaults.
- Python creates calibration data and records the saved weight file in JSON,
  but the current C++ driver parses neither calibration data nor `weight_files`.
  As a result, the package generator commonly writes a placeholder weight blob.
- The Python tokenizer output directory is not passed as
  `PackageGeneratorOptions::tokenizer_dir`, so packages use the tokenizer
  placeholder unless the C++ caller is extended.
- Kernel files and `selector.wasm` are placeholder stubs in the package
  generator. `selector_src.c` and `selector_index.json` are emitted to support
  a later WASM compilation step.
- The memory plan is created from the unoptimized source graph rather than one
  plan per compiled graph variant.
- The graph family can be large: before skips/pruning, a non-mixed single-SoC,
  single-dtype request produces 48 variants (3 contexts x 4 budgets x 2
  thermal states x 2 latency modes). No max-graph limit is set by the CLI.

## Relevant source map

| Concern | Main implementation |
| --- | --- |
| Python orchestration | `python/armcc/cli.py` |
| Model loading | `python/armcc/model_loader.py` |
| Torch export and JSON interchange | `python/armcc/graph_exporter.py` |
| C++ driver and JSON-to-IR conversion | `compiler/tools/armcc/main.cpp` |
| IR structure | `compiler/include/arm_ir/graph.h` |
| Pass ordering | `compiler/lib/passes/pass_manager.cpp` |
| Variant expansion | `compiler/lib/generator/graph_family_generator.cpp` |
| Memory planning | `compiler/lib/memory/memory_planner.cpp` |
| Archive generation | `compiler/lib/package/package_generator.cpp` |
| Intended archive format | `docs/package_format.md` |
