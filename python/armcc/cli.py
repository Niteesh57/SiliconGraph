"""
ARM AI Compiler — CLI Entry Point

Usage:
  armcc compile --model <model_id> --targets <targets> [OPTIONS]
  armcc inspect <package.armpack>
  armcc benchmark <package.armpack> --profile <profile>

Examples:
  armcc compile --model meta-llama/Llama-3.2-1B-Instruct \\
                --targets snapdragon_8_gen3,dimensity_9300,apple_a18 \\
                --quantization int4 --context-lengths 512,2048 \\
                --output ./llama-3.2-1b.armpack

  armcc compile --model HuggingFaceTB/SmolLM2-135M-Instruct \\
                --targets generic_arm64 --quantization int8 \\
                --output ./smollm2-135m.armpack

  armcc inspect ./smollm2-135m.armpack
"""

from __future__ import annotations
import argparse
import logging
import sys
import os
import json
import tempfile
from pathlib import Path

# Configure logging before any imports
logging.basicConfig(
    format="[armcc] %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("armcc.cli")


def _build_compile_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser("compile", help="Compile a HuggingFace model to .armpack")
    p.add_argument("--model", "-m", required=True,
                   help="HuggingFace model ID or local path")
    p.add_argument("--targets", "-t", default="generic_arm64",
                   help="Comma-separated list of target profiles "
                        "(e.g. snapdragon_8_gen3,dimensity_9300,apple_a18,generic_arm64)")
    p.add_argument("--quantization", "-q", default="int8",
                   choices=["fp32", "fp16", "int8", "int4", "fp8", "mixed"],
                   help="Quantization dtype (default: int8)")
    p.add_argument("--context-lengths", default="512,2048",
                   help="Comma-separated context lengths to optimize for "
                        "(default: 512,2048)")
    p.add_argument("--output", "-o", default=None,
                   help="Output .armpack file path (default: <model_name>.armpack)")
    p.add_argument("--calibration-samples", type=int, default=64,
                   help="Number of calibration samples for quantization (default: 64)")
    p.add_argument("--calibration-dataset", default="wikitext",
                   help="Calibration dataset name (default: wikitext)")
    p.add_argument("--skip-calibration", action="store_true",
                   help="Skip calibration (use default quant params)")
    p.add_argument("--no-mixed", action="store_true",
                   help="Disable mixed precision (use uniform quantization)")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Allow remote code for custom model architectures")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable verbose compiler output")
    return p


def _buildCppCompileCommand(
    armcc_bin: str,
    graph_json: str,
    targets: str,
    quantization: str,
    context_lengths: str,
    work_dir: str,
    output_path: str,
    calibration_path: str | None = None,
    verbose: bool = False,
) -> list[str]:
    """Build the C++ compiler command with all generated package assets."""
    work_path = Path(work_dir)
    cmd = [
        armcc_bin, "compile",
        "--graph", graph_json,
        "--targets", targets,
        "--quant", quantization,
        "--context-lengths", context_lengths,
        "--tokenizer-dir", str(work_path / "tokenizer"),
        "--runtime-config", str(work_path / "runtime_config.json"),
        "--output", output_path,
    ]
    if calibration_path:
        cmd += ["--calibration", calibration_path]
    if verbose:
        cmd += ["--verbose"]
    return cmd


def _cmd_compile(args) -> int:
    """Execute the compile command."""
    from armcc.model_loader    import ModelLoader
    from armcc.graph_exporter  import GraphExporter
    from armcc.calibration     import CalibrationRunner
    from armcc.hf_metadata     import HFMetadataExtractor

    import torch

    if args.verbose:
        logging.getLogger("armcc").setLevel(logging.DEBUG)

    # Determine output path
    model_slug = args.model.split("/")[-1].lower().replace("_", "-")
    output_path = args.output or f"./{model_slug}.armpack"
    targets = [t.strip() for t in args.targets.split(",")]
    context_lengths = [int(x) for x in args.context_lengths.split(",")]

    print(f"\n{'='*60}")
    print(f" ARM AI Compiler  v0.1.0")
    print(f"{'='*60}")
    print(f" Model   : {args.model}")
    print(f" Targets : {', '.join(targets)}")
    print(f" Quant   : {args.quantization}")
    print(f" Ctx len : {context_lengths}")
    print(f" Output  : {output_path}")
    print(f"{'='*60}\n")

    work_dir = tempfile.mkdtemp(prefix="armcc_")
    logger.info(f"Working directory: {work_dir}")

    # ── Step 1: Load model ────────────────────────────────────────────────────
    print("[1/5] Loading model from HuggingFace ...")
    loader = ModelLoader(cache_dir=None)
    loaded = loader.load(
        args.model,
        dtype=torch.float16,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"      ✓ {loaded.total_params/1e6:.0f}M params, "
          f"family={loaded.arch_family}")

    # ── Step 2: Extract metadata ──────────────────────────────────────────────
    print("[2/5] Extracting tokenizer and metadata ...")
    meta = HFMetadataExtractor(work_dir)
    runtime_config = meta.extract(loaded)
    print(f"      ✓ Tokenizer saved, runtime config generated")

    # ── Step 3: Export graph ──────────────────────────────────────────────────
    print(f"[3/5] Exporting computation graph (seq_len={context_lengths[0]}) ...")
    exporter = GraphExporter(work_dir)
    exported = exporter.export(loaded, context_length=context_lengths[0])
    graph_json = exporter.saveJSON(exported)
    print(f"      ✓ {len(exported.nodes)} ops exported")

    # ── Step 4: Calibration (optional) ───────────────────────────────────────
    calib_path = None
    if not args.skip_calibration and args.quantization in ("int8", "int4", "mixed"):
        print(f"[4/5] Running calibration ({args.calibration_samples} samples) ...")
        runner = CalibrationRunner(
            loaded,
            num_samples=args.calibration_samples,
            max_seq_len=context_lengths[0],
            dataset_name=args.calibration_dataset,
        )
        calib_result = runner.run()
        calib_path = os.path.join(work_dir, "calibration.json")
        runner.save(calib_result, calib_path)
        print(f"      ✓ {len(calib_result.stats)} tensors profiled")
    else:
        print("[4/5] Skipping calibration")

    # ── Step 5: Invoke C++ compiler ──────────────────────────────────────────
    print("[5/5] Running C++ optimization pipeline ...")
    print(f"      → Targets: {', '.join(targets)}")
    print(f"      → Quantization: {args.quantization}")
    print(f"      → This step requires the compiled armcc binary.")
    print(f"      → Graph JSON:   {graph_json}")
    print(f"      → Calib JSON:   {calib_path or '(skipped)'}")
    print(f"      → Output:       {output_path}")

    # Build the armcc C++ command
    armcc_bin = _findArmccBinary()
    if armcc_bin:
        import subprocess
        cmd = _buildCppCompileCommand(
            armcc_bin=armcc_bin,
            graph_json=graph_json,
            targets=args.targets,
            quantization=args.quantization,
            context_lengths=args.context_lengths,
            work_dir=work_dir,
            output_path=output_path,
            calibration_path=calib_path,
            verbose=args.verbose,
        )

        logger.info(f"Running: {' '.join(cmd)}")
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"\n✗ C++ compiler failed (exit code {ret.returncode})")
            return 1
    else:
        print("\n⚠  armcc C++ binary not found in PATH or build/.")
        print("   Exported graph and metadata are ready in:")
        print(f"   {work_dir}")
        print("   Build the C++ compiler first:")
        print("   cmake -B build -G Ninja && cmake --build build")
        return 0

    print(f"\n✓ Compiled successfully → {output_path}")
    return 0


def _cmd_inspect(args) -> int:
    """Print a summary of an .armpack file."""
    pack_path = args.package
    if not os.path.exists(pack_path):
        print(f"✗ File not found: {pack_path}")
        return 1

    import zipfile
    try:
        with zipfile.ZipFile(pack_path, "r") as zf:
            names = zf.namelist()
            manifest_data = None
            if "manifest.json" in names:
                manifest_data = json.loads(zf.read("manifest.json"))

        print(f"\n.armpack: {pack_path}")
        print(f"{'─'*50}")
        if manifest_data:
            print(f"  Model    : {manifest_data.get('model_id', 'unknown')}")
            print(f"  Family   : {manifest_data.get('model_family', 'unknown')}")
            print(f"  Compiler : {manifest_data.get('compiler_version', 'unknown')}")
            print(f"  Created  : {manifest_data.get('created_at', 'unknown')}")
            print(f"")
            print(f"  Graphs   : {len(manifest_data.get('graphs', []))}")
            for g in manifest_data.get("graphs", []):
                print(f"    [{g['index']:2d}] {g['id']}")
                print(f"         TTFT={g['estimated_ttft_ms']:.1f}ms, "
                      f"TPOT={g['estimated_tpot_ms']:.1f}ms, "
                      f"Peak={g['peak_memory_bytes']//1024//1024}MB")
            print(f"")
            print(f"  Kernels  : {len(manifest_data.get('kernels', []))}")
            print(f"  Weights  : {len(manifest_data.get('weights', []))}")
        else:
            print(f"  Files ({len(names)}):")
            for n in names[:20]:
                print(f"    {n}")
            if len(names) > 20:
                print(f"    ... and {len(names)-20} more")
    except Exception as e:
        print(f"✗ Could not read package: {e}")
        return 1
    return 0


def _findArmccBinary() -> str | None:
    """Find the compiled armcc C++ binary."""
    import shutil
    # Check PATH first
    found = shutil.which("armcc-compiler")
    if found:
        return found
    # Check common build directories
    for candidate in ["./build/compiler/tools/armcc/armcc",
                       "./build/bin/armcc",
                       "./out/armcc"]:
        if os.path.isfile(candidate):
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        prog="armcc",
        description="ARM AI Compiler — Universal AI Model Optimizer for ARM",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compile subcommand
    compile_p = _build_compile_parser(sub)
    compile_p.set_defaults(func=_cmd_compile)

    # inspect subcommand
    inspect_p = sub.add_parser("inspect", help="Inspect a compiled .armpack file")
    inspect_p.add_argument("package", help="Path to .armpack file")
    inspect_p.set_defaults(func=_cmd_inspect)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
