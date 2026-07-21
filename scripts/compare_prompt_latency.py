"""Compare the same prompt on Hugging Face and an executable .armpack runtime.

The optimized runtime command must print only the generated text to stdout and
may use these placeholders in any argument: {package}, {prompt},
{max_new_tokens}. It is intentionally required so the script never labels a
Hugging Face rerun as an optimized-device result.

Example, once an Android/package runner exists:
  python scripts/compare_prompt_latency.py ^
    --package output/model.armpack ^
    --prompt "Explain KV cache in one sentence." ^
    --optimized-command C:/tools/armpack-runner.exe --package {package} --prompt {prompt}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def package_model_id(package: Path) -> str:
    with zipfile.ZipFile(package) as archive:
        return json.loads(archive.read("manifest.json"))["model_id"]


def generate_huggingface(model_id: str, prompt: str, max_new_tokens: int,
                         device: str, local_files_only: bool) -> tuple[str, float]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    ).to(device).eval()

    messages = [{"role": "user", "content": prompt}]
    if tokenizer.chat_template:
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}

    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip(), elapsed


def run_optimized(command_template: list[str], package: Path, prompt: str,
                  max_new_tokens: int) -> tuple[str, float]:
    values = {
        "package": str(package.resolve()),
        "prompt": prompt,
        "max_new_tokens": str(max_new_tokens),
    }
    command = [item.format(**values) for item in command_template]
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"optimized runner exited {result.returncode}")
    return result.stdout.strip(), elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Hugging Face and .armpack runtime latency")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--optimized-command", nargs="+", required=True,
        help="Runtime executable and arguments; supports {package}, {prompt}, {max_new_tokens}",
    )
    args = parser.parse_args()

    if not args.package.is_file():
        parser.error(f"package not found: {args.package}")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            parser.error("CUDA was selected but is not available")

    model_id = package_model_id(args.package)
    print(f"Model: {model_id}")
    print("Running Hugging Face baseline...")
    hf_output, hf_seconds = generate_huggingface(
        model_id, args.prompt, args.max_new_tokens, args.device, args.local_files_only,
    )

    print("Running optimized package runtime...")
    optimized_output, optimized_seconds = run_optimized(
        args.optimized_command, args.package, args.prompt, args.max_new_tokens,
    )

    faster = "optimized runtime" if optimized_seconds < hf_seconds else "Hugging Face baseline"
    ratio = max(hf_seconds, optimized_seconds) / max(min(hf_seconds, optimized_seconds), 1e-9)
    print("\nResults")
    print(f"  Hugging Face:      {hf_seconds:.3f} s")
    print(f"  Optimized runtime: {optimized_seconds:.3f} s")
    print(f"  Faster:            {faster} ({ratio:.2f}x)")
    print("\nHugging Face output:\n" + hf_output)
    print("\nOptimized output:\n" + optimized_output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
