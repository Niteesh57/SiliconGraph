"""Run a text prompt against the Hugging Face model named in an .armpack.

This is a host-side model-output check. It does not execute the .armpack on an
Android phone; that requires the pending Android inference runtime.

Example:
  python scripts/run_prompt.py --package output/model.armpack --prompt "What is 2 + 2?"
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path


def model_id_from_package(package: Path) -> str:
    with zipfile.ZipFile(package) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    model_id = manifest.get("model_id")
    if not model_id:
        raise ValueError("manifest.json does not contain model_id")
    return model_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a prompt using an .armpack model's HF source")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--package", type=Path, help="Compiled .armpack file")
    source.add_argument("--model", help="Hugging Face model ID")
    parser.add_argument("--prompt", required=True, help="Prompt to send to the model")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Fail instead of downloading a model that is not cached")
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            parser.error("CUDA was selected but is not available")

    if args.package:
        if not args.package.is_file():
            parser.error(f"package not found: {args.package}")
        model_id = model_id_from_package(args.package)
    else:
        model_id = args.model

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Host-side model-output test; this does not run Android .armpack kernels.")
    print(f"Model: {model_id}")
    print(f"Device: {args.device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()

    messages = [{"role": "user", "content": args.prompt}]
    if tokenizer.chat_template:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {name: value.to(args.device) for name, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    print("\nOutput:\n")
    print(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
