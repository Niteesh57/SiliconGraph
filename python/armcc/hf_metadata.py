"""
ARM AI Compiler — HuggingFace Metadata Extractor
Extracts tokenizer assets, generation config, and model card metadata
and writes them to the output directory for packaging into .armpack.
"""

from __future__ import annotations
import json
import shutil
import logging
from pathlib import Path
from typing import Optional

from .model_loader import LoadedModel

logger = logging.getLogger("armcc.hf_metadata")


class HFMetadataExtractor:
    """Extracts and saves all metadata needed for the .armpack tokenizer/ dir."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def extract(self, loaded: LoadedModel) -> dict:
        """Extract all metadata and write tokenizer assets to output_dir.

        Returns:
            dict with runtime_config (sampling params, special tokens, etc.)
        """
        tok_dir = self.output_dir / "tokenizer"
        tok_dir.mkdir(parents=True, exist_ok=True)

        # Save tokenizer files
        try:
            loaded.tokenizer.save_pretrained(str(tok_dir))
            logger.info(f"  Tokenizer saved → {tok_dir}")
        except Exception as e:
            logger.warning(f"  Could not save tokenizer: {e}")

        # Build runtime config
        config = loaded.config
        gen_config = getattr(config, "generation_config", None)

        runtime_config = {
            "temperature":         1.0,
            "top_p":               0.9,
            "top_k":               50,
            "repetition_penalty":  1.0,
            "max_new_tokens":      512,
            "min_new_tokens":      0,
            "bos_token_id":        int(getattr(config, "bos_token_id", 1) or 1),
            "eos_token_id":        int(getattr(config, "eos_token_id", 2) or 2),
            "pad_token_id":        int(getattr(config, "pad_token_id", 0) or 0),
            "chat_template":       getattr(loaded.tokenizer, "chat_template", "") or "",
        }

        if gen_config:
            runtime_config.update({
                "temperature": float(getattr(gen_config, "temperature", 1.0) or 1.0),
                "top_p":       float(getattr(gen_config, "top_p", 0.9) or 0.9),
                "top_k":       int(getattr(gen_config, "top_k", 50) or 50),
            })

        # Save runtime config
        rt_path = self.output_dir / "runtime_config.json"
        with open(rt_path, "w") as f:
            json.dump(runtime_config, f, indent=2)
        logger.info(f"  Runtime config saved → {rt_path}")

        # Save model architecture summary
        arch = {
            "model_id":    loaded.model_id,
            "arch_family": loaded.arch_family,
            "num_layers":  loaded.num_layers,
            "hidden_size": loaded.hidden_size,
            "num_heads":   loaded.num_heads,
            "num_kv_heads": loaded.num_kv_heads,
            "intermediate_size": loaded.intermediate_size,
            "vocab_size":  loaded.vocab_size,
            "max_position_embeddings": loaded.max_position_embeddings,
            "total_params": loaded.total_params,
            "total_bytes":  loaded.total_bytes,
        }
        arch_path = self.output_dir / "model_arch.json"
        with open(arch_path, "w") as f:
            json.dump(arch, f, indent=2)

        return runtime_config
