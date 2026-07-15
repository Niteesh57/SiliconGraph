"""
ARM AI Compiler — Model Loader
Loads any HuggingFace model (Llama, Qwen, Gemma, Phi, SmolLM, etc.)
in PyTorch/Safetensors format and prepares it for graph export.
"""

from __future__ import annotations
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)

logger = logging.getLogger("armcc.model_loader")


@dataclass
class LoadedModel:
    """Container for a loaded HuggingFace model and its metadata."""
    model_id:   str
    model:      PreTrainedModel
    config:     PretrainedConfig
    tokenizer:  object  # PreTrainedTokenizer
    arch_family: str    # "llama_style" | "gemma_style" | "gpt2_style" | ...

    # Architectural dimensions (extracted from config)
    num_layers:              int = 0
    hidden_size:             int = 0
    num_heads:               int = 0
    num_kv_heads:            int = 0
    intermediate_size:       int = 0
    vocab_size:              int = 0
    max_position_embeddings: int = 0

    # Weight statistics for quantization
    weight_names: list[str] = field(default_factory=list)
    total_params:  int = 0
    total_bytes:   int = 0


class ModelLoader:
    """Loads a HuggingFace model into a LoadedModel container.

    Supports:
      - Llama 2/3, Mistral, Mixtral
      - Qwen 2/2.5
      - Gemma / Gemma 2
      - Phi-3 / Phi-3.5
      - SmolLM / SmolLM2
      - Any AutoModelForCausalLM-compatible model
    """

    # Model ID prefix → architecture family mapping
    ARCH_FAMILIES: dict[str, str] = {
        "llama":    "llama_style",
        "mistral":  "llama_style",
        "mixtral":  "llama_style",   # MoE
        "qwen":     "llama_style",   # Qwen2 uses Llama-style arch
        "phi":      "llama_style",   # Phi-3
        "gemma":    "gemma_style",
        "gpt2":     "gpt2_style",
        "smollm":   "gpt2_style",
        "t5":       "t5_style",
        "llava":    "vision_lm",
    }

    def __init__(self, cache_dir: Optional[str] = None, device: str = "cpu"):
        self.cache_dir = cache_dir
        self.device    = device

    def load(self,
             model_id: str,
             dtype: torch.dtype = torch.float16,
             trust_remote_code: bool = False) -> LoadedModel:
        """Load a HuggingFace model by ID or local path.

        Args:
            model_id: HuggingFace Hub model ID (e.g. 'meta-llama/Llama-3.2-1B')
                      or a local directory path.
            dtype: PyTorch dtype to load weights in (default: float16).
            trust_remote_code: Allow custom model code from HF Hub.

        Returns:
            LoadedModel with the model, config, tokenizer, and metadata.
        """
        logger.info(f"Loading model: {model_id}")

        # --- Load config first (fast, no weights) ---
        config = AutoConfig.from_pretrained(
            model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=trust_remote_code,
        )

        # --- Load tokenizer ---
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=trust_remote_code,
        )

        # --- Load model weights ---
        logger.info(f"  Loading weights (dtype={dtype}) ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,  # Stream weights, don't duplicate in RAM
        )
        model = model.eval()  # inference mode
        if self.device != "cpu":
            model = model.to(self.device)

        # --- Extract architectural metadata ---
        arch_family = self._detectFamily(model_id, config)
        dims = self._extractDimensions(config)

        # --- Weight inventory ---
        weight_names = [n for n, _ in model.named_parameters()]
        total_params = sum(p.numel() for p in model.parameters())
        total_bytes  = sum(p.numel() * p.element_size() for p in model.parameters())

        logger.info(f"  Loaded: {total_params/1e9:.2f}B params, "
                    f"{total_bytes/1e9:.2f}GB, family={arch_family}")

        return LoadedModel(
            model_id=model_id,
            model=model,
            config=config,
            tokenizer=tokenizer,
            arch_family=arch_family,
            weight_names=weight_names,
            total_params=total_params,
            total_bytes=total_bytes,
            **dims,
        )

    def _detectFamily(self, model_id: str, config: PretrainedConfig) -> str:
        """Detect the model architecture family from model_id and config."""
        model_type = getattr(config, "model_type", "").lower()
        model_id_lower = model_id.lower()

        for prefix, family in self.ARCH_FAMILIES.items():
            if prefix in model_id_lower or prefix in model_type:
                return family

        # Fallback: check config attributes
        if hasattr(config, "num_key_value_heads"):
            return "llama_style"
        if hasattr(config, "num_attention_heads"):
            return "gpt2_style"

        return "unknown"

    def _extractDimensions(self, config: PretrainedConfig) -> dict:
        """Extract numerical architecture dimensions from config."""
        def _get(*keys, default=0):
            for k in keys:
                v = getattr(config, k, None)
                if v is not None:
                    return int(v)
            return default

        return {
            "num_layers":              _get("num_hidden_layers", "n_layer"),
            "hidden_size":             _get("hidden_size", "n_embd"),
            "num_heads":               _get("num_attention_heads", "n_head"),
            "num_kv_heads":            _get("num_key_value_heads", "num_attention_heads"),
            "intermediate_size":       _get("intermediate_size", "n_inner"),
            "vocab_size":              _get("vocab_size"),
            "max_position_embeddings": _get("max_position_embeddings",
                                           "n_positions", default=2048),
        }
