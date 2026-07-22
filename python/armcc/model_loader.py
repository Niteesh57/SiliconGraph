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
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)
from armcc.model_reference import normalize_model_reference
from armcc.modalities import ModalityCapabilities, detect_modalities

logger = logging.getLogger("armcc.model_loader")


@dataclass
class LoadedModel:
    """Container for a loaded HuggingFace model and its metadata."""
    model_id:   str
    model:      PreTrainedModel
    config:     PretrainedConfig
    tokenizer:  object  # PreTrainedTokenizer
    processor:  object | None  # AutoProcessor for image/audio/video models
    arch_family: str    # "llama_style" | "gemma_style" | "gpt2_style" | ...
    modalities: ModalityCapabilities

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
      - Text LLMs, vision-language models, and speech models whose model class
        is exposed by the installed Transformers version
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
        model_id = normalize_model_reference(model_id)
        logger.info(f"Loading model: {model_id}")

        # --- Load config first (fast, no weights) ---
        config = AutoConfig.from_pretrained(
            model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=trust_remote_code,
        )

        # --- Discover modality and load the most specific processor ---
        # AutoProcessor is required for vision/audio/video inputs.  A tokenizer
        # remains the fallback for text-only models and older Transformers.
        processor = self._loadProcessor(model_id, trust_remote_code)
        modalities = detect_modalities(config, processor)
        self._configureStaticMultimodalProcessor(config, processor, modalities)
        tokenizer = self._loadTokenizer(model_id, processor, trust_remote_code)

        # --- Load model weights ---
        logger.info(f"  Loading weights (dtype={dtype}) ...")
        model_class = self._modelClassFor(modalities)
        model = model_class.from_pretrained(
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
        arch_family = self._detectFamily(model_id, config, modalities)
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
            processor=processor,
            arch_family=arch_family,
            modalities=modalities,
            weight_names=weight_names,
            total_params=total_params,
            total_bytes=total_bytes,
            **dims,
        )

    def _loadProcessor(self, model_id: str, trust_remote_code: bool) -> object | None:
        try:
            return AutoProcessor.from_pretrained(
                model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=trust_remote_code,
            )
        except Exception as exc:
            logger.info("  No AutoProcessor available: %s", exc)
            return None

    def _loadTokenizer(self, model_id: str, processor: object | None,
                       trust_remote_code: bool) -> object:
        try:
            return AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            processor_tokenizer = getattr(processor, "tokenizer", None)
            if processor_tokenizer is not None:
                return processor_tokenizer
            raise

    @staticmethod
    def _configureStaticMultimodalProcessor(
        config: PretrainedConfig, processor: object | None,
        modalities: ModalityCapabilities,
    ) -> None:
        """Apply package-safe processor settings for static mobile graphs.

        Idefics3/SmolVLM normally splits one image into a variable number of
        tiles. That produces a variable count of visual tokens and cannot be a
        single fixed mobile graph. A one-image, unsplit 512px path is still a
        supported processor mode, keeps the image token budget bounded, and is
        saved with the package's processor assets for runtime parity.
        """
        if modalities.task != "image_text_to_text":
            return
        if str(getattr(config, "model_type", "")).lower() != "idefics3":
            return
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None and getattr(image_processor, "do_image_splitting", None):
            image_processor.do_image_splitting = False
            logger.info("  Disabled Idefics3 image splitting for the static one-image mobile graph")

    @staticmethod
    def _modelClassFor(modalities: ModalityCapabilities):
        """Choose a Transformers auto-model class without hard pinning a family.

        Imports are deliberately lazy: Transformers exposes some auto-model
        classes only in newer versions.  If an installed release lacks a
        specialised class, ``AutoModel`` still lets the compiler inspect and
        package the model instead of applying the invalid causal-LM loader.
        """
        if modalities.task in {"audio_text_to_text", "image_audio_text_to_text"}:
            # Audio/omni configs are not universally registered under the
            # vision-to-sequence auto class. AutoModel is the conservative
            # generic dispatch for their model-specific forward signature.
            return AutoModel
        if modalities.task in {"image_text_to_text", "video_text_to_text"}:
            try:
                # Current Transformers releases register Idefics3/SmolVLM
                # under this class. It includes the conditional-generation
                # language head, unlike the base AutoModel fallback.
                from transformers import AutoModelForImageTextToText
                return AutoModelForImageTextToText
            except ImportError:
                pass
            try:
                from transformers import AutoModelForVision2Seq
                return AutoModelForVision2Seq
            except ImportError:
                return AutoModel
        if modalities.task == "speech_to_text":
            try:
                from transformers import AutoModelForSpeechSeq2Seq
                return AutoModelForSpeechSeq2Seq
            except ImportError:
                return AutoModel
        if modalities.task == "text_to_audio":
            try:
                from transformers import AutoModelForTextToWaveform
                return AutoModelForTextToWaveform
            except ImportError:
                return AutoModel
        return AutoModelForCausalLM

    def _detectFamily(self, model_id: str, config: PretrainedConfig,
                      modalities: ModalityCapabilities | None = None) -> str:
        """Detect the model architecture family from model_id and config."""
        model_type = getattr(config, "model_type", "").lower()
        model_id_lower = model_id.lower()

        if modalities and modalities.task in {
            "image_text_to_text", "video_text_to_text", "image_audio_text_to_text",
        }:
            return "vision_lm"
        if modalities and modalities.task in {"speech_to_text", "audio_text_to_text"}:
            return "audio_lm"
        if modalities and modalities.task == "text_to_audio":
            return "audio_generation"
        if modalities and modalities.task == "text_to_image":
            return "image_generation"
        if modalities and modalities.task == "text_embedding":
            return "embedding"

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
