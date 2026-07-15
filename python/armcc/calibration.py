"""
ARM AI Compiler — Calibration Runner
Runs a calibration dataset through the model to collect activation statistics
for post-training quantization (INT8/INT4 scale/zero-point computation).
"""

from __future__ import annotations
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Callable

import torch
import torch.nn as nn

from .model_loader import LoadedModel

logger = logging.getLogger("armcc.calibration")


@dataclass
class CalibrationStats:
    """Per-tensor quantization statistics collected during calibration."""
    name:    str
    min_val: float
    max_val: float
    mean:    float
    std:     float

    # Computed quantization parameters
    scale_int8:      float = 1.0
    zero_point_int8: int   = 0
    scale_int4:      float = 1.0
    zero_point_int4: int   = 0

    def computeQuantParams(self):
        """Compute INT8 and INT4 scale/zero-point from min/max stats."""
        # INT8: range [-128, 127]
        s8_range = self.max_val - self.min_val
        if s8_range > 0:
            self.scale_int8 = s8_range / 255.0
            self.zero_point_int8 = round(-128 - self.min_val / self.scale_int8)
            self.zero_point_int8 = max(-128, min(127, self.zero_point_int8))

        # INT4: range [-8, 7]
        if s8_range > 0:
            self.scale_int4 = s8_range / 15.0
            self.zero_point_int4 = round(-8 - self.min_val / self.scale_int4)
            self.zero_point_int4 = max(-8, min(7, self.zero_point_int4))


@dataclass
class CalibrationResult:
    """Output of the calibration pass."""
    model_id: str
    num_samples: int
    stats: dict[str, CalibrationStats] = field(default_factory=dict)

    # Per-layer quantization sensitivity (0.0=insensitive, 1.0=very sensitive)
    # Used by MixedPrecisionPass to decide INT4 vs FP16 per layer
    sensitivity: dict[str, float] = field(default_factory=dict)

    def toJSON(self) -> dict:
        return {
            "model_id":    self.model_id,
            "num_samples": self.num_samples,
            "stats": {
                name: {
                    "min": s.min_val, "max": s.max_val,
                    "mean": s.mean,   "std": s.std,
                    "scale_int8": s.scale_int8,
                    "zero_point_int8": s.zero_point_int8,
                    "scale_int4": s.scale_int4,
                    "zero_point_int4": s.zero_point_int4,
                }
                for name, s in self.stats.items()
            },
            "sensitivity": self.sensitivity,
        }


class CalibrationRunner:
    """
    Runs calibration data through a model to collect quantization statistics.

    Usage:
        runner = CalibrationRunner(loaded_model, num_samples=128)
        result = runner.run()
        runner.save(result, "calibration_stats.json")
    """

    def __init__(self,
                 loaded: LoadedModel,
                 num_samples: int = 128,
                 max_seq_len: int = 512,
                 dataset_name: str = "wikitext",
                 dataset_split: str = "train"):
        self.loaded       = loaded
        self.num_samples  = num_samples
        self.max_seq_len  = max_seq_len
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self._hooks: list = []
        self._stats: dict[str, CalibrationStats] = {}

    def run(self) -> CalibrationResult:
        """Run calibration and return statistics."""
        logger.info(f"Calibration: {self.num_samples} samples from '{self.dataset_name}'")

        # Load calibration dataset
        texts = self._loadCalibrationTexts()

        # Register forward hooks on Linear layers to capture activations
        self._registerHooks()

        model = self.loaded.model
        tokenizer = self.loaded.tokenizer

        model.eval()
        with torch.no_grad():
            for i, text in enumerate(texts[:self.num_samples]):
                if i % 16 == 0:
                    logger.info(f"  Calibrating sample {i}/{self.num_samples} ...")
                try:
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_seq_len,
                    )
                    _ = model(**inputs)
                except Exception as e:
                    logger.debug(f"  Sample {i} failed: {e}")
                    continue

        self._removeHooks()

        # Compute quant params from collected stats
        for stats in self._stats.values():
            stats.computeQuantParams()

        # Estimate per-layer sensitivity (simple heuristic: high variance = sensitive)
        sensitivity = {}
        for name, stats in self._stats.items():
            # Normalize std by mean to get coefficient of variation
            cv = abs(stats.std / stats.mean) if abs(stats.mean) > 1e-8 else 0.0
            # Clip to [0, 1]
            sensitivity[name] = min(1.0, cv / 2.0)

        result = CalibrationResult(
            model_id=self.loaded.model_id,
            num_samples=self.num_samples,
            stats=self._stats,
            sensitivity=sensitivity,
        )
        logger.info(f"  Calibration done: {len(self._stats)} tensors profiled")
        return result

    def save(self, result: CalibrationResult, path: str) -> None:
        """Save calibration results to JSON."""
        with open(path, "w") as f:
            json.dump(result.toJSON(), f, indent=2)
        logger.info(f"  Calibration saved → {path}")

    # ── Private ────────────────────────────────────────────────────────────────

    def _loadCalibrationTexts(self) -> list[str]:
        """Load calibration texts from a dataset or use built-in fallback."""
        try:
            from datasets import load_dataset
            ds = load_dataset(
                self.dataset_name,
                "wikitext-2-raw-v1" if self.dataset_name == "wikitext" else None,
                split=self.dataset_split,
                trust_remote_code=False,
            )
            texts = [row["text"] for row in ds
                     if row.get("text", "").strip()]
            logger.info(f"  Loaded {len(texts)} calibration texts from '{self.dataset_name}'")
            return texts
        except Exception as e:
            logger.warning(f"  Dataset load failed: {e}. Using built-in fallback texts.")
            return [
                "The quick brown fox jumps over the lazy dog. " * 20,
                "Artificial intelligence is transforming the way we interact with technology. " * 10,
                "Large language models are trained on vast amounts of text data. " * 10,
            ] * (self.num_samples // 3 + 1)

    def _registerHooks(self) -> None:
        """Register forward hooks on all Linear layers."""
        for name, module in self.loaded.model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(
                    self._makeHook(name)
                )
                self._hooks.append(hook)

    def _makeHook(self, name: str):
        def hook(module, inputs, output):
            tensor = output.detach().float()
            mn  = float(tensor.min())
            mx  = float(tensor.max())
            mean = float(tensor.mean())
            std  = float(tensor.std())
            if name not in self._stats:
                self._stats[name] = CalibrationStats(
                    name=name, min_val=mn, max_val=mx, mean=mean, std=std
                )
            else:
                s = self._stats[name]
                # Running min/max
                s.min_val = min(s.min_val, mn)
                s.max_val = max(s.max_val, mx)
                # Running mean (exponential moving average)
                alpha = 0.1
                s.mean = (1 - alpha) * s.mean + alpha * mean
                s.std  = (1 - alpha) * s.std  + alpha * std
        return hook

    def _removeHooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
