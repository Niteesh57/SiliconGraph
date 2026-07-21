"""
ARM AI Compiler — Python Ingestion Layer
Package: armcc

Entry point for the Python-side compiler tooling.
Handles:
  - HuggingFace model loading
  - torch.export graph extraction
  - Calibration dataset execution
  - Tokenizer + metadata extraction
  - CLI interface
"""

__version__ = "0.1.0"
__author__  = "ARM AI Compiler"

__all__ = [
    "ModelLoader",
    "GraphExporter",
    "CalibrationRunner",
    "HFMetadataExtractor",
]


def __getattr__(name: str):
    """Load heavyweight ML dependencies only when their public API is used."""
    if name == "ModelLoader":
        from .model_loader import ModelLoader
        return ModelLoader
    if name == "GraphExporter":
        from .graph_exporter import GraphExporter
        return GraphExporter
    if name == "CalibrationRunner":
        from .calibration import CalibrationRunner
        return CalibrationRunner
    if name == "HFMetadataExtractor":
        from .hf_metadata import HFMetadataExtractor
        return HFMetadataExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
