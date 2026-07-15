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

from .model_loader   import ModelLoader
from .graph_exporter import GraphExporter
from .calibration    import CalibrationRunner
from .hf_metadata    import HFMetadataExtractor

__all__ = [
    "ModelLoader",
    "GraphExporter",
    "CalibrationRunner",
    "HFMetadataExtractor",
]
