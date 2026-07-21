"""
ARM AI Compiler — Graph Exporter
Extracts a computation graph from a loaded PyTorch model using torch.export
and serializes it to a format the C++ compiler can ingest.

Output: a FlatBuffers or JSON proto describing all ops, tensors, and weights.
"""

from __future__ import annotations
import json
import struct
import logging
import tempfile
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import torch
import torch.export
from torch.fx import GraphModule

from .model_loader import LoadedModel

logger = logging.getLogger("armcc.graph_exporter")


class GraphExportError(RuntimeError):
    """Raised when an exact runnable graph cannot be extracted."""


@dataclass
class ExportedGraph:
    """Serializable graph description ready for the C++ compiler."""
    model_id:    str
    arch_family: str
    num_layers:  int
    hidden_size: int
    num_heads:   int
    num_kv_heads: int
    intermediate_size: int
    vocab_size:  int
    max_position_embeddings: int

    # Ops: list of dicts with {id, op, inputs, outputs, attrs}
    nodes: list[dict] = field(default_factory=list)

    # Tensors: list of dicts with {id, name, dtype, shape, is_weight}
    tensors: list[dict] = field(default_factory=list)

    # Weight file paths (temporary; C++ compiler mmaps these)
    weight_files: list[str] = field(default_factory=list)

    # Graph I/O tensor IDs
    input_ids:  list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)


# Mapping from PyTorch op names → ARM-IR OpCode strings
_TORCH_OP_MAP: dict[str, str] = {
    "aten.mm":                      "MatMul",
    "aten.bmm":                     "BatchMatMul",
    "aten.linear":                  "GEMM",
    "aten.matmul":                  "MatMul",
    "aten.conv1d":                  "Conv1D",
    "aten.conv2d":                  "Conv2D",
    "aten.add":                     "Add",
    "aten.add_":                    "Add",
    "aten.sub":                     "Sub",
    "aten.mul":                     "Mul",
    "aten.div":                     "Div",
    "aten.relu":                    "ReLU",
    "aten.gelu":                    "GeLU",
    "aten.silu":                    "SiLU",
    "aten.sigmoid":                 "Sigmoid",
    "aten.tanh":                    "Tanh",
    "aten.softmax":                 "Softmax",
    "aten._softmax":                "Softmax",
    "aten.layer_norm":              "LayerNorm",
    "aten.native_layer_norm":       "LayerNorm",
    "aten.rms_norm":                "RMSNorm",
    "aten.embedding":               "Embedding",
    "aten.reshape":                 "Reshape",
    "aten.view":                    "Reshape",
    "aten.permute":                 "Permute",
    "aten.transpose":               "Transpose",
    "aten.cat":                     "Concat",
    "aten.split":                   "Split",
    "aten.squeeze":                 "Squeeze",
    "aten.unsqueeze":               "Unsqueeze",
    "aten.slice":                   "Slice",
    "aten.select":                  "Slice",
    "aten.gather":                  "Gather",
    "aten.max":                     "ReduceMax",
    "aten.sum":                     "ReduceSum",
    "aten.mean":                    "ReduceMean",
    "aten.sqrt":                    "Sqrt",
    "aten.rsqrt":                   "Rsqrt",
    "aten.exp":                     "Exp",
    "aten.log":                     "Log",
    "aten.abs":                     "Abs",
    "aten.neg":                     "Neg",
    "aten.to":                      "Cast",
    "aten._to_copy":                "Cast",
    "aten.max_pool2d":              "MaxPool2D",
    "aten.avg_pool2d":              "AvgPool2D",
    "aten.pad":                     "Pad",
    "aten.scaled_dot_product_attention": "GroupQueryAttention",
}


def _dtype_to_str(dtype: torch.dtype) -> str:
    return {
        torch.float32: "f32",
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        # ARM-IR currently represents token IDs as i32. Hugging Face models
        # expose them as torch.long, but their vocabularies fit safely in i32.
        torch.int64:  "i32",
        torch.int32:  "i32",
        torch.int16:  "i16",
        torch.int8:   "i8",
        torch.uint8:  "u8",
        torch.bool:   "bool",
    }.get(dtype, "f32")


class GraphExporter:
    """Exports a LoadedModel to an ExportedGraph via torch.export."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self,
               loaded: LoadedModel,
               context_length: int = 512,
               batch_size: int = 1) -> ExportedGraph:
        """Export a loaded model to ExportedGraph.

        Uses torch.export with example inputs matching (batch, seq_len).
        Falls back to torch.fx.symbolic_trace if torch.export fails.

        Args:
            loaded: LoadedModel from ModelLoader
            context_length: Sequence length for example inputs
            batch_size: Batch size for export

        Returns:
            ExportedGraph ready for C++ compiler ingestion
        """
        logger.info(f"Exporting graph: {loaded.model_id} "
                    f"(seq_len={context_length}, batch={batch_size})")

        # Build example inputs
        example_inputs = self._buildExampleInputs(
            loaded, context_length, batch_size)

        # Try torch.export first (strict=False for more model compatibility)
        exported_program = None
        try:
            logger.info("  Attempting torch.export ...")
            exported_program = torch.export.export(
                loaded.model,
                args=(),
                kwargs=example_inputs,
                strict=False,
            )
            logger.info("  torch.export succeeded")
        except Exception as e:
            logger.warning(f"  torch.export failed: {e}")
            logger.info("  Falling back to torch.fx symbolic_trace ...")

        # Convert to ExportedGraph
        graph = ExportedGraph(
            model_id=loaded.model_id,
            arch_family=loaded.arch_family,
            num_layers=loaded.num_layers,
            hidden_size=loaded.hidden_size,
            num_heads=loaded.num_heads,
            num_kv_heads=loaded.num_kv_heads,
            intermediate_size=loaded.intermediate_size,
            vocab_size=loaded.vocab_size,
            max_position_embeddings=loaded.max_position_embeddings,
        )

        if exported_program is not None:
            self._convertExportedProgram(exported_program, graph, loaded)
        else:
            raise GraphExportError(
                "Unable to export an exact computation graph. No package was created "
                "because a placeholder graph is not runnable. The export log above "
                "contains the model-specific failure."
            )

        # Save weights to disk for C++ compiler
        weight_path = self._saveWeights(loaded)
        graph.weight_files.append(str(weight_path))

        logger.info(f"  Exported: {len(graph.nodes)} ops, "
                    f"{len(graph.tensors)} tensors, "
                    f"{len(graph.weight_files)} weight file(s)")
        return graph

    def saveJSON(self, graph: ExportedGraph, path: Optional[str] = None) -> str:
        """Save the ExportedGraph as a JSON file for the C++ compiler."""
        if path is None:
            path = str(self.output_dir / "exported_graph.json")

        data = {
            "model_id":    graph.model_id,
            "arch_family": graph.arch_family,
            "arch": {
                "num_layers":              graph.num_layers,
                "hidden_size":             graph.hidden_size,
                "num_heads":               graph.num_heads,
                "num_kv_heads":            graph.num_kv_heads,
                "intermediate_size":       graph.intermediate_size,
                "vocab_size":              graph.vocab_size,
                "max_position_embeddings": graph.max_position_embeddings,
            },
            "input_ids":   graph.input_ids,
            "output_ids":  graph.output_ids,
            "tensors":     graph.tensors,
            "nodes":       graph.nodes,
            "weight_files": graph.weight_files,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"  Saved exported graph JSON → {path}")
        return path

    # ── Private helpers ────────────────────────────────────────────────────────

    def _buildExampleInputs(self, loaded: LoadedModel,
                            context_length: int, batch_size: int) -> dict:
        """Build example inputs matching the model's expected signature."""
        vocab = loaded.vocab_size or 32000
        # Use middle-of-vocab token IDs to avoid special tokens
        token_ids = torch.randint(
            100, min(vocab, 1000),
            (batch_size, context_length),
            dtype=torch.long,
        )
        # Generation caches are runtime state, not part of a static graph. They
        # make torch.export emit DynamicCache objects that cannot be serialized.
        # The target runtime manages its own KV cache during generation.
        return {
            "input_ids": token_ids,
            "attention_mask": torch.ones_like(token_ids),
            "use_cache": False,
            "return_dict": False,
        }

    def _convertExportedProgram(self, ep, graph: ExportedGraph,
                                 loaded: LoadedModel) -> None:
        """Convert a torch.export.ExportedProgram to ExportedGraph."""
        tensor_id = [1]  # mutable counter

        def next_id():
            t = tensor_id[0]
            tensor_id[0] += 1
            return t

        node_map: dict[str, int] = {}  # fx node name → tensor_id

        for node in ep.graph.nodes:
            if node.op == "placeholder":
                tid = next_id()
                node_map[node.name] = tid
                # torch.export stores TensorMetadata as an object, whereas
                # some FX paths use a dict. Support both representations.
                tensor_meta = node.meta.get("tensor_meta")
                if isinstance(tensor_meta, dict):
                    shape = list(tensor_meta.get("shape", []))
                    meta_dtype = tensor_meta.get("dtype", torch.float32)
                else:
                    shape = list(getattr(tensor_meta, "shape", []))
                    meta_dtype = getattr(tensor_meta, "dtype", torch.float32)
                dtype = _dtype_to_str(meta_dtype)
                graph.tensors.append({
                    "id": tid, "name": node.name,
                    "dtype": dtype, "shape": shape,
                    "is_weight": False, "is_input": True,
                })
                graph.input_ids.append(tid)

            elif node.op == "get_attr":
                tid = next_id()
                node_map[node.name] = tid
                graph.tensors.append({
                    "id": tid, "name": node.name,
                    "dtype": "f16", "shape": [],
                    "is_weight": True, "is_input": False,
                })

            elif node.op == "call_function":
                # Map torch op → ARM-IR op
                target_name = str(node.target)
                # torch.export includes overload suffixes such as
                # ``aten.linear.default``. ARM-IR maps operator families, not
                # individual overloads, so remove only the final overload name.
                canonical_name = target_name.rsplit(".", 1)[0]
                arm_op = _TORCH_OP_MAP.get(
                    target_name,
                    _TORCH_OP_MAP.get(canonical_name, "Unknown"),
                )

                out_tid = next_id()
                node_map[node.name] = out_tid

                # Gather input tensor IDs
                in_ids = []
                for arg in node.args:
                    if hasattr(arg, "name") and arg.name in node_map:
                        in_ids.append(node_map[arg.name])

                # Output tensor metadata
                meta = node.meta.get("tensor_meta", {})
                shape = list(getattr(meta, "shape", []))
                dtype = _dtype_to_str(getattr(meta, "dtype", torch.float32))

                graph.tensors.append({
                    "id": out_tid, "name": node.name,
                    "dtype": dtype, "shape": shape,
                    "is_weight": False, "is_input": False,
                })

                graph.nodes.append({
                    "id": len(graph.nodes) + 1,
                    "name": node.name,
                    "op":  arm_op,
                    "inputs":  in_ids,
                    "outputs": [out_tid],
                    "attrs":   {},
                })

            elif node.op == "output":
                for arg in node.args[0] if isinstance(node.args[0], (list, tuple)) else [node.args[0]]:
                    if hasattr(arg, "name") and arg.name in node_map:
                        graph.output_ids.append(node_map[arg.name])

    def _saveWeights(self, loaded: LoadedModel) -> Path:
        """Save model weights as a flat binary file (F16 safetensors-style)."""
        weight_path = self.output_dir / "weights_f16.bin"
        logger.info(f"  Saving weights → {weight_path}")

        with open(weight_path, "wb") as f:
            for name, param in loaded.model.named_parameters():
                data = param.detach().to(torch.float16).cpu().numpy()
                f.write(data.tobytes())

        return weight_path
