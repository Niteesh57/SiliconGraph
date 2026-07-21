from types import SimpleNamespace

from types import SimpleNamespace

import torch

from armcc.graph_exporter import ExportedGraph, GraphExporter


def test_export_inputs_disable_transformers_dynamic_cache(tmp_path):
    loaded = SimpleNamespace(vocab_size=256)
    inputs = GraphExporter(tmp_path)._buildExampleInputs(
        loaded=loaded,
        context_length=32,
        batch_size=1,
    )

    assert inputs["input_ids"].shape == (1, 32)
    assert inputs["attention_mask"].shape == (1, 32)
    assert inputs["use_cache"] is False
    assert inputs["return_dict"] is False


def test_export_conversion_reads_object_tensor_metadata(tmp_path):
    placeholder = SimpleNamespace(
        op="placeholder",
        name="input_ids",
        meta={"tensor_meta": SimpleNamespace(shape=(1, 4), dtype=torch.int64)},
    )
    output = SimpleNamespace(op="output", args=([placeholder],))
    exported_program = SimpleNamespace(
        graph=SimpleNamespace(nodes=[placeholder, output]),
    )
    graph = ExportedGraph(
        model_id="test/model",
        arch_family="llama_style",
        num_layers=1,
        hidden_size=4,
        num_heads=1,
        num_kv_heads=1,
        intermediate_size=8,
        vocab_size=32,
        max_position_embeddings=16,
    )

    GraphExporter(tmp_path)._convertExportedProgram(exported_program, graph, None)

    assert graph.tensors[0]["shape"] == [1, 4]
    assert graph.tensors[0]["dtype"] == "i32"
    assert graph.input_ids == [1]
    assert graph.output_ids == [1]
