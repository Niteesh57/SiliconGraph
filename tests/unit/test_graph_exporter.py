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


def test_export_conversion_marks_unresolved_symbolic_shapes_as_dynamic(tmp_path):
    class SymbolicDimension:
        def __int__(self):
            raise RuntimeError("not statically known")

    placeholder = SimpleNamespace(
        op="placeholder",
        name="pixel_values",
        meta={"tensor_meta": SimpleNamespace(shape=(1, SymbolicDimension(), 3), dtype=torch.float16)},
    )
    output = SimpleNamespace(op="output", args=([placeholder],))
    exported_program = SimpleNamespace(graph=SimpleNamespace(nodes=[placeholder, output]))
    graph = ExportedGraph(
        model_id="test/model", arch_family="vision_lm", num_layers=1,
        hidden_size=4, num_heads=1, num_kv_heads=1, intermediate_size=8,
        vocab_size=32, max_position_embeddings=16,
    )

    GraphExporter(tmp_path)._convertExportedProgram(exported_program, graph, None)

    assert graph.tensors[0]["shape"] == [1, -1, 3]


def test_multimodal_export_uses_the_processor_chat_template_for_image_tokens(tmp_path):
    class Processor:
        def __init__(self):
            self.messages = None
            self.text = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            assert kwargs == {"add_generation_prompt": True, "tokenize": False}
            return "<image>Describe the supplied media."

        def __call__(self, **kwargs):
            self.text = kwargs["text"]
            assert len(kwargs["images"]) == 1
            return {"input_ids": torch.tensor([[1]]), "pixel_values": torch.zeros((1, 3, 2, 2))}

    processor = Processor()
    loaded = SimpleNamespace(
        processor=processor,
        modalities=SimpleNamespace(task="image_text_to_text", inputs=("text", "image")),
    )

    exported_inputs = GraphExporter(tmp_path)._buildMultimodalExampleInputs(loaded, batch_size=1)

    assert processor.messages[0]["content"][0] == {"type": "image"}
    assert processor.text == ["<image>Describe the supplied media."]
    assert set(exported_inputs) == {"input_ids", "pixel_values"}
