"""
ARM AI Compiler — Integration Test
Smoke test for the full Python pipeline on SmolLM2 (135M params).
Does NOT require the compiled C++ armcc binary.
"""
import pytest
import torch
import tempfile
import os
from pathlib import Path


@pytest.fixture(scope="module")
def model_id():
    return "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def loaded_model(model_id):
    """Load SmolLM2 once for all tests."""
    from armcc.model_loader import ModelLoader
    loader = ModelLoader(cache_dir=None)
    try:
        model = loader.load(model_id, dtype=torch.float16, trust_remote_code=False)
        return model
    except Exception as e:
        pytest.skip(f"Cannot load {model_id}: {e}")


def test_model_loader(loaded_model):
    """Test that the model loads correctly."""
    assert loaded_model.model_id is not None
    assert loaded_model.arch_family in ("llama_style", "gpt2_style", "gemma_style")
    assert loaded_model.total_params > 0
    assert loaded_model.num_layers > 0
    assert loaded_model.hidden_size > 0


def test_hf_metadata_extraction(loaded_model):
    """Test tokenizer and config extraction."""
    from armcc.hf_metadata import HFMetadataExtractor
    with tempfile.TemporaryDirectory() as tmpdir:
        extractor = HFMetadataExtractor(tmpdir)
        config = extractor.extract(loaded_model)
        assert isinstance(config, dict)
        assert "bos_token_id" in config
        assert "eos_token_id" in config
        assert os.path.exists(os.path.join(tmpdir, "tokenizer"))
        assert os.path.exists(os.path.join(tmpdir, "runtime_config.json"))


def test_graph_export(loaded_model):
    """Test graph export with torch.export."""
    from armcc.graph_exporter import GraphExporter
    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = GraphExporter(tmpdir)
        graph = exporter.export(loaded_model, context_length=64, batch_size=1)

        assert graph.model_id == loaded_model.model_id
        assert graph.arch_family == loaded_model.arch_family
        # Should have at least input/output tensors
        assert len(graph.tensors) > 0
        assert len(graph.input_ids) > 0
        assert len(graph.output_ids) > 0

        # Save JSON
        json_path = exporter.saveJSON(graph)
        assert os.path.exists(json_path)

        # Verify JSON has required keys
        import json
        with open(json_path) as f:
            data = json.load(f)
        assert "model_id" in data
        assert "arch" in data
        assert "nodes" in data
        assert "tensors" in data


def test_calibration_runner(loaded_model):
    """Test calibration with small number of samples."""
    from armcc.calibration import CalibrationRunner
    runner = CalibrationRunner(
        loaded_model,
        num_samples=4,
        max_seq_len=64,
        dataset_name="invalid_dataset",  # Will use fallback texts
    )
    result = runner.run()

    assert result.num_samples == 4
    assert isinstance(result.stats, dict)
    assert isinstance(result.sensitivity, dict)

    # Check quant params are reasonable
    for name, stats in result.stats.items():
        assert stats.scale_int8 > 0
        assert -128 <= stats.zero_point_int8 <= 127


def test_full_pipeline_no_cpp(loaded_model):
    """Test the full Python pipeline without the C++ compiler."""
    from armcc.graph_exporter import GraphExporter
    from armcc.hf_metadata import HFMetadataExtractor
    from armcc.calibration import CalibrationRunner

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Metadata
        meta = HFMetadataExtractor(tmpdir)
        config = meta.extract(loaded_model)

        # Step 2: Export
        exporter = GraphExporter(tmpdir)
        graph = exporter.export(loaded_model, context_length=64)
        json_path = exporter.saveJSON(graph)

        # Step 3: Calibration (4 samples)
        runner = CalibrationRunner(loaded_model, num_samples=4, max_seq_len=64)
        result = runner.run()
        calib_path = os.path.join(tmpdir, "calibration.json")
        runner.save(result, calib_path)

        assert os.path.exists(json_path)
        assert os.path.exists(calib_path)
        assert os.path.exists(os.path.join(tmpdir, "tokenizer"))
        print(f"\n✓ Full Python pipeline completed for {loaded_model.model_id}")
        print(f"  Graph: {len(graph.nodes)} ops, {len(graph.tensors)} tensors")
        print(f"  Calib: {len(result.stats)} tensors profiled")
