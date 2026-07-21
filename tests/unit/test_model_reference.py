import pytest

from armcc.model_reference import normalize_model_reference


def test_normalizes_hugging_face_browser_urls_to_repo_ids():
    assert normalize_model_reference(
        "https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct"
    ) == "HuggingFaceTB/SmolVLM-256M-Instruct"
    assert normalize_model_reference(
        "https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/tree/main"
    ) == "HuggingFaceTB/SmolVLM-256M-Instruct"


def test_keeps_model_ids_and_windows_paths_unchanged():
    assert normalize_model_reference("HuggingFaceTB/SmolVLM-256M-Instruct") == "HuggingFaceTB/SmolVLM-256M-Instruct"
    assert normalize_model_reference(r"C:\models\smolvlm") == r"C:\models\smolvlm"


def test_rejects_hugging_face_dataset_urls():
    with pytest.raises(ValueError, match="model repositories"):
        normalize_model_reference("https://huggingface.co/datasets/HuggingFaceTB/some-dataset")
