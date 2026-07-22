from types import SimpleNamespace

from armcc.hf_metadata import HFMetadataExtractor
from armcc.model_loader import ModelLoader
from armcc.modalities import ModalityCapabilities, detect_modalities


def test_detects_vision_language_models_without_using_the_model_name():
    capabilities = detect_modalities(SimpleNamespace(model_type="idefics3", architectures=[]))

    assert capabilities.task == "image_text_to_text"
    assert capabilities.inputs == ("text", "image")
    assert capabilities.outputs == ("text",)
    assert capabilities.native_status == "experimental"


def test_detects_speech_and_generative_media_contracts():
    speech = detect_modalities(SimpleNamespace(model_type="whisper", architectures=[]))
    image_generation = detect_modalities(SimpleNamespace(model_type="stable_diffusion", architectures=[]))
    audio_generation = detect_modalities(SimpleNamespace(model_type="vits", architectures=[]))

    assert (speech.task, speech.inputs, speech.outputs) == ("speech_to_text", ("audio",), ("text",))
    assert (image_generation.task, image_generation.outputs) == ("text_to_image", ("image",))
    assert (audio_generation.task, audio_generation.outputs) == ("text_to_audio", ("audio",))


def test_detects_models_that_can_fuse_image_audio_and_text_in_one_request():
    capabilities = detect_modalities(
        SimpleNamespace(model_type="qwen2_5_omni", architectures=[]),
    )

    assert capabilities.task == "image_audio_text_to_text"
    assert capabilities.inputs == ("text", "image", "audio")
    assert capabilities.outputs == ("text",)
    assert "concurrently" in capabilities.notes[0]


def test_idefics3_processor_is_configured_for_a_static_single_image_graph():
    processor = SimpleNamespace(image_processor=SimpleNamespace(do_image_splitting=True))
    modalities = ModalityCapabilities(
        task="image_text_to_text", inputs=("text", "image"), outputs=("text",),
        primary_input="image", primary_output="text", native_status="experimental",
    )

    ModelLoader._configureStaticMultimodalProcessor(
        SimpleNamespace(model_type="idefics3"), processor, modalities,
    )

    assert processor.image_processor.do_image_splitting is False


def test_metadata_writes_processor_assets_and_modality_contract(tmp_path):
    class Tokenizer:
        chat_template = "{{ messages }}"

        def save_pretrained(self, path):
            (tmp_path / "tokenizer").mkdir(exist_ok=True)

    class Processor:
        def save_pretrained(self, path):
            from pathlib import Path
            Path(path, "preprocessor_config.json").write_text("{}", encoding="utf-8")

    modalities = ModalityCapabilities(
        task="image_text_to_text", inputs=("text", "image"), outputs=("text",),
        primary_input="image", primary_output="text", native_status="experimental",
    )
    loaded = SimpleNamespace(
        config=SimpleNamespace(bos_token_id=1, eos_token_id=2, pad_token_id=0),
        tokenizer=Tokenizer(), processor=Processor(), modalities=modalities,
        model_id="org/vision-model", arch_family="vision_lm", num_layers=1,
        hidden_size=16, num_heads=2, num_kv_heads=1, intermediate_size=32,
        vocab_size=128, max_position_embeddings=512, total_params=1, total_bytes=2,
    )

    runtime = HFMetadataExtractor(tmp_path).extract(loaded)

    assert runtime["modalities"]["inputs"] == ["text", "image"]
    assert runtime["modalities"]["native_status"] == "experimental"
    assert (tmp_path / "processor" / "preprocessor_config.json").is_file()
