"""Model modality discovery and package-facing capability descriptions.

The compiler must never assume that every Hugging Face model is a text-only
causal LLM.  This module deliberately separates *what a model can consume and
produce* from the native runtime support available today.  A package can carry
the processor assets and a truthful contract before every modality has a
hand-written ARM/Vulkan execution path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_VISION_LANGUAGE_TYPES = frozenset({
    "blip", "blip_2", "chameleon", "gemma3", "gemma4", "idefics", "idefics2",
    "idefics3", "internvl", "llava", "llava_next", "mllama", "paligemma",
    "phi3_v", "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "smolvlm", "video_llava",
})
_AUDIO_LANGUAGE_TYPES = frozenset({
    "gemma3n", "gemma4", "qwen2_audio", "qwen2_5_omni", "ultravox",
})
_OMNI_LANGUAGE_TYPES = frozenset({"gemma3n", "qwen2_5_omni"})
_SPEECH_TO_TEXT_TYPES = frozenset({
    "parakeet", "seamless_m4t", "speech_to_text", "wav2vec2", "whisper",
})
_TEXT_TO_AUDIO_TYPES = frozenset({"bark", "musicgen", "speecht5", "vits"})
_IMAGE_GENERATION_TYPES = frozenset({
    "autoencoder_kl", "controlnet", "flux", "stable-diffusion", "stable_diffusion",
    "stable_diffusion_3", "unet_2d_condition",
})
_EMBEDDING_TYPES = frozenset({"bert", "nomic_bert", "roberta", "t5_encoder"})
_VIDEO_TYPES = frozenset({"llava_next_video", "qwen2_5_vl", "qwen3_vl", "video_llava"})


@dataclass(frozen=True)
class ModalityCapabilities:
    """A stable, serializable model I/O contract.

    ``native_status`` describes the SiliconGraph runtime, not the Hugging Face
    model.  It prevents a package from falsely advertising an unimplemented
    on-device media pipeline as runnable.
    """

    task: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    primary_input: str
    primary_output: str
    native_status: str
    notes: tuple[str, ...] = ()

    def to_runtime_config(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "task": self.task,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "primary_input": self.primary_input,
            "primary_output": self.primary_output,
            "native_status": self.native_status,
            "notes": list(self.notes),
        }


def _model_type(config: Any) -> str:
    return str(getattr(config, "model_type", "") or "").lower()


def _architectures(config: Any) -> str:
    return " ".join(str(name).lower() for name in (getattr(config, "architectures", None) or ()))


def _has(config: Any, *names: str) -> bool:
    return any(getattr(config, name, None) is not None for name in names)


def _processor_has(processor: Any, name: str) -> bool:
    return processor is not None and getattr(processor, name, None) is not None


def detect_modalities(config: Any, processor: Any = None) -> ModalityCapabilities:
    """Infer a conservative I/O contract from config and processor metadata.

    The mapping is intentionally conservative: unfamiliar models remain
    text-generation rather than being labelled as a modality that the caller
    cannot actually provide.  The model's processor is then the authoritative
    source for image/audio/video preprocessing details.
    """

    model_type = _model_type(config)
    architectures = _architectures(config)

    is_video = model_type in _VIDEO_TYPES or "video" in model_type or _has(config, "video_config")
    is_vision_language = (
        model_type in _VISION_LANGUAGE_TYPES
        or "vision2seq" in architectures
        or _has(config, "vision_config", "image_token_index")
    )
    is_audio_language = model_type in _AUDIO_LANGUAGE_TYPES or _has(config, "audio_config")
    is_speech_to_text = (
        model_type in _SPEECH_TO_TEXT_TYPES
        or "speechseq2seq" in architectures
        or "ctc" in architectures
    )
    is_text_to_audio = model_type in _TEXT_TO_AUDIO_TYPES or "texttowaveform" in architectures
    is_image_generation = model_type in _IMAGE_GENERATION_TYPES

    # A processor may expose a modality even when a model uses a custom config
    # class.  It only expands an already plausible route, except for a video
    # processor which is unambiguous.
    has_video_processor = _processor_has(processor, "video_processor")
    has_image_processor = _processor_has(processor, "image_processor")
    has_feature_extractor = _processor_has(processor, "feature_extractor")
    is_video = is_video or has_video_processor
    is_vision_language = is_vision_language or (has_image_processor and "vision" in architectures)
    is_audio_language = is_audio_language or (has_feature_extractor and "audio" in architectures)

    # Models such as Qwen2.5-Omni and Gemma 3n can fuse image, audio, and text
    # in a single request.  Keep this separate from an image-only or audio-only
    # VLM contract: the runtime may preprocess the independent streams in
    # parallel, but must join them before the model's cross-modal forward pass.
    is_omni_language = (
        model_type in _OMNI_LANGUAGE_TYPES
        or (is_vision_language and is_audio_language)
    )

    if is_image_generation:
        return ModalityCapabilities(
            task="text_to_image", inputs=("text",), outputs=("image",),
            primary_input="text", primary_output="image", native_status="planned",
            notes=("Diffusion/image-generation graph lowering is not implemented yet.",),
        )
    if is_text_to_audio:
        return ModalityCapabilities(
            task="text_to_audio", inputs=("text",), outputs=("audio",),
            primary_input="text", primary_output="audio", native_status="planned",
            notes=("Audio synthesis graph lowering and audio-file output are pending.",),
        )
    if is_speech_to_text:
        return ModalityCapabilities(
            task="speech_to_text", inputs=("audio",), outputs=("text",),
            primary_input="audio", primary_output="text", native_status="experimental",
            notes=("Processor assets are packaged; native ASR graph coverage is model-specific.",),
        )
    if is_video:
        return ModalityCapabilities(
            task="video_text_to_text", inputs=("text", "video"), outputs=("text",),
            primary_input="video", primary_output="text", native_status="experimental",
            notes=("Video is decoded to sampled frames before model preprocessing.",),
        )
    if is_omni_language:
        return ModalityCapabilities(
            task="image_audio_text_to_text", inputs=("text", "image", "audio"), outputs=("text",),
            primary_input="text", primary_output="text", native_status="experimental",
            notes=(
                "Image and audio preprocessing may run concurrently; tensors are synchronized before the fused model forward pass.",
                "The packaged processor owns the model-specific multimodal prompt format.",
            ),
        )
    if is_vision_language:
        return ModalityCapabilities(
            task="image_text_to_text", inputs=("text", "image"), outputs=("text",),
            primary_input="image", primary_output="text", native_status="experimental",
            notes=("Image preprocessing is supplied by the packaged Hugging Face processor.",),
        )
    if is_audio_language:
        return ModalityCapabilities(
            task="audio_text_to_text", inputs=("text", "audio"), outputs=("text",),
            primary_input="audio", primary_output="text", native_status="experimental",
            notes=("Audio preprocessing is supplied by the packaged Hugging Face processor.",),
        )
    if model_type in _EMBEDDING_TYPES and "causallm" not in architectures:
        return ModalityCapabilities(
            task="text_embedding", inputs=("text",), outputs=("embedding",),
            primary_input="text", primary_output="embedding", native_status="planned",
            notes=("Embedding pooling/output serialization is pending in the native runtime.",),
        )
    return ModalityCapabilities(
        task="text_generation", inputs=("text",), outputs=("text",),
        primary_input="text", primary_output="text", native_status="supported",
    )


def model_requires_processor(capabilities: ModalityCapabilities) -> bool:
    """Whether a package needs a processor directory in addition to tokenizer files."""
    return any(modality in {"image", "audio", "video"} for modality in capabilities.inputs)
