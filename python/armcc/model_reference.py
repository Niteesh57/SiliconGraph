"""Normalization for user-supplied Hugging Face model references."""

from __future__ import annotations

from urllib.parse import unquote, urlparse


_HUGGING_FACE_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"}
_NON_MODEL_ROOTS = {"datasets", "spaces"}


def normalize_model_reference(reference: str) -> str:
    """Return a Hub model ID for a Hugging Face browser URL.

    Hugging Face loaders accept ``owner/repository`` or a local directory, but
    not a browser URL. Other values are deliberately left unchanged so local
    paths (including Windows paths) continue to work as supplied.
    """
    candidate = str(reference or "").strip()
    if not candidate:
        raise ValueError("Model reference must not be blank.")

    parsed = urlparse(candidate)
    if parsed.scheme.lower() not in {"http", "https"}:
        return candidate
    if parsed.netloc.lower() not in _HUGGING_FACE_HOSTS:
        return candidate

    segments = [unquote(segment) for segment in parsed.path.split("/") if segment]
    if len(segments) < 2:
        raise ValueError(
            "A Hugging Face model URL must include both owner and repository, "
            "for example https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct."
        )
    if segments[0].lower() in _NON_MODEL_ROOTS:
        raise ValueError("Only Hugging Face model repositories are supported; dataset and Space URLs cannot be compiled as models.")

    return f"{segments[0]}/{segments[1]}"
