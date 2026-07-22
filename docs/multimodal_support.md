# Multimodal support

SiliconGraph packages describe the input and output modalities of a model; they
do not assume every Hugging Face repository is a text-only causal LLM.

## Package contract

`runtime_config.json` contains a `modalities` object:

```json
{
  "task": "image_text_to_text",
  "inputs": ["text", "image"],
  "outputs": ["text"],
  "primary_input": "image",
  "primary_output": "text",
  "native_status": "experimental"
}
```

For image, audio, or video inputs, the package also contains `processor/`.
Those assets come from the source model's Hugging Face `AutoProcessor` and own
normalization, resizing, feature extraction, frame sampling, and special media
tokens. A runtime must use them rather than inventing a new preprocessing path.

## Multimodal requests and safe admission

A model with the `image_audio_text_to_text` contract can receive a prompt,
image, and audio in one request. For example: *"Look at this image and listen
to this audio. What object is being discussed?"* The processor formats those
inputs for the source model; the tensors are then passed to **one fused model
forward pass** so cross-modal attention sees all of them.

Image decoding/resizing, audio decoding/resampling/feature extraction, and text
tokenization are independent preparation jobs. The runtime may run them in
parallel, then synchronizes them before the fused graph begins. It must not run
the image and audio branches as separate answers and try to merge their text.

Every package now carries `execution_policy.json` input-admission rules:

- Full: FP32/FP16; media is eligible.
- Medium: INT8, or the INT8 member of a mixed graph family; media is eligible.
- Low: INT4/FP8; text-only. Image, audio, and video are refused before their
  buffers are allocated.
- Media is also refused when current safe RAM is below 768 MB or cannot cover
  the model's declared media working set. The runtime can then ask for a
  text-only prompt or for more free memory instead of risking an OOM crash.

For a mixed package, media input sets a hard graph-selection constraint: the
runtime selects an INT8 graph and never falls back to the INT4 graph.

## Current state

| Model task | Inputs | Outputs | Package support | Native ARM runtime |
|---|---|---|---|---|
| Text generation | text | text | supported | supported foundation |
| Vision-language | text, image | text | supported | experimental graph export/lowering |
| Speech-to-text | audio | text | supported | experimental graph export/lowering |
| Audio-language | text, audio | text | supported | experimental graph export/lowering |
| Omni language | text, image, audio | text | supported | experimental graph export/lowering |
| Video-language | text, video | text | supported | experimental graph export/lowering |
| Text embedding | text | embedding | supported | output adapter planned |
| Text-to-image | text | image | declared | planned |
| Text-to-audio | text | audio | declared | planned |

`native_status` is deliberately separate from the model capability. It prevents
a package from claiming that a text-to-image or text-to-audio model can already
run on Android when its output decoder and Vulkan/CPU kernels have not yet been
implemented.

## Implementation order

1. Image-to-text models: processor adapter, image tensor graph lowering,
   CPU/Vulkan parity tests, then Android camera/file input.
2. Audio-to-text: PCM/WAV input, resampling and feature-extractor adapter,
   streaming chunk state, then microphone input.
3. Video-to-text: deterministic frame sampling plus the image path; optional
   audio track follows the audio adapter.
4. Embeddings: pooled tensor output and a binary/JSON vector result.
5. Image/audio generation: model-family-specific decoder graphs and safe file
   writers. These are separate workloads from token generation and must not be
   forced through the LLM KV-cache planner.
