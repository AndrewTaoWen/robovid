"""Serve an open vision-language model with vLLM on Modal GPUs.

The video containers in `modal_app.py` (CPU) decode frames and call this service
once per clip; nothing goes to an external API. Deploy it once:

    modal deploy -m robovid.modal_vlm

then annotate with `robovid annotate --backend modal --model-provider vllm ...`.
`modal run -m robovid.modal_vlm` sends one synthetic clip as a smoke test.

Design: an `@app.cls` that loads vLLM in `@modal.enter()` and exposes ONE
`@modal.batched` method. Each caller sends a single clip; Modal collects calls
from many video containers into one list, and that list goes to a single
`LLM.chat()` call, which vLLM batches on the GPU. Calls use Modal's own RPC, so
there's no public URL or API key to manage, and callers simply wait through a cold
start instead of hitting HTTP timeouts.

Configuration is read from env vars when you deploy and baked into the image env,
so the GPU container sees the same values:

    ROBOVID_VLLM_MODEL           Hugging Face model id (default Qwen/Qwen2.5-VL-7B-Instruct)
    ROBOVID_VLLM_GPU             Modal GPU type (default L40S, 48 GB)
    ROBOVID_VLLM_MAX_CONTAINERS  max GPU containers (default 2)
    ROBOVID_VLLM_MAX_BATCH       max clips per LLM.chat call (default 16)
    ROBOVID_VLLM_MAX_IMAGES      max frames per clip (default 16, the annotator's default)
    ROBOVID_VLLM_MAX_MODEL_LEN   context length (default 16384)

This module imports only `os` and `modal` at the top; vLLM is imported inside the
GPU container, so it's never needed locally.
"""
from __future__ import annotations

import os

import modal

APP_NAME = "robovid-vlm"  # = annotate.VLLM_APP_NAME
CLS_NAME = "VLMServer"  # = annotate.VLLM_CLS_NAME

MODEL_NAME = os.environ.get("ROBOVID_VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
GPU = os.environ.get("ROBOVID_VLLM_GPU", "L40S")
MAX_CONTAINERS = int(os.environ.get("ROBOVID_VLLM_MAX_CONTAINERS", "2"))
MAX_BATCH = int(os.environ.get("ROBOVID_VLLM_MAX_BATCH", "16"))
MAX_IMAGES = int(os.environ.get("ROBOVID_VLLM_MAX_IMAGES", "16"))
MAX_MODEL_LEN = int(os.environ.get("ROBOVID_VLLM_MAX_MODEL_LEN", "16384"))
MAX_TOKENS = 2000  # same output budget as ClaudeAnnotator
VLLM_VERSION = "0.10.1.1"  # Qwen2.5-VL has been supported since 0.7.x

HF_CACHE = "/root/.cache/huggingface"
VLLM_CACHE = "/root/.cache/vllm"  # torch.compile artifacts, speeds up later cold starts

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", "transformers<5")
    .env({
        "HF_HOME": HF_CACHE,
        # bake the deploy-time config in, so the container's import reads the same values
        "ROBOVID_VLLM_MODEL": MODEL_NAME,
        "ROBOVID_VLLM_GPU": GPU,
        "ROBOVID_VLLM_MAX_CONTAINERS": str(MAX_CONTAINERS),
        "ROBOVID_VLLM_MAX_BATCH": str(MAX_BATCH),
        "ROBOVID_VLLM_MAX_IMAGES": str(MAX_IMAGES),
        "ROBOVID_VLLM_MAX_MODEL_LEN": str(MAX_MODEL_LEN),
    })
    .add_local_python_source("robovid")  # so the container can import this module
)

app = modal.App(APP_NAME, image=image)
hf_cache = modal.Volume.from_name("robovid-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("robovid-vllm-cache", create_if_missing=True)


def generate_batch(llm, sampling_params, model_name: str, requests: list[dict]) -> list[dict]:
    """Run one batch of chat requests; one result dict per request, in order.

    If the batched call fails (e.g. one clip overflows the context), retry each
    request alone so a single bad clip doesn't fail every caller in the batch.
    """
    conversations = [r["messages"] for r in requests]
    try:
        outs = llm.chat(conversations, sampling_params, use_tqdm=False)
        return [{"model": model_name, "text": o.outputs[0].text} for o in outs]
    except Exception:
        if len(requests) == 1:
            return [{"model": model_name, "error": _error_text()}]
    results = []
    for conv in conversations:
        try:
            out = llm.chat([conv], sampling_params, use_tqdm=False)[0]
            results.append({"model": model_name, "text": out.outputs[0].text})
        except Exception:
            results.append({"model": model_name, "error": _error_text()})
    return results


def _error_text() -> str:
    import traceback

    return traceback.format_exc(limit=3)


@app.cls(
    gpu=GPU,
    volumes={HF_CACHE: hf_cache, VLLM_CACHE: vllm_cache},
    timeout=20 * 60,           # per batch, including a cold start's model load
    scaledown_window=5 * 60,   # stay warm between bursts of clips
    max_containers=MAX_CONTAINERS,
)
class VLMServer:
    @modal.enter()
    def load(self):
        from vllm import LLM, SamplingParams

        # First start downloads the weights into the robovid-hf-cache volume;
        # later cold starts read them from there.
        self.llm = LLM(
            model=MODEL_NAME,
            max_model_len=MAX_MODEL_LEN,
            limit_mm_per_prompt={"image": MAX_IMAGES},
            gpu_memory_utilization=0.90,
        )
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
        hf_cache.commit()

    @modal.batched(max_batch_size=MAX_BATCH, wait_ms=500)
    def generate(self, requests: list[dict]) -> list[dict]:
        """Callers send ONE {"messages": [...]} dict and get ONE result dict back."""
        return generate_batch(self.llm, self.sampling_params, MODEL_NAME, requests)


@app.local_entrypoint()
def smoke_test():
    """`modal run -m robovid.modal_vlm`: label one synthetic 4-frame clip."""
    import cv2
    import numpy as np

    from robovid.annotate import VLLMAnnotator
    from robovid.frames import Clip, Frame

    frames = []
    for i in range(4):
        img = np.zeros((240, 320, 3), np.uint8)
        cv2.circle(img, (60 + 60 * i, 120), 20, (0, 255, 0), -1)
        frames.append(Frame(t=float(i), jpeg=cv2.imencode(".jpg", img)[1].tobytes()))
    annotator = VLLMAnnotator(VLMServer().generate.remote)
    ann = annotator.annotate(Clip(0.0, 3.0, frames), video_id="smoke")
    print(ann.model_dump_json(indent=2))
