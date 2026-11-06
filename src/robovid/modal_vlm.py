"""Run an open vision model on Modal GPUs, so labeling needs no outside API.

What this is: a service on Modal that loads Qwen2.5-VL-7B-Instruct, an open
vision-language model (a model that reads images and text), onto a GPU. It uses
vLLM, open-source software for running models like this efficiently. The video
containers in modal_app.py decode each video on ordinary CPUs and send this
service one clip at a time to label.

How it works:
1. You deploy it once. The first start downloads the model's files (about 16 GB)
   into Modal storage. Later starts read them from there.
2. Each video container sends one clip and waits for the answer.
3. Modal holds incoming clips for up to half a second and hands this service up
   to 16 at a time, possibly from different videos. vLLM labels them in one pass
   on the GPU, which is much faster than one clip at a time.
4. If that combined pass fails (for example, one clip is too long), each clip is
   retried on its own, so one bad clip doesn't fail the others.
5. After 5 minutes with no clips, the GPU shuts down so you stop paying for it.

Calls go through Modal's own connection between functions. There is no web
address or API key to set up. If the model is still loading, callers just wait.

Try it:

    modal deploy -m robovid.modal_vlm      # start the service
    modal run -m robovid.modal_vlm         # label one made-up test clip

Settings, read when you deploy and copied into the GPU container:

    ROBOVID_VLLM_MODEL           model to load from Hugging Face (default Qwen/Qwen2.5-VL-7B-Instruct)
    ROBOVID_VLLM_GPU             Modal GPU type (default L40S, which has 48 GB of memory)
    ROBOVID_VLLM_MAX_CONTAINERS  most GPUs to run at once (default 2)
    ROBOVID_VLLM_MAX_BATCH       most clips per pass (default 16)
    ROBOVID_VLLM_MAX_IMAGES      most frames per clip (default 16, same as the labeler)
    ROBOVID_VLLM_MAX_MODEL_LEN   most tokens (pieces of text and image) per request (default 16384)
    ROBOVID_VLLM_MEMORY_MB       regular memory (RAM) for the GPU container, in MB (default 32768)
    ROBOVID_VLLM_CPU             CPU cores for the GPU container (default 4)

vLLM is only imported inside the GPU container, so you never need it installed
on your own machine.
"""
from __future__ import annotations

import os

import modal

APP_NAME = "robovid-vlm"  # must match annotate.VLLM_APP_NAME (a test checks)
CLS_NAME = "VLMServer"  # must match annotate.VLLM_CLS_NAME (a test checks)

MODEL_NAME = os.environ.get("ROBOVID_VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
GPU = os.environ.get("ROBOVID_VLLM_GPU", "L40S")
MAX_CONTAINERS = int(os.environ.get("ROBOVID_VLLM_MAX_CONTAINERS", "2"))
MAX_BATCH = int(os.environ.get("ROBOVID_VLLM_MAX_BATCH", "16"))
MAX_IMAGES = int(os.environ.get("ROBOVID_VLLM_MAX_IMAGES", "16"))
MAX_MODEL_LEN = int(os.environ.get("ROBOVID_VLLM_MAX_MODEL_LEN", "16384"))
# Regular memory and CPUs for the GPU container. Without these, Modal gives a
# container only a small default, and vLLM's engine was killed as soon as it
# started preparing the first clip's images. 32 GB leaves room for the model
# loading step and for decoding 16 clips' worth of frames at once.
MEMORY_MB = int(os.environ.get("ROBOVID_VLLM_MEMORY_MB", "32768"))
CPU = float(os.environ.get("ROBOVID_VLLM_CPU", "4"))
MAX_TOKENS = 2000  # longest answer allowed, the same limit Claude gets
VLLM_VERSION = "0.10.1.1"  # pinned so the setup doesn't change under us

HF_CACHE = "/root/.cache/huggingface"
VLLM_CACHE = "/root/.cache/vllm"  # vLLM's prepared GPU code, kept so later starts are faster

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", "transformers<5")
    .env({
        "HF_HOME": HF_CACHE,
        # Copy the settings into the container, so it sees the same values you deployed with.
        "ROBOVID_VLLM_MODEL": MODEL_NAME,
        "ROBOVID_VLLM_GPU": GPU,
        "ROBOVID_VLLM_MAX_CONTAINERS": str(MAX_CONTAINERS),
        "ROBOVID_VLLM_MAX_BATCH": str(MAX_BATCH),
        "ROBOVID_VLLM_MAX_IMAGES": str(MAX_IMAGES),
        "ROBOVID_VLLM_MAX_MODEL_LEN": str(MAX_MODEL_LEN),
        "ROBOVID_VLLM_MEMORY_MB": str(MEMORY_MB),
        "ROBOVID_VLLM_CPU": str(CPU),
    })
    .add_local_python_source("robovid")  # ship our code so the container can import it
)

app = modal.App(APP_NAME, image=image)
hf_cache = modal.Volume.from_name("robovid-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("robovid-vllm-cache", create_if_missing=True)


def generate_batch(llm, sampling_params, model_name: str, requests: list[dict]) -> list[dict]:
    """Label a group of clips in one pass. Returns one result per clip, in the same order.

    If the group fails (for example, one clip is too long for the model), each
    clip is retried on its own, so one bad clip doesn't fail every caller.
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
    memory=MEMORY_MB,          # regular memory (RAM), in MB
    cpu=CPU,                   # for preparing images while the GPU works
    volumes={HF_CACHE: hf_cache, VLLM_CACHE: vllm_cache},
    timeout=20 * 60,           # per group of clips, including loading the model on a fresh start
    scaledown_window=5 * 60,   # stay on for 5 idle minutes, then shut down
    max_containers=MAX_CONTAINERS,
)
class VLMServer:
    @modal.enter()
    def load(self):
        from vllm import LLM, SamplingParams

        # The first start downloads the model's files into Modal storage.
        # Later starts read them from there instead of downloading again.
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
        """Label clips. Each caller sends one clip and gets one answer back.

        Modal gathers calls from many callers into the `requests` list for us.
        """
        return generate_batch(self.llm, self.sampling_params, MODEL_NAME, requests)


@app.local_entrypoint()
def smoke_test():
    """Check the service works: label one made-up 4-frame clip and print the result.

    Run it with `modal run -m robovid.modal_vlm`.
    """
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
