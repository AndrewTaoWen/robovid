"""Run annotation at scale on Modal: one container per video, fanned out with .starmap().

Requires the optional extra (`pip install -e .[modal]`) and `modal setup`. With the
default model provider ("claude") it also needs a Modal Secret named "anthropic"
holding ANTHROPIC_API_KEY. With "vllm" the containers call the open-model GPU
service from `modal_vlm.py` instead and need no secret. Normally used via

    robovid annotate --backend modal data/videos/*.mp4 --out annotations.jsonl
    robovid annotate --backend modal --model-provider vllm data/videos/*.mp4

Nothing else in the package imports this module, so `import robovid` never needs Modal.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import modal

from robovid.parallel import (
    VideoJob,
    annotate_video_to_jsonl,
    append_jsonl,
    merge_results,
    plan_jobs,
)

APP_NAME = "robovid"
VOLUME_NAME = os.environ.get("ROBOVID_MODAL_VOLUME", "robovid-videos")
VIDEO_DIR = "/videos"
# Upper bound on parallel containers. Read at import time because Modal takes it
# from the decorator; the CLI's --max-containers flag sets this env var first.
MAX_CONTAINERS = int(os.environ.get("ROBOVID_MODAL_MAX_CONTAINERS", "20"))
TIMEOUT_S = 30 * 60  # per video
# "claude" (Anthropic API) or "vllm" (open model on Modal GPUs, see modal_vlm.py).
# Read at import time because it decides which secrets the function gets; the
# CLI's --model-provider flag sets this env var first.
MODEL_PROVIDER = os.environ.get("ROBOVID_MODEL_PROVIDER", "claude")
PROVIDERS = ("claude", "vllm")
if MODEL_PROVIDER not in PROVIDERS:
    raise ValueError(f"ROBOVID_MODEL_PROVIDER must be one of {PROVIDERS}, got {MODEL_PROVIDER!r}")
# Only the Claude path needs the Anthropic key; the vLLM path must run without it.
SECRETS = [modal.Secret.from_name("anthropic")] if MODEL_PROVIDER == "claude" else []

# OpenCV's wheel bundles FFmpeg for decoding, so no apt ffmpeg is needed.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("opencv-python-headless>=4.8", "pydantic>=2", "numpy", "anthropic>=0.40")
    .env({"ROBOVID_MODEL_PROVIDER": MODEL_PROVIDER})  # containers see the same provider
    .add_local_python_source("robovid")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def make_annotator(provider: str, model: str | None):
    """The annotator a video container uses for `provider`."""
    if provider == "vllm":
        from robovid.annotate import VLLMAnnotator

        return VLLMAnnotator.from_modal()  # the deployed GPU service picks the model
    from robovid.annotate import ClaudeAnnotator

    return ClaudeAnnotator(model=model)


@app.function(
    secrets=SECRETS,
    volumes={VIDEO_DIR: volume},
    timeout=TIMEOUT_S,
    retries=modal.Retries(max_retries=2, initial_delay=5.0, backoff_coefficient=2.0),
    max_containers=MAX_CONTAINERS,
    cpu=2.0,
)
def annotate_one(
    remote_path: str,
    video_id: str,
    fps: float,
    clip_len_s: float,
    model: str | None,
    source_url: str | None = None,
    license: str | None = None,
    provider: str = "claude",
) -> list[str]:
    """Annotate ONE video from the volume; returns its JSONL lines.

    The container always decodes the video itself; with provider="vllm" each clip
    is then labeled by the GPU service instead of the Anthropic API.
    """
    annotator = make_annotator(provider, model)
    return annotate_video_to_jsonl(
        f"{VIDEO_DIR}/{remote_path}", video_id, annotator, fps=fps, clip_len_s=clip_len_s,
        source_url=source_url, license=license,
    )


def upload_videos(jobs: Sequence[VideoJob]) -> None:
    with volume.batch_upload(force=True) as batch:
        for job in jobs:
            batch.put_file(job.local_path, f"/{job.remote_path}")


def check_vlm_service() -> None:
    """Fail fast, before uploading anything, if the GPU service isn't deployed."""
    from robovid.annotate import VLLM_APP_NAME, VLLM_CLS_NAME

    try:
        modal.Cls.from_name(VLLM_APP_NAME, VLLM_CLS_NAME).hydrate()
    except Exception as e:
        raise RuntimeError(
            f"open-model service {VLLM_APP_NAME}/{VLLM_CLS_NAME} not found on Modal; "
            f"deploy it first: modal deploy -m robovid.modal_vlm ({e})"
        ) from e


def run_on_modal(
    videos: Sequence[str | Path],
    out: str | Path,
    fps: float = 1.0,
    clip_len_s: float = 16.0,
    model: str | None = None,
    model_provider: str = MODEL_PROVIDER,
) -> tuple[int, list[tuple[VideoJob, BaseException]]]:
    """Upload videos, annotate them in parallel containers, append results to `out`.

    Returns (clips written, failed videos). Output is in input-video order and
    uses the same JSONL schema as the local backend.
    """
    if model_provider != MODEL_PROVIDER:
        raise ValueError(f"model_provider={model_provider!r} but this module was imported "
                         f"with ROBOVID_MODEL_PROVIDER={MODEL_PROVIDER!r}; set the env var "
                         "before importing robovid.modal_app")
    if model_provider == "vllm":
        check_vlm_service()
    jobs = plan_jobs(videos)
    missing = [j.video_id for j in jobs if not j.source_url]
    if missing:
        print(f"warning: no provenance sidecar for {len(missing)} video(s) "
              f"({', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}); "
              "run `robovid fetch` on their URLs to backfill")
    print(f"uploading {len(jobs)} videos to Modal volume {VOLUME_NAME!r}")
    upload_videos(jobs)
    with modal.enable_output(), app.run():
        results = list(annotate_one.starmap(
            [(j.remote_path, j.video_id, fps, clip_len_s, model, j.source_url, j.license,
              model_provider)
             for j in jobs],
            order_outputs=True,
            return_exceptions=True,
        ))
    lines, failures = merge_results(jobs, results)
    return append_jsonl(lines, out), failures
