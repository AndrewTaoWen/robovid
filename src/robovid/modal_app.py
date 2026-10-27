"""Run annotation at scale on Modal: one container per video, fanned out with .starmap().

Requires the optional extra (`pip install -e .[modal]`), `modal setup`, and a
Modal Secret named "anthropic" holding ANTHROPIC_API_KEY. Normally used via

    robovid annotate --backend modal data/videos/*.mp4 --out annotations.jsonl

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

# OpenCV's wheel bundles FFmpeg for decoding, so no apt ffmpeg is needed.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("opencv-python-headless>=4.8", "pydantic>=2", "numpy", "anthropic>=0.40")
    .add_local_python_source("robovid")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    secrets=[modal.Secret.from_name("anthropic")],
    volumes={VIDEO_DIR: volume},
    timeout=TIMEOUT_S,
    retries=modal.Retries(max_retries=2, initial_delay=5.0, backoff_coefficient=2.0),
    max_containers=MAX_CONTAINERS,
    cpu=2.0,
)
def annotate_one(
    remote_path: str, video_id: str, fps: float, clip_len_s: float, model: str | None
) -> list[str]:
    """Annotate ONE video from the volume; returns its JSONL lines."""
    from robovid.annotate import ClaudeAnnotator

    annotator = ClaudeAnnotator(model=model)
    return annotate_video_to_jsonl(
        f"{VIDEO_DIR}/{remote_path}", video_id, annotator, fps=fps, clip_len_s=clip_len_s
    )


def upload_videos(jobs: Sequence[VideoJob]) -> None:
    with volume.batch_upload(force=True) as batch:
        for job in jobs:
            batch.put_file(job.local_path, f"/{job.remote_path}")


def run_on_modal(
    videos: Sequence[str | Path],
    out: str | Path,
    fps: float = 1.0,
    clip_len_s: float = 16.0,
    model: str | None = None,
) -> tuple[int, list[tuple[VideoJob, BaseException]]]:
    """Upload videos, annotate them in parallel containers, append results to `out`.

    Returns (clips written, failed videos). Output is in input-video order and
    uses the same JSONL schema as the local backend.
    """
    jobs = plan_jobs(videos)
    print(f"uploading {len(jobs)} videos to Modal volume {VOLUME_NAME!r}")
    upload_videos(jobs)
    with modal.enable_output(), app.run():
        results = list(annotate_one.starmap(
            [(j.remote_path, j.video_id, fps, clip_len_s, model) for j in jobs],
            order_outputs=True,
            return_exceptions=True,
        ))
    lines, failures = merge_results(jobs, results)
    return append_jsonl(lines, out), failures
