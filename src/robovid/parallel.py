"""Backend-agnostic helpers for annotating many videos in parallel.

Nothing here imports Modal, so this logic is unit-testable without credentials;
`modal_app.py` wires it to Modal containers.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .annotate import Annotator, annotate_video
from .ingest import read_sidecar


@dataclass(frozen=True)
class VideoJob:
    local_path: str
    remote_path: str  # path relative to the shared volume root
    video_id: str
    source_url: str | None = None  # from the fetch sidecar, if any
    license: str | None = None


def _sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def plan_jobs(videos: Sequence[str | Path]) -> list[VideoJob]:
    """One job per input video, in input order.

    Remote paths are keyed by content hash, so re-uploads are idempotent and two
    files with the same name in different folders don't collide on the volume.
    """
    jobs = []
    for v in videos:
        p = Path(v)
        meta = read_sidecar(p)
        jobs.append(VideoJob(
            local_path=str(p),
            remote_path=f"{_sha256(p)[:16]}/{p.name}",
            video_id=p.stem,  # same id the local backend uses
            source_url=meta.get("url"),
            license=meta.get("license"),
        ))
    return jobs


def annotate_video_to_jsonl(
    path: str | Path,
    video_id: str,
    annotator: Annotator,
    fps: float = 1.0,
    clip_len_s: float = 16.0,
    source_url: str | None = None,
    license: str | None = None,
) -> list[str]:
    """Annotate one video and return its JSONL lines (one per clip, no newline)."""
    return [
        ann.model_dump_json()
        for ann in annotate_video(path, annotator, fps=fps, clip_len_s=clip_len_s,
                                  video_id=video_id, source_url=source_url, license=license)
    ]


def merge_results(
    jobs: Sequence[VideoJob], results: Iterable[list[str] | BaseException]
) -> tuple[list[str], list[tuple[VideoJob, BaseException]]]:
    """Combine per-video results (in job order) into JSONL lines plus failures.

    `results` is what an order-preserving map with return_exceptions=True yields:
    either a video's list of lines or the exception it failed with.
    """
    results = list(results)
    if len(results) != len(jobs):
        raise ValueError(f"expected {len(jobs)} results, got {len(results)}")
    lines: list[str] = []
    failures: list[tuple[VideoJob, BaseException]] = []
    for job, res in zip(jobs, results):
        if isinstance(res, BaseException):
            failures.append((job, res))
        else:
            lines.extend(res)
    return lines, failures


def append_jsonl(lines: Iterable[str], out: str | Path) -> int:
    """Append lines to a JSONL file (same append behaviour as the local backend)."""
    n = 0
    with Path(out).open("a") as fh:
        for line in lines:
            fh.write(line + "\n")
            n += 1
    return n
