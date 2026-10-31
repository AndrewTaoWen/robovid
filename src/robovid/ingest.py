"""Download videos, keeping only ones whose license allows reuse.

Internet video is not free to train on by default. This keeps Creative
Commons videos and records the license with every annotation, so the
dataset's provenance is auditable from day one.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

ALLOWED_LICENSE_MARKERS = ("creative commons", "cc-by", "cc by", "cc0", "public domain")


@dataclass
class VideoMeta:
    video_id: str
    url: str
    title: str
    license: str | None
    duration_s: float | None
    path: Path | None = None


def license_ok(license_str: str | None) -> bool:
    if not license_str:
        return False
    s = license_str.lower()
    return any(m in s for m in ALLOWED_LICENSE_MARKERS)


def fetch_meta(url: str) -> VideoMeta:
    out = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-playlist", "--", url],
        check=True, capture_output=True, text=True,
    ).stdout
    info = json.loads(out)
    return VideoMeta(
        video_id=info["id"],
        url=url,
        title=info.get("title", ""),
        license=info.get("license"),
        duration_s=info.get("duration"),
    )


def sidecar_path(video: str | Path) -> Path:
    """Where a video's provenance metadata lives: same name, .json extension."""
    return Path(video).with_suffix(".json")


def write_sidecar(meta: VideoMeta, video: str | Path) -> Path:
    path = sidecar_path(video)
    path.write_text(json.dumps({
        "video_id": meta.video_id,
        "url": meta.url,
        "title": meta.title,
        "license": meta.license,
        "duration_s": meta.duration_s,
    }, indent=2) + "\n")
    return path


def read_sidecar(video: str | Path) -> dict:
    """Provenance for a video, or {} if it has no (readable) sidecar."""
    try:
        data = json.loads(sidecar_path(video).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def download(url: str, out_dir: str | Path, require_open_license: bool = True) -> VideoMeta | None:
    """Download one video if its license allows it. Returns None if skipped.

    Always writes a provenance sidecar (<video_id>.json). If the video file is
    already there, only the sidecar is (re)written, so re-running fetch
    backfills metadata for older downloads without downloading again.
    """
    meta = fetch_meta(url)
    if require_open_license and not license_ok(meta.license):
        print(f"skip {meta.video_id}: license={meta.license!r}")
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{meta.video_id}.mp4"
    if target.exists():
        meta.path = target
        write_sidecar(meta, target)
        print(f"have {meta.video_id}: wrote metadata only")
        return meta
    subprocess.run(
        [
            "yt-dlp",
            # Prefer H.264 (avc1): OpenCV's pip wheels can't decode AV1, which
            # YouTube now serves by default, so AV1 files yield zero frames.
            "-f",
            "b[vcodec^=avc1][height<=720]/bv*[vcodec^=avc1][height<=720]+ba"
            "/mp4[height<=720]/best[height<=720]",
            "--merge-output-format", "mp4",
            "-o", str(target), "--", url,
        ],
        check=True,
    )
    meta.path = target
    write_sidecar(meta, target)
    return meta
