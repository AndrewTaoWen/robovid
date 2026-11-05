"""Sample frames from a video and group them into fixed-length clips."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class Frame:
    t: float  # seconds
    jpeg: bytes


@dataclass
class Clip:
    start_s: float
    end_s: float
    frames: list[Frame]


def sample_frames(path: str | Path, fps: float = 1.0, max_side: int = 768) -> list[Frame]:
    """Read a video and return JPEG frames sampled at roughly `fps`."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native_fps / fps))

    frames: list[Frame] = []
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if idx % step == 0:
            h, w = img.shape[:2]
            scale = max_side / max(h, w)
            if scale < 1:
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                frames.append(Frame(t=idx / native_fps, jpeg=buf.tobytes()))
        idx += 1
    cap.release()
    if not frames:
        raise ValueError(
            f"Decoded 0 frames from {path}; the codec may be unsupported by OpenCV "
            "(AV1 is a common cause). Re-encode to H.264 or re-fetch the video."
        )
    return frames


def chunk_into_clips(frames: list[Frame], clip_len_s: float = 16.0) -> list[Clip]:
    """Group frames into consecutive clips of about `clip_len_s` seconds."""
    clips: list[Clip] = []
    current: list[Frame] = []
    clip_start = frames[0].t if frames else 0.0
    for f in frames:
        if current and f.t - clip_start >= clip_len_s:
            clips.append(Clip(clip_start, current[-1].t, current))
            current, clip_start = [], f.t
        current.append(f)
    if current:
        clips.append(Clip(clip_start, current[-1].t, current))
    return clips
