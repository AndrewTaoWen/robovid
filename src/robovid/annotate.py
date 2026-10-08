"""Annotate clips with a vision-language model.

The annotator is a small interface so you can swap providers (Claude, GPT,
Gemini, an open model) and compare label quality on the same clips.
"""
from __future__ import annotations

import base64
import json
import os
import re
from typing import Protocol

from .frames import Clip
from .schema import ClipAnnotation

PROMPT = """You are labeling video for robot learning. You will see frames sampled
from a {duration:.0f}-second clip, each tagged with its timestamp in seconds.

Return ONLY a JSON object with these fields:
- task: the high-level task being performed (short phrase)
- viewpoint: "egocentric", "third_person", "mixed", or "unknown"
- has_human_hands: true/false
- usable_for_robotics: true only if a person is clearly manipulating objects
  with hands visible (not a talking head, slideshow, or montage)
- steps: list of atomic manipulation steps, each with
  start_s, end_s (absolute seconds, between {start:.1f} and {end:.1f}),
  verb, objects (list), hand ("left"/"right"/"both"/"none"/"unknown"), description

Only label what is visible. If unsure, use "unknown" or leave steps empty."""


class Annotator(Protocol):
    name: str

    def annotate(self, clip: Clip, video_id: str) -> ClipAnnotation: ...


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in model output: {text[:200]}")
    return json.loads(match.group(0))


def build_annotation(raw: dict, clip: Clip, video_id: str, model: str) -> ClipAnnotation:
    return ClipAnnotation(
        video_id=video_id,
        clip_start_s=clip.start_s,
        clip_end_s=clip.end_s,
        model=model,
        **{k: v for k, v in raw.items() if k in ClipAnnotation.model_fields},
    )


class ClaudeAnnotator:
    """Annotator backed by the Anthropic API (needs ANTHROPIC_API_KEY)."""

    def __init__(self, model: str | None = None, max_frames: int = 16):
        import anthropic  # imported lazily so the rest of the package works without it

        self.client = anthropic.Anthropic()
        self.name = model or os.environ.get("ROBOVID_MODEL", "claude-sonnet-5-5")
        self.max_frames = max_frames

    def annotate(self, clip: Clip, video_id: str) -> ClipAnnotation:
        frames = clip.frames
        if len(frames) > self.max_frames:  # evenly subsample long clips
            stride = len(frames) / self.max_frames
            frames = [frames[int(i * stride)] for i in range(self.max_frames)]

        content: list[dict] = []
        for f in frames:
            content.append({"type": "text", "text": f"t={f.t:.1f}s"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(f.jpeg).decode(),
                },
            })
        content.append({
            "type": "text",
            "text": PROMPT.format(
                duration=clip.end_s - clip.start_s, start=clip.start_s, end=clip.end_s
            ),
        })

        resp = self.client.messages.create(
            model=self.name,
            max_tokens=2000,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return build_annotation(_extract_json(text), clip, video_id, self.name)
