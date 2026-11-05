"""Annotate clips with a vision-language model.

The annotator is a small interface so you can swap providers (Claude, GPT,
Gemini, an open model) and compare label quality on the same clips.
"""
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Callable, Iterator, Protocol

from .frames import Clip, chunk_into_clips, sample_frames
from .schema import ClipAnnotation
from .verbs import VERBS, normalize_verb

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

verb must be exactly one of: {verbs}.
Use "other" only when none fits, and say what happened in the description.

Only label what is visible. If unsure, use "unknown" or leave steps empty."""


class Annotator(Protocol):
    name: str

    def annotate(self, clip: Clip, video_id: str) -> ClipAnnotation: ...


def annotate_video(
    video: str | Path,
    annotator: Annotator,
    fps: float = 1.0,
    clip_len_s: float = 16.0,
    video_id: str | None = None,
    source_url: str | None = None,
    license: str | None = None,
) -> Iterator[ClipAnnotation]:
    """Sample frames from one video, chunk them into clips and annotate each clip.

    This is the per-video unit of work shared by the local CLI and the Modal backend.
    `source_url` and `license` (normally from the sidecar `fetch` writes) are
    stamped on every clip so the dataset's provenance stays auditable.
    """
    video_id = video_id or Path(video).stem
    frames = sample_frames(video, fps=fps)
    provenance = {k: v for k, v in (("source_url", source_url), ("license", license)) if v}
    for clip in chunk_into_clips(frames, clip_len_s=clip_len_s):
        ann = annotator.annotate(clip, video_id=video_id)
        yield ann.model_copy(update=provenance) if provenance else ann


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in model output: {text[:200]}")
    return json.loads(match.group(0))


def _normalize_steps(steps: object) -> object:
    """Map each step's verb onto the fixed vocabulary, keeping the original."""
    if not isinstance(steps, list):
        return steps
    out = []
    for step in steps:
        if isinstance(step, dict) and "verb" in step:
            raw = step.get("verb")
            verb = normalize_verb(raw if isinstance(raw, str) else None)
            step = {**step, "verb": verb}
            if isinstance(raw, str) and raw.strip().lower() != verb:
                step["raw_verb"] = raw
        out.append(step)
    return out


def build_annotation(raw: dict, clip: Clip, video_id: str, model: str) -> ClipAnnotation:
    fields = {k: v for k, v in raw.items() if k in ClipAnnotation.model_fields}
    # provenance comes from fetch metadata, never from the model
    fields.pop("source_url", None)
    fields.pop("license", None)
    if "steps" in fields:
        fields["steps"] = _normalize_steps(fields["steps"])
    return ClipAnnotation(
        video_id=video_id,
        clip_start_s=clip.start_s,
        clip_end_s=clip.end_s,
        model=model,
        **fields,
    )


def clip_parts(clip: Clip, max_frames: int) -> list[tuple[str, str | bytes]]:
    """Provider-neutral request content for one clip, shared by every annotator.

    Returns ("text", str) and ("image", jpeg_bytes) parts in order: each sampled
    frame's timestamp then the frame (evenly subsampled to at most `max_frames`),
    then the labeling prompt. Each provider only maps these onto its wire format,
    so Claude and open-model labels come from the same input.
    """
    frames = clip.frames
    if len(frames) > max_frames:  # evenly subsample long clips
        stride = len(frames) / max_frames
        frames = [frames[int(i * stride)] for i in range(max_frames)]
    parts: list[tuple[str, str | bytes]] = []
    for f in frames:
        parts.append(("text", f"t={f.t:.1f}s"))
        parts.append(("image", f.jpeg))
    parts.append(("text", PROMPT.format(
        duration=clip.end_s - clip.start_s, start=clip.start_s, end=clip.end_s,
        verbs=", ".join(VERBS),
    )))
    return parts


class ClaudeAnnotator:
    """Annotator backed by the Anthropic API (needs ANTHROPIC_API_KEY)."""

    def __init__(self, model: str | None = None, max_frames: int = 16):
        import anthropic  # imported lazily so the rest of the package works without it

        self.client = anthropic.Anthropic()
        self.name = model or os.environ.get("ROBOVID_MODEL", "claude-sonnet-5-5")
        self.max_frames = max_frames

    def annotate(self, clip: Clip, video_id: str) -> ClipAnnotation:
        content: list[dict] = []
        for kind, value in clip_parts(clip, self.max_frames):
            if kind == "text":
                content.append({"type": "text", "text": value})
            else:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(value).decode(),
                    },
                })

        resp = self.client.messages.create(
            model=self.name,
            max_tokens=2000,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return build_annotation(_extract_json(text), clip, video_id, self.name)


# Where the GPU service from robovid.modal_vlm is deployed. Kept here (not imported
# from modal_vlm) so this module never needs Modal; a test checks the two agree.
VLLM_APP_NAME = "robovid-vlm"
VLLM_CLS_NAME = "VLMServer"


def chat_messages(clip: Clip, max_frames: int) -> list[dict]:
    """The clip as one OpenAI-style chat turn, the format vLLM's `LLM.chat` takes.

    Frames are sent inline as base64 JPEG data URLs, interleaved with their
    timestamps, exactly like the Claude request.
    """
    content: list[dict] = []
    for kind, value in clip_parts(clip, max_frames):
        if kind == "text":
            content.append({"type": "text", "text": value})
        else:
            url = "data:image/jpeg;base64," + base64.b64encode(value).decode()
            content.append({"type": "image_url", "image_url": {"url": url}})
    return [{"role": "user", "content": content}]


class VLLMAnnotator:
    """Annotator backed by an open VLM served with vLLM on Modal GPUs.

    `generate` takes one request dict ({"messages": [...]}) and returns
    {"model": str, "text": str} or {"model": str, "error": str}. In practice it is
    the deployed `VLMServer.generate.remote` (see `from_modal`); tests pass a fake.
    The model's reply goes through the same JSON extraction, verb normalization
    and schema validation as the Claude path, so the two are comparable.
    """

    def __init__(self, generate: Callable[[dict], dict], max_frames: int = 16,
                 name: str = "vllm"):
        self._generate = generate
        self.max_frames = max_frames
        self.name = name  # replaced by the served model's name after the first reply

    @classmethod
    def from_modal(cls, app_name: str = VLLM_APP_NAME, cls_name: str = VLLM_CLS_NAME,
                   max_frames: int = 16) -> "VLLMAnnotator":
        import modal  # lazily, so `import robovid` never needs Modal

        server = modal.Cls.from_name(app_name, cls_name)()
        return cls(server.generate.remote, max_frames=max_frames)

    def build_request(self, clip: Clip) -> dict:
        return {"messages": chat_messages(clip, self.max_frames)}

    def annotate(self, clip: Clip, video_id: str) -> ClipAnnotation:
        resp = self._generate(self.build_request(clip))
        self.name = resp.get("model") or self.name
        if resp.get("error") or resp.get("text") is None:
            raise RuntimeError(f"vLLM request failed for {video_id} "
                               f"[{clip.start_s:.0f}s]: {resp.get('error')}")
        return build_annotation(_extract_json(resp["text"]), clip, video_id, self.name)
