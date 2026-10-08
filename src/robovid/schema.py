"""Annotation schema: what a robotics team would want from one video clip."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Step(BaseModel):
    """One atomic manipulation step, e.g. 'grasp mug handle with right hand'."""

    start_s: float = Field(ge=0, description="Start time in seconds")
    end_s: float = Field(ge=0, description="End time in seconds")
    verb: str = Field(description="Action verb: grasp, pour, place, open, wipe...")
    objects: list[str] = Field(default_factory=list, description="Objects acted on")
    hand: Literal["left", "right", "both", "none", "unknown"] = "unknown"
    description: str = ""


class ClipAnnotation(BaseModel):
    """Annotation for one clip (a window of sampled frames)."""

    video_id: str
    source_url: str | None = None
    license: str | None = None
    clip_start_s: float
    clip_end_s: float
    task: str = Field(description="High-level task, e.g. 'making pour-over coffee'")
    viewpoint: Literal["egocentric", "third_person", "mixed", "unknown"] = "unknown"
    has_human_hands: bool = False
    steps: list[Step] = Field(default_factory=list)
    usable_for_robotics: bool = Field(
        default=False,
        description="Clear manipulation, hands visible, not a static talking head",
    )
    model: str | None = None
