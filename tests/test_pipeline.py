import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from robovid.annotate import _extract_json, build_annotation
from robovid.frames import chunk_into_clips, sample_frames
from robovid.ingest import license_ok


def make_video(path: Path, seconds: int = 40, fps: int = 10) -> None:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for i in range(seconds * fps):
        img = np.zeros((240, 320, 3), np.uint8)
        cv2.circle(img, (i % 320, 120), 20, (0, 255, 0), -1)
        w.write(img)
    w.release()


class TestFrames(unittest.TestCase):
    def test_sample_and_chunk(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v.mp4"
            make_video(p)
            frames = sample_frames(p, fps=1.0)
            self.assertEqual(len(frames), 40)
            clips = chunk_into_clips(frames, clip_len_s=16)
            self.assertEqual(len(clips), 3)
            self.assertEqual(sum(len(c.frames) for c in clips), 40)


class TestAnnotationParsing(unittest.TestCase):
    def test_build_from_model_output(self):
        text = "Here you go:\n" + json.dumps({
            "task": "pouring water into a mug",
            "viewpoint": "egocentric",
            "has_human_hands": True,
            "usable_for_robotics": True,
            "steps": [{"start_s": 1, "end_s": 3, "verb": "grasp",
                       "objects": ["kettle"], "hand": "right"}],
            "extra_field": "ignored",
        })
        frames = []
        from robovid.frames import Clip
        ann = build_annotation(_extract_json(text), Clip(0, 15, frames), "vid1", "test")
        self.assertEqual(ann.steps[0].verb, "grasp")
        self.assertTrue(ann.usable_for_robotics)


class TestLicense(unittest.TestCase):
    def test_license_filter(self):
        self.assertTrue(license_ok("Creative Commons Attribution license (reuse allowed)"))
        self.assertFalse(license_ok(None))
        self.assertFalse(license_ok("Standard YouTube License"))


if __name__ == "__main__":
    unittest.main()
