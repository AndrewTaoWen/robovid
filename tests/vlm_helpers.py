"""Small helpers shared by the tests for labeling clips with different models.

None of these tests need a GPU, the network, or a Modal account.
"""
import json

from robovid.frames import Clip, Frame

QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"


def make_clip(n=20, start=10.0):
    """A clip with `n` one-second frames whose "JPEG" bytes are just labels."""
    frames = [Frame(t=start + i, jpeg=f"jpeg-{i}".encode()) for i in range(n)]
    return Clip(start, start + n - 1, frames)


def model_reply(**over):
    """A typical model answer: valid label JSON wrapped in a Markdown code block.

    It also includes a made-up source_url, so tests can check that we ignore it.
    """
    body = {
        "task": "loosening a bolt", "viewpoint": "egocentric", "has_human_hands": True,
        "usable_for_robotics": True,
        "steps": [{"start_s": 11, "end_s": 13, "verb": "unscrew", "objects": ["bolt"],
                   "hand": "right"},
                  {"start_s": 13, "end_s": 15, "verb": "grasp", "objects": ["wrench"]}],
        "source_url": "made-up-by-model",
    }
    body.update(over)
    return "```json\n" + json.dumps(body) + "\n```"
