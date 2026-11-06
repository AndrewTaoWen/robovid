"""Every model is shown the same thing for a clip: timestamped frames, then the instructions."""
import base64
import os
import sys
import types
import unittest
from unittest import mock

from robovid import annotate
from robovid.annotate import MAX_STEPS, PROMPT, clip_parts
from robovid.verbs import VERBS
from vlm_helpers import make_clip, model_reply


def expected_prompt(clip):
    return PROMPT.format(duration=clip.end_s - clip.start_s, start=clip.start_s,
                         end=clip.end_s, verbs=", ".join(VERBS), max_steps=MAX_STEPS)


class TestClipParts(unittest.TestCase):
    def test_frames_alternate_with_timestamps_then_instructions(self):
        clip = make_clip(n=20)
        parts = clip_parts(clip, max_frames=16)
        self.assertEqual(len(parts), 16 * 2 + 1)
        stride = 20 / 16  # 16 of the 20 frames, spread evenly
        for i in range(16):
            src = clip.frames[int(i * stride)]
            self.assertEqual(parts[2 * i], ("text", f"t={src.t:.1f}s"))
            self.assertEqual(parts[2 * i + 1], ("image", src.jpeg))
        self.assertEqual(parts[-1], ("text", expected_prompt(clip)))

    def test_short_clip_keeps_every_frame(self):
        parts = clip_parts(make_clip(n=5), max_frames=16)
        self.assertEqual(sum(kind == "image" for kind, _ in parts), 5)

    def test_claude_request_is_unchanged(self):
        """Claude still gets exactly the request it got before this change."""
        clip = make_clip(n=20)
        client = mock.Mock()
        client.messages.create.return_value = types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=model_reply())])
        fake_anthropic = types.SimpleNamespace(Anthropic=mock.Mock(return_value=client))
        with mock.patch.dict(sys.modules, {"anthropic": fake_anthropic}), \
                mock.patch.dict(os.environ, {}, clear=False) as env:
            env.pop("ROBOVID_MODEL", None)
            ann = annotate.ClaudeAnnotator().annotate(clip, "vid")

        # The request as the old code built it, written out by hand.
        frames = [clip.frames[int(i * 20 / 16)] for i in range(16)]
        expected = []
        for f in frames:
            expected.append({"type": "text", "text": f"t={f.t:.1f}s"})
            expected.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.b64encode(f.jpeg).decode()}})
        expected.append({"type": "text", "text": expected_prompt(clip)})
        client.messages.create.assert_called_once_with(
            model="claude-sonnet-5-5", max_tokens=2000,
            messages=[{"role": "user", "content": expected}])
        self.assertEqual(ann.model, "claude-sonnet-5-5")


if __name__ == "__main__":
    unittest.main()
