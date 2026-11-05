"""The open-model labeler: same input as Claude, same checks on the answer."""
import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from robovid.annotate import VLLMAnnotator, chat_messages, clip_parts
from robovid.parallel import annotate_video_to_jsonl
from test_pipeline import make_video
from vlm_helpers import QWEN, fake_modal, make_clip, model_reply


class TestChatFormat(unittest.TestCase):
    def test_frames_become_data_url_images_after_their_timestamps(self):
        clip = make_clip(n=20)
        [msg] = VLLMAnnotator(generate=None, max_frames=16).build_request(clip)["messages"]
        self.assertEqual(msg["role"], "user")
        parts = msg["content"]
        self.assertEqual(sum(p["type"] == "image_url" for p in parts), 16)
        stride = 20 / 16
        for i in range(16):
            src = clip.frames[int(i * stride)]
            self.assertEqual(parts[2 * i], {"type": "text", "text": f"t={src.t:.1f}s"})
            url = parts[2 * i + 1]["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/jpeg;base64,"))
            self.assertEqual(base64.b64decode(url.split(",", 1)[1]), src.jpeg)

    def test_both_models_get_the_same_text(self):
        clip = make_clip(n=8)
        shared = [v for k, v in clip_parts(clip, 16) if k == "text"]
        vllm = [p["text"] for p in chat_messages(clip, 16)[0]["content"] if p["type"] == "text"]
        self.assertEqual(shared, vllm)


class TestVLLMAnnotator(unittest.TestCase):
    def test_answer_goes_through_the_same_checks_as_claude(self):
        calls = []

        def generate(req):
            calls.append(req)
            return {"model": QWEN, "text": "Sure!\n" + model_reply()}

        annotator = VLLMAnnotator(generate)
        clip = make_clip(n=6)
        ann = annotator.annotate(clip, "vid1")
        self.assertEqual(len(calls), 1)
        self.assertEqual((ann.video_id, ann.clip_start_s, ann.clip_end_s),
                         ("vid1", clip.start_s, clip.end_s))
        self.assertEqual((ann.model, annotator.name), (QWEN, QWEN))
        self.assertEqual([s.verb for s in ann.steps], ["loosen", "grasp"])  # fixed verb list
        self.assertEqual(ann.steps[0].raw_verb, "unscrew")
        self.assertIsNone(ann.source_url)  # the source never comes from the model

    def test_service_error_raises(self):
        annotator = VLLMAnnotator(lambda req: {"model": QWEN, "error": "context overflow"})
        with self.assertRaisesRegex(RuntimeError, "context overflow"):
            annotator.annotate(make_clip(n=3), "v")

    def test_answer_without_json_raises(self):
        annotator = VLLMAnnotator(lambda req: {"model": QWEN, "text": "I can't tell."})
        with self.assertRaises(ValueError):
            annotator.annotate(make_clip(n=3), "v")

    def test_works_in_the_existing_per_video_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v.mp4"
            make_video(p, seconds=20)
            lines = annotate_video_to_jsonl(
                p, "v", VLLMAnnotator(lambda req: {"model": QWEN, "text": model_reply()}),
                source_url="https://example.com/v", license="CC BY")
        recs = [json.loads(line) for line in lines]
        self.assertEqual(len(recs), 2)
        self.assertTrue(all(r["model"] == QWEN and r["license"] == "CC BY" for r in recs))

    def test_from_modal_connects_to_the_deployed_service(self):
        stub = fake_modal()
        with mock.patch.dict(sys.modules, {"modal": stub}):
            annotator = VLLMAnnotator.from_modal()
        stub.Cls.from_name.assert_called_once_with("robovid-vlm", "VLMServer")
        server = stub.Cls.from_name.return_value.return_value
        self.assertIs(annotator._generate, server.generate.remote)


if __name__ == "__main__":
    unittest.main()
