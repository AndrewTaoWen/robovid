"""A model answer that isn't valid labels: skip that one clip, and stop the open
model from writing such answers in the first place."""
import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from robovid.annotate import MAX_STEPS, VLLMAnnotator, annotate_video, label_schema
from robovid.schema import ClipAnnotation
from robovid.verbs import VERBS
from test_pipeline import make_video
from vlm_helpers import QWEN, import_with_fake_modal, model_reply


def cut_off(text, keep=60):
    """An answer cut off partway, like one that hit the length limit."""
    return text[:keep]


class TestSkipBadClips(unittest.TestCase):
    def video(self, d, seconds=40):
        p = Path(d) / "v.mp4"
        make_video(p, seconds=seconds)  # 40 seconds makes 3 clips
        return p

    def test_a_cut_off_answer_skips_only_that_clip(self):
        answers = iter([model_reply(), cut_off(model_reply()), model_reply()])
        annotator = VLLMAnnotator(lambda req: {"model": QWEN, "text": next(answers)})
        with tempfile.TemporaryDirectory() as d:
            err = io.StringIO()
            with redirect_stderr(err):
                anns = list(annotate_video(self.video(d), annotator))
        self.assertEqual(len(anns), 2)  # the first and third clips are kept
        self.assertIn("skipped v at 16s", err.getvalue())

    def test_fields_that_fail_the_checks_skip_the_clip(self):
        bad = model_reply(viewpoint="sideways")  # not one of the allowed viewpoints
        annotator = VLLMAnnotator(lambda req: {"model": QWEN, "text": bad})
        with tempfile.TemporaryDirectory() as d, redirect_stderr(io.StringIO()):
            self.assertEqual(list(annotate_video(self.video(d, seconds=10), annotator)), [])

    def test_service_errors_still_stop_the_video(self):
        annotator = VLLMAnnotator(lambda req: {"model": QWEN, "error": "service down"})
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "service down"):
                list(annotate_video(self.video(d, seconds=10), annotator))


class TestLabelShape(unittest.TestCase):
    def test_shape_matches_the_instructions_and_the_label_fields(self):
        schema = label_schema()
        step = schema["properties"]["steps"]
        self.assertEqual(step["maxItems"], MAX_STEPS)
        self.assertEqual(step["items"]["properties"]["verb"]["enum"], list(VERBS))
        # every field the model is asked for is a real label field
        self.assertLessEqual(set(schema["properties"]), set(ClipAnnotation.model_fields))

    def test_gpu_service_forces_answers_into_that_shape(self):
        vlm, _ = import_with_fake_modal("robovid.modal_vlm")
        guided = mock.Mock(side_effect=lambda json: ("guided", json))
        fake_sp = types.ModuleType("vllm.sampling_params")
        fake_sp.GuidedDecodingParams = guided
        sp_cls = mock.Mock(side_effect=lambda **kw: kw)
        with mock.patch.dict(sys.modules, {"vllm": types.ModuleType("vllm"),
                                           "vllm.sampling_params": fake_sp}):
            params = vlm.make_sampling_params(sp_cls)
        self.assertEqual(params["temperature"], 0.0)
        self.assertEqual(params["guided_decoding"], ("guided", label_schema()))

    def test_gpu_service_still_runs_if_vllm_cannot_force_json(self):
        vlm, _ = import_with_fake_modal("robovid.modal_vlm")
        sp_cls = mock.Mock(side_effect=lambda **kw: kw)
        with mock.patch.dict(sys.modules, {"vllm.sampling_params": None}), \
                redirect_stdout_quiet():
            params = vlm.make_sampling_params(sp_cls)
        self.assertNotIn("guided_decoding", params)
        self.assertEqual(params["max_tokens"], vlm.MAX_TOKENS)


def redirect_stdout_quiet():
    from contextlib import redirect_stdout
    return redirect_stdout(io.StringIO())


if __name__ == "__main__":
    unittest.main()
