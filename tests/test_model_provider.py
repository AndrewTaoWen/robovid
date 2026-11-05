"""Choosing which model labels the clips: Claude by default, or the open model on Modal GPUs."""
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from robovid import annotate
from robovid.annotate import VLLMAnnotator
from test_pipeline import make_video
from vlm_helpers import QWEN, import_with_fake_modal, model_reply


class TestModalRun(unittest.TestCase):
    def test_claude_is_the_default_and_gets_the_anthropic_key(self):
        env = {k: v for k, v in os.environ.items() if k != "ROBOVID_MODEL_PROVIDER"}
        with mock.patch.dict(os.environ, env, clear=True):
            app, stub = import_with_fake_modal("robovid.modal_app")
        self.assertEqual(app.MODEL_PROVIDER, "claude")
        stub.Secret.from_name.assert_called_once_with("anthropic")
        self.assertEqual(len(app.SECRETS), 1)

    def test_open_model_needs_no_anthropic_key(self):
        app, stub = import_with_fake_modal("robovid.modal_app",
                                           {"ROBOVID_MODEL_PROVIDER": "vllm"})
        stub.Secret.from_name.assert_not_called()
        self.assertEqual(app.SECRETS, [])

    def test_unknown_model_choice_is_rejected(self):
        with self.assertRaises(ValueError):
            import_with_fake_modal("robovid.modal_app", {"ROBOVID_MODEL_PROVIDER": "gpt"})

    def test_each_video_container_uses_the_chosen_model(self):
        app, _ = import_with_fake_modal("robovid.modal_app",
                                        {"ROBOVID_MODEL_PROVIDER": "vllm"})
        fake = VLLMAnnotator(lambda req: {"model": QWEN, "text": model_reply()})
        with tempfile.TemporaryDirectory() as d:
            make_video(Path(d) / "v.mp4", seconds=10)
            with mock.patch.object(app, "VIDEO_DIR", d), \
                    mock.patch.object(VLLMAnnotator, "from_modal", return_value=fake) as fm, \
                    mock.patch.object(annotate, "ClaudeAnnotator") as claude:
                lines = app.annotate_one("v.mp4", "v", 1.0, 16.0, None, provider="vllm")
                fm.assert_called_once_with()
                claude.assert_not_called()
                app.make_annotator("claude", "some-claude")
                claude.assert_called_once_with(model="some-claude")
        self.assertEqual(json.loads(lines[0])["model"], QWEN)

    def test_mismatched_model_choice_is_rejected(self):
        app, _ = import_with_fake_modal("robovid.modal_app",
                                        {"ROBOVID_MODEL_PROVIDER": "claude"})
        with self.assertRaises(ValueError):
            app.run_on_modal(["a.mp4"], "o.jsonl", model_provider="vllm")

    def test_missing_gpu_service_stops_before_uploading(self):
        app, stub = import_with_fake_modal("robovid.modal_app",
                                           {"ROBOVID_MODEL_PROVIDER": "vllm"})
        stub.Cls.from_name.return_value.hydrate.side_effect = RuntimeError("not found")
        with mock.patch.object(app, "upload_videos") as upload:
            with self.assertRaisesRegex(RuntimeError, "modal deploy -m robovid.modal_vlm"):
                app.run_on_modal(["a.mp4"], "o.jsonl", model_provider="vllm")
        upload.assert_not_called()


class TestCommandLine(unittest.TestCase):
    def parse(self, *argv):
        from robovid.cli import build_parser
        return build_parser().parse_args(["annotate", "a.mp4", *argv])

    def run_modal(self, *argv):
        from robovid import cli

        fake = types.ModuleType("robovid.modal_app")
        fake.run_on_modal = mock.Mock(return_value=(3, []))
        with mock.patch.dict(sys.modules, {"robovid.modal_app": fake}), \
                mock.patch.dict(os.environ, {}) as env:
            cli.cmd_annotate(self.parse("--backend", "modal", *argv))
            provider = env["ROBOVID_MODEL_PROVIDER"]
        return fake.run_on_modal, provider

    def test_defaults_are_claude_and_this_machine(self):
        args = self.parse()
        self.assertEqual((args.model_provider, args.backend), ("claude", "local"))

    def test_open_model_on_modal(self):
        run, provider = self.run_modal("--model-provider", "vllm")
        self.assertEqual(provider, "vllm")
        run.assert_called_once_with(["a.mp4"], "annotations.jsonl", fps=1.0, clip_len_s=16.0,
                                    model=None, model_provider="vllm")

    def test_modal_defaults_to_claude(self):
        run, provider = self.run_modal()
        self.assertEqual(provider, "claude")
        self.assertEqual(run.call_args.kwargs["model_provider"], "claude")

    def test_this_machine_uses_the_chosen_labeler(self):
        from robovid import cli

        with mock.patch.object(VLLMAnnotator, "from_modal", return_value="vllm-ann") as fm, \
                mock.patch.object(annotate, "ClaudeAnnotator", return_value="claude-ann") as c:
            self.assertEqual(cli._make_annotator(self.parse("--model-provider", "vllm")),
                             "vllm-ann")
            self.assertEqual(cli._make_annotator(self.parse("--model", "m")), "claude-ann")
        fm.assert_called_once_with()
        c.assert_called_once_with(model="m")

    def test_model_flag_is_rejected_for_the_open_model(self):
        from robovid import cli

        with self.assertRaises(SystemExit):
            cli.cmd_annotate(self.parse("--model-provider", "vllm", "--model", "x"))

    def test_robovid_imports_without_vllm_or_modal(self):
        import robovid.annotate  # noqa: F401
        import robovid.cli  # noqa: F401

        self.assertNotIn("vllm", sys.modules)
        self.assertNotIn("modal", sys.modules)


if __name__ == "__main__":
    unittest.main()
