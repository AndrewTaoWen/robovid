"""Open-model (vLLM on Modal GPUs) backend. No GPU, network, vLLM or Modal needed:
Modal is replaced by a stub whose decorators return the undecorated code."""
import base64
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from robovid import annotate
from robovid.annotate import PROMPT, VLLMAnnotator, chat_messages, clip_parts
from robovid.frames import Clip, Frame
from robovid.parallel import annotate_video_to_jsonl
from robovid.verbs import VERBS
from test_pipeline import make_video

QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"


def make_clip(n=20, start=10.0):
    frames = [Frame(t=start + i, jpeg=f"jpeg-{i}".encode()) for i in range(n)]
    return Clip(start, start + n - 1, frames)


def model_reply(**over):
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


def fake_modal():
    """Just enough of Modal's API for robovid.modal_app / modal_vlm to import."""
    m = types.ModuleType("modal")

    class _Image:
        def __getattr__(self, name):  # .pip_install(), .env(), ... chain
            return lambda *a, **k: self

    class App:
        def __init__(self, name, **kw):
            self.name = name

        def function(self, **kw):
            return lambda f: f

        def cls(self, **kw):
            return lambda c: c

        def local_entrypoint(self, **kw):
            return lambda f: f

    m.App = App
    m.Image = types.SimpleNamespace(debian_slim=lambda *a, **k: _Image())
    m.Volume, m.Secret, m.Retries, m.Cls = mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock()
    m.enter = m.method = lambda *a, **k: (lambda f: f)
    m.batched = lambda *a, **k: (lambda f: f)
    return m


def import_with_fake_modal(module, env=None):
    """Import `module` fresh against the stub; returns (module, stub)."""
    stub = fake_modal()
    with mock.patch.dict(sys.modules, {"modal": stub}), \
            mock.patch.dict(os.environ, env or {}):
        sys.modules.pop(module, None)
        mod = importlib.import_module(module)
    return mod, stub


class TestSharedRequest(unittest.TestCase):
    def test_frames_become_interleaved_image_parts(self):
        clip = make_clip(n=20)
        req = VLLMAnnotator(generate=None, max_frames=16).build_request(clip)
        [msg] = req["messages"]
        self.assertEqual(msg["role"], "user")
        parts = msg["content"]
        images = [p for p in parts if p["type"] == "image_url"]
        self.assertEqual(len(images), 16)  # max_frames respected
        self.assertEqual(len(parts), 16 * 2 + 1)
        # timestamp text, then its frame, evenly subsampled like the Claude path
        stride = 20 / 16
        for i in range(16):
            src = clip.frames[int(i * stride)]
            self.assertEqual(parts[2 * i], {"type": "text", "text": f"t={src.t:.1f}s"})
            url = parts[2 * i + 1]["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/jpeg;base64,"))
            self.assertEqual(base64.b64decode(url.split(",", 1)[1]), src.jpeg)
        self.assertEqual(parts[-1], {"type": "text", "text": PROMPT.format(
            duration=clip.end_s - clip.start_s, start=clip.start_s, end=clip.end_s,
            verbs=", ".join(VERBS))})

    def test_short_clip_keeps_every_frame(self):
        parts = chat_messages(make_clip(n=5), max_frames=16)[0]["content"]
        self.assertEqual(sum(p["type"] == "image_url" for p in parts), 5)

    def test_claude_request_is_unchanged(self):
        """ClaudeAnnotator still sends exactly what it sent before the refactor."""
        clip = make_clip(n=20)
        client = mock.Mock()
        client.messages.create.return_value = types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=model_reply())])
        fake_anthropic = types.SimpleNamespace(Anthropic=mock.Mock(return_value=client))
        with mock.patch.dict(sys.modules, {"anthropic": fake_anthropic}), \
                mock.patch.dict(os.environ, {}, clear=False) as env:
            env.pop("ROBOVID_MODEL", None)
            ann = annotate.ClaudeAnnotator().annotate(clip, "vid")

        # the pre-refactor construction, inlined
        frames = [clip.frames[int(i * 20 / 16)] for i in range(16)]
        expected = []
        for f in frames:
            expected.append({"type": "text", "text": f"t={f.t:.1f}s"})
            expected.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.b64encode(f.jpeg).decode()}})
        expected.append({"type": "text", "text": PROMPT.format(
            duration=clip.end_s - clip.start_s, start=clip.start_s, end=clip.end_s,
            verbs=", ".join(VERBS))})
        client.messages.create.assert_called_once_with(
            model="claude-sonnet-5-5", max_tokens=2000,
            messages=[{"role": "user", "content": expected}])
        self.assertEqual(ann.model, "claude-sonnet-5-5")

    def test_both_providers_get_the_same_text(self):
        clip = make_clip(n=8)
        texts = [v for k, v in clip_parts(clip, 16) if k == "text"]
        vllm_texts = [p["text"] for p in chat_messages(clip, 16)[0]["content"]
                      if p["type"] == "text"]
        self.assertEqual(texts, vllm_texts)


class TestVLLMAnnotator(unittest.TestCase):
    def test_parses_through_shared_path(self):
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
        self.assertEqual(ann.model, QWEN)
        self.assertEqual(annotator.name, QWEN)
        self.assertEqual([s.verb for s in ann.steps], ["loosen", "grasp"])  # normalized
        self.assertEqual(ann.steps[0].raw_verb, "unscrew")
        self.assertIsNone(ann.source_url)  # provenance never from the model
        self.assertTrue(ann.usable_for_robotics)

    def test_service_error_raises(self):
        annotator = VLLMAnnotator(lambda req: {"model": QWEN, "error": "context overflow"})
        with self.assertRaisesRegex(RuntimeError, "context overflow"):
            annotator.annotate(make_clip(n=3), "v")

    def test_non_json_reply_raises(self):
        annotator = VLLMAnnotator(lambda req: {"model": QWEN, "text": "I can't tell."})
        with self.assertRaises(ValueError):
            annotator.annotate(make_clip(n=3), "v")

    def test_works_with_unchanged_per_video_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v.mp4"
            make_video(p, seconds=20)
            lines = annotate_video_to_jsonl(
                p, "v", VLLMAnnotator(lambda req: {"model": QWEN, "text": model_reply()}),
                source_url="https://example.com/v", license="CC BY")
        recs = [json.loads(line) for line in lines]
        self.assertEqual(len(recs), 2)
        self.assertTrue(all(r["model"] == QWEN and r["license"] == "CC BY" for r in recs))

    def test_from_modal_looks_up_deployed_class(self):
        stub = fake_modal()
        with mock.patch.dict(sys.modules, {"modal": stub}):
            annotator = VLLMAnnotator.from_modal()
        stub.Cls.from_name.assert_called_once_with("robovid-vlm", "VLMServer")
        server = stub.Cls.from_name.return_value.return_value
        self.assertIs(annotator._generate, server.generate.remote)


class TestGpuService(unittest.TestCase):
    def setUp(self):
        self.vlm, _ = import_with_fake_modal("robovid.modal_vlm")

    def test_names_match_the_client(self):
        self.assertEqual(self.vlm.APP_NAME, annotate.VLLM_APP_NAME)
        self.assertEqual(self.vlm.CLS_NAME, annotate.VLLM_CLS_NAME)
        self.assertEqual(self.vlm.VLMServer.__name__, annotate.VLLM_CLS_NAME)
        self.assertEqual(self.vlm.MODEL_NAME, QWEN)
        self.assertEqual(self.vlm.MAX_IMAGES, 16)  # matches the annotator's max_frames

    def test_batch_is_one_chat_call(self):
        llm = mock.Mock()
        llm.chat.return_value = [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=t)])
                                 for t in ("a", "b")]
        reqs = [{"messages": ["m1"]}, {"messages": ["m2"]}]
        out = self.vlm.generate_batch(llm, "sp", QWEN, reqs)
        llm.chat.assert_called_once_with([["m1"], ["m2"]], "sp", use_tqdm=False)
        self.assertEqual(out, [{"model": QWEN, "text": "a"}, {"model": QWEN, "text": "b"}])

    def test_one_bad_request_does_not_fail_the_batch(self):
        def chat(convs, sp, use_tqdm):
            if len(convs) > 1 or convs[0] == ["bad"]:
                raise ValueError("prompt too long")
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="ok")])]

        llm = mock.Mock()
        llm.chat.side_effect = chat
        out = self.vlm.generate_batch(llm, "sp", QWEN,
                                      [{"messages": ["good"]}, {"messages": ["bad"]}])
        self.assertEqual(out[0], {"model": QWEN, "text": "ok"})
        self.assertIn("prompt too long", out[1]["error"])
        self.assertEqual(len(out), 2)


class TestModalFanOut(unittest.TestCase):
    def test_claude_is_default_and_gets_anthropic_secret(self):
        env = {k: v for k, v in os.environ.items() if k != "ROBOVID_MODEL_PROVIDER"}
        with mock.patch.dict(os.environ, env, clear=True):
            app, stub = import_with_fake_modal("robovid.modal_app")
        self.assertEqual(app.MODEL_PROVIDER, "claude")
        stub.Secret.from_name.assert_called_once_with("anthropic")
        self.assertEqual(len(app.SECRETS), 1)

    def test_vllm_needs_no_anthropic_secret(self):
        app, stub = import_with_fake_modal("robovid.modal_app",
                                           {"ROBOVID_MODEL_PROVIDER": "vllm"})
        stub.Secret.from_name.assert_not_called()
        self.assertEqual(app.SECRETS, [])

    def test_bad_provider_rejected(self):
        with self.assertRaises(ValueError):
            import_with_fake_modal("robovid.modal_app", {"ROBOVID_MODEL_PROVIDER": "gpt"})

    def test_annotate_one_routes_by_provider(self):
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

    def test_run_on_modal_rejects_provider_mismatch(self):
        app, _ = import_with_fake_modal("robovid.modal_app",
                                        {"ROBOVID_MODEL_PROVIDER": "claude"})
        with self.assertRaises(ValueError):
            app.run_on_modal(["a.mp4"], "o.jsonl", model_provider="vllm")

    def test_missing_gpu_service_fails_fast(self):
        app, stub = import_with_fake_modal("robovid.modal_app",
                                           {"ROBOVID_MODEL_PROVIDER": "vllm"})
        stub.Cls.from_name.return_value.hydrate.side_effect = RuntimeError("not found")
        with mock.patch.object(app, "upload_videos") as upload:
            with self.assertRaisesRegex(RuntimeError, "modal deploy -m robovid.modal_vlm"):
                app.run_on_modal(["a.mp4"], "o.jsonl", model_provider="vllm")
        upload.assert_not_called()


class TestCliModelProvider(unittest.TestCase):
    def parse(self, *argv):
        from robovid.cli import build_parser
        return build_parser().parse_args(["annotate", "a.mp4", *argv])

    def test_default_is_claude(self):
        self.assertEqual(self.parse().model_provider, "claude")
        self.assertEqual(self.parse().backend, "local")

    def test_modal_vllm_routes_to_run_on_modal(self):
        from robovid import cli

        fake = types.ModuleType("robovid.modal_app")
        fake.run_on_modal = mock.Mock(return_value=(3, []))
        with mock.patch.dict(sys.modules, {"robovid.modal_app": fake}), \
                mock.patch.dict(os.environ, {}) as env:
            cli.cmd_annotate(self.parse("--backend", "modal", "--model-provider", "vllm"))
            self.assertEqual(env["ROBOVID_MODEL_PROVIDER"], "vllm")
        fake.run_on_modal.assert_called_once_with(
            ["a.mp4"], "annotations.jsonl", fps=1.0, clip_len_s=16.0, model=None,
            model_provider="vllm")

    def test_modal_default_sets_claude(self):
        from robovid import cli

        fake = types.ModuleType("robovid.modal_app")
        fake.run_on_modal = mock.Mock(return_value=(1, []))
        with mock.patch.dict(sys.modules, {"robovid.modal_app": fake}), \
                mock.patch.dict(os.environ, {}) as env:
            cli.cmd_annotate(self.parse("--backend", "modal"))
            self.assertEqual(env["ROBOVID_MODEL_PROVIDER"], "claude")
        self.assertEqual(fake.run_on_modal.call_args.kwargs["model_provider"], "claude")

    def test_local_backend_picks_annotator(self):
        from robovid import cli

        with mock.patch.object(VLLMAnnotator, "from_modal", return_value="vllm-ann") as fm, \
                mock.patch.object(annotate, "ClaudeAnnotator", return_value="claude-ann") as c:
            self.assertEqual(cli._make_annotator(self.parse("--model-provider", "vllm")),
                             "vllm-ann")
            self.assertEqual(cli._make_annotator(self.parse("--model", "m")), "claude-ann")
        fm.assert_called_once_with()
        c.assert_called_once_with(model="m")

    def test_model_flag_rejected_for_vllm(self):
        from robovid import cli

        with self.assertRaises(SystemExit):
            cli.cmd_annotate(self.parse("--model-provider", "vllm", "--model", "x"))

    def test_package_imports_without_vllm_or_modal(self):
        import robovid.annotate  # noqa: F401
        import robovid.cli  # noqa: F401

        self.assertNotIn("vllm", sys.modules)
        self.assertNotIn("modal", sys.modules)


if __name__ == "__main__":
    unittest.main()
