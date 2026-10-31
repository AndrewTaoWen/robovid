"""Tests for the parallel/Modal backend that need no Modal credentials or network."""
import argparse
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from robovid.frames import Clip
from robovid.parallel import (
    VideoJob,
    annotate_video_to_jsonl,
    append_jsonl,
    merge_results,
    plan_jobs,
)
from robovid.schema import ClipAnnotation
from test_pipeline import make_video


class FakeAnnotator:
    name = "fake"

    def __init__(self):
        self.calls = []

    def annotate(self, clip: Clip, video_id: str) -> ClipAnnotation:
        self.calls.append((video_id, clip.start_s, len(clip.frames)))
        return ClipAnnotation(video_id=video_id, clip_start_s=clip.start_s,
                              clip_end_s=clip.end_s, task="fake task", model=self.name)


class TestPerVideo(unittest.TestCase):
    def test_annotate_video_to_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v.mp4"
            make_video(p)
            ann = FakeAnnotator()
            lines = annotate_video_to_jsonl(p, "vid1", ann, fps=1.0, clip_len_s=16)
        self.assertEqual(len(lines), 3)  # 40s video -> 3 clips of <=16s
        self.assertEqual(sum(n for _, _, n in ann.calls), 40)
        recs = [json.loads(line) for line in lines]
        self.assertTrue(all(r["video_id"] == "vid1" for r in recs))
        self.assertEqual([r["clip_start_s"] for r in recs], sorted(r["clip_start_s"] for r in recs))
        # every line round-trips through the shared schema
        for line in lines:
            ClipAnnotation.model_validate_json(line)


class TestPlanJobs(unittest.TestCase):
    def test_ids_and_content_keyed_remote_paths(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a", Path(d) / "b"
            a.mkdir(), b.mkdir()
            (a / "clip.mp4").write_bytes(b"one")
            (b / "clip.mp4").write_bytes(b"two")
            (a / "other.mp4").write_bytes(b"one")
            jobs = plan_jobs([a / "clip.mp4", b / "clip.mp4", a / "other.mp4"])
        self.assertEqual([j.video_id for j in jobs], ["clip", "clip", "other"])
        # same name, different content -> no collision on the volume
        self.assertNotEqual(jobs[0].remote_path, jobs[1].remote_path)
        self.assertTrue(jobs[0].remote_path.endswith("/clip.mp4"))
        # same content -> same hash prefix
        self.assertEqual(jobs[0].remote_path.split("/")[0], jobs[2].remote_path.split("/")[0])


class TestMerge(unittest.TestCase):
    def jobs(self, n):
        return [VideoJob(f"v{i}.mp4", f"h{i}/v{i}.mp4", f"v{i}") for i in range(n)]

    def test_order_and_failures(self):
        jobs = self.jobs(3)
        err = RuntimeError("boom")
        lines, failures = merge_results(jobs, [["a1", "a2"], err, ["c1"]])
        self.assertEqual(lines, ["a1", "a2", "c1"])
        self.assertEqual(failures, [(jobs[1], err)])

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            merge_results(self.jobs(2), [["x"]])

    def test_fan_out_with_fake_map(self):
        """End to end: plan -> per-video function over a fake map -> merge -> JSONL."""
        with tempfile.TemporaryDirectory() as d:
            videos = []
            for name, secs in [("first", 20), ("bad", 5), ("second", 10)]:
                p = Path(d) / f"{name}.mp4"
                make_video(p, seconds=secs)
                videos.append(p)
            jobs = plan_jobs(videos)

            def per_video(job):
                if job.video_id == "bad":
                    raise RuntimeError("container died")
                return annotate_video_to_jsonl(job.local_path, job.video_id, FakeAnnotator(),
                                               clip_len_s=16)

            def fake_map(fn, items):  # mimics .map(order_outputs=True, return_exceptions=True)
                out = []
                for it in reversed(items):  # "finish" out of order
                    try:
                        out.append(fn(it))
                    except Exception as e:
                        out.append(e)
                return list(reversed(out))

            lines, failures = merge_results(jobs, fake_map(per_video, jobs))
            out = Path(d) / "out.jsonl"
            out.write_text('{"existing": true}\n')
            n = append_jsonl(lines, out)
            recs = [json.loads(line) for line in out.read_text().splitlines()]

        self.assertEqual(n, 3)  # 20s -> 2 clips, 10s -> 1 clip
        self.assertEqual(recs[0], {"existing": True})  # appended, like the local backend
        self.assertEqual([r["video_id"] for r in recs[1:]], ["first", "first", "second"])
        self.assertEqual([j.video_id for j, _ in failures], ["bad"])


class TestCliModalBackend(unittest.TestCase):
    def args(self, **kw):
        base = dict(videos=["a.mp4", "b.mp4"], out="o.jsonl", fps=1.0, clip_len=16.0,
                    model=None, backend="modal", max_containers=7)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_dispatches_to_modal_app(self):
        from robovid import cli

        fake = types.ModuleType("robovid.modal_app")
        fake.run_on_modal = mock.Mock(return_value=(4, []))
        with mock.patch.dict(sys.modules, {"robovid.modal_app": fake}), \
                mock.patch.dict("os.environ", {}, clear=False) as env:
            cli.cmd_annotate(self.args())
            self.assertEqual(env["ROBOVID_MODAL_MAX_CONTAINERS"], "7")
        fake.run_on_modal.assert_called_once_with(
            ["a.mp4", "b.mp4"], "o.jsonl", fps=1.0, clip_len_s=16.0, model=None)

    def test_exits_nonzero_on_failed_videos(self):
        from robovid import cli

        job = VideoJob("b.mp4", "h/b.mp4", "b")
        fake = types.ModuleType("robovid.modal_app")
        fake.run_on_modal = mock.Mock(return_value=(2, [(job, RuntimeError("x"))]))
        with mock.patch.dict(sys.modules, {"robovid.modal_app": fake}), \
                mock.patch.dict("os.environ", {}, clear=False):
            with self.assertRaises(SystemExit) as cm:
                cli.cmd_annotate(self.args())
        self.assertEqual(cm.exception.code, 1)


class TestOptionalDependency(unittest.TestCase):
    def test_package_imports_without_modal(self):
        import robovid  # noqa: F401
        import robovid.cli  # noqa: F401
        import robovid.parallel  # noqa: F401

        self.assertNotIn("modal", sys.modules)

    @unittest.skipUnless(importlib.util.find_spec("modal"), "modal not installed")
    def test_modal_app_constructs(self):
        from robovid import modal_app

        self.assertEqual(modal_app.app.name, "robovid")
        self.assertTrue(hasattr(modal_app.annotate_one, "starmap"))


if __name__ == "__main__":
    unittest.main()
