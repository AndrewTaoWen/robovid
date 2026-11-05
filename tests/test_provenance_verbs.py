"""Provenance sidecars and the fixed verb vocabulary."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from robovid.annotate import annotate_video, build_annotation
from robovid.frames import Clip
from robovid.ingest import VideoMeta, download, read_sidecar, write_sidecar
from robovid.parallel import annotate_video_to_jsonl, plan_jobs
from robovid.schema import ClipAnnotation
from robovid.verbs import VERBS, normalize_verb
from test_pipeline import make_video


class FakeAnnotator:
    name = "fake"

    def annotate(self, clip, video_id):
        return ClipAnnotation(video_id=video_id, clip_start_s=clip.start_s,
                              clip_end_s=clip.end_s, task="t", model=self.name)


META = VideoMeta(video_id="abc", url="https://youtu.be/abc", title="Coffee",
                 license="Creative Commons Attribution license (reuse allowed)",
                 duration_s=40)


class TestVerbs(unittest.TestCase):
    def test_variants_collapse(self):
        # the variants seen in the first Modal run
        for raw in ("point to", "point_to", "point at", "Point-At", "gesture"):
            self.assertEqual(normalize_verb(raw), "point", raw)
        self.assertEqual(normalize_verb("hold up"), "hold")
        self.assertEqual(normalize_verb("pull off"), "remove")

    def test_phrases_and_inflections(self):
        self.assertEqual(normalize_verb("grasp the handle"), "grasp")
        self.assertEqual(normalize_verb("pick up mug"), "grasp")
        self.assertEqual(normalize_verb("pouring"), "pour")
        self.assertEqual(normalize_verb("placed"), "place")
        self.assertEqual(normalize_verb("slices"), "cut")

    def test_unknown_and_empty(self):
        self.assertEqual(normalize_verb("contemplate"), "other")
        self.assertEqual(normalize_verb(""), "other")
        self.assertEqual(normalize_verb(None), "other")

    def test_every_output_is_in_vocabulary(self):
        from robovid.verbs import SYNONYMS
        for target in SYNONYMS.values():
            self.assertIn(target, VERBS)

    def test_build_annotation_normalizes_and_keeps_raw(self):
        raw = {"task": "t", "steps": [
            {"start_s": 1, "end_s": 2, "verb": "point_to", "objects": ["engine"]},
            {"start_s": 2, "end_s": 3, "verb": "grasp", "objects": ["wrench"]},
        ], "source_url": "model-made-this-up"}
        ann = build_annotation(raw, Clip(0, 15, []), "v", "m")
        self.assertEqual([s.verb for s in ann.steps], ["point", "grasp"])
        self.assertEqual(ann.steps[0].raw_verb, "point_to")
        self.assertIsNone(ann.steps[1].raw_verb)  # unchanged, so not duplicated
        self.assertIsNone(ann.source_url)  # provenance never comes from the model


class TestProvenance(unittest.TestCase):
    def test_sidecar_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / "abc.mp4"
            path = write_sidecar(META, video)
            self.assertEqual(path, Path(d) / "abc.json")
            meta = read_sidecar(video)
        self.assertEqual(meta["url"], META.url)
        self.assertEqual(meta["license"], META.license)

    def test_missing_or_bad_sidecar_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(read_sidecar(Path(d) / "none.mp4"), {})
            (Path(d) / "bad.json").write_text("not json")
            self.assertEqual(read_sidecar(Path(d) / "bad.mp4"), {})

    def test_download_backfills_existing_video_without_redownloading(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "abc.mp4").write_bytes(b"already here")
            with mock.patch("robovid.ingest.fetch_meta", return_value=META), \
                    mock.patch("robovid.ingest.subprocess.run") as run:
                download(META.url, d)
            run.assert_not_called()
            self.assertEqual(read_sidecar(Path(d) / "abc.mp4")["license"], META.license)

    def test_annotations_carry_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "abc.mp4"
            make_video(p, seconds=20)
            anns = list(annotate_video(p, FakeAnnotator(), source_url=META.url,
                                       license=META.license))
        self.assertEqual(len(anns), 2)
        self.assertTrue(all(a.source_url == META.url and a.license == META.license
                            for a in anns))

    def test_modal_path_reads_sidecar_into_jobs_and_output(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "abc.mp4"
            make_video(p, seconds=10)
            write_sidecar(META, p)
            plain = Path(d) / "plain.mp4"
            make_video(plain, seconds=10)
            jobs = plan_jobs([p, plain])
            lines = annotate_video_to_jsonl(p, jobs[0].video_id, FakeAnnotator(),
                                            source_url=jobs[0].source_url,
                                            license=jobs[0].license)
        self.assertEqual((jobs[0].source_url, jobs[0].license), (META.url, META.license))
        self.assertEqual((jobs[1].source_url, jobs[1].license), (None, None))
        rec = json.loads(lines[0])
        self.assertEqual((rec["source_url"], rec["license"]), (META.url, META.license))


if __name__ == "__main__":
    unittest.main()
