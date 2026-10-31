"""Command line entry point.

    robovid annotate path/to/video.mp4 --out annotations.jsonl
    robovid annotate --backend modal data/videos/*.mp4 --out annotations.jsonl
    robovid fetch --urls examples/urls.txt --out-dir data/videos
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def cmd_fetch(args: argparse.Namespace) -> None:
    from .ingest import download

    urls = [u.strip() for u in Path(args.urls).read_text().splitlines()
            if u.strip() and not u.startswith("#")]
    for url in urls:
        try:
            download(url, args.out_dir, require_open_license=not args.allow_any_license)
        except Exception as e:  # keep going on bad URLs
            print(f"error {url}: {e}")


def cmd_annotate(args: argparse.Namespace) -> None:
    if args.backend == "modal":
        return _annotate_modal(args)

    from .annotate import ClaudeAnnotator, annotate_video
    from .ingest import read_sidecar

    annotator = ClaudeAnnotator(model=args.model)
    out = Path(args.out)
    n = 0
    with out.open("a") as fh:
        for video in args.videos:
            meta = read_sidecar(video)
            if not meta:
                print(f"warning: no provenance sidecar for {video}; "
                      "source_url and license will be empty", file=sys.stderr)
            for ann in annotate_video(video, annotator, fps=args.fps, clip_len_s=args.clip_len,
                                      source_url=meta.get("url"), license=meta.get("license")):
                fh.write(ann.model_dump_json() + "\n")
                n += 1
                print(f"{ann.video_id} [{ann.clip_start_s:.0f}-{ann.clip_end_s:.0f}s] "
                      f"{ann.task} ({len(ann.steps)} steps)")
    print(f"wrote {n} clip annotations to {out}")


def _annotate_modal(args: argparse.Namespace) -> None:
    if args.max_containers is not None:  # read by modal_app at import time
        os.environ["ROBOVID_MODAL_MAX_CONTAINERS"] = str(args.max_containers)
    try:
        from .modal_app import run_on_modal
    except ImportError as e:
        sys.exit(f"--backend modal needs the modal extra: pip install -e '.[modal]' ({e})")

    n, failures = run_on_modal(args.videos, args.out, fps=args.fps,
                               clip_len_s=args.clip_len, model=args.model)
    print(f"wrote {n} clip annotations from {len(args.videos) - len(failures)} "
          f"videos to {args.out}")
    for job, err in failures:
        print(f"failed {job.local_path}: {err!r}", file=sys.stderr)
    if failures:
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(prog="robovid")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download openly licensed videos")
    f.add_argument("--urls", required=True)
    f.add_argument("--out-dir", default="data/videos")
    f.add_argument("--allow-any-license", action="store_true")
    f.set_defaults(func=cmd_fetch)

    a = sub.add_parser("annotate", help="label videos with a VLM")
    a.add_argument("videos", nargs="+")
    a.add_argument("--out", default="annotations.jsonl")
    a.add_argument("--fps", type=float, default=1.0)
    a.add_argument("--clip-len", type=float, default=16.0)
    a.add_argument("--model", default=None)
    a.add_argument("--backend", choices=["local", "modal"], default="local",
                   help="local: annotate here, one video at a time; "
                        "modal: one Modal container per video, in parallel")
    a.add_argument("--max-containers", type=int, default=None,
                   help="modal backend: max parallel containers (default 20)")
    a.set_defaults(func=cmd_annotate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
