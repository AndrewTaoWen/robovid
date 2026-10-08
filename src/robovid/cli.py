"""Command line entry point.

    robovid annotate path/to/video.mp4 --out annotations.jsonl
    robovid fetch --urls examples/urls.txt --out-dir data/videos
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .frames import chunk_into_clips, sample_frames


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
    from .annotate import ClaudeAnnotator

    annotator = ClaudeAnnotator(model=args.model)
    out = Path(args.out)
    n = 0
    with out.open("a") as fh:
        for video in args.videos:
            frames = sample_frames(video, fps=args.fps)
            for clip in chunk_into_clips(frames, clip_len_s=args.clip_len):
                ann = annotator.annotate(clip, video_id=Path(video).stem)
                fh.write(ann.model_dump_json() + "\n")
                n += 1
                print(f"{Path(video).stem} [{clip.start_s:.0f}-{clip.end_s:.0f}s] "
                      f"{ann.task} ({len(ann.steps)} steps)")
    print(f"wrote {n} clip annotations to {out}")


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
    a.set_defaults(func=cmd_annotate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
