"""Summarize a robovid annotations file, plus local-vs-Modal timing if given.

    python3 scripts/summarize.py scale-modal.jsonl --local-jsonl scale-local.jsonl \
        --local-seconds 300 --local-videos 3 --modal-seconds 240 --modal-videos 40
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def footage_seconds(rows: list[dict]) -> float:
    """Total video length covered: the last clip end of each video, summed."""
    ends: dict[str, float] = {}
    for r in rows:
        ends[r["video_id"]] = max(ends.get(r["video_id"], 0.0), r["clip_end_s"])
    return sum(ends.values())


def fmt(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--local-jsonl")
    ap.add_argument("--local-seconds", type=float)
    ap.add_argument("--local-videos", type=int)
    ap.add_argument("--modal-seconds", type=float)
    ap.add_argument("--modal-videos", type=int)
    ap.add_argument("--max-containers", type=int)
    a = ap.parse_args()

    rows = load(a.jsonl)
    if not rows:
        raise SystemExit(f"no annotations in {a.jsonl}")
    videos = {r["video_id"] for r in rows}
    steps = [s for r in rows for s in r.get("steps", [])]
    verbs = collections.Counter(s["verb"] for s in steps)
    usable = sum(bool(r.get("usable_for_robotics")) for r in rows)
    with_src = sum(bool(r.get("source_url")) for r in rows)
    footage = footage_seconds(rows)

    print(f"videos annotated:     {len(videos)}")
    print(f"footage:              {fmt(footage)}")
    print(f"clips:                {len(rows)}  ({usable} usable for robotics, "
          f"{usable / len(rows):.0%})")
    print(f"labeled steps:        {len(steps)}  ({len(verbs)} distinct verbs, "
          f"{verbs['other']} 'other')")
    print(f"clips with source:    {with_src} of {len(rows)}")
    print(f"top verbs:            {', '.join(f'{v} {n}' for v, n in verbs.most_common(8))}")

    if a.modal_seconds and a.local_seconds and a.local_jsonl:
        local_rows = load(a.local_jsonl)
        local_footage = footage_seconds(local_rows)
        if local_footage:
            # how long one-at-a-time annotation takes per second of video,
            # scaled to everything Modal processed
            serial = a.local_seconds / local_footage * footage
            print()
            print(f"local, one at a time: {fmt(a.local_seconds)} for {a.local_videos} videos "
                  f"({fmt(local_footage)} of footage)")
            print(f"estimated serial:     {fmt(serial)} for all {len(videos)} videos")
            print(f"on Modal:             {fmt(a.modal_seconds)} wall clock"
                  + (f", up to {a.max_containers} containers" if a.max_containers else ""))
            print(f"speedup:              {serial / a.modal_seconds:.1f}x")
            print()
            print("resume line:")
            print(f"  Scaled the annotation pipeline on Modal across up to {a.max_containers} "
                  f"parallel containers, labeling {len(videos)} videos ({len(rows)} clips) in "
                  f"{fmt(a.modal_seconds)} versus about {fmt(serial)} run one at a time")


if __name__ == "__main__":
    main()
