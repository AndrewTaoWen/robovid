# robovid

Turn internet video into labeled training data for robot learning.

Robot manipulation models need large amounts of footage of people handling
objects, labeled with what happens and when. Teleoperation data is slow and
expensive to collect. Online video already shows almost every everyday task,
but it's unlabeled. robovid samples frames from video, asks a vision-language
model to segment each clip into atomic manipulation steps, and writes the
result as a JSONL dataset.

## Pipeline

```
fetch (open-license filter) → sample frames → chunk into clips → VLM annotation → JSONL
```

Each line of the output is one clip:

```json
{
  "video_id": "abc123",
  "source_url": "https://www.youtube.com/watch?v=abc123",
  "license": "Creative Commons Attribution license (reuse allowed)",
  "clip_start_s": 16.0,
  "clip_end_s": 31.0,
  "task": "making pour-over coffee",
  "viewpoint": "egocentric",
  "has_human_hands": true,
  "usable_for_robotics": true,
  "steps": [
    {"start_s": 17.0, "end_s": 19.0, "verb": "grasp", "objects": ["kettle"], "hand": "right"},
    {"start_s": 19.0, "end_s": 27.0, "verb": "pour", "objects": ["kettle", "filter"], "hand": "right"}
  ]
}
```

## Quickstart

```bash
pip install -e .
export ANTHROPIC_API_KEY=...

# 1. Download only Creative Commons videos from a list of URLs
robovid fetch --urls examples/urls.txt --out-dir data/videos

# 2. Annotate
robovid annotate data/videos/*.mp4 --out annotations.jsonl
```

Run tests with `python -m unittest discover tests`.

## Run at scale on Modal

The `modal` backend runs the same per-video pipeline (frame sampling → clips →
VLM annotation) on [Modal](https://modal.com), one container per video, fanned
out in parallel. Results are merged back in input order into the same JSONL
schema as the local backend.

```bash
pip install -e '.[modal]'        # or: pip install modal
modal setup                      # authenticate once
modal secret create anthropic ANTHROPIC_API_KEY=...

robovid annotate --backend modal data/videos/*.mp4 --out annotations.jsonl \
    --max-containers 20
```

Local videos are uploaded to a Modal Volume (`robovid-videos`, keyed by content
hash so re-runs don't collide), each video gets its own container with a 30
minute timeout and up to 2 retries, and `--max-containers` caps parallelism
(keep it within your Anthropic rate limits). Videos that still fail are listed
at the end and the command exits non-zero; successful ones are still written.

## Data provenance

By default `fetch` skips any video that isn't Creative Commons or public
domain. For each video it keeps, it writes a sidecar (`<video_id>.json`) with
the source URL, title and license, and `annotate` stamps `source_url` and
`license` on every clip, so each annotation traces back to its source. Re-run
`fetch` on videos you already have to backfill missing sidecars without
downloading them again. Check each platform's terms before collecting at scale.

## Verb vocabulary

Step verbs come from a fixed list (`robovid/verbs.py`: grasp, place, pour,
turn, insert, remove, ...), so "point to", "point_to" and "point at" don't
become three different labels. The model is asked to use that list, and any
verb it returns is mapped onto it afterwards; when the wording changed, the
original is kept in `raw_verb`. Verbs that fit nothing become `other`.

## Roadmap

- [ ] Hand and object bounding boxes (detector pass alongside the VLM)
- [ ] Label-quality eval: compare VLM labels against a hand-labeled set, per model
- [ ] Filter step: drop clips that aren't usable for robotics before paying for full annotation
- [x] Parallel processing for large video sets (Modal: `--backend modal`)
- [ ] Export to formats robotics teams use (e.g. LeRobot, RLDS)
