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

## Label with an open model on your own Modal GPUs

**Why.** By default, every clip is labeled by Claude through Anthropic's API.
That costs money per clip, and Anthropic limits how many requests you can send
per minute, so adding more containers stops helping after a point. This option
labels clips with an open-source model that runs on GPUs you rent from Modal
instead. Nothing is sent to an outside API.

**How it works.**

1. You start a GPU service on Modal once. It loads Qwen2.5-VL-7B-Instruct, an
   open vision-language model (a model that reads images and text), using vLLM,
   open-source software for running such models efficiently.
2. You run `robovid annotate` as usual. Each video still gets its own ordinary
   (CPU) container, which decodes the video and cuts it into 16-second clips.
3. Each container sends its clips to the GPU service, one at a time.
4. The GPU service gathers up to 16 clips, from any of the videos, and labels
   them in one pass. That keeps the GPU busy instead of idle between clips.
5. The answers go through the same checks as Claude's: same instructions, same
   16 frames per clip, same fixed verb list. So you can compare the two models'
   labels clip by clip. Each label records which model made it.

**What to expect.**

- **One L40S GPU (48 GB of memory) per GPU container,** with at most 2 running
  at once by default.
- **The first start is slow.** It downloads about 16 GB of model files into
  Modal storage. Later starts read them from there, which should take a few
  minutes. This hasn't been timed yet.
- **It shuts itself off.** After 5 minutes with no clips, the GPU stops, so you
  only pay while it's working.
- **Not tested on a real GPU yet.** The code is covered by tests that use
  stand-ins for Modal and vLLM. Run the one-clip check below before a big batch.
- **Bad answers are skipped, not fatal.** The open model is forced to answer in the
  label format, with at most 12 steps per clip. If an answer still can't be read,
  only that clip is skipped (it's listed in the run's log), not the whole video.
- **Label quality is unknown.** Open models often give rougher labels than
  Claude. Compare the two on the same clips before relying on it.

**Try it.**

```bash
pip install -e '.[modal]'   # vLLM only gets installed inside the GPU container
modal setup

modal deploy -m robovid.modal_vlm        # start the GPU service (redo after changing settings)
modal run -m robovid.modal_vlm           # check it works by labeling one made-up clip

robovid annotate --backend modal --model-provider vllm data/videos/*.mp4 \
    --out annotations-qwen.jsonl --max-containers 20
```

`--model-provider vllm` also works without `--backend modal`. Then your own
machine decodes the videos and sends the clips to the GPU service.

**Settings.** Set these when you run `modal deploy`:

| Setting | What it changes | Default |
|---|---|---|
| `ROBOVID_VLLM_MODEL` | Which open model to load, by its Hugging Face name | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `ROBOVID_VLLM_GPU` | Modal GPU type, for example `A100-80GB` or `H100` | `L40S` |
| `ROBOVID_VLLM_MAX_CONTAINERS` | Most GPUs running at once | 2 |
| `ROBOVID_VLLM_MAX_BATCH` | Most clips labeled in one pass | 16 |
| `ROBOVID_VLLM_MAX_IMAGES` | Most frames per clip | 16 |
| `ROBOVID_VLLM_MAX_MODEL_LEN` | Most tokens (pieces of text and image) per request | 16384 |
| `ROBOVID_VLLM_MEMORY_MB` | Regular memory (RAM) for each GPU container, in MB | 32768 |
| `ROBOVID_VLLM_CPU` | CPU cores for each GPU container | 4 |

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
- [ ] Label-quality check: compare each model's labels against a hand-labeled set, and Claude against the open model
- [x] Label without an outside API (open model on Modal GPUs: `--model-provider vllm`)
- [ ] Filter step: drop clips that aren't usable for robotics before paying for full annotation
- [x] Parallel processing for large video sets (Modal: `--backend modal`)
- [ ] Export to formats robotics teams use (e.g. LeRobot, RLDS)
