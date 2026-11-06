#!/usr/bin/env bash
# One-command scale run: fetch videos, time a few locally, run all on Modal, summarize.
#   usage: scripts/scale_run.sh examples/urls-scale.txt
#   env:   N_LOCAL (videos to time locally, default 3), MAX_CONTAINERS (default 20),
#          SKIP_FETCH=1 to reuse videos already in data/scale
set -euo pipefail

# the local backend calls the Anthropic API from this machine, so load the key from .env
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -f .env ]; then
  set -a; . ./.env; set +a
fi
[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "ANTHROPIC_API_KEY is not set (add it to .env)"; exit 1; }
URLS="${1:?pass a file of video URLs, one per line}"
DIR="data/scale"
N_LOCAL="${N_LOCAL:-3}"
MAX_CONTAINERS="${MAX_CONTAINERS:-20}"

if [ "${SKIP_FETCH:-0}" = 1 ]; then
  echo "== 1/4 skipping fetch, using videos already in $DIR"
else
  echo "== 1/4 fetching Creative Commons videos into $DIR"
  robovid fetch --urls "$URLS" --out-dir "$DIR"
fi
shopt -s nullglob
videos=("$DIR"/*.mp4)
echo "   have ${#videos[@]} videos"
[ "${#videos[@]}" -gt 0 ] || { echo "no videos downloaded"; exit 1; }

if command -v ffprobe >/dev/null; then
  for v in "${videos[@]}"; do
    c=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$v" || true)
    [ "$c" = "av1" ] && echo "   warning: $v is AV1 and will fail to decode; re-encode it to H.264"
  done
fi

echo "== 2/4 timing $N_LOCAL video(s) locally (one at a time, as without Modal)"
rm -f scale-local.jsonl
t0=$(date +%s)
robovid annotate --backend local "${videos[@]:0:$N_LOCAL}" --out scale-local.jsonl
t_local=$(( $(date +%s) - t0 ))

echo "== 3/4 annotating all ${#videos[@]} videos on Modal (up to $MAX_CONTAINERS containers)"
rm -f scale-modal.jsonl
t0=$(date +%s)
robovid annotate --backend modal "${videos[@]}" --out scale-modal.jsonl \
  --max-containers "$MAX_CONTAINERS" || echo "   (some videos failed; see above)"
t_modal=$(( $(date +%s) - t0 ))

echo "== 4/4 summary"
python3 scripts/summarize.py scale-modal.jsonl --local-jsonl scale-local.jsonl \
  --local-seconds "$t_local" --local-videos "$N_LOCAL" \
  --modal-seconds "$t_modal" --modal-videos "${#videos[@]}" \
  --max-containers "$MAX_CONTAINERS" | tee scale-summary.txt
