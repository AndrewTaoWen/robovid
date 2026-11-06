#!/usr/bin/env bash
# Find Creative Commons hands-on videos on YouTube and write their URLs, one per line.
#   usage: scripts/find_videos.sh [out_file] [per_query]
#   env:   MAX_TOTAL (default 40), MIN_S / MAX_S (video length bounds in seconds)
# Uses YouTube's own "Creative Commons" search filter (sp=EgIwAQ==). `robovid fetch`
# still checks each video's license before downloading, so anything that slips
# through is skipped, not downloaded.
set -euo pipefail
OUT="${1:-examples/urls-scale.txt}"
PER_QUERY="${2:-2}"
MAX_TOTAL="${MAX_TOTAL:-40}"
MIN_S="${MIN_S:-90}"    # skip very short clips
MAX_S="${MAX_S:-900}"   # skip anything over 15 minutes (cost and talking-head risk)

QUERIES=(
  "how to make pour over coffee"   "how to knead bread dough"       "knife skills chopping onion"
  "how to make an omelette"        "how to peel and cut a mango"    "how to make dumplings by hand"
  "how to roll sushi at home"      "how to make fresh pasta"        "how to brew loose leaf tea"
  "how to fold a t-shirt"          "how to sew a button"            "hand sewing basic stitches"
  "how to tie a tie"               "how to tie knots tutorial"      "how to wrap a gift"
  "origami crane step by step"     "knitting for beginners cast on" "how to change a bike tube"
  "how to fix a bike chain"        "how to adjust bike brakes"      "how to repot a plant"
  "how to solder wires"            "how to change a door handle"    "how to replace a faucet aerator"
  "how to use a drill to drive screws" "how to paint with a roller" "how to assemble flat pack furniture"
  "how to clean a cast iron pan"   "how to sharpen a knife whetstone" "how to change a light switch"
)

tmp="$(mktemp)"
qi=0
for q in "${QUERIES[@]}"; do
  qi=$((qi + 1))
  enc="$(python3 -c 'import sys,urllib.parse; print(urllib.parse.quote_plus(sys.argv[1]))' "$q")"
  echo "searching: $q" >&2
  yt-dlp --flat-playlist --playlist-end 15 --ignore-errors --no-warnings \
    --match-filters "duration>=${MIN_S} & duration<=${MAX_S} & !is_live" \
    --print "https://www.youtube.com/watch?v=%(id)s" \
    "https://www.youtube.com/results?search_query=${enc}&sp=EgIwAQ%3D%3D" 2>/dev/null \
    | head -n "$PER_QUERY" | awk -v q="$qi" '{print NR "\t" q "\t" $0}' >> "$tmp" || true
done

{
  echo "# Found by scripts/find_videos.sh (YouTube Creative Commons filter). One URL per line."
  # every task's top result first, then second results, so a cap keeps variety
  sort -t "$(printf '\t')" -k1,1n -k2,2n "$tmp" | cut -f3 | awk '!seen[$0]++' | head -n "$MAX_TOTAL"
} > "$OUT"
rm -f "$tmp"
echo "wrote $(grep -c '^https' "$OUT") URLs to $OUT" >&2
