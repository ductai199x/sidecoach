#!/usr/bin/env bash
# The five master-game play-throughs, in parallel, each to its own .md and .log.
#   tools/eval/run_playthroughs.sh <suffix> [extra playthrough.py flags...]
# e.g. tools/eval/run_playthroughs.sh -sum --summarise-every 20
set -u
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/../.."
suffix="${1:-}"; shift || true
extra=("$@")
run() {
  local game="$1" side="$2" name="$3" dump="$4"
  uv run --no-sync python tools/eval/playthrough.py --game "$game" --side "$side" --dump "$dump" "${extra[@]}" \
    > "data/eval/playthrough-$name$suffix.md" 2> "data/eval/playthrough-$name$suffix.log"
  echo "DONE" >> "data/eval/playthrough-$name$suffix.log"
}
run fischer__017-fischer-tal black fischer-tal 16,47,70,100 &
run mammoth__080-kasparov-karpov white kasparov-karpov 20,51,66 &
run mammoth__124-carlsen-kramnik white carlsen-kramnik 18,44,64,80 &
run mammoth__119-anand-carlsen black anand-carlsen 16,28,52,66 &
run mammoth__105-kasparov-topalov white kasparov-topalov 14,35,62,84 &
wait
echo "ALL DONE"
