# tools/puzzles: the coach and the analysis against Lichess puzzles

Lichess puzzles are ground truth: a position, its solution and its themes. These scripts
check the analysis layer (does our engine find the solution, does our tagger see the theme)
and the offline coach (do its hints tell the story the solution tells) against them. The
sample is written under `data/puzzles/` and never committed; the analysis of its positions
is cached there too (`data/puzzles/analysis/`, keyed by the engine settings).

Run from the repo root:

```
uv run --with zstandard python tools/puzzles/fetch.py [per-theme]   # the sample, first
uv run --no-sync python tools/puzzles/validate.py [limit]               # 0: every puzzle
uv run --no-sync python tools/puzzles/coach_bench.py [--source sample|full|themes] [--n 400]
```

| Script | What it does |
|---|---|
| `fetch.py` | Streams the Lichess puzzle database and keeps up to K puzzles per theme of interest |
| `validate.py` | Our motif tagger against the themes Lichess assigned, and `describe_tactic`'s sentences checked |
| `analysis_cache.py` | The positions analysed once, in parallel, as the live coach reads them (imported by the benches) |
| `coach_bench.py` | The offline coach's idea, focus, move and lesson at each rung, checked against the solution and the board |
| `story_checks.py` | The story checks: the first hint's parts agree with the move the player has to find (imported by `coach_bench.py`, `tools/eval/read_batch.py` and `tools/eval/story_masters.py`) |

None of them calls a paid model. The same puzzles feed `tools/eval/puzzle_coach.py` and
`tools/eval/read_batch.py` (the live coach's words on them, for reading).
