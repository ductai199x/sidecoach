# tools/engine: the engine redesign's measurements

The scripts behind the Stockfish redesign of 2026-10-04/05 (every search one line to depth
18, the hint six lines to depth 12, a candidate's verdict from its own position's search):
what each search costs on real Stockfish, how right each depth is, and how the live
analyser should schedule its work. They measure; the app does not import them. Results and
their caches live under `data/engine-cache/` (not committed); `build_report.py` turns them
into one HTML report.

Run each from the repo root with `uv run --no-sync python tools/engine/<script>.py`; the
scripts with arguments document them in their docstring and `--help`, the rest take none.
Most run for minutes to hours on every core: run them in the background with a log.

| Script | What it measures |
|---|---|
| `sessions.py` | The workloads: dumped sessions, the player's chess.com games at their clocks, browsing, a fixed pace (read by the others) |
| `chesscom_games.py` | The player's chess.com games with clocks, into `data/engine-cache/` (`[user]`) |
| `measure.py` | Every workload position searched cleanly (one thread, depth 18, cold), kept in SQLite |
| `atoms.py` | One search's wall time and CPU by lines and threads, when depth 12 arrives, the side searches, searches side by side |
| `single.py` | The one-line design on all 8,886 workload positions, and its slowest positions depth by depth |
| `decided.py` | Whether a decided position's verdict can be read before depth 18 |
| `explode.py` | Where slow searches spend themselves |
| `quality_games.py`, `quality_puzzles.py` | How right each depth is on Tai's games (truth at depth 24) and on Lichess puzzles |
| `sweep2.py`, `sweep2_report.py` | The scheduling policies on a virtual clock, the front of latency against CPU |
| `workers_compare.py` | How many workers the live analyser needs (2, 3, 4, 6) |
| `bench_glue.py` | The app's engine glue end to end, before and after, on three of the player's games |
| `build_report.py` | The redesign report from the measurements (`<out.html>`) |
