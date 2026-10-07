# tools/bots: Nova bot games (development only)

Nova (novachess.ai) predicts the move a human of a given rating and style would play. These
scripts use it to make realistic amateur games for testing the coach (the benchmark's games,
`tools/eval/bench.py games`) and to calibrate the simulated students. Nova is a development
tool on this PC only: never in the app, never on the phone. Its weights and LICENSE live in
`data/models/nova/` (gitignored); personal, research and educational use is allowed and the
games it produces are ours, but it may not go into a commercial product or a competing
service (its LICENSE, sections 2 and 5). The yardstick is the Nova authors' Lichess rapid
sample, `data/models/nova/unified_sample_600k.jsonl`.

Run from the repo root with `uv run --no-sync python tools/bots/<script>.py`:

| Script | What it does |
|---|---|
| `nova.py` | The model: `NovaPolicy`, and `temperature_for` (the temperature at which Nova's mistakes match real players' at its rating). Imported, not run |
| `play.py` | Bot games between two Nova players, saved as PGN under `data/bots/games/` (`--games 10 --white 1500[:CLASSICAL[:AGGRESSION]] --black ...`) |
| `calibrate.py` | How far Nova's moves are from real players' of the same rating on the same positions, in expected score (`--per-band`, `--bands`, `--as-rating`) |
| `habit_rates.py` | How often real players meet and miss threats and tactics, by band: the students' targets (`tools/eval/habits.py` `HUMAN_MISS`) |

Ratings are Lichess rapid's: a chess.com 1000-1400 player is about Nova 1300-1700.
