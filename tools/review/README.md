# tools/review: Game Review and Stats calibration

The scripts that checked and fitted the numbers Game Review and Stats show against real
players' games: our accuracy against chess.com's, the performance rating, and Stats' peers
and category ratings. Designs: [docs/game-review.md](../../docs/game-review.md) and
[docs/stats.md](../../docs/stats.md). Inputs and outputs live under `data/review/` and
`data/models/nova/` (the Nova authors' Lichess rapid sample; not committed); what a fit
writes into the app (`--write`) is committed with the code it feeds (`src/sidecoach/profile/peers.json`).

Run from the repo root with `uv run --no-sync python tools/review/<script>.py`; each
script's docstring gives its usage and `--help` its options.

| Script | What it does | Example |
|---|---|---|
| `chesscom_games.py` | A chess.com player's games from the public API, one PGN each, with chess.com's accuracies | `chesscom_games.py <username> [--out data/review/chesscom]` |
| `accuracy_check.py` | Our accuracy (several forms) against chess.com's CAPS2 on games both reviewed | `accuracy_check.py data/review/runs/<user>` |
| `human_losses.py` | What every move of the Lichess rapid sample cost, judged as the review judges it | `human_losses.py --per-band 100 --out FILE` |
| `fit_performance.py` | The performance rating fitted on some players and tested on players it never saw | `fit_performance.py data/review/human/pilot.jsonl --held ... --runs ...` |
| `human_facts.py` | The sample's sides with the facts Stats keeps | `human_facts.py --per-band 200 --out FILE` |
| `fit_stats.py` | Stats' peers and category ratings fitted by rating band | `fit_stats.py data/review/human/facts.jsonl [--write]` |
| `ai_review.py` | The model's review paragraphs for every key move, for reading by hand | `ai_review.py DB [--game KEY \| --games N] [--model local] [--budget USD]` |

`ai_review.py` is free with `--model local` (the default); any other model spends through
`judge.Ledger` (run `ai-review`) up to `--budget` dollars, default 0.
