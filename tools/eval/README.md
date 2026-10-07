# tools/eval: the coach's evaluation and ICR tooling

Scripts that measure what the coach says: the benchmark, the judge, the rubric, the ICR
rounds that distilled the coach's lessons, and the free regression sweeps of the offline
templates. The method, the numbers and the decisions are in
[docs/eval-results.md](../../docs/eval-results.md); the data layout (all under `data/eval/`,
not committed) in `data/eval/README.md`.

Every script is run from the repo root with `uv run --no-sync python tools/eval/<script>.py`;
each documents its own arguments in its docstring and `--help`. The tests import some of
them as modules (`pyproject.toml`'s pyright `extraPaths`).

## Money

Scripts marked **paid** call a hosted model when given one. Every such call goes through
`judge.Ledger` (`data/eval/usage.jsonl`, and one file per process under `data/eval/usage/`),
whose budget stops the run once spent; a run started again with the same name counts what it
already spent. Before any paid run: Tai names the amount, `regex_scan.py` and a free dry run
(`--dry-run`, `estimate.py`, `dump_prompts.py`) come first, then a pilot of about 40 items.
The model specs (`providers.py`): `none` (the templates, free), `local` (the app's own local
server, from `sidecoach.toml`; free), `openai:<model>@<effort>`, `dashscope`, `flash`. The judge and the coach in the
tools default to `openai:gpt-5.6-luna@medium` where they take a model.

## The scripts

| Script | What it does | Paid |
|---|---|---|
| `bench.py` | The coach benchmark: Nova games at five levels, a frozen dev/test split, fixed moments (`games`, `draw`, `refresh`, `run`, `report`) | `run` |
| `parity.py` | The floor test: each AI reply against the template at the same moment, judged | yes |
| `rubric.py`, `rubric.toml` | The excellence rubric: yes/no criteria per reply (the criteria are in the TOML) | yes |
| `judge.py` | The judge (deterministic checks, then binary votes) and `Ledger` | yes |
| `gold.py` | The judge against hand labels; a run judged again by the judge as it is now | yes |
| `align.py` | Every judge demand asked for in the coach's request, class by class (free) | |
| `report.py`, `compare_rounds.py`, `pairs.py` | Scores per class, paired comparisons, replies beside their templates for reading | |
| `icr_play.py`, `playthrough.py`, `run_playthroughs.sh` | ICR rounds on master games played through the real coach; one play-through; the five in parallel | yes |
| `pool.py`, `split.py`, `taxonomy.py` | The demonstrations collected, the train/held-out split, the classes of what can be asked | |
| `coach_build.py`, `corpus.py` | The coach built as the app builds it; a game replayed through it from cached facts | |
| `facts.py`, `facts_parallel.py` | Engine facts for every ply of the eval games, cached (`facts_parallel.py bench` for the benchmark) | |
| `dump_prompts.py`, `estimate.py` | The requests the coach would send, written out, and what a run would cost: no call made | |
| `regex_scan.py` | Every regex of the coach and the tools against long inputs, for catastrophic backtracking | |
| `offline_sweep.py` | Every game through the coach with no model, every sentence checked against the board | |
| `masters.py`, `masters_notes.py`, `story_masters.py` | The annotated master games: aligned, the coach against the annotators' notes, the story checks on them | |
| `positional_eval.py` | The coach's strategic plans on the books' master positions | |
| `chessqa.py`, `chessqa_coach.py`, `chessqa_prompt_utils.py` | ChessQA on the coach's model; the offline coach against ChessQA | `chessqa.py` |
| `coach_qa.py` | The coach's replies to a user's claims, against puzzle ground truth | |
| `puzzle_coach.py`, `read_batch.py` | A few Lichess puzzles through the live coach; the fixed batch of hints for reading by hand | `puzzle_coach.py` with a provider |
| `coach_games.py` | Every recorded game coached from its PGN with the real LLM, logged for judging | yes |
| `students.py`, `student_loop.py`, `habits.py` | Simulated students with planted habits, and the round-by-round loop on them | yes |
| `providers.py` | The model specs and the ledger wrapper | |
| `stamp.py` | The code version a result was made with | |
| `validate.py` | The evaluation data checked against itself and the board | |
