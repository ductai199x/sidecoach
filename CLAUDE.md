# Sidecoach

Vision-first chess coaching companion. Watches the screen, finds the board, reconciles a legal game, annotates with Stockfish, and coaches Socratically through an LLM (or offline templates). Read `DESIGN.md` before changing architecture; it is the source of truth for scope and decisions. `docs/README.md` indexes the other docs. The Android app (`D:\sidecoach-android`) is a 1:1 Kotlin port of this code, checked by parity fixtures: a change here gets its Kotlin twin there.

## Commands

- `uv sync --all-groups` installs everything (plain `uv sync` installs only the dev group). Python 3.12 is pinned in `.python-version`.
- `uv run sidecoach --help` for the CLI. `uv run sidecoach doctor` checks the local setup.
- `uv run --no-sync python scripts/release_desktop.py --install` builds the desktop app (the sidecar, then the installer) and installs it; it refuses while the app is running. The phone: `tools\install.ps1 -Serial <serial>` in the Android repo.
- One version for every platform, bumped together even when only one changed: `uv run --no-sync python scripts/bump_version.py <version>` (desktop files and the phone's `versionName`/`versionCode`); `--check` compares them, and both release scripts refuse a mismatch.
- `uv run pytest` runs the suite: unit, synthetic-vision, panel and server tests, in parallel. Never the whole suite unless Tai asks: run the test files that import what you changed (a guard asks for `FULL_SUITE_ASKED=1` otherwise). `-m slow` adds the tests that need the voice models or the GPU; `--seeds N` widens the seeded sweeps. `uv run ruff check .` and `uv run ruff format .` for lint and format. `uv run pyright` for types.

## Conventions

- src layout, single package `sidecoach`, one subpackage per pipeline stage (`capture`, `vision`, `state`, `analysis`, `coach`, `profile`, `ui`, `voice`, `replay`, `llm`). Stages talk only through the pydantic events in `sidecoach/events.py`.
- No DOM reading, no browser automation. Board input is pixels only.
- Disclosure: since 2026-09-26 (Tai's decision) the AI coach sees everything, engine lines at hints included, and the rung is stated in its task. What a reply gives away of the answer is logged (`Speaker._leaks`, from `disclosure.leak_check`) as a debugging signal, never a reason to refuse it: an occasional hint that says too much is minor, a vague coach is not. False claims about the board still refuse a draft (`Speaker._problems`, `claims.check_draft`).
- Eval loss and decisive-moment gaps are in expected score, never centipawns.
- How to write a test (the suite's budget, logic-not-timing, caches, duplicates): `docs/testing.md`. Every new test follows it.
- Tests use rendered boards (`tests/synth.py`, real piece sets from `tests/pieceset.py` and the sprite sheets under `tests/fixtures/pieces/`) and a few committed crops from real games (`tests/fixtures/`: boards, and the result badges under `fixtures/badges/`); nothing under `data/` is needed to run them. `uv run pytest tests/test_stress.py --seeds 1000` sweeps random games over every theme, set, size and decoration (about seven minutes). Recordings under `data/` are for replays and the validation scripts. Do not commit `data/` or `tools/stockfish/`.
- `tools/eval/` is the coach's evaluation and ICR tooling (committed); `docs/eval-results.md` holds the method, the numbers and the decisions; `data/eval/README.md` the data layout. Every paid model call goes through `judge.Ledger`.
- The LLM sits behind `sidecoach.llm`: the default preset is OpenAI's gpt-5.6-luna (Responses API, reasoning effort none); the local server on `http://localhost:1234` and `none` (offline templates) are the others. Keep providers swappable.
- Move labels follow chess.com's published definitions: `docs/move-classification.md`.
