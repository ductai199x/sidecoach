# Sidecoach

Vision-first chess coaching companion. Watches the screen, finds the board, reconciles a legal game, annotates with Stockfish, and coaches Socratically through a local LLM. Read `DESIGN.md` before changing architecture; it is the source of truth for scope and decisions.

## Commands

- `uv sync` installs everything. Python 3.12 is pinned in `.python-version`.
- `uv run sidecoach --help` for the CLI. `uv run sidecoach doctor` checks the local setup.
- `uv run --no-sync python scripts/release_desktop.py --install` builds the desktop app (the sidecar, then the installer) and installs it; it refuses while the app is running. The phone: `tools\install.ps1 -Serial <serial>` in the Android repo.
- One version for every platform, bumped together even when only one changed: `uv run --no-sync python scripts/bump_version.py <version>` (desktop files and the phone's `versionName`/`versionCode`); `--check` compares them, and both release scripts refuse a mismatch.
- `uv run pytest` runs the suite: unit, synthetic-vision, panel and server tests, in parallel, about ten seconds; `-m slow` adds the tests that need the voice models. `uv run ruff check .` and `uv run ruff format .` for lint and format. `uv run pyright` for types.

## Conventions

- src layout, single package `sidecoach`, one subpackage per pipeline stage (`capture`, `vision`, `state`, `analysis`, `coach`, `profile`, `ui`, `voice`, `replay`, `llm`). Stages talk only through the pydantic events in `sidecoach/events.py`.
- No DOM reading, no browser automation. Board input is pixels only.
- Disclosure is enforced in code, never by prompt instructions alone. Since 2026-09-26 (Tai's decision) the AI coach sees everything, engine lines at hints included, and the reply is checked before it is spoken: one that gives away the answer is regenerated or replaced by the template. See `sidecoach/coach/disclosure.py` (`leak_check`) and `Speaker._problems`.
- Eval loss and decisive-moment gaps are in expected score, never centipawns.
- Tests use rendered boards (`tests/synth.py`, real piece sets from `tests/pieceset.py` and the sprite sheets under `tests/fixtures/pieces/`) and a few committed crops from real games (`tests/fixtures/`: boards, and the result badges under `fixtures/badges/`); nothing under `data/` is needed to run them. `uv run pytest tests/test_stress.py --seeds 1000` sweeps random games over every theme, set, size and decoration (about seven minutes). Recordings under `data/` are for replays and the validation scripts. Do not commit `data/` or `tools/stockfish/`.
- `tools/eval/` is the coach's evaluation and ICR tooling (committed); `docs/eval-results.md` holds the method, the numbers and the decisions; `data/eval/README.md` the data layout. Every paid model call goes through `judge.Ledger`.
- The LLM is the local server on `http://localhost:1234` behind `sidecoach.llm`. Keep providers swappable.
