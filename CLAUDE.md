# Sidecoach

Vision-first chess coaching companion. Watches the screen, finds the board, reconciles a legal game, annotates with Stockfish, and coaches Socratically through a local LLM. Read `DESIGN.md` before changing architecture; it is the source of truth for scope and decisions.

## Commands

- `uv sync` installs everything. Python 3.12 is pinned in `.python-version`.
- `uv run sidecoach --help` for the CLI. `uv run sidecoach doctor` checks the local setup.
- `uv run pytest` runs the default suite: the unit and synthetic-vision tests, in parallel, about ten seconds; `uv run pytest -m slow -n 0` adds the recording replays, the server and Stockfish (a minute or two). `uv run ruff check .` and `uv run ruff format .` for lint and format. `uv run pyright` for types.

## Conventions

- src layout, single package `sidecoach`, one subpackage per pipeline stage (`capture`, `vision`, `state`, `analysis`, `coach`, `profile`, `ui`, `voice`, `replay`, `llm`). Stages talk only through the pydantic events in `sidecoach/events.py`.
- No DOM reading, no browser automation. Board input is pixels only.
- Disclosure is enforced by filtering the fact bundle before the LLM sees it, never by prompt instructions alone. See `sidecoach/coach/disclosure.py`.
- Eval loss and decisive-moment gaps are in expected score, never centipawns.
- Recordings under `data/` are the test suite. Do not commit `data/` or `tools/stockfish/`.
- `tools/eval/` is the coach's evaluation and ICR tooling (committed); `docs/eval-results.md` holds the method, the numbers and the decisions; `data/eval/README.md` the data layout. Every paid model call goes through `judge.Ledger`.
- The LLM is the local server on `http://localhost:1234` behind `sidecoach.llm`. Keep providers swappable.
