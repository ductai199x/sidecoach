# Sidecoach

A vision-first chess coaching companion. It watches your screen, finds the chessboard on any site, reconciles a legal game, annotates every position with Stockfish and a motif tagger, and coaches you Socratically through a local LLM. Before you move it asks and hints; it hands over the move only when you find it, play it, or ask for it.

Design and decisions: [DESIGN.md](DESIGN.md).

## Quickstart

```
uv sync
uv run python scripts/fetch_stockfish.py
uv run sidecoach doctor
```

`doctor` checks Python, the vision libraries, Stockfish, the local LLM server on port 1234, and screen capture.

## Live

```
uv run sidecoach watch --monitor 1
```

Then start a new game against a bot on that monitor. The watcher locks onto the board, bootstraps piece recognition from the start position, and prints every move as it settles. The session is recorded under `data/recordings/`, so anything that looks wrong can be replayed:

```
uv run sidecoach replay data/recordings/<stamp> --moves --pgn game.pgn
uv run sidecoach cells data/recordings/<stamp> --squares e2,e4 --start 100 --end 140
uv run sidecoach shrink data/recordings/g07-name data/recordings/<stamp>   # half size, still lossless
```

## Status

Milestone M1 (see the board) is complete; see the roadmap in DESIGN.md.

## Development

```
uv run pytest
uv run ruff check . && uv run ruff format .
uv run pyright
```

Copy `sidecoach.example.toml` to `sidecoach.toml` to override defaults. `data/` and `tools/` are local and never committed.
