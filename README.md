# Sidecoach

A chess coach that watches your screen while you play. It finds the board from pixels
alone, follows the game move by move, analyses every position with Stockfish, and coaches
the way a good club coach does: it waits until you are stuck or ask, hints before it tells,
and after your move says what it was and what it teaches. It speaks, it listens, it reviews
your games, and it remembers how you play from one game to the next.

No browser extension and no reading of the site: the board is read from the screen, so it
works on chess.com, Lichess, or anything else that draws a chessboard. There is a desktop
app (Windows; Linux in progress) and an Android app, released together under one version.

---

## What it does

- **Reads the board from the screen.** A frozen pretrained net learns the pieces from one
  look at the start position; every later move is read from the pixels, through
  animations, drags, arrows, highlights and premoves. Takebacks and resyncs are followed; a
  game joined in the middle, a puzzle or an analysis board is read cold. The clocks are
  read too.
- **Analyses with Stockfish and a motif tagger.** Every position gets engine lines and the
  ideas in it (forks, pins, skewers, discovered attacks, loose pieces, passed pawns, plans),
  and every move a class as chess.com's Game Review defines them: brilliant, great, best,
  excellent, good, book, inaccuracy, mistake, miss, blunder
  ([docs/move-classification.md](docs/move-classification.md)). Costs are in expected
  score, never centipawns.
- **Coaches in steps.** Silent until you have thought for a while (45 seconds by default,
  sooner when your clock is short) or ask. Then a first hint, a lesson on the position; a
  second, the piece and the question that leads to the move; then the move with its
  explanation, and the line if you want it. After a mistake it says what the move allowed
  and what was better.
- **AI coach or offline templates.** With a model (OpenAI, DashScope, a model on your own
  machine through LM Studio, Ollama or vLLM, or any OpenAI-compatible service) the coach
  answers every question in its own words. The model sees the whole analysis, engine lines
  included, and is told which step of the ladder it is on; every draft is checked against
  the board and a false claim is never shown. With no model the coach speaks from the board
  and its templates: free and offline.
- **Speaks and listens.** A local voice (Kokoro) reads the coach aloud, notation included;
  local speech recognition (Parakeet) takes your questions, push-to-talk from any window or
  hands-free, corrected against the legal moves on the board.
- **Game Review and Stats.** Every watched game, and games brought in from chess.com,
  Lichess or a PGN, can be reviewed move by move with the coach. Stats rates your
  openings, tactics, strategy and endgames across your games, and the coach recommends
  Lichess puzzles for your weak spots, playable in the app.
- **Records nothing unless you ask.** The app keeps a few seconds of frames in memory.
  *Debug › Dump the last frames and the log* saves what it saw just now; *Debug › Record
  sessions* saves every session's frames and log to the data folder, to show a bug. API
  keys go to the operating system's credential store.

---

## Where it stands

- **The coach** (2026-09-26, `gpt-5.6-luna` at medium effort, 298 replies on the locked
  test games): judged excellent in .99 of AI replies against .86 for the templates; 165 of
  166 questions answered. A hand audit found a false claim of the model's own in about 8-12%
  of AI replies, mostly in explanations of lines; the draft check stops what it can name.
  Method and numbers: [docs/eval-results.md](docs/eval-results.md).
- **Board reading** is tested by a seeded sweep of random games over every piece set, theme,
  size and decoration (`tests/test_stress.py`); what still fails, and why, is in
  [docs/known-issues.md](docs/known-issues.md).

---

## Install (Windows)

Run `Sidecoach_<version>_x64-setup.exe`. It installs per user and starts; run it again later
and it opens the app. The installer is not code-signed, so the first run of a downloaded
copy shows SmartScreen's "Windows protected your PC": *More info*, then *Run anyway*.

On first run a setup page fetches what the coach needs into the app's data folder (about
3 GB: the voice, speech recognition, Stockfish, the opening names, and a sample of the
Lichess puzzles). On a machine with a suitable NVIDIA card it also offers the GPU pack, over
a gigabyte more; the optional items start ticked and can be left out.

Then, in the app:

1. **Settings → AI Coach**: pick a provider and paste its API key, point it at a model on
   your own machine, or choose *No model* for the free, offline coach.
2. Open the site you play on, press **Start**, and play.
3. Closing the window quits; a setting keeps the coach running in the tray instead.

Linux: an AppImage and a .deb build and run; screen capture on a real Linux desktop is still
open ([docs/app.md](docs/app.md)). Android: [android/](android/README.md).

---

## Run from source

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/). An NVIDIA GPU is optional.

```sh
uv sync --all-groups                     # plain `uv sync` installs only the dev group:
                                         # the panel and the voice are groups of their own
uv run python scripts/fetch_stockfish.py
uv run python scripts/fetch_openings.py
uv run sidecoach doctor                  # checks Python, the vision libraries, Stockfish,
                                         # the model, screen capture, the voice and the GPU
uv run sidecoach --help                  # every command
```

Copy `sidecoach.example.toml` to `sidecoach.toml` for the settings; API keys go in `.env`
(`OPENAI_API_KEY=...`, `DASHSCOPE_API_KEY=...`) or the settings page.

```sh
uv run sidecoach panel --monitor 1       # the panel at http://127.0.0.1:8765/ (--browser opens it)
uv run sidecoach watch --monitor 1       # the pipeline in the terminal, without the panel
uv run sidecoach replay data/recordings/<stamp> --moves --pgn game.pgn
uv run sidecoach review game.pgn         # what each move cost, where the game turned
uv run sidecoach puzzles fetch           # fill the puzzle store once
```

`panel --record` saves a live session (frames, events, the log) for `replay`.

### Building the desktop app

Rust (stable, MSVC), Node 24 and uv. One script builds the embedded Python (the sidecar),
then the installer, each step logged under `data/build/`:

```sh
uv run --no-sync python scripts/release_desktop.py             # sidecar + installer
uv run --no-sync python scripts/release_desktop.py --install   # and install it; refuses while the app runs
```

One version for every platform, the phone's included, bumped together even when only one
changed; both release scripts refuse a mismatch:

```sh
uv run --no-sync python scripts/bump_version.py <version>
uv run --no-sync python scripts/bump_version.py --check
```

In `desktop/`: `npm run tauri dev` runs the shell against the dev server, `npm run
build:app:fast` builds the executable without the installer, and `npm run build:web` copies
the built page into `src/sidecoach/ui/web/` (commit it: CI checks it is current). The
updater's artifacts need `TAURI_SIGNING_PRIVATE_KEY`; without it the build leaves them out.
More in [desktop/README.md](desktop/README.md).

---

## How it works

```
screen  →  capture  →  vision  →  state  →  analysis  →  coach  →  panel / voice
           frames      board,     legal     Stockfish,   facts,     Svelte page,
           in memory   pieces,    game,     motifs,      policy,    Kokoro, Parakeet
                       moves      clocks    classes      LLM
```

Each stage talks to the next only through typed events (`src/sidecoach/events.py`). The
desktop app is a Tauri 2 shell around the Python pipeline, which runs as a sidecar behind a
local HTTP API; the page is SvelteKit. The Android app is a 1:1 Kotlin port of the pipeline
that serves the same page from the phone.

---

## Development

```sh
uv run pytest tests/test_<what you changed>.py   # the files that import what you changed
uv run pytest                            # the whole suite (unit, synthetic vision, panel,
                                         # server), in parallel; on purpose only
uv run pytest -m slow                    # adds the tests that need the voice models or the GPU
uv run pytest tests/test_stress.py --seeds 1000   # the vision sweep (~7 min)
uv run ruff check . && uv run ruff format .
uv run pyright
```

The agent guard (`.claude/hooks/guard_commands.py`) asks for `FULL_SUITE_ASKED=1` on a
whole-suite run. How to write a test, and the suite's budget: [docs/testing.md](docs/testing.md).
No test reads anything under `data/`: boards are rendered from open-licensed piece sets, and
a few real crops sit under `tests/fixtures/`. `data/`, `tools/stockfish/` and `refs/` are
local and never committed.

Layout: one package, `sidecoach`, one subpackage per stage (`capture`, `vision`, `state`,
`analysis`, `coach`, `profile`, `voice`, `llm`, `ui`, `setup`, `replay`); the desktop shell
under `desktop/`; development tools under `tools/` (`eval/` the coach's evaluation,
`puzzles/` the tagger against Lichess puzzles, `engine/` Stockfish measurements) and build
scripts under `scripts/`.

---

## Documentation

The index of every document, with when to read it: [docs/README.md](docs/README.md). Start
with [DESIGN.md](DESIGN.md), the source of truth for scope and decisions.

Piece sets used for the synthetic tests are Lichess's (open licences, credited under
`tests/fixtures/pieces/`); chess.com's assets are only ever observed on the user's own
screen. The piece-naming net is Ultralytics YOLO26n-cls (AGPL-3.0 weights), frozen and
exported to ONNX.
