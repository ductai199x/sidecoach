# Sidecoach

A chess coach that watches your screen while you play. It finds the board on any site
from pixels alone, follows the game move by move, analyses every position with Stockfish,
and coaches the way a good club coach does: it asks before it tells, and it hands over the
move only when you find it, play it, or ask for it. It speaks, it listens, and it
remembers how you play from one game to the next.

No browser extension, no account, no reading of the site: the board is read from the
screen, so it works on chess.com, Lichess, or anything else that draws a chessboard.

---

## What it does

- **Reads the board from the screen.** A frozen pretrained net names the pieces from a
  single look at the start position; every later move is recognised from the pixels,
  through animations, drags, arrows, hints, highlights and premoves. Takebacks and
  resyncs are handled; a position joined mid-game is read cold.
- **Analyses with Stockfish and a motif tagger.** Every position gets engine lines, a
  verdict on the move played (book, best, excellent, good, inaccuracy, mistake, miss,
  blunder, brilliant), and the tactical and strategic ideas in it: forks, pins,
  skewers, discovered attacks, loose pieces, passed pawns, files, outposts, plans.
- **Coaches Socratically.** Before you move it asks and hints, from the theme down to the
  piece, never the move; after you move it says what the move was, and on a mistake,
  what was better and why. What the model may say is enforced by filtering the facts
  it sees, not by asking it to behave.
- **Speaks and listens.** A local voice (Kokoro) reads the coach aloud, notation
  included; local speech recognition (Parakeet) takes your questions, push-to-talk (a key
  held from any window) or hands-free, corrected against the legal moves on the board.
  Your words appear in the text box as you say them and go to the coach a second after
  you stop, or later if you keep going -- or type, and they are yours to edit and send.
- **Learns you.** Every game feeds a player model: where you lose ground and why. The
  coach carries that into the next session and recommends Lichess puzzles for the
  weak spot, playable inside the app.
- **Works without a model.** With no LLM configured, the coach still speaks from the
  board and its templates: the word on every move, the lesson on a mistake, the
  questions and hints, the review. Free and offline. With a model (OpenAI, DashScope,
  or one running on your own machine) it explains and answers in full.
- **Records nothing unless you ask.** The app keeps a few seconds of frames in memory and
  writes nothing to disk on its own; *Debug › Record this session* saves the screen, the
  events and the log to the data folder to show a bug, and stops when you say. API keys go
  to the operating system's credential store.

---

## Results

| What | Measure | Result |
|---|---|---|
| Coach replies, gpt-5.6-luna, held out (320 moments, 20 games) | judged pass / fabrication / disclosure leak | **.994 / .00 / .00** (from .89 / .06 / .00 before the harness fixes and the distilled lessons) |
| Board reading, synthetic sweep (38 piece sets, 8 themes, 44–144 px, every decoration a site draws) | games read ply for ply | **901 of 901** |
| Board reading, recorded live games | games read ply for ply and ended | **6 of 6** |
| Tactic tagging vs Lichess puzzle themes (4,570 puzzles) | recall: pin / discovered attack / skewer / fork / x-ray | **100% / 99% / 97% / 93% / 97%** |
| Tactic description | true sentence on every agreed tactic | **100%** (the coach never fabricates a tactic) |
| Speech recognition (Parakeet-TDT-0.6B, CPU) | word error rate, raw / after chess correction; latency | **4.5% / 0.0%; ~100 ms** |
| Offline coach, 199 master games from both sides | sentences checked against the board | **20,997 sentences, 0 wrong** |
| ChessQA (board-reading benchmark, 250 items), gpt-5.6-luna with the structured prompt | accuracy | **.89** at medium effort, **.90** at extra-high |
| Desktop app, 2-hour soak | page frame rate / long frames / memory | **140 fps / 0.03% / flat once every list was bounded** |

Method and details: [docs/eval-results.md](docs/eval-results.md) (the coach, the judge,
the ICR rounds, spend), [DESIGN.md](DESIGN.md) (every decision, with its numbers),
[docs/app-plan.md](docs/app-plan.md) (the desktop app's measurements).

---

## Install (Windows)

Download `Sidecoach_<version>_x64-setup.exe` from the releases page and run it. It
installs per user and starts; run it again later and it simply opens the app. The
installer is not code-signed, so the first run of a downloaded copy shows SmartScreen's
"Windows protected your PC" — *More info*, then *Run anyway*.

On first run a setup window fetches what the coach needs into the app's own data
folder (about 4 GB: the voice, Stockfish, the opening names, speech recognition, a puzzle sample),
and offers GPU acceleration when an NVIDIA card can take it. The CPU is the default;
the GPU is one checkbox.

Then, in the app:

1. **Settings → AI Coach**: pick a provider and paste its API key, choose a model on
   your own machine, or choose *No model* for the free, offline coach.
2. Open the site you play on, press **Start**, and play.
3. Closing the window quits; a setting keeps the coach running in the tray instead.
   Settings apply the moment you save them; the two that need a restart say so.

Linux is in progress (an AppImage and a .deb build and run; screen capture on a real
desktop is the open piece), Android is planned; the plan and its measurements are in
[docs/app-plan.md](docs/app-plan.md). On Linux the voice and the microphone need
`libportaudio2` (the .deb declares it; the AppImage needs it installed), and the GPU
needs the NVIDIA driver.

---

## Run from source

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/). A CUDA-capable NVIDIA
GPU is optional; everything runs on the CPU.

```sh
uv sync                                  # add --group voice for the voice and speech recognition
uv run python scripts/fetch_stockfish.py
uv run python scripts/fetch_openings.py
uv run sidecoach doctor                  # checks Python, the vision libraries, Stockfish,
                                         # the model endpoint and screen capture
```

Copy `sidecoach.example.toml` to `sidecoach.toml` for the settings; API keys go in
`.env` (`OPENAI_API_KEY=...`, `DASHSCOPE_API_KEY=...`) or the app's settings window.

```sh
uv run sidecoach panel --monitor 1       # the panel at http://127.0.0.1:8765/ (--browser opens it)
uv run sidecoach watch --monitor 1       # the same pipeline in the terminal, without the coach
```

The panel shows the board as the pipeline sees it, the coach's messages, the voice
controls and a debug terminal. When a game ends it shows the review — accuracy, the
tally, where the game turned and why — and the coach sums it up. A development session
can be recorded (`--record`) and replayed stage by stage:

```sh
uv run sidecoach replay data/recordings/<stamp> --moves --pgn game.pgn
uv run sidecoach review game.pgn         # what each move cost, where the game turned
uv run --with zstandard sidecoach puzzles fetch   # fill the puzzle store once
```

### Building the desktop app

Rust (stable, MSVC), Node 24 and uv.

```sh
uv run python scripts/build_sidecar.py        # the embedded interpreter (desktop/resources/python)
cd desktop && npm ci && npm run tauri build   # the installer under desktop/src-tauri/target/release/bundle/nsis/
```

`npm run build:app:fast` builds the same executable without the installer in about half
a minute (no whole-program optimisation; `target/fast/`), for trying a change on the
installed app. `npm run tauri dev` runs the shell against the Vite dev server; `npm run build:web`
copies the built page into `src/sidecoach/ui/web/` for the browser flow. The updater's
signature needs `TAURI_SIGNING_PRIVATE_KEY`; without it, `npm run tauri build --
--no-bundle` builds the executable alone.

---

## How it works

```
screen  →  capture  →  vision  →  state  →  analysis  →  coach  →  panel / voice
           frames      board,     legal     Stockfish,   facts,     Svelte page,
           in memory   pieces,    game,     motifs,      policy,    Kokoro, Parakeet
                       moves      clocks    verdicts     LLM
```

Each stage talks to the next only through typed events (`sidecoach/events.py`). The
coach never sees the whole analysis: a disclosure machine filters the fact bundle by
level — nature, theme, region, piece, candidate, line — and the level rises only as you
ask, or find the move, or play it. Eval loss is measured in expected score (a move's
cost in winning chances), never in centipawns.

The model is any OpenAI-compatible endpoint: a preset (OpenAI, DashScope, a local server
such as LM Studio) or a custom one. The coach was evaluated on `gpt-5.6-luna` at medium
effort; its prompt carries twelve distilled lessons produced by iterative context
refinement over its own judged replies, and every draft is checked against the board
before it is spoken.

---

## Development

```sh
uv run pytest                            # ~1000 tests, parallel, about a minute and a half
uv run pytest tests/test_stress.py --seeds 1000   # the vision sweep: random games over every
                                         # piece set, theme, size and decoration (~7 min)
uv run python tools/eval/offline_sweep.py   # the offline coach over 199 master games (~2 min)
uv run ruff check . && uv run ruff format .
uv run pyright
```

No test reads anything under `data/`: boards are rendered from open-licensed piece sets,
and a few real board crops sit under `tests/fixtures/`. `tools/eval/` holds the coach's
evaluation and the ICR tooling; `tools/puzzles/` validates the tactic tagger against
Lichess puzzles. `data/`, `tools/stockfish/` and `refs/` are local and never committed.

Layout: one package, `sidecoach`, one subpackage per stage — `capture`, `vision`,
`state`, `analysis`, `coach`, `profile`, `voice`, `llm`, `ui`, `setup`, `replay` — and
the desktop shell under `desktop/` (Tauri 2, SvelteKit).

---

## Documentation

- [DESIGN.md](DESIGN.md) — scope, policy, architecture, every stage's design and the
  decisions taken, with their numbers.
- [docs/eval-results.md](docs/eval-results.md) — how the coach is evaluated and trained
  in context, and what it scores.
- [docs/app-plan.md](docs/app-plan.md) — the desktop and mobile apps: the plan, the
  spike's measurements, the soaks.
- [docs/coach-plan.md](docs/coach-plan.md) — the coach's roadmap.

Piece sets used for the synthetic tests are Lichess's (open licences, credited under
`tests/fixtures/pieces/`); chess.com's assets are only ever observed on the user's own
screen. The piece-naming net is Ultralytics YOLO26n-cls (AGPL-3.0 weights), frozen and
exported to ONNX.
