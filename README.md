# Sidecoach

A vision-first chess coaching companion. It watches your screen, finds the chessboard on any site, reconciles a legal game, annotates every position with Stockfish and a motif tagger, and coaches you Socratically through an LLM. Before you move it asks and hints; it hands over the move only when you find it, play it, or ask for it.

Design and decisions: [DESIGN.md](DESIGN.md). The coach's evaluation, the ICR method and the numbers: [docs/eval-results.md](docs/eval-results.md).

## Quickstart

```
uv sync                                  # add --group voice for the TTS/STT models
uv run python scripts/fetch_stockfish.py
uv run python scripts/fetch_openings.py
uv run sidecoach doctor
```

`doctor` checks Python, the vision libraries, the video codec, Stockfish, the LLM endpoint, and screen capture. Pieces are named by a frozen pretrained net (YOLO26n-cls, ImageNet weights, nothing trained) shipped as `src/sidecoach/vision/models/yolo26n-embed.onnx`; it runs on the GPU when onnxruntime's CUDA provider initialises (`SIDECOACH_EMBED_DEVICE=cpu` to keep it off). To regenerate the file, `uv sync --group classifier && uv run python tools/classifier/export.py`. The LLM is any OpenAI-compatible endpoint: a preset (`local` for LM Studio on port 1234, `dashscope`, `openai`) or a custom `base_url` + `model`, set under `[llm]` in `sidecoach.toml` (copy `sidecoach.example.toml`); API keys live in `.env`.

## Play

```
uv run sidecoach panel --monitor 1
```

serves the panel at http://127.0.0.1:8765/ (add `--browser` to have it opened for you): the board as the pipeline sees it, the coach's messages, a voice, and a debug terminal. The header's controls are grouped by what they are about: *board* -- which side is at the bottom of the board the coach reads (a puzzle or a position joined mid-way is re-read the other way up; a game seen from its start only turns the panel's board around) and a re-read of the screen; *game* -- a menu for what the board cannot tell (a resignation, a flag, a draw) and for forgetting a game; *voice* -- mute and which voice. The status strip above the terminal keeps the session's figures: the tokens the model calls have used (and their cost, at the prices set under `[llm]` in sidecoach.toml), and, amortised over the session, the time from a move to the first sound of the coach's word on it and the model's time to its first token. A board joined in the middle (a puzzle, an analysis board, a game already under way) is read cold from the cached theme on its own, the side at the bottom taken as the player and, on a puzzle, as the one to move; it is coached like a game but recorded by nobody and summed up by no one. The page fits any window: board beside the coach on a desktop or a landscape phone (drag the dividers to resize, double-click one to reset), stacked on a phone or a tablet held upright. When a game ends the panel shows its review -- accuracy, the tally, where it turned and why -- and the coach sums it up; every game feeds the player model under `data/profile.sqlite`, whose insights the coach carries into the next session (a replay teaches it nothing). After the review the coach recommends a few Lichess puzzles for the weak spot the profile sees, on the card: click one and solve it on the panel's own board (click a piece, click a square; the coach stays quiet unless you press Hint), or open it on Lichess and the panel follows it like any position; either way it is scored (`uv run --with zstandard sidecoach puzzles fetch` fills the puzzle store once; `sidecoach puzzles import <csv>` loads the whole database). Start a game against a bot on that monitor; the pipeline locks onto the board, learns the piece set from the start position, and follows every move. `uv run sidecoach watch --monitor 1` is the same pipeline in the terminal, without the coach.

Every session is recorded under `data/recordings/` (lossless, change-gated), so anything odd can be replayed stage by stage:

```
uv run sidecoach replay data/recordings/<stamp> --moves --pgn game.pgn
uv run sidecoach cells data/recordings/<stamp> --squares e2,e4 --start 100 --end 140
uv run sidecoach review game.pgn        # what each move cost, where the game turned
```

## Development

```
uv run pytest                            # ~800 tests, parallel, about half a minute
uv run pytest tests/test_stress.py --seeds 1000   # the sweep: random games through the
                                         # pipeline over every theme, piece set, size and
                                         # decoration a site draws (about seven minutes)
uv run ruff check . && uv run ruff format .
uv run pyright
```

No test reads anything under `data/`: boards are rendered (`tests/synth.py`), pieces come from the cburnett vectors (`tests/pieceset.py`) and 35 lichess sets as sprite sheets under `tests/fixtures/pieces/`, and seven real board crops sit beside them. `tools/eval/` holds the coach's evaluation and ICR tooling (committed); `data/`, `tools/stockfish/` and `refs/` are local and never committed.
