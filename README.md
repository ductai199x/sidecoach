# Sidecoach for Android

The chess coaching companion on the phone: open it once, grant screen capture, switch to
the chess app and play; the coach speaks. Phase 3 of `docs/app-plan.md` in the desktop
repo (`D:\chess-coaching-companion`), which is the source of truth for scope and order.

The desktop app is Python behind a Tauri shell. The phone cannot run that interpreter, so
the pipeline is ported to Kotlin file by file. The rule for the port is one to one: the
Kotlin says what the Python says, in the same order, and is held to it by tests that
compare against the Python's own recorded output rather than against a reading of the code.

## Where it stands

**The phone watches a game, follows it, and assesses every move.** Capture, board detection
and tracking, the theme learned from the start position, the recogniser, the legal game
reconciled from what the squares show, and Stockfish on every move. On the emulator,
watching its own screen play a rendered thirty-ply game, it reads every move in order --
each committed within half a second of appearing -- and names what it cost:

```
game started (game), WHITE at the bottom
1. e4 (e2e4) you        1. e4 — best, cost 0.000, depth 18
5. Bc4 (f1c4) you       5. Bc4 — excellent, cost 0.009, best 3. Bb5 Nf6 4. O-O Nxe4
```

Not there yet, and absent rather than faked: the coach (what to say), the voice, the clock,
the profile of the player, the motif tagger and the opening book.

| Ported | From | Held to the Python by |
| --- | --- | --- |
| `chess/` | `python-chess` (the subset the reconciler uses) | `ChessTest`: 685 positions, 20,314 legal moves in the library's own order, every SAN and resulting FEN, perft to depth four |
| `vision/BoardDetect`, `BoardRefine`, `BoardTracker` | `vision/board_detect.py`, `tracker.py` | `BoardDetectTest` |
| `vision/Silhouette`, `Covers`, `Pieces`, `ThemeModel`, `CellEmbedder` | `vision/pieces.py` | `SilhouetteTest` (fractions to four decimals), `ClassifyTest` (0 of 64 cells differ) |
| `vision/Coordinates`, `ResultBadges`, `ThemeCache`, `ChangeGate` | `vision/coordinates.py`, `result_badges.py`, `theme_cache.py`, `replay/recording.py` | through `PipelineTest` |
| `state/Reconcile` | `state/reconcile.py` | `ReconcileTest`: eight games, every frame's events identical |
| `state/Pipeline` | `state/pipeline.py` | `PipelineTest`: three games from the pixels up, every gate decision and every event identical |
| `analysis/Score`, `Threats`, `Annotator`, `Assessing` | `analysis/score.py`, `threats.py`, `annotator.py`, `cli.py`'s assessing stage | `ScoreTest`: 400 curve points, 600 labels, 42 positions, 30 games, exactly |
| `analysis/Engine` | `analysis/engine.py`, speaking UCI directly | `EngineTest`, `AnnotatorTest`: against a running engine, since a search is not a pure function |

### What it costs

Measured on the x86-64 emulator over a thirty-ply game played at a move every 2.5 seconds,
which is four to twelve times faster than a real one:

```
app (capture + vision, ten ticks a second)   31-36% of a core
stockfish (2 threads, depth 18)              134-189% of a core
```

The engine is four to five times the rest of the pipeline, and its number is a cost per
move rather than a rate. An unchanged tick costs three to six milliseconds; a move costs the
eight to thirteen cells that changed. The levers, when they are needed, are the ones the
desktop already has as settings: threads, depth, the time cap, and the hurried analysis the
annotator already falls back to when it is behind.

The phone ticks ten times a second because the desktop does and everything downstream counts
in those ticks -- the tracker's three misses before it drops a lock and its re-fit every ten,
how long the reconciler remembers a covered cell, how long a cell may stay unexplained
before a resync. Reading more slowly stretches all of them.

## Building and testing

The SDK is the one Android Studio installs (`%LOCALAPPDATA%\Android\Sdk`); `local.properties`
points at it and is not committed. JDK 17.

```
gradlew.bat assembleDebug                      # the APK
gradlew.bat installDebug                       # onto the phone over adb
powershell -File tools\test.ps1                # the parity suite, on a device or emulator
powershell -File tools\test.ps1 -Class com.sidecoach.state.PipelineTest
```

The tests are instrumented, not unit tests: they need OpenCV's native library, the ONNX
runtime and the engine, so they run on a device. Twenty-five seconds on the emulator for
twenty-three of them. `tools/test.ps1` rather than Gradle's `connectedDebugAndroidTest`,
which uninstalls the app when it is done and takes the engine's network with it.

The engine is built from source with the NDK, without its 98 MB network; the app fetches the
network on first run, as the desktop fetches the engine:

```
python tools/fetch_stockfish.py         # the source, into third_party/ (not committed)
python tools/fetch_stockfish.py --net   # ... and the network, for a dev push
```

### The fixtures

Every parity test reads what the desktop recorded. The generators live in `tools/` and run
from the desktop repo, which has python-chess and the renderer:

```
cd D:/chess-coaching-companion
uv run --no-sync python D:/sidecoach-android/tools/make_fixtures.py            # boards
uv run --no-sync python D:/sidecoach-android/tools/make_chess_fixtures.py      # positions
uv run --no-sync python D:/sidecoach-android/tools/make_reconcile_fixtures.py  # readings
uv run --no-sync python D:/sidecoach-android/tools/make_pipeline_fixtures.py   # frames
uv run --no-sync python D:/sidecoach-android/tools/make_game.py                # a game to play
```

`make_game.py` renders a game into the debug build's assets; `tools/play_game.ps1` installs
the app, starts the capture, plays that game on the device's own screen and prints what the
watcher made of it. That is the end-to-end check, and the frames it renders are not
committed.

## Layout

```
app/src/main/java/com/sidecoach/
  capture/   MediaProjection, the virtual display, the frame the vision reads
  chess/     the board, the moves, the notation: python-chess's subset
  vision/    the board found, the theme learned, every square named
  state/     the frames folded into a legal game
  analysis/  the engine, what a move cost, what the position holds
  events/    what one stage tells the next
  debug/     a rendered board or game on screen, for testing without a chess app
```
