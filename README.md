# Sidecoach for Android

The chess coaching companion on the phone: open it once, grant screen capture, switch to
the chess app and play; the coach speaks. Phase 3 of `docs/app-plan.md` in the desktop
repo (`D:\chess-coaching-companion`), which is the source of truth for scope and order.

The desktop app is Python behind a Tauri shell. The phone cannot run that interpreter, so
the pipeline is ported to Kotlin file by file. The rule for the port is one to one: the
Kotlin says what the Python says, in the same order, and is held to it by tests that
compare against the Python's own recorded output rather than against a reading of the code.

## Where it stands

**The phone watches a game and follows it.** Capture, board detection and tracking, the
theme learned from the start position, the recogniser, and the legal game reconciled from
what the squares show. On the emulator, watching its own screen play a rendered game, it
reads every move of the line with the mover and the time taken; a move costs the eight to
thirteen cells that changed and a few tens of milliseconds.

Not there yet, and absent rather than faked: the clock, the engine, the coach, the voice.

| Ported | From | Held to the Python by |
| --- | --- | --- |
| `chess/` | `python-chess` (the subset the reconciler uses) | `ChessTest`: 685 positions, 20,314 legal moves in the library's own order, every SAN and resulting FEN, perft to depth four |
| `vision/BoardDetect`, `BoardRefine`, `BoardTracker` | `vision/board_detect.py`, `tracker.py` | `BoardDetectTest` |
| `vision/Silhouette`, `Covers`, `Pieces`, `ThemeModel`, `CellEmbedder` | `vision/pieces.py` | `SilhouetteTest` (fractions to four decimals), `ClassifyTest` (0 of 64 cells differ) |
| `vision/Coordinates`, `ResultBadges`, `ThemeCache`, `ChangeGate` | `vision/coordinates.py`, `result_badges.py`, `theme_cache.py`, `replay/recording.py` | through `PipelineTest` |
| `state/Reconcile` | `state/reconcile.py` | `ReconcileTest`: eight games, every frame's events identical |
| `state/Pipeline` | `state/pipeline.py` | `PipelineTest`: three games from the pixels up, every gate decision and every event identical |

## Building and testing

The SDK is the one Android Studio installs (`%LOCALAPPDATA%\Android\Sdk`); `local.properties`
points at it and is not committed. JDK 17.

```
gradlew.bat assembleDebug              # the APK
gradlew.bat installDebug               # onto the phone over adb
gradlew.bat connectedDebugAndroidTest  # the parity suite, on a device or emulator
```

The tests are instrumented, not unit tests: they need OpenCV's native library and the ONNX
runtime, so they run on a device. Twenty-two seconds on the emulator.

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
  events/    what one stage tells the next
  debug/     a rendered board or game on screen, for testing without a chess app
```
