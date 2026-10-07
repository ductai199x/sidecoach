# Sidecoach for Android

The chess coaching companion on the phone: open it once, grant screen capture, switch to
the chess app and play; the coach speaks. It lives in `android/` of the Sidecoach repo; the
repo's `DESIGN.md` is the source of truth for scope and decisions, and `docs/app.md` describes
the phone app.

The desktop is Python behind a Tauri shell. The phone cannot run that interpreter, so the
pipeline is ported to Kotlin file by file. The rule for the port is one to one: the Kotlin
says what the Python says, in the same order, and is held to it by tests that compare
against the Python's own recorded output rather than against a reading of the code. Every
release carries the desktop's version number, bumped together by the desktop's
`scripts/bump_version.py`.

## What runs on the phone

The whole pipeline, as on the desktop: capture (MediaProjection in a foreground service),
the board found and tracked, the pieces learned from the start position, the clocks, the
legal game reconciled from what the squares show, Stockfish built from source with the NDK,
the motif tagger and the opening book, the coach (templates offline, or the AI coach over
the same OpenAI-compatible providers), the voice (Kokoro, with espeak-ng built from source
for the phonemes) and speech recognition (Parakeet TDT-CTC 110M), the player profile, Game
Review, Stats and puzzles.

The screens are the desktop's own page (`sidecoach/ui/web`, copied into `assets/web` by
`tools/sync_web.py`), served from the phone by `ui/LocalServer` on the loopback address and
shown in a WebView; the settings page reads the desktop's schema (`tools/sync_settings.py`),
trimmed to what the phone honours.

The phone's engine defaults differ from the desktop's: two workers of one thread each,
64 MB of hash, depth 14 (the desktop: three workers, 256 MB, depth 18).

chess.com's Android app blocks capture of live games (`FLAG_SECURE`); nothing can read
those.

## Building and installing

The SDK is the one Android Studio installs (`%LOCALAPPDATA%\Android\Sdk`); `local.properties`
points at it and is not committed. JDK 17 or newer (the code targets Java 17); CMake 3.22.1
and the NDK from the SDK.

The engine and the voice's phonemizer are built from source, which is fetched into
`third_party/` (not committed); the build fails without either:

```
python tools/fetch_stockfish.py         # Stockfish's source
python tools/fetch_stockfish.py --net   # ... and its network, for a dev push (tools\push_net.ps1)
cd ..   # the repo root, whose environment has espeakng_loader
uv run --no-sync python android/tools/fetch_espeak.py   # espeak-ng's source and data
```

The engine ships without its 98 MB network; the app fetches the network, the voice, speech
recognition, the opening tables and (when asked) a puzzle sample on first run, as the
desktop's setup does.

A release is installed with:

```
powershell -ExecutionPolicy Bypass -File tools\install.ps1 -Serial <serial>
```

It builds the `phone` build type and installs it on that one device (the serial is
required), then prints what landed. It refuses a version that differs from the desktop's, a
device that is absent or offline, and one where a coaching session is live (`-Force`
overrides that). `-Voice` also pushes the faster Kokoro the desktop's `tools/kokoro_fast.py`
makes.

The `phone` build type is the debug build in all but being debuggable, signed with the same
debug key, so it installs over a debug build and keeps the app's files and settings; the
tests run it too (`testBuildType`). The debug build (`installDebug`) is for
`tools/play_game.ps1`'s rendered game and for `run-as`, which only a debuggable app allows.

Every script here that builds or talks to a device dot-sources `tools/Session.ps1`, which
reconnects a network device first and stops Gradle's daemon and adb's server in a `finally`.

adb from Git Bash: a device path (`/sdcard/...`, `/data/...`) is rewritten into a Git path
before adb sees it. Set `MSYS_NO_PATHCONV=1` on the command, or use PowerShell.

## Testing

How to write a test (the budget, logic never timing, where a test goes): `docs/testing.md`.

```
powershell -File tools\test.ps1 -Class com.sidecoach.state.PipelineTest   # one device class
gradlew.bat :app:testPhoneUnitTest --tests com.sidecoach.coach.CoachTest  # one host class
powershell -File tools\test.ps1                # the whole suite, device and host: on purpose only
powershell -File tools\test.ps1 -OneAtATime    # the device's classes one after another
python tools/lane_seconds.py                   # the lanes' cost table, after a -OneAtATime run
```

`tools\test.ps1` runs on `emulator-5556` unless `-Serial` says otherwise. Without `-Class` it
is the whole suite, which the repo's agent guard asks to confirm with
`FULL_SUITE_ASKED=1`.

The suite has two halves, and `tools/test.ps1` runs both at once; it exits 0 only if every
test of both passed.

- **The device's** (`app/src/androidTest`): the tests that need OpenCV's native library, the
  ONNX runtime, the engine, SQLite, the screen or the app's own state. `tools/test.ps1` rather
  than Gradle's `connectedPhoneAndroidTest`, which uninstalls the app when it is done and takes
  the engine's network and the voice's models with it. It installs only what changed, and
  pushes the model files the tests load once (through the app's external files:
  `com.sidecoach.FilesIn` moves them in). The runner (`androidx.test.runner.SidecoachRunner`)
  runs the classes in lanes side by side (`com.sidecoach.Lanes`): the app lane, one class at a
  time, for whatever touches the app's shared state (a class that names no lane runs there);
  the model classes and the free ones in pools, longest first; a class that holds 85 MB of the
  heap or more waits for a heap slot, since every class shares the one process's 192 MB. The
  classes that churn through the heap run in processes of their own (`@InProcess`): the
  `apart` build types are the phone build under other package names (`com.sidecoach.apartN`,
  never installed on the phone), each instrumented by its own runner
  (`SidecoachRunnerApartN`). Each test's process, lane, seconds, CPU, peak memory, heap
  allocated, collections and result land in `app/build/test-timings.tsv`; the run prints every
  skipped test by name.
- **The host's** (`app/src/test`, Gradle's `testPhoneUnitTest`): the pure-logic tests, on the
  JVM, with Android's own `org.json` (`:libcore-json`) and the same fixtures
  (`com.sidecoach.TestAssets`). A call into Android fails there; it never passes by accident.
  Helpers both halves use live in `app/src/sharedTest`.

### The fixtures

Every parity test reads what the desktop recorded. The generators live in `tools/` and run
from the repo root, whose environment has python-chess, the renderer and the code being ported; they
write under `app/src/androidTest/assets/`. There is one per ported module
(`make_chess_fixtures.py`, `make_reconcile_fixtures.py`, `make_pipeline_fixtures.py`,
`make_coach_fixtures.py`, `make_voice_fixtures.py`, ...; `make_fixtures.py` renders the
boards); after a change on the desktop, run the ones for the modules it touched:

```
cd ..   # the repo root
uv run --no-sync python android/tools/make_<module>_fixtures.py
```

The `sync_*.py` scripts copy what the app ships from the desktop the same way: the page
(`sync_web.py`), the settings schema (`sync_settings.py`), Kokoro's vocabulary
(`sync_kokoro.py`), the recogniser's preprocessor (`sync_asr.py`), the logo (`sync_brand.py`).

`make_game.py` renders a game into the debug build's assets (not committed);
`tools/play_game.ps1` installs the app, starts the capture, plays that game on the device's
own screen and prints what the watcher made of it. That is the end-to-end check.

## Layout

```
app/src/main/java/com/sidecoach/
  capture/   MediaProjection, the foreground service, the frame the vision reads
  chess/     the board, the moves, the notation, PGN: python-chess's subset
  vision/    the board found, the theme learned, every square named, the clocks
  state/     the frames folded into a legal game
  analysis/  the engine, what a move cost, motifs, openings, Game Review
  coach/     what to say, when, from which facts, checked before it is shown
  llm/       the model providers, offline templates included
  voice/     Kokoro, espeak-ng, Parakeet, the microphone
  profile/   the player model, the games library, Stats, puzzles
  setup/     the first-run downloads
  ui/        the local server behind the desktop's page
  events/    what one stage tells the next
app/src/main/cpp/   Stockfish's and espeak-ng's native builds
app/src/debug/      a rendered board or game on screen, for testing without a chess app
```
