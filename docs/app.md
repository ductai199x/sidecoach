# Sidecoach as an application

How the desktop app and the phone app are built, where they keep things, and how a release
is made. [DESIGN.md](../DESIGN.md) is the source of truth for the pipeline (capture, vision,
state, analysis, coach, voice); this file covers what wraps it. The phone's app is
`android/` (its [README.md](../android/README.md) and `docs/testing.md`); the desktop's test rules
are in [docs/testing.md](testing.md).

There is one pipeline, written in Python, and two hosts for it:

- **Desktop (Windows):** a Tauri 2 shell in Rust, a Svelte 5 page in the shell's webview, and
  the Python package running as a sidecar process behind its HTTP API.
- **Android:** a native Kotlin app. The pipeline is ported to Kotlin one to one, and the same
  Svelte page runs in a WebView, served by a Kotlin HTTP server that answers the same API.

Both platforms carry one version number and are released together (see
[Building and releasing](#building-and-releasing)).

## The desktop app

### Processes

```
sidecoach.exe (Tauri shell, Rust)
 ├─ WebView2 windows: main, splash, settings, games, stats, puzzle/<id>, review/<key>,
 │                    captions, captions-cross       (the Svelte page, from the shell's origin)
 └─ python.exe -I -m sidecoach panel --port 0 --announce --parent-pid <shell>
                     --no-persist --no-record        (the sidecar: FastAPI on 127.0.0.1:<port>)
      └─ stockfish (the analyser's workers, the coach's engine)
```

The page calls the sidecar cross-origin at `http://127.0.0.1:<port>`; the shell hands it the
port and a bearer token. The shell knows nothing about chess. Everything app-specific is in
the page or the sidecar.

### The shell

`desktop/src-tauri/src/lib.rs` (the sidecar's life, the commands the page calls, the
windows) and `places.rs` (window positions). Plugins: single-instance, opener, updater,
process. The executable is `sidecoach.exe`; the Tauri identifier is `sidecoach`.

**Starting the sidecar.** At launch the shell makes a 32-byte random token, opens the splash
window, and spawns the sidecar on a thread:

- In a build with a bundled interpreter (`resources/python/python.exe`, or
  `bin/python3` on Linux), it runs `python -I -Xpycache_prefix=<data>/pycache -m sidecoach
  panel ...` with `PYTHONHOME` and `PYTHONPATH` removed. `-I` keeps the environment's Python
  variables and the user's site-packages out (an AppImage's launcher exports `PYTHONHOME`).
  The bytecode cache goes under the data folder, never into the install folder.
- Without one (`tauri dev`, a debug build), it runs `uv run --no-sync sidecoach panel ...` in
  the repository root, found by walking up to `pyproject.toml`. The data folder is then the
  repository's `data/`.
- The flags are always `--port 0 --announce --parent-pid <shell pid> --no-persist
  --no-record`. The token goes in the environment (`SIDECOACH_PANEL_TOKEN`), not on the
  command line. `SIDECOACH_DATA_DIR` and `SIDECOACH_CONFIG` are set to the per-user folders
  (below); a `SIDECOACH_CONFIG` already in the environment is kept.
- On Windows the process is created with `CREATE_NO_WINDOW`. Its stdout is piped and drained
  for its whole life; its stderr goes to `<data>/sidecar.log`, and the previous launch's log is
  kept as `sidecar.1.log`.

The sidecar binds a free port and prints one JSON line, `{"port": N, "pid": P}`. The shell
waits up to 120 s for that line and for `GET /api/health` to answer 200, then stores the
port. On failure it stores the reason, which the splash shows with a way to open the window
anyway.

**Stopping it.** On exit the shell sends `POST /api/shutdown` with the token, waits up to
25 s for the process to end (the sidecar closes the session, the last game's review, the
profile and Stockfish), then kills it and removes `<data>/panel.lock`. The same stop runs from
a guard in the app's resource table, because the updater's install exits without
`RunEvent::Exit`. A shell that dies without any of this is covered by the sidecar's parent
watchdog (below). `restart_sidecar` lets the page start a new sidecar after one has died.

**Commands the page calls:** `sidecar_status` and `sidecar_port` (the port, the error, whether
the process has exited), `sidecar_token`, `restart_sidecar`, `splash_reveal` and
`splash_done`, `data_dir`, `shell_settings` and `set_shell_settings`, `open_window(route)`,
`open_captions`.

**Windows.**

- `main` is built hidden (`tauri.conf.json`) and shown when the splash says so.
- The splash (`/splash`) is a 720x480 transparent, undecorated card that polls `/api/health`
  for the warm-up's stages and draws them as White's first rank filling. `splash_reveal` shows
  the main window behind it, so the card fades out over the app; `splash_done` closes it.
  Closing the splash before the main window shows quits, unless closing is set to the tray.
  `SIDECOACH_NO_SPLASH` skips it.
- `open_window` opens a page in a window of its own (`settings`, `setup`, `games`, `stats`,
  `puzzle/<id>`, `review/<key>`), one window per route; a second request brings it to the
  front, and a `#place` in the route is sent to the open window. The command is async on
  purpose: on Windows a window built from a synchronous command comes up white.
- `open_captions` makes the two captions windows (`w-captions`, `w-captions-cross`): always on
  top, undecorated, not focusable, out of the taskbar.
- `places.rs` records each kind of window's size, position and maximised state into
  `window.json` and restores it at the next open; a place on a monitor that is gone is
  centred.

**No menu bar.** The page's own header and left drawer are the way in, on the desktop as on
the phone (Tai, 2026-09-24). The tray icon (Show Sidecoach, Quit Sidecoach; a left click shows
the window) is built from the page with Tauri's JS API (`desktop/src/lib/shell.svelte.ts`).
Always on top is a toggle in the header.

**Closing the main window** quits the app by default; the shell setting `close = "tray"`
hides it to the tray instead (Tai, 2026-09-21). The shell keeps its own settings in
`shell.json`, because it needs them before the sidecar is up.

**One instance.** The single-instance plugin is registered first: a second launch (the
shortcut again, the installer of the installed version) shows the running window instead.

**The updater** (`tauri-plugin-updater`) reads `latest.json` from the GitHub release
(`tauri.conf.json`, `plugins.updater`), verified with the minisign public key there. The page
checks once a day, 20 s after start, and from the drawer's Check for updates; it downloads and
installs only after a yes.

### The page

`desktop/src` is a SvelteKit app built as a static SPA (`adapter-static`, fallback
`index.html`). Routes: `/` (the panel), `splash`, `settings`, `setup`, `games`, `stats`,
`puzzle/[id]`, `review/[key]`, `captions`, `captions-cross`. The same build has three hosts:

| Host | Served from | How it finds the API |
|---|---|---|
| The desktop shell | the bundle (`frontendDist: ../build`), origin `http://tauri.localhost` | `sidecar_status` and `sidecar_token` from the shell; calls go cross-origin with `Authorization: Bearer` |
| A browser (`sidecoach panel`) | the sidecar itself, from `src/sidecoach/ui/web/` | same origin, no token |
| The phone | the app's `LocalServer`, from the asset folder `web/` | same origin; the token is a cookie |

`src/lib/api.ts` holds that choice. `src/lib/shell.svelte.ts` wraps what only the shell can do
(the tray, windows, always on top, the data folder, quit, the updater); in a browser those
calls do nothing. The phone's server marks the document `data-app="phone"`, which the page
reads as `onPhone()`.

What the page does at every size (fixed after the audit of 2026-10-07 and Tai's walks of
2026-10-08/09):

- **Stacked** (the board over the coach), the board's side is at most the main area's height
  less what shares the first screen with it, measured: the rows under the board, the moves'
  line, the ask box, the chips and the coach at its minimum height (`lib/layout.ts`
  `mobileBoard`; the puzzle page shares it). At 720x900, Tai's panel window, the ask box and
  the Hint / Best move / Line chips had sat under the fold. Where the whole width fits (a
  phone at 360 or 412, every side-by-side window) nothing is set; a dragged size is used
  wherever it fits and the saved value is never written.
- **A screen whose first request fails** (a review, a puzzle, Your games, Settings, Setup) says
  so in its own content area, with `api.ts`'s `failureText` and Try again; a review of a game
  not in the library and a puzzle the store does not have say what to open instead, their
  controls disabled. A reload that fails after a first success keeps what is shown.
- **Setup** fits 360 px: its title line wraps under the back arrow and logo.
- **The offline banner** sits 8 px below `--inset-top`, clear of the phone's status bar.
- **Reset layout** (the drawer) forgets only the layout, `LAYOUT_KEYS` in `shell.svelte.ts`
  (the remembered sizes, which panes show, the terminal's fold); Stats' filters, Game Review's
  choices and the update check stay.
- **Your games** says what its filters hide: "N games hidden by your filters." and Clear
  filters under a cut list, the same link after "No game matches." and after an import whose
  new games the filters hide (`lib/gamefilters.ts`). Clear filters puts back the page's
  starting filters and search, on the player's press only.

`npm run build:web` builds the page and copies it into `src/sidecoach/ui/web/`
(`desktop/sync-web.mjs`). That copy is committed: the browser flow and the server tests serve
it, the phone copies it from there, and CI fails when it differs from a fresh build.

### The sidecar

`sidecoach panel` (`src/sidecoach/cli.py`, `ui/server.py`, `ui/panel.py`). The browser flow
uses the same command with its defaults (port 8765, no token, a session folder kept).

- **Port and announce.** `--port 0` binds a free port. `--announce` prints the port line for
  the shell.
- **Lock file.** `<data>/panel.lock` holds the port, the pid and a digest of the token. A
  second announcing sidecar with the same token prints the first one's port (`"attached":
  true`) and exits; one with another token waits for the old owner to exit. A development
  panel on the same data folder leaves the lock alone.
- **Parent watchdog.** With `--parent-pid`, a thread checks every 2 s that the shell is alive;
  when it is gone, the sidecar closes, removes its lock and exits.
- **Requests.** With a token, every `/api/` route except `/api/health` needs `Authorization:
  Bearer <token>` (compared in constant time). Before that, a request is refused when its
  `Host` is not a local address or `localhost` (DNS rebinding), when it comes from a page
  that is neither this server's nor the shell's, or when a browser sends a state change
  without a JSON body. CORS allows exactly the shell's origins: `http://tauri.localhost`,
  `tauri://localhost` and the Vite dev server `http://localhost:1420`.
- **Warm-up.** Everything a session needs loads when the sidecar starts, so Start only
  starts reading the screen (Tai, 2026-09-21). Piece recognition, the vision's first pass and
  Stockfish load side by side; the voice and speech recognition load after them, so they do
  not slow the piece net on the GPU; a local model gets one short request in the background,
  and a paid provider is never called. `/api/health` reports the stages as `warm` (pieces,
  voice, stt, mic, vision, engine, coach, ready). Stop keeps Stockfish and the coach for the
  next Start; only the exit closes them.
- **A session that cannot be built** (the stages, the capture, the profile, in `panel._run`)
  ends as a failure in the loop does: "engine stopped: {exc!r}" in the log, what the build had
  opened closed (`_unbuilt`), the panel back to stopped. It had been left on "running" with
  nothing said.
- **No board-reading model** (the embedding net missing, or its load raising): the status
  line and the coach pane's unread-board note say `panel.NO_BOARD_MODEL`, "The board-reading
  model is missing or could not load: reinstall Sidecoach.", in place of "waiting for a start"
  and "Show me the starting position", which cannot help. The app ships the model; Setup does
  not fetch it.
- **The GPU pack** is activated (`setup/gpu.py`, `activate`) before onnxruntime is first
  imported.
- **Settings.** `GET/PUT /api/settings`. The settings page is generated from the pydantic
  schema in `config.py`: each field's description, its choices, and whether a change applies
  `now`, at the `next` Start, or after a `restart`. In the app the settings file never records
  `data_dir` or `tools_dir`; the shell owns them. API keys never go in the file: they come
  from the environment, `.env`, or the OS credential store through `keyring` (service
  `sidecoach`).
- **Background review.** The sidecar that owns the data folder reviews the library's games
  in the background for Stats (`ui/backlog.py`), below normal priority; only the player's
  Pause stops it. A replay or a second panel beside the app runs none.

### Where everything lives

One folder per kind, named for the app, and nothing written anywhere else (Tai,
2026-09-21). On Windows (checked on the installed 0.1.84):

| What | Where |
|---|---|
| The program: `sidecoach.exe`, `python\` (the embedded interpreter), `uninstall.exe` | `%LOCALAPPDATA%\Sidecoach\` |
| The webview's own storage and cache | `%LOCALAPPDATA%\Sidecoach\EBWebView\` |
| Data | `%LOCALAPPDATA%\Sidecoach\data\` |
| Settings: `sidecoach.toml` (the sidecar's), `shell.json` (closing the window), `window.json` (window places) | `%APPDATA%\Sidecoach\` |
| API keys | the Windows Credential Manager |

Inside `data\`:

| Path | What |
|---|---|
| `models\kokoro\` | the voice: `kokoro-v1.0.onnx`, `voices-v1.0.bin` |
| `models\hf\` | speech recognition (Parakeet TDT 0.6B v3) and the Silero gate, in a Hugging Face cache layout |
| `tools\stockfish\` | Stockfish (the settings' `tools_dir` is forced here in the app) |
| `packs\cuda\<version>\` | the CUDA pack, when installed |
| `openings\`, `puzzles\`, `puzzles.sqlite` | the opening names and the puzzle sample |
| `profile.sqlite` | the player's profile, games and reviews |
| `themes\`, `pieces\` | the learned board looks and piece sets |
| `pycache\` | the interpreter's bytecode cache |
| `sidecar.log`, `sidecar.1.log` | the sidecar's stderr, this launch and the one before |
| `panel.lock` | the running sidecar's port and pid |
| `dumps\<stamp>\` | Debug > Dump only |
| `recordings\<stamp>\` | only when the player turns on Record sessions |
| `perf\`, `shell-trace.log` | development hooks only |

An uninstall or an upgrade keeps the data folder (several GB of downloads and the player's
profile). On Linux the same folders are `~/.local/share/sidecoach/` (data and the webview's
storage) and `~/.config/sidecoach/` (settings), with the keys in the Secret Service; see
[Linux](#linux).

### First-run setup

The installer carries the code and nothing else. What the app needs beyond that is fetched
into the data folder by the sidecar (`src/sidecoach/setup/__init__.py`), from the `/setup`
window with one progress bar, through `/api/setup/status`, `start` and `cancel`:

| Item | Source | Notes |
|---|---|---|
| The voice (Kokoro) | the kokoro-onnx GitHub release | about 350 MB, SHA-256-checked as it comes (`KOKORO_SHA256`; files already on disk are not re-checked) |
| Stockfish | the official release, pinned to `sf_19` | the `universal` build for the platform; unpacked in a staging folder and moved in only when an executable came out |
| The opening names | lichess-org/chess-openings | five TSV files |
| Speech recognition | Hugging Face, Parakeet TDT 0.6B v3 and Silero VAD at pinned commits | file by file, size- and SHA-256-checked, into the hub cache's layout |
| Puzzles (optional) | the Lichess puzzle database, streamed | 200 per theme over 27 themes; written to `sample.csv.part` and renamed whole |
| The CUDA pack (optional) | PyPI | offered only when an eligible GPU is found |

Every download resumes from its `.part`, is verified, and stops at the next chunk on Cancel.
A free-space check runs before each item. Inside the app the Hugging Face hub is set offline
(`HF_HUB_OFFLINE=1`): nothing downloads behind the player's back, only the setup does.

Since the audit of 2026-10-07 (fixed 2026-10-08, both platforms): an item whose presence check
raises (a folder the system refuses to read) is listed as not there with the error in its own
line (`Setup.status`, `run.errors`), and Setup is not ready; it had been left out, and Setup
said everything was in place. Voice mode asked for before Setup had fetched the Silero file
comes as soon as the file is there, without a restart: the gate (`voice/vad.py`) looks again
when its failure was the file missing, and any other failure stands. A download that did not
finish is never taken as done: the voice by its SHA-256, the puzzle sample by its `.part`.

### The CUDA pack

The app ships the CPU build of onnxruntime. GPU acceleration for the voice, speech
recognition and the piece net is a pack fetched on the machine that can use it
(`src/sidecoach/setup/gpu.py`; Tai, 2026-09-21: a new user with nothing must get the GPU on
their own). Nothing is hosted by us:

- `scripts/build_pack.py` writes one manifest per platform into the package
  (`src/sidecoach/setup/packs/cuda-windows-x86_64.json`, `cuda-linux-x86_64.json`). Each names
  the exact PyPI wheel files of onnxruntime-gpu and the NVIDIA CUDA 13 and cuDNN 9 wheels at
  this environment's versions, with URL, size and SHA-256 (8 wheels; 1.31 GB on Windows,
  1.53 GB on Linux). Run it after `uv lock` moves any of them; `--check` reports a difference.
  If the wheels change, raise `PACK_VERSION` in both `build_pack.py` and `gpu.py`, or an
  installed pack of the old version is taken as current.
- **Detection** (`detect`, through NVML): an NVIDIA GPU with a driver from the R580 branch or
  later and at least 4 GB of VRAM is eligible.
- **Install:** the wheels are downloaded from PyPI, verified, and unpacked (their
  `.data/purelib|platlib` merged in) into `<data>/packs/cuda/<version>.staging/`, which is
  renamed to `<version>/` when complete. A staging folder survives a failure or Cancel and
  resumes.
- **Activate:** the pack's `site-packages` goes first on `sys.path` before onnxruntime is
  imported, and its DLLs are preloaded. If CUDA does not come up, the CPU build stays in
  charge and the reason is reported. The base environment is never modified; removing the
  pack is deleting its folder (Settings, `/api/setup/remove_pack`).
- `gpu.mode`: `auto` (default), `cpu` (never detect or offer), `cuda` (a failure is an error).
  Applies after a restart.

### Nothing of a session is recorded

The app never writes frames or session files on its own (Tai, 2026-09-21). The shell always
starts the sidecar with `--no-persist --no-record`: no session folder, the log in memory, and
the recent frames in an in-memory ring (`FrameRing`: 150 frames as half-scale JPEGs). They
reach the disk only when the player asks:

- **Debug > Dump** in the drawer (`POST /api/dump`) writes the ring and the log to
  `<data>/dumps/<stamp>/`.
- **Record sessions** (the drawer's Debug section, or the setting `capture.record`) saves each
  session's frames, events and log to `<data>/recordings/<stamp>/` until it is turned off.

Recording by default is a development feature of the CLI (`sidecoach panel` without
`--no-record`, `sidecoach watch`).

### Development hooks

Environment variables the shell reads, none of them for the product:

| Variable | Effect |
|---|---|
| `SIDECOACH_PYTHON` | an interpreter to run instead of the bundled one (the app's mode from a checkout) |
| `SIDECOACH_DATA_DIR`, `SIDECOACH_CONFIG` | the data folder and settings file, passed to the sidecar |
| `SIDECOACH_SIDECAR_ARGS` | extra sidecar arguments (`--replay <recording>`, `--config <file>`) |
| `SIDECOACH_NO_SPLASH` | show the main window at once |
| `SIDECOACH_OPEN_WINDOW=<route>` | open a second window at start (`settings`, `setup`, `puzzle/<id>`) |
| `SIDECOACH_PAGE_TRACE` | log every asset a window asks for, and the answer, to `<data>/shell-trace.log`, and report the page's boot to the terminal |
| `SIDECOACH_PAGE_PERF=<label>` | the page's meter on, samples to `<data>/perf/<label>.jsonl` |
| `SIDECOACH_PAGE_QUERY` | point the main window at the sidecar-served page instead of the bundle |

The sidecar reads `SIDECOACH_PACK_MANIFEST` (another CUDA pack manifest, a path or URL). A
page's uncaught errors reach the panel's terminal through `POST /api/client_log`, always.

A window that works under `tauri dev` can still fail in the installed build (the bundle's asset
protocol differs from the dev server): open every window route from the installed build
before a release.

## The Android app

`android/` (its own Gradle build): Kotlin, `minSdk` 33, `targetSdk` 37, ABIs arm64-v8a and x86_64.
The test device is Tai's Pixel 10 Pro XL.

### The port

The phone cannot run the Python interpreter, so the pipeline is ported file by file, one to
one: the Kotlin says what the Python says, in the same order (chess rules, vision, state,
analysis, coach, LLM, profile, voice, the panel engine and its API). It is held to that by
parity fixtures, not by reading (see [Parity fixtures](#parity-fixtures)). Where the phone
must differ, the difference is stated in the code.

### The page and the shared page contract

The phone shows the desktop's own page. `MainActivity` starts a `LocalServer` (a small HTTP
server on 127.0.0.1, an ephemeral port remembered between launches so the page's
`localStorage` origin stays the same) and loads it in a WebView. `/api/...` goes to
`PanelApi.kt`, which is `ui/server.py` route for route; anything else is a file from the
asset folder `web/`. HTTP was kept instead of a JavaScript bridge because a WebView's request
interception does not see POST bodies, and so that the page talks to both platforms the same
way.

- **Auth:** a per-launch token, set as a cookie in the WebView's private cookie store; the
  server also takes `Authorization: Bearer`. Links to other hosts open in the browser.
- **The page's browser features** need the WebView wired for them: `alert`/`confirm` are app
  dialogs, file inputs open Android's document picker (`onShowFileChooser`), the system bar
  heights are injected as `--inset-top`/`--inset-bottom`.
- **Settings** are one JSON document in app storage over defaults shipped from `config.py` by
  `tools/sync_settings.py`, with the schema trimmed to what the phone honours. The API key is
  kept apart from it.
- **Back** walks the page's own history (out of the settings, back to the panel) and leaves
  the app only when there is nowhere left to go, every time (`PageBack` in
  `MainActivity.kt`): the callback steps aside for the one press that leaves and is back at
  once. Until 2026-10-08 it stayed off after the first exit, and after a return Back left the
  app from the settings.

**The contract.** The page is one program with two servers. Any change to what a page sends
(a route, a body's shape, a whole document against a partial one, a new field) must be made in
`ui/server.py` and in the phone's `PanelApi` and its helpers (`SettingsApi`, `SetupApi`,
`GamesApi`, `StatsApi`, ...) in the same change, with a phone test that sends the new shape
(Tai, 2026-10-01, after a partial settings save on the phone was laid over the defaults and
reset every other setting). A save that writes a whole document starts from the current
values, never the defaults.

### Capture

Start on the page reaches back to the activity: Android asks for screen-capture consent, and
`CaptureService` (a foreground service of type `mediaProjection|microphone`) takes the
projection and mirrors the display into an `ImageReader`. The person can stop it from the
status-bar chip; the system stops it on the lock screen. Before a session the app names any
installed chess app that can detect screen recording (`RecordingNotice`). Some chess apps
block capture of live games (`FLAG_SECURE`); nothing can read those.

Start builds the session's stages (the net, Stockfish, the coach, the watcher) on the vision
thread. A build that throws is caught (`CaptureService.builtOrSaid`), said in the panel's log
as the desktop says it ("engine stopped: {exc!r}"), and the service stops as after a refused
screen share; uncaught on that thread, it had taken the whole app down. With no board-reading
model, or one whose load failed, the status line says `PanelEngine.NO_BOARD_MODEL`, the
desktop's words, in place of "waiting for a board".

### Engine and voice

- **Stockfish** is built from source with the NDK, without its embedded network, and shipped
  in the APK as `libstockfish.so`: Android 10 and later only let an app execute files from its
  library folder. The source comes from `tools/fetch_stockfish.py` into `third_party/` (not
  committed); the network is a data file the app downloads at first run.
- **espeak-ng** (the phonemizer behind Kokoro) is built from source in the app's CMake, at
  the tag the desktop's `espeakng_loader` uses (1.52.0). `tools/fetch_espeak.py` (run from the
  repo root in its Python environment) puts the source in `third_party/espeak-ng` and the English data subset into
  the committed `app/src/main/assets/espeak-ng-data`. The CMake build stops with an error when
  the source is missing; an APK without `libsidecoach_espeak.so` dies after the splash.
- **The voice** is Kokoro on onnxruntime-android, the graph made by the desktop's
  `tools/kokoro_fast.py` (8-bit decoder convolutions, the last stage in fp16), fetched by the
  phone's setup from Tai's Hugging Face repository. It plays on the **media stream**
  (`USAGE_MEDIA`), so it follows the media volume (Tai, 2026-10-02: as `USAGE_ASSISTANT` it had
  its own volume, at 1 of 15 on his phone).
- **Speech recognition** is Parakeet TDT-CTC 110M int8 where the desktop runs the 0.6B v3
  (Tai, 2026-09-23), with the same Silero gate.
- **Captions** over other apps are an overlay window (`SYSTEM_ALERT_WINDOW`), phone only.

### Data, review and dumps

Everything is in the app's private files (`profile.sqlite`, the openings, the models, the
learned sets); backups are off. First-run downloads run in a `dataSync` foreground service
(`SetupService`) with the same item model as the desktop's setup. The phone reviews a game only
when the player asks (Tai, 2026-09-30: no background review, as chess.com has none); a review
stopped part way resumes. Debug > Dump writes to the app's external files (`dumps/`), where
`adb pull` reaches it. The profile holds the same text as the desktop's: the puzzle rating
(`puzzle_perf`, DESIGN section 5.7) is written through `pyJsonDumps` in `json.dumps`'s exact
form, `{"r": 1637.3, ...}`, Python's separators and float repr in the same key order
(`Puzzles.perfJson`; org.json's compact text had failed the release gate's `ProfileTest`).
Either form is read.

## Building and releasing

### One version

Every platform carries the same version, bumped together even when only one changed (Tai,
2026-09-25):

```
uv run --no-sync python scripts/bump_version.py 0.1.85    # everywhere
uv run --no-sync python scripts/bump_version.py --check   # exit 1 when they differ
```

It writes `pyproject.toml`, `src/sidecoach/__init__.py`, `uv.lock`, `desktop/package.json`,
`desktop/src-tauri/tauri.conf.json`, `Cargo.toml` and `Cargo.lock`, and the phone's
`app/build.gradle.kts` (`versionName`, and `versionCode` as the version read as a number:
0.1.85 is 185). The phone is the `android/` folder (`--android` otherwise). Nothing is
committed. Both release scripts refuse a mismatch.

### The desktop

```
uv run --no-sync python scripts/release_desktop.py --install
```

1. Checks the versions (when `android/` exists).
2. `scripts/build_sidecar.py`: copies uv's managed CPython 3.12 (python-build-standalone) to
   `desktop/resources/python/` (gitignored), installs the lock's requirement set for the
   `panel` and `voice` groups with the CPU onnxruntime in place of onnxruntime-gpu and without
   the NVIDIA wheels, torch or the export tools, installs the package itself, prunes tests,
   bytecode and parts of the standard library, and runs the result's `doctor`. The interpreter
   holds a copy of the package, so it is rebuilt before every installer build
   (`--skip-sidecar` keeps it).
3. `npm run tauri build` in `desktop/`: the page, the shell, and the NSIS installer at
   `desktop/src-tauri/target/release/bundle/nsis/Sidecoach_<version>_x64-setup.exe`. Without
   `TAURI_SIGNING_PRIVATE_KEY` the updater's artifacts are left out
   (`tauri.no-updater.json`).
4. With `--install` (Windows): refuses while `sidecoach.exe` is running, and refuses a version
   that is already installed (its installer would only start the installed app). Runs the
   installer with `/S` and checks the installed exe's version and that it changed.

Logs of each step go to `data/build/<step>-<version>.log`. For a quick try of a shell change,
`npm run build:app:fast` builds the executable without LTO and without the installer.

**The installer** is Tauri's NSIS bundle: per user, no administrator prompt, the WebView2
bootstrapper if needed. Its hook (`desktop/src-tauri/nsis/hooks.nsh`) compares its own file
version with the installed exe's; when they match it starts the installed app and quits, so
the same file serves as "install, or just run". A different version installs over the old
one.

### The phone

```
powershell -File android\tools\install.ps1 -Serial <serial>
```

It builds and installs the `phone` build type (the debug build in all but being debuggable,
signed with the debug key, so it installs over earlier builds and keeps the app's data). It
refuses a version that differs from the desktop's `pyproject.toml`, a device that is absent,
and a device where `CaptureService` is running (a live session; `-Force` overrides). The
serial is always named. `-Voice` also pushes the fast Kokoro graph from the desktop's
`data/models/kokoro`. Before a phone release, check that the APK has
`lib/arm64-v8a/libsidecoach_espeak.so`.

The phone's tests run through `android\tools\test.ps1` (android/README.md). It pushes the
files the tests need (the engine's network, Silero, Parakeet, Kokoro) under the app's external
files, then opens the push folder to the app (`chmod -R a+rwX`): on a fresh emulator the
folders are the shell's (mode 2770, group `ext_data_rw`), the app's listener saw them empty,
and in the 0.1.88 gate 54 device tests skipped and three failed for want of a file (2026-10-09).
A device that had the files from an earlier install never showed it.

### Line endings

LF everywhere in the repo, no CR byte in any committed text file (Tai, 2026-10-08: "Do not have
any windows line ending anywhere in this repo"). `.gitattributes` has `* text=auto eol=lf`, so
every checkout is LF whatever `core.autocrlf` says; `android/.gitattributes` makes the page's
copy and the parity fixtures `text eol=lf` and their png, bin and wav binary, never guessed, and
keeps `espeak-ng-data` byte for byte (`-text`), which `tools/fetch_espeak.py` writes with LF;
ruff formats with `line-ending = "lf"`. Check with
`git grep -I -l $'\r' HEAD` (nothing found is the pass).

### Moving shared things to the phone

Each runs from the repo root with `uv run --no-sync python android/tools/<script>`:

| Script | Copies |
|---|---|
| `sync_web.py` | the built page, `src/sidecoach/ui/web/` to `app/src/main/assets/web/` (after `npm run build:web`) |
| `sync_settings.py` | the settings schema and defaults from `config.py` |
| `sync_brand.py` | the launcher, splash and notification icons from `desktop/brand/logo.svg` |
| `sync_kokoro.py`, `sync_asr.py` | Kokoro's phoneme vocabulary, onnx-asr's mel preprocessor |
| `fetch_espeak.py`, `fetch_stockfish.py` | the native sources into `third_party/` (`fetch_espeak.py` also writes the espeak data asset) |

### Parity fixtures

The phone's port is checked against what the Python itself answered. Each
`android/tools/make_*_fixtures.py` runs in the desktop's environment, calls the
desktop code (often the desktop's own test cases, through `tools/desktop_tests.py`), and writes
the inputs and answers under `app/src/androidTest/assets/` (`chess/`, `vision/`, `pipeline/`,
`profile/`, `voice/`, `panel/`, ...). An instrumented test feeds the Kotlin the same inputs and
compares. Stages that call a model or an engine record a deterministic stub's answers, and
random draws are counted on both sides. Regenerate a stage's fixture whenever the desktop code
it covers changes, in the same change as the Kotlin:

```
uv run --no-sync python android/tools/make_coach_fixtures.py
```

`tools/make_fixtures.py` makes the board fixtures and `tools/make_game.py` a rendered game for
the debug build's end-to-end check. The phone's test rules are its `docs/testing.md`.

### CI

`.github/workflows/desktop.yml`, Windows only, two jobs: the Python checks and the tests
that need no GPU; and the app (the page built and diffed against the committed copy, the
embedded interpreter, the Tauri build, a smoke test of three launch/quit cycles through
`scripts/shell_cycles.ps1`). On a `v*` tag it would also publish the installer and
`latest.json` for the updater. No tag has been pushed, so the updater has never had a release
to find; releases are installed locally with the scripts above.

## Decisions

- **Tauri 2 with a Python sidecar, not Electron and not a frozen binary** (2026-09-21). Both
  shells render with Chromium on Windows; Tauri is lighter, and the pipeline stays the Python
  package as tested. The interpreter is a real python-build-standalone with a normal
  `site-packages`, because onnxruntime's DLL preloading, onnx-asr's cache and kokoro-onnx's
  file paths assume one.
- **CPU by default, the GPU as a pack fetched from PyPI** (Tai, 2026-09-21). One download at
  first run, no runtime choice for the player.
- **One setup exe** that installs or, when that version is installed, just runs (Tai,
  2026-09-21).
- **No code signing** (Tai, 2026-09-21: no Azure account, and no other signing service). The
  installer ships unsigned; a downloaded copy shows SmartScreen's "More info, Run anyway". The
  updater's own minisign signature is separate and stays.
- **The app never records frames** on its own; only Dump and the opt-in Record sessions write
  any (Tai, 2026-09-21).
- **Everything under one Sidecoach folder** per kind (program and data, settings), the
  identifier `sidecoach` rather than a reverse-domain name, and nothing outside them (Tai,
  2026-09-21).
- **Closing the window quits**; the tray is opt-in (Tai, 2026-09-21).
- **No menu bar**: the drawer is the one way in, as on the phone (Tai, 2026-09-24).
- **Everything loads at launch**; Start and Stop only start and stop reading the screen (Tai,
  2026-09-21).
- **No model is a supported mode**: `llm.provider = "none"` is the offline coach, also the
  fallback when the provider cannot be reached.
- **Android is native Kotlin**, not a JavaScript runtime: the app is mostly a background
  service, which a WebView cannot run, and the native parts (onnxruntime-android,
  MediaProjection, the NDK) are first-party. The page is shared; the pipeline is ported.
- **One version for every platform**, released and installed together (Tai, 2026-09-25 and
  2026-09-29).

## Linux

Linux was worked on for one day, 2026-09-21, under WSL2 with WSLg on Tai's machine, and not
since. What was done then: `build_sidecar.py` knows the POSIX layout (`bin/python3`,
`opencv-python-headless`, libpython stripped); the Stockfish chooser takes the Linux release
assets; the shell runs the interpreter isolated; `tauri.conf.json` bundles an AppImage and a
`.deb` (which depends on `libportaudio2`); the CUDA pack has a Linux manifest; and the
AppImage built and ran under WSLg with every warm-up stage on the CPU and, with the pack, on
CUDA. `release_desktop.py` runs on Linux and sets `APPIMAGELAUNCHER_DISABLE=1`; under WSL2 the
Tauri-cached `linuxdeploy-plugin-appimage` had to be replaced by its 1-alpha-20230713-1
release.

0.1.88's AppImage and `.deb` were built the same way (2026-10-13) and are on the GitHub
release as a preview. The only capture backends are WGC and mss (X11): under Xvfb, the
AppImage's own Python grabbed the screen with mss and `sidecoach detect` found a drawn board at
its exact place; under WSLg (Wayland with XWayland) the grab fails with an X11 protocol error.

The 0.1.88 AppImage was then run on Ubuntu 24.04 (GNOME 46, a VMware Workstation VM,
2026-10-13): under Xorg it followed a whole game; under Wayland capture failed ("31 grabs
failed in a row: X11 Protocol Error"), and it would not start without FUSE 2, drew a blank
window, waited on the keyring at its start, took Left Ctrl for push-to-talk, and masked its own
board 37 px high. Tai (2026-10-13): Linux must work on Wayland and X11 alike, the app choosing
for itself, and behave as on Windows, 1:1.

**What Linux does now** (2026-10-14; the code and the tests named):

- **The screen.** A Wayland session (`XDG_SESSION_TYPE`, `sidecoach.linux.wayland_session`)
  is read through the xdg-desktop-portal ScreenCast interface (`capture/portal_backend.py`):
  the desktop's "Share Screen" dialog at the first Start, its choice remembered (the restore
  token, `data/screencast-restore-token`; GNOME's "Remember this selection" is ticked), the
  frames from PipeWire through the system's GStreamer (ctypes: `pipewiresrc`, `videoconvert`,
  `appsink`; the .deb depends on gstreamer1.0-pipewire and -plugins-base, stock on Ubuntu). A
  cancelled dialog ends the session at once (`fatal`, not thirty retries). X11 keeps `mss`.
  The doctor reports the portal without opening the dialog. The screen-rectangle mask is not
  used on the portal's frames (Wayland tells no window where it is); the board's signature
  (`vision/ownboard.py`) keeps the panel's board out, and the tracker now checks it again on
  every changed frame and before taking a lock back in place: a lock taken while GNOME dimmed
  the panel behind the share dialog followed the panel's own board (26.04, 2026-10-14).
- **The push-to-talk key** is read from the keyboard devices (evdev, `voice/hotkey.py`
  `EvdevListener`): held in any window, Wayland's included, Left and Right told apart by key
  code, keyboards plugged in later found. Access is a udev rule (`linux/70-sidecoach-input.rules`,
  logind `uaccess`: the user at the desktop, at once, no new login). The .deb installs it
  (`postinst.sh`); from the AppImage the page offers "Allow in every window" once, which asks for
  the password (`linux.allow_keyboards`, pkexec of a helper named for what it does). Until then
  the X11 listener hears the key in Sidecoach's window. On X11 without the rule, the X11
  listener reads the hand from the keysym (Left Ctrl is plain "ctrl" there); the mouse side
  buttons are `button8`/`button9`.
- **The key for the model** lives in a file only the user may read (`config.keys_file`,
  `~/.config/sidecoach/keys.json`, 0600), as Windows' Credential Manager asks for nothing. A
  key found in the keyring is moved there once: silently if the keyring is open, else with one
  unlock dialog beside the running app (`panel._unlock_key`), never again. The start never waits
  on a keyring.
- **The AppImage** carries AppImage's static type2 runtime (`release_desktop.py`
  `_static_runtime`, pinned and checked): no FUSE 2 needed. Its updater signature is made again
  after the swap. It carries a PortAudio built for ALSA alone (`build_sidecar.py`, pinned
  v19.7.0; Ubuntu ships none), which sounddevice is pointed at (`linux.use_bundled_portaudio`).
- **The shell** turns WebKitGTK's DMA-BUF renderer off and sends its windows through XWayland
  (`GDK_BACKEND=x11,wayland`) unless the user set them, so captions, Keep on top and window
  places work from the .deb as from the AppImage; it starts the sidecar without the AppImage's
  library and plugin paths, in the app's own folder (no stray `~/.env`); it realizes the
  captions' cross before making it click-through (tao unwrapped its missing GDK window and the
  app aborted when captions were turned on). The page takes its board's place from the shell
  (`innerPosition`), exact on X11 and Windows.
- **The AppImage in the menu** (Tai, 2026-10-14): at each start it writes its desktop entry and
  icon into the user's data folder (`lib.rs` `appimage_menu_entry`:
  `~/.local/share/applications/sidecoach-appimage.desktop`, the icon under `icons/hicolor`),
  again only when they changed (the file moved); `TryExec` hides it once the file is gone; not
  when the .deb's entry is installed. The dock shows the knight and the app grid has Sidecoach.
- **Close to the tray without a tray** (plain GNOME, Fedora's): no StatusNotifierWatcher on the
  session bus, so the window is minimized to the dock instead of hidden (`lib.rs` `tray_host`);
  a hidden window could only have come back by starting the app again.
- **Push-to-talk held over silence** hands nothing over (Tai, 2026-10-14, every platform; the
  phone's `Mic.kt` the same): the VAD hears the held frames on the listener thread, and a press
  with no speech in it comes to nothing, as a tap too short does. The recogniser made "yeah" of
  silence and it went to the coach.
- **The build machine is Ubuntu 22.04**: the packages need the build's glibc, and built on 24.04
  they needed 2.39 (no Ubuntu 22.04, Debian 12 or Mint 21). Built on 22.04 (a VM, 2026-10-14)
  the newest any file in the AppImage needs is 2.35 (WebKitGTK's own libraries); the shell 2.34.
- **The AppImage takes libwayland and libdrm from the system** (`release_desktop.py`
  `APPIMAGE_SYSTEM_LIBRARIES`: removed from the AppDir, the image packed again with the packer
  Tauri downloads; linuxdeploy as Tauri runs it takes no exclusions). With 22.04's in it,
  WebKit's page process aborted under Fedora 44's Mesa ("Could not create default EGL display:
  EGL_BAD_PARAMETER") and the window stayed empty.
- **The coach's voice on Linux**: the sweep of espeak's copies read `/proc/*/maps` and failed on
  anonymous memory (a line with no path), so the voice never loaded ("list index out of range");
  fixed in `kokoro._mapped_libraries` (the 2026-10-13 round's own bug, never released).
- **A second launch, or the tray's Show**, brings the window to the front with the X server's
  time (`lib.rs` `raise`): GNOME took GTK's untimed raise for focus stealing and left the window
  behind the browser or minimized, with a "Sidecoach is ready" notice. The tray's Show is the
  shell's own command (`show_main_window`).
- **Sharing stopped from the desktop** (GNOME's or KDE's indicator) ends the session and says
  so on the status line ("screen sharing stopped from the desktop · Start shares it again"):
  the portal's session `Closed` signal is watched while sharing (`PortalCapture`, the stream
  itself says nothing and its last picture stayed), and it is final (`PortalStopped`): the loop
  reopened the capture with the remembered choice, sharing the screen again behind the user's
  back. A cancelled dialog and a refusal say theirs too (`panel._ended`); the log alone said why.
- **Logout with the AppImage running**: SIGTERM reaches the runtime too, which takes its mount
  away; the server's orderly end then hit missing files (SIGBUS, Fedora's crash notice at the
  next login). From an AppImage the sidecar ends at once on SIGTERM (`linux.quick_term`); the
  app's own quit asks over HTTP and is unchanged.

**Checked** in VMs (VMware Workstation, headless), the game on a self-playing page:
- Ubuntu 24.04, GNOME 46, Wayland, the AppImage: no FUSE 2; the window drawn; the keyring locked
  with a key in it: the app up at once, one unlock dialog, the key moved, no dialog at the next
  start with the keyring locked again; "Allow in every window", the password, then Right Ctrl
  held over Firefox recorded 1.5 s and Left Ctrl nothing; the Share dialog, a whole game read
  move for move and coached (`e4 e5 Nf3 d6 Bc4 Bg4 Nc3 g6 Nxe5 Bxd1 Bxf7+ Ke7 Nd5#`), Stop and
  Start with no dialog; captions over the browser; the window closed, no process left.
- Ubuntu 24.04, Xorg, the AppImage: `mss`, a whole game; push-to-talk Right yes, Left no; the
  mask at the panel's board exactly ((80,140) maximized, (860,138) tiled).
- Ubuntu 26.04, GNOME 50, Wayland only, stock (no PortAudio, no FUSE 2), the .deb: installed
  with apt, keyboards readable straight away; Setup's downloads (Stockfish's Linux build
  included); the bundled PortAudio listing PipeWire's `default` and using it; the Share dialog
  with the panel's board on screen: no lock on it; a whole game read and coached; push-to-talk
  Right yes, Left no, with no prompt; closed, no process left. The AppImage starts there too.

Checked again (2026-10-14) with the packages built on 22.04:
- Ubuntu 22.04, GNOME 42, Wayland, the AppImage (no FUSE 2): the window drawn, the knight in
  the dock and the tray; Setup's downloads; the Share dialog, a whole game read and coached, the
  choice remembered (no dialog at the next Start); the voice loaded; "Allow in every window",
  the password, Right Ctrl over Firefox taken (and, nothing said, handed over as nothing), Left
  Ctrl nothing; closed to the tray, a second launch brought the window to the front, focused.
  The .deb: installed with apt (its dependencies all in 22.04), the rule in place, the key in
  every window with no prompt, the window drawn, no AppImage menu entry written.
- Fedora 44, GNOME 50 (no tray), Wayland, the AppImage: the window drawn (after the libwayland
  fix); Setup; the Share dialog, a whole game read and coached; Allow through SELinux, Right yes,
  Left no; closed with the tray setting: minimized, and a second launch restored it, focused;
  Alt+F4 with the default: no process, no mount, no crash report; three stops by SIGTERM: no
  crash report (each one had one before).
- Fedora 44, KDE Plasma 6.7, Wayland, the AppImage: the knight in the taskbar and the system
  tray; KDE's share dialog ("Allow restoring" ticked), whole games read move for move, Start again
  with no dialog; Allow with KDE's own password dialog, Right yes, Left no; the session closed
  from KDE's side: the session ended and said so, the screen not shared again.
- Ubuntu 24.04, Xorg, the AppImage: `mss`, a whole game; with no rule and the device access of a
  fresh boot, X11's own key listening: Right yes, Left no.
- Ubuntu 26.04, GNOME 50, the .deb built on 22.04: installed with apt; the voice loaded; the
  Share dialog, the game followed to its mate and the next game seen; Right yes, Left no.

The Xorg segfaults of 2026-10-13 (GNOME Shell twice, then Sidecoach) were in GTK 3's own
`libgdk-3` at the same offset as a crash of GNOME's own "something went wrong" dialog between
them: a GTK bug, set off by the test tool making and removing virtual input devices within
milliseconds; real devices do not come and go like that, and nothing of Sidecoach's is on the
stack.

KDE's share dialog once answered 2 (failure) to a choice made with Enter ("Pipewire stream is not
ready" in its log); the same choice with Space, and every one after, shared. The status line says
"the desktop did not share the screen · Start tries again". An AppImage stopped by SIGTERM
leaves its runtime's mount point in /tmp (the runtime's own behaviour); nothing reads it.

Not like Windows: Wayland's share dialog at the first Start (the desktop's rule; remembered
after), and the push-to-talk permission from the AppImage (the .deb needs none). CI does not
build Linux, and `release_desktop.py --install` is Windows only.

## Not done

- **Linux:** Linux in CI, and a release of the work under [Linux](#linux).
- **The updater's release:** v0.1.88 is on GitHub (Windows, Android, the Linux preview) without
  the updater's `latest.json`, so the updater still has nothing to find.
- **Start with the system:** `shell.json` has an `autostart` field and the page carries it,
  but the settings page shows no control for it and the shell does nothing with it.
- **The phone's distribution:** it is installed over adb with the debug key; there is no
  signed release build and no store listing.
