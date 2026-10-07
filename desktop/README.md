# Sidecoach desktop

The desktop app: a Tauri 2 shell (`src-tauri/`, Rust) around a Svelte 5 page (`src/`,
SvelteKit as a static SPA). The Python pipeline runs as a sidecar process behind its HTTP API;
the shell starts it, hands its port and a bearer token to the page, and stops it on exit. How
it all fits together, where the app keeps its files, and how a release is made:
[../docs/app.md](../docs/app.md).

## Layout

- `src/routes/`: the pages. `/` is the panel; `splash`, `settings`, `setup`, `games`, `stats`,
  `puzzle/[id]`, `review/[key]`, `captions` and `captions-cross` open in windows of their own.
- `src/lib/`: the components and the page's modules. `api.ts` finds the API (the shell's
  sidecar, or the same origin in a browser or on the phone); `shell.svelte.ts` is everything
  only the shell can do (the tray, windows, always on top, the updater).
- `src-tauri/`: the shell (`src/lib.rs`, `src/places.rs`), `tauri.conf.json`, the NSIS hooks
  (`nsis/hooks.nsh`) and the icons.
- `brand/`: the logo and the icon source. `tests/`: node tests of `src/lib`, run by
  `tests/test_page_api.py` in the Python suite.
- `resources/python/`: the embedded interpreter, built by `scripts/build_sidecar.py`
  (gitignored).

There is no menu bar. The page's header and left drawer are the way in, on the desktop as on
the phone; the tray icon (Show, Quit) is built from the page with Tauri's JS API. Always on top
is a toggle in the header. Closing the window quits; keeping the app in the tray is a setting.

## Developing

```
npm install
npm run tauri dev       # the shell against the Vite dev server (port 1420)
npm run check           # svelte-check
```

In development the shell starts the sidecar with `uv run --no-sync sidecoach panel --port 0
--announce --parent-pid <shell> --no-persist --no-record` in the repository root, so the data
folder is the repository's `data/`. `SIDECOACH_SIDECAR_ARGS` adds sidecar arguments (a
`--replay`, a `--config`); the other development variables are listed in
[../docs/app.md](../docs/app.md#development-hooks).

The page also runs without the shell: `uv run sidecoach panel` serves the copy in
`src/sidecoach/ui/web/` at `http://127.0.0.1:8765/`.

## Building

- `npm run build:web` builds the page and copies it into `../src/sidecoach/ui/web/`. That copy
  is committed: the browser flow, the server tests and the phone app use it, and CI fails when
  it differs from a fresh build. Run it after every change to `src/`.
- `npm run build:app:fast` builds the shell's executable without the installer and without
  LTO, to try a change quickly.
- The installer is built from the repository root with
  `uv run --no-sync python scripts/release_desktop.py` (`--install` installs it; it refuses
  while the app is running). It rebuilds the embedded interpreter first, because the
  interpreter holds a copy of the package.

Nothing of a session is written to disk by the app: the sidecar keeps the recent frames in
memory, and they are saved only by the drawer's Debug > Dump or the opt-in Record sessions.
