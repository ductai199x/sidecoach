# Sidecoach desktop

The desktop shell: a Tauri 2 window, menu bar and tray around the Python pipeline, which
runs as a sidecar behind its HTTP API. The plan, the decisions and the measurements are
in `../docs/app-plan.md`.

- `npm install` once; `npm run tauri dev` opens the window. In development the shell
  starts the sidecar with `uv run --no-sync sidecoach panel --port 0 --announce
  --parent-pid <shell> --no-persist --no-record` from the repository root and points the
  window at the port it announces. `SIDECOACH_SIDECAR_ARGS` adds arguments (a `--replay`,
  a `--config`), `SIDECOACH_PAGE_QUERY` a query string for the page (`?perf=label` turns
  the page's own meter on).
- `src/` is the SvelteKit frontend (SPA, adapter-static). In Phase 0 it is only the
  starting page; the panel is still served by the sidecar. `src-tauri/` is the shell.
- The menu bar and the tray are built from the page (`sidecoach/ui/web/shell.js`) with
  Tauri's JS API; in a browser the same page shows its header instead.
- Nothing of a session is written to disk by the app: the sidecar runs with `--no-persist
  --no-record`, keeps the recent frames in memory, and writes them only on File > Dump.
