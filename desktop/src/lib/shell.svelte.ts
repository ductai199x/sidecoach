// The desktop shell's native menu bar and tray, built from the page with Tauri's JS API
// when the page runs inside the shell. In a browser this module does nothing: the
// header's own controls stay the way in. The menus drive the same store and API the
// header does, so the shell adds a menu bar and takes nothing away.

import { isTauri, invoke } from "@tauri-apps/api/core";
import { panel } from "./api";
import { prefs, store } from "./panel.svelte";

export type ShellSettings = { close: "quit" | "tray"; autostart: boolean };

export const inShell = (): boolean => isTauri();

export async function shellSettings(): Promise<ShellSettings> {
  if (!isTauri()) return { close: "quit", autostart: false };
  return invoke<ShellSettings>("shell_settings").catch(() => ({ close: "quit", autostart: false }));
}

export async function setShellSettings(s: Partial<ShellSettings>): Promise<void> {
  if (!isTauri()) return;
  await invoke("set_shell_settings", { settings: s });
}

export type PaneId = "board" | "moves" | "strip" | "terminal";

/** What the View menu switches; the page reads these and the choice is remembered. */
export const panes = $state({
  board: prefs.getStr("view.board") !== "0",
  moves: prefs.getStr("view.moves") !== "0",
  strip: prefs.getStr("view.strip") !== "0",
});

export function togglePane(id: "board" | "moves" | "strip"): void {
  panes[id] = !panes[id];
  prefs.setStr("view." + id, panes[id] ? "1" : "0");
}

type MenuHandles = {
  refresh: () => Promise<void>;
};

/** Build the menu bar and the tray. `terminal` reads/toggles the terminal fold, owned by the page. */
export async function buildShell(terminal: { shown: () => boolean; toggle: () => void }): Promise<MenuHandles | null> {
  if (!isTauri()) return null;
  const { Menu, Submenu, MenuItem, CheckMenuItem, PredefinedMenuItem } = await import("@tauri-apps/api/menu");
  const { getCurrentWindow } = await import("@tauri-apps/api/window");
  const win = getCurrentWindow();
  const opener = await import("@tauri-apps/plugin-opener").catch(() => null);

  const item = (text: string, action: () => void, id?: string) => MenuItem.new({ text, action, id });
  const check = (text: string, checked: boolean, action: () => void, id?: string) => CheckMenuItem.new({ text, checked, action, id });
  const sep = () => PredefinedMenuItem.new({ item: "Separator" });

  const startItem = await item("Start watching", () => store.toggleRunning(), "start");
  const file = await Submenu.new({
    text: "File",
    items: [
      startItem,
      await sep(),
      await item("Dump the last frames and the log", () => panel.dump()),
      await item("Settings…", () => openWindow("settings")),
      await sep(),
      await PredefinedMenuItem.new({ item: "Quit", text: "Quit Sidecoach" }),
    ],
  });
  const game = await Submenu.new({
    text: "Game",
    items: [
      await item("I resigned / lost on time", () => panel.end("resigned")),
      await item("Opponent resigned / flagged", () => panel.end("won")),
      await item("Draw agreed", () => panel.end("draw")),
      await sep(),
      await item("Forget this game", () => panel.end("forget")),
      await sep(),
      await item("Re-read the screen", () => panel.analyze()),
      await item("Flip the board", () => panel.flip()),
    ],
  });
  const boardItem = await check("Board", panes.board, () => togglePane("board"), "view-board");
  const movesItem = await check("Move list", panes.moves, () => togglePane("moves"), "view-moves");
  const stripItem = await check("Status strip", panes.strip, () => togglePane("strip"), "view-strip");
  const termItem = await check("Terminal", terminal.shown(), () => terminal.toggle(), "view-terminal");
  const topItem = await check("Always on top", false, async () => { await win.setAlwaysOnTop(!(await win.isAlwaysOnTop())); }, "view-top");
  const view = await Submenu.new({
    text: "View",
    items: [
      boardItem, movesItem, stripItem, termItem,
      await sep(),
      topItem,
      await sep(),
      await item("Reset layout", () => {
        try { for (const k of Object.keys(localStorage)) if (k.startsWith("sidecoach.")) localStorage.removeItem(k); } catch { /* private mode */ }
        location.reload();
      }),
    ],
  });
  const muteItem = await check("Mute", !!store.state.muted, () => store.setMuted(!store.state.muted), "voice-mute");
  const listen: Record<string, Awaited<ReturnType<typeof check>>> = {};
  for (const [mode, text] of [["off", "Not listening"], ["ptt", "Push to talk"], ["vad", "Voice activity"]] as const) {
    listen[mode] = await check(text, store.listenMode === mode, () => { store.listenMode = mode; panel.setListen(mode); }, "listen-" + mode);
  }
  const voice = await Submenu.new({ text: "Voice", items: [muteItem, await sep(), listen.off, listen.ptt, listen.vad] });
  const data = await Submenu.new({
    text: "Data",
    items: [
      await item("Clear caches", () => panel.clearCaches()),
      await sep(),
      await item("Reset profile…", async () => {
        if (window.confirm("Reset the profile? Every game, verdict, puzzle attempt and insight the player model holds is deleted. This cannot be undone.")) await store.resetProfile();
      }),
    ],
  });
  const verboseItem = await check("Verbose terminal", store.filters.verbose, () => (store.filters.verbose = !store.filters.verbose), "debug-verbose");
  // Recording is off unless asked: the frames, the events and the log of this session go
  // to the data folder's recordings/, to show a bug. Starts and stops within a second of a
  // running session; before Start it applies from Start.
  const recordItem = await check("Record this session (to show a bug)", !!store.state.record_wanted, async () => {
    const on = !store.state.record_wanted;
    const r = await panel.setRecord(on);
    store.state.record_wanted = r.on;
  }, "debug-record");
  const debug = await Submenu.new({
    text: "Debug",
    items: [
      recordItem,
      verboseItem,
      await item("Open the panel in the browser", async () => {
        const url = (await import("./api")).apiBase() + "/";
        if (opener) await opener.openUrl(url); else window.open(url);
      }),
      await item("Open the data folder", async () => {
        const path = await invoke<string>("data_dir").catch(() => "");
        if (path && opener) await opener.openPath(path);
      }),
    ],
  });
  const help = await Submenu.new({
    text: "Help",
    items: [
      await item("Set up (models, Stockfish, GPU)…", () => openWindow("setup")),
      await item("Check for updates…", () => checkForUpdates(true)),
      await sep(),
      await item("Sidecoach on GitHub", async () => { if (opener) await opener.openUrl("https://github.com/ductai199x/sidecoach"); }),
      await PredefinedMenuItem.new({ item: { About: { name: "Sidecoach", comments: "Vision-first chess coaching companion" } }, text: "About Sidecoach" }),
    ],
  });
  // Settings is its own entry in the bar (Tai, 2026-09-21): one click, not a submenu.
  const settingsItem = await item("Settings", () => openWindow("settings"), "settings");
  const menu = await Menu.new({ items: [file, game, view, voice, data, settingsItem, debug, help] });
  // The panel's window only: an app-wide menu would sit on the puzzle and settings windows
  // too, which carry nothing of the panel's (Tai, 2026-09-21: those stay in the parent).
  // macOS has no per-window menu; there the app menu is the one there is.
  if (navigator.userAgent.includes("Mac")) await menu.setAsAppMenu();
  else await menu.setAsWindowMenu(win);

  // Keep the check items honest against the page's state.
  const refresh = async () => {
    try {
      await startItem.setText(store.running ? "Stop watching" : "Start watching");
      await boardItem.setChecked(panes.board);
      await movesItem.setChecked(panes.moves);
      await stripItem.setChecked(panes.strip);
      await termItem.setChecked(terminal.shown());
      await muteItem.setChecked(!!store.state.muted);
      await recordItem.setChecked(!!store.state.record_wanted);
      await verboseItem.setChecked(!!store.filters.verbose);
      for (const [mode, it] of Object.entries(listen)) await it.setChecked(store.listenMode === mode);
      await topItem.setChecked(await win.isAlwaysOnTop());
    } catch { /* the menu may be gone while the window closes */ }
  };
  setInterval(refresh, 1000);
  return { refresh };
}

/** The tray icon by the clock: Show, and Quit. The shell decides what closing does. */
export async function buildTray(): Promise<void> {
  if (!isTauri()) return;
  try {
    const { TrayIcon } = await import("@tauri-apps/api/tray");
    const { Menu, MenuItem, PredefinedMenuItem } = await import("@tauri-apps/api/menu");
    const { defaultWindowIcon } = await import("@tauri-apps/api/app");
    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    const win = getCurrentWindow();
    const show = async () => { await win.show(); await win.unminimize(); await win.setFocus(); };
    const menu = await Menu.new({
      items: [
        await MenuItem.new({ text: "Show Sidecoach", action: show }),
        await PredefinedMenuItem.new({ item: "Separator" }),
        await PredefinedMenuItem.new({ item: "Quit", text: "Quit Sidecoach" }),
      ],
    });
    await TrayIcon.new({
      id: "sidecoach",
      menu,
      tooltip: "Sidecoach",
      icon: (await defaultWindowIcon()) ?? undefined,
      showMenuOnLeftClick: false,
      action: (event) => { if (event.type === "Click" && event.button === "Left" && event.buttonState === "Up") show(); },
    });
  } catch (e) {
    console.warn("tray:", e);
  }
}

export async function openWindow(route: string): Promise<void> {
  if (!isTauri()) { window.open("/" + route, "_blank"); return; }
  await invoke("open_window", { route }).catch((e) => console.warn("open window:", e));
}


/** The updater: a signed `latest.json` on the release page. Asked from Help, and once
 * a day on its own; the download and the install happen only after a yes. */
export async function checkForUpdates(interactive: boolean): Promise<void> {
  if (!isTauri()) return;
  try {
    const { check } = await import("@tauri-apps/plugin-updater");
    const { relaunch } = await import("@tauri-apps/plugin-process");
    const update = await check();
    if (!update) {
      if (interactive) window.alert("Sidecoach is up to date.");
      return;
    }
    const notes = update.body ? "\n\n" + update.body.slice(0, 600) : "";
    if (!window.confirm(`Sidecoach ${update.version} is available (you have ${update.currentVersion}). Download and install it now?${notes}`)) return;
    await update.downloadAndInstall();
    if (window.confirm("Installed. Restart Sidecoach now?")) await relaunch();
  } catch (e) {
    if (interactive) window.alert("Could not check for updates: " + String(e));
    else console.warn("updater:", e);
  }
}
