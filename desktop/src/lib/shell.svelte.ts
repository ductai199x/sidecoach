// What the page needs from the desktop shell, with Tauri's JS API when it runs inside it:
// the tray, the windows of their own, the updater, and the few things only the shell can do
// (always on top, the data folder, quitting), which the drawer offers. The shell has no
// menu bar: the drawer is the one way in, on the desktop as on the phone (Tai, 2026-09-24:
// "replace the ugly top bar with the collapsible nav bar on the left too"). In a browser
// the shell's own calls do nothing.

import { isTauri, invoke } from "@tauri-apps/api/core";
import { prefs } from "./panel.svelte";

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
  // The status strip is for reporting a problem (frame rate, timings, memory): off until asked for.
  strip: prefs.getStr("view.strip") === "1",
});

export function togglePane(id: "board" | "moves" | "strip"): void {
  panes[id] = !panes[id];
  prefs.setStr("view." + id, panes[id] ? "1" : "0");
}

/** What the View menu shows, for the page's own controls where there is no menu bar: in a
 * browser, and on the phone, whose panel is a web view (Tai, 2026-09-24: "on mobile it's
 * missing the feature where I want to hide or unhide elements on the panel"). */
export const PANES: { id: "board" | "moves" | "strip"; name: string; hint: string }[] = [
  { id: "board", name: "Board", hint: "The position as the coach reads it" },
  { id: "moves", name: "Move list", hint: "The game's moves and their verdicts" },
  { id: "strip", name: "Status strip", hint: "Frame rate, timings and memory, for reporting a problem" },
];

/** The View menu's Reset layout: every remembered size and pane back to its default. */
export function resetLayout(): void {
  try {
    for (const k of Object.keys(localStorage)) if (k.startsWith("sidecoach.")) localStorage.removeItem(k);
  } catch {
    /* private mode */
  }
  location.reload();
}

/** The terminal under the status strip, open or folded: the drawer's Terminal and the strip's
 * own button. It starts folded (Tai, 2026-09-21: the terminal is off by default) and the
 * choice is remembered. */
export const terminal = $state({ open: prefs.getStr("fold") === "0" });

export function toggleTerminal(): void {
  terminal.open = !terminal.open;
  prefs.setStr("fold", terminal.open ? "0" : "1");
}

// What only the shell has, for the drawer's rows (they were the menu bar's).

export async function alwaysOnTop(): Promise<boolean> {
  if (!isTauri()) return false;
  const { getCurrentWindow } = await import("@tauri-apps/api/window");
  return getCurrentWindow().isAlwaysOnTop().catch(() => false);
}

export async function setAlwaysOnTop(on: boolean): Promise<void> {
  if (!isTauri()) return;
  const { getCurrentWindow } = await import("@tauri-apps/api/window");
  await getCurrentWindow().setAlwaysOnTop(on);
}

export async function openInBrowser(): Promise<void> {
  const url = (await import("./api")).apiBase() + "/";
  const opener = isTauri() ? await import("@tauri-apps/plugin-opener").catch(() => null) : null;
  if (opener) await opener.openUrl(url);
  else window.open(url);
}

export async function openDataFolder(): Promise<void> {
  if (!isTauri()) return;
  const path = await invoke<string>("data_dir").catch(() => "");
  const opener = await import("@tauri-apps/plugin-opener").catch(() => null);
  if (path && opener) await opener.openPath(path);
}

/** Quit the app, whatever closing the window is set to do: the shell kills the sidecar on
 * its way out. */
export async function quitApp(): Promise<void> {
  if (!isTauri()) return;
  const { exit } = await import("@tauri-apps/plugin-process");
  await exit(0);
}

export async function appVersion(): Promise<string> {
  if (!isTauri()) return "";
  const { getVersion } = await import("@tauri-apps/api/app");
  return getVersion().catch(() => "");
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

/** Settings, at one setting ("profile.chesscom_user"): its tab open, the field in view and lit
 *  for a moment. */
export function openSetting(field: string): Promise<void> {
  return openWindow(`settings#${field}`);
}

// What one window changed that the others show: "settings" saved, "games" brought in or
// reviewed. Across the shell's windows by its events; in a browser, between its tabs.
type Topic = "settings" | "games";
const CHANNEL = "sidecoach";

/** Tell every other window `topic` changed. */
export async function announce(topic: Topic): Promise<void> {
  if (isTauri()) {
    const { emit } = await import("@tauri-apps/api/event");
    await emit(`${CHANNEL}:${topic}`).catch((e) => console.warn("announce:", e));
    return;
  }
  const channel = new BroadcastChannel(CHANNEL);
  channel.postMessage(topic);
  channel.close();
}

/** `run` whenever another window says `topic` changed; returns the unsubscribe. */
export function onAnnounce(topic: Topic, run: () => void): () => void {
  if (isTauri()) {
    let stop: (() => void) | null = null;
    let gone = false;
    import("@tauri-apps/api/event").then(({ listen }) =>
      listen(`${CHANNEL}:${topic}`, () => run()).then((un) => { if (gone) un(); else stop = un; }),
    );
    return () => { gone = true; stop?.(); };
  }
  const channel = new BroadcastChannel(CHANNEL);
  channel.onmessage = (e) => { if (e.data === topic) run(); };
  return () => channel.close();
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
