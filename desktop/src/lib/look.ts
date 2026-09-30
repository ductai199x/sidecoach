// How the boards look, in every window: the pieces and the squares' colours the settings chose
// (looks.py, /api/look), set as the stylesheet's variables (panel.css --pc-*, --sq-*). Read
// when a window opens, again when Settings saves (the shell's "settings" event) and when a
// window comes back to the front. A player's own pieces arrive inline (data URLs). Until the
// sidecar answers, and wherever it cannot, the stylesheet's defaults stand.
import { connect, get } from "$lib/api";
import { onAnnounce } from "$lib/shell.svelte";

export type BoardColours = {
  light: string; light_odd: string; dark: string; dark_odd: string;
  light_hl: string; light_hl_odd: string; dark_hl: string; dark_hl_odd: string;
};
export type Look = { pieces: string; theme: string; images: Record<string, string>; board: BoardColours };

const VARS: [keyof BoardColours, string][] = [
  ["light", "--sq-light"], ["light_odd", "--sq-light-odd"], ["dark", "--sq-dark"], ["dark_odd", "--sq-dark-odd"],
  ["light_hl", "--sq-light-hl"], ["light_hl_odd", "--sq-light-hl-odd"], ["dark_hl", "--sq-dark-hl"], ["dark_hl_odd", "--sq-dark-hl-odd"],
];

/** The look set on the page: a piece's image as a CSS url(), the squares' colours. */
export function applyLook(look: Look): void {
  const root = document.documentElement.style;
  for (const [code, src] of Object.entries(look.images)) root.setProperty(`--pc-${code}`, `url("${src}")`);
  for (const [key, name] of VARS) if (look.board[key]) root.setProperty(name, look.board[key]);
}

let reading: Promise<void> | null = null;
async function read(): Promise<void> {
  try {
    await connect();
    const look = await get<Look>("/api/look");
    if (look?.images) applyLook(look);
  } catch {
    /* the sidecar not up yet, or an older one: the defaults stand */
  }
}
export function refreshLook(): Promise<void> {
  reading ??= read().finally(() => (reading = null));
  return reading;
}

/** Follow the settings from now on; returns the unsubscribe. */
export function followLook(): () => void {
  void refreshLook();
  const stop = onAnnounce("settings", () => void refreshLook());
  const onFocus = () => void refreshLook();
  window.addEventListener("focus", onFocus);
  return () => {
    stop();
    window.removeEventListener("focus", onFocus);
  };
}
