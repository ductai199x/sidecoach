// The push-to-talk key, as the settings keep it (pynput's names, voice/hotkey.py: ctrl_r, f9, a
// letter, mouse_x1) and as a person reads it ("Right Ctrl"); and the key a press in the page
// names, for Settings' "press the key you want" (Tai, 2026-10-01: no typing "ctrl_r" by hand).

const BY_CODE: Record<string, string> = {
  ControlRight: "ctrl_r", ControlLeft: "ctrl_l", AltRight: "alt_r", AltLeft: "alt_l",
  ShiftRight: "shift_r", ShiftLeft: "shift_l", MetaRight: "cmd_r", MetaLeft: "cmd_l",
  CapsLock: "caps_lock", ScrollLock: "scroll_lock", NumLock: "num_lock", Pause: "pause",
  Insert: "insert", Delete: "delete", Home: "home", End: "end", PageUp: "page_up", PageDown: "page_down",
  Backspace: "backspace", Tab: "tab", Space: "space", Enter: "enter", ContextMenu: "menu", PrintScreen: "print_screen",
  ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
};

const NAMES: Record<string, string> = {
  ctrl_r: "Right Ctrl", ctrl_l: "Left Ctrl", ctrl: "Ctrl", alt_r: "Right Alt", alt_l: "Left Alt", alt: "Alt",
  shift_r: "Right Shift", shift_l: "Left Shift", shift: "Shift", cmd_r: "Right Windows key", cmd_l: "Left Windows key", cmd: "Windows key",
  caps_lock: "Caps Lock", scroll_lock: "Scroll Lock", num_lock: "Num Lock", pause: "Pause",
  insert: "Insert", delete: "Delete", home: "Home", end: "End", page_up: "Page Up", page_down: "Page Down",
  backspace: "Backspace", tab: "Tab", space: "Space", enter: "Enter", menu: "Menu key", print_screen: "Print Screen",
  up: "Up arrow", down: "Down arrow", left: "Left arrow", right: "Right arrow",
  mouse_x1: "Mouse back button", mouse_x2: "Mouse forward button", mouse_middle: "Middle mouse button",
};

/** The key a keyboard press names, or null for one that cannot be the key (Escape cancels). */
export function keyFromEvent(e: KeyboardEvent): string | null {
  if (BY_CODE[e.code]) return BY_CODE[e.code];
  const f = /^F([1-9]|1\d|2[0-4])$/.exec(e.code);
  if (f) return "f" + f[1];
  const letter = /^Key([A-Z])$/.exec(e.code);
  if (letter) return letter[1].toLowerCase();
  const digit = /^Digit(\d)$/.exec(e.code);
  if (digit) return digit[1];
  return e.key.length === 1 && e.key !== " " ? e.key.toLowerCase() : null;
}

/** The mouse button a press names (the side buttons, the wheel), or null for left and right. */
export function keyFromMouse(e: MouseEvent): string | null {
  return e.button === 3 ? "mouse_x1" : e.button === 4 ? "mouse_x2" : e.button === 1 ? "mouse_middle" : null;
}

const LINUX = typeof navigator !== "undefined" && /Linux/.test(navigator.userAgent) && !/Android/.test(navigator.userAgent);

/** The key as a person reads it: "Right Ctrl", "F9", "A". */
export function keyName(key: string | null | undefined): string {
  const k = (key ?? "").trim().toLowerCase();
  if (!k) return "None";
  // Linux names that key Super.
  if (NAMES[k] && LINUX) return NAMES[k].replace("Windows key", "Super key");
  if (NAMES[k]) return NAMES[k];
  if (/^f\d{1,2}$/.test(k)) return k.toUpperCase();
  return k.length === 1 ? k.toUpperCase() : k;
}

/** Whether holding it would type into the game too (a letter, a digit, Space, Enter...). */
export function typesText(key: string | null | undefined): boolean {
  const k = (key ?? "").trim().toLowerCase();
  return k.length === 1 || ["space", "enter", "tab", "backspace", "delete"].includes(k);
}
