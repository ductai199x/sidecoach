"""A global push-to-talk key: hold it from any window, and the coach listens.

While a game is on, chess.com has the keyboard focus, not the panel, so the panel's own
button and Space bar cannot be the way to talk mid-game. pynput installs a system-wide
listener instead: the configured key (a lone modifier such as Right Ctrl by default, which
types nothing into the game; a function key; a single character; or a mouse side button)
presses and releases straight through to the listener's push-to-talk. Key auto-repeat is
collapsed to one press. Missing pynput, or a key name it does not know, leaves the panel's
own controls as the only ones -- stated, never crashed.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Callable
from importlib.util import find_spec
from typing import Any

log = logging.getLogger(__name__)

MOUSE_PREFIX = "mouse_"

# Windows reports the hand of a modifier in two ways: the sided virtual key (VK_RCONTROL) and,
# whatever the virtual key says, the hardware scan code with its "extended" flag -- the keypad's
# and the right-hand Ctrl and Alt carry it, the left-hand ones do not; Shift's two keys have
# scan codes of their own (0x2A left, 0x36 right).
_LEFT_SHIFT_SCAN = 0x2A
_RIGHT_SHIFT_SCAN = 0x36


_SIDED_VK = {
    "ctrl_l": 0xA2,
    "ctrl_r": 0xA3,
    "shift_l": 0xA0,
    "shift_r": 0xA1,
    "alt_l": 0xA4,
    "alt_r": 0xA5,
    "alt_gr": 0xA5,
}
_GENERIC_VK = {"ctrl": 0x11, "shift": 0x10, "alt": 0x12}
_MOUSE_VK = {"x1": 0x05, "x2": 0x06, "middle": 0x04}


def windows_keys(name: str) -> tuple[int, ...]:
    """The Windows virtual keys the push-to-talk key `name` reads as held by: its own, and for
    a modifier its generic one too (a keyboard or a remapper reporting Right Ctrl as plain Ctrl
    must never read as up while held). Empty for a key this does not know."""
    name = (name or "").strip().lower()
    if name.startswith(MOUSE_PREFIX):
        vk = _MOUSE_VK.get(name.removeprefix(MOUSE_PREFIX))
        return (vk,) if vk is not None else ()
    if name in _SIDED_VK:
        return (_SIDED_VK[name], _GENERIC_VK[name.split("_")[0]])
    if name in _GENERIC_VK:
        return (_GENERIC_VK[name],)
    if name.startswith("f") and name[1:].isdigit() and 1 <= int(name[1:]) <= 24:
        return (0x70 + int(name[1:]) - 1,)
    if len(name) == 1 and name.isalnum():
        return (ord(name.upper()),)
    return ()


def _windows_key_down(vk: int) -> bool:
    import ctypes

    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)  # type: ignore[attr-defined]


def hand_of(generic: str, scan: int, extended: bool) -> str | None:
    """ "l" or "r" for a modifier ("ctrl", "alt" or "shift") from the hardware's scan data, or
    None for a key this does not know."""
    if generic in ("ctrl", "alt"):
        return "r" if extended else "l"
    if generic == "shift":
        if scan == _RIGHT_SHIFT_SCAN:
            return "r"
        if scan == _LEFT_SHIFT_SCAN:
            return "l"
    return None


def matches_key(
    wanted: str, got: str, scan: int | None = None, extended: bool | None = None
) -> bool:
    """Whether a key reported as `got` is the key named `wanted` ("ctrl_r", "ctrl", "f9").

    A modifier's generic name matches its sided keys ("ctrl" for ctrl_l and ctrl_r). And a sided
    name matches the generic key, because some keyboards, drivers and remappers report Right Ctrl
    as plain Ctrl (Tai's, 2026-09-22): the press arrives, just not under the name the setting
    uses. Then the hardware's scan data says whose hand it was, so Left Ctrl, which a plain
    "ctrl" report would equally be, never starts push-to-talk (Tai, 2026-10-01: "why does my
    keyboard report right ctrl as only ctrl?" -- whatever the cause in his stack, the hand is
    read from the scan code). Without scan data (another system), a plain report matches."""
    if got == wanted:
        return True
    if got.startswith(wanted + "_"):
        return True  # "ctrl" wants either hand
    generic, _, side = wanted.partition("_")
    if side not in ("l", "r") or got != generic:
        return False
    if scan is None or extended is None:
        return True
    return hand_of(generic, scan, extended) == side


class GlobalPushToTalk:
    """Calls `on_press` once when the key goes down and `on_release` when it comes up."""

    def __init__(
        self,
        key: str,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
    ) -> None:
        self.key = (key or "").strip().lower()
        self._on_press = on_press
        self._on_release = on_release
        self._listener: Any = None
        self._down = False
        self._lock = threading.Lock()
        self.detail = "off"
        # Whether a virtual key is down now (Windows: GetAsyncKeyState); None elsewhere.
        self.key_down: Callable[[int], bool] | None = _windows_key_down if os.name == "nt" else None

    def healthcheck(self) -> tuple[bool, str]:
        if not self.key:
            return False, "no global push-to-talk key configured"
        if find_spec("pynput") is None:
            return False, "pynput not installed; run `uv sync --group voice`"
        return True, f"hold {self.key} from any window"

    def start(self) -> bool:
        if self._listener is not None:
            return True
        ok, detail = self.healthcheck()
        self.detail = detail
        if not ok:
            return False
        try:
            if self.key.startswith(MOUSE_PREFIX):
                self._listener = self._mouse_listener(self.key.removeprefix(MOUSE_PREFIX))
            else:
                self._listener = self._keyboard_listener(self.key)
            self._listener.daemon = True
            self._listener.start()
            return True
        except Exception as exc:  # noqa: BLE001 - a bad key name or a headless box
            self.detail = f"global key unavailable: {exc}"
            self._listener = None
            return False

    def stop(self) -> None:
        listener, self._listener = self._listener, None
        if listener is not None:
            with contextlib.suppress(Exception):
                listener.stop()
        self._release()

    def let_go_if_up(self) -> bool:
        """Whether the key, taken as held, is in fact up: then it is let go of here (its release
        called) and True. Windows removes a keyboard hook that answered late without a word,
        and the key-up never came: the coach stayed silent, push-to-talk "held", until the
        next press (round 3 audit, 2026-10-02). False when it is down, not taken as held, or
        its state cannot be read."""
        vks = windows_keys(self.key)
        if not self._down or self.key_down is None or not vks:
            return False
        try:
            if any(self.key_down(vk) for vk in vks):
                return False
        except Exception:  # noqa: BLE001 - unreadable: taken as held, as before
            return False
        self._release()
        return True

    def _press(self) -> None:
        with self._lock:
            if self._down:
                return  # key auto-repeat
            self._down = True
        self._on_press()

    def _release(self) -> None:
        with self._lock:
            if not self._down:
                return
            self._down = False
        self._on_release()

    def _keyboard_listener(self, name: str) -> Any:
        from pynput import keyboard  # type: ignore[reportMissingImports]

        wanted = (
            keyboard.Key[name]
            if name in keyboard.Key.__members__
            else keyboard.KeyCode.from_char(name)
        )

        # The hardware's scan data of the key being handled: the hook's filter runs just before
        # the listener's callbacks, on the same thread, for the same event.
        # By the key's virtual code, so a quick second key cannot overwrite it before the
        # listener has handled the first.
        by_vk: dict[int, tuple[int, bool]] = {}

        def note(_msg: int, data: Any) -> bool:
            by_vk[int(getattr(data, "vkCode", 0))] = (
                int(getattr(data, "scanCode", 0)),
                bool(int(getattr(data, "flags", 0)) & 0x01),
            )
            return True  # nothing is suppressed: the key goes on to the focused window

        def same(key: Any) -> bool:
            if key == wanted:
                return True
            got = getattr(key, "name", "") or ""
            if not isinstance(wanted, keyboard.Key):
                return False
            vk = getattr(getattr(key, "value", None), "vk", None)
            scan, extended = by_vk.get(vk, (None, None)) if vk is not None else (None, None)
            return matches_key(name, got, scan, extended)

        # No key but the push-to-talk key is ever reported: the hook sees every key in every
        # window. Naming them for debugging became a keylogger (a live ring, 2026-09-22, held
        # a whole typed message), and named keys alone still traced the typing (the space and
        # backspace of every message, 2026-09-28), so it is gone.
        def pressed(key: Any) -> None:
            if same(key):
                self._press()

        def released(key: Any) -> None:
            if same(key):
                self._release()

        if os.name == "nt":
            return keyboard.Listener(on_press=pressed, on_release=released, win32_event_filter=note)
        return keyboard.Listener(on_press=pressed, on_release=released)

    def _mouse_listener(self, name: str) -> Any:
        from pynput import mouse  # type: ignore[reportMissingImports]

        wanted = mouse.Button[name]

        def on_click(_x: int, _y: int, button: Any, pressed: bool) -> None:
            if button != wanted:
                return
            if pressed:
                self._press()
            else:
                self._release()

        return mouse.Listener(on_click=on_click)
