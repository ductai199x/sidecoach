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
import threading
from collections.abc import Callable
from importlib.util import find_spec
from typing import Any

log = logging.getLogger(__name__)

MOUSE_PREFIX = "mouse_"


class GlobalPushToTalk:
    """Calls `on_press` once when the key goes down and `on_release` when it comes up."""

    def __init__(
        self, key: str, on_press: Callable[[], None], on_release: Callable[[], None]
    ) -> None:
        self.key = (key or "").strip().lower()
        self._on_press = on_press
        self._on_release = on_release
        self._listener: Any = None
        self._down = False
        self._lock = threading.Lock()
        self.detail = "off"

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

        def same(key: Any) -> bool:
            if key == wanted:
                return True
            # A modifier's generic name matches its sided key: "ctrl" for ctrl_l / ctrl_r.
            return (
                getattr(key, "name", "").startswith(name + "_")
                if isinstance(wanted, keyboard.Key)
                else False
            )

        return keyboard.Listener(
            on_press=lambda key: self._press() if same(key) else None,
            on_release=lambda key: self._release() if same(key) else None,
        )

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
