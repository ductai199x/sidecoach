"""Non-blocking single-key input for the terminal watcher.

Windows reads the console directly; elsewhere the terminal is switched to cbreak mode
while a reader is open. Without a console (a pipe, a service) no key is ever reported.
"""

from __future__ import annotations

import importlib
import sys


class KeyReader:
    def __init__(self) -> None:
        self._posix_state: object = None
        self._enabled = sys.stdin is not None and sys.stdin.isatty()

    def __enter__(self) -> KeyReader:
        if self._enabled and sys.platform != "win32":
            try:
                termios = importlib.import_module("termios")
                tty = importlib.import_module("tty")
                fd = sys.stdin.fileno()
                self._posix_state = (fd, termios.tcgetattr(fd))
                tty.setcbreak(fd)
            except Exception:  # noqa: BLE001
                self._enabled = False
        return self

    @property
    def enabled(self) -> bool:
        """Whether keys can be read at all: stdin is a terminal the reader could open."""
        return self._enabled

    def __exit__(self, *exc: object) -> None:
        if self._posix_state is not None:
            termios = importlib.import_module("termios")
            fd, state = self._posix_state  # type: ignore[misc]
            termios.tcsetattr(fd, termios.TCSADRAIN, state)
            self._posix_state = None

    def read(self) -> str | None:
        """The key pressed since the last call, lower-cased, or None."""
        if not self._enabled:
            return None
        if sys.platform == "win32":
            import msvcrt

            if not msvcrt.kbhit():
                return None
            key = msvcrt.getwch()
            return key.lower() if key else None
        import select

        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        return sys.stdin.read(1).lower()
