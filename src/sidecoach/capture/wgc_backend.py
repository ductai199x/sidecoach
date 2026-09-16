"""Capture through the Windows Graphics Capture API (the one OBS uses), via `windows-capture`.

GDI screen grabs (mss, PIL) read the desktop surface, and on Windows a hardware-accelerated
window -- Chrome with chess.com in it -- may be presented on a multi-plane overlay that GDI
does not see update. The symptom is a board that looks frozen to the pipeline until the
window is moved or resized, which knocks it out of the overlay. Windows Graphics Capture
returns the composed output, overlays included, so the pipeline sees what is on screen.

WGC delivers a frame only when the screen changes; `grab` returns the latest one, so a still
screen simply yields the same frame again, which is what a paced capture loop wants. A
session is opened per monitor, once, and kept until `close`.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any

import numpy as np

from sidecoach.capture.mss_backend import MssCapture
from sidecoach.events import Region

FIRST_FRAME_TIMEOUT_S = 3.0


class _Session:
    """One WGC capture of one monitor, keeping the newest frame."""

    def __init__(self, monitor_index: int) -> None:
        from windows_capture import (  # type: ignore[reportMissingImports]
            Frame,
            InternalCaptureControl,
            WindowsCapture,
        )

        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._arrived = threading.Event()
        self.frames = 0
        self.closed = False
        capture = WindowsCapture(
            cursor_capture=False, draw_border=False, monitor_index=monitor_index
        )

        @capture.event
        def on_frame_arrived(frame: Frame, control: InternalCaptureControl) -> None:
            bgr = frame.convert_to_bgr().frame_buffer
            with self._lock:
                self._frame = np.ascontiguousarray(bgr[:, :, :3]).copy()
                self.frames += 1
            self._arrived.set()

        @capture.event
        def on_closed() -> None:
            self.closed = True

        self._control = capture.start_free_threaded()

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def wait_first(self, timeout: float) -> bool:
        return self._arrived.wait(timeout)

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            self._control.stop()


class WgcCapture:
    """A `CaptureBackend` on Windows Graphics Capture; monitor geometry from the OS."""

    name = "wgc"

    def __init__(self) -> None:
        self._geometry = MssCapture()  # the monitor list; not used to grab pixels
        self._sessions: dict[int, _Session] = {}

    def monitors(self) -> list[Region]:
        return self._geometry.monitors()

    def _session_for(self, region: Region) -> tuple[_Session, Region]:
        """The session of the monitor holding `region` (started once), and that monitor."""
        monitors = self.monitors()
        holding = [
            (i, m)
            for i, m in enumerate(monitors)
            if m.x <= region.x < m.x + m.w and m.y <= region.y < m.y + m.h
        ]
        if not holding:
            raise RuntimeError(f"region {region} lies on no monitor")
        index, mon = holding[0]
        session = self._sessions.get(index)
        if session is None or session.closed:
            # windows-capture counts monitors from 1, in the same order as the OS list.
            session = _Session(index + 1)
            if not session.wait_first(FIRST_FRAME_TIMEOUT_S):
                session.stop()
                raise RuntimeError(f"no frame from monitor {index} in {FIRST_FRAME_TIMEOUT_S}s")
            self._sessions[index] = session
        return session, mon

    def grab(self, region: Region) -> np.ndarray:
        session, mon = self._session_for(region)
        frame = session.latest()
        if frame is None:
            raise RuntimeError("no frame captured yet")
        y0, x0 = region.y - mon.y, region.x - mon.x
        crop = frame[y0 : y0 + region.h, x0 : x0 + region.w]
        if crop.shape[:2] != (region.h, region.w):
            raise RuntimeError(f"captured {frame.shape[1]}x{frame.shape[0]}, wanted {region}")
        return np.ascontiguousarray(crop)

    def reopen(self) -> None:
        self.close()
        time.sleep(0.2)
        self._geometry = MssCapture()

    def close(self) -> None:
        for session in self._sessions.values():
            session.stop()
        self._sessions.clear()
        with contextlib.suppress(Exception):
            self._geometry.close()


def available() -> bool:
    try:
        import windows_capture  # type: ignore[reportMissingImports]  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def make_capture(backend: str = "auto") -> Any:
    """The capture backend: "wgc" (Windows Graphics Capture), "mss" (GDI), or "auto" --
    WGC when it is installed, which on Windows it should be, else mss."""
    if backend == "mss":
        return MssCapture()
    if backend == "wgc":
        return WgcCapture()
    if backend != "auto":
        raise ValueError(f"capture backend must be auto, wgc or mss, got {backend!r}")
    return WgcCapture() if available() else MssCapture()
