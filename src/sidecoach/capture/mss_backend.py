"""Capture through `mss`. Simple and portable; swap for dxcam if CPU cost matters."""

from __future__ import annotations

import contextlib

import mss
import numpy as np

from sidecoach.events import Region


class MssCapture:
    name = "mss"

    def __init__(self) -> None:
        self._sct = mss.MSS()

    def monitors(self) -> list[Region]:
        # Index 0 is the virtual screen spanning all monitors; physical monitors follow.
        return [
            Region(m["left"], m["top"], m["width"], m["height"]) for m in self._sct.monitors[1:]
        ]

    def grab(self, region: Region) -> np.ndarray:
        shot = self._sct.grab(
            {"left": region.x, "top": region.y, "width": region.w, "height": region.h}
        )
        bgra = np.asarray(shot)
        return np.ascontiguousarray(bgra[:, :, :3])

    def reopen(self) -> None:
        with contextlib.suppress(Exception):
            self._sct.close()
        self._sct = mss.MSS()

    def close(self) -> None:
        self._sct.close()
