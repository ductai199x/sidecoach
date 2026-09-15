"""Capture backend protocol. Frames are HxWx3 BGR uint8 numpy arrays."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from sidecoach.events import Region


class CaptureBackend(Protocol):
    def monitors(self) -> list[Region]: ...

    def grab(self, region: Region) -> np.ndarray: ...

    def reopen(self) -> None:
        """Rebuild the capture after a failed grab (a locked session, a sleeping monitor)."""
        ...

    def close(self) -> None: ...


def primary_monitor(monitors: list[Region]) -> int:
    """Index of the primary monitor: the one at the desktop origin, else the first."""
    for i, m in enumerate(monitors):
        if m.x == 0 and m.y == 0:
            return i
    return 0
