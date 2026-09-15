"""Capture backend protocol. Frames are HxWx3 BGR uint8 numpy arrays."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from sidecoach.events import Region


class CaptureBackend(Protocol):
    def monitors(self) -> list[Region]: ...

    def grab(self, region: Region) -> np.ndarray: ...

    def close(self) -> None: ...
