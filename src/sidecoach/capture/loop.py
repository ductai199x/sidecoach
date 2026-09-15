"""Paced capture loop: grab a region at a fixed rate and yield (frame_id, ts, frame)."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import numpy as np

from sidecoach.capture.base import CaptureBackend
from sidecoach.events import Region


def capture_loop(
    backend: CaptureBackend,
    region: Region,
    fps: int,
    *,
    max_seconds: float | None = None,
    stop: Callable[[], bool] | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    interval = 1.0 / fps
    started = time.monotonic()
    next_due = started
    frame_id = 0
    while True:
        now = time.monotonic()
        if max_seconds is not None and now - started >= max_seconds:
            return
        if stop is not None and stop():
            return
        yield frame_id, now, backend.grab(region)
        frame_id += 1
        next_due += interval
        delay = next_due - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_due = time.monotonic()
