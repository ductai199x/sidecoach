"""Paced capture loop: grab a region at a fixed rate and yield (frame_id, ts, frame)."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator

import numpy as np

from sidecoach.capture.base import CaptureBackend
from sidecoach.events import Region


class CaptureFailed(RuntimeError):
    """The backend kept failing after every retry; the session cannot see the screen."""


def capture_loop(
    backend: CaptureBackend,
    region: Region,
    fps: int | float | Callable[[], float],
    *,
    max_seconds: float | None = None,
    stop: Callable[[], bool] | None = None,
    grab_retries: int = 30,
    retry_delay: float = 0.5,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Frames at a fixed rate; a grab that fails is retried, not fatal.

    A locked session, a sleeping monitor, or a display change makes the backend raise.
    The capture is reopened and tried again every `retry_delay` seconds, up to
    `grab_retries` times in a row before the error is raised; a session survives a screen
    lock of a quarter minute and simply misses the frames.
    """
    rate: Callable[[], float] = fps if callable(fps) else (lambda: float(fps))  # type: ignore[assignment]
    if rate() <= 0:
        raise ValueError(f"fps must be positive, got {rate()}")
    interval = 1.0 / rate()
    started = time.monotonic()
    next_due = started
    frame_id = 0
    failures = 0
    while True:
        now = time.monotonic()
        if max_seconds is not None and now - started >= max_seconds:
            return
        if stop is not None and stop():
            return
        try:
            frame = backend.grab(region)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            if failures > grab_retries:
                raise CaptureFailed(f"{failures} grabs failed in a row: {exc}") from exc
            with contextlib.suppress(Exception):  # the next grab decides if the capture is back
                backend.reopen()
            time.sleep(retry_delay)
            next_due = time.monotonic()
            continue
        failures = 0
        yield frame_id, now, frame
        frame_id += 1
        next_due += interval
        delay = next_due - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_due = time.monotonic()
