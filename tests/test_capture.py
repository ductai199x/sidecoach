"""The capture loop survives failed grabs."""

import numpy as np
import pytest

from sidecoach.capture.base import primary_monitor
from sidecoach.capture.loop import CaptureFailed, capture_loop
from sidecoach.events import Region


class FlakyBackend:
    name = "flaky"

    def __init__(self, failures: int, reopen_failures: int = 0) -> None:
        self.failures = failures
        self.reopen_failures = reopen_failures
        self.reopened = 0
        self.grabs = 0

    def monitors(self) -> list[Region]:
        return [Region(0, 0, 8, 8)]

    def grab(self, region: Region) -> np.ndarray:
        self.grabs += 1
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("screen locked")
        return np.zeros((region.h, region.w, 3), dtype=np.uint8)

    def reopen(self) -> None:
        self.reopened += 1
        if self.reopen_failures > 0:
            self.reopen_failures -= 1
            raise RuntimeError("no display")

    def close(self) -> None:
        pass


def test_capture_loop_retries_failed_grabs_even_when_the_reopen_fails() -> None:
    backend = FlakyBackend(failures=2, reopen_failures=2)
    frames = []
    for frame_id, _ts, frame in capture_loop(
        backend, Region(0, 0, 8, 8), fps=100, retry_delay=0.0, stop=lambda: len(frames) >= 3
    ):
        assert frame.shape == (8, 8, 3)
        frames.append(frame_id)
    assert frames == [0, 1, 2]
    assert backend.reopened == 2 and backend.grabs == 5


def test_capture_loop_gives_up_after_too_many_failures() -> None:
    backend = FlakyBackend(failures=10)
    with pytest.raises(CaptureFailed, match="screen locked"):
        for _ in capture_loop(
            backend, Region(0, 0, 8, 8), fps=100, grab_retries=3, retry_delay=0.0
        ):
            raise AssertionError("no frame should come")
    assert backend.reopened == 3


def test_primary_monitor_is_the_one_at_the_origin() -> None:
    side = Region(-1080, 0, 1080, 1920)
    main = Region(0, 0, 2560, 1440)
    assert primary_monitor([side, main]) == 1
    assert primary_monitor([main, side]) == 0
    assert primary_monitor([Region(100, 0, 800, 600)]) == 0
    assert primary_monitor([]) == 0


def test_capture_loop_rejects_a_zero_rate() -> None:
    with pytest.raises(ValueError, match="fps"):
        next(capture_loop(FlakyBackend(failures=0), Region(0, 0, 8, 8), fps=0))


def test_capture_loop_reads_the_rate_every_frame() -> None:
    # capture.fps applies now: the loop asks for the rate at every frame, not once.
    asked: list[float] = []

    def rate() -> float:
        asked.append(1000.0)
        return 1000.0

    frames = capture_loop(FlakyBackend(failures=0), Region(0, 0, 8, 8), fps=rate)
    for _ in range(5):
        next(frames)
    before = len(asked)
    for _ in range(5):
        next(frames)
    assert len(asked) - before == 5
