"""The capture backend: Windows Graphics Capture when available, mss otherwise, by config."""

from __future__ import annotations

import sys

import pytest

from sidecoach.capture.mss_backend import MssCapture
from sidecoach.capture.wgc_backend import WgcCapture, available, make_capture


def test_backends_are_chosen_by_name_and_unknown_names_are_refused() -> None:
    assert isinstance(make_capture("mss"), MssCapture)
    with pytest.raises(ValueError):
        make_capture("gdi++")
    auto = make_capture("auto")
    assert isinstance(auto, WgcCapture if available() else MssCapture)
    assert auto.name in ("wgc", "mss")


@pytest.mark.skipif(sys.platform != "win32" or not available(), reason="WGC is Windows-only")
def test_wgc_grabs_the_primary_monitor_at_its_own_size() -> None:
    from sidecoach.cli import _monitor_region

    cap = make_capture("wgc")
    try:
        monitor, region = _monitor_region(cap, None)
        frame = cap.grab(region)
        again = cap.grab(region)
    finally:
        cap.close()
    assert frame.shape == (region.h, region.w, 3) and frame.dtype.name == "uint8"
    assert again.shape == frame.shape


def test_wgc_is_paced_and_copies_each_frame_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """WGC delivered a frame on every screen update, up to the refresh rate, and each cost
    two copies of the monitor: 19.6 ms a frame at 1440p, most of a core at 144 Hz for a
    pipeline reading ten frames a second (round 2 finding 6). The session asks for frames
    no faster than the fastest capture rate, keeps one plain copy of each, and `grab` drops
    the alpha from the crop it returns. A fake `windows_capture` stands in for the API."""
    import types

    import numpy as np

    from sidecoach.capture import wgc_backend
    from sidecoach.events import Region

    made: dict[str, object] = {}

    class _Frame:
        def __init__(self, buffer: np.ndarray) -> None:
            self.frame_buffer = buffer

    class _Capture:
        def __init__(self, **kwargs: object) -> None:
            made.update(kwargs)
            self.handlers: dict[str, object] = {}

        def event(self, fn):  # noqa: ANN001, ANN201
            self.handlers[fn.__name__] = fn
            return fn

        def start_free_threaded(self) -> object:
            buffer = np.zeros((4, 6, 4), dtype=np.uint8)
            buffer[..., 0], buffer[..., 1], buffer[..., 2], buffer[..., 3] = 10, 20, 30, 255
            self.handlers["on_frame_arrived"](_Frame(buffer), None)  # type: ignore[operator]
            buffer[...] = 0  # WGC reuses its buffer for the next frame
            return types.SimpleNamespace(stop=lambda: None)

    fake = types.ModuleType("windows_capture")
    fake.Frame = _Frame  # type: ignore[attr-defined]
    fake.InternalCaptureControl = object  # type: ignore[attr-defined]
    fake.WindowsCapture = _Capture  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "windows_capture", fake)
    cap = wgc_backend.WgcCapture()
    monkeypatch.setattr(cap, "monitors", lambda: [Region(0, 0, 6, 4)])
    frame = cap.grab(Region(1, 1, 3, 2))
    assert made["minimum_update_interval"] == wgc_backend.MIN_UPDATE_INTERVAL_MS > 0
    assert frame.shape == (2, 3, 3) and frame.flags.c_contiguous
    assert frame[..., 0].max() == 10 and frame[..., 2].min() == 30
