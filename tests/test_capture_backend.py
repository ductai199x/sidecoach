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
