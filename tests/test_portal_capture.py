"""The Wayland capture through the desktop's screen-sharing portal: the parts that need no
desktop -- reading its answers, keeping the user's choice, cutting the shared screen, and not
asking again once refused. The portal and PipeWire themselves are checked on a desktop
(docs/app.md, Linux)."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from sidecoach.capture import portal_backend
from sidecoach.capture.portal_backend import (
    PortalCancelled,
    PortalCapture,
    PortalError,
    read_token,
    request_path,
    stream_of,
    wayland_session,
    write_token,
)
from sidecoach.events import Region


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"XDG_SESSION_TYPE": "wayland"}, True),
        ({"XDG_SESSION_TYPE": "Wayland", "WAYLAND_DISPLAY": "wayland-0"}, True),
        ({"XDG_SESSION_TYPE": "x11", "WAYLAND_DISPLAY": "wayland-0"}, False),
        ({"XDG_SESSION_TYPE": "tty"}, False),
        ({"WAYLAND_DISPLAY": "wayland-0"}, True),
        ({"DISPLAY": ":0"}, False),
        ({}, False),
    ],
)
def test_a_wayland_session_is_known_by_its_type_first(env: dict[str, str], expected: bool) -> None:
    assert wayland_session(env) is expected


def test_a_request_answers_on_the_path_the_portal_names() -> None:
    # The sender's unique name without its colon, dots made underscores.
    assert (
        request_path(":1.234", "sidecoach1_ab")
        == "/org/freedesktop/portal/desktop/request/1_234/sidecoach1_ab"
    )


def test_the_shared_stream_is_read_from_the_start_answer() -> None:
    # jeepney gives variants as (signature, value).
    results = {
        "streams": (
            "a(ua{sv})",
            [(57, {"position": ("(ii)", (1920, 0)), "size": ("(ii)", (2560, 1440))})],
        ),
        "restore_token": ("s", "tok"),
    }
    assert stream_of(results) == (57, (1920, 0))
    assert stream_of({"streams": ("a(ua{sv})", [(9, {})])}) == (9, None)
    with pytest.raises(PortalError, match="no screen"):
        stream_of({"streams": ("a(ua{sv})", [])})


def test_the_users_choice_is_kept_and_replaced(tmp_path: Path) -> None:
    """A token shares once: each Start's new one replaces the last; none clears it."""
    assert read_token(tmp_path) is None
    write_token(tmp_path, "first")
    assert read_token(tmp_path) == "first"
    write_token(tmp_path, "second")
    assert read_token(tmp_path) == "second"
    write_token(tmp_path, None)
    assert read_token(tmp_path) is None
    # Without a data folder nothing is kept, and nothing fails.
    write_token(None, "x")
    assert read_token(None) is None


def test_gstreamers_map_info_has_the_c_layout() -> None:
    """GstMapInfo as gst/gstmemory.h lays it out on a 64-bit desktop: a wrong field order would
    read the frame from the wrong address."""
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        pytest.skip("the layout checked is the 64-bit one")
    info = portal_backend._GstMapInfo
    assert info.data.offset == 16 and info.size.offset == 24 and ctypes.sizeof(info) == 104


class _Stream:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    def latest(self, timeout: float) -> np.ndarray:
        return self.frame

    def close(self) -> None:
        pass


def _capture(frame: np.ndarray, origin: tuple[int, int] = (0, 0)) -> PortalCapture:
    cap = PortalCapture.__new__(PortalCapture)
    cap._state_dir = None
    cap._portal = None
    cap._stream = _Stream(frame)  # type: ignore[assignment]
    cap._origin = origin
    cap._refused = None
    return cap


def test_the_shared_screen_is_cut_where_the_desktop_put_it() -> None:
    frame = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    cap = _capture(frame, origin=(100, 50))
    assert cap.monitors() == [Region(100, 50, 8, 6)]
    cut = cap.grab(Region(102, 51, 4, 3))
    assert np.array_equal(cut, frame[1:4, 2:6])
    # A screen that changed size is said, for the capture loop to reopen.
    with pytest.raises(PortalError, match="not 9x6"):
        cap.grab(Region(100, 50, 9, 6))


def test_once_refused_the_screen_is_not_asked_for_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dialog cancelled at a reopen: that answer stands for the session."""
    cap = _capture(np.zeros((4, 4, 3), dtype=np.uint8))
    asked: list[int] = []

    def ask() -> None:
        asked.append(1)
        raise PortalCancelled("screen sharing was cancelled in the desktop's dialog")

    monkeypatch.setattr(cap, "_open", ask)
    with pytest.raises(PortalCancelled):
        cap.reopen()
    cap.reopen()  # asks nothing now
    assert asked == [1]
    with pytest.raises(PortalCancelled) as raised:
        cap.grab(Region(0, 0, 4, 4))
    assert raised.value.fatal
