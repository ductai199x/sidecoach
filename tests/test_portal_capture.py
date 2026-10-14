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
from tests.fakes import HANG_S


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
    cap._closed = portal_backend.threading.Event()
    cap._watch = None
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


def test_the_desktop_closing_the_session_is_heard_and_ends_the_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop sharing from GNOME's or KDE's indicator closes the portal's session, and the stream
    says nothing: the last picture stayed and the session ran on it (KDE, 2026-10-14). The
    session's Closed signal is watched while the screen is shared: once heard, every grab is
    final. Closing the capture lets the watch go first (it holds the portal's connection)."""
    from sidecoach.capture.portal_backend import PortalStopped

    told = portal_backend.threading.Event()
    watched: list[str] = []

    class Portal:
        def start(self, token: str | None) -> tuple[int, tuple[int, int], str]:
            return 7, (0, 0), "token"

        def pipewire_fd(self) -> int:
            return 3

        def wait_closed(self, give_up: object) -> bool:
            watched.append("watching")
            told.wait(HANG_S)
            return not give_up.is_set()  # type: ignore[attr-defined]

        def close(self) -> None:
            watched.append("closed")

    monkeypatch.setattr(portal_backend, "_Portal", Portal)
    monkeypatch.setattr(portal_backend, "_Stream", lambda fd, node: _Stream(np.zeros((4, 4, 3))))
    cap = PortalCapture()
    assert cap.grab(Region(0, 0, 4, 4)).shape == (4, 4, 3)
    told.set()  # the desktop's Closed
    assert cap._watch is not None
    cap._watch[1].join(HANG_S)
    with pytest.raises(PortalStopped) as stopped:
        cap.grab(Region(0, 0, 4, 4))
    assert stopped.value.fatal

    # A second capture, closed by the app: the watch is let go before the portal closes.
    class Quiet(Portal):
        def wait_closed(self, give_up: object) -> bool:
            watched.append("watching")
            give_up.wait(HANG_S)  # type: ignore[attr-defined]
            return False

    monkeypatch.setattr(portal_backend, "_Portal", Quiet)
    watched.clear()
    other = PortalCapture()
    watch = other._watch
    assert watch is not None
    other.close()
    assert not watch[1].is_alive() and other._watch is None
    assert watched == ["watching", "closed"] and not other._closed.is_set()


def test_sharing_stopped_from_the_desktop_ends_the_session_without_sharing_again() -> None:
    """The user stops sharing from the desktop's indicator: the stream ends. The capture loop
    took it for a passing failure and reopened the capture with the remembered choice, which
    shared the screen again without a word. It is final now: the loop ends at once, never
    reopening, and the panel says why on its status line, as for a cancelled dialog."""
    from sidecoach.capture.loop import CaptureFailed, capture_loop
    from sidecoach.ui.panel import _ended

    stream = portal_backend._Stream.__new__(portal_backend._Stream)
    stream._first = portal_backend.threading.Event()
    stream._lock = portal_backend.threading.Lock()
    stream._frame = np.zeros((4, 4, 3), dtype=np.uint8)
    stream.failure = "the desktop stopped sharing the screen"  # what the reader saw: EOS
    stream._first.set()
    cap = _capture(np.zeros((4, 4, 3), dtype=np.uint8))
    cap._stream = stream
    reopened: list[int] = []
    cap.reopen = lambda: reopened.append(1)  # type: ignore[method-assign]
    with pytest.raises(CaptureFailed) as failed:
        next(capture_loop(cap, Region(0, 0, 4, 4), fps=10, retry_delay=0.0))
    assert reopened == [] and "1 grabs failed" in str(failed.value)
    assert _ended(failed.value) == "screen sharing stopped from the desktop · Start shares it again"
    assert _ended(PortalCancelled("x")) == "screen sharing cancelled · Start asks again"
    assert _ended(PortalError("x")).startswith("the desktop did not share the screen")
    assert _ended(RuntimeError("a backend that says nothing")) == "stopped"
