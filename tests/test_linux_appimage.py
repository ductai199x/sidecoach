"""The sidecar from an AppImage, told to end from outside: a logout's SIGTERM reaches the
AppImage's runtime too, which takes its mount away, and the server's orderly end then ran over
files no longer there -- SIGBUS, and a crash report at the next login (Fedora 44, 2026-10-14).
From an AppImage, SIGTERM ends the sidecar at once; every other signal, and every other
install, keeps the server's own handling."""

from __future__ import annotations

import signal

import pytest

from sidecoach import linux


def test_only_an_appimage_is_known_as_one() -> None:
    assert linux.from_appimage({"APPIMAGE": "/home/a/Sidecoach.AppImage"})
    assert not linux.from_appimage({"APPDIR": "/tmp/.mount_x"})
    assert not linux.from_appimage({})


def test_sigterm_ends_at_once_and_the_rest_goes_to_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exits: list[int] = []
    handled: list[int] = []

    def fake_exit(code: int) -> None:
        exits.append(code)

    monkeypatch.setattr(linux.os, "_exit", fake_exit)
    handle = linux.quick_term(lambda sig, frame: handled.append(sig))
    handle(signal.SIGINT, None)
    assert handled == [signal.SIGINT] and exits == []
    handle(signal.SIGTERM, None)
    assert exits == [0]
