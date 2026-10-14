"""The app's own PortAudio on Linux: sounddevice looks only for the system's, which stock Ubuntu
does not have -- no voice and no microphone from the AppImage (the Linux audit, 2026-10-13).
The build puts one beside the embedded Python (scripts/build_sidecar.py), and the package
points sounddevice at it before sounddevice is first imported."""

from __future__ import annotations

import ctypes.util
import sys
from pathlib import Path

import pytest

from sidecoach import linux


def test_the_bundled_library_is_found_beside_the_embedded_python(tmp_path: Path) -> None:
    (tmp_path / "bin").mkdir()
    (tmp_path / "lib").mkdir()
    exe = tmp_path / "bin" / "python3"
    exe.write_bytes(b"")
    assert linux.bundled_portaudio(str(exe)) is None  # a development checkout has none
    (tmp_path / "lib" / linux.PORTAUDIO_LIB).write_bytes(b"\x7fELF")
    assert linux.bundled_portaudio(str(exe)) == (tmp_path / "lib" / linux.PORTAUDIO_LIB).resolve()


def test_sounddevice_is_pointed_at_it_and_nothing_else_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundled = tmp_path / linux.PORTAUDIO_LIB
    bundled.write_bytes(b"\x7fELF")
    asked: list[str] = []

    def system(name: str) -> str | None:
        asked.append(name)
        return f"lib{name}.so.6"

    monkeypatch.setattr(ctypes.util, "find_library", system)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(linux, "bundled_portaudio", lambda executable=None: bundled)
    assert linux.use_bundled_portaudio() is True
    assert linux.use_bundled_portaudio() is True  # once: not wrapped twice
    assert ctypes.util.find_library("portaudio") == str(bundled)
    assert ctypes.util.find_library("c") == "libc.so.6" and asked == ["c"]


def test_windows_and_a_checkout_without_one_are_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ctypes.util.find_library
    monkeypatch.setattr(sys, "platform", "win32")
    assert linux.use_bundled_portaudio() is False
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(linux, "bundled_portaudio", lambda executable=None: None)
    assert linux.use_bundled_portaudio() is False
    assert ctypes.util.find_library is original
