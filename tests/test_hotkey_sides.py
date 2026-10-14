"""Which hand pressed a modifier, from the hardware's scan data, whatever name the driver gives
the key (Tai, 2026-10-01: "why does my keyboard report right ctrl as only ctrl?" -- the app
reads the hand from the scan code, so Right Ctrl starts push-to-talk and Left Ctrl never does)."""

from __future__ import annotations

import pytest

from sidecoach.voice.hotkey import hand_of, matches_key


@pytest.mark.parametrize(
    ("wanted", "got", "scan", "extended", "expected"),
    [
        # Reported properly: by name, no scan data needed.
        ("ctrl_r", "ctrl_r", None, None, True),
        ("ctrl_r", "ctrl_l", None, None, False),
        ("ctrl", "ctrl_r", None, None, True),  # a generic setting wants either hand
        ("ctrl", "ctrl_l", None, None, True),
        ("f9", "f9", None, None, True),
        ("f9", "f10", None, None, False),
        # Reported as plain Ctrl: the scan data says whose hand.
        ("ctrl_r", "ctrl", 0x1D, True, True),
        ("ctrl_r", "ctrl", 0x1D, False, False),  # the left key, never push-to-talk
        ("ctrl_l", "ctrl", 0x1D, False, True),
        ("ctrl_l", "ctrl", 0x1D, True, False),
        ("alt_r", "alt", 0x38, True, True),
        ("alt_r", "alt", 0x38, False, False),
        ("shift_r", "shift", 0x36, False, True),
        ("shift_r", "shift", 0x2A, False, False),
        # No scan data (another system): a plain report matches, as it did.
        ("ctrl_r", "ctrl", None, None, True),
        # Another key altogether.
        ("ctrl_r", "shift", 0x36, False, False),
        ("ctrl_r", "alt", 0x38, True, False),
    ],
)
def test_the_hand_is_read_from_the_scan_data(
    wanted: str, got: str, scan: int | None, extended: bool | None, expected: bool
) -> None:
    assert matches_key(wanted, got, scan, extended) is expected


@pytest.mark.parametrize(
    ("wanted", "got", "keysym", "expected"),
    [
        # X11 (pynput's xorg backend): Left Ctrl comes as plain "ctrl", Control_L its keysym.
        # Ctrl+C in any window started push-to-talk (Ubuntu 24.04, 2026-10-13).
        ("ctrl_r", "ctrl", 0xFFE3, False),
        ("ctrl_r", "ctrl_r", 0xFFE4, True),
        ("ctrl_l", "ctrl", 0xFFE3, True),
        ("ctrl", "ctrl", 0xFFE3, True),  # a generic setting wants either hand
        ("shift_r", "shift", 0xFFE1, False),
        ("shift_r", "shift_r", 0xFFE2, True),
        ("alt_r", "alt", 0xFFE9, False),
        ("alt_gr", "alt_gr", 0xFE03, True),
        # A keysym that names no hand: a plain report matches, as it did.
        ("ctrl_r", "ctrl", 0x61, True),
    ],
)
def test_on_x11_the_hand_is_read_from_the_keysym(
    wanted: str, got: str, keysym: int, expected: bool
) -> None:
    from sidecoach.voice.hotkey import x11_hand

    assert matches_key(wanted, got, hand=x11_hand(keysym)) is expected


def test_hand_of_knows_the_modifiers_only() -> None:
    assert hand_of("ctrl", 0x1D, True) == "r"
    assert hand_of("ctrl", 0x1D, False) == "l"
    assert hand_of("shift", 0x36, False) == "r"
    assert hand_of("shift", 0x99, False) is None
    assert hand_of("f9", 0x43, False) is None


def test_the_keys_a_held_key_reads_as_down_by() -> None:
    """The push-to-talk key is let go of only when Windows reads it up by every key it may be
    held by: a modifier's generic key too, so a Right Ctrl reported as plain Ctrl never reads as
    up while held (round 3 audit, 2026-10-02)."""
    from sidecoach.voice.hotkey import windows_keys

    assert windows_keys("ctrl_r") == (0xA3, 0x11)
    assert windows_keys("shift_l") == (0xA0, 0x10)
    assert windows_keys("alt_gr") == (0xA5, 0x12)
    assert windows_keys("ctrl") == (0x11,)
    assert windows_keys("f9") == (0x78,)
    assert windows_keys("v") == (0x56,)
    assert windows_keys("mouse_x1") == (0x05,)
    assert windows_keys("menu") == ()  # unknown: never second-guessed
