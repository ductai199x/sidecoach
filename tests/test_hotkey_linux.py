"""The push-to-talk key on Linux, read from the keyboard devices (evdev): held in any window,
Wayland's included, exactly as on Windows -- an X11 listener heard nothing a Wayland window was
given (Firefox focused, Ubuntu 24.04, 2026-10-13). A fake `evdev.ecodes` stands in for the
kernel's codes; the devices themselves are checked on a desktop (docs/app.md, Linux)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from sidecoach.voice.hotkey import EvdevListener, evdev_names

ROOT = Path(__file__).resolve().parents[1]
EV_KEY = 1
CODES = {"KEY_RIGHTCTRL": 97, "KEY_LEFTCTRL": 29, "KEY_A": 30, "BTN_SIDE": 0x113}


@pytest.mark.parametrize(
    ("key", "names"),
    [
        ("ctrl_r", ("KEY_RIGHTCTRL",)),
        ("ctrl_l", ("KEY_LEFTCTRL",)),
        ("ctrl", ("KEY_LEFTCTRL", "KEY_RIGHTCTRL")),  # either hand
        ("alt_gr", ("KEY_RIGHTALT",)),
        ("f9", ("KEY_F9",)),
        ("F24", ("KEY_F24",)),
        ("f25", ()),
        ("a", ("KEY_A",)),
        ("7", ("KEY_7",)),
        (";", ("KEY_SEMICOLON",)),
        ("print_screen", ("KEY_SYSRQ",)),
        ("menu", ("KEY_COMPOSE",)),
        ("mouse_x1", ("BTN_SIDE",)),
        ("mouse_x2", ("BTN_EXTRA",)),
        ("mouse_middle", ("BTN_MIDDLE",)),
        ("", ()),
        ("hyper_x", ()),
    ],
)
def test_every_key_the_settings_can_name_has_its_kernel_code(
    key: str, names: tuple[str, ...]
) -> None:
    assert evdev_names(key) == names


def test_every_key_the_page_can_record_is_known_to_linux() -> None:
    """Every name Settings' "press the key you want" can store (desktop/src/lib/hotkey.ts)
    holds on Linux too: none falls back to a key that never works."""
    import re

    page = (ROOT / "desktop" / "src" / "lib" / "hotkey.ts").read_text(encoding="utf-8")
    stored = set(re.findall(r':\s*"([a-z0-9_]+)"', page.split("const NAMES")[0]))
    stored |= {"mouse_x1", "mouse_x2", "mouse_middle"}
    assert stored, "no key names read from the page"
    assert [k for k in sorted(stored) if not evdev_names(k)] == []


class _Event:
    def __init__(self, code: int, value: int, type_: int = EV_KEY) -> None:
        self.type, self.code, self.value = type_, code, value


@pytest.fixture
def listener(monkeypatch: pytest.MonkeyPatch) -> tuple[EvdevListener, list[str]]:
    ecodes = types.SimpleNamespace(EV_KEY=EV_KEY, **CODES)
    fake = types.ModuleType("evdev")
    fake.ecodes = ecodes  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "evdev", fake)
    monkeypatch.setitem(sys.modules, "evdev.ecodes", ecodes)  # type: ignore[arg-type]
    said: list[str] = []
    return (
        EvdevListener(("KEY_RIGHTCTRL",), lambda: said.append("down"), lambda: said.append("up")),
        said,
    )


def test_held_is_one_press_and_the_other_hand_is_nothing(
    listener: tuple[EvdevListener, list[str]],
) -> None:
    ptt, said = listener
    right, left = CODES["KEY_RIGHTCTRL"], CODES["KEY_LEFTCTRL"]
    # Left Ctrl (Ctrl+C in the browser) never starts it; a held Right Ctrl repeats (2): once.
    ptt._handle("kbd", [_Event(left, 1), _Event(left, 0)], EV_KEY)
    ptt._handle("kbd", [_Event(right, 1), _Event(right, 2), _Event(right, 2)], EV_KEY)
    ptt._handle("kbd", [_Event(right, 0)], EV_KEY)
    assert said == ["down", "up"]
    # Not a key event at all (a mouse move, a sync) with the same number: nothing.
    ptt._handle("kbd", [_Event(right, 1, type_=2)], EV_KEY)
    assert said == ["down", "up"]


def test_two_keyboards_hold_it_until_the_last_lets_go(
    listener: tuple[EvdevListener, list[str]],
) -> None:
    ptt, said = listener
    right = CODES["KEY_RIGHTCTRL"]
    ptt._handle("laptop", [_Event(right, 1)], EV_KEY)
    ptt._handle("usb", [_Event(right, 1)], EV_KEY)
    ptt._handle("laptop", [_Event(right, 0)], EV_KEY)
    assert said == ["down"]
    ptt._handle("usb", [_Event(right, 0)], EV_KEY)
    assert said == ["down", "up"]


def test_a_keyboard_unplugged_while_held_lets_the_key_go(
    listener: tuple[EvdevListener, list[str]],
) -> None:
    ptt, said = listener
    ptt._handle("usb", [_Event(CODES["KEY_RIGHTCTRL"], 1)], EV_KEY)
    ptt._drop("usb")
    assert said == ["down", "up"]


def test_the_packaged_udev_rule_is_the_one_the_app_installs() -> None:
    """The .deb's rule file and the AppImage's (written by `allow_keyboards`) are one rule."""
    from sidecoach.linux import UDEV_RULE

    packaged = ROOT / "desktop" / "src-tauri" / "linux" / "70-sidecoach-input.rules"
    assert packaged.read_bytes().decode("utf-8") == UDEV_RULE
    assert 'TAG+="uaccess"' in UDEV_RULE and "ID_INPUT_KEYBOARD" in UDEV_RULE


@pytest.mark.parametrize(
    ("readable", "wayland", "expected", "needs"),
    [
        (True, True, "evdev", False),  # allowed: every window, Wayland too
        (True, False, "evdev", False),
        (False, False, "x11", False),  # X11 hears every window itself
        (False, True, "x11", True),  # Wayland, not allowed: this window only, until allowed
    ],
)
def test_linux_reads_the_keyboards_when_it_may_and_says_when_it_may_not(
    monkeypatch: pytest.MonkeyPatch, readable: bool, wayland: bool, expected: str, needs: bool
) -> None:
    from sidecoach import linux
    from sidecoach.voice import hotkey

    class _Thread:
        daemon = True

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    monkeypatch.setattr(hotkey.sys, "platform", "linux")
    monkeypatch.setattr(linux, "keyboards_readable", lambda: readable)
    monkeypatch.setattr(linux, "wayland_session", lambda env=None: wayland)
    monkeypatch.setattr(hotkey, "EvdevListener", type("Evdev", (_Thread,), {}))
    monkeypatch.setattr(hotkey.GlobalPushToTalk, "healthcheck", lambda self: (True, "ok"))
    monkeypatch.setattr(
        hotkey.GlobalPushToTalk,
        "_keyboard_listener",
        lambda self, name: type("X11", (_Thread,), {})(),
    )
    ptt = hotkey.GlobalPushToTalk("ctrl_r", on_press=lambda: None, on_release=lambda: None)
    assert ptt.start()
    assert type(ptt._listener).__name__.lower() == expected
    assert ptt.needs_permission is needs
    assert ("Sidecoach's window" in ptt.detail) is needs
