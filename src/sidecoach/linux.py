"""What the desktop app needs to know of a Linux desktop: Wayland or X11, and whether the
push-to-talk key can be read in every window.

The push-to-talk key works as on Windows -- held in any window, the game's browser too, the
coach listens; let go, it stops -- by reading the keyboard devices themselves (evdev,
voice/hotkey.py). An X11 key listener hears nothing a Wayland window is given (Firefox focused,
Ubuntu 24.04, 2026-10-13), and the desktop's GlobalShortcuts portal is absent from GNOME 46 and
asks for every binding where it exists. Reading the devices needs one permission: a udev rule
that gives the user at the desktop -- the active local session, nobody else -- read access to
keyboards and mice (logind's `uaccess`, at once, no new login). The .deb installs it; from an
AppImage the user allows it once, with their password (`allow_keyboards`).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

RULE_NAME = "70-sidecoach-input.rules"
# The .deb's copy is the package's; an AppImage's goes where an administrator's rules go.
RULE_PATHS = (Path("/usr/lib/udev/rules.d") / RULE_NAME, Path("/etc/udev/rules.d") / RULE_NAME)
UDEV_RULE = (
    "# Sidecoach: the user at the desktop may read keyboards and mice, for the push-to-talk key\n"
    "# held in any window (the active local session only: logind's uaccess).\n"
    'SUBSYSTEM=="input", KERNEL=="event*", ENV{ID_INPUT_KEYBOARD}=="1", TAG+="uaccess"\n'
    'SUBSYSTEM=="input", KERNEL=="event*", ENV{ID_INPUT_MOUSE}=="1", TAG+="uaccess"\n'
)


def wayland_session(env: Mapping[str, str] | None = None) -> bool:
    """Whether this is a Wayland desktop session.

    XDG_SESSION_TYPE is the session's own word; a Wayland socket alone (WAYLAND_DISPLAY, which
    an X11 session can carry too) counts only when the session type is not given."""
    env = os.environ if env is None else env
    kind = env.get("XDG_SESSION_TYPE", "").strip().lower()
    if kind:
        return kind == "wayland"
    return bool(env.get("WAYLAND_DISPLAY"))


PORTAUDIO_LIB = "libportaudio.so.2"


def bundled_portaudio(executable: str | None = None) -> Path | None:
    """The PortAudio the app carries beside its Python (scripts/build_sidecar.py: lib/), or
    None (a development checkout, the system's)."""
    import sys

    exe = Path(executable or sys.executable)
    path = exe.resolve().parent.parent / "lib" / PORTAUDIO_LIB
    return path if path.is_file() else None


def use_bundled_portaudio() -> bool:
    """Point sounddevice at the app's own PortAudio, before it is first imported: on Linux it
    looks for the system's (ctypes.util.find_library), which stock Ubuntu does not have -- no
    voice and no microphone (the Linux audit, 2026-10-13). The app's, built for ALSA, is the
    one used wherever it is there, so every desktop runs the build that was tested. True when
    it is in use."""
    import ctypes.util
    import sys

    if not sys.platform.startswith("linux"):
        return False
    bundled = bundled_portaudio()
    if bundled is None:
        return False
    original = ctypes.util.find_library
    if getattr(original, "sidecoach_portaudio", None):
        return True

    def find_library(name: str) -> str | None:
        return str(bundled) if name == "portaudio" else original(name)

    find_library.sidecoach_portaudio = True  # type: ignore[attr-defined]
    ctypes.util.find_library = find_library
    return True


def input_devices(folder: Path = Path("/dev/input")) -> list[Path]:
    try:
        return sorted(folder.glob("event*"))
    except OSError:
        return []


def keyboards_readable() -> bool:
    """Whether this user can read at least one keyboard device (asking nothing)."""
    try:
        import evdev  # type: ignore[import-not-found]
        from evdev import ecodes  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - no evdev: not Linux, or a broken install
        return False
    for path in input_devices():
        if not os.access(path, os.R_OK):
            continue
        try:
            device = evdev.InputDevice(str(path))
        except OSError:
            continue
        try:
            if ecodes.KEY_A in device.capabilities().get(ecodes.EV_KEY, []):
                return True
        finally:
            device.close()
    return False


def rule_installed() -> bool:
    return any(path.exists() for path in RULE_PATHS)


HELPER_DIR = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "sidecoach"
HELPER_NAME = "allow-push-to-talk-in-every-window"


def _helper_script() -> str:
    """The commands run as administrator: the rule written where an administrator's rules
    go, and applied to the devices plugged in now."""
    target = RULE_PATHS[1]
    rule = UDEV_RULE.replace("'", "'\\''")
    return (
        "#!/bin/sh\n"
        "# Sidecoach: lets the user at the desktop read the keyboard, for the push-to-talk key\n"
        "# held in any window. Written and run once, then removed.\n"
        "set -e\n"
        "umask 022\n"
        f"mkdir -p '{target.parent}'\n"
        f"printf '%s' '{rule}' > '{target}'\n"
        "udevadm control --reload\n"
        "udevadm trigger --subsystem-match=input --action=change\n"
        "udevadm settle --timeout=5 || true\n"
    )


def allow_keyboards() -> tuple[bool, str]:
    """Install the udev rule (UDEV_RULE) with the desktop's own password dialog (pkexec), and
    apply it to the devices now: the push-to-talk key works in every window at once. Waited
    on: the person types their password. (True, "") when the keyboards can be read after."""
    pkexec = shutil.which("pkexec")
    if pkexec is None:
        return False, "this desktop has no pkexec to ask for the password"
    # The desktop's password dialog names the program it runs as administrator: a helper
    # whose name says what it does, not a line of shell (which it showed whole, 2026-10-13).
    helper = HELPER_DIR / HELPER_NAME
    try:
        HELPER_DIR.mkdir(parents=True, exist_ok=True)
        helper.write_text(_helper_script(), encoding="utf-8", newline="\n")
        helper.chmod(0o700)
    except OSError as exc:
        return False, f"the helper could not be written: {exc}"
    try:
        done = subprocess.run([pkexec, str(helper)], capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    finally:
        with contextlib.suppress(OSError):
            helper.unlink()
    if done.returncode in (126, 127):  # the dialog dismissed, or not authorised
        return False, "not allowed"
    if done.returncode != 0:
        return False, (done.stderr or done.stdout).strip() or f"exit {done.returncode}"
    return (True, "") if keyboards_readable() else (False, "the keyboards are still not readable")
