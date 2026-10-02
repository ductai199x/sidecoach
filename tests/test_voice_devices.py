"""With no device configured the voice follows the system default as it changes."""

from __future__ import annotations

import sys

import pytest

from sidecoach.voice.devices import system_default

WINDOWS = [
    {"name": "Microsoft Sound Mapper - Input", "max_input_channels": 2, "max_output_channels": 0},
    {"name": "Microphone (USB V7 AUDIO)", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "Microsoft Sound Mapper - Output", "max_input_channels": 0, "max_output_channels": 2},
    {"name": "Sceptre Y32 (NVIDIA High Defini", "max_input_channels": 0, "max_output_channels": 2},
    {"name": "Speakers (USB V7 AUDIO)", "max_input_channels": 0, "max_output_channels": 2},
]
ELSEWHERE = [
    {"name": "default", "max_input_channels": 2, "max_output_channels": 2},
    {"name": "HDA Intel", "max_input_channels": 2, "max_output_channels": 2},
]


def test_the_sound_mapper_is_chosen_on_windows_and_portaudio_decides_elsewhere() -> None:
    """PortAudio's default output is a concrete device fixed at start (the monitor's
    speakers in the first live game); the Sound Mapper opens whatever Windows' default is
    at that moment, so earphones connected mid-game are heard from the next line."""
    assert system_default("output", WINDOWS) == 2
    assert system_default("input", WINDOWS) == 0
    assert system_default("output", ELSEWHERE) is None
    assert system_default("input", ELSEWHERE) is None
    # A mapper with no channels of the wanted kind is not the one.
    assert system_default("input", [WINDOWS[2]]) is None


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="Windows' Core Audio")
def test_windows_saying_the_default_output_changed_is_counted() -> None:
    """The voice learns of a new default output from Windows itself (IMMNotificationClient,
    registered through ctypes): the watcher is set up with the real device enumerator, and the
    object Windows calls is called here as Windows would -- asked for the interface, then told
    of a change of the render default (counted) and of the capture default (not). No device
    on the machine is changed. Before, nothing watched (voice finding 6)."""
    import ctypes
    from ctypes import WINFUNCTYPE, byref, c_long, c_void_p, c_wchar_p

    from sidecoach.voice import devices

    start = devices.output_changes()
    watcher = devices._watcher  # noqa: SLF001
    assert watcher is not None and not devices._watch_failed  # noqa: SLF001
    this = ctypes.addressof(watcher._object)  # noqa: SLF001
    table = ctypes.cast(this, ctypes.POINTER(ctypes.POINTER(c_void_p)))[0]
    iid = (ctypes.c_byte * 16)()
    ctypes.WinDLL("ole32").CLSIDFromString(
        c_wchar_p("{7991EEC9-7E89-4D85-8390-6C703CEC60C0}"), byref(iid)
    )
    out = c_void_p()
    query = WINFUNCTYPE(c_long, c_void_p, c_void_p, c_void_p)(table[0])
    assert query(this, ctypes.addressof(iid), ctypes.addressof(out)) == 0 and out.value == this
    changed = WINFUNCTYPE(c_long, c_void_p, c_long, c_long, c_wchar_p)(table[6])
    assert changed(this, 0, 0, "{headphones}") == 0  # eRender, eConsole
    assert changed(this, 1, 0, "{headset mic}") == 0  # eCapture: not the voice's
    assert devices.output_changes() == start + 1


def test_a_configured_device_is_used_as_given() -> None:
    from sidecoach.voice.kokoro import KokoroTTS
    from sidecoach.voice.mic import Microphone

    assert Microphone(device=1).capture_device() == 1
    assert Microphone(device="USB V7").capture_device() == "USB V7"
    tts = KokoroTTS.__new__(KokoroTTS)
    tts.output_device = 8
    tts._playback = None
    assert tts.playback_device() == 8
