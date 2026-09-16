"""With no device configured the voice follows the system default as it changes."""

from __future__ import annotations

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


def test_a_configured_device_is_used_as_given() -> None:
    from sidecoach.voice.kokoro import KokoroTTS
    from sidecoach.voice.mic import Microphone

    assert Microphone(device=1).capture_device() == 1
    assert Microphone(device="USB V7").capture_device() == "USB V7"
    tts = KokoroTTS.__new__(KokoroTTS)
    tts.output_device = 8
    tts._playback = None
    assert tts.playback_device() == 8
