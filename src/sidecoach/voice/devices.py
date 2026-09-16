"""Which audio device the voice uses when none is configured.

PortAudio enumerates devices once, when it starts, and its default output is a concrete
device -- the monitor's speakers -- so Bluetooth earphones connected during a game would
never be heard even though Windows switched to them. Windows' MME host has a device that
follows the system default at every open: the Sound Mapper. With nothing configured the
voice plays through it, and the microphone reads from its input twin, so a device change
takes effect on the next line spoken (for the mic, the next time listening is switched on,
since its stream stays open). A configured index or name is used as given, and elsewhere
than Windows PortAudio's own default is fine.
"""

from __future__ import annotations

import sys
from typing import Any, Literal

Kind = Literal["input", "output"]

_MAPPER = {"input": "Microsoft Sound Mapper - Input", "output": "Microsoft Sound Mapper - Output"}
_CHANNELS = {"input": "max_input_channels", "output": "max_output_channels"}


def system_default(kind: Kind, devices: list[dict[str, Any]] | None = None) -> int | None:
    """The index of the device that follows the system default, or None for PortAudio's.

    `devices` is the sounddevice device list, for tests; queried when not given.
    """
    if devices is None:
        if not sys.platform.startswith("win"):
            return None
        try:
            import sounddevice as sd  # type: ignore[reportMissingImports]

            devices = [dict(device) for device in sd.query_devices()]
        except Exception:  # noqa: BLE001 - no sounddevice, no audio host: PortAudio decides
            return None
    for index, device in enumerate(devices):
        if str(device.get("name", "")).startswith(_MAPPER[kind]) and device.get(_CHANNELS[kind], 0):
            return index
    return None


def follows_system_default(device: int | str | None, kind: Kind) -> bool:
    """Whether `device` is the one that tracks the system default."""
    if device is None:
        return False
    try:
        import sounddevice as sd  # type: ignore[reportMissingImports]

        return str(sd.query_devices(device).get("name", "")).startswith(_MAPPER[kind])
    except Exception:  # noqa: BLE001
        return False
