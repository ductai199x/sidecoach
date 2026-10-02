"""Which audio device the voice uses when none is configured, and when the system's changes.

PortAudio enumerates devices once, when it starts, and its default output is a concrete
device -- the monitor's speakers -- so Bluetooth earphones connected during a game would
never be heard even though Windows switched to them. Windows' MME host has a device that
follows the system default at every open: the Sound Mapper. With nothing configured the
voice plays through it, and the microphone reads from its input twin. A configured index or
name is used as given, and elsewhere than Windows PortAudio's own default is fine.

A stream stays open, though, and PortAudio's list never says the default changed (voice
finding 6). `output_changes` counts the system's changes of its default output as Windows
tells them (Core Audio's IMMNotificationClient, no polling): the voice reopens its stream on
the new default when the count moves, mid-line if it must (Tai, 2026-10-02: the voice follows
headphones plugged in or the default switched, as other apps do, without a restart).
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any, Literal

log = logging.getLogger(__name__)

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


# ---- the system's default output, as Windows tells its changes ------------------------------

_watch_lock = threading.Lock()
_watcher: _DefaultOutputWatcher | None = None
_watch_failed = False


def output_changes() -> int:
    """How many times the system's default output has changed since the voice first asked: a
    counter read for free, moved by Windows' own notification the moment the default changes.
    0 for ever off Windows, or where the notification could not be set up."""
    global _watcher, _watch_failed
    watcher = _watcher
    if watcher is not None:
        return watcher.changes
    if _watch_failed or not sys.platform.startswith("win"):
        return 0
    with _watch_lock:
        if _watcher is None and not _watch_failed:
            # Set up on a thread of its own, kept: COM is joined there (its multithreaded
            # apartment, alive as long as the thread is), never on the voice's or the caller's.
            made: list[_DefaultOutputWatcher | Exception] = []
            ready = threading.Event()

            def run() -> None:
                try:
                    made.append(_DefaultOutputWatcher())
                except Exception as exc:  # noqa: BLE001 - no audio service: nothing to follow
                    made.append(exc)
                ready.set()
                threading.Event().wait()  # holds the apartment for the process's life

            threading.Thread(target=run, name="audio-devices", daemon=True).start()
            ready.wait(5.0)
            if made and isinstance(made[0], _DefaultOutputWatcher):
                _watcher = made[0]
            else:
                why = made[0] if made else "no answer"
                log.warning("the system's output device will not be followed: %s", why)
                _watch_failed = True
                return 0
        return _watcher.changes if _watcher is not None else 0


class _DefaultOutputWatcher:
    """An IMMNotificationClient registered with Core Audio's device enumerator, through ctypes
    (no COM package). Windows calls `OnDefaultDeviceChanged` on a thread of its own; for the
    render default of the console role (what the Sound Mapper plays through) the count moves.
    Kept for the process's life: the callbacks and the object must outlive every call."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import POINTER, WINFUNCTYPE, byref, c_long, c_ulong, c_void_p, c_wchar_p

        self.changes = 0
        ole32 = ctypes.WinDLL("ole32")
        ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED on this thread; "already" is fine

        def guid(text: str) -> Any:
            out = (ctypes.c_byte * 16)()
            if ole32.CLSIDFromString(c_wchar_p(text), byref(out)) != 0:
                raise OSError(f"bad GUID {text}")
            return out

        clsid = guid("{BCDE0395-E52F-467C-8E3D-C4579291692E}")  # MMDeviceEnumerator
        iid_enumerator = guid("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
        self._iids = {
            bytes(guid("{00000000-0000-0000-C000-000000000046}")),  # IUnknown
            bytes(guid("{7991EEC9-7E89-4D85-8390-6C703CEC60C0}")),  # IMMNotificationClient
        }
        enumerator = c_void_p()
        hr = ole32.CoCreateInstance(
            byref(clsid), None, 0x17, byref(iid_enumerator), byref(enumerator)
        )
        if hr != 0 or not enumerator.value:
            raise OSError(f"no device enumerator (0x{hr & 0xFFFFFFFF:08X})")

        hresult = c_long  # what a callback returns (ctypes' HRESULT checks only calls made)
        this = c_void_p

        def query(_this: Any, riid: Any, out: Any) -> int:
            asked = bytes(ctypes.string_at(riid, 16))
            if asked in self._iids:
                ctypes.cast(out, POINTER(c_void_p))[0] = ctypes.addressof(self._object)
                return 0
            ctypes.cast(out, POINTER(c_void_p))[0] = None
            return -2147467262  # E_NOINTERFACE

        def default_changed(_this: Any, flow: int, role: int, _device: Any) -> int:
            if flow == 0 and role == 0:  # eRender, eConsole
                self.changes += 1
            return 0

        def nothing(*_args: Any) -> int:
            return 0

        def count(_this: Any) -> int:
            return 1  # a static object: never freed by its count

        # The vtable: IUnknown's three, then OnDeviceStateChanged, OnDeviceAdded,
        # OnDeviceRemoved, OnDefaultDeviceChanged, OnPropertyValueChanged (its PROPERTYKEY
        # passed by address on x64).
        self._functions = (
            WINFUNCTYPE(hresult, this, c_void_p, c_void_p)(query),
            WINFUNCTYPE(c_ulong, this)(count),
            WINFUNCTYPE(c_ulong, this)(count),
            WINFUNCTYPE(hresult, this, c_wchar_p, c_ulong)(nothing),
            WINFUNCTYPE(hresult, this, c_wchar_p)(nothing),
            WINFUNCTYPE(hresult, this, c_wchar_p)(nothing),
            WINFUNCTYPE(hresult, this, c_long, c_long, c_wchar_p)(default_changed),
            WINFUNCTYPE(hresult, this, c_wchar_p, c_void_p)(nothing),
        )
        self._table = (c_void_p * len(self._functions))(
            *[ctypes.cast(f, c_void_p) for f in self._functions]
        )
        self._object = (c_void_p * 1)(ctypes.addressof(self._table))
        self._enumerator = enumerator
        table = ctypes.cast(enumerator, POINTER(POINTER(c_void_p)))[0]
        # IMMDeviceEnumerator::RegisterEndpointNotificationCallback is its seventh method.
        register = WINFUNCTYPE(ctypes.HRESULT, this, c_void_p)(table[6])
        register(enumerator, ctypes.addressof(self._object))
