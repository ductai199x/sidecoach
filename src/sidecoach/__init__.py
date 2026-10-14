"""Sidecoach: vision-first chess coaching companion."""

import sys as _sys

__version__ = "0.1.88"

if _sys.platform.startswith("linux"):
    # The app's own PortAudio, found before sounddevice is first imported (sidecoach.linux).
    from sidecoach.linux import use_bundled_portaudio as _use_bundled_portaudio

    _use_bundled_portaudio()
