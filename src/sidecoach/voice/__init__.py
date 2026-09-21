"""Voice stage: text-to-speech now (Kokoro), local STT and VAD later.

The coach speaks through a `TTSProvider` (see `base`); `make_tts` builds the configured one,
falling back to silence when voice is off or a backend cannot load. Chess notation is turned
into words by `speak_notation` before it is voiced. See DESIGN.md section 5.9.
"""

from __future__ import annotations

from sidecoach.voice.base import NullTTS, TTSProvider
from sidecoach.voice.factory import make_recogniser, make_tts
from sidecoach.voice.notation import speak_notation
from sidecoach.voice.stt import NullSTT, STTProvider

__all__ = [
    "NullSTT",
    "NullTTS",
    "STTProvider",
    "TTSProvider",
    "make_recogniser",
    "make_tts",
    "speak_notation",
]
