"""Build the voice providers: Kokoro speaks, Parakeet listens. Voice never fails the
session: a backend that will not load comes back as the silent NullTTS or the deaf NullSTT
with a reason, so the caller always has a `say` and a `transcribe`. `off` gives those
outright -- the tests, and a terminal watcher without a voice -- so no model is loaded.
"""

from __future__ import annotations

from sidecoach.config import Settings
from sidecoach.voice.base import NullTTS, TTSProvider
from sidecoach.voice.kokoro import KokoroTTS
from sidecoach.voice.stt import STTProvider, make_stt


def make_tts(settings: Settings, *, off: bool = False) -> TTSProvider:
    voice = settings.voice
    if off:
        return NullTTS("voice off")
    return KokoroTTS(
        settings.voice_model_dir(),
        voice=voice.voice,
        speed=voice.speed,
        lang=voice.lang,
        device=voice.tts_device,
        output_device=_device(voice.output_device),
    )


def _device(spec: str | None) -> int | str | None:
    """A sounddevice device: an index if the setting is a number, else a name, else None."""
    if spec is None or spec == "":
        return None
    return int(spec) if spec.isdigit() else spec


def make_ear(settings: Settings, *, off: bool = False) -> STTProvider:
    """The speech recogniser, or a deaf one with a reason."""
    if off:
        return make_stt(None, off=True)
    return make_stt(settings.voice.stt_device)
