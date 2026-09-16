"""Build the configured voice providers. Voice never fails the session: anything unusable --
turned off, an unknown provider, a backend that will not load -- comes back as the silent
NullTTS or the deaf NullSTT with a reason, so the caller always has a `say` and a `transcribe`.
"""

from __future__ import annotations

from sidecoach.config import Settings
from sidecoach.voice.base import NullTTS, TTSProvider
from sidecoach.voice.kokoro import KokoroTTS
from sidecoach.voice.stt import STTProvider, make_stt


def make_tts(settings: Settings) -> TTSProvider:
    voice = settings.voice
    if not voice.enabled:
        return NullTTS("voice disabled")
    if voice.provider == "null":
        return NullTTS("voice provider is null")
    if voice.provider == "kokoro":
        return KokoroTTS(
            settings.voice_model_dir(),
            voice=voice.voice,
            speed=voice.speed,
            lang=voice.lang,
            device=voice.tts_device,
        )
    return NullTTS(f"unknown voice.provider {voice.provider!r}")


def make_ear(settings: Settings) -> STTProvider:
    """The configured speech recogniser, or a deaf one with a reason."""
    voice = settings.voice
    if not voice.enabled:
        return make_stt("null", None, "cpu")
    return make_stt(voice.stt_provider, voice.stt_model, voice.stt_device)
