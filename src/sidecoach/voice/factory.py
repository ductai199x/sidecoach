"""Build the configured text-to-speech provider. Voice never fails the session: anything
unusable -- turned off, an unknown provider, a backend that will not load -- comes back as
the silent NullTTS with a reason, so the caller always has a `say` to call.
"""

from __future__ import annotations

from sidecoach.config import Settings
from sidecoach.voice.base import NullTTS, TTSProvider
from sidecoach.voice.kokoro import KokoroTTS


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
        )
    return NullTTS(f"unknown voice.provider {voice.provider!r}")
