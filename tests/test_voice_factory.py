"""The voice factory always returns something with a `say`, and voice never raises."""

from __future__ import annotations

from pathlib import Path

from sidecoach.config import Settings, VoiceSettings
from sidecoach.voice import make_tts
from sidecoach.voice.base import NullTTS
from sidecoach.voice.kokoro import KokoroTTS


def test_disabled_voice_is_silent() -> None:
    tts = make_tts(Settings(voice=VoiceSettings(enabled=False)))
    assert isinstance(tts, NullTTS)
    ready, reason = tts.healthcheck()
    assert ready is False and reason
    # A silent provider must accept calls and do nothing, never raise.
    tts.say("Nf3 is a good move")
    tts.stop()


def test_unknown_provider_degrades_to_silence_with_a_reason() -> None:
    tts = make_tts(Settings(voice=VoiceSettings(enabled=True, provider="festival")))
    assert isinstance(tts, NullTTS)
    ready, reason = tts.healthcheck()
    assert ready is False
    assert "festival" in reason


def test_kokoro_is_built_with_the_configured_voice_and_model_dir(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        voice=VoiceSettings(enabled=True, provider="kokoro", voice="am_adam", speed=1.1),
    )
    tts = make_tts(settings)
    assert isinstance(tts, KokoroTTS)
    assert tts.voice == "am_adam"
    assert tts.speed == 1.1
    assert tts.model_dir == tmp_path / "models" / "kokoro"
    # No model files and (in CI) no extras: healthcheck reports not-ready without loading, and
    # neither healthcheck nor stop raises.
    ready, reason = tts.healthcheck()
    assert ready is False and reason
    tts.stop()


def test_explicit_model_dir_overrides_the_default() -> None:
    settings = Settings(voice=VoiceSettings(enabled=True, model_dir="C:/models/kokoro"))
    assert settings.voice_model_dir() == Path("C:/models/kokoro")


def test_a_failed_kokoro_say_is_a_noop_and_starts_no_worker(tmp_path: Path) -> None:
    tts = KokoroTTS(tmp_path)
    tts._failed = True  # simulate a backend that could not load
    tts.say("Qh5+")
    assert tts._worker is None
