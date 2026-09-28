"""The voice factory always returns something with a `say`, and voice never raises."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import numpy as np

from sidecoach.config import Settings, VoiceSettings
from sidecoach.voice import make_recogniser, make_tts
from sidecoach.voice.base import NullTTS
from sidecoach.voice.kokoro import KokoroTTS
from sidecoach.voice.stt import PARAKEET_MODEL, NullSTT, ParakeetSTT


def test_voice_turned_off_is_silent_and_deaf_with_a_reason() -> None:
    tts = make_tts(Settings(), off=True)
    assert isinstance(tts, NullTTS)
    ready, reason = tts.healthcheck()
    assert ready is False and reason
    # A silent provider must accept calls and do nothing, never raise.
    tts.say("Nf3 is a good move")
    tts.stop()
    stt = make_recogniser(Settings(), off=True)
    assert isinstance(stt, NullSTT)
    ready, reason = stt.healthcheck()
    assert ready is False and "off" in reason
    assert stt.transcribe(np.zeros(1600, dtype=np.float32)) == ""


def test_kokoro_is_built_with_the_configured_voice_and_model_dir(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, voice=VoiceSettings(voice="am_adam", speed=1.1))
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


def test_speech_recognition_is_parakeet_on_the_configured_device() -> None:
    stt = make_recogniser(Settings(voice=VoiceSettings(stt_device="cpu")))
    assert isinstance(stt, ParakeetSTT) and stt.device_wanted == "cpu"
    # It states its device before loading anything.
    ready, detail = stt.healthcheck()
    if find_spec("onnx_asr") is None:
        # Without the voice group (CI installs dev and panel only) the ear says so.
        assert not ready and "onnx-asr" in detail
    else:
        assert ready and detail == f"{PARAKEET_MODEL}, cpu (not loaded yet)"


def test_explicit_model_dir_overrides_the_default() -> None:
    settings = Settings(voice=VoiceSettings(model_dir="C:/models/kokoro"))
    assert settings.voice_model_dir() == Path("C:/models/kokoro")


def test_a_failed_kokoro_say_is_a_noop_and_starts_no_worker(tmp_path: Path) -> None:
    tts = KokoroTTS(tmp_path)
    tts._failed = True  # simulate a backend that could not load
    tts.say("Qh5+")
    assert tts._worker is None
