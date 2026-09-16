"""The coach's hearing: one runtime for both voice models, loaded once; the VAD endpointer;
the ear providers; and the whole hear -> correct -> ask path over HTTP with a real clip.
"""

from __future__ import annotations

import wave
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from sidecoach.config import Settings, VoiceSettings
from sidecoach.voice import runtime
from sidecoach.voice.kokoro import KokoroTTS
from sidecoach.voice.stt import NullSTT, ParakeetSTT, make_stt
from sidecoach.voice.vad import Endpointer

AUDIO = Path(__file__).parent / "audio"


def clip(name: str) -> np.ndarray:
    with wave.open(str(AUDIO / name), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


# ---- the runtime: device choice --------------------------------------------------------------


def test_cpu_is_cpu_and_bad_devices_are_refused() -> None:
    providers, where = runtime.providers_for("cpu")
    assert providers == ["CPUExecutionProvider"]
    assert where.name == "cpu"
    with pytest.raises(ValueError):
        runtime.providers_for("tpu")


def test_auto_and_cuda_land_somewhere_stated() -> None:
    for wanted in ("auto", "cuda"):
        providers, where = runtime.providers_for(wanted)
        assert providers[-1] == "CPUExecutionProvider"
        assert where.name in ("cpu", "cuda")
        if where.name == "cuda":
            assert providers[0] == "CUDAExecutionProvider"
        else:
            assert "cuda" in where.detail.lower() or "no usable" in where.detail


# ---- Kokoro loads its graph once, whatever the voice does ------------------------------------


def test_kokoro_loads_once_and_a_voice_change_reloads_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "kokoro-v1.0.onnx").write_bytes(b"x")
    (tmp_path / "voices-v1.0.bin").write_bytes(b"x")
    sessions = 0

    def fake_session(path: Path | str, device: str, **_: object) -> tuple[object, runtime.Device]:
        nonlocal sessions
        sessions += 1
        return object(), runtime.Device("cuda", "cuda (fake)")

    class FakeKokoro:
        @classmethod
        def from_session(cls, session: object, voices: str) -> FakeKokoro:
            return cls()

    monkeypatch.setattr("sidecoach.voice.kokoro.make_session", fake_session)
    import importlib.machinery
    import sys
    import types

    fake_mod = types.ModuleType("kokoro_onnx")
    fake_mod.Kokoro = FakeKokoro  # type: ignore[attr-defined]
    fake_mod.__spec__ = importlib.machinery.ModuleSpec("kokoro_onnx", None)
    fake_sd = types.ModuleType("sounddevice")
    fake_sd.__spec__ = importlib.machinery.ModuleSpec("sounddevice", None)
    monkeypatch.setitem(sys.modules, "kokoro_onnx", fake_mod)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    tts = KokoroTTS(tmp_path, voice="af_sarah", device="auto")
    assert tts.load() and tts.load() and tts.load()
    tts.set_voice("bm_george")
    tts.set_voice("af_heart")
    assert tts.load()
    assert sessions == 1, "the graph must be loaded exactly once"
    assert tts.voice == "af_heart"
    assert tts.device is not None and tts.device.name == "cuda"
    assert "cuda" in tts.healthcheck()[1]


# ---- the endpointer ----------------------------------------------------------------------------


def test_speech_starts_after_two_frames_and_ends_after_silence() -> None:
    ep = Endpointer(end_ms=96, min_utterance_ms=64)  # 3 silent frames end it; 2 frames is an ask
    assert ep.feed(0.9) is None  # one confident frame is not yet speech
    assert ep.feed(0.9) == "start"
    assert ep.speaking
    for _ in range(5):
        assert ep.feed(0.8) is None
    assert ep.feed(0.1) is None
    assert ep.feed(0.1) is None
    assert ep.feed(0.1) == "end"
    assert not ep.speaking


def test_a_blip_is_dropped_and_the_coach_talking_raises_the_bar() -> None:
    ep = Endpointer(end_ms=64, min_utterance_ms=320)
    ep.feed(0.9)
    assert ep.feed(0.9) == "start"
    assert ep.feed(0.1) is None
    assert ep.feed(0.1) == "drop"  # 64 ms of speech is a click, not an ask
    # While the coach speaks, the mic hears the speakers: 0.6 is not a start, 0.9 is.
    ep = Endpointer()
    assert ep.feed(0.6, coach_speaking=True) is None
    assert ep.feed(0.6, coach_speaking=True) is None
    assert ep.feed(0.9, coach_speaking=True) is None
    assert ep.feed(0.9, coach_speaking=True) == "start"


# ---- the ear providers -------------------------------------------------------------------------


def test_null_and_unknown_ears_are_deaf_with_a_reason() -> None:
    assert isinstance(make_stt("null", None, "cpu"), NullSTT)
    ear = make_stt("bat-signal", None, "cpu")
    assert isinstance(ear, NullSTT)
    assert "bat-signal" in ear.healthcheck()[1]
    assert ear.transcribe(np.zeros(1600, dtype=np.float32)) == ""


def test_parakeet_states_its_device_before_loading() -> None:
    ear = make_stt("parakeet", None, "cuda")
    assert isinstance(ear, ParakeetSTT)
    ready, detail = ear.healthcheck()
    assert "parakeet" in detail
    if ready:
        assert "cuda" in detail and "not loaded yet" in detail


# ---- hear -> correct -> ask over HTTP, with the real ear -------------------------------------


@pytest.mark.skipif(find_spec("onnx_asr") is None, reason="voice extras not installed")
def test_a_spoken_move_reaches_the_coach_as_san(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from sidecoach.ui.panel import PanelEngine
    from sidecoach.ui.server import build_app

    settings = Settings(
        data_dir=tmp_path,
        voice=VoiceSettings(enabled=True, stt_provider="parakeet", stt_device="cpu", listen="off"),
    )
    engine = PanelEngine(settings, analysis=False, coach=False, frame_source=iter(()))
    # A position where Nf3 is legal: the start position, as the engine would report it.
    engine._state["fen"] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    client = TestClient(build_app(engine))

    if not engine._stt.load():
        pytest.skip(f"parakeet not loadable here: {engine._stt.healthcheck()[1]}")
    assert client.post("/api/hear", content=(AUDIO / "knight_f3.wav").read_bytes()).json()["ok"]

    lines = client.get("/api/poll").json()["lines"]
    heard = [ln for ln in lines if ln["chan"] == "voice" and ln["text"].startswith("heard:")]
    assert heard, f"nothing heard; lines: {[ln['text'] for ln in lines][-5:]}"
    assert heard[-1]["text"] == "heard: Nf3", heard[-1]
    assert heard[-1]["data"]["moves"] == ["Nf3"]
    assert heard[-1]["data"]["stt_ms"] < 2000
    # The corrected text was queued for the coach as an ask.
    name, payload = engine._commands.get_nowait()
    assert (name, payload) == ("ask_spoken", "Nf3")
    # The ear was loaded once and says where.
    assert engine._stt.device is not None
    assert engine._stt.device.name == "cpu"
