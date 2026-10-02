"""The coach's hearing: one runtime for both voice models, loaded once; the VAD endpointer;
and the whole hear -> correct -> ask path over HTTP with a real clip. The ear providers
the factory builds are test_voice_factory.py's.
"""

from __future__ import annotations

import wave
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from sidecoach.config import Settings, VoiceSettings
from sidecoach.ui.panel import PanelEngine
from sidecoach.voice import runtime
from sidecoach.voice.kokoro import KokoroTTS
from sidecoach.voice.stt import ParakeetSTT
from sidecoach.voice.vad import Endpointer
from tests.fakes import FedMic

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


def test_cuda_is_the_default_and_falls_back_to_cpu_with_a_reason() -> None:
    from sidecoach.config import VoiceSettings

    assert VoiceSettings().tts_device == "cuda" and VoiceSettings().stt_device == "cuda"
    for wanted in ("auto", "cuda"):
        providers, where = runtime.providers_for(wanted)
        assert providers[-1] == "CPUExecutionProvider"  # the CPU is always the last resort
        assert where.name in ("cpu", "cuda")
        if where.name == "cuda":
            assert providers[0] == "CUDAExecutionProvider"
        else:
            assert "fell back" in where.detail


def test_an_onnx_that_will_not_import_trusts_the_provider_list(monkeypatch) -> None:  # noqa: ANN001
    """The probe builds its graph with `onnx`; an `onnx` that fails to import for any reason
    -- in the dev venv a broken ml_dtypes under it raised AttributeError -- leaves the
    provider list to say, where it made the GPU go unused."""
    import builtins

    real_import = builtins.__import__

    def broken(name: str, *args: object, **kwargs: object) -> object:
        if name == "onnx":
            raise AttributeError("module 'ml_dtypes' has no attribute 'bfloat16'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", broken)
    assert runtime._probe_cuda(object()) is True


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


def test_speech_starts_after_two_frames_over_the_bar_and_ends_after_silence() -> None:
    """A blip too short to be an ask is dropped: test_an_utterance_that_comes_to_nothing_is_said."""
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
    # While the coach speaks, the mic hears the speakers and the bar is raised: 0.6 is not a
    # start, 0.9 is.
    ep = Endpointer()
    assert ep.feed(0.6, coach_speaking=True) is None
    assert ep.feed(0.6, coach_speaking=True) is None
    assert ep.feed(0.9, coach_speaking=True) is None
    assert ep.feed(0.9, coach_speaking=True) == "start"


class _ScriptedGate:
    """A voice gate that answers a script of probabilities, one per frame."""

    def __init__(self, probs: list[float]) -> None:
        self.probs = list(probs)

    def load(self) -> bool:
        return True

    def healthcheck(self) -> tuple[bool, str]:
        return True, "scripted"

    def reset(self) -> None:
        pass

    def probability(self, frame: np.ndarray) -> float:
        return self.probs.pop(0) if self.probs else 0.0


def test_an_utterance_that_comes_to_nothing_is_said() -> None:
    """Whoever was told an utterance began is told when it comes to nothing: a click the
    endpointer drops in voice mode, a push-to-talk tap too short to hold a word. Neither was
    said before (2026-09-23), and the panel's dictation stayed held -- a cough in voice mode
    kept the words said just before it from ever being sent. The listener reads the frames as
    the test steps it (tests/fakes.py SteppedMic): every frame is handled when it is asked."""
    from sidecoach.voice.mic import HOP, Listener
    from tests.fakes import SteppedMic

    starts: list[str] = []
    dropped: list[str] = []
    heard: list[float] = []
    mic = SteppedMic()
    # Two confident frames start speech; the silence after it is a click, not an ask.
    gate = _ScriptedGate([0.9, 0.9] + [0.1] * 20)
    listener = Listener(
        mic,  # type: ignore[arg-type]
        gate,  # type: ignore[arg-type]
        on_utterance=lambda audio, seconds: heard.append(seconds),
        on_speech_start=lambda: starts.append("start"),
        on_dropped=lambda: dropped.append("dropped"),
    )
    listener.set_mode("vad")
    try:
        for _ in range(22):
            mic.frames.put(np.zeros(HOP, dtype=np.float32))
        mic.steps.idle()  # all 22 read and handled; the listener waits for the next
        assert starts == ["start"] and dropped == ["dropped"] and heard == []
        # Push-to-talk: a press and a release with nothing kept in between.
        listener.set_mode("ptt")
        listener.press()
        listener.release()
        assert dropped == ["dropped", "dropped"] and heard == []
        # Nothing was ever handed to the utterances' thread (so none is on its way).
        assert listener._deliverer is None and listener._deliveries.empty()
    finally:
        mic.steps.free()
        listener.set_mode("off")


# ---- hear -> correct -> ask over HTTP, with the real ear -------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(find_spec("onnx_asr") is None, reason="voice extras not installed")
def test_a_spoken_move_reaches_the_coach_as_san(
    make_panel: Callable[..., PanelEngine], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real ear on a real clip. The second of quiet before the ask is the panel's timer on
    a clock the test moves; the panel's own clock stays real, so `stt_ms` is the ear's real
    time."""
    from fastapi.testclient import TestClient

    from sidecoach.ui import panel as panel_mod
    from sidecoach.ui.server import build_app
    from tests.fakes import ManualClock

    clock = ManualClock()
    monkeypatch.setattr(panel_mod, "threading", clock.threading_module())
    settings = Settings(data_dir=tmp_path, voice=VoiceSettings(stt_device="cpu", listen="off"))
    engine = make_panel(settings, voice_models=True)
    # A position where Nf3 is legal: the start position, as the engine would report it.
    engine._state["fen"] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    engine._state["running"] = True  # a session: the ask is queued for its loop
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))

    if not engine._stt.load():
        pytest.skip(f"parakeet not loadable here: {engine._stt.healthcheck()[1]}")
    assert client.post("/api/hear", content=(AUDIO / "knight_f3.wav").read_bytes()).json()["ok"]

    lines = client.get("/api/poll").json()["lines"]
    heard = [ln for ln in lines if ln["chan"] == "voice" and ln["text"].startswith("heard:")]
    assert heard, f"nothing heard; lines: {[ln['text'] for ln in lines][-5:]}"
    assert heard[-1]["text"] == "heard: Nf3", heard[-1]
    assert heard[-1]["data"]["moves"] == ["Nf3"]
    assert heard[-1]["data"]["stt_ms"] < 2000
    # The corrected text is in the text box, and asked of the coach after a second of quiet.
    assert client.get("/api/poll").json()["state"]["dictation"]["text"] == "Nf3"
    assert engine._commands.empty(), "not before the second of quiet"
    clock.advance(panel_mod.DICTATION_SEND_DELAY_S)
    name, payload = engine._commands.get_nowait()
    assert (name, payload) == ("ask_spoken", "Nf3")
    # The ear was loaded once and says where.
    ear = engine._stt
    assert isinstance(ear, ParakeetSTT)
    assert ear.device is not None
    assert ear.device.name == "cpu"


class _MishearingEar:
    """A recogniser that hears every clip the way an ear mishears a spoken move."""

    def transcribe(self, audio: np.ndarray) -> str:
        return "Night F3"

    def healthcheck(self) -> tuple[bool, str]:
        return True, "a test ear"


def test_the_microphone_test_keeps_what_it_hears_and_asks_no_coach(
    make_panel: Callable[..., PanelEngine], tmp_path: Path
) -> None:
    """The settings page's live test (Tai, 2026-09-21): while it is on, an utterance is
    transcribed and corrected into the test's list, the coach is not asked, and the
    listening mode it interrupted comes back when it stops. No voice model and no
    microphone: the real ear is test_a_spoken_move_reaches_the_coach_as_san's."""
    from fastapi.testclient import TestClient

    from sidecoach.ui.server import build_app

    engine = make_panel(Settings(data_dir=tmp_path, voice=VoiceSettings(listen="off")))
    mic = FedMic()
    engine._stt = _MishearingEar()  # type: ignore[assignment]
    engine._mic = mic  # type: ignore[assignment]
    engine._listener.mic = mic  # type: ignore[assignment]
    engine._listener.gate = _ScriptedGate([])  # type: ignore[assignment]
    engine._state["fen"] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))
    status = client.get("/api/hear_test").json()
    assert status["on"] is False and status["heard"] == []
    started = client.post("/api/hear_test", json={"on": True}).json()
    assert started["ok"] and started["on"] and engine._listener.mode == "vad"
    # An utterance while the test is on: kept for the page, corrected, and not an ask.
    queued_before = engine._commands.qsize()
    client.post("/api/hear", content=(AUDIO / "knight_f3.wav").read_bytes())
    status = client.get("/api/hear_test").json()
    assert status["heard"] and status["heard"][0]["heard"] == "Nf3", "corrected to the legal move"
    assert status["heard"][0]["raw"] == "Night F3" and status["heard"][0]["stt_ms"] >= 0
    assert "level" in status and status["ok"]
    assert engine._commands.qsize() == queued_before, "nothing was asked of the coach"
    stopped = client.post("/api/hear_test", json={"on": False}).json()
    assert stopped["on"] is False and engine._listener.mode == "off"
