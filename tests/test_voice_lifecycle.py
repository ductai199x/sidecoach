"""The voice's and the recogniser's life: loaded once the files arrive, a stop that lands while
the stream opens, a dead output stream, a replaced voice released (voice findings 4, 6, 11 and
13, 2026-10-01). A fake model and a fake sounddevice: nothing is synthesised, nothing plays."""

from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import sidecoach.voice.kokoro as kk
from sidecoach.voice.kokoro import ONNX_FILE, VOICES_FILE, KokoroTTS
from sidecoach.voice.stt import ParakeetSTT


class FakeKokoro:
    def create(self, text: str, voice: str, speed: float, lang: str) -> tuple[Any, int]:
        time.sleep(0.02)
        return np.ones(2400, dtype=np.float32), 24000  # 0.1 s


class FakeOutputStream:
    # When set: `opening` is set as a stream begins to open, and it opens once `gate` is.
    gate: threading.Event | None = None
    opening: threading.Event | None = None
    instances: list[FakeOutputStream] = []

    def __init__(self, samplerate: int, channels: int, dtype: str, device: Any, callback: Any):
        if FakeOutputStream.opening is not None:
            FakeOutputStream.opening.set()
        if FakeOutputStream.gate is not None:
            assert FakeOutputStream.gate.wait(5), "the test never let the stream open"
        self.callback = callback
        self.played = 0
        self.closed = False
        self.dead = False
        FakeOutputStream.instances.append(self)

    def start(self) -> None:
        def run() -> None:
            while not self.closed:
                if not self.dead:
                    out = np.zeros((480, 1), dtype=np.float32)
                    self.callback(out, 480, None, None)
                    self.played += int(np.count_nonzero(out))
                time.sleep(0.002)

        threading.Thread(target=run, daemon=True).start()

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakeSD:
    OutputStream = FakeOutputStream

    class default:
        device = (0, 1)

    @staticmethod
    def query_devices(index: Any) -> dict[str, str]:
        return {"name": "Speakers"}

    @staticmethod
    def stop() -> None:
        pass


@pytest.fixture
def voice(monkeypatch: pytest.MonkeyPatch) -> KokoroTTS:
    monkeypatch.setattr(kk, "follows_system_default", lambda device, kind: False)
    FakeOutputStream.gate = None
    FakeOutputStream.opening = None
    FakeOutputStream.instances = []
    tts = KokoroTTS(model_dir=Path("."), output_device=1)
    tts._kokoro = FakeKokoro()
    tts._sd = FakeSD()
    return tts


def _until(cond: Any, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_a_stop_while_the_stream_opens_is_honoured(
    voice: KokoroTTS, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Voice finding 13: a stop landing while the stream opened was ignored, and the line
    played over a mute or a barge-in. The stream opens only when the test lets it, and the stop
    lands while it opens: by that state, not by a 0.3 s open raced against a 0.05 s sleep
    (round 3 audit, 2026-10-02)."""
    queued: list[int] = []
    real_queue = kk._Stream.queue

    def queue(self: Any, samples: Any) -> None:
        queued.append(len(samples))
        real_queue(self, samples)

    monkeypatch.setattr(kk._Stream, "queue", queue)
    FakeOutputStream.gate = threading.Event()
    FakeOutputStream.opening = threading.Event()
    voice.on_play = None
    voice.say("Your knight on f5 is loose.")
    assert FakeOutputStream.opening.wait(5), "the stream never began to open"
    voice.stop()
    FakeOutputStream.gate.set()
    assert _until(lambda: not voice.busy)
    assert queued == [], "the line was queued after the stop"


def test_a_dead_output_stream_is_dropped_and_reopened(
    voice: KokoroTTS, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Voice finding 6: a dead stream (the HDMI monitor asleep) held each line for its length
    and five seconds, silent, and kept the voice busy. On a clock the test moves: past the
    dead-stream limit with no call from the device, the stream is dropped, and the next line
    opens a new one. It asserted a wall-clock duration (round 3 audit, 2026-10-02)."""
    now = [0.0]
    monkeypatch.setattr(
        kk,
        "time",
        types.SimpleNamespace(monotonic=lambda: now[0], perf_counter=time.perf_counter),
    )
    voice.say("First line.")
    assert _until(lambda: FakeOutputStream.instances and not voice.busy)
    first = FakeOutputStream.instances[0]
    first.dead = True  # the HDMI monitor went to sleep
    voice.say("Second line.")
    stream = voice._stream
    assert stream is not None
    assert _until(lambda: bool(stream._chunks)), "the second line never reached the stream"
    assert voice.busy
    now[0] += kk.STREAM_DEAD_S + 0.1  # no call from the device since
    assert _until(lambda: not voice.busy)
    assert first.closed and voice._stream is None
    voice.say("Third line.")
    assert _until(lambda: len(FakeOutputStream.instances) == 2 and not voice.busy)
    assert FakeOutputStream.instances[1].played > 0


def test_a_closed_voice_releases_its_worker_model_and_stream(voice: KokoroTTS) -> None:
    voice.say("A line.")
    assert _until(lambda: FakeOutputStream.instances and not voice.busy)
    worker = voice._worker
    assert worker is not None
    voice.close()
    worker.join(2)
    assert not worker.is_alive()
    assert voice._kokoro is None
    assert FakeOutputStream.instances[0].closed


def test_the_voice_loads_once_setup_has_fetched_its_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions: list[Path] = []

    def session(path: Path, device: str) -> Any:
        sessions.append(path)
        if not path.exists():
            raise FileNotFoundError(path)
        return object(), None

    monkeypatch.setattr(kk, "make_session", session)
    fake_onnx = types.ModuleType("kokoro_onnx")
    fake_onnx.Kokoro = types.SimpleNamespace(from_session=lambda s, v: FakeKokoro())  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kokoro_onnx", fake_onnx)
    monkeypatch.setitem(sys.modules, "sounddevice", FakeSD)
    tts = KokoroTTS(tmp_path)
    assert tts.load() is False  # the warm-up, before Setup
    assert tts.load() is False and len(sessions) == 1  # nothing changed: not tried again
    (tmp_path / ONNX_FILE).write_bytes(b"x")
    (tmp_path / VOICES_FILE).write_bytes(b"x")
    assert tts.load() is True


def test_the_recogniser_loads_once_setup_has_fetched_it(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    fake = types.ModuleType("onnx_asr")

    def load_model(model: str, **kw: Any) -> Any:
        calls.append(model)
        if len(calls) == 1:
            raise RuntimeError("Cannot reach the hub: outgoing traffic has been disabled")
        return object()

    fake.load_model = load_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnx_asr", fake)
    stt = ParakeetSTT(device="cpu")
    assert stt.load() is False
    assert stt._failed is not None and stt._failed.startswith("not downloaded")
    assert stt.load() is True  # Setup fetched it: no restart needed


def test_speech_recognition_is_on_once_setup_has_fetched_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Voice finding 4, the player's path: the recogniser failed to load at start (its files not
    there yet), so listening stayed off, the microphone button hidden and its test refused,
    until a restart -- Setup fetched the files and nothing loaded them (round 3 audit,
    2026-10-02). Setup's run ends, the recogniser is loaded, and listening is applied as the
    settings say."""
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    import sidecoach.setup as setup_module
    from sidecoach.config import Settings
    from sidecoach.setup import Item
    from sidecoach.ui.panel import PanelEngine
    from sidecoach.ui.server import build_app

    monkeypatch.setenv("SIDECOACH_CONFIG", str(tmp_path / "sidecoach.toml"))
    monkeypatch.delenv("SIDECOACH_DATA_DIR", raising=False)
    settings = Settings(data_dir=tmp_path)
    settings.voice.listen = "ptt"
    engine = PanelEngine(settings, analysis=False, coach=False, frame_source=iter(()))
    files = {"there": False}

    class Recogniser:
        loaded = False

        def healthcheck(self) -> tuple[bool, str]:
            return (True, "fake") if self.loaded or files["there"] else (False, "not downloaded")

        def load(self) -> bool:
            self.loaded = files["there"]
            return self.loaded

    modes: list[str] = []

    def set_mode(mode: str) -> str:
        modes.append(mode)
        listener.mode = mode
        return mode

    listener = SimpleNamespace(mode="off", level=0.0, set_mode=set_mode, mic=None)
    engine._listener = listener  # type: ignore[assignment]
    engine._hotkey = SimpleNamespace(start=lambda: True, stop=lambda: None, detail="")  # type: ignore[assignment]
    engine._mic = SimpleNamespace(healthcheck=lambda: (True, "mic"))  # type: ignore[assignment]
    engine._stt = Recogniser()  # type: ignore[assignment]
    engine._tts = SimpleNamespace(healthcheck=lambda: (True, "voice"), busy=False)  # type: ignore[assignment]
    engine._warm_voice()  # the start: no files yet
    assert engine.poll(0)["state"]["stt_ready"] is False and listener.mode == "off"

    def fetch(_progress: object) -> None:
        files["there"] = True

    item = Item("stt", "Speech recognition", 1, lambda: files["there"], fetch)
    monkeypatch.setattr(setup_module, "items", lambda _settings: [item])
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))
    assert client.post("/api/setup/start", json={"ids": ["stt"]}).json()["ok"]
    runs = [t for t in threading.enumerate() if t.name == "setup"]
    for run in runs:
        run.join(10)
    for after in [t for t in threading.enumerate() if t.name == "voice-after-setup"]:
        after.join(10)
    state = engine.poll(0)["state"]
    assert state["stt_ready"] is True, state["stt_detail"]
    assert listener.mode == "ptt" and modes[-1] == "ptt"


class _QuietGate:
    def load(self) -> bool:
        return True

    def healthcheck(self) -> tuple[bool, str]:
        return True, "fake"

    def reset(self) -> None:
        pass

    def probability(self, frame: Any) -> float:
        return 0.0


def test_push_to_talk_stops_at_the_longest_utterance_and_a_mode_switch_lets_go() -> None:
    """Voice finding 7: a held key buffered without end; a switch with the key down left it
    pressed, so the next press was ignored."""
    from sidecoach.voice.mic import MAX_UTTERANCE_S, Listener
    from sidecoach.voice.vad import HOP, SAMPLE_RATE
    from tests.fakes import FedMic

    utterances: list[float] = []
    mic = FedMic()
    ear = Listener(
        mic,  # type: ignore[arg-type]
        _QuietGate(),  # type: ignore[arg-type]
        on_utterance=lambda audio, s: utterances.append(len(audio) / SAMPLE_RATE),
    )
    ear.set_mode("ptt")
    try:
        ear.press()
        for _ in range(int((MAX_UTTERANCE_S + 5) * SAMPLE_RATE // HOP)):
            mic.frames.put(np.full(HOP, 0.01, dtype=np.float32))
        assert _until(lambda: mic.frames.empty() and utterances)
        assert utterances[0] <= MAX_UTTERANCE_S + 0.1
        assert len(ear._buffer) * HOP <= SAMPLE_RATE  # holding on keeps nothing more
        ear.release()
        ear.press()
        ear.set_mode("vad")
        ear.set_mode("ptt")
        assert not ear._ptt
        ear.press()
        assert ear._started_at is not None  # the fresh press is taken
    finally:
        ear.set_mode("off")


def test_a_microphone_whose_stream_died_is_reopened(monkeypatch: pytest.MonkeyPatch) -> None:
    """Voice finding 17: a dead input stream was never reopened; the coach stayed deaf."""
    import sidecoach.voice.mic as mic_mod
    from sidecoach.voice.mic import Listener
    from tests.fakes import FedMic

    monkeypatch.setattr(mic_mod, "MIC_REOPEN_S", 0.0)

    class DyingMic(FedMic):
        def __init__(self) -> None:
            super().__init__()
            self.opens = 0
            self.dead = False

        def open(self) -> bool:
            self.opens += 1
            self.dead = False
            return super().open()

        def stalled(self) -> bool:
            return self.dead

    mic = DyingMic()
    ear = Listener(mic, _QuietGate(), on_utterance=lambda a, s: None)  # type: ignore[arg-type]
    ear.set_mode("ptt")
    try:
        assert mic.opens == 1
        mic.dead = True  # the headset unplugged
        assert _until(lambda: mic.opens == 2)
    finally:
        ear.set_mode("off")
