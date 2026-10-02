"""The voice's and the recogniser's life: loaded once the files arrive, a stop that lands while
the stream opens, a dead output stream, a replaced voice released (voice findings 4, 6, 11 and
13, 2026-10-01). A fake model and a fake sounddevice: nothing is synthesised, nothing plays.

No test waits on the real time: the voice reads a clock the test moves (tests/fakes.py
ManualClock), a line's end is said by the voice itself (`WatchedTTS.finished`), and the
listener reads its frames as the test steps it (tests/fakes.py SteppedMic)."""

from __future__ import annotations

import queue
import sys
import threading
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import sidecoach.voice.kokoro as kk
from sidecoach.voice.kokoro import ONNX_FILE, VOICES_FILE, KokoroTTS
from sidecoach.voice.stt import ParakeetSTT
from tests.fakes import HANG_S, ManualClock, SteppedMic


class FakeKokoro:
    def create(self, text: str, voice: str, speed: float, lang: str) -> tuple[Any, int]:
        return np.ones(2400, dtype=np.float32), 24000  # 0.1 s


class FakeOutputStream:
    """A sound card's output stream: from `start` its own thread asks the callback for samples,
    as a device does, until it is closed -- or `kill`ed, the device gone: no callback after
    that. With `gate` set, opening one waits for the test to let it go (`opening` says it
    began)."""

    gate: threading.Event | None = None
    opening = threading.Event()
    instances: list[FakeOutputStream] = []

    def __init__(self, samplerate: int, channels: int, dtype: str, device: Any, callback: Any):
        FakeOutputStream.opening.set()
        if FakeOutputStream.gate is not None:
            FakeOutputStream.gate.wait(HANG_S)
        self.callback = callback
        self.played = 0
        self.closed = False
        self.dead = False
        self._device = threading.Lock()
        self._thread: threading.Thread | None = None
        FakeOutputStream.instances.append(self)

    def start(self) -> None:
        def run() -> None:
            while not self.closed:
                with self._device:
                    if not self.dead:
                        out = np.zeros((480, 1), dtype=np.float32)
                        self.callback(out, 480, None, None)
                        self.played += int(np.count_nonzero(out))
                time.sleep(0.002)  # the device's own period; nothing waits on it

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def kill(self) -> None:
        """The device went away: once this returns, no callback runs again."""
        with self._device:
            self.dead = True

    def halt(self) -> None:
        """Closed, and its thread ended: what it played is all it will play."""
        self.closed = True
        if self._thread is not None:
            self._thread.join(HANG_S)
            assert not self._thread.is_alive()

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


class WatchedTTS(KokoroTTS):
    """The voice, saying when its worker is done with a line -- taken up, played or dropped,
    and no longer `busy` with it: `finished` gets one item per line."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.finished: queue.Queue[str] = queue.Queue()
        super().__init__(*args, **kwargs)

    @property  # type: ignore[override]
    def _playing(self) -> bool:
        return bool(self.__dict__.get("_playing_now", False))

    @_playing.setter
    def _playing(self, value: bool) -> None:
        was = self.__dict__.get("_playing_now", False)
        self.__dict__["_playing_now"] = value
        if was and not value:
            self.finished.put("done")

    def line_done(self) -> None:
        assert self.finished.get(timeout=HANG_S) == "done", "the line never ended"


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> ManualClock:
    """The voice's clock, moved by the test only: a stream is never taken for dead because
    the machine was slow."""
    clock = ManualClock()
    monkeypatch.setattr(kk, "time", clock.time_module())
    return clock


@pytest.fixture
def voice(monkeypatch: pytest.MonkeyPatch, clock: ManualClock) -> Iterator[WatchedTTS]:
    monkeypatch.setattr(kk, "follows_system_default", lambda device, kind: False)
    FakeOutputStream.gate = None
    FakeOutputStream.opening = threading.Event()
    FakeOutputStream.instances = []
    tts = WatchedTTS(model_dir=Path("."), output_device=1)
    tts._kokoro = FakeKokoro()
    tts._sd = FakeSD()
    yield tts
    if FakeOutputStream.gate is not None:
        FakeOutputStream.gate.set()
    tts.close()
    for stream in FakeOutputStream.instances:
        stream.close()


def test_a_stop_while_the_stream_opens_is_honoured(voice: WatchedTTS) -> None:
    FakeOutputStream.gate = threading.Event()
    voice.on_play = None
    voice.say("Your knight on f5 is loose.")
    assert FakeOutputStream.opening.wait(HANG_S), "the line never reached its stream"
    assert FakeOutputStream.instances == [], "the stream is still opening"
    voice.stop()  # ... and the player stops the voice now
    FakeOutputStream.gate.set()
    voice.line_done()
    assert not voice.busy
    stream = FakeOutputStream.instances[0]
    stream.halt()
    assert stream.played == 0


def test_a_dead_output_stream_is_dropped_and_reopened(
    voice: WatchedTTS, clock: ManualClock
) -> None:
    voice.say("First line.")
    voice.line_done()
    assert FakeOutputStream.instances and not voice.busy
    first = FakeOutputStream.instances[0]
    first.kill()  # the HDMI monitor went to sleep
    begun = clock.now
    voice.say("Second line.")
    # The device's silence is what tells: STREAM_DEAD_S of it and the line ends, long before
    # the line's own timeout (its length and five seconds), which the clock never reaches.
    clock.advance(kk.STREAM_DEAD_S + 0.25)
    voice.line_done()
    assert not voice.busy
    assert clock.now - begun < kk.STREAM_DEAD_S + 1.0  # not the line's length + 5 s
    assert first.closed
    voice.say("Third line.")
    voice.line_done()
    assert len(FakeOutputStream.instances) == 2 and not voice.busy
    second = FakeOutputStream.instances[1]
    second.halt()
    assert second.played > 0


def test_a_closed_voice_releases_its_worker_model_and_stream(voice: WatchedTTS) -> None:
    voice.say("A line.")
    voice.line_done()
    assert FakeOutputStream.instances and not voice.busy
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

    utterances: list[float] = []
    handed: queue.Queue[float] = queue.Queue()

    def on_utterance(audio: Any, seconds: float) -> None:
        utterances.append(len(audio) / SAMPLE_RATE)
        handed.put(seconds)

    mic = SteppedMic()
    ear = Listener(
        mic,  # type: ignore[arg-type]
        _QuietGate(),  # type: ignore[arg-type]
        on_utterance=on_utterance,
    )
    ear.set_mode("ptt")
    try:
        ear.press()
        for _ in range(int((MAX_UTTERANCE_S + 5) * SAMPLE_RATE // HOP)):
            mic.frames.put(np.full(HOP, 0.01, dtype=np.float32))
        mic.steps.idle()  # every frame read and handled; the listener waits for the next
        handed.get(timeout=HANG_S)  # the utterance's own thread handed it over
        assert mic.frames.empty() and utterances
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
        mic.steps.free()
        ear.set_mode("off")


def test_a_microphone_whose_stream_died_is_reopened(monkeypatch: pytest.MonkeyPatch) -> None:
    """Voice finding 17: a dead input stream was never reopened; the coach stayed deaf. It is
    tried again in the listener's quiet rounds, at most once every MIC_REOPEN_S (the
    listener's clock is one the test moves; its rounds are the test's steps)."""
    import sidecoach.voice.mic as mic_mod
    from sidecoach.voice.mic import Listener

    clock = ManualClock()
    monkeypatch.setattr(mic_mod, "time", clock.time_module())

    class DyingMic(SteppedMic):
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
        mic.steps.idle()
        assert mic.opens == 1, "a live stream is left alone"
        mic.dead = True  # the headset unplugged
        mic.steps.idle()  # the listener's next quiet round
        assert mic.opens == 2
        # Dead again at once: not tried again until MIC_REOPEN_S have passed.
        tried = clock.now
        mic.dead = True
        mic.steps.idle()
        clock.advance_to(tried + mic_mod.MIC_REOPEN_S - 0.01)
        mic.steps.idle()
        assert mic.opens == 2, "tried again too soon"
        clock.advance_to(tried + mic_mod.MIC_REOPEN_S)
        mic.steps.idle()
        assert mic.opens == 3
    finally:
        mic.steps.free()
        ear.set_mode("off")
