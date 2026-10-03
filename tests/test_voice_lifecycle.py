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
    # Set: the device is there but asks for no samples yet (a busy audio stack).
    held = False

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
                    if not self.dead and not FakeOutputStream.held:
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
        self.closing.set()

    @property
    def closing(self) -> threading.Event:
        """Set once the stream is closed."""
        event = self.__dict__.get("_closing")
        if event is None:
            event = self.__dict__["_closing"] = threading.Event()
        return event


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
    FakeOutputStream.held = False
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


def test_a_live_stream_slow_to_take_a_line_is_waited_for_not_dropped(
    voice: WatchedTTS, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other side of the dead stream: a device that is there but has not asked for the
    line yet is waited on, round after round, and the line plays on the same stream once it
    asks. The voice's clock does not move, so the device is never silent for STREAM_DEAD_S;
    the test lets the device ask only after the voice has looked at least twice."""
    voice.say("First line.")
    voice.line_done()
    stream = FakeOutputStream.instances[0]
    looks = threading.Condition()
    seen = [0]
    read = kk.time.monotonic

    def monotonic() -> float:
        with looks:
            seen[0] += 1
            looks.notify_all()
        return read()

    monkeypatch.setattr(kk.time, "monotonic", monotonic)
    FakeOutputStream.held = True
    played = stream.played
    voice.say("Second line.")
    with looks:
        # The wait's own start, then a round that found the line still queued and the
        # device alive (the device asks for nothing while held, so these are the wait's).
        assert looks.wait_for(lambda: seen[0] >= 3, HANG_S), "the voice never looked again"
    assert voice.busy and not stream.dead
    FakeOutputStream.held = False
    voice.line_done()
    assert not voice.busy and not stream.dead
    assert FakeOutputStream.instances == [stream], "the same stream, not reopened"
    assert stream.played > played, "the second line played on it"


class SystemOutput:
    """The system's default output as the voice reads it (`devices.output_changes`): changed
    when the test says (headphones plugged in, the default switched in Windows)."""

    def __init__(self) -> None:
        self.changes = 0

    def switch(self) -> None:
        self.changes += 1


@pytest.fixture
def system(monkeypatch: pytest.MonkeyPatch, voice: WatchedTTS) -> SystemOutput:
    """The voice on the device that follows the system default, as it is when none is set."""
    output = SystemOutput()
    monkeypatch.setattr(kk, "follows_system_default", lambda device, kind: True)
    monkeypatch.setattr(kk, "output_changes", lambda: output.changes, raising=False)
    voice.output_device = 0
    return output


def test_the_voice_follows_the_systems_output_at_the_next_line(
    voice: WatchedTTS, system: SystemOutput
) -> None:
    """Headphones plugged in between two lines: the second is said on a stream opened on the new
    default, the old one closed. Before, the voice compared PortAudio's default as it read it at
    start, which never changes, and kept playing to the speakers (voice finding 6)."""
    voice.say("First line.")
    voice.line_done()
    first = FakeOutputStream.instances[0]
    voice.say("Still the speakers.")
    voice.line_done()
    assert FakeOutputStream.instances == [first], "nothing changed: the same stream"
    system.switch()
    voice.say("Second line.")
    voice.line_done()
    assert len(FakeOutputStream.instances) == 2 and first.closed
    second = FakeOutputStream.instances[1]
    second.halt()
    assert second.played > 0


def test_the_voice_follows_the_systems_output_mid_line(
    voice: WatchedTTS, system: SystemOutput
) -> None:
    """The default switched while a line plays: the rest of the line goes on a stream opened on
    the new output, at once, and the old stream is closed without playing it."""
    voice.say("First line.")
    voice.line_done()
    first = FakeOutputStream.instances[0]
    played = first.played
    FakeOutputStream.held = True  # the line queued on the speakers, not yet asked for
    FakeOutputStream.opening = threading.Event()
    queued = threading.Event()
    voice.on_play = lambda line: queued.set()  # said by the voice once the line is queued
    voice.say("A long line about the knight on f5.")
    assert queued.wait(HANG_S), "the line never reached its stream"
    assert voice._stream._stream is first and voice.busy  # noqa: SLF001 - on the speakers, playing
    system.switch()  # ... and the default changes under it
    assert FakeOutputStream.opening.wait(HANG_S), "no stream opened on the new output"
    assert first.closing.wait(HANG_S), "the old stream was never closed"
    FakeOutputStream.held = False
    voice.line_done()
    assert not voice.busy
    second = FakeOutputStream.instances[1]
    second.halt()
    first.halt()
    assert first.played == played, "the old output said nothing of the line"
    assert second.played >= 2400, "the whole line on the new output"


class HeldKokoro(FakeKokoro):
    """The model, holding the line that starts with `held` in synthesis until the test lets it
    go (`synthesising` says it began): the lines queued behind it wait their turn."""

    def __init__(self, held: str) -> None:
        self.held = held
        self.synthesising = threading.Event()
        self.go = threading.Event()

    def create(self, text: str, voice: str, speed: float, lang: str) -> tuple[Any, int]:
        if text.startswith(self.held):
            self.synthesising.set()
            assert self.go.wait(HANG_S), "the test never let the line go"
        return super().create(text, voice, speed, lang)


def _played(voice: WatchedTTS) -> list[str]:
    played: list[str] = []
    voice.on_play = played.append  # the moment a line's audio is queued to the device
    return played


def test_a_rooms_stop_cuts_its_own_line_and_leaves_the_coachs_queued_one(
    voice: WatchedTTS,
) -> None:
    """Round 3 audit, E5 (Tai, 2026-10-02): a card is being said, a coach line is queued after
    it, and the review is closed. Before, the room's stop stopped the voice only when a room
    had given the last line -- here the coach had, so the card played on after the review
    closed. Now the review's own lines stop, the playing card and a later one, and the coach's
    line is said."""
    model = HeldKokoro("Card one")
    voice._kokoro = model
    played = _played(voice)
    voice.say("Card one.", owner="review")
    assert model.synthesising.wait(HANG_S)
    voice.say("Coach line.")
    voice.say("Card two.", owner="review")
    voice.stop(owner="review")
    model.go.set()
    voice.line_done()  # the card, cut
    voice.line_done()  # the coach's line
    assert not voice.busy
    assert played == ["Coach line."]


def test_a_rooms_stop_leaves_the_coachs_playing_line_and_drops_its_queued_card(
    voice: WatchedTTS,
) -> None:
    """The other order: the coach is speaking and a card is queued after it, then the review is
    closed. Before, the card having given the last line, the stop cut the coach's line off; now
    the coach's line is said whole and the card never is."""
    model = HeldKokoro("Coach line")
    voice._kokoro = model
    played = _played(voice)
    voice.say("Coach line.")
    assert model.synthesising.wait(HANG_S)
    voice.say("Card one.", owner="review")
    voice.stop(owner="review")
    model.go.set()
    voice.line_done()
    assert not voice.busy
    assert played == ["Coach line."]
    # A card said after the room's stop is said.
    voice.say("Card two.", owner="review")
    voice.line_done()
    assert played == ["Coach line.", "Card two."]


class MadeKokoro(FakeKokoro):
    """The model, noting each line it makes (`made`) and saying when one is made (`one`)."""

    def __init__(self) -> None:
        self.made: list[str] = []
        self.one = threading.Semaphore(0)

    def create(self, text: str, voice: str, speed: float, lang: str) -> tuple[Any, int]:
        self.made.append(text)
        self.one.release()
        return super().create(text, voice, speed, lang)


def test_the_next_line_is_made_while_the_one_before_plays(voice: WatchedTTS) -> None:
    """Tai, 2026-10-03: "zero waits between sentences". The device holds the first line (it
    asks for no samples yet): the second is made meanwhile, before the first has played, and
    follows it at once; the voice stays busy from one into the other."""
    model = MadeKokoro()
    voice._kokoro = model
    played = _played(voice)
    FakeOutputStream.held = True
    voice.say("One.")
    voice.say("Two.")
    assert model.one.acquire(timeout=HANG_S) and model.one.acquire(timeout=HANG_S)
    assert model.made == ["One.", "Two."], "the second line made while the first waits to play"
    assert played == ["One."], "the second is not handed over before the first has played"
    assert voice.waiting == 1 and voice.busy
    FakeOutputStream.held = False
    voice.line_done()  # both, one after the other: busy throughout
    assert played == ["One.", "Two."]
    assert voice.waiting == 0


def test_a_line_that_comes_while_one_plays_is_made_before_it_ends(voice: WatchedTTS) -> None:
    """The model writes a reply's next sentence after the first has begun to play: it is made
    the moment it comes, not once the first has ended (the bench, 2026-10-03: looked for once,
    as the first began, it waited 0.39 s on the GPU, a second on the phone)."""
    model = MadeKokoro()
    voice._kokoro = model
    played: list[str] = []
    began = threading.Event()

    def on_play(line: str) -> None:
        played.append(line)
        began.set()

    voice.on_play = on_play
    FakeOutputStream.held = True
    voice.say("One.")
    assert began.wait(HANG_S), "the first handed to the device: it plays"
    assert model.one.acquire(timeout=HANG_S)
    voice.say("Two.")  # comes while "One." plays
    assert model.one.acquire(timeout=HANG_S), "made while the first plays"
    assert played == ["One."]
    FakeOutputStream.held = False
    voice.line_done()
    assert played == ["One.", "Two."]


def test_dropping_what_waits_leaves_the_line_playing(voice: WatchedTTS) -> None:
    """The pace plan's `drop_waiting`: the lines behind the one playing -- the one made ahead
    too -- go; the one playing is said to its end."""
    model = MadeKokoro()
    voice._kokoro = model
    played = _played(voice)
    FakeOutputStream.held = True
    voice.say("One.")
    voice.say("Two.")
    voice.say("Three.")
    assert model.one.acquire(timeout=HANG_S) and model.one.acquire(timeout=HANG_S)
    voice.drop_waiting()
    assert voice.waiting <= 1  # the one made ahead is dropped when its turn comes
    FakeOutputStream.held = False
    voice.line_done()
    assert played == ["One."]
    assert not voice.busy


def test_a_line_prepared_ahead_is_not_made_again(voice: WatchedTTS) -> None:
    """A fast line's word, held to the line's end, is made while it waits (`prepare`): said --
    even while it is still being made -- it plays without being made again."""
    model = MadeKokoro()
    voice._kokoro = model
    played = _played(voice)
    voice.prepare("Well found.")
    voice.say("Well found.")
    voice.line_done()
    assert played == ["Well found."]
    assert model.made == ["Well found."], "made once, ahead"


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


def test_a_45_second_question_held_on_the_key_is_handed_over_whole() -> None:
    """E3 (Tai, 2026-10-02): the longest utterance is 60 s, not 20 s -- a long question asked on
    the held key lost everything past 20 s."""
    from sidecoach.voice.mic import Listener
    from sidecoach.voice.vad import HOP, SAMPLE_RATE

    handed: queue.Queue[float] = queue.Queue()
    mic = SteppedMic()
    ear = Listener(
        mic,  # type: ignore[arg-type]
        _QuietGate(),  # type: ignore[arg-type]
        on_utterance=lambda audio, seconds: handed.put(len(audio) / SAMPLE_RATE),
    )
    ear.set_mode("ptt")
    try:
        ear.press()
        for _ in range(45 * SAMPLE_RATE // HOP):
            mic.frames.put(np.full(HOP, 0.01, dtype=np.float32))
        mic.steps.idle()
        assert handed.empty()  # still held: nothing handed over at 20 s
        ear.release()
        assert handed.get(timeout=HANG_S) >= 45 - HOP / SAMPLE_RATE
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
