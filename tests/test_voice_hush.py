"""The voice cut off by the player -- a mute, a barge-in -- while a reply is written sentence
by sentence: silent from there, the reply whole in the chat and the captions (voice
findings 1 and 2, 2026-10-01). A fake voice: nothing loads, nothing plays."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from sidecoach.coach.coach import _FirstCandidate
from sidecoach.config import Settings
from sidecoach.events import Trigger
from sidecoach.ui.panel import PanelEngine

REPLY = (
    "Your knight on f5 is loose and the queen can reach g6 with check. "
    "That check forks the king and the knight. "
    "So before anything else, ask what defends f5. "
    "Then look at Nh4 and Ne3. "
)
CHUNKS = [REPLY[i : i + 20] for i in range(0, len(REPLY), 20)]


class FakeTTS:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.busy = False

    def healthcheck(self) -> tuple[bool, str]:
        return True, "fake"

    def say(self, text: str, *, voice: str | None = None) -> None:
        self.calls.append("say:" + text)
        self.busy = True

    def stop(self) -> None:
        self.calls.append("stop")
        self.busy = False


def _engine(tmp_path: Path) -> tuple[PanelEngine, FakeTTS]:
    engine = PanelEngine(Settings(data_dir=tmp_path), record=False, persist=False)
    fake = FakeTTS()
    engine._tts = fake  # type: ignore[assignment]
    return engine, fake


def _stream(engine: PanelEngine, fake: FakeTTS, cut: str) -> _FirstCandidate:
    sink = engine._stream_sink(Trigger.USER_ASK, None)
    assert sink is not None
    reader = _FirstCandidate(sink, lambda text: [])
    done = False
    for chunk in CHUNKS:
        reader.feed(chunk)
        if not done and any(c.startswith("say:") for c in fake.calls):
            done = True
            if cut == "mute":
                engine.set_muted(True)
            else:
                engine._on_speech_start()
    reader.finish()
    return reader


def _captioned(engine: PanelEngine) -> str:
    return " ".join(line[1] for line in engine.poll()["state"]["caption_lines"])


def test_mute_mid_reply_keeps_the_whole_reply_and_only_stops_the_sound(tmp_path: Path) -> None:
    engine, fake = _engine(tmp_path)
    reader = _stream(engine, fake, "mute")
    assert not reader.declined
    assert reader.spoken == REPLY.strip()  # the chat's text: the whole reply
    assert "Then look at Nh4 and Ne3." in _captioned(engine)
    assert fake.calls.index("stop") == len(fake.calls) - 1  # nothing said after the mute


def test_a_barge_in_silences_the_rest_of_the_reply(tmp_path: Path) -> None:
    engine, fake = _engine(tmp_path)
    reader = _stream(engine, fake, "barge")
    said_after = fake.calls[fake.calls.index("stop") + 1 :]
    assert said_after == []
    assert reader.spoken == REPLY.strip()
    assert "Then look at Nh4 and Ne3." in _captioned(engine)


def _feed_all(engine: PanelEngine) -> _FirstCandidate:
    sink = engine._stream_sink(Trigger.USER_ASK, None)
    assert sink is not None
    reader = _FirstCandidate(sink, lambda text: [])
    return reader


def test_a_mute_before_the_first_sentence_keeps_the_ai_reply(tmp_path: Path) -> None:
    """Muted while the model still thinks: the sink declined the first sentence, and the coach
    put the offline template in the AI reply's place (round 3 audit, 2026-10-02). The reply is
    the text, shown, and nothing is said."""
    engine, fake = _engine(tmp_path)
    reader = _feed_all(engine)
    engine.set_muted(True)
    for chunk in CHUNKS:
        reader.feed(chunk)
    reader.finish()
    assert not reader.declined
    assert reader.spoken == REPLY.strip()
    assert "Then look at Nh4 and Ne3." in _captioned(engine)
    assert not any(c.startswith("say:") for c in fake.calls)


def test_a_reply_held_for_the_pace_is_held_whole(tmp_path: Path) -> None:
    """A fast game holds a reply back for the pace ("text"): the sink declined it, and the
    coach's template took its place. The whole reply is the text and the word held; nothing is
    said until the game slows."""
    engine, fake = _engine(tmp_path)
    engine.voice_action = lambda *_args: "text"  # type: ignore[method-assign]
    reader = _feed_all(engine)
    for chunk in CHUNKS:
        reader.feed(chunk)
    reader.finish()
    assert not reader.declined
    assert reader.spoken == REPLY.strip()
    assert engine._held_word is not None and engine._held_word[1] == REPLY.strip()
    assert "Then look at Nh4 and Ne3." in _captioned(engine)
    assert not any(c.startswith("say:") for c in fake.calls)


class HeldSTT:
    """A recogniser that works only when the test lets it: the test acts while a recognition
    is under way by that state, not by how long one takes."""

    name = "held"

    def __init__(self, words: list[str]) -> None:
        self.words = words
        self.working = threading.Event()
        self.go = threading.Event()

    def healthcheck(self) -> tuple[bool, str]:
        return True, "fake"

    def load(self) -> bool:
        return True

    def transcribe(self, audio: Any, sample_rate: int = 16000) -> str:
        self.working.set()
        assert self.go.wait(5), "the test never let the recogniser go"
        return self.words.pop(0)


class FakeTimer:
    """`threading.Timer` that the test fires: the second of quiet is a step, not a wait."""

    made: list[FakeTimer] = []

    def __init__(self, delay: float, call: Any) -> None:
        self.delay, self.call = delay, call
        self.started = self.cancelled = False
        self.daemon = False
        FakeTimer.made.append(self)

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True


def test_a_press_while_the_last_words_are_recognised_keeps_one_question(
    tmp_path: Path,
) -> None:
    """Voice finding 5: the key pressed again while the first half was being recognised; the
    second of quiet started when that recognition ended, and sent the first half alone while
    the player still held the key. Driven by events and a timer the test fires: it slept to
    land the press and waited on the real second (round 3 audit, 2026-10-02)."""
    engine, _ = _engine(tmp_path)
    stt = HeldSTT(["what about", "the knight"])
    engine._stt = stt  # type: ignore[assignment]
    FakeTimer.made = []
    engine._make_timer = FakeTimer  # type: ignore[assignment]
    engine._listener.mode = "ptt"
    engine._state["running"] = True  # a session: the ask is queued for its loop
    audio = np.zeros(16000, np.float32)
    engine.ptt(True)
    first = threading.Thread(target=engine._on_utterance, args=(audio, 1.0))
    first.start()  # released: the first half goes to the recogniser ...
    assert stt.working.wait(5)
    engine.ptt(True)  # ... and the key is pressed again while it works
    stt.go.set()
    first.join()
    assert engine.poll()["state"]["dictation"]["sends_at"] is None
    assert not any(t.started and not t.cancelled for t in FakeTimer.made), "a send armed"
    assert engine._commands.empty()
    engine._on_utterance(audio, 1.0)  # released
    armed = [t for t in FakeTimer.made if t.started and not t.cancelled]
    assert len(armed) == 1
    armed[0].call()  # the second of quiet
    assert engine._commands.get_nowait() == ("ask_spoken", "what about the knight")
