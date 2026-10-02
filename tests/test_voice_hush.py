"""The voice cut off by the player -- a mute, a barge-in -- while a reply is written sentence
by sentence: silent from there, the reply whole in the chat and the captions (voice
findings 1 and 2, 2026-10-01). A fake voice: nothing loads, nothing plays."""

from __future__ import annotations

from pathlib import Path

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
