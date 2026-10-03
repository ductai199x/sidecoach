"""The panel's voice following the player's pace (`voice.pace = "follow"`, the default): the
plan (`voice/pace.py`) decides, the panel captions every sentence the moment it exists, keeps
the voice one line ahead (the next is made while one plays: no wait between sentences), and
writes the reply into the chat as it comes (Tai, 2026-10-03)."""

from __future__ import annotations

from collections.abc import Callable

from sidecoach.events import Trigger
from sidecoach.ui.panel import PanelEngine


class PacedTTS:
    """A voice that plays one line and makes the next meanwhile: `busy` while any line is
    with it, `waiting` the lines behind the one playing. The test ends a line (`end`)."""

    name = "paced"

    def __init__(self) -> None:
        self.lines: list[str] = []  # handed over, the first playing
        self.calls: list[str] = []

    def healthcheck(self) -> tuple[bool, str]:
        return True, "paced"

    @property
    def busy(self) -> bool:
        return bool(self.lines)

    @property
    def waiting(self) -> int:
        return max(0, len(self.lines) - 1)

    def say(self, text: str, **_: object) -> None:
        self.lines.append(text)
        self.calls.append("say:" + text)

    def stop(self, owner: str | None = None) -> None:
        self.lines.clear()
        self.calls.append("stop")

    def drop_waiting(self, owner: str = "coach") -> None:
        self.lines[1:] = []
        self.calls.append("drop")

    def prepare(self, text: str, **_: object) -> None:
        self.calls.append("prepare:" + text)

    def end(self, engine: PanelEngine) -> None:
        """The line playing ends; the next starts (its `on_play`)."""
        self.lines.pop(0)
        if self.lines:
            engine._on_audio_starts(self.lines[0])


def _panel(make_panel: Callable[..., PanelEngine]) -> tuple[PanelEngine, PacedTTS]:
    engine = make_panel()
    assert engine._pace is not None, "follow my pace is the default"
    tts = PacedTTS()
    engine._tts = tts  # type: ignore[assignment]
    engine._position_moved_on("a new game")
    return engine, tts


def test_the_voice_is_kept_one_line_ahead(make_panel: Callable[..., PanelEngine]) -> None:
    engine, tts = _panel(make_panel)
    engine._position_moved_on("the player moved", ply=1, think_s=20.0)
    sink = engine._stream_sink(Trigger.POST_MOVE, 0, None, None)
    assert sink is not None
    for sentence in ("One.", "Two.", "Three."):
        sink(sentence)
    # The first plays, the second is made behind it; the third waits in the plan.
    assert tts.lines == ["One.", "Two."]
    tts.end(engine)  # "Two." plays: "Three." goes to the voice now, to be made meanwhile
    assert tts.lines == ["Two.", "Three."]


def test_the_players_move_finishes_the_sentence_and_drops_what_waits(
    make_panel: Callable[..., PanelEngine],
) -> None:
    engine, tts = _panel(make_panel)
    engine._position_moved_on("the player moved", ply=1, think_s=20.0)
    sink = engine._stream_sink(Trigger.POST_MOVE, 0, None, None)
    assert sink is not None
    sink("One.")
    sink("Two.")
    engine._position_moved_on("the player moved", ply=3, think_s=4.0)
    assert "stop" not in tts.calls, "the sentence playing finishes"
    assert tts.lines == ["One."], "the one waiting is no longer the news"


def test_a_hint_whose_position_is_gone_is_cut(make_panel: Callable[..., PanelEngine]) -> None:
    engine, tts = _panel(make_panel)
    sink = engine._stream_sink(Trigger.STUCK, 0, None, None)
    assert sink is not None
    sink("Only one move holds.")
    assert tts.lines == ["Only one move holds."]
    engine._position_moved_on("the player moved", ply=1, think_s=50.0)
    assert tts.calls[-1] == "stop" and not tts.lines


def test_a_reply_streams_into_the_chat_muted_or_not(make_panel: Callable[..., PanelEngine]) -> None:
    engine, tts = _panel(make_panel)
    engine.set_muted(True)
    sink = engine._stream_sink(Trigger.USER_ASK, 0, None, None)
    assert sink is not None, "muted, the reply is still written sentence by sentence"
    sink("Play Qa8+.")
    assert engine.poll()["state"]["coach_draft"] == "Play Qa8+."
    sink("It deflects the rook.")
    assert engine.poll()["state"]["coach_draft"] == "Play Qa8+. It deflects the rook."
    assert not any(c.startswith("say:") for c in tts.calls), "muted: nothing said"


def test_the_setting_switches_to_the_simple_voice_and_back(
    make_panel: Callable[..., PanelEngine],
) -> None:
    engine, _ = _panel(make_panel)
    engine.set_voice_pace("simple")
    assert engine._pace is None
    engine.set_voice_pace("follow")
    assert engine._pace is not None
