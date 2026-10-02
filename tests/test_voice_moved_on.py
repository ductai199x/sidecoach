"""The position moves on: the voice about the old one stops, and a reply that lands after is
shown but not spoken (Tai, 2026-10-02: "There is no need to drop any replies, the text message
should come out in the same correct order. The only thing that might be good is to interrupt the
tts so it stops its long rant"). Logic on events and on the moved-on flag, never on timing."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import chess

from sidecoach.events import GameStarted, MovePlayed, Resync, Side, Takeback, Trigger
from sidecoach.ui.panel import PanelEngine
from tests.test_voice_ptt_quiet import FakeTTS, _captioned


def _panel(make_panel: Callable[..., PanelEngine]) -> tuple[PanelEngine, FakeTTS]:
    engine = make_panel()
    fake = FakeTTS()
    engine._tts = fake  # type: ignore[assignment]
    engine._state["ply"] = 12
    return engine, fake


def _dispatch(engine: PanelEngine, event: object) -> None:
    """The game event as the engine loop hands it to the panel (`_dispatch_game`)."""
    assessing = SimpleNamespace(submit=lambda event: None)
    engine._dispatch_game(event, assessing, chess, followed=None)  # type: ignore[arg-type]


def _move(by_user: bool, ply: int = 13) -> MovePlayed:
    board = chess.Board()
    move = board.parse_san("e4")
    return MovePlayed(
        game_id="g",
        ply=ply,
        san="e4",
        uci=move.uci(),
        fen_before=board.fen(),
        fen_after=chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").fen(),
        by_user=by_user,
    )


def test_the_players_move_a_takeback_a_reread_and_a_new_game_stop_the_voice(
    make_panel: Callable[..., PanelEngine],
) -> None:
    events = {
        "move": _move(True),
        "takeback": Takeback(game_id="g", plies=1, fen=chess.STARTING_FEN),
        "re-read": Resync(game_id="g", fen=chess.STARTING_FEN, reason="re-read"),
        "new game": GameStarted(game_id="h", user_side=Side.WHITE, start_fen=chess.STARTING_FEN),
    }
    for name, event in events.items():
        engine, fake = _panel(make_panel)
        assert engine._speak("stuck", "A long hint about the knight on f7.", 12) == "speak"
        assert fake.busy
        before = engine._hushed
        _dispatch(engine, event)
        assert fake.stops >= 1 and not fake.busy, name
        assert engine._hushed == before + 1, name


def test_the_opponents_move_alone_does_not_stop_the_voice(
    make_panel: Callable[..., PanelEngine],
) -> None:
    """Tai: the opponent's move alone must not stop the voice mid-sentence."""
    engine, fake = _panel(make_panel)
    assert engine._speak("stuck", "A long hint about the knight on f7.", 12) == "speak"
    _dispatch(engine, _move(False))
    assert fake.stops == 0 and fake.busy


def test_a_reply_asked_for_a_moment_that_is_over_is_shown_not_spoken(
    make_panel: Callable[..., PanelEngine],
) -> None:
    engine, fake = _panel(make_panel)
    gone = {"over": True}
    sink = engine._stream_sink(Trigger.USER_ASK, 12, lambda: gone["over"])
    assert sink is not None
    # Taken, every sentence, so the reply is whole in the record: declined, the coach said a
    # template in its place. Captioned in order. Not spoken.
    assert sink("The knight is loose.") is True
    assert sink("What defends it?") is True
    assert fake.said == []
    assert _captioned(engine) == ["The knight is loose.", "What defends it?"]
    # The moment still holds: spoken as ever.
    gone["over"] = False
    sink = engine._stream_sink(Trigger.USER_ASK, 12, lambda: gone["over"])
    assert sink is not None and sink("Look at f7.") is True
    assert fake.said == ["Look at f7."]
    # Not streamed: a reply that is late for the voice is captioned and said to be stale.
    assert engine._speak("user_ask", "Late word.", 12, moved_on=True) == "stale"
    assert fake.said == ["Look at f7."]
    assert _captioned(engine)[-1] == "Late word."


def test_a_reply_that_began_when_the_position_moves_on_is_cut_and_goes_on_silent(
    make_panel: Callable[..., PanelEngine],
) -> None:
    engine, fake = _panel(make_panel)
    sink = engine._stream_sink(Trigger.USER_ASK, 12, lambda: False)
    assert sink is not None
    assert sink("First, the idea.") is True
    assert fake.said == ["First, the idea."]
    _dispatch(engine, _move(True))  # the player moves: the rest is only text
    assert sink("Then the second sentence.") is True
    assert fake.said == ["First, the idea."] and fake.stops >= 1
    assert _captioned(engine) == ["First, the idea.", "Then the second sentence."]
