"""The reconciler's record and what its listener is told, at the level of its stories.

The rendered tests (test_reconcile.py) are replayed on the phone from the reconciler's own
calls; these set the stories directly, which no frame can stage on cue, and their 1:1 twin
is the phone's `ReconcileRecordTest`."""

import io

import chess
import chess.pgn
import pytest

from sidecoach.events import (
    Event,
    GameEnded,
    GameStarted,
    MovePlayed,
    Resync,
    Side,
    Takeback,
    TurnPending,
)
from sidecoach.state.pipeline import GamePipeline
from sidecoach.state.reconcile import Reconciler
from sidecoach.state.stories import Story


class Listener:
    """What the coach holds of the side to move (`Coach.on_event`): pending as the last
    TurnPending said, and known again from any event that tells a position whole."""

    def __init__(self) -> None:
        self.pending = False

    def hear(self, events: list[Event]) -> None:
        for event in events:
            if isinstance(event, GameStarted | Takeback | Resync | GameEnded):
                self.pending = False
            if isinstance(event, TurnPending):
                self.pending = event.pending


def _played(*sans: str) -> chess.Board:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


def _reconciler(*sans: str) -> Reconciler:
    rec = Reconciler(bottom=Side.WHITE, user_side=Side.WHITE)
    rec.start()
    rec.board = _played(*sans)
    return rec


def test_a_takeback_played_as_the_stories_part_tells_the_turn_pending_again() -> None:
    """The stories disagree on the side to move and the coach was told so; in one frame the
    record takes back a move and plays the reply every story begins with, and the stories
    still disagree after it. The Takeback tells the coach the turn is known: TurnPending
    must follow it, or the coach speaks for a side to move nobody knows (Tai's first
    criterion; the audit of 2026-10-07)."""
    rec = _reconciler("e4", "e5")
    listener = Listener()
    back = (("back", 1),)
    d5 = chess.Move.from_uci("d7d5")
    nf3 = chess.Move.from_uci("g1f3")
    rec._stories = [Story(head=back, moves=(d5,)), Story(head=back, moves=(d5, nf3))]
    told = rec._tell_turn()
    listener.hear(told)
    assert listener.pending and rec.turn_pending
    stories = list(rec._stories)
    rec._seen = {s.ops[: i + 1]: 5 for s in stories for i in range(len(s.ops))}
    events = rec._decide(stories)
    listener.hear(events)
    assert any(isinstance(e, Takeback) for e in events)
    assert [e.san for e in events if isinstance(e, MovePlayed)] == ["d5"]
    assert rec.turn_pending, "the stories still disagree after the common moves"
    assert listener.pending == rec.turn_pending, [type(e).__name__ for e in events]


def test_the_clock_settling_a_turn_tells_it_pending_again_while_the_stories_disagree() -> None:
    """The clock settles a side to move a re-read left open: the Resync it sends tells the
    coach the turn is known, and the stories, kept, may still disagree on it."""
    rec = Reconciler(bottom=Side.WHITE, user_side=Side.WHITE)
    rec.start_from(chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"), ts=0.0)
    assert rec.turn_unknown and rec.board.turn == chess.WHITE
    listener = Listener()
    king = chess.Move.from_uci("e8d8")
    pawn = chess.Move.from_uci("e2e4")
    rec._stories = [Story(moves=(pawn,)), Story(moves=(pawn, king))]
    listener.hear(rec._tell_turn())
    assert listener.pending
    events = rec.turn_from_clock(chess.BLACK)
    listener.hear(events)
    assert any(isinstance(e, Resync) for e in events)
    assert listener.pending == rec.turn_pending


# What the reconciler remembers of the position it stands on (`reconcile.POSITION_MEMORY`).
MEMORY = (
    "_turn_pending_told",
    "_queue_ts",
    "_resync_candidate",
    "_resync_seen",
    "_unmarked",
    "_unmarked_view",
    "_mark_denied",
    "_prev_obs",
)


@pytest.mark.parametrize("how", ["new game", "position set"])
def test_a_position_set_forgets_what_was_remembered_of_the_last(how: str) -> None:
    """Every position set -- a new game, a position re-read or corrected -- leaves nothing of
    the last position's own memory: the queue drawn on it, the mark's count, a resync's
    candidate and the boards it saw, the turn told pending; and a position set with its side
    to move settled has no turn open (the audit of 2026-10-07)."""
    rec = Reconciler(bottom=Side.WHITE, user_side=Side.WHITE)
    rec.start_from(chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"), ts=0.0)
    assert rec.turn_unknown
    rec._turn_pending_told = True
    rec._queue_ts = 1.0
    rec._resync_candidate = ("x", 2, 0.5)
    rec._resync_seen = [(chess.Board(), frozenset(), "x", 0.5)]
    rec._unmarked = 2
    rec._unmarked_view = ("x", ())
    rec._mark_denied = True
    rec._prev_obs = {chess.E1: None}  # type: ignore[dict-item]
    if how == "new game":
        rec.start()
    else:
        rec._set_position(chess.Board("4k3/8/8/8/8/4P3/8/4K3 b - - 0 1"))
    fresh = Reconciler(bottom=Side.WHITE, user_side=Side.WHITE)
    for name in MEMORY:
        assert getattr(rec, name) == getattr(fresh, name), name
    assert not rec.turn_unknown


def test_a_game_replaced_ends_with_the_mate_its_stories_leave() -> None:
    """The mate under the site's dialog, its story pending, when the next game comes up the
    other way round (the bootstrap) or the board is read again: the game ends with the mate
    played and its result, as the reconciler's own new game ends it (the audit of
    2026-10-07: "*" and no mate)."""
    pipe = GamePipeline()
    rec = _reconciler("f3", "e5", "g4")
    rec._stories = [Story(moves=(chess.Move.from_uci("d8h4"),))]
    pipe.reconciler = rec
    events = pipe._end_in_progress("new game started")
    assert [e.san for e in events if isinstance(e, MovePlayed)] == ["Qh4#"]
    ended = [e for e in events if isinstance(e, GameEnded)]
    assert len(ended) == 1 and ended[0].result == "0-1"
    assert ended[0].pgn == "1. f3 e5 2. g4 Qh4#"


def test_a_board_read_whole_is_the_game_when_it_shows_its_position() -> None:
    """A re-read asked over a game being followed: the game's own position -- the same
    pieces, the side to move the mark names one the stories allow -- is the same game, kept
    (Tai, 2026-10-07); other pieces, or the mark naming the other side, are not."""
    rec = _reconciler("e4", "e5", "Nf3")
    shown = rec.board.copy(stack=False)
    marked = frozenset({chess.G1, chess.F3})
    assert rec.shows_the_game(shown, marked)
    assert rec.shows_the_game(shown), "no mark: the turn the stories have"
    assert not rec.shows_the_game(shown, frozenset({chess.E7, chess.E5})), "Black just moved"
    other = rec.board.copy()
    other.push_san("Nc6")
    assert not rec.shows_the_game(other.copy(stack=False))
    rec.end("1-0", "resignation")
    assert not rec.shows_the_game(shown, marked), "a game over is followed no more"


def test_the_record_of_a_position_is_a_game_from_it() -> None:
    """A game begun from a position, or corrected mid-game to one, is written with the
    position it began from: a PGN that plays out (the audit of 2026-10-07: the moves alone,
    numbered from the middle, with no FEN)."""
    rec = Reconciler(bottom=Side.WHITE, user_side=Side.WHITE)
    start = chess.Board("4k3/8/8/8/8/8/4P3/r3K3 w - - 0 1")  # in check: White to move
    rec.start_from(start, ts=0.0)
    assert not rec.turn_unknown
    rec._commit(chess.Move.from_uci("e1d2"))
    rec._commit(chess.Move.from_uci("e8d8"))
    game = chess.pgn.read_game(io.StringIO(rec.pgn()))
    assert game is not None and not game.errors
    assert game.board().fen() == start.fen()
    assert game.end().board().fen() == rec.board.fen()
    assert _reconciler().pgn() == "", "a game from the start is its moves alone"
