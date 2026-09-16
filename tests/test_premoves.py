"""Premoves, the way chess.com draws them, in every shape the live games have produced.

While the opponent thinks the user may queue moves. The site tints every square of a queue
red, draws the piece at the end of its queue and leaves its real square empty, and executes
the first queued move the instant the opponent moves -- so the screen is not a position,
and the third live game showed how far that goes: a real move whose settled frame the
capture never caught, then two queens queued at once, then a mate delivered in a second
and a half under the game-over dialog. Each scenario here renders the frames as the site
would and checks that the moves committed are the moves played, in order, and nothing else.
"""

from __future__ import annotations

from collections.abc import Iterable

import chess
import pytest

from sidecoach.events import GameEnded, MovePlayed, Side
from sidecoach.state.pipeline import GamePipeline
from tests.synth import synthetic_board
from tests.test_occlusion import with_dialog

BOTTOMS = [Side.WHITE, Side.BLACK]
Queue = tuple[chess.Move, ...]


class Game:
    """A pipeline fed synthetic frames; `board` is the truth, `sans` what was committed."""

    def __init__(self, bottom: Side, fen: str | None = None) -> None:
        self.bottom = bottom
        self.pipe = GamePipeline()
        self.board = chess.Board()
        self.ts = 0.0
        self.sans: list[str] = []
        self.ended: list[GameEnded] = []
        self.show(frames=2)
        assert self.pipe.reconciler is not None and self.pipe.reconciler.started
        if fen is not None:
            # A mid-game position: play the pieces there through a resync-free path is
            # not always possible, so the reconciler adopts it like a cold start would.
            self.board = chess.Board(fen)
            rec = self.pipe.reconciler
            rec.board = self.board.copy()
            rec._base_fen = fen
            self.show(frames=2)

    def frame(self, **kw: object):
        return synthetic_board(board=self.board, bottom=self.bottom, **kw)  # type: ignore[arg-type]

    def show(self, frames: int = 2, **kw: object) -> None:
        for _ in range(frames):
            for event in self.pipe.process(self.frame(**kw), self.ts):
                if isinstance(event, MovePlayed):
                    self.sans.append(event.san)
                elif isinstance(event, GameEnded):
                    self.ended.append(event)
            self.ts += 0.1

    def play(self, san: str, frames: int = 2, **kw: object) -> chess.Move:
        """A move made on the board, then its frames with the last-move highlight."""
        move = self.board.push_san(san)
        self.show(frames=frames, highlight=move, **kw)
        return move

    def queue(self, *queues: Iterable[str]) -> tuple[Queue, ...]:
        """Premove queues from SAN-like uci strings, checked to be the user's pieces."""
        out: list[Queue] = []
        for spec in queues:
            moves = tuple(chess.Move.from_uci(uci) for uci in spec)
            piece = self.board.piece_at(moves[0].from_square)
            assert piece is not None and piece.color == (self.bottom == Side.WHITE)
            out.append(moves)
        return tuple(out)


def _uci(*ucis: str) -> tuple[str, ...]:
    return ucis


@pytest.mark.parametrize("bottom", BOTTOMS)
def test_a_single_premove_executes_with_the_opponents_move(bottom: Side) -> None:
    g = Game(bottom)
    if bottom == Side.WHITE:
        g.play("e4"), g.play("e5"), g.play("Nf3")
        premove = g.queue(_uci("f3g5"))
        g.show(frames=4, premoves=premove)  # the queue forms; nothing has happened
        assert g.sans == ["e4", "e5", "Nf3"]
        reply = g.board.push_san("Nc6")
        executed = g.board.push_san("Ng5")
        g.show(frames=3, highlight=executed)
        assert g.sans == ["e4", "e5", "Nf3", "Nc6", "Ng5"], g.sans
        del reply
    else:
        g.play("e4"), g.play("e5")
        premove = g.queue(_uci("g8f6"))
        g.show(frames=4, premoves=premove)
        assert g.sans == ["e4", "e5"]
        g.board.push_san("Nf3")
        executed = g.board.push_san("Nf6")
        g.show(frames=3, highlight=executed)
        assert g.sans == ["e4", "e5", "Nf3", "Nf6"], g.sans


@pytest.mark.parametrize("bottom", BOTTOMS)
def test_a_queue_of_two_steps_executes_one_step_per_opponent_move(bottom: Side) -> None:
    g = Game(bottom)
    if bottom == Side.WHITE:
        g.play("e4"), g.play("e5"), g.play("Nf3")
        premove = g.queue(_uci("f3g5", "g5e6"))
        g.show(frames=4, premoves=premove)
        assert g.sans == ["e4", "e5", "Nf3"]
        g.board.push_san("d6")
        executed = g.board.push_san("Ng5")
        # The knight is really on g5 now, but the queue goes on to e6: g5 stays red and
        # empty, e6 shows the knight.
        g.show(frames=3, highlight=executed, premoves=g.queue(_uci("g5e6")))
        assert g.sans == ["e4", "e5", "Nf3", "d6", "Ng5"], g.sans
        g.board.push_san("Be7")
        executed = g.board.push_san("Ne6")
        g.show(frames=3, highlight=executed)
        assert g.sans == ["e4", "e5", "Nf3", "d6", "Ng5", "Be7", "Ne6"], g.sans
    else:
        g.play("e4"), g.play("e5")
        premove = g.queue(_uci("g8f6", "f6g4"))
        g.show(frames=4, premoves=premove)
        g.board.push_san("d4")
        executed = g.board.push_san("Nf6")
        g.show(frames=3, highlight=executed, premoves=g.queue(_uci("f6g4")))
        assert g.sans == ["e4", "e5", "d4", "Nf6"], g.sans
        g.board.push_san("Nc3")
        executed = g.board.push_san("Ng4")
        g.show(frames=3, highlight=executed)
        assert g.sans == ["e4", "e5", "d4", "Nf6", "Nc3", "Ng4"], g.sans


def test_two_pieces_queued_at_once_execute_in_order() -> None:
    """The third live game's shape: two queens premoved, the first executes on the
    opponent's move, the second stays queued, then executes on the next."""
    g = Game(Side.WHITE)
    g.play("e4"), g.play("e5"), g.play("Qh5"), g.play("Nc6"), g.play("Bc4")
    queues = g.queue(_uci("h5h4"), _uci("c4b3"))
    g.show(frames=4, premoves=queues)
    assert g.sans == ["e4", "e5", "Qh5", "Nc6", "Bc4"]
    g.board.push_san("Nf6")
    executed = g.board.push_san("Qh4")
    g.show(frames=3, highlight=executed, premoves=g.queue(_uci("c4b3")))
    assert g.sans == ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qh4"], g.sans
    g.board.push_san("d6")
    executed = g.board.push_san("Bb3")
    g.show(frames=3, highlight=executed)
    assert g.sans[-2:] == ["d6", "Bb3"], g.sans


def test_a_real_move_seen_only_in_flight_before_the_queue_formed_is_recovered() -> None:
    """The user moves and premoves within half a second: the real move's settled frame is
    never captured, only one frame with the piece in flight, then the queue picture with
    the piece's real square red and empty. Undoing the queue recovers the real move."""
    g = Game(Side.WHITE)
    g.play("e4"), g.play("e5"), g.play("Nf3"), g.play("Nc6")
    # Bf1-c4 is played: one frame mid-drag, never a settled one.
    real = g.board.push_san("Bc4")
    g.show(frames=1, highlight=real, dragging=(chess.C4, chess.C4, 0.15, 0.1))
    assert g.sans == ["e4", "e5", "Nf3", "Nc6"]
    # Then the queue: the bishop, really on c4, is drawn at b3 with c4 red and empty.
    g.show(frames=4, highlight=real, premoves=g.queue(_uci("c4b3")))
    assert g.sans == ["e4", "e5", "Nf3", "Nc6", "Bc4"], g.sans
    g.board.push_san("Nf6")
    executed = g.board.push_san("Bb3")
    g.show(frames=3, highlight=executed)
    assert g.sans[-2:] == ["Nf6", "Bb3"], g.sans


def test_a_queued_capture_is_drawn_over_its_victim() -> None:
    g = Game(Side.WHITE)
    g.play("e4"), g.play("e5"), g.play("Qh5"), g.play("Nc6"), g.play("Bc4")
    g.show(frames=4, premoves=g.queue(_uci("h5f7")))  # Qxf7 queued: drawn over the f7 pawn
    assert g.sans == ["e4", "e5", "Qh5", "Nc6", "Bc4"]
    g.board.push_san("Nf6")
    executed = g.board.push_san("Qxf7#")
    g.show(frames=3, highlight=executed)
    assert g.sans[-2:] == ["Nf6", "Qxf7#"], g.sans
    assert g.ended and g.ended[0].result == "1-0"


def test_a_premove_the_opponents_move_makes_illegal_is_cleared() -> None:
    """A check clears the queue: the piece snaps back to its square, nothing executes."""
    g = Game(Side.WHITE)
    g.play("e4"), g.play("e5"), g.play("Bc4"), g.play("Nc6"), g.play("d3")
    g.show(frames=4, premoves=g.queue(_uci("c4b3")))  # Bb3 queued
    assert g.sans == ["e4", "e5", "Bc4", "Nc6", "d3"]
    check = g.board.push_san("Bb4+")  # illegal to answer with Bb3: the queue is cleared
    g.show(frames=3, highlight=check)
    assert g.sans[-1:] == ["Bb4+"], g.sans
    g.play("c3")
    assert g.sans[-2:] == ["Bb4+", "c3"], g.sans


def test_premoves_that_end_in_mate_under_the_dialog_commit_the_real_line() -> None:
    """The third live game's finish, move for move: 65. Qgg3 (seen only in flight) Kc2
    66. Qhh2+ Kd1 67. Qgg1#, both queens premoved, the mate's one clean frame followed
    by the game-over dialog."""
    g = Game(Side.WHITE, fen="8/8/4K3/6Q1/7Q/8/8/1k6 w - - 0 65")
    assert g.sans == []
    real = g.board.push_san("Qgg3")
    g.show(frames=1, highlight=real, dragging=(chess.G3, chess.G3, 0.15, 0.1))
    # Both queens queued: h4-h2 first, g3-g1 second; g3 (real) and h4 (real) red and
    # empty, h2 and g1 red with the queens drawn there.
    g.show(frames=4, highlight=real, premoves=g.queue(_uci("h4h2"), _uci("g3g1")))
    assert g.sans == ["Qgg3"], g.sans
    g.board.push_san("Kc2")
    executed = g.board.push_san("Qhh2+")
    g.show(frames=3, highlight=executed, premoves=g.queue(_uci("g3g1")))
    assert g.sans == ["Qgg3", "Kc2", "Qhh2+"], g.sans
    g.board.push_san("Kd1")
    mate = g.board.push_san("Qgg1#")
    g.show(frames=1, highlight=mate)  # one clean frame...
    assert g.sans == ["Qgg3", "Kc2", "Qhh2+"], g.sans
    for _ in range(30):  # ...then the dialog
        for event in g.pipe.process(with_dialog(g.frame(highlight=mate), dark=True), g.ts):
            if isinstance(event, MovePlayed):
                g.sans.append(event.san)
            elif isinstance(event, GameEnded):
                g.ended.append(event)
        g.ts += 0.1
    assert g.sans == ["Qgg3", "Kc2", "Qhh2+", "Kd1", "Qgg1#"], g.sans
    assert g.ended and g.ended[0].result == "1-0"
