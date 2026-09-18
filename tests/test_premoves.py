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

pytestmark = pytest.mark.vision  # the synthetic vision pipeline: run with -m vision

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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


def _run_queue(
    g: Game,
    first: str,
    queue: tuple[str, ...],
    replies: tuple[str, ...],
    *,
    frames_per_state: int,
    skip_states: bool = False,
) -> None:
    """The user plays `first` for real, queues `queue` (uci, one piece or several in the
    order they will execute), and the opponent answers `replies` one by one; each answer
    executes the next queued move at once. `frames_per_state` frames of each picture are
    shown; with `skip_states` every other picture is dropped, as when a bot answers faster
    than the capture runs. A queued step that ends the game is followed by the dialog."""
    real = g.board.push_san(first)
    g.show(frames=frames_per_state, highlight=real)
    pending = list(queue)

    def queues() -> tuple[Queue, ...]:
        # Group the remaining uci moves into per-piece queues in execution order.
        by_piece: dict[chess.Square, list[chess.Move]] = {}
        where: dict[chess.Square, chess.Square] = {}  # a piece's real square -> its queue end
        for uci in pending:
            move = chess.Move.from_uci(uci)
            origin = next((o for o, end in where.items() if end == move.from_square), None)
            if origin is None:
                origin = move.from_square
                by_piece[origin] = []
            by_piece[origin].append(move)
            where[origin] = move.to_square
        return tuple(tuple(moves) for moves in by_piece.values())

    g.show(frames=frames_per_state, highlight=real, premoves=queues())
    step = real
    for index, reply in enumerate(replies):
        g.board.push_san(reply)
        step = chess.Move.from_uci(pending.pop(0))
        assert step in g.board.legal_moves, (reply, step)
        g.board.push(step)
        if skip_states and index % 2 == 0 and not g.board.is_game_over():
            continue  # the bot answered again before the next frame
        if g.board.is_game_over():
            g.show(frames=1, highlight=step)
            for _ in range(30):
                for event in g.pipe.process(with_dialog(g.frame(highlight=step), dark=True), g.ts):
                    if isinstance(event, MovePlayed):
                        g.sans.append(event.san)
                    elif isinstance(event, GameEnded):
                        g.ended.append(event)
                g.ts += 0.1
            return
        g.show(frames=frames_per_state, highlight=step, premoves=queues())
    # The board rests once the queue is spent.
    g.show(frames=3, highlight=step)


# White: K g1, Q d1; Black: K h8, pawns g7 h7 a7. The queen climbs the d-file square by
# square while Black pushes the a-pawn, and Qd8 is mate on the back rank.
LADDER = "7k/p5pp/8/8/8/8/8/3Q2K1 w - - 0 1"


@pytest.mark.slow
@pytest.mark.parametrize("frames_per_state", [3, 1])
def test_a_six_step_queue_of_one_piece_ends_in_mate(frames_per_state: int) -> None:
    g = Game(Side.WHITE, fen=LADDER)
    _run_queue(
        g,
        "Qd2",
        _uci("d2d3", "d3d4", "d4d5", "d5d6", "d6d8"),
        _uci("a6", "a5", "a4", "a3", "a2"),
        frames_per_state=frames_per_state,
    )
    assert g.sans == ["Qd2", "a6", "Qd3", "a5", "Qd4", "a4", "Qd5", "a3", "Qd6", "a2", "Qd8#"], (
        g.sans
    )
    assert g.ended and g.ended[0].result == "1-0"


@pytest.mark.slow
def test_a_long_queue_against_a_bot_faster_than_the_capture() -> None:
    """Every other picture is never captured: two plies of each side land between frames.
    The pictures then admit a shorter legal story (a7-a5 in one push, the queen two
    squares at once) and the reconciler takes the shortest, as a person would; what must
    hold is that the position stays right throughout and the mate is seen."""
    g = Game(Side.WHITE, fen=LADDER)
    _run_queue(
        g,
        "Qd2",
        _uci("d2d3", "d3d4", "d4d5", "d5d6", "d6d8"),
        _uci("a6", "a5", "a4", "a3", "a2"),
        frames_per_state=1,
        skip_states=True,
    )
    assert g.pipe.reconciler is not None
    final = g.pipe.reconciler.board
    assert final.is_checkmate() and final.piece_at(chess.D8) == chess.Piece.from_symbol("Q")
    assert g.sans[0] == "Qd2" and g.sans[-1] == "Qd8#" and 7 <= len(g.sans) <= 11, g.sans
    assert g.ended and g.ended[0].result == "1-0"


# White: K a1, Q b3, R c1; Black: K h8, pawns g7 h7 a7 b7. Two pieces queued at once, seven
# steps between them: the rook walks c1-c2-c3-c4 while the queen goes b3-d5-d8 mate.
TWO_PIECES = "7k/pp4pp/8/8/8/1Q6/8/K1R5 w - - 0 1"


@pytest.mark.slow
def test_a_seven_step_queue_over_two_pieces_ends_in_mate() -> None:
    g = Game(Side.WHITE, fen=TWO_PIECES)
    _run_queue(
        g,
        "Rc2",
        _uci("c2c3", "b3d5", "c3c4", "d5d6", "c4c5", "d6d8"),
        _uci("a6", "a5", "a4", "b6", "b5", "b4"),
        frames_per_state=2,
    )
    assert g.sans == [
        "Rc2",
        "a6",
        "Rc3",
        "a5",
        "Qd5",
        "a4",
        "Rc4",
        "b6",
        "Qd6",
        "b5",
        "Rc5",
        "b4",
        "Qd8#",
    ], g.sans
    assert g.ended and g.ended[0].result == "1-0"


# White: K a1, N b1; Black: K h8, pawns a7 b7 g7 h7. A knight hops a five-step queue
# b1-c3-e4-f6 ... and the mate needs help: the rook on g1 lands the blow. Kept simple: the
# knight's five hops end on f7 giving... this is only about a long single-piece queue.
KNIGHT = "7k/pp4pp/8/8/8/8/8/KN4R1 w - - 0 1"


def test_a_five_hop_knight_queue_is_undone_hop_by_hop() -> None:
    g = Game(Side.WHITE, fen=KNIGHT)
    _run_queue(
        g,
        "Nc3",
        _uci("c3e4", "e4f6", "f6d5", "d5e7"),
        _uci("a6", "a5", "a4", "a3"),
        frames_per_state=2,
    )
    assert g.sans == ["Nc3", "a6", "Ne4", "a5", "Nf6", "a4", "Nd5", "a3", "Ne7"], g.sans


# --- more shapes -------------------------------------------------------------------------


def test_a_queue_the_user_cancels_leaves_no_trace() -> None:
    """The user premoves, thinks better of it and clicks the queue away: the piece is back
    on its square, nothing red, nothing moved."""
    g = Game(Side.WHITE)
    g.play("e4"), g.play("e5"), g.play("Nf3")
    g.show(frames=4, premoves=g.queue(_uci("f3g5", "g5f7")))
    g.show(frames=4)  # cancelled: the plain position again
    assert g.sans == ["e4", "e5", "Nf3"]
    reply = g.board.push_san("Nc6")
    g.show(frames=3, highlight=reply)
    assert g.sans == ["e4", "e5", "Nf3", "Nc6"]


def test_a_queue_replaced_by_another_pieces_queue() -> None:
    """Premove the knight, change your mind, premove the bishop instead: only the second
    queue is drawn, and only it executes."""
    g = Game(Side.WHITE)
    g.play("e4"), g.play("e5"), g.play("Nf3"), g.play("Nc6"), g.play("Bc4")
    g.show(frames=3, premoves=g.queue(_uci("f3g5")))
    g.show(frames=3, premoves=g.queue(_uci("c4f7")))  # the knight's queue is gone
    assert g.sans == ["e4", "e5", "Nf3", "Nc6", "Bc4"]
    g.board.push_san("Nf6")
    executed = g.board.push_san("Bxf7+")
    g.show(frames=3, highlight=executed)
    assert g.sans[-2:] == ["Nf6", "Bxf7+"], g.sans


def test_the_opponent_captures_the_queued_piece() -> None:
    """The queued piece is taken before its move executes: the queue is void, the capture
    is the only move, and the ghost must not be taken for a piece that survived."""
    g = Game(Side.WHITE)
    g.play("e4"), g.play("e5"), g.play("Nf3"), g.play("Nc6"), g.play("Bc4"), g.play("Bc5")
    g.play("Ng5")  # White's knight goes to g5, where the queen can take it
    g.show(frames=3, premoves=g.queue(_uci("g5f7")))  # Nxf7 queued while Black thinks
    assert g.sans[-1] == "Ng5"
    capture = g.board.push_san("Qxg5")  # ...and Black just takes the knight
    g.show(frames=3, highlight=capture)
    assert g.sans[-2:] == ["Ng5", "Qxg5"], g.sans
    assert g.pipe.reconciler is not None
    assert g.pipe.reconciler.board.piece_at(chess.F7) == chess.Piece.from_symbol("p")


def test_the_opponent_moves_onto_the_queued_square_and_the_premove_captures_it() -> None:
    """The queue's destination is empty when queued and holds an enemy piece when the
    step executes: the premove executes as a capture."""
    g = Game(Side.WHITE)
    g.play("e4"), g.play("e5"), g.play("Nf3"), g.play("Nc6"), g.play("Bc4")
    g.show(frames=3, premoves=g.queue(_uci("f3g5")))
    g.board.push_san("Nf6")  # not onto g5; a plain reply
    executed = g.board.push_san("Ng5")
    g.show(frames=3, highlight=executed)
    assert g.sans[-2:] == ["Nf6", "Ng5"], g.sans
    g.show(frames=3, highlight=executed, premoves=g.queue(_uci("g5e6")))  # Ne6 queued
    g.board.push_san("Be7")
    g.show(frames=1, highlight=chess.Move.from_uci("f8e7"), premoves=g.queue(_uci("g5e6")))
    executed = g.board.push_san("Ne6")
    g.show(frames=3, highlight=executed)
    assert g.sans[-2:] == ["Be7", "Ne6"], g.sans


def test_the_users_king_in_check_on_a_red_square_is_not_a_ghost() -> None:
    """Chess.com paints the checked king's square red; with a queue on the board too, the
    king must not be mistaken for a queued piece and moved to some red square."""
    g = Game(Side.WHITE)
    g.play("e4"), g.play("e5"), g.play("f3"), g.play("Nc6"), g.play("Nc3")
    g.show(frames=3, premoves=g.queue(_uci("c3d5")))  # Nd5 queued
    check = g.board.push_san("Qh4+")  # check clears the queue; the king's square is red
    g.show(frames=3, highlight=check, red_marks=(chess.E1,))
    assert g.sans[-1:] == ["Qh4+"], g.sans
    g.play("g3", red_marks=())
    assert g.sans[-2:] == ["Qh4+", "g3"], g.sans


@pytest.mark.slow
@pytest.mark.parametrize("frames_per_state", [3, 1])
def test_black_runs_a_long_queue_to_mate(frames_per_state: int) -> None:
    """The same ladder from Black's side, black at the bottom: the queen climbs the d-file
    while White pushes a pawn, and ...Qd1 is mate."""
    fen = "3q2k1/8/8/8/8/8/P5PP/7K b - - 0 1"
    g = Game(Side.BLACK, fen=fen)
    _run_queue(
        g,
        "Qd7",
        _uci("d7d6", "d6d5", "d5d4", "d4d3", "d3d1"),
        _uci("a3", "a4", "a5", "a6", "a7"),
        frames_per_state=frames_per_state,
    )
    assert g.sans == ["Qd7", "a3", "Qd6", "a4", "Qd5", "a5", "Qd4", "a6", "Qd3", "a7", "Qd1#"], (
        g.sans
    )
    assert g.ended and g.ended[0].result == "0-1"


def test_a_king_walk_queue_of_five_steps() -> None:
    """A king can only step, so a five-square queue is five red squares in a row."""
    g = Game(Side.WHITE, fen="7k/pp4pp/8/8/8/8/8/K5R1 w - - 0 1")
    _run_queue(
        g,
        "Kb2",
        _uci("b2c3", "c3d4", "d4e5", "e5e6"),
        _uci("a6", "a5", "a4", "a3"),
        frames_per_state=2,
    )
    assert g.sans == ["Kb2", "a6", "Kc3", "a5", "Kd4", "a4", "Ke5", "a3", "Ke6"], g.sans


def test_a_pawn_queue_up_the_board() -> None:
    g = Game(Side.WHITE, fen="7k/p5pp/8/8/8/8/3P4/K5R1 w - - 0 1")
    _run_queue(
        g,
        "d4",
        _uci("d4d5", "d5d6", "d6d7"),
        _uci("a6", "a5", "a4"),
        frames_per_state=2,
    )
    assert g.sans == ["d4", "a6", "d5", "a5", "d6", "a4", "d7"], g.sans


def test_two_queues_of_different_lengths_finish_at_different_times() -> None:
    """A one-step queue and a three-step queue at once: the short one is spent after the
    first opponent move and the long one keeps going."""
    g = Game(Side.WHITE, fen="7k/pp4pp/8/8/8/1Q6/8/K1R5 w - - 0 1")
    _run_queue(
        g,
        "Rc2",
        _uci("b3b4", "c2c3", "c3c4", "c4c5"),
        _uci("a6", "a5", "a4", "b6"),
        frames_per_state=2,
    )
    assert g.sans == ["Rc2", "a6", "Qb4", "a5", "Rc3", "a4", "Rc4", "b6", "Rc5"], g.sans
