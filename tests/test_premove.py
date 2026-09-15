"""Premove queues: the piece is drawn at the end of the queue while its square is empty."""

import chess

from sidecoach.events import MovePlayed, Region, Resync, Side
from sidecoach.state.reconcile import Reconciler
from sidecoach.vision.pieces import ThemeModel
from sidecoach.vision.squares import BoardGeometry
from tests.synth import synthetic_board

GEOM = BoardGeometry(Region(300, 120, 512, 512), 64.0)
# After 59...Kf7 in the game of 2026-09-14 20:06: white Kc7, pawns a4 b6 f4 g3 h4, Be5.
ENDGAME = "8/2K2k2/1P6/4Bp1p/P4PpP/6P1/8/8 w - - 0 60"


def _reconciler_at(fen: str) -> tuple[ThemeModel, Reconciler, chess.Board]:
    start = synthetic_board()
    model = ThemeModel.from_start_position(start, GEOM, Side.WHITE)
    assert model is not None
    rec = Reconciler(bottom=Side.WHITE, user_side=Side.WHITE, settle_frames=2)
    rec.update(model.classify_frame(start, GEOM), 0.0)
    board = chess.Board(fen)
    rec.board = board.copy()
    rec._base_fen = fen
    for i in range(2):
        rec.update(model.classify_frame(synthetic_board(board=board), GEOM), 0.1 + 0.1 * i)
    return model, rec, board


def _displayed(board: chess.Board, origin: chess.Square, target: chess.Square, symbol: str):
    """The true position with the user's piece drawn at the queue's end instead."""
    shown = chess.Board(None)
    for sq, piece in board.piece_map().items():
        shown.set_piece_at(sq, piece)
    shown.remove_piece_at(origin)
    shown.set_piece_at(target, chess.Piece.from_symbol(symbol))
    return shown


class _Feed:
    def __init__(self, model: ThemeModel, rec: Reconciler) -> None:
        self.model, self.rec, self.ts = model, rec, 1.0

    def __call__(self, shown: chess.Board, frames: int) -> list[str]:
        played: list[str] = []
        for _ in range(frames):
            self.ts += 0.1
            events = self.rec.update(
                self.model.classify_frame(synthetic_board(board=shown), GEOM), self.ts
            )
            assert not [e for e in events if isinstance(e, Resync)]
            played += [e.san for e in events if isinstance(e, MovePlayed)]
        return played


def test_pawn_queue_commits_one_step_per_opponent_move() -> None:
    """The sequence of the game: a5, then a queue a6 a7 (a8=Q) resolved by the bot's king."""
    model, rec, board = _reconciler_at(ENDGAME)
    feed = _Feed(model, rec)

    board.push_san("a5")
    assert feed(board, 2) == ["a5"]

    # Queue a5-a6, a6-a7: the pawn is drawn on a7 and a5 is empty; nothing has happened.
    assert feed(_displayed(board, chess.A5, chess.A7, "P"), 3) == []
    assert rec.board.fen() == board.fen()

    # The bot plays Kg6 and the first queued move executes; the picture keeps the pawn on a7.
    board.push_san("Kg6")
    board.push_san("a6")
    assert feed(_displayed(board, chess.A6, chess.A7, "P"), 3) == ["Kg6", "a6"]

    # The user re-queues the promotion: the queen is drawn on a8, a6 stays empty.
    assert feed(_displayed(board, chess.A6, chess.A8, "Q"), 2) == []

    board.push_san("Kf7")
    board.push_san("a7")
    assert feed(_displayed(board, chess.A7, chess.A8, "Q"), 3) == ["Kf7", "a7"]

    # The last queued move executes and the queue is empty: the picture is the position.
    board.push_san("Kg8")
    board.push_san("a8=Q")
    assert feed(board, 3) == ["Kg8", "a8=Q+"]
    board.push_san("Kf7")
    assert feed(board, 2) == ["Kf7"]
    assert rec.board.fen() == board.fen()


def test_cancelled_queue_snaps_back_without_a_commit() -> None:
    model, rec, board = _reconciler_at(ENDGAME)
    feed = _Feed(model, rec)
    board.push_san("a5")
    assert feed(board, 2) == ["a5"]
    assert feed(_displayed(board, chess.A5, chess.A7, "P"), 4) == []
    assert feed(board, 3) == []  # the queue was cleared: the pawn is back on a5
    assert rec.board.fen() == board.fen()
    board.push_san("Kg6")
    assert feed(board, 2) == ["Kg6"]


def test_pawn_drawn_on_the_last_rank_waits_for_the_promotion_choice() -> None:
    """Before the piece is chosen a pawn sits on a8; the bot's move then cuts the queue."""
    model, rec, board = _reconciler_at(ENDGAME)
    feed = _Feed(model, rec)
    board.push_san("a5")
    assert feed(board, 2) == ["a5"]
    assert feed(_displayed(board, chess.A5, chess.A8, "P"), 3) == []
    board.push_san("Kg6")
    board.push_san("a6")
    assert feed(_displayed(board, chess.A6, chess.A7, "P"), 3) == ["Kg6", "a6"]


def test_single_king_premove_still_waits_for_the_opponent() -> None:
    model, rec, board = _reconciler_at(ENDGAME)
    feed = _Feed(model, rec)
    board.push_san("a5")
    assert feed(board, 2) == ["a5"]
    assert feed(_displayed(board, chess.C7, chess.C6, "K"), 3) == []
    board.push_san("Kg6")
    board.push_san("Kc6")
    assert feed(board, 3) == ["Kg6", "Kc6"]
