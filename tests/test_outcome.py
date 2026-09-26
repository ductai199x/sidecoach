"""Where a line leads (`coach/outcome.py`): what it takes and gives, piece by piece."""

from __future__ import annotations

import chess

from sidecoach.coach.outcome import line_outcome


def _uci(board: chess.Board, sans: str) -> list[str]:
    line = board.copy()
    out = []
    for san in sans.split():
        move = line.parse_san(san)
        out.append(move.uci())
        line.push(move)
    return out


def test_the_exchange_is_named_not_counted_in_pawns() -> None:
    """Be6 Kd8 Bxd7 Qxd7 wins a rook for a bishop. Said "the player wins two pawns", the
    model added two pawns to the exchange (the locked test set's audit, 2026-09-26)."""
    board = chess.Board("2k2b1r/ppqr2p1/3p3p/2pB1Q2/3pP3/3P3P/PPP2PP1/3R1RK1 w - - 4 17")
    said = line_outcome(board, _uci(board, "Be6 Kd8 Bxd7 Qxd7"), chess.WHITE)
    assert said.startswith("the player wins the exchange: a rook for a bishop")
    assert "two pawns" not in said


def test_a_plain_win_names_the_pieces_and_a_promotion_keeps_the_sum() -> None:
    board = chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    assert line_outcome(board, _uci(board, "exd5"), chess.WHITE).startswith(
        "the player wins a pawn"
    )
    queening = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    said = line_outcome(queening, _uci(queening, "a8=Q+"), chess.WHITE)
    assert said.startswith("the player wins material worth 8 pawns"), said
