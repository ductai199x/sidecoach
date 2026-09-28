"""Where a line leads (`coach/outcome.py`): what it takes and gives, piece by piece.

What is pinned is the material named and who gains it, not the sentence it is said in:
the first clause of what is said is the material."""

from __future__ import annotations

import re

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


def _gain(board: chess.Board, sans: str) -> tuple[str, str]:
    """What is said of the line, and its first clause: the material."""
    said = line_outcome(board, _uci(board, sans), chess.WHITE)
    return said, said.split(";")[0]


def test_the_exchange_is_named_not_counted_in_pawns() -> None:
    """Be6 Kd8 Bxd7 Qxd7 wins a rook for a bishop. Said "the player wins two pawns", the
    model added two pawns to the exchange (the locked test set's audit, 2026-09-26)."""
    board = chess.Board("2k2b1r/ppqr2p1/3p3p/2pB1Q2/3pP3/3P3P/PPP2PP1/3R1RK1 w - - 4 17")
    said, gain = _gain(board, "Be6 Kd8 Bxd7 Qxd7")
    assert "player" in gain and all(word in gain for word in ("exchange", "rook", "bishop")), said
    assert "two pawns" not in said, said


def test_a_plain_win_names_the_pieces_and_a_promotion_keeps_the_sum() -> None:
    said, gain = _gain(chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1"), "exd5")
    assert "player" in gain and "pawn" in gain, said
    # A pawn made a queen: the pieces no longer add up, the pawns' worth (9 - 1) stands.
    said, gain = _gain(chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"), "a8=Q+")
    assert "player" in gain and re.search(r"\b(8|eight)\b", gain), said
