"""Who can take, who can take back, and what a capture nets (`analysis/exchange.py`): the
positions the 2026-09-23 audit reproduced false sentences from."""

from __future__ import annotations

import chess

from sidecoach.analysis import exchange


def sq(name: str) -> chess.Square:
    return chess.parse_square(name)


def test_a_pinned_piece_cannot_take() -> None:
    # The c3 knight is pinned to the king by the b4 bishop: the e4 knight is not hanging.
    board = chess.Board("4k3/8/8/8/1b2n3/2N5/8/4K3 w - - 0 1")
    assert exchange.capturers(board, chess.WHITE, sq("e4")) == []
    assert not exchange.hanging(board, sq("e4"))
    assert not exchange.can_be_won(board, sq("e4"))


def test_a_pinned_piece_cannot_take_back() -> None:
    # The e4 pawn's only defender, the c3 knight, is pinned: the pawn hangs.
    board = chess.Board("4k3/8/5n2/8/1b2P3/2N5/8/4K3 b - - 0 1")
    assert exchange.defenders(board, chess.WHITE, sq("e4")) == []
    assert exchange.hanging(board, sq("e4"))


def test_the_king_cannot_take_back_on_a_guarded_square() -> None:
    # Scholar's mate: f7 is "defended" by the king, but the queen takes with the bishop
    # still guarding: nothing can take back.
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    assert not exchange.can_retake(board, chess.Move.from_uci("h5f7"))
    assert exchange.hanging(board, sq("f7"))


def test_an_x_ray_attacker_counts_in_the_exchange() -> None:
    # The d4 rook is attacked by the d7 rook with the queen behind it on d8, and defended
    # once by the queen on d1: "attacked once, defended once, so it holds" was said, but
    # Rxd4 Qxd4 Qxd4 wins, so White does not take back and loses the rook.
    board = chess.Board("3q2k1/3r4/8/8/3R4/8/8/3Q2K1 b - - 0 1")
    assert exchange.exchange(board, chess.Move.from_uci("d7d4")) == 5
    assert exchange.can_be_won(board, sq("d4"))


def test_an_exchange_wins_what_it_wins() -> None:
    # A pawn takes a defended knight: nets two.
    board = chess.Board("4k3/8/8/3n4/4P3/8/8/4K3 w - - 0 1")
    assert exchange.exchange(board, chess.Move.from_uci("e4d5")) == 3
    # A queen takes a pawn defended by a pawn: loses eight.
    board = chess.Board("4k3/2p5/3p4/8/8/8/8/3QK3 w - - 0 1")
    assert exchange.exchange(board, chess.Move.from_uci("d1d6")) == -8
    assert exchange.best_capture(board, chess.WHITE, sq("d6")) == 0
