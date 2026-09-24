"""What a move does, read off legal moves: a pinned piece attacks and defends nothing off its
line, a king takes nothing the other side still guards, a defence counts only when it saves
the piece, a fork only when it wins something, a pin only when it holds. Most positions are
ones the audit of 2026-09-23 reproduced a false sentence on; the rest are their sound
counterparts, and a check being answered."""

from __future__ import annotations

import chess

from sidecoach.coach.effects import describe_effects


def _effects(fen: str, san: str) -> list[str]:
    board = chess.Board(fen)
    return describe_effects(board, board.parse_san(san), limit=9)


def test_a_pinned_piece_or_a_king_that_cannot_take_attacks_nothing() -> None:
    # Nd2 blocks the check and is pinned there: it does not attack the knight on e4, and it
    # stands where the bishop and the knight both take it, the king unable to take back.
    assert _effects("4k3/8/8/8/1b2n3/8/8/1N2K3 w - - 0 1", "Nd2") == [
        "puts the knight where it is attacked and undefended"
    ]
    # The d5 pawn guards the knight: the king beside it cannot take it.
    assert _effects("4k3/8/8/3p4/4n3/8/6K1/8 w - - 0 1", "Kf3") == []
    # Blocking the check on c5 pins the bishop, so the d6 pawn is not attacked; the rook's
    # line to f7, which the bishop left, is a real discovery.
    assert _effects("4r3/2p2ppp/B1Pp3n/K5q1/4k3/3p4/PPP2BP1/RN1Q1R2 w - - 6 26", "Bc5") == [
        "uncovers an attack on the pawn on f7 by the rook on f1",
        "puts the bishop where it is attacked and undefended",
    ]
    # An unpinned bishop still attacks the knight it could take.
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    assert describe_effects(board, board.parse_san("Bb5")) == ["attacks the knight on c6"]


def test_attacked_is_a_legal_capture_and_defended_a_legal_recapture() -> None:
    # The knight on c6 is pinned to its king: the rook on d4 is not attacked.
    assert _effects("4k3/8/2n5/1B6/8/8/8/3RK3 w - - 0 1", "Rd4") == []
    # A discovered check: gxf5 does not answer it, so the knight is not attacked.
    assert _effects("3k4/8/6p1/8/3N4/8/8/3RK3 w - - 0 1", "Nf5+") == [
        "gives check",
        "uncovers an attack on the king on d8 by the rook on d1",
    ]
    # The c7 pawn, attacked by the bishop, steps to c5 beside the king on b5: the knight on
    # a6 guards c5, so the king cannot take and the pawn is out of attack.
    assert _effects("b4k2/2p3pQ/n3p1p1/BK6/4PP2/8/PP6/R2N4 b - - 0 28", "c5") == [
        "gets the pawn out of attack"
    ]


def test_defends_only_when_the_defence_saves_the_piece() -> None:
    # The knight blocking on d2 is pinned: it cannot take back on e4.
    assert _effects("4k3/8/5n2/8/1b2B3/8/8/1N2K3 w - - 0 1", "Nd2") == []
    # The b4 pawn wins the knight on c3 defended or not.
    assert _effects("4k3/8/8/8/1p6/2N5/8/2B1K3 w - - 0 1", "Bd2") == []
    # Against a knight the bishop's guard holds: Nxc3 Bxc3 is only a trade.
    assert _effects("4k3/8/8/1n6/8/2N5/8/2B1K3 w - - 0 1", "Bd2") == [
        "defends the knight on c3, which was attacked"
    ]


def test_a_fork_needs_two_targets_it_wins_from_a_square_it_keeps() -> None:
    # Two knights the pawns guard: the queen attacks one and forks nothing.
    assert _effects("k7/1p5p/2n3n1/8/8/8/8/4Q1K1 w - - 0 1", "Qe4") == ["attacks the knight on c6"]
    # Rooks are worth more than the knight; the king is always a target.
    assert _effects("r3r1k1/5ppp/8/1N6/8/8/8/6K1 w - - 0 1", "Nc7") == [
        "forks the rook on a8 and the rook on e8"
    ]
    assert _effects("r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1", "Nc7+") == [
        "gives check",
        "forks the king on e8 and the rook on a8",
    ]
    # The same knight fork with a bishop on b8 to take the knight: no fork.
    assert _effects("rb2k3/8/8/1N6/8/8/8/4K3 w - - 0 1", "Nc7+") == [
        "gives check",
        "attacks the king on e8",
        "puts the knight where it is attacked and undefended",
    ]


def test_a_pin_the_pinned_piece_breaks_at_a_profit_is_not_named() -> None:
    # Rxe1 wins the unguarded rook: the "pin" holds nothing.
    assert _effects("4k3/4r3/8/8/8/8/7K/7R w - - 0 1", "Re1") == [
        "attacks the rook on e7",
        "puts the rook where it is attacked and undefended",
        "takes the open e-file",
    ]
    # With the king on f1 guarding e1, Rxe1+ Kxe1 is only a trade: the pin stands.
    assert _effects("4k3/4r3/8/8/8/8/8/R4K2 w - - 0 1", "Re1") == [
        "attacks the rook on e7",
        "pins the rook on e7 to the king",
        "takes the open e-file",
    ]


def test_answering_a_check_does_not_make_every_attack_new() -> None:
    """In check, the only legal captures are of the checker; read that way, the knight on
    c2 "attacked" the bishop on b4 only once the checking queen was gone."""
    fen = "rn2k1nr/2pp1p2/P3p3/7p/1b1P2P1/4P3/PPNN1q1P/R1BQKR2 w Qkq - 0 15"
    assert _effects(fen, "Kxf2") == ["takes the queen on f2"]
    assert _effects(fen, "Rxf2") == [
        "takes the queen on f2",
        "attacks the pawn on f7",
        "gets the rook out of attack",
    ]
