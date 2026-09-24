"""What stands on the board, in the coach's words, and the examination of a piece."""

from __future__ import annotations

import chess

from sidecoach.coach.inventory import (
    batteries,
    forks_in_place,
    inventory_text,
    pin_target,
    skewer_target,
)
from sidecoach.coach.named import analyse_named, pieces_by_words
from sidecoach.events import DisclosureLevel


def test_hanging_pinned_batteries_and_captures_are_read_off_the_board() -> None:
    # Black to move; White's rook on f6 forks the loose pawns on b6 and h6. Black's rook on
    # e3 hits e2 and a3, but the bishop and the pawn are defended and worth less: no fork.
    board = chess.Board("b5k1/8/pp3R1p/8/3p4/P3r3/1P2BKPP/8 b - - 7 39")
    text = inventory_text(board, chess.BLACK)
    assert "The player's hanging: pawn on a6 (attacked by the bishop on e2); pawn on b6" in text
    assert "The opponent's forks: rook on f6 hits the pawn on b6 and the pawn on h6" in text
    assert "The player's forks" not in text
    assert "The player's checks: Rf3+, Rxe2+" in text
    assert "The player's captures: Rxe2+ (the bishop, defended)" in text
    # A queen pinning the f7 pawn to the king, read from White's side.
    board = chess.Board("6k1/2q2ppp/1pb5/2p5/4pN2/1Q2P3/PP1r1PPP/5RK1 w - - 0 23")
    assert pin_target(board, chess.F7) == (chess.B3, chess.G8)
    assert (
        "The opponent's pinned: pawn on f7 to the king on g8 by the queen on b3"
        in inventory_text(board, chess.WHITE)
    )
    # A rook behind a queen on a file is a battery; a skewer is the pin the other way round.
    board = chess.Board("2q4k/5pbp/5n2/1p3P2/2p3P1/P3Q3/1P4P1/4R1K1 b - - 0 32")
    assert batteries(board, chess.WHITE) == [[chess.E1, chess.E3]]
    # (The king on h2, off the rook's rank: in check it could not take at all.)
    board = chess.Board("6k1/8/8/8/8/8/7K/R2q1r2 w - - 0 1")
    assert skewer_target(board, chess.A1) == (chess.D1, chess.F1)
    assert forks_in_place(chess.Board(), chess.WHITE) == []
    assert inventory_text(chess.Board(), chess.WHITE) == (
        "On the board now: nothing hangs, nothing is pinned, no check or capture is on."
    )


def test_a_piece_named_with_its_square_is_examined_without_an_engine() -> None:
    board = chess.Board("rnb1nrk1/pp3pbp/3p2p1/q1pP4/2P5/2NB1N2/PP3PPP/R1BQK2R w KQ - 0 10")
    probes = pieces_by_words("what does the white bishop on d3 do?", board)
    assert [(p.label, p.square) for p in probes] == [("white bishop on d3", chess.D3)]
    (fact,) = analyse_named(
        "is my knight on c3 safe?",
        board,
        None,
        user_colour=chess.WHITE,
        level=DisclosureLevel.QUESTION,
    )
    assert fact.startswith(
        "Engine on my knight on c3, which the player asked about: "
        "the player's knight on c3 stands there"
    )
    assert "it has no legal move" in fact
    assert "it is attacked by the queen on a5 and the bishop on g7" in fact
    assert "it is pinned to the king on e1 by the queen on a5" in fact
    # An empty square: who controls it.
    (fact,) = analyse_named(
        "what about e5?", board, None, user_colour=chess.WHITE, level=DisclosureLevel.QUESTION
    )
    assert fact == (
        "Engine on e5, which the player asked about: e5 is empty; the player controls it with "
        "the knight on f3; the opponent controls it with the pawn on d6 and the bishop on g7."
    )
    # Nothing named, nothing examined; a line needs an engine.
    assert (
        analyse_named(
            "am I winning?", board, None, user_colour=chess.WHITE, level=DisclosureLevel.QUESTION
        )
        == []
    )
    assert (
        analyse_named(
            "what about Nf3?", board, None, user_colour=chess.WHITE, level=DisclosureLevel.QUESTION
        )
        == []
    )
