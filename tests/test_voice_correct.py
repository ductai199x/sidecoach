"""A raw transcript is corrected against the legal moves: homophones fixed, spoken moves as SAN."""

from __future__ import annotations

import chess
import pytest

from sidecoach.voice.correct import correct, normalise, spoken_forms

START = chess.Board()
# After 1.e4 f5 2.exf5 g6: White to move; fxg6, d4, Bd3, Nf3 are legal.
AFTER_G6 = chess.Board("rnbqkbnr/ppppp2p/6p1/5P2/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")


def test_known_confusions_are_normalised() -> None:
    confusions = [
        # "Night" and "Czech": test_without_a_board_only_the_confusions_apply.
        ("Can I take on D4 with the 9?", "can i take on d four with the knight"),
        ("The upon is passed, should I push it?", "the a pawn is passed should i push it"),
        ("Rooka 1 to D1.", "rook a one to d one"),
        ("Roka 1 to d1", "rook a one to d one"),
        ("Why was E-Tix D5 a mistake?", "why was e takes d five a mistake"),
        ("Why was eTakes D5 a mistake?", "why was e takes d five a mistake"),
        ("Why was Exd Five a mistake", "why was e takes d five a mistake"),
        ("Castle King's side.", "castle kingside"),
        ("should I push the pawn to e for", "should i push the pawn to e four"),
        ("bishop see six", "bishop c six"),
    ]
    wrong = [
        (heard, got, fixed) for heard, fixed in confusions if (got := normalise(heard)) != fixed
    ]
    assert not wrong, f"(heard, normalised, wanted): {wrong}"


def test_spoken_forms_cover_the_ways_a_move_is_said() -> None:
    forms = spoken_forms(START)
    assert forms["knight f three"] == "Nf3"
    assert forms["knight to f three"] == "Nf3"
    assert forms["e four"] == "e4"
    assert forms["pawn to e four"] == "e4"
    forms = spoken_forms(AFTER_G6)
    assert forms["f takes g six"] == "fxg6"
    assert forms["pawn takes g six"] == "fxg6"
    assert forms["bishop d three"] == "Bd3"
    # Castling forms only exist when castling is legal.
    assert "castle kingside" not in forms
    castled = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert spoken_forms(castled)["castle kingside"] == "O-O"
    assert spoken_forms(castled)["castles queenside"] == "O-O-O"


@pytest.mark.parametrize(
    ("heard", "board", "text", "moves"),
    [
        ("Night F3", START, "Nf3", ["Nf3"]),
        ("should I play knight to f three here", START, "should i play Nf3 here", ["Nf3"]),
        # exd5 is not legal after 1.e4 f5 2.exf5 g6: left as the user said it.
        ("why was E-Tix D5 a mistake", AFTER_G6, "why was e takes d five a mistake", []),
        (
            "what about bishop d three or knight f three",
            AFTER_G6,
            "what about Bd3 or Nf3",
            ["Bd3", "Nf3"],
        ),
        # Sentences with no move in them are never touched.
        ("what should I play here", START, "what should i play here", []),
        (
            "the a pawn is passed, should I push it",
            START,
            "the a pawn is passed should i push it",
            [],
        ),
    ],
)
def test_spoken_legal_moves_become_san(
    heard: str, board: chess.Board, text: str, moves: list[str]
) -> None:
    result = correct(heard, board)
    assert result.text == text
    assert result.moves == moves


def test_a_promotion_that_names_no_piece_is_a_queen() -> None:
    """Every promotion shares the forms that name no piece ("e eight", "pawn to e eight"); the
    knight's, generated last, used to take them all, so "e8 queen" came out as e8=N+
    (2026-09-23)."""
    board = chess.Board("8/4P1k1/8/8/8/8/5K2/8 w - - 0 1")
    promotions = [
        ("E8 Queen", ["e8=Q"]),
        ("e eight equals queen", ["e8=Q"]),
        ("pawn to e eight", ["e8=Q"]),
        # A piece named is still that piece.
        ("e eight knight", ["e8=N+"]),
        ("e eight promotes to rook", ["e8=R"]),
    ]
    wrong = [
        (heard, got, moves)
        for heard, moves in promotions
        if (got := correct(heard, board).moves) != moves
    ]
    assert not wrong, f"(heard, moves read, wanted): {wrong}"


SCANDINAVIAN = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")


@pytest.mark.parametrize(
    ("heard", "board", "text", "moves"),
    [
        # Only the king can go to f3: the knight the person named stays a knight, in words.
        ("Night to F3", chess.Board("8/4P1k1/8/8/8/8/5K2/8 w - - 0 1"), "knight to f three", []),
        # e4 was played: "pawn to e four" is not h4 because h4 sounds alike.
        ("pawn to e four", SCANDINAVIAN, "pawn to e four", []),
        # A square after "on" is a place, not a move there (exd5 is legal).
        ("is the pawn on d five weak", SCANDINAVIAN, "is the pawn on d five weak", []),
        # Fuzz still mends the words around the piece and the square.
        ("knight takes f three", START, "Nf3", ["Nf3"]),
        ("e takes d five", SCANDINAVIAN, "exd5", ["exd5"]),
    ],
)
def test_a_fuzzy_match_never_swaps_the_piece_or_the_square(
    heard: str, board: chess.Board, text: str, moves: list[str]
) -> None:
    """Close is not enough: the piece and the square the person named must be the move's.
    What matches no legal move stays as words, for the coach to read (Tai, 2026-09-23)."""
    result = correct(heard, board)
    assert (result.text, result.moves) == (text, moves)


def test_without_a_board_only_the_confusions_apply() -> None:
    assert correct("Night F3 Czech", None).text == "knight f three check"
