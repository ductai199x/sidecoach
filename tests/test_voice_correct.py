"""A raw transcript is corrected against the legal moves: homophones fixed, spoken moves as SAN."""

from __future__ import annotations

import chess
import pytest

from sidecoach.voice.correct import correct, normalise, spoken_forms

START = chess.Board()
# After 1.e4 f5 2.exf5 g6: White to move; fxg6, d4, Bd3, Nf3 are legal.
AFTER_G6 = chess.Board("rnbqkbnr/ppppp2p/6p1/5P2/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")


@pytest.mark.parametrize(
    ("heard", "fixed"),
    [
        ("Night F3", "knight f three"),
        ("Can I take on D4 with the 9?", "can i take on d four with the knight"),
        ("The upon is passed, should I push it?", "the a pawn is passed should i push it"),
        ("Rooka 1 to D1.", "rook a one to d one"),
        ("Roka 1 to d1", "rook a one to d one"),
        ("Why was E-Tix D5 a mistake?", "why was e takes d five a mistake"),
        ("Why was eTakes D5 a mistake?", "why was e takes d five a mistake"),
        ("Why was Exd Five a mistake", "why was e takes d five a mistake"),
        ("Queen H Five Czech", "queen h five check"),
        ("Castle King's side.", "castle kingside"),
        ("should I push the pawn to e for", "should i push the pawn to e four"),
        ("bishop see six", "bishop c six"),
    ],
)
def test_known_confusions_are_normalised(heard: str, fixed: str) -> None:
    assert normalise(heard) == fixed


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
        ("why was E-Tix D5 a mistake", AFTER_G6, "why was e takes d five a mistake", []),
        ("is f takes g six good", AFTER_G6, "is fxg6 good", ["fxg6"]),
        (
            "what about bishop d three or knight f three",
            AFTER_G6,
            "what about Bd3 or Nf3",
            ["Bd3", "Nf3"],
        ),
        ("hint", START, "hint", []),
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


@pytest.mark.parametrize(
    ("heard", "moves"),
    [
        ("E8 Queen", ["e8=Q"]),
        ("e eight, queen", ["e8=Q"]),
        ("e eight equals queen", ["e8=Q"]),
        ("e eight promotes to queen", ["e8=Q"]),
        ("pawn to e eight", ["e8=Q"]),
        ("e eight", ["e8=Q"]),
        ("e eight knight", ["e8=N+"]),
        ("e eight promotes to rook", ["e8=R"]),
    ],
)
def test_a_promotion_that_names_no_piece_is_a_queen(heard: str, moves: list[str]) -> None:
    """Every promotion shares the forms that name no piece ("e eight", "pawn to e eight"); the
    knight's, generated last, used to take them all, so "e8 queen" came out as e8=N+."""
    board = chess.Board("8/4P1k1/8/8/8/8/5K2/8 w - - 0 1")
    assert correct(heard, board).moves == moves


def test_an_illegal_move_is_left_as_words() -> None:
    # e takes d5 is not legal after 1.e4 f5 2.exf5 g6, so it stays as the user said it.
    result = correct("e takes d five", AFTER_G6)
    assert result.moves == []
    assert result.text == "e takes d five"


def test_without_a_board_only_the_confusions_apply() -> None:
    assert correct("Night F3 Czech", None).text == "knight f three check"
