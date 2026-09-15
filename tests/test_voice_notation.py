"""Chess notation is turned into words before it is spoken, and prose is left alone."""

from __future__ import annotations

import pytest

from sidecoach.voice.notation import speak_notation


@pytest.mark.parametrize(
    ("san", "spoken"),
    [
        ("e4", "e four"),
        ("Nf3", "knight f three"),
        ("Bxc6", "bishop takes c six"),
        ("exd5", "e takes d five"),
        ("O-O", "castles kingside"),
        ("O-O-O", "castles queenside"),
        ("Qh5+", "queen h five check"),
        ("Qh7#", "queen h seven checkmate"),
        ("O-O-O#", "castles queenside checkmate"),
        ("e8=Q", "e eight promotes to queen"),
        ("exd8=Q+", "e takes d eight promotes to queen check"),
        ("Nbd7", "knight b d seven"),
        ("R1e2", "rook one e two"),
        ("1-0", "white wins"),
        ("0-1", "black wins"),
        ("1/2-1/2", "draw"),
    ],
)
def test_tokens_are_spoken(san: str, spoken: str) -> None:
    assert speak_notation(san) == spoken


def test_notation_inside_prose_is_converted_and_the_rest_kept() -> None:
    line = "The move Nf3 develops and eyes e5; after O-O you are safe."
    assert speak_notation(line) == (
        "The move knight f three develops and eyes e five; after castles kingside you are safe."
    )


def test_a_check_mark_before_punctuation_still_speaks() -> None:
    assert speak_notation("Try Qh5+! now") == "Try queen h five check! now"


def test_ordinary_words_and_numbers_are_not_mangled() -> None:
    # No token is shaped like a move here: 'a' is not a square, 'backup' is a word, 1400 a rating.
    assert speak_notation("a backup plan, rated 1400, is fine") == (
        "a backup plan, rated 1400, is fine"
    )


def test_a_full_move_pair_with_numbers() -> None:
    assert speak_notation("1.e4 e5 2.Nf3 Nc6") == "1.e four e five 2.knight f three knight c six"
