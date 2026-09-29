"""Chess notation is turned into words before it is spoken, and prose is left alone."""

from __future__ import annotations

import pytest

from sidecoach.voice.notation import speak_notation


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("Bxc6", "bishop takes c six"),
        ("O-O", "castles kingside"),
        ("O-O-O", "castles queenside"),
        ("Qh7#", "queen h seven checkmate"),
        ("exd8=Q+", "e takes d eight promotes to queen check"),
        ("Nbd7", "knight b d seven"),
        ("R1e2", "rook one e two"),
        ("1-0", "white wins"),
        ("0-1", "black wins"),
        ("1/2-1/2", "draw"),
        pytest.param(
            "The move Nf3 develops and eyes e5; after O-O you are safe.",
            "The move knight f three develops and eyes e five;"
            " after castles kingside you are safe.",
            id="prose",
        ),
        pytest.param("Try Qh5+! now", "Try queen h five check! now", id="check-before-punctuation"),
        # No token is shaped like a move here: 'a' is not a square, 'backup' is a word, 1400 a
        # rating.
        pytest.param(
            "a backup plan, rated 1400, is fine", "a backup plan, rated 1400, is fine", id="no-move"
        ),
        pytest.param(
            "1.e4 e5 2.Nf3 Nc6", "1.e four e five 2.knight f three knight c six", id="move-numbers"
        ),
    ],
)
def test_notation_is_spoken_and_the_prose_around_it_kept(text: str, spoken: str) -> None:
    assert speak_notation(text) == spoken
