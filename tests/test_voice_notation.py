"""Chess notation is turned into words before it is spoken, and prose is left alone."""

from __future__ import annotations

import pytest

from sidecoach.voice.notation import speak_notation


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        # A piece's capture: "Bxc6 doubles them" in the capitals' test below.
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


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        # A capital, as the model starts a sentence: spelled out letter by letter before
        # ("dee ex ee five", Tai 2026-09-29), now the move or the square it is.
        ("Dxe5 was better.", "d takes e five was better."),
        ("G4 was a miss, costing you 28%.", "g four was a miss, costing you 28%."),
        ("B5 was a mistake", "b five was a mistake"),
        ("Exd8=Q+ wins", "e takes d eight promotes to queen check wins"),
        # ...but a capital B before a capture is the bishop's, as SAN writes it, and an
        # opening's code is no square.
        ("Bxc6 doubles them", "bishop takes c six doubles them"),
        ("the A40 Horwitz Defense", "the A40 Horwitz Defense"),
        # P for a pawn, as the model writes it now and then.
        ("defended by Rf8 and Pg7", "defended by rook f eight and pawn g seven"),
    ],
)
def test_capitals_and_pawn_letters_read_as_the_moves_they_are(text: str, spoken: str) -> None:
    assert speak_notation(text) == spoken


@pytest.mark.parametrize(
    ("text", "voiced"),
    [
        ("Ra4", "rook eh four"),
        ("bxa5", "b takes eh five"),
        ("Nad7", "knight eh d seven"),
        ("the pawn on a2", "the pawn on eh two"),
        ("Axb5!", "eh takes b five!"),
        # The article stays the article, and a file named in prose stays as written.
        ("a plan on the a-file", "a plan on the a-file"),
    ],
)
def test_the_voice_says_the_a_file_as_the_letter(text: str, voiced: str) -> None:
    """The letter a alone is the article to the voice's phonemizer: "rook a four" came back
    from the app's own ear as "rook of fours", "rook eh four" as "rook A4" (2026-09-29). The
    voice gets "eh"; the words the speech correction matches keep "a", as a person says it."""
    assert speak_notation(text, voice=True) == voiced
    assert "eh" not in speak_notation(text)


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("Castle with 0-0 now.", "Castle with castles kingside now."),
        ("0-0-0+ wins", "castles queenside check wins"),
        ("Then e8Q wins.", "Then e eight promotes to queen wins."),
        ("d1N+", "d one promotes to knight check"),
        ("The score was 10-0.", "The score was 10-0."),
    ],
)
def test_castling_with_zeros_and_a_promotion_without_the_sign(text: str, spoken: str) -> None:
    """Voice finding 19: "0-0" was read "zero dash zero", "e8Q" "e eight Q"."""
    assert speak_notation(text) == spoken
