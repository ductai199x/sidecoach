"""The AI coach's words as links to the board (coach/review_links.py): a move its arrow in the
colour of whose it is, a piece or square its square, a file its squares, a term its meaning --
on the short game of the review words' tests."""

from __future__ import annotations

from typing import Any

import chess

from sidecoach.coach import review_links
from tests.test_review_words import _game


def _links(text: str, ply: int, user: str | None = "black", **kw: Any) -> list[dict[str, Any]]:
    return [p for p in review_links.links(text, _game(), ply, user, **kw) if isinstance(p, dict)]


def test_a_move_draws_its_arrow_in_the_colour_of_whose_it_is() -> None:
    # 2. Nf3 (ply 3) is White's, the student's opponent's; Bc4 was the better move.
    said = _links("Your opponent's Nf3 left Bc4 for later.", 3)
    assert [(p["text"], p["arrow"]) for p in said] == [
        ("Nf3", ["g1", "f3", "threat"]),
        ("Bc4", ["f1", "c4", "best"]),
    ]
    # The student's own move is theirs (blue); a move of no line, read where it is legal.
    assert _links("You answered with Nc6.", 4)[0]["arrow"] == ["b8", "c6", "idea"]
    assert _links("Qh5 was possible too.", 3)[0]["arrow"] == ["d1", "h5", "threat"]
    # Side unknown: every move but the better line's in one colour.
    assert _links("Nf3", 3, None)[0]["arrow"] == ["g1", "f3", "idea"]


def test_pieces_and_squares_light_their_squares_and_a_file_spans() -> None:
    said = _links("Your knight on c6 eyes e5 and the d-file, taking on e4 later.", 4)
    assert [(p["kind"], p["text"], p.get("squares") or p.get("span")) for p in said] == [
        ("piece", "Your knight on c6", ["c6"]),
        ("square", "e5", ["e5"]),
        # A file end to end: a double-ended arrow along it, as notation marks one.
        ("file", "d-file", ["d1", "d8"]),
        # "taking on e4" names no king.
        ("square", "e4", ["e4"]),
    ]
    # A coordinate that is a move of the lines is the move: "with e5" at the pawn's own move.
    assert _links("Nf3 attacks e5.", 3)[1]["kind"] == "square"


def test_a_term_carries_its_meaning_and_the_text_is_kept_whole() -> None:
    text = "A quiet move keeps the initiative; the isolated pawn on d5 is a target."
    pieces = review_links.links(text, _game(), 3, "black")
    assert "".join(p if isinstance(p, str) else p["text"] for p in pieces) == text
    kinds = [(p["kind"], p["text"]) for p in pieces if isinstance(p, dict)]
    # The piece on its square wins over the term inside it.
    assert kinds == [
        ("term", "quiet move"),
        ("term", "initiative"),
        ("piece", "the isolated pawn on d5"),
    ]
    assert _links("It keeps the initiative.", 3)[0]["meaning"].startswith("Making the threats")
    # A term that names a file spans it: one link, its meaning and the file.
    (half_open,) = _links("Rooks belong on the half-open d-file.", 3)
    assert (half_open["kind"], half_open["text"]) == ("term", "half-open d-file")
    assert half_open["span"] == ["d1", "d8"]
    assert "squares" not in _links("A fork wins.", 3, general=True)[0]


def test_a_pawn_structure_lights_its_pawns() -> None:
    """ "doubled c-pawns" lights the pawns, read in the position: whose by the word before it,
    which file by the word itself."""
    board = chess.Board("4k3/8/8/2p5/2P5/8/1PP1P3/4K3 w - - 0 1")
    assert review_links._pawns(board, chess.WHITE, "doubled pawns", None) == {chess.C2, chess.C4}
    assert review_links._pawns(board, chess.WHITE, "isolated pawn", None) == {chess.E2}
    assert review_links._pawns(board, chess.WHITE, "passed pawn", None) == {chess.E2}
    assert review_links._pawns(board, chess.BLACK, "doubled pawns", 2) == set()
    assert review_links._owners("Your opponent's ", "black") == [chess.WHITE]
    assert review_links._owners("your ", "black") == [chess.BLACK]
    assert review_links._owners("the ", "black") == [chess.WHITE, chess.BLACK]
