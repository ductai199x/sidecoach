"""Board marks for a coach message never show more than the words may say.

They are built from the disclosure-filtered facts, level by level: nothing at the question
level, a region tint at the region level, the focus squares and the piece at the piece
level, an arrow at the candidate level, the line at the line level. A message about a
played move gets the played move coloured by how it scored and the better move in green.
"""

from __future__ import annotations

import chess

from sidecoach.coach.annotations import annotations_for, mentions_in
from sidecoach.coach.disclosure import PositionFacts
from sidecoach.events import DisclosureLevel, Trigger

# After 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6: a pin theme on the c6 knight, best 4.Ba4.
FEN = "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
FACTS = PositionFacts(
    fen=FEN,
    side_to_move="white",
    phase="opening",
    theme="pin",
    region="the queenside, around c6",
    piece="your bishop on b5",
    focus=["b5", "c6", "e8"],
    tactic="the knight on c6 is pinned to the king on e8 by the bishop on b5",
    plan="keeping the pressure on",
    ideas=["your bishop on b5 ties down the c6 knight"],
    candidates=["Ba4", "Bxc6"],
    threats=["axb5"],
    line=["Ba4", "Nf6", "O-O", "Be7"],
)
BOARD = chess.Board(FEN)


def kinds(marks: list) -> set[tuple[str, tuple[str, ...], str]]:
    return {(m.kind, tuple(m.squares), m.color) for m in marks}


def test_nothing_is_drawn_at_the_question_level() -> None:
    assert annotations_for(FACTS, DisclosureLevel.QUESTION, BOARD, trigger=Trigger.STUCK) == []
    assert annotations_for(FACTS, DisclosureLevel.THEME, BOARD, trigger=Trigger.STUCK) == []


def test_region_level_tints_the_region_only() -> None:
    marks = annotations_for(FACTS, DisclosureLevel.REGION, BOARD, trigger=Trigger.STUCK)
    assert marks, "the region should be tinted"
    assert all(m.kind == "square" for m in marks), "no arrows or rings below the piece level"
    squares = {m.squares[0] for m in marks}
    assert "c6" in squares  # the anchor named in the region words
    assert "a1" in squares and "e4" not in squares  # queenside files only
    # The ideas are visible from the region level: the squares they name are marked too.
    assert "b5" in squares


def test_piece_level_marks_focus_and_rings_the_piece_but_no_move() -> None:
    marks = annotations_for(FACTS, DisclosureLevel.PIECE, BOARD, trigger=Trigger.STUCK)
    ks = kinds(marks)
    assert ("ring", ("b5",), "focus") in ks
    assert ("square", ("e8",), "focus") in ks
    assert not any(m.kind == "arrow" for m in marks), "the candidate is not disclosed yet"


def test_candidate_level_draws_the_move_and_the_threat() -> None:
    marks = annotations_for(FACTS, DisclosureLevel.CANDIDATE, BOARD, trigger=Trigger.USER_ASK)
    arrows = [(tuple(m.squares), m.color) for m in marks if m.kind == "arrow"]
    assert (("b5", "a4"), "good") in arrows  # Ba4
    assert (("a6", "b5"), "threat") in arrows  # ...axb5, read after a null move
    # But not the whole line.
    assert (("g8", "f6"), "line") not in arrows


def test_line_level_draws_the_line_fading() -> None:
    marks = annotations_for(FACTS, DisclosureLevel.LINE, BOARD, trigger=Trigger.USER_ASK)
    line = [m for m in marks if m.kind == "arrow" and m.color == "line"]
    assert [m.label for m in line] == ["Ba4", "Nf6", "O-O", "Be7"]
    assert line[0].opacity > line[-1].opacity


def test_a_played_move_gets_its_score_colour_and_the_better_move() -> None:
    marks = annotations_for(
        FACTS,
        DisclosureLevel.LINE,
        BOARD,
        trigger=Trigger.POST_MISTAKE,
        played_san="Bxc6",
        severity="mistake",
    )
    arrows = {tuple(m.squares): m.color for m in marks if m.kind == "arrow"}
    assert arrows[("b5", "c6")] == "bad"  # what was played, coloured by severity
    assert arrows[("b5", "a4")] == "good"  # what was better
    # Two arrows and no more: the line behind the better move is not drawn unasked.
    assert len(marks) == 2


def test_a_good_played_move_is_neutral_with_no_better_move_arrow() -> None:
    marks = annotations_for(
        FACTS,
        DisclosureLevel.LINE,
        BOARD,
        trigger=Trigger.POST_MOVE,
        played_san="Ba4",
        severity="none",
    )
    assert [(tuple(m.squares), m.color) for m in marks] == [(("b5", "a4"), "played")]


def test_mentions_link_the_words_to_the_board() -> None:
    marks = mentions_in("Your bishop on b5 pins the knight; Ba4 keeps it. Watch e8.", BOARD)
    ks = kinds(marks)
    assert ("arrow", ("b5", "a4"), "mention") in ks
    assert ("square", ("b5",), "mention") in ks
    assert ("square", ("e8",), "mention") in ks
    # Ordinary words are not squares, and a move is not also a bare square.
    assert all(
        len(s) == 2 and s[0] in "abcdefgh" and s[1] in "12345678" for m in marks for s in m.squares
    )
    assert not any(m.kind == "square" and m.squares == ["a4"] for m in marks)
