"""The moves besides the best one: real Stockfish analyses (tests/fixtures/alternatives/, six
lines at depth 16) of a quiet position with three moves about equally good and of positions
with one clear best move, read by `coach/alternatives.py`."""

from __future__ import annotations

from pathlib import Path

import chess

from sidecoach.coach.alternatives import (
    WORSE_MARGIN,
    fine_said,
    worse_moves,
    worse_said,
    worse_text,
)
from sidecoach.coach.facts import CANDIDATE_MARGIN, facts_from_analysis
from sidecoach.events import Analysis

FIXTURES = Path(__file__).parent / "fixtures" / "alternatives"


def load(name: str) -> tuple[chess.Board, Analysis]:
    analysis = Analysis.model_validate_json((FIXTURES / f"{name}.json").read_text())
    return chess.Board(analysis.fen), analysis


def test_a_quiet_position_has_several_moves_about_as_good_and_none_clearly_worse() -> None:
    """Black after 1.e4 e5 2.Nf3 Nc6 3.Bc4: Bc5, Nf6 and Be7 are within a few hundredths of
    the best line's score. The facts name all three, and nothing is clearly worse among the
    six lines (nothing the board says of a move, then, is said)."""
    board, analysis = load("italian")
    facts = facts_from_analysis(board, analysis, last_move=None, user=board.turn)
    assert facts.candidates == ["Bc5", "Nf6", "Be7"]
    top = analysis.lines[0].expected_score
    assert all(top - line.expected_score <= CANDIDATE_MARGIN for line in analysis.lines[:3])
    assert top - analysis.lines[3].expected_score > CANDIDATE_MARGIN, "the fourth is not fine"
    assert worse_moves(board, analysis, board.turn) == []
    assert (
        fine_said(facts.candidates) == "Bc5, Nf6 and Be7 are all fine here -- about equally good."
    )


def test_one_clear_best_move_is_the_only_fine_one_and_the_others_pass_up_the_mate() -> None:
    """The scholar's mate: Qxf7# against five moves that each pass it up."""
    board, analysis = load("scholar")
    facts = facts_from_analysis(board, analysis, last_move=None, user=board.turn)
    assert facts.candidates == ["Qxf7#"]
    worse = worse_moves(board, analysis, board.turn)
    assert [w.san for w in worse] == ["Qd1", "Qe2"], "the two nearest the best, the tempting ones"
    assert all(w.wrong == "passes up the mate" for w in worse)
    assert all(w.drop >= WORSE_MARGIN for w in worse)
    assert worse_text(worse) == "Qd1 passes up the mate; Qe2 passes up the mate"
    assert worse_said(worse) == (
        "Qd1 is clearly worse: it passes up the mate. Qe2 is clearly worse: it passes up the mate."
    )
    assert fine_said(facts.candidates) == "Qxf7# is the move here."


def test_a_move_that_misses_a_won_queen_says_so_and_nothing_a_cut_off_line_would() -> None:
    """Black's queen has taken on h4 and hangs to Nxh4. Each move clearly behind passes up
    winning the queen; a line cut off in the middle of an exchange is not read for what it
    gives up."""
    board, analysis = load("queen_hangs")
    worse = worse_moves(board, analysis, board.turn, limit=6)
    assert worse and worse[0].san == "Nc3"
    assert all(w.wrong == "passes up winning a queen and a pawn" for w in worse)
    assert analysis.lines[0].moves[0] == "f3h4"


def test_nothing_is_said_when_it_is_not_the_players_move() -> None:
    board, analysis = load("scholar")
    assert worse_moves(board, analysis, chess.BLACK) == []
