"""The moves besides the best one: real Stockfish analyses (tests/fixtures/alternatives/, six
lines at depth 16) of a quiet position with three moves about equally good and of positions
with one clear best move, read by `coach/alternatives.py`."""

from __future__ import annotations

from pathlib import Path

import chess

from sidecoach.analysis.score import (
    DEFAULT_THRESHOLDS,
    about_as_good,
    clearly_worse,
    graded_good,
    label_move,
)
from sidecoach.coach.alternatives import (
    fine_said,
    good_said,
    worse_moves,
    worse_said,
    worse_text,
)
from sidecoach.coach.facts import facts_from_analysis
from sidecoach.events import Analysis, Label

FIXTURES = Path(__file__).parent / "fixtures" / "alternatives"


def load(name: str) -> tuple[chess.Board, Analysis]:
    analysis = Analysis.model_validate_json((FIXTURES / f"{name}.json").read_text())
    return chess.Board(analysis.fen), analysis


def _label(loss: float) -> Label:
    """The label the review gives a move this far behind the best, all else plain."""
    return label_move(
        loss,
        DEFAULT_THRESHOLDS,
        is_best=loss == 0.0,
        in_book=False,
        decisive=False,
        sacrifice=False,
        before_score=0.5,
        opponent_blundered=False,
    )


def test_a_quiet_position_has_moves_about_as_good_good_ones_and_none_clearly_worse() -> None:
    """Black after 1.e4 e5 2.Nf3 Nc6 3.Bc4. "About equally good" is what the move labels grade
    Best or Excellent, "good" what they grade Good (Tai, 2026-10-02: the grading bands, not
    0.03 of our own): Bc5 and Nf6 are about as good as each other; Be7, 0.029 behind, and d6
    are good, a little behind; h6 is an inaccuracy. Nothing is clearly worse among the six
    lines (nothing the board says of a move, then, is said)."""
    board, analysis = load("italian")
    facts = facts_from_analysis(board, analysis, last_move=None, user=board.turn)
    assert facts.candidates == ["Bc5", "Nf6"]
    assert facts.good == ["Be7", "d6"]
    top = analysis.lines[0].expected_score
    labels = [_label(top - line.expected_score) for line in analysis.lines]
    assert labels[:4] == [Label.BEST, Label.EXCELLENT, Label.GOOD, Label.GOOD]
    assert labels[4] is Label.INACCURACY
    for line, label in zip(analysis.lines, labels, strict=True):
        loss = top - line.expected_score
        assert about_as_good(loss) == (label in (Label.BEST, Label.EXCELLENT))
        assert graded_good(loss) == (label in (Label.BEST, Label.EXCELLENT, Label.GOOD))
    assert worse_moves(board, analysis, board.turn) == []
    assert fine_said(facts.candidates) == "Bc5 and Nf6 are both fine here -- about equally good."
    assert good_said(facts.good) == "Be7 and d6 are good too, a little behind."


def test_clearly_worse_is_a_mistake_or_worse_as_the_labels_grade_it() -> None:
    for loss in (0.0, 0.03, 0.05, 0.0999, 0.10, 0.2, 0.5):
        assert clearly_worse(loss) == (_label(loss) in (Label.MISTAKE, Label.BLUNDER)), loss


def test_one_clear_best_move_is_the_only_fine_one_and_the_others_pass_up_the_mate() -> None:
    """The scholar's mate: Qxf7# against five moves that each pass it up."""
    board, analysis = load("scholar")
    facts = facts_from_analysis(board, analysis, last_move=None, user=board.turn)
    assert facts.candidates == ["Qxf7#"]
    worse = worse_moves(board, analysis, board.turn)
    assert [w.san for w in worse] == ["Qd1", "Qe2"], "the two nearest the best, the tempting ones"
    assert all(w.wrong == "passes up the mate" for w in worse)
    assert all(clearly_worse(w.drop) for w in worse)
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
