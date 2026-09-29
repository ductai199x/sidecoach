"""Tactics that win something, and threats the opponent prepares (the master games' notes,
2026-09-24): a tactic's capture must leave the player up, and a quiet move of theirs that
would threaten mate or a piece next is a threat to meet when the best move takes it away --
the move's reason, not a plan it also serves."""

from __future__ import annotations

import chess
import pytest

from sidecoach.analysis import motifs
from sidecoach.coach import idea as idea_of
from sidecoach.coach import idea_words
from sidecoach.coach.facts import describe_tactic, facts_from_analysis
from sidecoach.events import MotifTag
from tests.lines import analysed, boards_along, san_moves

# Whether a tactic the first move makes pays off in the line (`motifs.pays_off`). (case, FEN,
# the line in SAN, the tag: name and squares, and whether it wins something.)
PAYS: list[tuple[str, str, str, str, list[str], bool]] = [
    # Smyslov-Tal: Na4 opens the long diagonal onto the bishop on g7, but it takes the bishop
    # on b2 first and the knight takes back: the dark-squared bishops traded.
    (
        "a loose piece traded off",
        "r1bqk2r/pp1pnpbp/2n1p1p1/2p5/2P5/1PN3P1/PB1PPPBP/R2QK1NR w KQkq - 2 7",
        "Na4 Bxb2 Nxb2 d5 e3 O-O",
        "hangingPiece",
        ["g7"],
        False,
    ),
    (
        "a loose piece taken for nothing",
        "4k3/8/8/3n4/8/8/8/R3K3 w - - 0 1",
        "Ra5 Ke7 Rxd5",
        "hangingPiece",
        ["d5"],
        True,
    ),
    # Nf2+ uncovers the queen on d1: Qxd1 Rxd1 Nxd1 wins the rook on the third capture
    # (Lichess 01q4A): the exchange on a square played out to its end.
    (
        "the exchange played out",
        "r7/ppk5/2n3pp/2b1pN1q/3pP1n1/1P4NP/1P4P1/R1BQ1R1K b - - 1 22",
        "Nf2+ Kh2 Qxd1 Rxd1 Nxd1",
        "discoveredAttack",
        ["g4", "h5", "d1"],
        True,
    ),
    # The knight steps off the e-file and the rook bears on the pawn on e7: a pawn, not a
    # piece, and "the rook now bears on the pawn" teaches nothing.
    (
        "a discovered attack on a pawn",
        "6k1/4p3/8/8/4N3/8/8/4R1K1 w - - 0 1",
        "Nc5 Kf7 Rxe7+ Kf6",
        "discoveredAttack",
        ["e4", "e1", "e7"],
        False,
    ),
]


@pytest.mark.parametrize(
    ("case", "fen", "sans", "name", "squares", "pays"), PAYS, ids=[r[0] for r in PAYS]
)
def test_a_tactic_is_named_only_when_the_line_wins_with_it(
    case: str, fen: str, sans: str, name: str, squares: list[str], pays: bool
) -> None:
    board = chess.Board(fen)
    line = san_moves(board, sans)
    tag = MotifTag(name=name, squares=squares, move=line[0].uci())
    assert bool(motifs.pays_off(boards_along(board, line), line, 0, tag)) is pays, case


# Short-Timman, before 11...Nd4: were Black to pass, Rd1 would come, and then Rxd8 wins the
# queen ("After anything else, White seizes the d-file by Rd1", Nunn).
SHORT_TIMMAN = "r1bq1rk1/1pp1ppbp/1nn3p1/p3P3/P7/1B3N1P/1PP1QPP1/RNB2RK1 b - - 0 11"


def test_a_quiet_move_that_prepares_to_win_the_queen_is_a_threat_to_meet() -> None:
    board = chess.Board(SHORT_TIMMAN)
    analysis = analysed(SHORT_TIMMAN, "Nd4 Nxd4 Qxd4 Re1", score=0.39, threats=["f1d1"])
    # Nd4 meets Rd1-Rxd8; that it also puts a knight in the centre is not why: no plan.
    assert facts_from_analysis(board, analysis).plan is None
    idea = idea_of.read(board, analysis)
    assert idea.goal == "defend" and idea.prepared and idea.threatened == "your queen"
    assert idea_words.goal(idea) == (
        "your opponent is preparing to win material: stop it before it comes"
    )
    assert idea_words.target(idea, board) == (
        "their rook on f1, one move from winning your queen on d8"
    )
    assert idea_words.why(idea, board, describe_tactic) == (
        "Nd4 meets their idea: Rd1, and then Rxd8"
    )


def test_a_best_move_that_lets_the_prepared_threat_come_is_no_defence() -> None:
    # Kh8 leaves Rd1 and Rxd8 as strong as before: whatever the move is for, it is not that.
    board = chess.Board(SHORT_TIMMAN)
    idea = idea_of.read(board, analysed(SHORT_TIMMAN, "Kh8 Rd1", score=0.39, threats=["f1d1"]))
    assert idea.goal != "defend"
