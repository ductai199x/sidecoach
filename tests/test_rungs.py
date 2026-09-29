"""The rungs say only what is so (the master games' notes read beside the three rungs, and
Tai's own games read move by move, 2026-09-24): two knights are "your knights", a pawn
attacked gives way to the move's own idea and the focus rung names no stray feature, a line
is read as far as an ear can follow -- a win a move later as one, a win far down the line not
move by move, a long mate only in part -- the right capture with the other piece is taught as
that, and a forcing capture points at what it takes. (A threat to meet keeps its own reason:
test_threats_to_meet.py; the lesson's "it" of the better move: test_allowed.py.)"""

from __future__ import annotations

import chess
import pytest

from sidecoach.coach import idea as idea_of
from sidecoach.coach import idea_words, plans
from sidecoach.coach.coach import RECAPTURE_RULE, focus_template, lesson_template
from sidecoach.coach.facts import describe_tactic, facts_from_analysis
from tests.lines import analysed


def test_two_knights_are_your_knights() -> None:
    plan = plans.LinePlan(
        "attack", square=chess.G8, pieces=(chess.C3, chess.F3), sans=("Ne4", "Ng5")
    )
    board = chess.Board("6k1/5ppp/8/8/8/2N2N2/5PPP/6K1 w - - 0 1")
    assert plans.why(plan, board) == "Ne4 and Ng5 bring your knights against their king"


def test_the_move_s_own_idea_leads_and_the_focus_rung_names_no_stray_feature() -> None:
    """d4 strikes at the pawn on e5; the pawn ending's point is the passed pawn it makes. The
    second hint never reaches for a feature elsewhere (g06, before 9.O-O: "your king is still
    in the centre and can still castle" when a3 was the move and castling the inaccuracy)."""
    fen = "4k3/8/8/4p3/8/8/3P4/4K3 w - - 0 1"
    facts = facts_from_analysis(chess.Board(fen), analysed(fen, "d4 Kd7"))
    assert facts.plan is not None and facts.plan != plans.THREAT_NAME["win"]
    # Not even with no target and no plan to point at: here, and in g06's middlegame, whose
    # hint asks its own question.
    g06 = "rnbqkb1r/pppp4/4p1p1/5n2/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 7"
    for faced in (facts, facts_from_analysis(chess.Board(g06), analysed(g06, "a3 d5", score=0.5))):
        bare = faced.model_copy(
            update={"target": None, "plan": None, "ideas": ["a stray feature elsewhere"]}
        )
        assert "stray" not in focus_template(bare)


WIN_FIRST = "What has to happen first, before you can take it?"
# A best line read out (`idea_words.why`) and the second hint after it (`idea_words.nudge`).
# (case, FEN, the line in SAN, its score, the mate in moves, the why, the nudge or None: not
# checked.)
READ_OUT: list[tuple[str, str, str, float, int | None, str, str | None]] = [
    # Tai's game g04, 12.Nxd4 for 12.O-O: castling first, and the pawn falls after it.
    (
        "a win a move later",
        "r1bq1rk1/2pnbppp/p2p4/1p1P4/3pP3/1B3P2/PPP1N1PP/R1BQK2R w KQ - 1 12",
        "O-O Nc5 Nxd4 Bd7 c3 Nxb3 axb3 a5",
        0.67,
        None,
        "O-O, and after Nc5, Nxd4; you win a pawn",
        WIN_FIRST,
    ),
    # g01, 7.Be3 for 7.h4: the knight falls on the seventh move of the line.
    (
        "a win far down the line",
        "rnbqkb1r/pppp4/4p1p1/5n2/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 7",
        "h4 d5 g4 Nxh4 Bg5 Be7 Bxh4 Bxh4 g5 Nc6 Rxh4 Rxh4 Nxh4 Qxg5",
        0.9,
        None,
        "h4: you win a knight for two pawns, a few moves on",
        WIN_FIRST,
    ),
    # Tai's game g02, 53.Kc5 for 53.b5: a mate in twelve, sixteen plies of it.
    (
        "a long mate",
        "8/k6n/2K3NP/8/1P4P1/2P5/8/8 w - - 1 53",
        "b5 Ka8 Ne5 Nf8 b6 Ne6 b7+ Ka7 b8=Q+ Kxb8 h7 Nd8+ Kb6 Nf7 Nxf7 Kc8",
        1.0,
        12,
        "b5 forces mate: b5 Ka8 Ne5 Nf8 b6, and mate follows",
        None,
    ),
]


@pytest.mark.parametrize(
    ("case", "fen", "sans", "score", "mate", "why", "nudge"),
    READ_OUT,
    ids=[r[0] for r in READ_OUT],
)
def test_a_line_is_read_as_far_as_an_ear_can_follow(
    case: str, fen: str, sans: str, score: float, mate: int | None, why: str, nudge: str | None
) -> None:
    board = chess.Board(fen)
    idea = idea_of.read(board, analysed(fen, sans, score=score, mate=mate))
    assert idea_words.why(idea, board, describe_tactic) == why, case
    if nudge is not None:
        assert idea_words.nudge(idea) == nudge, case


def test_the_right_capture_with_the_other_piece_is_taught_as_that() -> None:
    # Tai's game g03, 13.Raxd1 for 13.Rexd1: he took the queen back, with the other rook.
    fen = "r3k2r/p1p1bppp/2p5/4p3/2b1P3/2N1B2P/PPP2PP1/R2qR1K1 w kq - 0 13"
    board = chess.Board(fen)
    facts = facts_from_analysis(board, analysed(fen, "Rexd1 Bb4 Na4 Be2", score=0.54))
    text = lesson_template(
        facts, "Rexd1", "That one cost you: Rexd1 was better", board=board, played="a1d1"
    )
    assert text.startswith(
        "That one cost you: Rexd1 was better -- take with the rook on e1, not the one on a1."
    )
    # The recapture's rule, as a plain sentence (f983cd0: no lead-in announces it).
    assert text.endswith(f" {RECAPTURE_RULE[0].upper()}{RECAPTURE_RULE[1:]}."), text


def test_a_forcing_capture_points_at_what_it_takes() -> None:
    # Tai's game g04, 8.d5 for 8.dxe5: the focus was "your king is still in the centre".
    fen = "r1bqk2r/2p1bppp/p1np1n2/1p2p3/3PP3/1BN2N2/PPP2PPP/R1BQK2R w KQkq - 0 8"
    board = chess.Board(fen)
    facts = facts_from_analysis(board, analysed(fen, "dxe5 dxe5 Qxd8+ Bxd8", score=0.53))
    assert facts.target == "their pawn on e5"
