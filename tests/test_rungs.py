"""The rungs say only what is so (the master games' notes read beside the three rungs, and
Tai's own games read move by move, 2026-09-24): two knights are "your knights", a threat to
meet keeps its own reason, a pawn attacked gives way to the move's own idea, the focus rung
points at the plan, a win that comes a move later is read as one, the right capture with the
other piece is taught as that, and a long mate is read only as far as an ear can follow."""

from __future__ import annotations

import chess

from sidecoach.analysis import positional
from sidecoach.coach import idea as idea_of
from sidecoach.coach import idea_words, plans
from sidecoach.coach.coach import focus_template, lesson_template
from sidecoach.coach.facts import describe_tactic, facts_from_analysis
from sidecoach.events import Analysis, Line


def _analysis(
    fen: str,
    sans: str,
    *,
    threats: list[str] | None = None,
    score: float = 0.5,
    mate: int | None = None,
) -> Analysis:
    board = chess.Board(fen)
    moves = []
    for san in sans.split():
        move = board.parse_san(san)
        moves.append(move.uci())
        board.push(move)
    return Analysis(
        game_id="g",
        fen=fen,
        ply=0,
        lines=[Line(moves=moves, depth=20, expected_score=score, mate=mate)],
        threats=threats or [],
        decisive=False,
        only_move=False,
    )


def test_two_knights_are_your_knights() -> None:
    plan = plans.LinePlan(
        "attack", square=chess.G8, pieces=(chess.C3, chess.F3), sans=("Ne4", "Ng5")
    )
    board = chess.Board("6k1/5ppp/8/8/8/2N2N2/5PPP/6K1 w - - 0 1")
    assert plans.why(plan, board) == "Ne4 and Ng5 bring your knights against their king"


def test_a_threat_to_meet_keeps_its_own_reason() -> None:
    # Short-Timman: Nd4 meets Rd1-Rxd8; that it also puts a knight in the centre is not why.
    fen = "r1bq1rk1/1pp1ppbp/1nn3p1/p3P3/P7/1B3N1P/1PP1QPP1/RNB2RK1 b - - 0 11"
    facts = facts_from_analysis(
        chess.Board(fen), _analysis(fen, "Nd4 Nxd4 Qxd4 Re1", threats=["f1d1"])
    )
    assert facts.goal is not None and facts.goal.startswith("your opponent is preparing")
    assert facts.plan is None


def test_a_pawn_attacked_gives_way_to_the_move_s_own_idea() -> None:
    # d4 strikes at the pawn on e5; the pawn ending's point is the passed pawn it makes.
    fen = "4k3/8/8/4p3/8/8/3P4/4K3 w - - 0 1"
    facts = facts_from_analysis(chess.Board(fen), _analysis(fen, "d4 Kd7"))
    assert facts.plan != plans.THREAT_NAME["win"]
    assert facts.plan is not None


def test_the_focus_rung_points_at_the_plan_not_at_a_stray_feature() -> None:
    fen = "4k3/8/8/4p3/8/8/3P4/4K3 w - - 0 1"
    facts = facts_from_analysis(chess.Board(fen), _analysis(fen, "d4 Kd7"))
    facts = facts.model_copy(update={"target": None, "ideas": ["a stray feature elsewhere"]})
    assert "stray" not in focus_template(facts)


def test_a_capture_away_from_the_centre_is_not_the_fight_for_it() -> None:
    board = chess.Board("rnbqkbnr/ppppp1pp/8/5p2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    assert not positional._is_central_break(board, chess.Move.from_uci("e4f5"), chess.WHITE)
    board = chess.Board("4k3/8/8/4p3/5P2/8/8/4K3 w - - 0 1")
    assert positional._is_central_break(board, chess.Move.from_uci("f4e5"), chess.WHITE)


def test_a_promotion_is_praised_plainly() -> None:
    # c8=Q was "prophylaxis: stop their plan before pushing your own" (Tai's game g02).
    board = chess.Board("8/2P5/8/8/8/k7/8/K7 w - - 0 1")
    move = chess.Move.from_uci("c7c8q")
    assert positional.best_move_idea(board, move, chess.WHITE, phase="endgame") is None


def test_a_win_that_comes_a_move_later_is_read_as_one() -> None:
    # Tai's game g04, 12.Nxd4 for 12.O-O: castling first, and the pawn falls after it.
    fen = "r1bq1rk1/2pnbppp/p2p4/1p1P4/3pP3/1B3P2/PPP1N1PP/R1BQK2R w KQ - 1 12"
    board = chess.Board(fen)
    idea = idea_of.read(board, _analysis(fen, "O-O Nc5 Nxd4 Bd7 c3 Nxb3 axb3 a5", score=0.67))
    assert idea.goal == "win" and idea.gain_ply == 2
    assert idea_words.why(idea, board, describe_tactic) == (
        "O-O, and after Nc5, Nxd4; you win a pawn"
    )
    assert idea_words.nudge(idea) == "What has to happen first, before you can take it?"


def test_the_right_capture_with_the_other_piece_is_taught_as_that() -> None:
    # Tai's game g03, 13.Raxd1 for 13.Rexd1: he took the queen back, with the other rook.
    fen = "r3k2r/p1p1bppp/2p5/4p3/2b1P3/2N1B2P/PPP2PP1/R2qR1K1 w kq - 0 13"
    board = chess.Board(fen)
    facts = facts_from_analysis(board, _analysis(fen, "Rexd1 Bb4 Na4 Be2", score=0.54))
    text = lesson_template(
        facts, "Rexd1", "That one cost you: Rexd1 was better", board=board, played="a1d1"
    )
    assert text == (
        "That one cost you: Rexd1 was better -- take with the rook on e1, not the one on a1. "
        "Keep this one: when two pieces can take, ask which you want to keep where it is."
    )


def test_a_lesson_says_it_of_the_better_move() -> None:
    # Tai's game g03, 12.Be3 for 12.Qg4: "Qg4 was better -- goes after..." lacked its "it".
    fen = "r2qk2r/p1p1bppp/2p5/4p3/2b1P3/2N4P/PPP2PP1/R1BQR1K1 w kq - 0 12"
    board = chess.Board(fen)
    facts = facts_from_analysis(board, _analysis(fen, "Qg4 g6 b3 Be6", score=0.6))
    text = lesson_template(facts, "Qg4", "That one cost you: Qg4 was better")
    assert text.startswith("That one cost you: Qg4 was better -- it goes after their pawn on g7")


def test_a_long_mate_is_read_as_far_as_an_ear_can_follow() -> None:
    # Tai's game g02, 53.Kc5 for 53.b5: a mate in twelve, sixteen plies of it.
    fen = "8/k6n/2K3NP/8/1P4P1/2P5/8/8 w - - 1 53"
    board = chess.Board(fen)
    line = "b5 Ka8 Ne5 Nf8 b6 Ne6 b7+ Ka7 b8=Q+ Kxb8 h7 Nd8+ Kb6 Nf7 Nxf7 Kc8"
    idea = idea_of.read(board, _analysis(fen, line, score=1.0, mate=12))
    assert idea_words.why(idea, board, describe_tactic) == (
        "b5 forces mate: b5 Ka8 Ne5 Nf8 b6, and mate follows"
    )


def test_a_forcing_capture_points_at_what_it_takes() -> None:
    # Tai's game g04, 8.d5 for 8.dxe5: the focus was "your king is still in the centre".
    fen = "r1bqk2r/2p1bppp/p1np1n2/1p2p3/3PP3/1BN2N2/PPP2PPP/R1BQK2R w KQkq - 0 8"
    board = chess.Board(fen)
    facts = facts_from_analysis(board, _analysis(fen, "dxe5 dxe5 Qxd8+ Bxd8", score=0.53))
    assert facts.target == "their pawn on e5"
