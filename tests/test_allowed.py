"""What a mistake allowed (Tai's own games, 2026-09-24): the lesson after a mistake says what
the move let the opponent do -- their best line in the position it left, read by the same
idea reader -- and a win far down a line is not read out move by move."""

from __future__ import annotations

import chess

from sidecoach.coach import idea as idea_of
from sidecoach.coach import idea_words
from sidecoach.coach.coach import focus_template, lesson_template
from sidecoach.coach.facts import describe_tactic, facts_from_analysis
from sidecoach.events import Analysis, Line


def _analysis(fen: str, sans: str, *, score: float) -> Analysis:
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
        lines=[Line(moves=moves, depth=20, expected_score=score)],
        decisive=False,
        only_move=False,
    )


def test_a_blunder_that_drops_a_rook_says_so() -> None:
    # g02, 13.Qxd5??: the lesson named Rxa6 and never the rook Rxa1+ took.
    fen = "1n1qk2r/2pp1pp1/r7/3Qp2p/8/2B2P2/1PP3PP/R3K1NR b KQk - 0 13"
    analysis = _analysis(fen, "Rxa1+ Kf2 Qf6 Bxe5 Qf5 Qe4 Qxe4 fxe4", score=0.92)
    assert idea_words.allowed(chess.Board(fen), analysis, "Qxd5") == (
        "after Qxd5, Rxa1+ wins a rook"
    )


def test_a_move_that_lets_them_trade_into_a_won_ending_says_so() -> None:
    # g05, 18.Qd4: the queens come off and the ending is theirs.
    fen = "r1b3k1/ppp2ppp/3b4/8/3Qq3/P4N2/1BP2PPP/R5K1 b - - 1 18"
    analysis = _analysis(fen, "Qxd4 Bxd4 Bf5 c3 b6 Be5 Rd8 Rd1", score=0.82)
    text = idea_words.allowed(chess.Board(fen), analysis, "Qd4")
    assert text is not None and text.startswith("after Qd4, Qxd4 trades them into a won")
    assert "mixed" not in text


def test_a_quiet_line_allows_nothing_to_say() -> None:
    fen = "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1"
    analysis = _analysis(fen, "Kd7 Kd2 Kd6 Kd3", score=0.5)
    assert idea_words.allowed(chess.Board(fen), analysis, "e2") is None


def test_the_lesson_says_what_the_move_allowed() -> None:
    fen = "r2qk2r/p1p1bppp/2p5/4p3/2b1P3/2N4P/PPP2PP1/R1BQR1K1 w kq - 0 12"
    board = chess.Board(fen)
    facts = facts_from_analysis(board, _analysis(fen, "Qg4 g6 b3 Be6", score=0.6))
    text = lesson_template(
        facts,
        "Qg4",
        "That one cost you: Qg4 was better",
        allowed="after Be3, Qxd1 wins your queen",
    )
    assert "After Be3, Qxd1 wins your queen." in text


def test_a_win_far_down_the_line_is_not_read_out_move_by_move() -> None:
    # g01, 7.Be3 for 7.h4: the knight falls on the seventh move of the line.
    fen = "rnbqkb1r/pppp4/4p1p1/5n2/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 7"
    board = chess.Board(fen)
    line = "h4 d5 g4 Nxh4 Bg5 Be7 Bxh4 Bxh4 g5 Nc6 Rxh4 Rxh4 Nxh4 Qxg5"
    idea = idea_of.read(board, _analysis(fen, line, score=0.9))
    assert idea.goal == "win" and idea.gain_ply > idea_words.LATER_READ
    assert idea_words.why(idea, board, describe_tactic) == (
        f"h4: you win {idea.won}, a few moves on"
    )


def test_the_focus_rung_names_no_stray_feature() -> None:
    # g06, before 9.O-O: "your king is still in the centre and can still castle" when a3
    # was the move and castling the inaccuracy.
    fen = "rnbqkb1r/pppp4/4p1p1/5n2/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 7"
    facts = facts_from_analysis(chess.Board(fen), _analysis(fen, "a3 d5", score=0.5))
    facts = facts.model_copy(
        update={"target": None, "plan": None, "ideas": ["a stray feature elsewhere"]}
    )
    assert "stray" not in focus_template(facts)
