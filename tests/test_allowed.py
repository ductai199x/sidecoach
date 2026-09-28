"""What a mistake allowed (Tai's own games, 2026-09-24): the lesson after a mistake says what
the move let the opponent do -- their best line in the position it left, read by the same
idea reader -- and a win far down a line is not read out move by move."""

from __future__ import annotations

import chess
import pytest

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


# What the move played let the opponent do (`idea_words.allowed`), read off their best line
# in the position it left. (case, FEN (None: the start), the moves played on it, their line
# in SAN, its score, the move played, the words -- ending "..." for their start only -- or
# None, and a word the words must not hold.)
RUY = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
ALLOWED: list[tuple[str, str | None, list[str], str, float, str, str | None, str | None]] = [
    # g02, 13.Qxd5??: the lesson named Rxa6 and never the rook Rxa1+ took.
    (
        "a blunder that drops a rook",
        "1n1qk2r/2pp1pp1/r7/3Qp2p/8/2B2P2/1PP3PP/R3K1NR b KQk - 0 13",
        [],
        "Rxa1+ Kf2 Qf6 Bxe5 Qf5 Qe4 Qxe4 fxe4",
        0.92,
        "Qxd5",
        "after Qxd5, Rxa1+ wins a rook",
        None,
    ),
    # g05, 18.Qd4: the queens come off and the ending is theirs.
    (
        "a trade into a won ending",
        "r1b3k1/ppp2ppp/3b4/8/3Qq3/P4N2/1BP2PPP/R5K1 b - - 1 18",
        [],
        "Qxd4 Bxd4 Bf5 c3 b6 Be5 Rd8 Rd1",
        0.82,
        "Qd4",
        "after Qd4, Qxd4 trades them into a won...",
        "mixed",
    ),
    (
        "a quiet line",
        "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1",
        [],
        "Kd7 Kd2 Kd6 Kd3",
        0.5,
        "e2",
        None,
        None,
    ),
    # The player's 4.Bxc6, Black to move: taking back is not something the capture allowed
    # (Tai's puzzles, 2026-09-26; test_winning_back.py).
    ("a recapture", None, [*RUY, "b5c6"], "dxc6 O-O", 0.5, "Bxc6", None, None),
]


@pytest.mark.parametrize(
    ("case", "fen", "played", "sans", "score", "san", "said", "never"),
    ALLOWED,
    ids=[r[0] for r in ALLOWED],
)
def test_the_lesson_says_what_a_mistake_allowed(
    case: str,
    fen: str | None,
    played: list[str],
    sans: str,
    score: float,
    san: str,
    said: str | None,
    never: str | None,
) -> None:
    board = chess.Board(fen) if fen else chess.Board()
    for uci in played:
        board.push_uci(uci)  # the moves stay on the board, as the coach keeps them
    text = idea_words.allowed(board, _analysis(board.fen(), sans, score=score), san)
    if said is None or not said.endswith("..."):
        assert text == said, case
    else:
        assert text is not None and text.startswith(said[:-3]), f"{case}: {text}"
    if never is not None:
        assert text is not None and never not in text, f"{case}: {text}"


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
