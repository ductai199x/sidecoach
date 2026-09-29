"""Taking back is not winning (Tai's puzzles, 2026-09-26: "why is the coaching telling me to
win a knight when I have to trade equal value to it"): what the opponent's last capture took
counts against the line (`idea.exchange_owed`), so a recapture is "you can win back a knight",
a net win is named by what it nets, and a lesson never says a move "allowed" a recapture
(test_allowed.py).

The positions are Lichess puzzles from the sample (the setup move played on the board, as the
coach has it), with the engine's line written out."""

from __future__ import annotations

import chess
import pytest

from sidecoach.coach import idea as idea_of
from sidecoach.coach import idea_words
from sidecoach.coach.coach import board_after, board_back
from sidecoach.events import GameStarted, MovePlayed, Side
from tests.coaching import scripted_coach
from tests.lines import analysed

RUY = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]


def _moves(ucis: list[str]) -> list[MovePlayed]:
    board = chess.Board()
    out = []
    for ply, uci in enumerate(ucis, 1):
        move = chess.Move.from_uci(uci)
        before, san = board.fen(), board.san(move)
        board.push(move)
        out.append(
            MovePlayed(
                game_id="g",
                ply=ply,
                san=san,
                uci=uci,
                fen_before=before,
                fen_after=board.fen(),
                by_user=ply % 2 == 0,
            )
        )
    return out


def _board(fen: str | None, moves: list[str]) -> chess.Board:
    board = chess.Board(fen) if fen else chess.Board()
    for uci in moves:
        board.push_uci(uci)
    return board


def _goal(board: chess.Board, sans: str, *, score: float) -> tuple[idea_of.Idea, str | None]:
    idea = idea_of.read(board, analysed(board.fen(), sans, score=score))
    return idea, idea_words.goal(idea)


def test_the_exchange_in_progress_is_what_the_last_captures_took() -> None:
    board = _board(None, [*RUY, "b5c6"])  # 4.Bxc6: Black is owed the knight
    assert idea_of.exchange_owed(board) == {chess.KNIGHT: 1}
    board.push_uci("d7c6")  # 4...dxc6: the trade is done, nobody is owed
    assert idea_of.exchange_owed(board) == {}
    assert idea_of.exchange_owed(_board(None, RUY)) == {}  # the last move took nothing
    # A position set up from a FEN has no moves to read: nothing is assumed.
    assert idea_of.exchange_owed(chess.Board(_board(None, [*RUY, "b5c6"]).fen())) == {}


# The goal of a line after the opponent captured (`idea.read`, `idea_words.goal`). (case,
# the FEN, the move played on it, the line in SAN, its score, whether it wins back what was
# taken, the goal in words.) The plain recapture, 4.Bxc6 dxc6, is the live coach's below.
GOALS: list[tuple[str, str, list[str], str, float, bool, str]] = [
    # Lichess 00tMQ: 20...Nxh5 took the bishop; Qxh5 takes the knight back.
    (
        "00tMQ",
        "r5rk/p2nq2p/2pppn1b/1p2p1PB/7P/1PN5/PBPRQ3/2K3R1 b - - 0 20",
        ["f6h5"],
        "Qxh5 Bg7",
        0.7,
        True,
        "you can win back a knight",
    ),
    # Lichess 00s16: 10...Rxb8 took the bishop; the check wins the one on b4 back.
    (
        "00s16",
        "rBbqk2r/1p3ppp/p3pn2/8/1b1P4/2N2N1P/PP3PP1/R2QKB1R b KQkq - 0 10",
        ["a8b8"],
        "Qa4+ b5 Qxb4",
        0.75,
        True,
        "you can win back a bishop",
    ),
    # Lichess 00ISm: 29...Bxg5 took the queen; Nxg5+ Kh6 Nxe6 takes the bishop, then theirs.
    (
        "their queen back after they took yours",
        "5r2/4bp1k/2ppq1p1/4p1Q1/4N2P/3P4/1P1R1P2/4K1R1 b - - 0 29",
        ["e7g5"],
        "Nxg5+ Kh6 Nxe6 fxe6",
        0.6,
        True,
        "you can win back their queen",
    ),
    # Lichess 00d6g: 23...Nxe6 took the rook; Bxf5+ g6 Bxe6 takes both knights: a net win is
    # named by what it nets.
    (
        "a net win",
        "r4n2/1p4pk/2p1R3/p2p1nPp/7P/3P1P1B/PP6/2K4R b - - 0 23",
        ["f8e6"],
        "Bxf5+ g6 Bxe6 Rf8",
        0.9,
        False,
        "you can win two knights for a rook",
    ),
]


@pytest.mark.parametrize(
    ("case", "fen", "played", "sans", "score", "back", "words"), GOALS, ids=[r[0] for r in GOALS]
)
def test_taking_back_is_winning_back(
    case: str, fen: str, played: list[str], sans: str, score: float, back: bool, words: str
) -> None:
    idea, goal = _goal(_board(fen, played), sans, score=score)
    assert idea.back is back and goal == words, f"{case}: {goal}"


def test_the_live_coach_keeps_the_moves_that_led_to_a_position() -> None:
    """The coach built each position from the move's FEN, so the exchange in progress was
    never seen live: the fix reached only the benches (the Android port's review, 2026-09-26).
    After 4.Bxc6 the plain recapture dxc6 is "you can win back a bishop"."""
    coach, _ = scripted_coach([])
    start = chess.Board().fen()
    coach.on_event(GameStarted(game_id="g", user_side=Side.BLACK, start_fen=start), 0.0)
    for i, event in enumerate(_moves([*RUY, "b5c6"])):
        coach.on_event(event, float(i))
    position = coach.positions[7]
    assert [m.uci() for m in position.board.move_stack] == [*RUY, "b5c6"]
    assert idea_of.exchange_owed(position.board) == {chess.KNIGHT: 1}
    analysis = analysed(position.board.fen(), "dxc6 O-O", score=0.5)
    coach.on_event(analysis.model_copy(update={"ply": 7}), 8.0)
    assert position.facts is not None and position.facts.goal == "you can win back a bishop"


def test_a_board_keeps_its_moves_only_when_the_event_agrees_with_it() -> None:
    first, second = _moves(["e2e4", "e7e5"])
    board = board_after(chess.Board(), first)
    assert [m.uci() for m in board.move_stack] == ["e2e4"]
    # Played from somewhere else than the board: from the position the event gives.
    elsewhere = board_after(chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1"), second)
    assert [m.uci() for m in elsewhere.move_stack] == ["e7e5"]
    # A FEN the board does not reach (the counters differ): the event's own, no moves.
    odd = second.model_copy(update={"fen_after": second.fen_after.replace(" 0 2", " 0 9")})
    assert (
        board_after(board, odd).move_stack == [] and board_after(board, odd).fen() == odd.fen_after
    )


def test_a_takeback_keeps_the_moves_before_it() -> None:
    board = chess.Board()
    for uci in RUY:
        board.push_uci(uci)
    back = board_back(board, _moves(RUY)[1].fen_after)
    assert [m.uci() for m in back.move_stack] == RUY[:2]
    assert board_back(board, "8/8/8/8/8/8/8/K6k w - - 0 1").move_stack == []
