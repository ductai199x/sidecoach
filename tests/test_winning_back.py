"""Taking back is not winning (Tai's puzzles, 2026-09-26: "why is the coaching telling me to
win a knight when I have to trade equal value to it"): what the opponent's last capture took
counts against the line (`idea.exchange_owed`), so a recapture is "you can win back a knight",
a net win is named by what it nets, and a lesson never says a move "allowed" a recapture.

The positions are Lichess puzzles from the sample (the setup move played on the board, as the
coach has it), with the engine's line written out."""

from __future__ import annotations

import chess

from sidecoach.coach import idea as idea_of
from sidecoach.coach import idea_words
from sidecoach.events import Analysis, Line

RUY = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]


def _board(fen: str | None, moves: list[str]) -> chess.Board:
    board = chess.Board(fen) if fen else chess.Board()
    for uci in moves:
        board.push_uci(uci)
    return board


def _analysis(board: chess.Board, sans: str, *, score: float) -> Analysis:
    b = board.copy()
    moves = []
    for san in sans.split():
        move = b.parse_san(san)
        moves.append(move.uci())
        b.push(move)
    return Analysis(
        game_id="g",
        fen=board.fen(),
        ply=0,
        lines=[Line(moves=moves, depth=20, expected_score=score)],
        decisive=False,
        only_move=False,
    )


def _goal(board: chess.Board, sans: str, *, score: float) -> tuple[idea_of.Idea, str | None]:
    idea = idea_of.read(board, _analysis(board, sans, score=score))
    return idea, idea_words.goal(idea)


def test_the_exchange_in_progress_is_what_the_last_captures_took() -> None:
    board = _board(None, [*RUY, "b5c6"])  # 4.Bxc6: Black is owed the knight
    assert idea_of.exchange_owed(board) == {chess.KNIGHT: 1}
    board.push_uci("d7c6")  # 4...dxc6: the trade is done, nobody is owed
    assert idea_of.exchange_owed(board) == {}
    assert idea_of.exchange_owed(_board(None, RUY)) == {}  # the last move took nothing
    # A position set up from a FEN has no moves to read: nothing is assumed.
    assert idea_of.exchange_owed(chess.Board(_board(None, [*RUY, "b5c6"]).fen())) == {}


def test_a_plain_recapture_is_winning_back() -> None:
    idea, goal = _goal(_board(None, [*RUY, "b5c6"]), "dxc6 O-O", score=0.5)
    assert idea.back and goal == "you can win back a bishop"


def test_a_puzzle_recapture_is_winning_back() -> None:
    # Lichess 00tMQ: 20...Nxh5 took the bishop; Qxh5 takes the knight back.
    board = _board("r5rk/p2nq2p/2pppn1b/1p2p1PB/7P/1PN5/PBPRQ3/2K3R1 b - - 0 20", ["f6h5"])
    idea, goal = _goal(board, "Qxh5 Bg7", score=0.7)
    assert idea.back and goal == "you can win back a knight"
    # Lichess 00s16: 10...Rxb8 took the bishop; the check wins the one on b4 back.
    board = _board("rBbqk2r/1p3ppp/p3pn2/8/1b1P4/2N2N1P/PP3PP1/R2QKB1R b KQkq - 0 10", ["a8b8"])
    idea, goal = _goal(board, "Qa4+ b5 Qxb4", score=0.75)
    assert idea.back and goal == "you can win back a bishop"


def test_winning_their_queen_back_after_they_took_yours() -> None:
    # Lichess 00ISm: 29...Bxg5 took the queen; Nxg5+ Kh6 Nxe6 takes the bishop, then theirs.
    board = _board("5r2/4bp1k/2ppq1p1/4p1Q1/4N2P/3P4/1P1R1P2/4K1R1 b - - 0 29", ["e7g5"])
    idea, goal = _goal(board, "Nxg5+ Kh6 Nxe6 fxe6", score=0.6)
    assert idea.back and goal == "you can win back their queen"


def test_a_net_win_is_named_by_what_it_nets() -> None:
    # Lichess 00d6g: 23...Nxe6 took the rook; Bxf5+ g6 Bxe6 takes both knights.
    board = _board("r4n2/1p4pk/2p1R3/p2p1nPp/7P/3P1P1B/PP6/2K4R b - - 0 23", ["f8e6"])
    idea, goal = _goal(board, "Bxf5+ g6 Bxe6 Rf8", score=0.9)
    assert not idea.back and goal == "you can win two knights for a rook"


def test_a_lesson_never_says_a_capture_allowed_the_recapture() -> None:
    board = _board(None, [*RUY, "b5c6"])  # the player's 4.Bxc6, Black to move
    analysis = _analysis(board, "dxc6 O-O", score=0.5)
    assert idea_words.allowed(board, analysis, "Bxc6") is None
