"""Engine lines written by hand: the moves of a SAN line, the boards along it, and the
analysis of a position whose best line it is."""

from __future__ import annotations

from collections.abc import Sequence

import chess

from sidecoach.events import Analysis, Line


def san_moves(board: chess.Board, sans: str) -> list[chess.Move]:
    """The moves of `sans`, SAN separated by spaces, played from `board` (left as it was)."""
    b = board.copy(stack=False)
    moves = []
    for san in sans.split():
        moves.append(b.parse_san(san))
        b.push(moves[-1])
    return moves


def boards_along(board: chess.Board, moves: Sequence[chess.Move]) -> list[chess.Board]:
    """`board`, then the board after each of `moves`."""
    boards = [board.copy(stack=False)]
    for move in moves:
        after = boards[-1].copy(stack=False)
        after.push(move)
        boards.append(after)
    return boards


def analysed(
    fen: str,
    sans: str,
    *,
    score: float = 0.5,
    mate: int | None = None,
    threats: Sequence[str] = (),
    decisive: bool = False,
) -> Analysis:
    """The engine's word on `fen`: one best line, `sans`, at depth 20."""
    moves = [m.uci() for m in san_moves(chess.Board(fen), sans)]
    return Analysis(
        game_id="g",
        fen=fen,
        ply=0,
        lines=[Line(moves=moves, depth=20, expected_score=score, mate=mate)],
        threats=list(threats),
        decisive=decisive,
        only_move=False,
    )
