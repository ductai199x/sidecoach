"""What the opponent would do if it were their turn.

A null move hands them the move without changing anything else, so the engine's answer is
exactly the threat the position carries. This is what lets the coach ask "what is your
opponent trying to do?" instead of naming a move, which is the first rung of the hint
ladder in DESIGN.md section 5.5.
"""

from __future__ import annotations

import chess

from sidecoach.analysis.engine import Engine
from sidecoach.events import Line


def threat_lines(engine: Engine, board: chess.Board, *, depth: int, multipv: int = 1) -> list[Line]:
    """The opponent's best lines after a null move, from their point of view.

    Empty when passing is not a question that can be asked: the side to move is in check
    (they cannot pass out of it), or the game is over.
    """
    if board.is_check() or board.is_game_over(claim_draw=True):
        return []
    passed = board.copy(stack=False)
    passed.push(chess.Move.null())
    # Re-read the position from its own description: a board whose history holds a null move
    # cannot be sent to the engine as a move list, and python-chess warns about it once per
    # call. The position is all the engine needs.
    passed = chess.Board(passed.fen())
    if not passed.is_valid():
        return []
    return engine.analyse(passed, depth=depth, multipv=multipv)


def threat_moves(engine: Engine, board: chess.Board, *, depth: int, multipv: int = 1) -> list[str]:
    """The first move of each threat line, in UCI."""
    return [line.moves[0] for line in threat_lines(engine, board, depth=depth, multipv=multipv)]
