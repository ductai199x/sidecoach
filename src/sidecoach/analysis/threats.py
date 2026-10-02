"""What the opponent would do if it were their turn.

A null move hands them the move without changing anything else, so the engine's answer is
exactly the threat the position carries. This is what lets the coach ask "what is your
opponent trying to do?" instead of naming a move, which is the first rung of the hint
ladder in DESIGN.md section 5.5.
"""

from __future__ import annotations

import chess

from sidecoach.analysis.engine import Engine
from sidecoach.analysis.score import game_over
from sidecoach.events import Line


def threat_lines(engine: Engine, board: chess.Board, *, depth: int, multipv: int = 1) -> list[Line]:
    """The opponent's best lines after a null move, from their point of view.

    Empty when passing is not a question that can be asked: the side to move is in check
    (they cannot pass out of it), or the game is over.
    """
    if board.is_check() or game_over(board):
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


# The search for what a quiet best move threatens: enough to see a mate or a piece won.
AIM_DEPTH = 14
# Only a best line at least this good for the side to move is asked what it threatens: a
# quiet move that wins, not one that holds.
AIM_MIN = 0.6


def aim_lines(
    engine: Engine, board: chess.Board, lines: list[Line], *, depth: int = AIM_DEPTH
) -> list[Line]:
    """What the best move threatens, when it is quiet: the side to move's best line after it
    and a null move, from the side to move's view. A quiet move -- no capture, no check, no
    promotion -- that wins has a threat as its point, and the threat is the lesson: "the first
    move is quiet, but it threatens mate" where "their knight on c3 is the target" said
    nothing of how (the 28 wins of the puzzle sample whose first hint was bare, 2026-09-25).
    Empty for a move that is not quiet, a line that does not win, and a move that gives check
    or ends the game (no free move can be handed over then)."""
    if not lines or not lines[0].moves or lines[0].expected_score < AIM_MIN:
        return []
    try:
        first = chess.Move.from_uci(lines[0].moves[0])
    except ValueError:
        return []
    if (
        first not in board.legal_moves
        or board.is_capture(first)
        or board.gives_check(first)
        or first.promotion is not None
    ):
        return []
    after = board.copy(stack=False)
    after.push(first)
    return threat_lines(engine, after, depth=depth)


def threat_moves(engine: Engine, board: chess.Board, *, depth: int, multipv: int = 1) -> list[str]:
    """The first move of each threat line, in UCI."""
    return [line.moves[0] for line in threat_lines(engine, board, depth=depth, multipv=multipv)]
