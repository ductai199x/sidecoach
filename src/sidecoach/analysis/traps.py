"""The captures that look free and are not.

A player at the board sees a piece to take -- the exchange on its square nets something by the
rules -- and takes it. When the engine's good moves do not include that capture, it is either
merely worse or a trap: after it, the opponent has something much better than the piece.
Capablanca-Marshall, 14.Qf3: the knight on g4 looks free, and 14.hxg4? Qh4 wins for Black --
the annotator's main point, and the lesson a stuck player most needs ("the knight is bait").

The engine's multi-PV lines already score the captures among them; the rest of the tempting
captures, at most TRAPS_READ of the most tempting, are searched on their own. A capture is a
trap when it scores TRAP_DROP or more below the best move, or walks into mate where the best
move does not: in a lost position the scores are all near nothing, and 34.Kxh3 in
Johner-Nimzowitsch, mate in six, scored 0.06 below 34.Bxe4.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import chess

from sidecoach.analysis import exchange
from sidecoach.events import Line

# Below the best move's expected score, a tempting capture is a trap: 14.hxg4 in
# Capablanca-Marshall scores 0.49 to the best move's 0.60 at the search depth here (the
# refutation shows only at depth 22), and it is the annotator's main point.
TRAP_DROP = 0.10
TRAP_DEPTH = 14  # the search after the capture: enough to see the punishment
TRAPS_READ = 2  # the most tempting captures searched per position


class _Engine(Protocol):
    def analyse(
        self, board: chess.Board, *, depth: int | None = None, multipv: int | None = None
    ) -> list[Line]: ...


def tempting(board: chess.Board, lines: Sequence[Line]) -> list[chess.Move]:
    """The captures that look like they win something -- the exchange on the square nets
    material, recaptures played out -- that are not among the good lines, the most tempting
    first."""
    if not lines:
        return []
    good = {line.moves[0] for line in lines if line.moves and not _worse(line, lines[0])}
    found: list[tuple[int, chess.Move]] = []
    for move in board.legal_moves:
        if not board.is_capture(move) or move.uci() in good:
            continue
        gain = exchange.exchange(board, move)
        if gain > 0:
            found.append((gain, move))
    found.sort(key=lambda item: -item[0])
    return [move for _, move in found]


def trap_lines(
    engine: _Engine, board: chess.Board, lines: Sequence[Line], *, depth: int = TRAP_DEPTH
) -> list[Line]:
    """The tempting captures that are traps, each as a line from the player's side: the
    capture, then the engine's best play after it, scored for the player."""
    if not lines or board.is_game_over():
        return []
    scored = {line.moves[0]: line for line in lines if line.moves}
    out: list[Line] = []
    searched = 0
    for move in tempting(board, lines):
        line = scored.get(move.uci())
        if line is None:
            if searched >= TRAPS_READ:
                continue
            searched += 1
            after = board.copy(stack=False)
            after.push(move)
            if after.is_game_over():
                continue
            replies = engine.analyse(chess.Board(after.fen()), depth=depth, multipv=1)
            if not replies:
                continue
            reply = replies[0]
            line = Line(
                moves=[move.uci(), *reply.moves],
                depth=reply.depth,
                expected_score=1.0 - reply.expected_score,
                mate=-reply.mate if reply.mate is not None else None,
            )
        if _worse(line, lines[0]):
            out.append(line)
    return out


def _mated(line: Line) -> bool:
    return line.mate is not None and line.mate < 0


def _worse(line: Line, best: Line) -> bool:
    """A capture's line is a trap's: TRAP_DROP below the best, or mate where the best is not."""
    return best.expected_score - line.expected_score >= TRAP_DROP or (
        _mated(line) and not _mated(best)
    )
