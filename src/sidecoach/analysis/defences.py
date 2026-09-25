"""The recaptures a winning line declines, and what refutes them.

A player who sees a capture in the engine's line expects it to be taken back: Qxg3 is a queen
trade to anyone who sees fxg3. When the line's reply is not the recapture, taking back must
lose more, and that is the lesson -- "the pawn on f2 cannot take back: it shields their king".
Without it, "you can win their queen" read as a lie (Tai's chess.com puzzle, 2026-09-24:
1...Qxg3 2.Rfe1, because 2.fxg3 Bc5+ is worse), and a rook that could not take back on b8
because it had to stop a pawn queening on f8 was taught as "a loose piece".

Each of the player's captures in the first PLAYER_MOVES moves of the best line whose reply
is no recapture, while something could take back, has the natural recapture -- the cheapest
piece taking back -- played and searched. It is kept as a line from the root: the line up to
the capture, the recapture, and the engine's best play after it, scored for the player; kept
only when the recapture is no better for the opponent than the line's own reply.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import chess

from sidecoach.events import Line

DEFENCE_DEPTH = 14  # the search after the recapture: enough to see the punishment
PLAYER_MOVES = 3  # the player's first moves of the line looked at
DEFENCES_READ = 2  # recaptures searched per position
# The recapture is kept when it scores no better for the opponent than this below the line's
# own reply (search noise between two depths).
DEFENCE_SLACK = 0.05

_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


class _Engine(Protocol):
    def analyse(
        self, board: chess.Board, *, depth: int | None = None, multipv: int | None = None
    ) -> list[Line]: ...


def declined(board: chess.Board, line: Line) -> list[tuple[int, chess.Move]]:
    """The player's captures the line does not see taken back, as (the capture's index in
    the line, the opponent's natural recapture): the cheapest piece that could take back."""
    out: list[tuple[int, chess.Move]] = []
    at = board.copy(stack=False)
    moves = [chess.Move.from_uci(u) for u in line.moves]
    for i, move in enumerate(moves):
        if i // 2 >= PLAYER_MOVES or move not in at.legal_moves:
            break
        if i % 2 == 0 and at.is_capture(move) and i + 1 < len(moves):
            after = at.copy(stack=False)
            after.push(move)
            reply = moves[i + 1]
            takers = [m for m in after.legal_moves if m.to_square == move.to_square]
            if reply.to_square != move.to_square and takers:
                cheapest = min(
                    takers,
                    key=lambda m: _VALUE.get(after.piece_type_at(m.from_square) or chess.PAWN, 0),
                )
                out.append((i, cheapest))
        at.push(move)
    return out


def defence_lines(
    engine: _Engine, board: chess.Board, lines: Sequence[Line], *, depth: int = DEFENCE_DEPTH
) -> list[Line]:
    """The declined recaptures of the best line, each as a line from the root: the line up to
    the player's capture, the recapture, then the engine's best play, scored for the player."""
    if not lines or not lines[0].moves or board.is_game_over():
        return []
    best = lines[0]
    out: list[Line] = []
    for i, recapture in declined(board, best)[:DEFENCES_READ]:
        at = board.copy(stack=False)
        for uci in best.moves[: i + 1]:
            at.push_uci(uci)
        at.push(recapture)
        if at.is_game_over():
            continue
        replies = engine.analyse(chess.Board(at.fen()), depth=depth, multipv=1)
        if not replies:
            continue
        reply = replies[0]
        line = Line(
            moves=[*best.moves[: i + 1], recapture.uci(), *reply.moves],
            depth=reply.depth,
            expected_score=reply.expected_score,
            mate=reply.mate,
        )
        mates = line.mate is not None and line.mate > 0
        if mates or line.expected_score >= best.expected_score - DEFENCE_SLACK:
            out.append(line)
    return out
