"""Expected score, the loss a move costs, and what makes a moment decisive.

Everything here is in expected score, never centipawns: 0.5 is a dead draw, 1.0 is a win
for the side to move. A 50-centipawn slip at +8 costs nothing and the same slip at level
costs the game, so a threshold in centipawns says nothing about how much a move mattered.

Pure functions over engine lines, so they are cheap to test and carry no engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import chess

from sidecoach.config import CoachSettings
from sidecoach.events import Line, Severity

DRAW = 0.5


@dataclass(frozen=True)
class Thresholds:
    """How much loss earns a name, and what makes a position a decisive moment."""

    inaccuracy: float = 0.05
    mistake: float = 0.10
    blunder: float = 0.20
    decisive_gap: float = 0.10
    # An only move: everything else throws away this much. Wider than a decisive moment,
    # since "there is one move here" is a stronger claim than "this move matters".
    only_move_gap: float = 0.30

    @classmethod
    def from_settings(cls, coach: CoachSettings) -> Thresholds:
        return cls(
            inaccuracy=coach.inaccuracy,
            mistake=coach.mistake,
            blunder=coach.blunder,
            decisive_gap=coach.decisive_gap,
            only_move_gap=coach.only_move_gap,
        )


DEFAULT_THRESHOLDS = Thresholds()


def terminal_expected_score(board: chess.Board) -> float | None:
    """The value of a finished position to the side to move, or None if it is not finished."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return DRAW
    return 1.0 if outcome.winner == board.turn else 0.0


def position_score(board: chess.Board, lines: Sequence[Line]) -> float:
    """The position's expected score to the side to move, whether or not it is finished."""
    finished = terminal_expected_score(board)
    if finished is not None:
        return finished
    if not lines:
        return DRAW  # nothing was measured; claim nothing
    return lines[0].expected_score


def eval_loss(before: Sequence[Line], played_uci: str, after_score: float) -> float:
    """What the played move cost its mover, in expected score.

    `after_score` is the position after the move, scored for the player who is to move
    there, which is the opponent; the mover's own prospects are one minus that. Comparing
    the two needs both to have been measured at the same depth, which is why the annotator
    analyses every position of a game the same way.

    The engine's own first choice costs nothing by definition. Without that rule a search
    that returns a slightly different number for the same line at the same depth would have
    the coach calling the best move on the board an inaccuracy.
    """
    if not before:
        return 0.0
    best = before[0]
    if best.moves and played_uci == best.moves[0]:
        return 0.0
    return max(0.0, best.expected_score - (1.0 - after_score))


def severity(loss: float, thresholds: Thresholds | None = None) -> Severity:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    if loss >= thresholds.blunder:
        return Severity.BLUNDER
    if loss >= thresholds.mistake:
        return Severity.MISTAKE
    if loss >= thresholds.inaccuracy:
        return Severity.INACCURACY
    return Severity.NONE


def best_gap(lines: Sequence[Line]) -> float | None:
    """How much better the best line is than the second best, or None without a second."""
    if len(lines) < 2:
        return None
    return max(0.0, lines[0].expected_score - lines[1].expected_score)


def is_only_move(board: chess.Board, lines: Sequence[Line], thresholds: Thresholds) -> bool:
    """One move holds the position: either it is the only legal one, or the rest throw it away."""
    legal = board.legal_moves.count()
    if legal == 0:
        return False
    if legal == 1:
        return True
    gap = best_gap(lines)
    return gap is not None and gap >= thresholds.only_move_gap


def is_decisive(board: chess.Board, lines: Sequence[Line], thresholds: Thresholds) -> bool:
    """A moment where the choice matters: the lines diverge, or there is only one move.

    This is what the coach may say out loud before a move is played, because it points at
    the position without naming the move. Which move it is stays behind the disclosure
    filter until the user finds it, plays it, or asks.
    """
    if is_only_move(board, lines, thresholds):
        return True
    gap = best_gap(lines)
    return gap is not None and gap >= thresholds.decisive_gap
