"""Expected score, the loss a move costs, and what makes a moment decisive.

Everything here is in expected score, never centipawns: 0.5 is a dead draw, 1.0 is a win
for the side to move. A 50-centipawn slip at +8 costs nothing and the same slip at level
costs the game, so a threshold in centipawns says nothing about how much a move mattered.

Pure functions over engine lines, so they are cheap to test and carry no engine.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import chess

from sidecoach.config import CoachSettings
from sidecoach.events import Label, Line, Severity

DRAW = 0.5
# Lichess's curve from centipawns to winning chances, fitted to games between people.
# Stockfish's own win-draw-loss numbers answer a different question: how often Stockfish
# beats Stockfish from here, which is 100% from a knight up and says nothing about the rest
# of a club game. A coach needs the human curve, where a knight up is about 0.88 and the
# moves that follow can still cost something.
HUMAN_K = 0.00368208
CP_CAP = 1000


def human_expected_score(score_cp: int | None, mate: int | None) -> float:
    """Expected score for the side to move, on the curve fitted to human games."""
    if mate is not None:
        return 1.0 if mate > 0 else 0.0
    if score_cp is None:
        return DRAW
    capped = max(-CP_CAP, min(CP_CAP, score_cp))
    return 1.0 / (1.0 + math.exp(-HUMAN_K * capped))


def engine_expected_score(wdl: tuple[int, int, int] | None) -> float:
    """Expected score from the engine's own win-draw-loss numbers: (W + D/2) / 1000."""
    if wdl is None or sum(wdl) == 0:
        return DRAW
    wins, draws, _ = wdl
    return (wins + draws / 2) / sum(wdl)


def expected_score(
    model: str, score_cp: int | None, mate: int | None, wdl: tuple[int, int, int] | None
) -> float:
    """Expected score under the named model: "human" (the default) or "engine"."""
    if model == "engine":
        return engine_expected_score(wdl)
    return human_expected_score(score_cp, mate)


@dataclass(frozen=True)
class Thresholds:
    """How much loss earns a name, and what makes a position a decisive moment.

    The three severity lines were fitted to chess.com's review tallies over six of the
    user's games (671 plies, 2026-09-15): they put our inaccuracy, mistake, miss and blunder
    counts at 55, 33, 29 and 9 against their 55, 38, 24 and 10. A blunder throws away more
    than a quarter of the game.
    """

    inaccuracy: float = 0.04
    mistake: float = 0.10
    blunder: float = 0.28
    decisive_gap: float = 0.10
    # An only move: everything else throws away this much. Wider than a decisive moment,
    # since "there is one move here" is a stronger claim than "this move matters".
    only_move_gap: float = 0.30
    # Below this loss a move is "excellent" rather than merely "good".
    excellent: float = 0.02
    # A position this far gone, either way, is decided: a sacrifice in it is not brilliant.
    decided: float = 0.90

    @classmethod
    def from_settings(cls, coach: CoachSettings) -> Thresholds:
        return cls(
            inaccuracy=coach.inaccuracy,
            mistake=coach.mistake,
            blunder=coach.blunder,
            decisive_gap=coach.decisive_gap,
            only_move_gap=coach.only_move_gap,
            excellent=coach.excellent,
            decided=coach.decided,
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


def label_move(
    loss: float,
    thresholds: Thresholds,
    *,
    is_best: bool,
    in_book: bool,
    only_move: bool,
    sacrifice: bool,
    before_score: float,
    opponent_erred: bool,
    obvious: bool = False,
) -> Label:
    """What kind of move it was, in the vocabulary the user reads on chess.com.

    Two axes. The loss says how good the move was: best, excellent, good, then the three
    names for losing ground. The moment says whether it mattered, which is what great,
    brilliant and miss add. `opponent_erred` says the move before this one was a mistake
    or worse, so this move either took the chance (great, when it is the best move) or let
    it go (miss, whatever the size of the loss); `only_move` says everything else lost, so
    finding it was great; a sound `sacrifice` is brilliant unless the position was already
    decided, where giving material back is just technique. Neither great nor brilliant
    applies to an `obvious` move (a recapture, a free capture nobody can take back, the
    only legal move), nor in a lost position, where the best move changes nothing. A great
    move in a won position stays great; chess.com's review awards it there and the user
    reads that.
    """
    flagged = loss >= thresholds.inaccuracy
    if in_book and not flagged:
        # The opening tables hold dubious gambits too; a move that loses ground is judged
        # by what it lost, whatever a book calls it. 1. e4 f5 is in the tables.
        return Label.BOOK
    won = before_score >= thresholds.decided
    lost = before_score <= 1.0 - thresholds.decided
    if opponent_erred and flagged:
        return Label.MISS
    if not flagged:
        if not obvious and not lost:
            if sacrifice and not won:
                return Label.BRILLIANT
            if only_move or (opponent_erred and is_best):
                return Label.GREAT
        if is_best:
            return Label.BEST
        return Label.EXCELLENT if loss < thresholds.excellent else Label.GOOD
    verdict = severity(loss, thresholds)
    if verdict is Severity.BLUNDER:
        return Label.BLUNDER
    return Label.MISTAKE if verdict is Severity.MISTAKE else Label.INACCURACY


# Per-move accuracy: 100 for no loss, falling off exponentially with the drop in winning
# chances. The shape is Lichess's; the steepness is not. Lichess uses 0.04354, on which an
# inaccuracy scores about 80, and the plain mean of that ran fifteen points above
# chess.com's accuracy on every one of six games. At 0.25 an inaccuracy scores 26, a
# mistake 5, and the mean lands within 4.4 points of chess.com's number over twelve
# game-sides with no bias to speak of: chess.com's accuracy is, in effect, how often you
# played a near-best move.
ACCURACY_A = 103.1668
ACCURACY_K = 0.25
ACCURACY_C = 3.1669


def move_accuracy(loss: float) -> float:
    """0 to 100 for one move, from its loss in expected score."""
    raw = ACCURACY_A * math.exp(-ACCURACY_K * loss * 100.0) - ACCURACY_C
    return round(max(0.0, min(100.0, raw)), 2)


def game_accuracy(accuracies: Sequence[float]) -> float | None:
    """One number for a side's game: the mean of the per-move accuracies.

    Lichess averages a volatility-weighted mean with a harmonic mean; the harmonic mean
    collapses on a single move that flipped the game, which club games have. Chess.com's
    own formula is not published; the plain mean on the steepened curve is what fitted.
    """
    if not accuracies:
        return None
    return sum(accuracies) / len(accuracies)


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
