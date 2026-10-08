"""The post-game review: the game in numbers, from the engine's verdicts on the player's
moves, and in sentences the coach can sum it up from.

Nothing here is opinion. The tally, the accuracy, where the game turned and what was
better, how each phase went and what the mistakes have in common are all read off the
assessments and the causes, so the coach's summary has something true to say.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import chess

from sidecoach.analysis.score import game_accuracy
from sidecoach.events import GameReview, PhaseSummary, Side, TurningPoint
from sidecoach.profile.causes import HABIT
from sidecoach.profile.insights import motif_words

ERRORS = ("mistake", "blunder")
GOOD = ("brilliant", "great", "best", "excellent")
TURNING_POINTS = 3
PATTERN_MIN = 2  # mistakes of one cause in a game that make a pattern worth naming


@dataclass
class UserMove:
    """A judged move of the player's, as the profiler keeps it."""

    ply: int
    san: str
    uci: str
    fen_before: str
    label: str | None
    severity: str
    expected_loss: float
    best_uci: str | None
    accuracy: float | None
    phase: str
    cause: str | None = None
    motif: str | None = None
    segment: int = 0


def numbered(fen_before: str, san: str) -> str:
    board = chess.Board(fen_before)
    n = board.fullmove_number
    return f"{n}. {san}" if board.turn == chess.WHITE else f"{n}... {san}"


def best_san(fen_before: str, best_uci: str | None) -> str | None:
    if best_uci is None:
        return None
    board = chess.Board(fen_before)
    try:
        move = chess.Move.from_uci(best_uci)
    except ValueError:
        return None
    return board.san(move) if move in board.legal_moves else None


def _outcome(result: str, user_side: Side) -> str:
    if result == "1/2-1/2":
        return "drew"
    if result not in ("1-0", "0-1"):
        return "played"
    won = (result == "1-0") == (user_side is Side.WHITE)
    return "won" if won else "lost"


def build_review(
    *,
    game_id: str,
    user_side: Side,
    result: str,
    reason: str | None,
    opening: str | None,
    moves: list[UserMove],
    fullmoves: int,
) -> GameReview:
    """The review of one game from the player's judged moves."""
    judged = [m for m in moves if m.severity is not None]
    tallies = Counter(m.label or m.severity for m in judged)
    accuracy = game_accuracy([m.accuracy for m in judged if m.accuracy is not None])
    turning = sorted((m for m in judged if m.severity in ERRORS), key=lambda m: -m.expected_loss)[
        :TURNING_POINTS
    ]
    turning_points = [
        TurningPoint(
            ply=m.ply,
            segment=m.segment,
            san=numbered(m.fen_before, m.san),
            label=m.label or m.severity,
            expected_loss=round(m.expected_loss, 3),
            best_san=best_san(m.fen_before, m.best_uci),
            cause=m.cause,
            motif=m.motif,
            phase=m.phase,
            fen_before=m.fen_before,
        )
        for m in turning
    ]
    by_phase: dict[str, list[UserMove]] = defaultdict(list)
    for m in judged:
        by_phase[m.phase].append(m)
    phases = [
        PhaseSummary(
            phase=phase,
            moves=len(items),
            expected_loss=round(sum(i.expected_loss for i in items), 3),
            mistakes=sum(1 for i in items if i.severity in ERRORS),
        )
        for phase in ("opening", "middlegame", "endgame")
        if (items := by_phase.get(phase))
    ]
    causes: dict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    for m in judged:
        if m.severity in ERRORS and m.cause:
            causes[m.cause] += m.expected_loss
            counts[m.cause] += 1
    ordered = dict(sorted(((k, round(v, 3)) for k, v in causes.items()), key=lambda kv: -kv[1]))
    review = GameReview(
        game_id=game_id,
        user_side=user_side,
        result=result,
        reason=reason,
        opening=opening,
        moves=len(judged),
        accuracy=None if accuracy is None else round(accuracy, 1),
        tallies=dict(tallies),
        turning_points=turning_points,
        phases=phases,
        causes=ordered,
    )
    review.lines = lines_for(review, fullmoves)
    pattern = pattern_line(counts, causes)
    if pattern:
        # Second, after the result: the offline review says the first four lines.
        review.lines.insert(1, pattern)
    return review


def pattern_line(counts: Counter[str], losses: dict[str, float]) -> str | None:
    """The one thing to work on, when one cause is behind two or more of the game's mistakes,
    and a habit that helps: said as a skill to practise, not a count of the times it went
    wrong (Tai, 2026-10-02)."""
    if not counts:
        return None
    cause, n = max(counts.items(), key=lambda kv: (kv[1], losses.get(kv[0], 0.0)))
    if n < PATTERN_MIN or cause not in HABIT:
        return None
    words, habit = HABIT[cause]
    return f"One thing to work on: {words}. A habit that helps: {habit}."


# How a game ended, as the review's first sentence says it: the reason where it reads as
# English after "by", these where it does not ("lost by user resigned or lost on time",
# "won by game over on board": the parity audit, 2026-09-26).
HOW_IT_ENDED = {
    "draw agreed": " by agreement",
    "lost on time": " on time",
    "fifty-move rule": " by the fifty-move rule",
    "user resigned or lost on time": " by resignation or on time",
    "opponent resigned or lost on time": " by resignation or on time",
    "game over on board": "",
}


def lines_for(review: GameReview, fullmoves: int) -> list[str]:
    """The review in sentences."""
    out: list[str] = []
    how = HOW_IT_ENDED.get(review.reason, f" by {review.reason}") if review.reason else ""
    colour = review.user_side.value
    outcome = _outcome(review.result, review.user_side)
    # "1 move", not "1 moves" (the audit, 2026-10-07).
    moves = f"{fullmoves} move{'' if fullmoves == 1 else 's'}"
    if review.result not in ("1-0", "0-1", "1/2-1/2"):
        out.append(f"You played {colour} for {moves}; I did not see how it ended.")
    else:
        out.append(f"You played {colour} and {outcome}{how} in {moves}.")
    if review.opening:
        out.append(f"The opening was the {review.opening}.")
    if review.moves:
        order = (
            "brilliant",
            "great",
            "best",
            "excellent",
            "good",
            "book",
            "inaccuracy",
            "mistake",
            "miss",
            "blunder",
        )
        # The kinds that are nouns take a plural; best, excellent, good and book do not.
        plural = {
            "inaccuracy": "inaccuracies",
            "mistake": "mistakes",
            "miss": "misses",
            "blunder": "blunders",
        }
        parts = [
            f"{n} {plural.get(k, k) if n > 1 else k}"
            for k in order
            if (n := review.tallies.get(k, 0))
        ]
        acc = f"Accuracy {review.accuracy:.1f}. " if review.accuracy is not None else ""
        out.append(f"{acc}Your {review.moves} moves: " + ", ".join(parts) + ".")
    for tp in review.turning_points:
        better = f"; {tp.best_san} was better" if tp.best_san else ""
        why = ""
        if tp.cause:
            why = f" ({tp.cause}"
            if tp.motif:
                why += f": {motif_words(tp.motif)}"
            why += ")"
        out.append(
            f"Ground lost at {tp.san}, a {tp.label} costing {tp.expected_loss:.2f}{better}{why}."
        )
    if not review.turning_points and review.moves:
        out.append("No move cost you real ground.")
    if len(review.phases) > 1:
        bits = [f"the {p.phase} {p.expected_loss:.2f} over {p.moves} moves" for p in review.phases]
        out.append("Expected score lost by phase: " + ", ".join(bits) + ".")
    if review.causes:
        cause, loss = next(iter(review.causes.items()))
        out.append(f"Most ground went to {cause} ({loss:.2f}).")
    return out
