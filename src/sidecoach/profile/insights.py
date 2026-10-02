"""What the cells add up to, in words (DESIGN section 5.6): the biggest cost, whether it
is a gap or a habit, the motif missed most often, the phase that costs most.

Everything is computed from counts and the expected score they cost, and an insight
needs a few mistakes behind it, so a single event never makes a weakness. The words are
for the coach's continuity and the panel; the numbers stay in the cells.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from sidecoach.events import ProfileCell
from sidecoach.profile.store import CellRow

# A cause needs this many mistakes before it is named; a motif this many occasions. A habit
# (a gap, a habit, a tactic missed) needs its mistakes from this many games: three mistakes in
# one game were "a habit" (the review of 2026-10-01).
MIN_ERRORS = 3
MIN_CHANCES = 4
MIN_GAMES = 3
# A cause whose cost sits this much in one kind of position is a gap; spread out, a habit.
GAP_SHARE = 0.6


def motif_words(name: str | None) -> str:
    """A tagger name in words: hangingPiece -> hanging piece, mateIn2 -> mate in 2."""
    if name is None:
        return ""
    out: list[str] = []
    for ch in name:
        if ch.isupper():
            out.append(" " + ch.lower())
        elif ch.isdigit() and out and not out[-1][-1].isdigit():
            out.append(" " + ch)
        else:
            out.append(ch)
    return "".join(out).replace("f 2 f 7", "f2/f7").replace("x ray", "x-ray")


def as_cells(rows: Sequence[CellRow]) -> list[ProfileCell]:
    return [
        ProfileCell(
            cause=r.cause,
            motif=r.motif,
            phase=r.phase,
            errors=r.errors,
            chances=r.chances,
            expected_loss=round(r.expected_loss, 3),
        )
        for r in rows
    ]


def motif_rates(rows: Sequence[CellRow]) -> list[tuple[str, float, int, int]]:
    """Each motif missed often enough, over enough of its chances and games: (motif, the rate
    it is missed, smoothed as (misses + 1) / (chances + 2) to rank them, misses, chances), in
    the order its chances came. The cells without a cause count a motif's chances and, in
    their errors, the chances missed; a mistake's own motif may be a tactic it allowed, never
    a chance the player had."""
    chances: dict[str, int] = defaultdict(int)
    misses: dict[str, int] = defaultdict(int)
    games: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.motif is None or r.cause is not None:
            continue
        chances[r.motif] += r.chances
        misses[r.motif] += r.errors
        games[r.motif] |= r.games
    return [
        (m, (misses[m] + 1) / (chances[m] + 2), misses[m], chances[m])
        for m in chances
        if chances[m] >= MIN_CHANCES and misses[m] >= MIN_ERRORS and len(games[m]) >= MIN_GAMES
    ]


def insights(rows: Sequence[CellRow], games: int) -> list[str]:
    """The player model in a few sentences, or nothing while the counts are small."""
    out: list[str] = []
    by_cause: dict[str, list[CellRow]] = defaultdict(list)
    for r in rows:
        if r.cause is not None and r.errors > 0:
            by_cause[r.cause].append(r)
    if by_cause:
        cause, cells = max(by_cause.items(), key=lambda kv: sum(c.expected_loss for c in kv[1]))
        errors = sum(c.errors for c in cells)
        loss = sum(c.expected_loss for c in cells)
        if errors >= MIN_ERRORS:
            over = f" over {games} games" if games > 1 else ""
            out.append(f"Your biggest cost{over} is {cause}: {errors} mistakes.")
            top = max(cells, key=lambda c: c.expected_loss)
            seen_in = len(frozenset[str]().union(*(c.games for c in cells)))
            if seen_in < MIN_GAMES:
                pass  # a gap or a habit needs several games behind it, never one bad game
            elif loss > 0 and top.expected_loss / loss >= GAP_SHARE and (top.motif or top.phase):
                what = motif_words(top.motif) if top.motif else "positions"
                where = f" in the {top.phase}" if top.phase else ""
                out.append(f"That looks like a gap: it shows up as {what}{where}.")
            else:
                kinds = len({(c.motif, c.phase) for c in cells})
                out.append(
                    f"That looks like a habit: it shows up across {kinds} kinds of position"
                    f" in {seen_in} games."
                )
    # The motif missed most often, as a rate over the occasions it was on the board.
    rated = motif_rates(rows)
    if rated:
        motif, _, missed, seen = max(rated, key=lambda t: t[1])
        out.append(f"You missed a {motif_words(motif)} {missed} of the {seen} times you had one.")
    # The phase that costs most per move.
    phase_moves: dict[str, int] = defaultdict(int)
    phase_loss: dict[str, float] = defaultdict(float)
    for r in rows:
        if r.phase is None:
            continue
        if r.cause is None and r.motif is None:
            phase_moves[r.phase] += r.chances
        phase_loss[r.phase] += r.expected_loss
    busy = [p for p in phase_moves if phase_moves[p] >= 10]
    if len(busy) >= 2:
        worst = max(busy, key=lambda p: phase_loss[p] / phase_moves[p])
        if phase_loss[worst] > 0:
            out.append(f"The {worst} costs you most per move.")
    return out
