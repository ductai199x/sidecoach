"""What the player's record adds up to, in words (DESIGN section 5.6): the biggest cost (the
area Stats' "Work on this" puts first), the motif missed most often, the phase that costs most.

Everything is computed from counts and the expected score they cost, and an insight
needs a few mistakes behind it, so a single event never makes a weakness. The words are
for the coach's continuity and the panel; the numbers stay in the cells.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from sidecoach.events import ProfileCell
from sidecoach.profile.stats import AREA_NAMES
from sidecoach.profile.store import CellRow

# A motif is named after this many mistakes of it and this many occasions, from this many
# games: three mistakes in one game were "a habit" (the review of 2026-10-01).
MIN_ERRORS = 3
MIN_CHANCES = 4
MIN_GAMES = 3


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


def insights(rows: Sequence[CellRow], games: int, cost: dict[str, Any] | None = None) -> list[str]:
    """The player model in a few sentences, or nothing while the counts are small. `cost` is the
    area that costs the player most (`stats.biggest_cost`): the one rule Stats' "Work on this"
    uses, so the coach and the page name the same thing. The cells' causes (a mistake's why)
    were a second taxonomy beside Stats' areas -- the coach said "missed opponent threat" where
    the page said "forks you missed" (Tai, 2026-10-02: "fine", one) -- and no longer name the
    biggest cost; the sentences that said a gap or a habit were about that cause and go with
    it."""
    out: list[str] = []
    if cost is not None:
        name = AREA_NAMES.get(cost["area"], cost["area"])
        out.append(
            f"Your biggest cost over {cost['games']} games is {name}: {cost['moves']} moves."
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
