"""The coach's wordings, drawn like marbles from a bag: every wording of a template once
before any comes again, and a fresh bag never opens with the one just said (Tai,
2026-09-25: "are you doing taking out a marbles in a bag without replacement? probably
should do that for anything that has multiple templates -- it's a small thing but
important UX"). "Random, but never the last one" still said A, B, A, B while C, D and E
waited.

One bag per key: a template ("cost", "keep"), a label's praise ("praise:good"), a lesson's
clauses ("clause:develop"), the first hint's lead ("lead:tactic"). A wording no longer among
the options is dropped from its bag; an empty bag is refilled and shuffled.

A wording written ahead of time -- the first hint is in every position's facts, and said
only in the few where the player asks -- is `peek`ed when it is written and `take`n when it
is said, so the hints never said do not use up the bag.
"""

from __future__ import annotations

import random
from collections.abc import Sequence


class Bags:
    """The bags of one coach (a game's worth of wordings, or a session's)."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()
        self._bags: dict[str, list[str]] = {}
        self._last: dict[str, str] = {}

    def draw(self, key: str, options: Sequence[str]) -> str:
        """One of `options`, from `key`'s bag, said now."""
        chosen = self.peek(key, options)
        self.take(key, chosen)
        return chosen

    def peek(self, key: str, options: Sequence[str]) -> str:
        """The wording `key`'s next draw gives, left in the bag."""
        wordings = list(dict.fromkeys(options))
        if not wordings:
            raise ValueError(f"no wordings for {key!r}")
        if len(wordings) == 1:
            return wordings[0]
        bag = [w for w in self._bags.get(key, []) if w in wordings]
        if not bag:
            bag = list(wordings)
            self.rng.shuffle(bag)
            # Drawn from the end: a new bag never opens with the wording just said.
            if bag[-1] == self._last.get(key):
                bag[0], bag[-1] = bag[-1], bag[0]
        self._bags[key] = bag
        return bag[-1]

    def take(self, key: str, wording: str) -> None:
        """`wording` said: out of `key`'s bag until the bag is refilled."""
        bag = self._bags.get(key, [])
        if wording in bag:
            bag.remove(wording)
        self._last[key] = wording
