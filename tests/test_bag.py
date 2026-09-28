"""The coach's wordings drawn like marbles from a bag (coach/bag.py)."""

from __future__ import annotations

import random

import pytest

from sidecoach.coach.bag import Bags

WORDINGS = ["first", "second", "third", "fourth", "fifth"]


def test_every_wording_once_before_any_comes_again_and_never_twice_running() -> None:
    """Four decisive moments in one game got the same sentence four times (a master-game
    pass, 2026-09-21); "random, but never the last one" still said A, B, A, B."""
    for seed in range(50):
        bags = Bags(random.Random(seed))
        said = [bags.draw("cost", WORDINGS) for _ in range(len(WORDINGS) * 8)]
        for start in range(0, len(said), len(WORDINGS)):
            assert sorted(said[start : start + len(WORDINGS)]) == sorted(WORDINGS), seed
        # Across a refill too: the new bag never opens with the wording just said.
        assert all(a != b for a, b in zip(said, said[1:], strict=False)), seed


def test_each_key_is_its_own_bag_and_a_list_of_one_is_said_again() -> None:
    bags = Bags(random.Random(1))
    first = [bags.draw("cost", ["a", "b", "c"]) for _ in range(2)]
    [bags.draw("fine", ["a", "b", "c"]) for _ in range(3)]
    assert sorted([*first, bags.draw("cost", ["a", "b", "c"])]) == ["a", "b", "c"]
    assert [bags.draw("one", ["only"]) for _ in range(3)] == ["only"] * 3
    with pytest.raises(ValueError):
        bags.draw("none", [])


def test_a_wording_no_longer_offered_leaves_the_bag() -> None:
    bags = Bags(random.Random(2))
    bags.draw("praise", ["a", "b", "c", "d"])
    said = [bags.draw("praise", ["a", "b"]) for _ in range(6)]
    assert set(said) == {"a", "b"}


def test_a_peek_is_the_next_draw_and_uses_nothing_up_until_taken() -> None:
    """The first hint is written for every position and said in few: written, it only peeks;
    said, it takes."""
    bags = Bags(random.Random(3))
    peeked = [bags.peek("lead", WORDINGS) for _ in range(4)]
    assert len(set(peeked)) == 1
    bags.take("lead", peeked[0])
    rest = [bags.draw("lead", WORDINGS) for _ in range(len(WORDINGS) - 1)]
    assert sorted([peeked[0], *rest]) == sorted(WORDINGS)
    assert bags.peek("lead", WORDINGS) != rest[-1]
