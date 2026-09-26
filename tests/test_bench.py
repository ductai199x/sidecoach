"""The coach's benchmark (tools/eval/bench.py): the split is by game and frozen, the draw is
balanced, only the drawn moments reach the model, and the numbers carry their uncertainty."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import chess

from sidecoach.events import DisclosureLevel, Trigger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "eval"))

import bench  # noqa: E402  # pyright: ignore[reportMissingImports]
import corpus  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_the_split_is_by_game_half_and_half_within_each_level_and_the_same_every_time():
    stems = [f"L{level}__{i:03d}" for level in bench.LEVELS for i in range(12)]
    split = bench.split_games(stems)
    assert split == bench.split_games(list(reversed(stems)))  # frozen by the seed
    for level in bench.LEVELS:
        parts = [split[s] for s in stems if bench.level_of(s) == level]
        assert parts.count("dev") == 6 and parts.count("test") == 6


def _items(n_games: int, per_class: int) -> list[dict]:
    out = []
    for g in range(n_games):
        for cls in bench.FAMILY_OF:
            for k in range(per_class):
                out.append({"id": f"g{g}/{cls}/{k}", "game": f"g{g}", "class": cls})
    return out


def test_the_draw_takes_a_third_from_each_family_spread_over_classes_and_games():
    import random

    items = _items(n_games=6, per_class=3)
    chosen = bench.draw_moments(items, 30, random.Random(1))
    assert len(chosen) == 30
    families = [bench.FAMILY_OF[it["class"]] for it in chosen]
    assert all(families.count(f) == 10 for f in bench.FAMILIES)
    # No class crowds out the rest: every class of a family with at least as many
    # classes as its share gets one before any gets two.
    for family, classes in bench.FAMILIES.items():
        got = [it["class"] for it in chosen if bench.FAMILY_OF[it["class"]] == family]
        counts = [got.count(c) for c in classes]
        assert max(counts) - min(counts) <= 1, (family, counts)
    # At most an even share (+1) of a family's moments from any one game.
    for family in bench.FAMILIES:
        games = [it["game"] for it in chosen if bench.FAMILY_OF[it["class"]] == family]
        assert max(games.count(g) for g in set(games)) <= -(-10 // 6) + 1
    assert chosen == bench.draw_moments(items, 30, random.Random(1))  # frozen by the seed


def test_the_intervals_and_the_paired_test():
    lo, hi = bench.wilson(8, 10)
    assert 0.48 < lo < 0.50 and 0.94 < hi < 0.95
    assert all(math.isnan(x) for x in bench.wilson(0, 0))  # no moments, no rate
    assert bench.mcnemar(0, 0) == 1.0
    assert bench.mcnemar(10, 0) < 0.01
    assert bench.mcnemar(5, 5) == 1.0
    import random

    lo, hi = bench.mean_interval([0.0, 1.0] * 50, random.Random(3))
    assert lo < 0.5 < hi


def _request(ply: int, trigger: str, text: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        ply=ply,
        trigger=Trigger(trigger),
        user_text=text,
        fallback=f"template {ply}",
        level=DisclosureLevel.IDEA,
        game_id="g",
        board=chess.Board(),
        annotations=[],
    )


def test_only_the_drawn_moments_reach_the_model():
    asked: list[int] = []

    def model(request):
        asked.append(request.ply)
        return SimpleNamespace(message=SimpleNamespace(text="model"))

    seen: dict = {}
    drawn = {corpus.moment_key(_request(4, "post_move"), seen)}  # type: ignore[arg-type]
    capture = corpus.Capture(model, only=drawn)
    for request in (
        _request(3, "stuck"),
        _request(4, "post_move"),
        _request(5, "user_ask", "hint"),
    ):
        capture(request)  # type: ignore[arg-type]
    assert [s.message.text for s in capture.spoken] == ["template 3", "model", "template 5"]
    assert asked == [4], "the model answered the drawn moment and nothing else"
    # The same moment asked twice in a game is two moments.
    seen = {}
    first = corpus.moment_key(_request(9, "user_ask", "hint"), seen)  # type: ignore[arg-type]
    second = corpus.moment_key(_request(9, "user_ask", "hint"), seen)  # type: ignore[arg-type]
    assert first != second and first[:3] == second[:3]
