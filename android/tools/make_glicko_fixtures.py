"""Record what the desktop's puzzle rating (sidecoach/profile/glicko.py) answers, for the Kotlin
port (profile/Glicko.kt) to be held to.

- `games`: computeGame over random players and outcomes, with and without the period's growth.
- `finishes`: a puzzle attempt over random perfs, puzzles (provisional ones among them), wins and
  losses, themes of every kind of weight, and times since the last puzzle: the perf after it
  (read as of now first, as Lichess reads it), the change shown and its text.
- `live`: the deviation as of a time, forwards and as stored.
- `weights`: the theme weight and the player weight of every theme kind, won and lost.

    cd D:/chess-coaching-companion
    uv run --no-sync python android/tools/make_glicko_fixtures.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DESKTOP / "src"))

from sidecoach.profile.glicko import (  # noqa: E402
    Glicko,
    Perf,
    compute_game,
    diff_text,
    finish,
    live_deviation,
    player_weight,
    rating_diff,
    read,
    theme_weight,
)

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "profile"
DAY_MS = 24 * 60 * 60 * 1000
THEMES = [None, "mix", "mateIn1", "backRankMate", "fork", "mateIn2", "mate", "endgame", "superGM"]


def _g(g: Glicko) -> list[float]:
    return [g.rating, g.deviation, g.volatility]


def _perf(p: Perf) -> dict:
    return {"glicko": _g(p.glicko), "nb": p.nb, "latest": p.latest}


def _number(x: float) -> float | None:
    """A deviation shrunk back past nothing is NaN, on the JVM too: null in the JSON."""
    return None if math.isnan(x) else x


def _random_glicko(rng: random.Random) -> Glicko:
    return Glicko(
        rng.uniform(300, 3200), rng.choice([rng.uniform(30, 520), 75.0, 110.0, 500.0]),
        rng.uniform(0.03, 0.12),
    )  # fmt: skip


def main() -> None:
    rng = random.Random(20261008)
    games = []
    for _ in range(300):
        white, black = _random_glicko(rng), _random_glicko(rng)
        score = rng.choice([0.0, 0.5, 1.0])
        skip = rng.random() < 0.5
        w, b = compute_game(white, black, score, skip_deviation_increase=skip)
        games.append(
            {"white": _g(white), "black": _g(black), "score": score, "skip": skip,
             "out": [_g(w), _g(b)]}
        )  # fmt: skip
    finishes = []
    for _ in range(400):
        latest = rng.choice([None, 0])
        now = rng.choice([1, DAY_MS, 30 * DAY_MS, 400 * DAY_MS])
        stored = Perf(_random_glicko(rng), nb=rng.randint(0, 3000), latest=latest)
        puzzle = Glicko(rng.uniform(400, 3000), rng.choice([60.0, 75.0, 109.9, 110.0, 300.0]), 0.09)
        win = rng.random() < 0.5
        theme = rng.choice(THEMES)
        before = read(stored, now)
        after = finish(before, puzzle, win, now, theme=theme)
        diff = rating_diff(before, after)
        finishes.append(
            {"stored": _perf(stored), "now": now, "puzzle": _g(puzzle), "win": win,
             "theme": theme, "before": _perf(before), "after": _perf(after), "diff": diff,
             "diff_text": diff_text(diff), "display": after.glicko.display()}
        )  # fmt: skip
    live = []
    for _ in range(100):
        perf = Perf(_random_glicko(rng), nb=10, latest=rng.choice([None, 0, 5 * DAY_MS]))
        now = rng.choice([5 * DAY_MS, 100 * DAY_MS, 3000 * DAY_MS])
        live.append(
            {"perf": _perf(perf), "now": now, "live": live_deviation(perf, now),
             "stored": _number(live_deviation(perf, now, reverse=True))}
        )  # fmt: skip
    weights = [
        {"theme": theme, "win": win, "deviation": dev, "theme_weight": theme_weight(theme, win),
         "player_weight": player_weight(theme, win, Glicko(1500, dev, 0.09))}
        for theme in THEMES for win in (True, False) for dev in (75.0, 110.0)
    ]  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "glicko.json"
    body = {"games": games, "finishes": finishes, "live": live, "weights": weights}
    path.write_text(
        json.dumps(body, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"{path}: {len(games)} games, {len(finishes)} finishes, {len(live)} live")


if __name__ == "__main__":
    main()
