"""The player's chess.com games, with their clocks, fetched from the public archive into
data/engine-cache/chesscom-<user>.json, for the engine simulator (tools/engine/simulate.py).

    uv run python tools/engine/chesscom_games.py [user]"""

from __future__ import annotations

import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

CACHE = Path(__file__).resolve().parents[2] / "data" / "engine-cache"


def fetch(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "sidecoach-dev"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def main() -> None:
    user = sys.argv[1] if len(sys.argv) > 1 else "nosegazm"
    archives = fetch(f"https://api.chess.com/pub/player/{user}/games/archives")["archives"]
    games = []
    for month in archives:
        games += [g for g in fetch(month).get("games", []) if g.get("pgn") and "%clk" in g["pgn"]]
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"chesscom-{user}.json"
    out.write_text(
        json.dumps(
            [
                {
                    "pgn": g["pgn"],
                    "time_class": g.get("time_class"),
                    "time_control": g.get("time_control"),
                    "rules": g.get("rules"),
                }
                for g in games
            ]
        ),
        encoding="utf-8",
    )
    print(f"{len(games)} games with clocks -> {out}")
    print(
        Counter(
            (g.get("time_class"), g.get("time_control"), g.get("rules")) for g in games
        ).most_common()
    )


if __name__ == "__main__":
    main()
