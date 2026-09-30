"""A chess.com player's games from the official public API (api.chess.com/pub): every month's
archive, one request at a time as the API asks, saved as one PGN per game with chess.com's own
accuracies (when chess.com reviewed the game) in the headers -- the check on our accuracy and
performance rating against theirs (docs/game-review-plan.md, phase 0).

    uv run --no-sync python tools/review/chesscom_games.py <username> [--out data/review/chesscom]
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import chess.pgn
import httpx

API = "https://api.chess.com/pub/player/{user}/games/archives"
UA = {"User-Agent": "Sidecoach (personal chess coach; development)"}


def fetch(user: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    kept = with_acc = 0
    with httpx.Client(headers=UA, timeout=30) as client:
        months = client.get(API.format(user=user.lower())).json().get("archives", [])
        for month in months:
            for game in client.get(month).json().get("games", []):
                if game.get("rules") != "chess" or not game.get("pgn"):
                    continue
                pgn = chess.pgn.read_game(io.StringIO(game["pgn"]))
                if pgn is None:
                    continue
                acc = game.get("accuracies") or {}
                pgn.headers["TimeClass"] = str(game.get("time_class", ""))
                pgn.headers["Link"] = str(game.get("url", ""))
                if acc:
                    pgn.headers["WhiteAccuracy"] = str(acc.get("white"))
                    pgn.headers["BlackAccuracy"] = str(acc.get("black"))
                    with_acc += 1
                name = str(game.get("url", "")).rstrip("/").rsplit("/", 1)[-1] or str(
                    game.get("end_time")
                )
                (out / f"{name}.pgn").write_text(str(pgn) + "\n", encoding="utf-8")
                kept += 1
    print(
        json.dumps(
            {
                "user": user,
                "months": len(months),
                "games": kept,
                "with_accuracy": with_acc,
                "out": str(out),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("user")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    fetch(args.user, args.out or Path("data/review/chesscom") / args.user.lower())
