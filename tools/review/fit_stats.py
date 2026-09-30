"""Stats' peers and category ratings (docs/stats-plan.md, S2), from the real players' game sides
`human_facts.py` judged (Lichess rapid, six rating bands).

The peers: each side becomes a game of Stats' own shape (its strategy themes read again from the
sample's positions, which the facts do not keep), the sides of a band are added up by Stats' own
functions (`stats.figures`), and every figure gets a line through the bands -- its value against
the band's mean rating, weighted by how many it counts -- saved with how well it fits
(`profile/peers.json`). Two figures are left out: the book moves before leaving book (it needs
the opponent's moves, which the sample does not hold) and the time a move (the sample's
increment is not known).

The category ratings: the game rating's model (`performance`) fitted on each category's moves
alone (`stats.CATEGORIES`), on part of the players, and tested on the rest -- a side by itself
(one game), and sides of close ratings pooled ten at a time (a player's page of games, as Stats
reads them) -- beside the game model on the same moves (`profile/categories.json`).

    uv run --no-sync python tools/review/fit_stats.py data/review/human/facts.jsonl --write
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np
from fit_performance import fit, held_out
from human_losses import game_sides

from sidecoach.profile import performance, stats
from sidecoach.profile.stat_facts import themes_of

LEFT_OUT = ("openings.book_moves", "time.by_phase")
POOL = 10  # sides pooled into one pseudo-player


def load(path: Path) -> list[dict]:
    """The judged sides, each move with its position's themes (from the sample, by the side's
    key and the ply)."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    wanted = {(r["player"], r["rating"], r["opp_rating"], r["result"]) for r in rows}
    fens: dict[tuple, dict[int, str]] = {}
    for side in game_sides():
        first = side[0]
        key = (first["player_id"], first["rating"], first["opp_rating"], first["result"])
        if key in wanted:
            fens[key] = {r["ply"]: r["fen"] for r in side}
    for i, row in enumerate(rows):
        by_ply = fens[(row["player"], row["rating"], row["opp_rating"], row["result"])]
        for move in row["moves"]:
            move["themes"] = ",".join(themes_of(chess.Board(by_ply[move["ply"]])))
        if (i + 1) % 300 == 0:
            print(f"  themes: {i + 1}/{len(rows)} sides", flush=True)
    return rows


def as_stats(rows: list[dict]) -> stats._Data:
    """The sides as Stats' games and moves: a game a side, the player's side known."""
    games, moves = [], []
    for row in rows:
        key = f"{row['player']}:{row['rating']}:{row['opp_rating']}:{row['result']}"
        white = row["side"] == "white"
        result = {1: "1-0" if white else "0-1", -1: "0-1" if white else "1-0"}.get(
            row["result"], "1/2-1/2"
        )
        games.append(
            {
                "key": key,
                "user_side": row["side"],
                "result": result,
                "time_class": "rapid",
                "played_at": None,
                "source": "lichess",
                "opening": None,
                "termination": None,
                "bot": False,
            }
        )
        for m in row["moves"]:
            clock = m.get("clock")
            moves.append(
                {
                    **m,
                    "key": key,
                    "side": row["side"],
                    "ply": m["ply"] + 1,
                    "clock_ms": None if clock is None else clock * 1000,
                    "spent_ms": None,
                }
            )
    return stats._Data(games, moves)


def peer_lines(rows: list[dict]) -> tuple[dict, dict]:
    """Each band's figures by Stats' functions, then a line a figure through the bands."""
    by_band: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_band[row["band"]].append(row)
    bands, points = {}, defaultdict(list)
    for band, members in sorted(by_band.items(), key=lambda kv: int(kv[0].split("-")[0])):
        rating = float(np.mean([r["rating"] for r in members]))
        bands[band] = {"sides": len(members), "rating": round(rating)}
        for path, figure in stats.walk(stats.figures(as_stats(members))):
            if figure["value"] is not None and figure["n"] and not path.startswith(LEFT_OUT):
                points[path].append((rating, float(figure["value"]), float(figure["n"])))
    lines = {}
    for path, pts in points.items():
        if len(pts) < 3:
            continue
        x, y, w = (np.array(c) for c in zip(*pts, strict=True))
        b, a = np.polyfit(x, y, 1, w=np.sqrt(w))
        mean = np.average(y, weights=w)
        total = np.average((y - mean) ** 2, weights=w)
        r2 = 1.0 - np.average((y - (a + b * x)) ** 2, weights=w) / total if total > 0 else 0.0
        lines[path] = {
            "a": round(float(a), 6),
            "b": round(float(b), 8),
            "r2": round(float(r2), 3),
            "lo": round(float(x.min())),
            "hi": round(float(x.max())),
            "n": int(w.sum()),
        }
    return bands, lines


def only(rows: list[dict], select) -> list[dict]:
    return [{**r, "moves": [m for m in r["moves"] if select(m)]} for r in rows]


def errors(rows: list[dict], model: dict | None) -> tuple[float, float, int]:
    """Mean absolute error of the estimate from the ratings: a side alone, and sides pooled ten
    at a time by rating (their moves together against their mean rating)."""
    alone = [
        abs(performance.estimate(r["moves"], model=model).rating - r["rating"])
        for r in rows
        if len(r["moves"]) >= performance.MIN_MOVES
    ]
    ordered = sorted((r for r in rows if r["moves"]), key=lambda r: r["rating"])
    pooled = []
    for i in range(0, len(ordered) - POOL + 1, POOL):
        group = ordered[i : i + POOL]
        moves = [m for r in group for m in r["moves"]]
        pooled.append(
            abs(
                performance.estimate(moves, model=model).rating
                - np.mean([r["rating"] for r in group])
            )
        )
    return float(np.mean(alone)), float(np.mean(pooled)), len(alone)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("facts", type=Path)
    parser.add_argument("--test-share", type=float, default=0.3)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows = load(args.facts)
    print(f"{len(rows)} sides")

    bands, lines = peer_lines(rows)
    print("\n== the peers: a line a figure through the bands (value at 1000 / 1500 / 2000, r²)")
    for band, info in bands.items():
        print(f"  {band:>10}: {info['sides']} sides, mean rating {info['rating']}")
    for path, line in sorted(lines.items()):
        at = [line["a"] + line["b"] * r for r in (1000, 1500, 2000)]
        values = " ".join(f"{v:8.2f}" for v in at)
        print(f"  {path:36} {values}   r² {line['r2']:.2f}   n {line['n']}")

    train = [r for r in rows if not held_out({"player": r["player"]}, args.test_share)]
    test = [r for r in rows if held_out({"player": r["player"]}, args.test_share)]
    print(f"\n== the category ratings: {len(train)} sides to fit, {len(test)} to test")
    head = f"  {'category':10} {'moves/side':>10} {'one game':>9} {'ten pooled':>11}"
    print(f"{head}   (the game model on the same moves)")
    models = {}
    for name, select in stats.CATEGORIES.items():
        models[name] = fit(only(train, select))
        mine = only(test, select)
        per_side = np.mean([len(r["moves"]) for r in mine])
        alone, pooled, n = errors(mine, models[name])
        base_alone, base_pooled, _ = errors(mine, None)
        base = f"({base_alone:.0f} / {base_pooled:.0f}; n {n})"
        print(f"  {name:10} {per_side:10.1f} {alone:9.0f} {pooled:11.0f}   {base}")
    alone, pooled, _ = errors(test, None)
    per_side = np.mean([len(r["moves"]) for r in test])
    print(f"  {'all moves':10} {per_side:10.1f} {alone:9.0f} {pooled:11.0f}   (the game model)")

    if args.write:
        stats.PEERS_PATH.write_text(
            json.dumps(
                {
                    "source": "Lichess rapid, judged at depth 18 (tools/review/human_facts.py)",
                    "bands": bands,
                    "figures": lines,
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        stats.CATEGORIES_PATH.write_text(json.dumps(models) + "\n", encoding="utf-8")
        print(f"\nwritten {stats.PEERS_PATH} and {stats.CATEGORIES_PATH}")


if __name__ == "__main__":
    main()
