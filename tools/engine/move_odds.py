"""How likely each of a position's lines is to be the move played: the rank of the move actually
played among the six lines, by who played it (the player or the opponent) and by how far the line
falls behind the best one -- from the sessions' cached searches (tools/engine/simulate.py). The
table the adaptive look-ahead starts from.

    uv run python tools/engine/move_odds.py [dump] [--chesscom]"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import simulate as sim  # noqa: E402

# How far behind the best a line is, in expected score: the buckets the odds are kept by.
GAPS = (0.0, 0.02, 0.05, 0.10, 0.20, 1.01)


def bucket(gap: float) -> int:
    return next(i for i in range(len(GAPS) - 1) if gap < GAPS[i + 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path, nargs="?")
    parser.add_argument("--chesscom", action="store_true")
    args = parser.parse_args()
    sessions = {}
    if args.dump:
        sessions["puzzles"] = sim.session(args.dump)
    sessions.update(
        sim.chesscom_sessions("nosegazm") if args.chesscom else sim.library_sessions("nosegazm", 4)
    )
    cache = sim.load_cache(Path("all"))
    # For each mover and gap bucket: how often a line in it was the move played, of the lines
    # seen in it -- the chance one line of that kind is played.
    seen: dict[tuple[str, int], int] = Counter()
    played: dict[tuple[str, int], int] = Counter()
    ranks: dict[str, Counter] = defaultdict(Counter)
    for name, steps in sessions.items():
        kind = "puzzle" if name == "puzzles" else "game"
        for step in steps:
            if step.kind != "move":
                continue
            found = cache.get(sim.cache_key(step.board))
            if not found or not found["lines"]:
                continue
            lines = found["lines"]
            best = lines[0]["expected_score"]
            who = f"{kind}:{'you' if step.by_user else 'opponent'}"
            hit = None
            for rank, line in enumerate(lines):
                b = bucket(max(0.0, best - line["expected_score"]))
                seen[(who, b)] += 1
                if line["moves"] and line["moves"][0] == step.move.uci():
                    played[(who, b)] += 1
                    hit = rank
            ranks[who][hit if hit is not None else "not in six"] += 1
    for who in sorted(ranks):
        total = sum(ranks[who].values())
        dist = ", ".join(
            f"{k}: {v / total:.0%}"
            for k, v in sorted(ranks[who].items(), key=lambda kv: str(kv[0]))
        )
        print(f"{who:18} moves {total:4}  rank played -> {dist}")
    print("\nchance a line is the one played, by how far behind the best it is:")
    print(f"{'':18} " + " ".join(f"<{GAPS[i + 1]:.2f}".rjust(9) for i in range(len(GAPS) - 1)))
    for who in sorted(ranks):
        cells = []
        for b in range(len(GAPS) - 1):
            n = seen[(who, b)]
            cells.append(f"{played[(who, b)] / n:.2f}({n})".rjust(9) if n else "-".rjust(9))
        print(f"{who:18} " + " ".join(cells))
    table = {f"{who}|{b}": [played[(who, b)], seen[(who, b)]] for (who, b) in seen}
    (sim.CACHE / "move-odds.json").write_text(json.dumps(table), encoding="utf-8")


if __name__ == "__main__":
    main()
