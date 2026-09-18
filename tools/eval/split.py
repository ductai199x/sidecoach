"""The train / held-out split of the games with facts, by game: 20% of each book's master
games (a stratum per book) and of the recordings, plus the recorded game the review
question was first asked on, always held out. `icr_play.run_games` draws its games from
the two sides of this split.

  data/eval/split.json   {"heldout_games": ["masters/<book>__<game>", ...], "train_games": [...]}

Run:  uv run python tools/eval/split.py [--seed 7] [--fraction 0.2]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FACTS = ROOT / "data" / "eval" / "facts"
SPLIT = ROOT / "data" / "eval" / "split.json"
ALWAYS_HELD = {"20260916-102830"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fraction", type=float, default=0.2)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    train: list[str] = []
    heldout: list[str] = []
    for source_dir in sorted(p for p in FACTS.iterdir() if p.is_dir()):
        names = sorted(p.stem for p in source_dir.glob("*.json"))
        # Held out per stratum: the book of a master game ("mammoth__050-..."), else the source.
        strata: dict[str, list[str]] = {}
        for n in names:
            key = n.split("__", 1)[0] if "__" in n else source_dir.name
            strata.setdefault(key, []).append(n)
        held: set[str] = set()
        for _, members in sorted(strata.items()):
            forced = [n for n in members if n in ALWAYS_HELD]
            rest = [n for n in members if n not in ALWAYS_HELD]
            rng.shuffle(rest)
            k = max(1, round(len(members) * args.fraction)) - len(forced)
            held.update(forced + rest[: max(0, k)])
        for n in names:
            (heldout if n in held else train).append(f"{source_dir.name}/{n}")
        print(f"{source_dir.name}: {len(names)} games, {len(held)} held out")
    SPLIT.write_text(
        json.dumps({"heldout_games": heldout, "train_games": train}, indent=1), encoding="utf-8"
    )
    print("wrote", SPLIT)


if __name__ == "__main__":
    main()
