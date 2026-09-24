"""The fixed reading batch: the same puzzles every round, the three hints before and after, for
a person to read.

The checks (`tools/puzzles/story_checks.py`) catch only what has been written down; Tai's rule
(2026-09-24) is that the hints are judged by reading them as a player. This prints a fixed set
-- a few puzzles per theme, chosen once by a seed and kept in `data/eval/read/` -- from two
bench runs side by side, so each round is read on the same positions and a change shows as a
change.

  uv run --no-sync python tools/eval/read_batch.py --before A.jsonl --after B.jsonl --out read.md
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEPT = ROOT / "data" / "eval" / "read"
THEMES = (
    "mateIn1",
    "mateIn2",
    "mateIn3",
    "fork",
    "pin",
    "skewer",
    "discoveredAttack",
    "deflection",
    "hangingPiece",
    "trappedPiece",
    "advancedPawn",
    "defensiveMove",
    "quietMove",
    "intermezzo",
    "zugzwang",
    "equality",
)
_BORING = {"middlegame", "endgame", "short", "long", "crushing", "advantage", "master", "veryLong"}


def _records(path: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            out[row["id"]] = row
    return out


def batch(after: dict[str, dict], per_theme: int, seed: int) -> list[str]:
    """The fixed puzzle ids: chosen once, kept, and read back every round."""
    kept = KEPT / f"batch-{seed}-{per_theme}.json"
    if kept.exists():
        return json.loads(kept.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    chosen: list[str] = []
    for theme in THEMES:
        pool = sorted(
            pid
            for pid, row in after.items()
            if theme in row["themes"] and not row.get("engine_differs") and pid not in chosen
        )
        chosen += rng.sample(pool, min(per_theme, len(pool)))
    KEPT.mkdir(parents=True, exist_ok=True)
    kept.write_text(json.dumps(chosen), encoding="utf-8")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-theme", type=int, default=3)
    parser.add_argument("--seed", type=int, default=24)
    args = parser.parse_args()
    before, after = _records(args.before), _records(args.after)
    ids = batch(after, args.per_theme, args.seed)
    lines = [f"# Reading batch ({len(ids)} puzzles)", ""]
    changed = 0
    for pid in ids:
        b, a = before.get(pid), after.get(pid)
        if a is None:
            continue
        themes = " ".join(t for t in a["themes"] if t not in _BORING)
        lines.append(f"## {pid} -- {themes}")
        lines.append(f"`{a['fen']}` -- the answer: **{a['solution']}**")
        lines.append("")
        for rung, name in (("idea", "1"), ("focus", "2"), ("move", "3")):
            new = a["texts"][rung]
            old = b["texts"][rung] if b else None
            if old is not None and old != new:
                changed += 1
                lines.append(f"- {name} before: {old}")
                lines.append(f"- {name} after: **{new}**")
            else:
                lines.append(f"- {name}: {new}")
        lines.append("")
    lines.insert(1, f"{changed} hints changed.")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(ids)} puzzles, {changed} hints changed -> {args.out}")


if __name__ == "__main__":
    main()
