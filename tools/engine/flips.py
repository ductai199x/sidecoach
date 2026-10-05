"""How bad the severity flips of the decided-position rule are (decided.py): each flip's
severities before and after, the loss's move, and the same counted for a baseline that is no
rule at all -- two depth-18 searches of one position (one line, and six lines) disagreeing on
each candidate's severity by themselves (single.py's sample: each candidate's own position
searched to depth 17, judged against each search's score of the position).

    uv run --no-sync python tools/engine/flips.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sidecoach.analysis.score import Thresholds, severity  # noqa: E402
from tools.engine.decided import BUDGETS, _taken  # noqa: E402

CACHE = ROOT / "data" / "engine-cache"
ORDER = ["none", "inaccuracy", "mistake", "blunder"]


def name(loss: float, th: Thresholds) -> str:
    return str(severity(loss, th).value if hasattr(severity(loss, th), "value") else severity(loss, th)).lower()


def table(flips: Counter, total: int, title: str) -> None:
    n = sum(flips.values())
    two = sum(c for (a, b), c in flips.items() if abs(ORDER.index(a) - ORDER.index(b)) >= 2)
    print(f"{title}: {n}/{total} severities differ ({n / max(1, total):.1%}); by two levels or more: {two}")
    for (a, b), c in flips.most_common():
        print(f"    {a:>10} -> {b:<10} {c}")


def main() -> None:
    th = Thresholds()
    out = json.loads((CACHE / "decided.json").read_text(encoding="utf-8"))
    for budget in BUDGETS:
        flips: Counter = Counter()
        total = 0
        moved = []
        for r in out:
            taken = _taken(r["marks"], budget)
            if taken is None:
                continue
            final = r["marks"][-1][2]
            for value in r["children"].values():
                total += 1
                a, b = name(max(0.0, final - value), th), name(max(0.0, taken[1] - value), th)
                if a != b:
                    flips[(a, b)] += 1
                    moved.append(abs(taken[1] - final))
        table(flips, total, f"rule at {budget}s (depth 18 -> the rule's)")
        if moved:
            moved.sort()
            print(f"    the position's score moved, in those flips: median {moved[len(moved) // 2]:.3f} max {moved[-1]:.3f}")
    # The baseline: no rule, two depth-18 searches of the same position.
    found = json.loads((CACHE / "single.json").read_text(encoding="utf-8"))
    db = sqlite3.connect(CACHE / "clean.sqlite")
    clean = {k: json.loads(f) for k, f in db.execute("SELECT key, found FROM clean")}
    db.close()
    flips = Counter()
    total = 0
    for r in found:
        if not r.get("children") or not r["one_lines"]:
            continue
        six = clean[r["key"]]["6"]["lines"]
        if not six:
            continue
        one_score, six_score = r["one_lines"][0]["expected_score"], six[0]["expected_score"]
        for c in r["children"]:
            if not c["lines"]:
                continue
            value = 1.0 - c["lines"][0]["expected_score"]
            total += 1
            a, b = name(max(0.0, one_score - value), th), name(max(0.0, six_score - value), th)
            if a != b:
                flips[(a, b)] += 1
    table(flips, total, "baseline, no rule (one-line depth 18 -> six-line depth 18)")


if __name__ == "__main__":
    main()
