"""What single.py found: the one-line depth-18 search's time on every position against the
two- and six-line searches', and on the sample the candidates found as their own positions'
one-line searches against the six-line search's lines -- time (all of it; and on three workers
side by side, the slowest worker's), the best move, the order, and each candidate's verdict.

    uv run --no-sync python tools/engine/single_report.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sidecoach.analysis.score import Thresholds, severity  # noqa: E402

CACHE = ROOT / "data" / "engine-cache"
SIX_AT_ONCE = 1.13


def q(v: list[float], p: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def tail(name: str, v: list[float]) -> None:
    print(
        f"  {name:24s} p50 {q(v, .5):.2f} p90 {q(v, .9):.2f} p99 {q(v, .99):.2f} "
        f"p99.9 {q(v, .999):.2f} max {max(v):5.1f} | over 2s {sum(x > 2 for x in v):4d} "
        f"over 5s {sum(x > 5 for x in v):3d}"
    )


def makespan(times: list[float], workers: int) -> float:
    """The slowest worker's total, the longest first to the least loaded (LPT)."""
    load = [0.0] * workers
    for t in sorted(times, reverse=True):
        load[load.index(min(load))] += t
    return max(load)


def score_of(line: dict) -> float:
    """A line's expected score for the side to move in its position."""
    return float(line["expected_score"])


def main() -> None:
    found = json.loads((CACHE / "single.json").read_text(encoding="utf-8"))
    db = sqlite3.connect(CACHE / "clean.sqlite")
    clean = {k: json.loads(f) for k, f in db.execute("SELECT key, found FROM clean")}
    db.close()
    print(f"{len(found)} positions, one thread, cold, depth 18 (times over one search alone)")
    tail("one line", [r["one"] for r in found])
    tail("two lines", [clean[r["key"]]["2"]["wall"] / SIX_AT_ONCE for r in found])
    tail("six lines", [clean[r["key"]]["6"]["wall"] / SIX_AT_ONCE for r in found])
    sample = [r for r in found if r.get("children")]
    print(f"\n{len(sample)} positions: the six candidates as their own positions' one-line searches")
    total = [r["one"] + sum(c["s"] for c in r["children"]) for r in sample]
    wall3 = [makespan([r["one"]] + [c["s"] for c in r["children"]], 3) for r in sample]
    worst_child = [max([c["s"] for c in r["children"]] or [0]) for r in sample]
    six = [clean[r["key"]]["6"]["wall"] / SIX_AT_ONCE for r in sample]
    tail("children: all of it", total)
    tail("children: 3 workers", wall3)
    tail("children: slowest one", worst_child)
    tail("six lines (same sample)", six)
    print(f"  CPU, all of it: children {sum(total):.0f}s vs six lines {sum(six):.0f}s")
    th = Thresholds()
    same_best = same_order = agree = moves = 0
    for r in sample:
        lines6 = clean[r["key"]]["6"]["lines"]
        # A candidate's value for the side to move here: one minus its position's own.
        kids = [
            (c["move"], 1.0 - score_of(c["lines"][0]) if c["lines"] else None)
            for c in r["children"]
        ]
        kids = [(m, v) for m, v in kids if v is not None]
        if not kids or not lines6:
            continue
        by_kids = [m for m, _ in sorted(kids, key=lambda mv: -mv[1])]
        by_six = [ln["moves"][0] for ln in lines6 if ln["moves"]]
        same_best += by_kids[0] == by_six[0]
        same_order += by_kids == by_six[: len(by_kids)]
        best_k = max(v for _, v in kids)
        best_6 = score_of(lines6[0])
        six_of = {ln["moves"][0]: score_of(ln) for ln in lines6 if ln["moves"]}
        for m, v in kids:
            if m not in six_of:
                continue
            moves += 1
            agree += severity(max(0.0, best_k - v), th) == severity(max(0.0, best_6 - six_of[m]), th)
    n = len(sample)
    print(
        f"  best move the same: {same_best}/{n}; the whole order the same: {same_order}/{n}; "
        f"each candidate's verdict (severity) the same: {agree}/{moves}"
    )


if __name__ == "__main__":
    main()
