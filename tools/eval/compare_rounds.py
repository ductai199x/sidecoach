"""Paired comparison of two judged runs on the same items: per class and overall, the
change in claim support, fabrication, completeness and pass, with a sign test on the
per-item claim-support deltas so a small pilot's noise is visible.

Run:  uv run python tools/eval/compare_rounds.py --a pilot-r1 --b pilot-r2
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JUDGMENTS = ROOT / "data" / "eval" / "judgments"


def load(run: str) -> dict[str, dict]:
    out = {}
    for path in sorted((JUDGMENTS / run).glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def sign_test(wins: int, losses: int) -> float:
    """Two-sided binomial p-value that wins and losses are equally likely."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    ids = sorted(set(a) & set(b))
    print(f"{len(ids)} items judged in both {args.a} and {args.b}")
    rows: dict[str, list[tuple[dict, dict]]] = {"all": []}
    for i in ids:
        rows["all"].append((a[i], b[i]))
        rows.setdefault(a[i]["class"], []).append((a[i], b[i]))

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    def fmt(v):
        return "-" if v is None else f"{v:+.2f}" if isinstance(v, float) else str(v)

    print(
        f"{'class':18s} {'n':>3s}  {'claim_support':>22s}  {'fabrication':>20s}"
        f"  {'completeness':>22s}  {'pass':>16s}   wins/losses  p"
    )
    for cls in ["all"] + sorted(k for k in rows if k != "all"):
        pairs = rows[cls]
        sa = [x["scores"] for x, _ in pairs]
        sb = [y["scores"] for _, y in pairs]
        cs_a, cs_b = (
            mean([s.get("claim_support") for s in sa]),
            mean([s.get("claim_support") for s in sb]),
        )
        fb_a, fb_b = (
            mean([1.0 if s["fabrication"] else 0.0 for s in sa]),
            mean([1.0 if s["fabrication"] else 0.0 for s in sb]),
        )
        co_a, co_b = mean([s["completeness"] for s in sa]), mean([s["completeness"] for s in sb])
        pa_a, pa_b = (
            mean([1.0 if s["pass"] else 0.0 for s in sa]),
            mean([1.0 if s["pass"] else 0.0 for s in sb]),
        )
        wins = losses = 0
        for x, y in pairs:
            u, v = x["scores"].get("claim_support"), y["scores"].get("claim_support")
            if u is None or v is None:
                continue
            if v > u + 1e-9:
                wins += 1
            elif u > v + 1e-9:
                losses += 1
        d = lambda p, q: None if p is None or q is None else q - p  # noqa: E731
        print(
            f"{cls:18s} {len(pairs):3d}  {fmt(cs_a) if cs_a is None else f'{cs_a:.2f}'}"
            f" -> {fmt(cs_b) if cs_b is None else f'{cs_b:.2f}'} ({fmt(d(cs_a, cs_b))})"
            f"  {fb_a:.2f} -> {fb_b:.2f} ({fmt(d(fb_a, fb_b))})"
            f"  {fmt(co_a) if co_a is None else f'{co_a:.2f}'}"
            f" -> {fmt(co_b) if co_b is None else f'{co_b:.2f}'} ({fmt(d(co_a, co_b))})"
            f"  {pa_a:.2f} -> {pa_b:.2f} ({fmt(d(pa_a, pa_b))})"
            f"   {wins}/{losses}  {sign_test(wins, losses):.2f}"
        )


if __name__ == "__main__":
    main()
