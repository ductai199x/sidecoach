"""The atoms measured by atoms.py, summed up: per variant, the wall time and CPU of one search
alone (each position's against its own one-thread time), when depth 12 arrives, the side
searches, the run-to-run noise, and the slowdown of searches side by side.

    uv run --no-sync python tools/engine/atoms_report.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "engine-cache" / "atoms.json"


def q(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[round(p * (len(values) - 1))]


def main() -> None:
    d = json.loads(DATA.read_text(encoding="utf-8"))
    rows = [r for r in d["alone"] if not r["repeat"]]
    repeats = [r for r in d["alone"] if r["repeat"]]
    print(f"{len(rows)} positions, {len(repeats)} searched again")
    base = {r["fen"]: r["6x1"]["wall"] for r in rows}
    print("\nA. one search alone (depth 18)")
    print(
        f"  {'variant':>14} {'wall p50':>9} {'p90':>6} {'mean':>6} {'cpu mean':>9} "
        f"{'wall/1x6 p50':>13} {'faster than 1x6':>16} {'d12 p50':>8}"
    )
    for name in ("1x1", "6x1", "1x2", "6x2", "1x4", "6x4", "6x8"):
        wall = [r[name]["wall"] for r in rows]
        cpu = [r[name]["cpu"] for r in rows]
        ratio = [r[name]["wall"] / r["6x1"]["wall"] for r in rows]
        faster = sum(1 for x in ratio if x < 1) / len(ratio)
        d12 = [r[name]["at12"] for r in rows if r[name]["at12"] is not None]
        lines, threads = name.split("x")
        print(
            f"  {lines} line(s) {threads}T {q(wall, 0.5):9.3f} {q(wall, 0.9):6.2f} "
            f"{statistics.mean(wall):6.3f} {statistics.mean(cpu):9.3f} {q(ratio, 0.5):13.2f} "
            f"{faster:15.0%} {q(d12, 0.5):8.3f}"
        )
    side = [r["side"]["wall"] for r in rows]
    side_ratio = [r["side"]["wall"] / r["6x1"]["wall"] for r in rows]
    print(
        f"  side searches: wall p50 {q(side, 0.5):.3f} p90 {q(side, 0.9):.3f} mean "
        f"{statistics.mean(side):.3f}; over the 6-line search p50 {q(side_ratio, 0.5):.2f}"
    )
    one_over_six = [r["1x1"]["wall"] / r["6x1"]["wall"] for r in rows]
    print(
        f"  1 line over 6 lines (1 thread): p50 {q(one_over_six, 0.5):.2f} "
        f"mean of sums {sum(r['1x1']['wall'] for r in rows) / sum(r['6x1']['wall'] for r in rows):.2f}"
    )
    noise = [abs(r["6x1"]["wall"] - base[r["fen"]]) / base[r["fen"]] for r in repeats]
    print(f"  noise, one thread searched again: p50 {q(noise, 0.5):.1%} max {max(noise):.1%}")
    print("\nB. one-thread 6-line searches side by side (each on its own physical core)")
    by_k: dict[int, list[float]] = {}
    for r in d["side_by_side"]:
        by_k.setdefault(r["k"], []).append(r["wall"] / base[r["fen"]])
    for k, ratios in sorted(by_k.items()):
        print(
            f"  k={k:2d}: slower than alone p50 x{q(ratios, 0.5):.2f} mean x{statistics.mean(ratios):.2f}"
        )


if __name__ == "__main__":
    main()
