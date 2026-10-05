"""sweep2's results summed up: per policy and per kind of workload (puzzles, the time controls,
bullet-like, browsing, the bursts), each answer's latency (p50, p90, max), what was left
unanswered while wanted, the cores used, the CPU per move and the share thrown away; and the
front: the policies no other beats on cores and every latency at once, over all workloads.

    uv run --no-sync python tools/engine/sweep2_report.py [--top 15]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CACHE = Path(__file__).resolve().parents[2] / "data" / "engine-cache"
ANSWERS = ("verdict_user", "verdict_opp", "hint", "deep")


def q(v: list[float], p: float) -> float | None:
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def kind(session: str) -> str:
    return session.split(" #")[0]


def pooled(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for a in ANSWERS:
        v = [x for r in runs for x in r[a]]
        out[a] = {
            "p50": q(v, 0.5),
            "p90": q(v, 0.9),
            "p99": q(v, 0.99),
            "max": max(v) if v else None,
            "n": len(v),
        }
    cpu = sum(r["cpu"] for r in runs)
    wall = sum(r["wall"] for r in runs)
    out["cores"] = cpu / wall if wall else 0.0
    out["cpu_per_move"] = cpu / max(1, sum(r["moves"] for r in runs))
    out["wasted"] = sum(r["wasted"] for r in runs) / cpu if cpu else 0.0
    out["left_hint"] = sum(r["left_hint"] for r in runs)
    out["left_deep"] = sum(r["left_deep"] for r in runs)
    return out


def summarise(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        by: dict[str, list[dict[str, Any]]] = {}
        for name, run in r["sessions"].items():
            by.setdefault(kind(name), []).append(run)
        every = [run for runs in by.values() for run in runs]
        rows.append(
            {
                "name": r["name"],
                "policy": r["policy"],
                "all": pooled(every),
                "kinds": {k: pooled(v) for k, v in by.items()},
            }
        )
    return rows


def front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The policies no other is at least as good as on every measure and better on one."""

    def measures(r: dict[str, Any]) -> list[float]:
        a = r["all"]
        return [a["cores"], *(a[x]["p90"] or 0.0 for x in ANSWERS), a["left_hint"], a["left_deep"]]

    m = [measures(r) for r in rows]
    keep = []
    for i, mi in enumerate(m):
        beaten = any(
            all(x <= y for x, y in zip(mj, mi, strict=True))
            and any(x < y for x, y in zip(mj, mi, strict=True))
            for j, mj in enumerate(m)
            if j != i
        )
        if not beaten:
            keep.append(rows[i])
    return sorted(keep, key=lambda r: r["all"]["cores"])


def fmt(v: float | None) -> str:
    return "  -  " if v is None else f"{v:5.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    results = json.loads((CACHE / "sweep2.json").read_text(encoding="utf-8"))
    rows = summarise(results)
    (CACHE / "sweep2_summary.json").write_text(json.dumps(rows), encoding="utf-8")
    best = front(rows)
    print(
        f"{len(rows)} policies; {len(best)} on the front "
        "(cores; p90 of verdict you/them, hint, deep; left)"
    )
    for r in best[: args.top]:
        a = r["all"]
        print(
            f"  {r['name']:42s} cores {a['cores']:.2f} | verdict {fmt(a['verdict_user']['p90'])} "
            f"{fmt(a['verdict_opp']['p90'])} | hint {fmt(a['hint']['p90'])} | deep "
            f"{fmt(a['deep']['p90'])} | left {a['left_hint']}/{a['left_deep']} "
            f"| waste {a['wasted']:.0%}"
        )


if __name__ == "__main__":
    main()
