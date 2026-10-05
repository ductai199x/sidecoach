"""One policy through every workload, timed and summed up: the check before a sweep.

    uv run --no-sync python tools/engine/sweep_one.py --workers 3 --look 2
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.engine.measure import workloads  # noqa: E402
from tools.engine.sweep import Missing, Policy, Run, bursts, load  # noqa: E402


def q(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[round(p * (len(values) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reserve", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--passed", default="2")
    parser.add_argument("--look", type=int, default=2)
    parser.add_argument("--reach", type=float, default=0.15)
    args = parser.parse_args()
    policy = Policy(args.workers, args.reserve, args.quick, args.passed, args.look, args.reach)
    atoms = load()
    sessions = workloads()
    sessions.update(bursts(sessions))
    by_kind: dict[str, list[dict]] = {}
    missing = 0
    began = time.perf_counter()
    for name, steps in sessions.items():
        run = Run(policy, atoms, steps)
        try:
            result = run.run()
        except Missing as m:
            print(f"{name}: a game position unmeasured ({len(m.boards)})")
            continue
        missing += len(run.missing)
        by_kind.setdefault(name.split(" #")[0], []).append(result)
    print(f"{policy.name()}: {len(sessions)} sessions in {time.perf_counter() - began:.1f}s; "
          f"{missing} look-ahead positions unmeasured")
    for kind, runs in by_kind.items():
        vu = [x for r in runs for x in r["verdict_user"]]
        vo = [x for r in runs for x in r["verdict_opp"]]
        first = [x for r in runs for x in r["first"]]
        whole = [x for r in runs for x in r["whole"]]
        cpu = sum(r["cpu"] for r in runs)
        wall = sum(r["wall"] for r in runs)
        moves = sum(r["moves"] for r in runs)
        fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
        print(
            f"  {kind:14s} verdict you p50 {fmt(q(vu, .5))} p90 {fmt(q(vu, .9))} | them p90 "
            f"{fmt(q(vo, .9))} | first p50 {fmt(q(first, .5))} p90 {fmt(q(first, .9))} | whole p50 "
            f"{fmt(q(whole, .5))} p90 {fmt(q(whole, .9))} | left {sum(r['left_first'] for r in runs)}"
            f"/{sum(r['left_whole'] for r in runs)} | cores {cpu / wall:.2f} cpu/move "
            f"{cpu / max(1, moves):.2f}s wasted {sum(r['wasted'] for r in runs) / max(cpu, 1e-9):.0%}"
        )
    _ = statistics


if __name__ == "__main__":
    main()
