"""How many workers the live analyser needs: the design as built (no look-ahead, each position's
candidates searched, verdicts first, the AI coach's requests on) simulated on the measured
atoms for 2, 3, 4 and 6 workers, over every workload sweep2 uses. Prints each answer's latency
(p50 / p90 / max, seconds), the cores used and the CPU per move.

    uv run --no-sync python tools/engine/workers_compare.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.engine.sweep2 import Policy, _init, _one  # noqa: E402
from tools.engine.sweep2_report import q  # noqa: E402

WORKERS = (2, 3, 4, 6)


def main() -> None:
    from tqdm import tqdm

    from tools.engine.measure import workloads
    from tools.engine.sessions import bursts

    sessions = workloads()
    sessions.update(bursts(sessions))
    policies = [Policy(w, 0, 6, 0, 0.15, "verdicts", coach=True) for w in WORKERS]
    with ProcessPoolExecutor(len(policies), initializer=_init) as pool:
        jobs = pool.map(_one, [(p, sessions) for p in policies])
        results = list(tqdm(jobs, total=len(policies), desc="policies"))
    for policy, out, missing in results:
        runs = list(out.values())
        print(f"\n{policy.workers} workers ({len(missing)} positions unmeasured, left out)")
        series = {
            "verdict, user's move": [x for r in runs for x in r["verdict_user"]],
            "verdict, opponent's": [x for r in runs for x in r["verdict_opp"]],
            "first answer (hint)": [x for r in runs for x in r["hint"]],
            "position whole": [x for r in runs for x in r["deep"]],
        }
        for name in sorted({n for r in runs for n in r["asked"]}):
            series[f"coach: {name}"] = [x for r in runs for x in r["asked"].get(name, [])]
        for name, v in series.items():
            if v:
                p50, p90, top = q(v, 0.5), q(v, 0.9), max(v)
                print(f"  {name:<24} p50 {p50:5.2f}  p90 {p90:5.2f}  max {top:5.2f}  (n={len(v)})")
        cpu = sum(r["cpu"] for r in runs)
        wall = sum(r["wall"] for r in runs)
        moves = sum(r["moves"] for r in runs)
        print(f"  cores used {cpu / wall:.2f}, CPU per move {cpu / max(1, moves):.2f} s")


if __name__ == "__main__":
    main()
