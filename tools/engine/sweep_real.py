"""The sweep's clock checked against the real engine: the same scheduler (sweep.Run), its searches
made by real Stockfish workers (one thread each, each pinned to a physical core of its own) in
real time instead of taken from the measurements. Its decisions use the measured lines, so the
two runs decide alike and only the times can differ: what is compared is the timing model --
each search's measured time against its real one, and every answer's latency and the CPU.

    uv run --no-sync python tools/engine/sweep_real.py
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.engine.measure import STOCKFISH, workloads  # noqa: E402
from tools.engine.sweep import BESIDE, Job, Policy, Run, bursts, load  # noqa: E402

CORES = [12, 14, 16, 18, 20, 22]


class RealRun(Run):
    """Run, its searches real."""

    def __init__(self, policy: Policy, atoms: dict, steps: list[Any], engines: list[Any]) -> None:
        super().__init__(policy, atoms, steps)
        self.engines = engines
        self.done_q: queue.Queue[tuple[Job, float]] = queue.Queue()
        self.first_at: dict[int, float] = {}  # id(job) -> when its depth 12 came
        self.timed: list[tuple[str, float, float]] = []  # (kind, predicted, real)
        self.t0 = 0.0
        # One thread per worker, each taking its engine's searches one after another, as the
        # analyser's workers do: a search stopped ends before the next begins on that engine.
        self.inbox: list[queue.Queue[tuple[Job, float] | None]] = [queue.Queue() for _ in engines]
        self.threads = [
            threading.Thread(target=self._worker, args=(w,), daemon=True) for w in range(len(engines))
        ]
        for t in self.threads:
            t.start()

    def _worker(self, w: int) -> None:
        from sidecoach.analysis.engine import Interrupted
        from sidecoach.analysis.threats import threat_moves

        engine = self.engines[w]
        while True:
            item = self.inbox[w].get()
            if item is None:
                return
            job, predicted = item
            board = self.pos[job.key].board.copy()

            def first(found: list, job: Job = job) -> None:
                if found and found[0].depth >= 12 and id(job) not in self.first_at:
                    self.first_at[id(job)] = self.clock()

            try:
                with engine.interruptible():
                    if self.jobs.get(w) is not job:
                        continue  # stopped before it began
                    # It begins now: a search stopped on this engine just before took its time.
                    job.start = self.clock()
                    if job.kind == "threats":
                        threat_moves(engine, board, depth=18)
                    else:
                        engine.analyse(board, depth=18, multipv=int(job.kind), on_depth=first)
            except Interrupted:
                continue
            self.done_q.put((job, predicted))

    def close(self) -> None:
        for box in self.inbox:
            box.put(None)
        for t in self.threads:
            t.join()

    def clock(self) -> float:
        return time.perf_counter() - self.t0

    def assign(self) -> None:
        before = dict(self.jobs)
        super().assign()
        for w, job in self.jobs.items():
            if before.get(w) is not job:
                self._start(job)

    def _start(self, job: Job) -> None:
        predicted = job.end - job.start
        job.end = float("inf")  # known when it is done
        self.inbox[job.worker].put((job, predicted))

    def reframe(self) -> None:
        before = dict(self.jobs)
        super().reframe()
        for w, job in before.items():
            if self.jobs.get(w) is not job:
                self.engines[w].interrupt()

    def tell(self, key: str) -> None:
        """Run.tell with the first answer's real moment (its search's depth 12 as it came),
        never the measured one."""
        p = self.pos[key]
        if key in self._open_first:
            if "6" in p.done:
                self.first.append(max(0.0, min(p.done["6"], self.now) - self._open_first.pop(key)))
            else:
                job = next((j for j in self.jobs.values() if j.key == key and j.kind == "6"), None)
                if job is not None and id(job) in self.first_at:
                    at = self.first_at[id(job)]
                    self.first.append(max(0.0, at - self._open_first.pop(key)))
        if key in self._open_u and "6" in p.done and "threats" in p.done:
            self.whole.append(max(0.0, self.now - self._open_u.pop(key)))

    def run(self) -> dict[str, Any]:
        steps = sorted(self.steps, key=lambda s: s.t)
        base = steps[0].t
        self.t0 = time.perf_counter()
        i = 0
        while True:
            self.now = self.clock()
            self.assign()
            for key in list(self._open_first):
                self.tell(key)
            if i >= len(steps) and not self.owed():
                break
            wait = (steps[i].t - base) - self.clock() if i < len(steps) else 1.0
            try:
                job, predicted = self.done_q.get(timeout=max(0.0, min(wait, 0.02)))
            except queue.Empty:
                if i < len(steps) and self.clock() >= steps[i].t - base:
                    self.now = self.clock()
                    self.step(steps[i])
                    i += 1
                continue
            self.now = self.clock()
            if self.jobs.get(job.worker) is not job:
                continue  # stopped meanwhile
            job.end = self.now
            self.timed.append((job.kind, predicted, job.end - job.start))
            self.finish(job)
        return {
            "verdict_user": [x for u, x in self.verdicts if u],
            "verdict_opp": [x for u, x in self.verdicts if not u],
            "first": self.first,
            "whole": self.whole,
            "left_first": self.left_first,
            "left_whole": self.left_whole,
            "cpu": self.cpu,
            "wasted": self.wasted,
            "wall": self.clock(),
            "timed": self.timed,
        }


def main() -> None:
    import psutil

    from sidecoach.analysis.engine import Engine

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--look", type=int, default=2)
    args = parser.parse_args()
    policy = Policy(args.workers, 1, False, "2", args.look, 0.15, "newest")
    atoms = load()
    sessions = workloads()
    sessions.update(bursts(sessions))
    puzzle = sessions["puzzles #0"]
    slices = {
        "burst 0.5s #0": sessions["burst 0.5s #0"],
        "blitz x3": next(v for k, v in sessions.items() if k.startswith("blitz x3")),
        "puzzles, 2 min": [s for s in puzzle if s.t - puzzle[0].t <= 120],
    }
    psutil.Process().cpu_affinity([CORES[-1] + 1])
    engines = []
    for w in range(policy.workers):
        e = Engine(path=STOCKFISH, threads=1, hash_mb=256, multipv=6, pv_plies=12)
        e.open()
        assert e._engine is not None
        psutil.Process(e._engine.transport.get_pid()).cpu_affinity([CORES[w]])
        e.analyse(__import__("chess").Board(), depth=1)
        engines.append(e)
    try:
        for name, steps in slices.items():
            sim = Run(policy, atoms, steps).run()
            for e in engines:
                assert e._engine is not None
                e._engine.configure({"Clear Hash": None})
            real_run = RealRun(policy, atoms, steps, engines)
            try:
                real = real_run.run()
            finally:
                real_run.close()
            ratios = sorted(r / p for _, p, r in real["timed"] if p > 0.02)
            n = len(ratios)
            print(f"\n{name}: {n} searches; real over measured p10 {ratios[n // 10]:.2f} "
                  f"p50 {ratios[n // 2]:.2f} p90 {ratios[9 * n // 10]:.2f}")
            for key in ("verdict_user", "verdict_opp", "first", "whole"):
                s, r = sorted(sim[key]), sorted(real[key])
                def p(v: list[float], q: float) -> str:
                    return "-" if not v else f"{v[round(q * (len(v) - 1))]:.2f}"
                print(f"  {key:13s} sim p50 {p(s, .5)} p90 {p(s, .9)} (n {len(s)}) | "
                      f"real p50 {p(r, .5)} p90 {p(r, .9)} (n {len(r)})")
            print(f"  cpu sim {sim['cpu']:.1f}s real {real['cpu']:.1f}s; left {sim['left_whole']} / "
                  f"{real['left_whole']}")
    finally:
        for e in engines:
            e.close()
    _ = BESIDE


if __name__ == "__main__":
    main()
