"""The one-line search's slowest positions, depth by depth: the 300 slowest of the 8,886 at
depth 18 (single.py) searched again, cold, one line, each depth's arrival timed. Every other
position reached depth 18 sooner than these did, so it reached every depth before it sooner too:
the worst time to each depth over all 8,886 is the worst of these.

    uv run --no-sync python tools/engine/single_tail.py
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.engine.measure import _W, CORES, _worker_init  # noqa: E402

CACHE = ROOT / "data" / "engine-cache"
OUT = CACHE / "single_tail.json"
SIX_AT_ONCE = 1.13


def _one(fen: str) -> dict[str, Any]:
    engine = _W["engine"]
    engine._engine.configure({"Clear Hash": None})
    marks: dict[int, float] = {}
    t0 = time.perf_counter()

    def on_depth(found: list) -> None:
        if found and found[0].depth not in marks:
            marks[found[0].depth] = (time.perf_counter() - t0) / SIX_AT_ONCE

    engine.analyse(chess.Board(fen), depth=18, multipv=1, on_depth=on_depth)
    return {"fen": fen, "total": (time.perf_counter() - t0) / SIX_AT_ONCE, "depths": marks}


def main() -> None:
    from tqdm import tqdm

    found = json.loads((CACHE / "single.json").read_text(encoding="utf-8"))
    slowest = sorted(found, key=lambda r: -r["one"])[:300]
    fens = [" ".join(r["key"].split(" ")[:6]) for r in slowest]
    out = []
    pool = ProcessPoolExecutor(len(CORES), initializer=_worker_init, initargs=(CORES,))
    try:
        futures = [pool.submit(_one, f) for f in fens]
        for fut in tqdm(as_completed(futures), total=len(futures), file=sys.stdout, mininterval=2):
            out.append(fut.result())
    finally:
        processes = list(pool._processes.values()) if pool._processes else []
        pool.shutdown(wait=False, cancel_futures=True)
        for p in processes:
            p.terminate()
    OUT.write_text(json.dumps(out), encoding="utf-8")

    def q(v: list[float], p: float) -> float:
        v = sorted(v)
        return v[min(len(v) - 1, int(p * len(v)))]

    print("the 300 slowest one-line searches of 8,886: time to each depth (s)")
    for d in range(10, 19):
        at = [r["depths"].get(d, r["depths"].get(str(d), r["total"])) for r in out]
        print(
            f"  d{d}: max {max(at):5.2f}  p99-of-300 {q(at, 0.99):.2f}  "
            f"over 1s {sum(x > 1 for x in at)}"
        )
    worst = max(out, key=lambda r: r["total"])
    print(f"slowest: {worst['total']:.1f}s {worst['fen']}")
    print(
        "   depths: "
        + " ".join(
            f"{d}:{t:.2f}" for d, t in sorted(worst["depths"].items(), key=lambda x: int(x[0]))
        )
    )


if __name__ == "__main__":
    main()
