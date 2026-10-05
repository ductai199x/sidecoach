"""The single-line design measured on every workload position: one line to depth 18, cold, on
each of the 8,886 positions (does a one-line search ever explode, as two- and six-line ones
did?), and on a sample of 1,000 the candidates the coach reads -- the six best moves -- found as
their own positions' one-line searches to depth 17 (18 from the position), against the
six-line search's: their time, their total, and whether the moves and their order agree.

Six processes, each on a physical core of its own (measured: six at once run 1.13 times as long
as one alone; times here are divided by it).

    uv run --no-sync python tools/engine/single.py
    Get-Content "D:\\chess-coaching-companion\\data\\engine-cache\\single.log" -Wait -Tail 2
"""

from __future__ import annotations

import json
import random
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.engine.measure import _W, CORES, DB, _board, _worker_init  # noqa: E402

OUT = ROOT / "data" / "engine-cache" / "single.json"
SIX_AT_ONCE = 1.13
SAMPLE = 1000


def _timed(board: chess.Board, depth: int) -> tuple[float, list[dict[str, Any]]]:
    engine = _W["engine"]
    engine._engine.configure({"Clear Hash": None})
    t0 = time.perf_counter()
    found = engine.analyse(board, depth=depth, multipv=1)
    return (time.perf_counter() - t0) / SIX_AT_ONCE, [ln.model_dump() for ln in found]


def _one(task: tuple[str, str, list[str], list[str] | None]) -> dict[str, Any]:
    key, root, moves, candidates = task
    board = _board(root, moves)
    took, lines = _timed(board, 18)
    out: dict[str, Any] = {"key": key, "one": took, "one_lines": lines}
    if candidates:
        kids = []
        for uci in candidates:
            child = board.copy()
            child.push_uci(uci)
            if child.is_game_over():
                kids.append({"move": uci, "s": 0.0, "lines": []})
                continue
            t, ln = _timed(child, 17)
            kids.append({"move": uci, "s": t, "lines": ln})
        out["children"] = kids
    return out


def main() -> None:
    from tqdm import tqdm

    db = sqlite3.connect(DB)
    rows = {k: json.loads(f) for k, f in db.execute("SELECT key, found FROM clean")}
    db.close()
    keys = sorted(rows)
    random.Random(5).shuffle(keys)
    sample = set(keys[:SAMPLE])
    # Each position from its FEN alone: the moves a key carries after it matter to the engine
    # only for repetitions, never to what a search costs.
    tasks = []
    for key in keys:
        fen = " ".join(key.split(" ")[:6])
        six = rows[key]["6"]["lines"]
        cands = [ln["moves"][0] for ln in six if ln["moves"]] if key in sample else None
        tasks.append((key, fen, [], cands))
    out: list[dict[str, Any]] = []
    pool = ProcessPoolExecutor(len(CORES), initializer=_worker_init, initargs=(CORES,))
    try:
        futures = [pool.submit(_one, t) for t in tasks]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="positions",
            file=sys.stdout,
            mininterval=2,
        ):
            out.append(fut.result())
    finally:
        processes = list(pool._processes.values()) if pool._processes else []
        pool.shutdown(wait=False, cancel_futures=True)
        for p in processes:
            p.terminate()
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"{len(out)} positions -> {OUT}")


if __name__ == "__main__":
    main()
