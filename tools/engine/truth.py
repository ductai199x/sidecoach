"""Which way of finding a position's candidates is right where the two disagree: the six-line
search at depth 18, or each candidate's own position searched with one line to depth 17
(single.py's sample). The truth: each candidate's own position searched to depth 23 (24 from
here), one line, cold. For 300 of the sample's positions: the true best move, and how much
each method's best loses against it (in expected score).

    uv run --no-sync python tools/engine/truth.py
"""

from __future__ import annotations

import json
import random
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.engine.measure import CORES, _W, _worker_init  # noqa: E402

CACHE = ROOT / "data" / "engine-cache"
OUT = CACHE / "truth.json"
DEEP = 23


def _truth(task: tuple[str, list[str]]) -> dict[str, Any]:
    fen, moves = task
    engine = _W["engine"]
    board = chess.Board(fen)
    values = {}
    for uci in moves:
        child = board.copy()
        child.push_uci(uci)
        if child.is_game_over():
            values[uci] = 1.0 if child.is_checkmate() else 0.5
            continue
        engine._engine.configure({"Clear Hash": None})
        lines = engine.analyse(child, depth=DEEP, multipv=1)
        values[uci] = 1.0 - lines[0].expected_score if lines else 0.5
    return {"fen": fen, "values": values}


def main() -> None:
    from tqdm import tqdm

    found = json.loads((CACHE / "single.json").read_text(encoding="utf-8"))
    sample = [r for r in found if r.get("children")]
    random.Random(9).shuffle(sample)
    sample = sample[:300]
    db = sqlite3.connect(CACHE / "clean.sqlite")
    clean = {k: json.loads(f) for k, f in db.execute("SELECT key, found FROM clean")}
    db.close()
    tasks, meta = [], {}
    for r in sample:
        fen = " ".join(r["key"].split(" ")[:6])
        six = [ln["moves"][0] for ln in clean[r["key"]]["6"]["lines"] if ln["moves"]]
        kids = {c["move"]: (1.0 - c["lines"][0]["expected_score"]) for c in r["children"] if c["lines"]}
        if not six or not kids:
            continue
        tasks.append((fen, six))
        meta[fen] = (six[0], max(kids, key=kids.get))
    out = []
    pool = ProcessPoolExecutor(len(CORES), initializer=_worker_init, initargs=(CORES,))
    try:
        futures = [pool.submit(_truth, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), file=sys.stdout, mininterval=2):
            out.append(fut.result())
    finally:
        processes = list(pool._processes.values()) if pool._processes else []
        pool.shutdown(wait=False, cancel_futures=True)
        for p in processes:
            p.terminate()
    OUT.write_text(json.dumps(out), encoding="utf-8")
    six_right = kids_right = differ = 0
    six_loss, kids_loss = [], []
    for r in out:
        v = r["values"]
        best = max(v, key=v.get)
        by_six, by_kids = meta[r["fen"]]
        six_right += by_six == best
        kids_right += by_kids == best
        differ += by_six != by_kids
        six_loss.append(v[best] - v.get(by_six, v[best]))
        kids_loss.append(v[best] - v.get(by_kids, v[best]))
    n = len(out)

    def big(v: list[float]) -> int:
        return sum(x > 0.02 for x in v)

    print(f"{n} positions, the truth each candidate searched to depth {DEEP + 1}:")
    print(f"  the two methods' best moves differ in {differ}")
    print(f"  six lines' best is the true best in {six_right}; loses over 0.02 in {big(six_loss)}; "
          f"mean loss {sum(six_loss) / n:.4f}")
    print(f"  candidates' own searches' best is the true best in {kids_right}; loses over 0.02 in "
          f"{big(kids_loss)}; mean loss {sum(kids_loss) / n:.4f}")


if __name__ == "__main__":
    main()
