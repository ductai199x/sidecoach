"""The last atoms, on the same positions as atoms.py (one thread, depth 18): the six-line search
of a position right after its two-line search on the same engine (what the hash saves), and each
side search on its own -- the threats, the traps, the declined recaptures, what a quiet best move
threatens -- cold.

    uv run --no-sync python tools/engine/atoms_more.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "engine"))

from atoms import ME, OUT as ALONE, cpu, make  # noqa: E402

from sidecoach.analysis.defences import DEFENCE_DEPTH, defence_lines  # noqa: E402
from sidecoach.analysis.threats import AIM_DEPTH, aim_lines, threat_moves  # noqa: E402
from sidecoach.analysis.traps import TRAP_DEPTH, trap_lines  # noqa: E402

OUT = ALONE.parent / "atoms_more.json"


def main() -> None:
    import psutil
    from tqdm import tqdm

    psutil.Process().cpu_affinity([ME])
    fens = [
        r["fen"] for r in json.loads(ALONE.read_text(encoding="utf-8"))["alone"] if not r["repeat"]
    ]
    engine, process = make(6, 1, [12])
    assert engine._engine is not None
    rows: list[dict[str, Any]] = []

    def timed(fn: Any) -> tuple[Any, dict[str, float]]:
        c0, t0 = cpu(process), time.perf_counter()
        out = fn()
        return out, {"wall": time.perf_counter() - t0, "cpu": cpu(process) - c0}

    try:
        for fen in tqdm(fens, desc="positions", file=sys.stdout, mininterval=2):
            board = chess.Board(fen)
            row: dict[str, Any] = {"fen": fen}
            engine._engine.configure({"Clear Hash": None})
            _, row["2"] = timed(lambda: engine.analyse(board, depth=18, multipv=2))
            six, row["6_after_2"] = timed(lambda: engine.analyse(board, depth=18, multipv=6))
            for name, fn in (
                ("threats", lambda: threat_moves(engine, board, depth=18)),
                ("traps", lambda: trap_lines(engine, board, six, depth=min(TRAP_DEPTH, 18))),
                (
                    "defences",
                    lambda: defence_lines(engine, board, six, depth=min(DEFENCE_DEPTH, 18)),
                ),
                ("aims", lambda: aim_lines(engine, board, six, depth=min(AIM_DEPTH, 18))),
            ):
                engine._engine.configure({"Clear Hash": None})
                _, row[name] = timed(fn)
            rows.append(row)
    finally:
        engine.close()
    OUT.write_text(json.dumps(rows), encoding="utf-8")
    base = {
        r["fen"]: r
        for r in json.loads(ALONE.read_text(encoding="utf-8"))["alone"]
        if not r["repeat"]
    }

    def q(v: list[float], p: float) -> float:
        v = sorted(v)
        return v[round(p * (len(v) - 1))]

    print(f"{len(rows)} positions")
    six_cold = [base[r["fen"]]["6x1"]["wall"] for r in rows]
    after = [r["6_after_2"]["wall"] for r in rows]
    print(
        f"  6 lines cold p50 {q(six_cold, 0.5):.3f}; right after its 2-line search p50 {q(after, 0.5):.3f}; "
        f"sum ratio {sum(after) / sum(six_cold):.2f}"
    )
    print(
        f"  2 lines then 6, together over 6 cold: {sum(r['2']['wall'] + r['6_after_2']['wall'] for r in rows) / sum(six_cold):.2f}"
    )
    for name in ("threats", "traps", "defences", "aims"):
        w = [r[name]["wall"] for r in rows]
        print(
            f"  {name}: p50 {q(w, 0.5):.3f} p90 {q(w, 0.9):.3f} sum over 6 cold {sum(w) / sum(six_cold):.3f}"
        )


if __name__ == "__main__":
    main()
