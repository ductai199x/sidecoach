"""Two more atoms, on consecutive positions of real games: what a one-thread depth-18 search costs
by its number of lines (1, 2, 3, 4, 6), cold (the hash cleared first), and warm -- the game's
positions searched in order on one engine that keeps its hash, as a worker following one game
would. Each game runs in a process of its own, its engines pinned to a physical core of their
own (four at once: side by side costs four searches about 4%, atoms.py).

    uv run --no-sync python tools/engine/atoms_lines.py
    Get-Content "D:\\chess-coaching-companion\\data\\engine-cache\\atoms_lines.log" -Wait -Tail 2
"""

from __future__ import annotations

import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "engine"))

from atoms import STOCKFISH, cpu, engine_process  # noqa: E402

from sidecoach.analysis.engine import Engine  # noqa: E402

CACHE = ROOT / "data" / "engine-cache"
OUT = CACHE / "atoms_lines.json"
LINES = [1, 2, 3, 4, 6]
GAMES, PLIES = 8, 30
CORES = [12, 14, 16, 18]  # one physical core per process, on the second die


def _game(job: tuple[int, str, int]) -> list[dict[str, Any]]:
    import psutil

    index, pgn, core = job
    psutil.Process().cpu_affinity([core + 1])  # this process on the core's other half
    game = chess.pgn.read_game(io.StringIO(pgn))
    assert game is not None
    engines: dict[str, tuple[Engine, Any]] = {}
    for lines in LINES:
        for mode in ("cold", "warm"):
            if mode == "warm" and lines not in (1, 2, 6):
                continue
            e = Engine(path=STOCKFISH, threads=1, hash_mb=256, multipv=lines, pv_plies=12)
            e.open()
            p = engine_process(e)
            p.cpu_affinity([core])
            e.analyse(chess.Board(), depth=1)
            assert e._engine is not None
            e._engine.configure({"Clear Hash": None})
            engines[f"{lines}{mode}"] = (e, p)
    rows = []
    try:
        board = game.board()
        for ply, move in enumerate(game.mainline_moves()):
            if ply >= PLIES or board.is_game_over():
                break
            row: dict[str, Any] = {"game": index, "ply": ply, "fen": board.fen()}
            for name, (engine, process) in engines.items():
                assert engine._engine is not None
                if name.endswith("cold"):
                    engine._engine.configure({"Clear Hash": None})
                lines = int(name[0])
                c0, t0 = cpu(process), time.perf_counter()
                found = engine.analyse(board, depth=18, multipv=lines)
                row[name] = {"wall": time.perf_counter() - t0, "cpu": cpu(process) - c0}
                assert found and found[0].depth == 18
            rows.append(row)
            board.push(move)
    finally:
        for engine, _ in engines.values():
            engine.close()
    return rows


def main() -> None:
    from tqdm import tqdm

    archive = json.loads((CACHE / "chesscom-nosegazm.json").read_text(encoding="utf-8"))
    long_games = [g["pgn"] for g in archive if g["pgn"].count("%clk") >= 2 * PLIES][:GAMES]
    jobs = [(i, pgn, CORES[i % len(CORES)]) for i, pgn in enumerate(long_games)]
    rows: list[dict[str, Any]] = []
    # A pool of one process per core: a core's games run one after another, never two at once.
    pools = {core: ProcessPoolExecutor(1) for core in CORES}
    try:
        futures = [pools[job[2]].submit(_game, job) for job in jobs]
        with tqdm(total=len(jobs), desc="games", file=sys.stdout, mininterval=2) as bar:
            for fut in as_completed(futures):
                rows.extend(fut.result())
                bar.update(1)
                OUT.write_text(json.dumps(rows), encoding="utf-8")
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True)
    print(f"{len(rows)} positions -> {OUT}")


if __name__ == "__main__":
    main()
