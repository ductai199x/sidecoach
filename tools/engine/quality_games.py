"""How right each depth's move verdicts are in real games: every position of Tai's chess.com
games searched once, one line, one thread, cold, to depth `--truth` (24), its expected score at
every depth on the way kept (a single-thread search to depth d is exactly the first d iterations
of a deeper one). Each played move's loss at depth d -- the position before's score against the
position after's -- and its severity, against the same at the truth's depth.

    uv run --no-sync python tools/engine/quality_games.py --games 12 --die 1
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CACHE = ROOT / "data" / "engine-cache"
OUT = CACHE / "quality_games.json"
STOCKFISH = next(
    (Path(os.environ["LOCALAPPDATA"]) / "Sidecoach" / "data" / "tools" / "stockfish").glob("*.exe")
)
_W: dict[str, Any] = {}


def _init(cores: list[int]) -> None:
    import atexit

    import psutil

    from sidecoach.analysis.engine import Engine

    psutil.Process().cpu_affinity(cores)
    engine = Engine(path=STOCKFISH, threads=1, hash_mb=64, multipv=1, pv_plies=4)
    engine.open()
    assert engine._engine is not None
    psutil.Process(engine._engine.transport.get_pid()).cpu_affinity(cores)
    atexit.register(engine.close)
    _W["engine"] = engine


def _scores(task: tuple[str, list[str], int]) -> tuple[str, dict[int, float]]:
    """The position's expected score for the side to move, at every depth to `truth`."""
    fen, moves, truth = task
    board = chess.Board(fen)
    for m in moves:
        board.push_uci(m)
    if board.is_game_over():
        return board.fen(), {}
    engine = _W["engine"]
    engine._engine.configure({"Clear Hash": None})
    at: dict[int, float] = {}

    def on_depth(found: list) -> None:
        if found:
            at[found[0].depth] = found[0].expected_score

    lines = engine.analyse(board, depth=truth, multipv=1, on_depth=on_depth)
    if lines:
        at[lines[0].depth] = lines[0].expected_score
    return board.fen(), at


def main() -> None:
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--truth", type=int, default=24)
    parser.add_argument("--die", type=int, default=1)
    args = parser.parse_args()
    archive = json.loads((CACHE / "chesscom-nosegazm.json").read_text(encoding="utf-8"))
    games = [chess.pgn.read_game(io.StringIO(g["pgn"])) for g in archive[: args.games]]
    tasks, plies = [], []
    for g, game in enumerate(games):
        if game is None:
            continue
        root = game.board().fen()
        moves = [m.uci() for m in game.mainline_moves()]
        for i in range(len(moves) + 1):
            tasks.append((root, moves[:i], args.truth))
        plies.append((g, root, moves))
    cores = list(range(12 * args.die, 12 * args.die + 12))
    scores: dict[tuple[int, int], dict[int, float]] = {}
    index = {(t[0], len(t[1]), tuple(t[1])): k for k, t in enumerate(tasks)}
    pool = ProcessPoolExecutor(12, initializer=_init, initargs=(cores,))
    try:
        futures = {pool.submit(_scores, t): k for k, t in enumerate(tasks)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="positions", file=sys.stdout, mininterval=2):
            scores[futures[fut]] = fut.result()[1]
    finally:
        processes = list(pool._processes.values()) if pool._processes else []
        pool.shutdown(wait=False, cancel_futures=True)
        for p in processes:
            p.terminate()
    rows = []
    for g, root, moves in plies:
        for i in range(len(moves)):
            before = scores[index[(root, i, tuple(moves[:i]))]]
            after = scores[index[(root, i + 1, tuple(moves[: i + 1]))]]
            rows.append({"game": g, "ply": i + 1, "before": before, "after": after})
    OUT.write_text(json.dumps(rows), encoding="utf-8")
    report(rows, args.truth)


def report(rows: list[dict[str, Any]], truth: int) -> None:
    from sidecoach.analysis.score import Thresholds, severity

    th = Thresholds()

    def loss(r: dict[str, Any], d: int) -> float | None:
        b = r["before"].get(str(d), r["before"].get(d))
        if not r["after"]:
            return 0.0 if b is not None else None  # the move ended the game
        a = r["after"].get(str(d), r["after"].get(d))
        if b is None or a is None:
            return None
        return max(0.0, b - (1.0 - a))

    print(f"{len(rows)} moves; the truth: depth {truth}")
    for d in range(10, truth):
        same = n = two = 0
        for r in rows:
            t, x = loss(r, truth), loss(r, d)
            if t is None or x is None:
                continue
            n += 1
            st, sx = severity(t, th), severity(x, th)
            same += st == sx
            order = ["none", "inaccuracy", "mistake", "blunder"]
            two += abs(order.index(str(st.value)) - order.index(str(sx.value))) >= 2
        print(f"  depth {d}: severity as the truth's {same}/{n} ({same / max(1, n):.2%}); off by two levels or more: {two}")


if __name__ == "__main__":
    main()
