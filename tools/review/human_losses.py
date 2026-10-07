"""What every move of real players' games cost, for fitting the performance rating
(docs/game-review.md, phase 0). The games are the Nova authors' sample of Lichess rapid
(March 2026; `data/models/nova/unified_sample_600k.jsonl`, 100,000 positions a band from 800 to
2300 and up), which holds each game side whole, from the player's first move to their last,
with their rating. Each move is judged as the review judges it -- `GameAnnotator.assess_move`,
the loss in expected score from the best line at depth 18 -- several engines side by side.
Written as one JSON line a game side: the player, their rating (Lichess rapid), and per move
the ply, the loss, the accuracy, the label, the mover's expected score before it and their clock.

    uv run --no-sync python tools/review/human_losses.py --per-band 100 --out FILE
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import random
import threading
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import chess

from sidecoach.analysis.annotator import GameAnnotator
from sidecoach.analysis.engine import Engine
from sidecoach.analysis.score import Thresholds
from sidecoach.config import Settings, find_stockfish

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data" / "models" / "nova" / "unified_sample_600k.jsonl"


def game_sides(path: Path = SAMPLE) -> list[list[dict]]:
    """The sample's game sides that are whole: a player's positions of one game, every other
    ply without a gap from their first move. A side is keyed by the player and the game's
    ratings and result, which the sample repeats on each of its rows."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["player_id"], row["rating"], row["opp_rating"], row["result"])
            grouped[key].append(row)
    sides = []
    for rows in grouped.values():
        rows.sort(key=lambda r: r["ply"])
        plies = [r["ply"] for r in rows]
        if plies[0] in (0, 1) and all(b - a == 2 for a, b in pairwise(plies)):
            sides.append(rows)
    return sides


def chosen(sides: list[list[dict]], per_band: int, seed: int) -> list[list[dict]]:
    """`per_band` sides of each band, drawn at random with a fixed seed."""
    by_band: dict[str, list[list[dict]]] = defaultdict(list)
    for side in sides:
        by_band[side[0]["band"]].append(side)
    rng = random.Random(seed)
    out = []
    for band in sorted(by_band):
        pool = by_band[band]
        out.extend(rng.sample(pool, min(per_band, len(pool))))
    return out


def paired(sides: list[list[dict]]) -> list[tuple[str, list[dict]]]:
    """The games the sample holds both sides of, each side with its game: the other side has
    the ratings swapped and the result negated, and each side's first move leads to the
    other's next position."""

    def after(row: dict) -> str:
        board = chess.Board(row["fen"])
        board.push(played(board, row["actual"]))
        return " ".join(board.fen().split()[:2])

    def head(row: dict) -> str:
        return " ".join(row["fen"].split()[:2])

    by_game: dict[tuple, list[list[dict]]] = defaultdict(list)
    for side in sides:
        first = side[0]
        by_game[(first["rating"], first["opp_rating"], first["result"])].append(side)
    out = []
    for white in sides:
        first = white[0]
        if first["ply"] != 0 or len(white) < 2:
            continue
        for black in by_game[(first["opp_rating"], first["rating"], -first["result"])]:
            if (
                black[0]["ply"] == 1
                and head(black[0]) == after(first)
                and head(white[1]) == after(black[0])
            ):
                game = f"{first['player_id']}-{black[0]['player_id']}-{first['rating']}"
                out.extend([(game, white), (game, black)])
                break
    return out


def played(board: chess.Board, uci: str) -> chess.Move:
    """The sample's move: a promotion is written without its piece, and is to a queen."""
    move = chess.Move.from_uci(uci)
    if move not in board.legal_moves:
        move = chess.Move.from_uci(uci + "q")
    return move


def judged(side: list[dict], annotator: GameAnnotator, game: str | None = None) -> dict:
    """One side's moves judged in order; `game` names the game when both its sides are."""
    annotator.reset()
    first = side[0]
    game_id = f"{first['player_id']}:{first['rating']}:{first['opp_rating']}"
    moves = []
    for row in side:
        board = chess.Board(row["fen"])
        before = annotator.measure_for_verdict(board)
        assessment = annotator.assess_move(game_id, board, played(board, row["actual"]), row["ply"])
        clock = row["white_clock"] if board.turn == chess.WHITE else row["black_clock"]
        moves.append(
            {
                "ply": row["ply"],
                "loss": round(assessment.expected_loss, 4),
                "accuracy": round(assessment.accuracy or 0.0, 2),
                "label": assessment.label.value if assessment.label else None,
                "score": round(before.score, 4),
                "clock": clock,
            }
        )
    return {
        "player": first["player_id"],
        "band": first["band"],
        "rating": first["rating"],
        "opp_rating": first["opp_rating"],
        "result": first["result"],
        "side": "white" if first["ply"] % 2 == 0 else "black",
        "game": game,
        "moves": moves,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-band", type=int, default=100)
    parser.add_argument(
        "--paired", action="store_true", help="the games the sample holds both sides of, instead"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=22)
    parser.add_argument("--depth", type=int, default=18)
    args = parser.parse_args()
    if os.name == "nt":  # hours of every core: the engines, started after, yield to the desktop
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # type: ignore[attr-defined]
    path = find_stockfish(Settings())
    assert path is not None, "no Stockfish"
    done_keys: set[tuple] = set()
    if args.out.exists():  # a run stopped part way goes on where it was
        for line in args.out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done_keys.add((row["player"], row["rating"], row["opp_rating"], row["result"]))
    todo: queue.SimpleQueue = queue.SimpleQueue()
    sides = game_sides()
    if args.paired:
        work_list = paired(sides)
    else:
        work_list = [(None, s) for s in chosen(sides, args.per_band, args.seed)]
    for game, side in work_list:
        first = side[0]
        key = (first["player_id"], first["rating"], first["opp_rating"], first["result"])
        if key not in done_keys:
            todo.put((game, side))
    total = todo.qsize()
    print(f"{total} sides to judge ({len(done_keys)} done before)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    done = [0]

    def work() -> None:
        with Engine(path=path, threads=1, hash_mb=64, multipv=1, max_seconds=5.0) as engine:
            annotator = GameAnnotator(
                engine=engine,
                depth=args.depth,
                multipv=1,
                thresholds=Thresholds(),
                with_threats=False,
            )
            while True:
                try:
                    game, side = todo.get_nowait()
                except queue.Empty:
                    return
                try:
                    row = judged(side, annotator, game)
                except Exception as error:  # one bad side is skipped, the worker goes on
                    print(f"skipped player {side[0]['player_id']}: {error}", flush=True)
                    continue
                with lock:
                    with args.out.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row) + "\n")
                    done[0] += 1
                    if done[0] % 25 == 0 or done[0] == total:
                        print(f"{done[0]}/{total} sides", flush=True)

    threads = [threading.Thread(target=work, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
