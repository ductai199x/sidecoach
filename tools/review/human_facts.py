"""Real players' games judged as the review judges them, with the facts Stats keeps
(docs/stats-plan.md, S2): the peers' column, and the category ratings' fit. The games are the
Nova authors' Lichess rapid sample (`human_losses.game_sides`); each position the player moved
from, and the one their move led to, is measured whole as the review measures it (two lines, the
tagger's motifs), the move assessed, and its facts read by the app's own definitions
(`profile.stat_facts`: the tactic chance and whether it was found, the one allowed, the
structure, the endgame's kind). One JSON line a side.

    uv run --no-sync python tools/review/human_facts.py --per-band 200 --out FILE
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import threading
from pathlib import Path

import chess
from human_losses import chosen, game_sides, played

from sidecoach.analysis.annotator import GameAnnotator
from sidecoach.analysis.engine import Engine
from sidecoach.analysis.phase import phase_of
from sidecoach.analysis.score import Thresholds, position_score
from sidecoach.cli import _load_book, _load_tagger
from sidecoach.config import Settings, find_stockfish
from sidecoach.profile.stat_facts import (
    FOUND_LOSS,
    chance_in,
    endgame_of,
    structure_of,
)

MULTIPV = 2  # what the decisive call needs: the best line against the next


def judged(side: list[dict], annotator: GameAnnotator) -> dict:
    """One side's moves, each with its facts."""
    annotator.reset()
    first = side[0]
    game_id = f"{first['player_id']}:{first['rating']}:{first['opp_rating']}"
    moves = []
    for row in side:
        board = chess.Board(row["fen"])
        # The sample writes every position's counters as "0 1": the phase reads the move number.
        board.fullmove_number = row["ply"] // 2 + 1
        move = played(board, row["actual"])
        before = annotator.analysis_event(game_id, board, row["ply"])
        after_board = board.copy(stack=False)
        after_board.push(move)
        after = annotator.analysis_event(game_id, after_board, row["ply"] + 1)
        assessment = annotator.assess_move(game_id, board, move, row["ply"])
        phase = phase_of(board)
        chance = chance_in(before)
        moves.append(
            {
                "ply": row["ply"],
                "phase": phase,
                "loss": round(assessment.expected_loss, 4),
                "label": assessment.label.value if assessment.label else None,
                "accuracy": round(assessment.accuracy or 0.0, 2),
                "score": round(position_score(board, before.lines), 4),
                "chance": chance,
                "found": chance is not None and assessment.expected_loss < FOUND_LOSS,
                "allowed": chance_in(after),
                "structure": structure_of(board),
                "endgame": endgame_of(board) if phase == "endgame" else None,
                "clock": row["white_clock"] if board.turn == chess.WHITE else row["black_clock"],
            }
        )
    return {
        "player": first["player_id"],
        "band": first["band"],
        "rating": first["rating"],
        "opp_rating": first["opp_rating"],
        "result": first["result"],
        "side": "white" if first["ply"] % 2 == 0 else "black",
        "moves": moves,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-band", type=int, default=200)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=22)
    parser.add_argument("--depth", type=int, default=18)
    args = parser.parse_args()
    if os.name == "nt":  # hours of every core: the engines, started after, yield to the desktop
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x4000)
    settings = Settings()
    path = find_stockfish(settings)
    assert path is not None, "no Stockfish"
    tagger, book = _load_tagger(), _load_book(settings)
    done_keys: set[tuple] = set()
    if args.out.exists():  # a run stopped part way goes on where it was
        for line in args.out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done_keys.add((row["player"], row["rating"], row["opp_rating"], row["result"]))
    todo: queue.SimpleQueue = queue.SimpleQueue()
    for side in chosen(game_sides(), args.per_band, args.seed):
        first = side[0]
        key = (first["player_id"], first["rating"], first["opp_rating"], first["result"])
        if key not in done_keys:
            todo.put(side)
    total = todo.qsize()
    print(f"{total} sides to judge ({len(done_keys)} done before)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    done = [0]

    def work() -> None:
        with Engine(path=path, threads=1, hash_mb=64, multipv=MULTIPV, max_seconds=5.0) as engine:
            annotator = GameAnnotator(
                engine=engine,
                depth=args.depth,
                multipv=MULTIPV,
                thresholds=Thresholds.from_settings(settings.coach),
                tagger=tagger,
                book=book,
                with_threats=False,
            )
            while True:
                try:
                    side = todo.get_nowait()
                except queue.Empty:
                    return
                try:
                    row = judged(side, annotator)
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
