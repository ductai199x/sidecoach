"""What every move of games at known strength cost, for fitting the performance rating
(docs/game-review-plan.md, phase 0): each game judged move by move as the review judges it --
the loss in expected score at depth 18 from the best line alone, which is all accuracy needs --
several engines side by side, one game each at a time. Written as one JSON line a game side:
its level, the rating its PGN gives, and the losses and accuracies of its moves.

    uv run --no-sync python tools/review/level_losses.py data/bots/games/*.pgn --out FILE
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import threading
from pathlib import Path

import chess
import chess.pgn

from sidecoach.analysis.annotator import GameAnnotator
from sidecoach.analysis.engine import Engine
from sidecoach.analysis.score import Thresholds
from sidecoach.config import Settings, find_stockfish


def games(paths: list[Path]) -> list[tuple[str, int, chess.pgn.Game]]:
    out = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            index = 0
            while (game := chess.pgn.read_game(handle)) is not None:
                index += 1
                out.append((path.stem, index, game))
    return out


def sides(name: str, index: int, game: chess.pgn.Game, annotator: GameAnnotator) -> list[dict]:
    annotator.reset()
    board = game.board()
    moves: dict[bool, list[tuple[float, float]]] = {chess.WHITE: [], chess.BLACK: []}
    for ply, move in enumerate(game.mainline_moves(), start=1):
        assessment = annotator.assess_move(f"{name}#{index}", board, move, ply)
        moves[board.turn].append((round(assessment.expected_loss, 4), assessment.accuracy or 0.0))
        board.push(move)
    out = []
    for colour, key in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        rating = re.sub(r"\D", "", game.headers.get(f"{key}Elo", "")) or None
        level = re.search(r"Nova (\d+)", game.headers.get(key, ""))
        out.append(
            {
                "game": f"{name}#{index}",
                "side": key.lower(),
                "level": int(level.group(1)) if level else None,
                "rating": int(rating) if rating else None,
                "result": game.headers.get("Result"),
                "loss": [m[0] for m in moves[colour]],
                "accuracy": [m[1] for m in moves[colour]],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--depth", type=int, default=18)
    args = parser.parse_args()
    settings = Settings()
    path = find_stockfish(settings)
    assert path is not None, "no Stockfish"
    todo: queue.SimpleQueue = queue.SimpleQueue()
    for item in games(args.pgn):
        todo.put(item)
    total = todo.qsize()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    done = [0]

    def work() -> None:
        with Engine(path=path, threads=2, hash_mb=64, multipv=1, max_seconds=5.0) as engine:
            annotator = GameAnnotator(
                engine=engine,
                depth=args.depth,
                multipv=1,
                thresholds=Thresholds(),
                with_threats=False,
            )
            while True:
                try:
                    name, index, game = todo.get_nowait()
                except queue.Empty:
                    return
                rows = sides(name, index, game, annotator)
                with lock:
                    with args.out.open("a", encoding="utf-8") as handle:
                        for row in rows:
                            handle.write(json.dumps(row) + "\n")
                    done[0] += 1
                    if done[0] % 20 == 0 or done[0] == total:
                        print(f"{done[0]}/{total} games", flush=True)

    threads = [threading.Thread(target=work, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
