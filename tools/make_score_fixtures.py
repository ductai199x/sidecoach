"""Record what the desktop's scoring says, for the Kotlin port to be held to.

Everything the coach says about a move comes through these functions: what the move cost in
expected score, what that loss is called, how accurate it was, and whether the position was
a moment where the choice mattered. They are pure, so unlike a search they can be compared
exactly -- and they must be, because the thresholds were fitted to chess.com's own review
over the user's games and a port that rounds differently would rename his mistakes.

    cd D:/chess-coaching-companion
    uv run --no-sync python D:/sidecoach-android/tools/make_score_fixtures.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402

from sidecoach.analysis.score import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    best_gap,
    engine_expected_score,
    eval_loss,
    game_accuracy,
    human_expected_score,
    is_decisive,
    is_only_move,
    label_move,
    move_accuracy,
    position_score,
    severity,
    terminal_expected_score,
)
from sidecoach.events import Line  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"

POSITIONS = [
    chess.STARTING_FEN,
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
    "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",   # checkmate
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",   # stalemate
    "8/8/8/4k3/8/8/8/4KB2 w - - 0 1",   # king and bishop: insufficient
    "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
]


def a_line(rng: random.Random) -> dict:
    """A plausible engine line: a score or a mate, with win-draw-loss numbers."""
    mate = rng.choice([None, None, None, None, rng.choice([-5, -2, -1, 1, 2, 6])])
    cp = None if mate is not None else rng.randint(-1500, 1500)
    wins = rng.randint(0, 1000)
    draws = rng.randint(0, 1000 - wins)
    wdl = [wins, draws, 1000 - wins - draws]
    return {
        "moves": [m.uci() for m in _some_moves(rng)],
        "depth": rng.randint(8, 24),
        "score_cp": cp,
        "mate": mate,
        "wdl": wdl,
    }


def _some_moves(rng: random.Random) -> list[chess.Move]:
    board = chess.Board()
    out = []
    for _ in range(rng.randint(1, 6)):
        moves = list(board.legal_moves)
        if not moves:
            break
        move = rng.choice(moves)
        out.append(move)
        board.push(move)
    return out


def to_line(raw: dict, model: str) -> Line:
    wdl = tuple(raw["wdl"])
    if model == "engine":
        expected = engine_expected_score(wdl)
    else:
        expected = human_expected_score(raw["score_cp"], raw["mate"])
    return Line(
        moves=raw["moves"],
        depth=raw["depth"],
        expected_score=expected,
        score_cp=raw["score_cp"],
        mate=raw["mate"],
        wdl=wdl,
    )


def main() -> None:
    rng = random.Random(20260922)
    curves = []
    for _ in range(400):
        mate = rng.choice([None, None, None, rng.randint(-8, 8) or 1])
        cp = None if mate is not None else rng.randint(-2000, 2000)
        wins = rng.randint(0, 1000)
        draws = rng.randint(0, 1000 - wins)
        wdl = [wins, draws, 1000 - wins - draws]
        curves.append(
            {
                "score_cp": cp,
                "mate": mate,
                "wdl": wdl,
                "human": human_expected_score(cp, mate),
                "engine": engine_expected_score(tuple(wdl)),
            }
        )

    labels = []
    for _ in range(600):
        loss = rng.choice(
            [rng.uniform(0.0, 0.05), rng.uniform(0.0, 0.5), rng.choice([0.0, 0.02, 0.04, 0.10, 0.28])]
        )
        args = {
            "loss": loss,
            "is_best": rng.random() < 0.3,
            "in_book": rng.random() < 0.2,
            "only_move": rng.random() < 0.2,
            "sacrifice": rng.random() < 0.15,
            "before_score": rng.uniform(0.0, 1.0),
            "opponent_erred": rng.random() < 0.25,
            "obvious": rng.random() < 0.2,
        }
        labels.append(
            {
                **args,
                "label": label_move(
                    loss,
                    DEFAULT_THRESHOLDS,
                    is_best=args["is_best"],
                    in_book=args["in_book"],
                    only_move=args["only_move"],
                    sacrifice=args["sacrifice"],
                    before_score=args["before_score"],
                    opponent_erred=args["opponent_erred"],
                    obvious=args["obvious"],
                ).value,
                "severity": severity(loss, DEFAULT_THRESHOLDS).value,
                "accuracy": move_accuracy(loss),
            }
        )

    positions = []
    for fen in POSITIONS:
        board = chess.Board(fen)
        for _ in range(6):
            model = rng.choice(["human", "engine"])
            raw = [a_line(rng) for _ in range(rng.randint(0, 5))]
            lines = [to_line(r, model) for r in raw]
            # The lines a real engine gives are sorted best first.
            order = sorted(range(len(lines)), key=lambda i: -lines[i].expected_score)
            raw = [raw[i] for i in order]
            lines = [lines[i] for i in order]
            played = next(iter(board.legal_moves), None)
            positions.append(
                {
                    "fen": fen,
                    "model": model,
                    "lines": raw,
                    "expected": [line.expected_score for line in lines],
                    "best_gap": best_gap(lines),
                    "only_move": is_only_move(board, lines, DEFAULT_THRESHOLDS),
                    "decisive": is_decisive(board, lines, DEFAULT_THRESHOLDS),
                    "position_score": position_score(board, lines),
                    "terminal": terminal_expected_score(board),
                    "played": played.uci() if played is not None else None,
                    "after_score": rng.uniform(0.0, 1.0),
                }
            )
    for entry in positions:
        if entry["played"] is None:
            entry["eval_loss"] = None
            continue
        model = entry["model"]
        lines = [to_line(r, model) for r in entry["lines"]]
        entry["eval_loss"] = eval_loss(lines, entry["played"], entry["after_score"])

    games = []
    for _ in range(30):
        accuracies = [move_accuracy(rng.uniform(0.0, 0.6)) for _ in range(rng.randint(1, 40))]
        games.append({"accuracies": accuracies, "game": game_accuracy(accuracies)})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "score.json").write_text(
        json.dumps(
            {"curves": curves, "labels": labels, "positions": positions, "games": games}
        ),
        encoding="utf-8",
    )
    print(
        f"{len(curves)} curve points, {len(labels)} labels, {len(positions)} positions, "
        f"{len(games)} games -> {OUT / 'score.json'}"
    )


if __name__ == "__main__":
    main()
