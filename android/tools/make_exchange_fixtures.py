"""Record what the desktop's exchange module says, for the Kotlin port to be held to.

`analysis/exchange.py` answers who can take on a square, who can take back, and what a
capture nets -- by legal moves, so a piece pinned to its own king takes nothing and a king
never takes back onto a guarded square. Every tactic the tagger names, every "hanging" and
"defended" the coach says, rests on it, and on the phone it runs many times per position, so
it is held to the Python square by square.

The positions: random games (every kind of middlegame mess, checks and en passant rights
included), the Lichess puzzle sample walked move by move (real tactics: pins, x-rays,
overloaded defenders), and the desktop's own test positions for this module. For each
position and each of the 64 squares, both colours: the captures onto the square (in the
order the move generator gives them, which decides which of two equal pieces recaptures),
the capturers, the defenders, the best capture, and for the piece standing there whether it
can be won and whether it hangs. For every capture of the side to move, and of the side not
to move after a pass: the exchange it starts and whether it can be taken back.

    cd D:/chess-coaching-companion
    uv run --no-sync python android/tools/make_exchange_fixtures.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402

from sidecoach.analysis import exchange  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"
PUZZLES = DESKTOP / "data" / "puzzles" / "sample.csv"

# The desktop's tests/test_exchange.py positions, and a few built ones: a pinned capturer, a
# pinned defender, the king that cannot take back, an x-ray behind the capturer, promotions
# onto a guarded square, en passant.
BUILT = [
    "4k3/8/8/8/1b2n3/2N5/8/4K3 w - - 0 1",
    "4k3/8/5n2/8/1b2P3/2N5/8/4K3 b - - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "3q2k1/3r4/8/8/3R4/8/8/3Q2K1 b - - 0 1",
    "4k3/8/8/3n4/4P3/8/8/4K3 w - - 0 1",
    "4k3/2p5/3p4/8/8/8/8/3QK3 w - - 0 1",
    "1r2k3/2P5/8/8/8/8/8/4K2R w K - 0 1",
    "r3k2r/1P6/8/8/8/8/6p1/R3K2R w KQkq - 0 1",
    "4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 2",
    "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1",
    "4k3/8/8/K2pP2r/8/8/8/8 w - d6 0 2",
    "3rk3/8/8/3p4/8/3Q4/8/3RK3 w - - 0 1",
    "4k3/8/5n2/8/1b2B3/2N5/8/4K3 b - - 0 1",
    "4k3/8/5n2/8/4B3/2N5/8/4K3 b - - 0 1",
    "4k3/8/5n2/8/1b6/2N2B2/8/4K3 w - - 0 1",
]


def squares(values) -> list[int]:
    return [int(s) for s in values]


# Each square's answers, in this order (white's first, then black's, then the piece's own).
ROW = ["on_w", "cap_w", "def_w", "best_w", "on_b", "cap_b", "def_b", "best_b", "won", "hang"]


def record(board: chess.Board) -> dict:
    per_square = []
    for sq in chess.SQUARES:
        row: list = []
        for color in (chess.WHITE, chess.BLACK):
            row.append([m.uci() for m in exchange.captures_on(board, color, sq)])
            row.append(squares(exchange.capturers(board, color, sq)))
            row.append(squares(exchange.defenders(board, color, sq)))
            row.append(exchange.best_capture(board, color, sq))
        row.append(exchange.can_be_won(board, sq))
        row.append(exchange.hanging(board, sq))
        per_square.append(row)
    captures = []
    views = [board]
    if not board.is_check():
        passed = board.copy(stack=False)
        passed.push(chess.Move.null())
        views.append(passed)
    for view in views:
        for move in view.legal_moves:
            if view.is_capture(move):
                captures.append(
                    {
                        "fen": view.fen(),
                        "move": move.uci(),
                        "exchange": exchange.exchange(view, move),
                        "retake": exchange.can_retake(view, move),
                    }
                )
    return {"fen": board.fen(), "squares": per_square, "captures": captures}


def main() -> None:
    rng = random.Random(20260923)
    boards: list[chess.Board] = [chess.Board(fen) for fen in BUILT]

    # Random games: checks, en passant rights, pins of every kind, positions a puzzle avoids.
    for _ in range(40):
        board = chess.Board()
        for ply in range(rng.randint(10, 90)):
            moves = list(board.legal_moves)
            if not moves:
                break
            # Captures twice as likely, so the board thins and the exchanges get long.
            captures = [m for m in moves if board.is_capture(m)]
            board.push(rng.choice(captures if captures and rng.random() < 0.5 else moves))
            if ply % 9 == 4:
                boards.append(board.copy(stack=False))

    # The puzzle sample: real tactics, every position of the solution.
    with PUZZLES.open(newline="", encoding="utf-8") as handle:
        for row in list(csv.DictReader(handle))[:90]:
            board = chess.Board(row["FEN"])
            for uci in row["Moves"].split():
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    break
                board.push(move)
                boards.append(board.copy(stack=False))

    cases = [record(board) for board in boards]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exchange.json").write_text(
        json.dumps({"row": ROW, "positions": cases}, separators=(",", ":")), encoding="utf-8"
    )
    exchanges = sum(len(c["captures"]) for c in cases)
    won = sum(r[8] for c in cases for r in c["squares"])
    hang = sum(r[9] for c in cases for r in c["squares"])
    print(
        f"{len(cases)} positions, {len(cases) * 64} squares, {exchanges} exchanges; "
        f"{won} can be won, {hang} hanging -> {OUT / 'exchange.json'}"
    )


if __name__ == "__main__":
    main()
