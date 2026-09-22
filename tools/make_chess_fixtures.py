"""Record what python-chess answers, for the Kotlin port to be held to.

The desktop reconciler is written against python-chess and depends on it exactly: on which
moves are legal, on the order the generator yields them in (a story of missed moves keeps
the first of several transpositions), on the SAN it writes, on the FEN it prints, and on
what it calls a valid position. So the port is not judged by eye. This walks random games
and a set of awkward positions, writes down every one of those answers, and
`app/src/androidTest/.../ChessTest.kt` reads them back on the phone and compares.

Run from the desktop repo's environment, which has python-chess:

    cd D:/chess-coaching-companion
    uv run python D:/sidecoach-android/tools/make_chess_fixtures.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import chess

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"

# Positions worth more than a random game: castling both ways and blocked, en passant with
# and without the pin behind it, promotions with capture, mate, stalemate, bare kings.
AWKWARD = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # Kiwipete
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",  # en passant available
    "8/8/8/8/k1pP3R/8/8/4K3 b - d3 0 1",  # en passant that would skewer the king
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",  # stalemate
    "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",  # checkmate
    "8/8/8/4k3/8/8/8/4K3 w - - 0 1",  # bare kings
    "8/8/8/4k3/8/8/8/4KB2 w - - 0 1",  # king and bishop
    "8/8/8/4k3/8/8/8/4KN2 w - - 0 1",  # king and knight
    "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",  # castling both ways
    "r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1",
    "4k3/8/8/8/8/8/6p1/4K3 b - - 0 1",  # promotion
    "n3k3/1P6/8/8/8/8/8/4K3 w - - 0 1",  # promotion with capture
]


def measure(board: chess.Board) -> dict:
    """Everything the port must agree with for this position."""
    moves = list(board.legal_moves)
    sans = []
    for move in moves:
        sans.append(board.san(move))
    validity = {}
    for name, turn in (("white", chess.WHITE), ("black", chess.BLACK)):
        trial = board.copy(stack=False)
        trial.turn = turn
        validity[name] = trial.is_valid()
    return {
        "fen": board.fen(),
        # The object's own en-passant square, which `fen()` prints only when the capture is
        # legal -- and which `is_valid()` judges whatever the FEN says.
        "ep": chess.square_name(board.ep_square) if board.ep_square is not None else None,
        "board_fen": board.board_fen(),
        "moves": [move.uci() for move in moves],
        "sans": sans,
        "check": board.is_check(),
        "checkmate": board.is_checkmate(),
        "stalemate": board.is_stalemate(),
        "insufficient": board.is_insufficient_material(),
        "game_over": board.is_game_over(),
        "result": board.result(),
        "valid": validity,
        # The position each move leads to: a push that differs anywhere shows up here.
        "after": [
            (lambda b: (b.push(move), b.fen())[1])(board.copy(stack=False)) for move in moves
        ],
    }


def perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += perft(board, depth - 1)
        board.pop()
    return total


def main() -> None:
    random.seed(20260922)
    positions: list[dict] = []
    for fen in AWKWARD:
        positions.append(measure(chess.Board(fen)))

    # Random games, sampled every few plies: openings, middlegames, and the odd endgame.
    for game in range(60):
        board = chess.Board()
        for ply in range(random.randint(4, 120)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(random.choice(moves))
            if ply % 7 == 0 or not board.legal_moves:
                positions.append(measure(board))
        positions.append(measure(board))

    perfts = [
        {"fen": fen, "depth": depth, "nodes": perft(chess.Board(fen), depth)}
        for fen, depth in (
            (chess.STARTING_FEN, 4),
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4),
            ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3),
            ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3),
        )
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "positions.json").write_text(
        json.dumps({"positions": positions, "perft": perfts}), encoding="utf-8"
    )
    print(f"{len(positions)} positions, {len(perfts)} perfts -> {OUT / 'positions.json'}")
    print(f"{sum(len(p['moves']) for p in positions)} legal moves recorded")


if __name__ == "__main__":
    main()
