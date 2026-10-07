"""Record what python-chess answers, for the Kotlin port to be held to.

The desktop reconciler is written against python-chess and depends on it exactly: on which
moves are legal, on the order the generator yields them in (a story of missed moves keeps
the first of several transpositions), on the SAN it writes, on the FEN it prints, and on
what it calls a valid position. So the port is not judged by eye. This walks random games
and a set of awkward positions, writes down every one of those answers, and
`app/src/androidTest/.../ChessTest.kt` reads them back on the phone and compares.

Beside them, the rest of what the desktop leans on (the parity audit of 2026-09-26): which
of `parse_san`'s three exceptions a text raises, the draw rules ply by ply through repeating
and long games, the null move pushed and written, moves the generator never yields
(king-takes-rook castling, drops, a promotion on a king move) judged legal or not, and FENs
with odd fields -- a right with no rook, file letters -- or ones python-chess refuses.

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


# SAN the desktop parses (coach prose, the opening tables, spoken moves): each with which of
# python-chess's three exceptions it raises, when it raises. The coach tells them apart.
SANS = [
    (chess.STARTING_FEN, ["e4", "e5", "Nf3", "Nd2", "N1f3", "Ngf3", "Nbf3", "exd5", "e2e4", "g1f3",
                          "e2e5", "Ke2", "Qz9", "xyz", "", "O-O", "0-0", "O-O-O", "--", "Z0", "0000",
                          "@@@@", "e8=Q", "Nf3+", "Nf3#", "Nxf3", "N-f3", "e4,e5", "P@e4", "Ph3"]),
    ("4k3/8/8/8/8/8/8/N1N1K3 w - - 0 1", ["Nb3", "Nab3", "Ncb3", "N1b3", "Na1b3", "Nd3", "Ne2"]),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["O-O", "O-O+", "O-O#", "0-0", "O-O-O", "0-0-0",
                                             "Kg1", "Kf1", "Kc1", "Rh1", "e1g1", "e1h1", "e1c1",
                                             "e1a1", "Rxa8", "Rxh8+", "Rb1", "Rab1"]),
    ("r3k2r/8/8/8/8/8/8/R3KB1R w KQkq - 0 1", ["O-O", "O-O-O", "e1h1", "e1g1"]),
    ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", ["O-O", "O-O-O", "e8h8", "e8g8", "e8a8"]),
    ("n3k3/1P6/8/8/8/8/8/4K3 w - - 0 1", ["b8=Q", "b8Q", "b8=q", "b8", "bxa8=N", "bxa8", "b8=K",
                                          "b7b8", "b7b8=R", "b8=Q+", "bxa8=Q+"]),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", ["exd6", "exd6e.p.", "e5d6", "ed6", "ed", "d6"]),
    ("8/8/8/8/k1pP3R/8/8/4K3 b - d3 0 1", ["cxd3", "c4d3", "c3", "Kb5"]),
    ("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1", ["Kxe2", "Kf1", "Kd1", "Kd2", "O-O"]),
]

# Games that repeat, run out the clock, or both: after every ply what python-chess makes of
# the draw rules, claimable or not.
DRAWS = [
    (chess.STARTING_FEN, ["g1f3", "g8f6", "f3g1", "f6g8"] * 4 + ["g1f3"]),
    # The rights change on the way round: the same squares, not the same position.
    (chess.STARTING_FEN, ["g1f3", "g8f6", "h1g1", "h8g8", "g1h1", "g8h8", "f3g1", "f6g8",
                          "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]),
    # An en passant square that is not a capture does not make a new position.
    (chess.STARTING_FEN, ["e2e4", "g8f6", "g1f3", "f6g8", "f3g1", "g8f6", "g1f3", "f6g8",
                          "f3g1", "g8f6"]),
    ("8/8/8/4k3/8/8/8/4K2R w K - 97 60", ["h1h2", "e5d5", "h2h1", "d5e5", "h1h2", "e5d5"]),
    ("8/8/8/4k3/8/8/8/4K2R w - - 147 90", ["h1h2", "e5d5", "h2h1", "d5e5", "h1h2"]),
    # Mate on the hundredth half-move is mate, not fifty moves.
    ("7k/8/6K1/8/8/8/8/6Q1 w - - 99 80", ["g1g7"]),
    # The pawn move resets the clock.
    ("4k3/8/8/8/8/8/4P3/4K3 w - - 99 70", ["e2e4", "e8d7"]),
]

# Moves as a screen or an engine may write them, generated or not.
LEGALITY = [
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["e1h1", "e1a1", "e1g1", "e1c1", "e1f1", "e1d1",
                                             "e1b1", "e1h1q", "h1h8", "a1a8", "e1e2", "0000"]),
    ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", ["e8h8", "e8a8", "e8g8", "e8c8", "h8h1"]),
    ("r3k2r/8/8/8/8/8/8/R3KB1R w KQkq - 0 1", ["e1h1", "e1g1", "e1a1", "e1c1"]),
    ("r3k2r/8/8/8/8/8/5r2/R3K2R w KQkq - 0 1", ["e1h1", "e1g1", "e1a1", "e1c1"]),
    ("r3k2r/8/8/8/8/8/4r3/R3K2R w KQkq - 0 1", ["e1h1", "e1g1", "e1a1", "e1c1", "e1d1"]),
    ("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1", ["e1h1", "e1g1", "e1a1", "e1c1"]),
    ("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1", ["e1h1", "e1a1", "e1g1", "e1c1"]),
    ("4k3/8/8/8/8/8/8/R4K1R w - - 0 1", ["f1h1", "f1g1", "f1e1"]),
    (chess.STARTING_FEN, ["e2e4", "e2e5", "e7e5", "a2a1", "g1f3", "g1g3", "e1g1", "e1h1",
                          "b1d2", "a2a3q", "P@e4", "0000"]),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", ["e5d6", "e5f6", "e5e6"]),
    ("8/8/8/8/k1pP3R/8/8/4K3 b - d3 0 1", ["c4d3", "c4c3", "a4b5"]),
    ("n3k3/1P6/8/8/8/8/8/4K3 w - - 0 1", ["b7b8", "b7b8q", "b7a8n", "b7b8k", "b7a8"]),
]

# FENs with odd fields, and ones python-chess refuses.
FENS = [
    "4k3/8/8/8/8/8/8/4K3 w K - 0 1",
    "4k3/8/8/8/8/8/8/4K3 w Q - 0 1",
    "4k3/8/8/8/8/8/8/4K3 b k - 0 1",
    "4k3/8/8/8/8/8/8/4K3 b q - 0 1",
    "4k3/8/8/8/8/8/8/1R2K1R1 w KQ - 0 1",
    "4k3/8/8/8/8/8/8/R3K2R w HA - 0 1",
    "4k3/8/8/8/8/8/8/R3K2R w AH - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w Kk - 0 1",
    "4k3/8/8/8/8/8/8/R6R w KQ - 0 1",
    "4k3/8/8/8/8/8/8/R2K3R w KQ - 0 1",
    "rk6/8/8/8/8/8/8/1K5R w Kq - 0 1",
    "4k3/8/8/8/8/8/8/4K3",
    "4k3/8/8/8/8/8/8/4K3 b",
    "4k3/8/8/8/8/8/8/4K3 w -",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    "rnbqkbnr/pppp1ppp/8/8/3pP3/8/PPP2PPP/RNBQKBNR b KQkq e3 0 3",
    "4k3/8/8/8/8/8/8/4K3 w - - 0 0",
    "  4k3/8/8/8/8/8/8/4K3   w  -  -  3  9 ",
    "4k3/8/8/8/8/8/8/Q~3K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/4K3 x - - 0 1",
    "4k3/8/8/8/8/8/8 w - - 0 1",
    "4k3/8/8/8/8/8/8/4K4 w - - 0 1",
    "4k3/8/8/8/8/8/8/44 w - - 0 1",
    "4k3/8/8/8/8/8/8/3K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/4X3 w - - 0 1",
    "4k3/8/8/8/8/8/8/4K3 w - z9 0 1",
    "4k3/8/8/8/8/8/8/4K3 w - - -1 1",
    "4k3/8/8/8/8/8/8/4K3 w - - 0 -1",
    "4k3/8/8/8/8/8/8/4K3 w - - a 1",
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1 7",
    "4k3/8/8/8/8/8/8/4K3 w kqKQ - 0 1",
    "4k3/8/8/8/8/8/8/4K3 w KQQ - 0 1",
    "4k3/8/8/8/8/8/8/4K3 w KQx - 0 1",
    "~4k3/8/8/8/8/8/8/4K3 w - - 0 1",
    "",
    "4k3/8/8/8/8/8/8/9 w - - 0 1",
]


def parse_sans() -> list[dict]:
    out = []
    for fen, sans in SANS:
        board = chess.Board(fen)
        for san in sans:
            case = {"fen": fen, "san": san, "uci": None, "failure": None}
            try:
                case["uci"] = board.parse_san(san).uci()
            except chess.AmbiguousMoveError:
                case["failure"] = "ambiguous"
            except chess.IllegalMoveError:
                case["failure"] = "illegal"
            except chess.InvalidMoveError:
                case["failure"] = "invalid"
            out.append(case)
    return out


def draws() -> list[dict]:
    out = []
    for fen, ucis in DRAWS:
        board = chess.Board(fen)
        steps = []
        for uci in ucis:
            board.push(chess.Move.from_uci(uci))
            steps.append({
                "fen": board.fen(),
                "repetition": board.is_repetition(),
                "repetition2": board.is_repetition(2),
                "threefold": board.can_claim_threefold_repetition(),
                "fifty": board.is_fifty_moves(),
                "claim_fifty": board.can_claim_fifty_moves(),
                "fivefold": board.is_fivefold_repetition(),
                "seventyfive": board.is_seventyfive_moves(),
                "game_over": board.is_game_over(),
                "game_over_claim": board.is_game_over(claim_draw=True),
                "result_claim": board.result(claim_draw=True),
            })
        out.append({"fen": fen, "moves": ucis, "steps": steps})
    return out


def nulls() -> list[dict]:
    """A null move pushed where it is legal to pass, and where it is not (in check), and
    what the board says after it: the FEN, the SAN python-chess writes for it (no suffix,
    whatever the position), and the FEN again once it is popped."""
    out = []
    for fen in (
        chess.STARTING_FEN,
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 5 20",
        "4k3/8/8/8/8/8/4r3/4K3 b - - 0 1",
        "4k3/8/8/8/8/8/4q3/4K3 w - - 0 1",
        "rnbqkbnr/pppp1ppp/8/8/3pP3/8/PPP2PPP/RNBQKBNR b KQkq e3 0 3",
    ):
        board = chess.Board(fen)
        san = board.san(chess.Move.null())
        board.push(chess.Move.null())
        after = board.fen()
        stack = [m.uci() for m in board.move_stack]
        board.pop()
        out.append({"fen": fen, "san": san, "after": after, "stack": stack, "popped": board.fen()})
    return out


def legality() -> list[dict]:
    out = []
    for fen, ucis in LEGALITY:
        board = chess.Board(fen)
        for uci in ucis:
            move = chess.Move.from_uci(uci)
            case = {
                "fen": fen,
                "uci": uci,
                "pseudo_legal": board.is_pseudo_legal(move),
                # `move in board.legal_moves` asks the same: `is_legal`.
                "legal": board.is_legal(move),
                "san": None,
                "after": None,
                "stacked": None,
            }
            if case["legal"]:
                case["san"] = board.san(move)
                after = board.copy(stack=False)
                after.push(move)
                case["after"] = after.fen()
                case["stacked"] = after.move_stack[-1].uci()
            out.append(case)
    return out


def fens() -> list[dict]:
    out = []
    for fen in FENS:
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            out.append({"given": fen, "error": str(exc)})
            continue
        out.append({
            "given": fen,
            "error": None,
            "fen": board.fen(),
            "fen_ep": board.fen(en_passant="fen"),
            "castling_rights": format(board.castling_rights, "x"),
            "promoted": format(board.promoted, "x"),
            "valid": board.is_valid(),
            "ply": board.ply(),
        })
    return out


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

    extra = {
        "parse_san": parse_sans(),
        "draws": draws(),
        "nulls": nulls(),
        "legality": legality(),
        "fens": fens(),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "positions.json").write_text(
        json.dumps({"positions": positions, "perft": perfts, **extra}), encoding="utf-8"
    )
    print(f"{len(positions)} positions, {len(perfts)} perfts -> {OUT / 'positions.json'}")
    print(f"{sum(len(p['moves']) for p in positions)} legal moves recorded")
    print(", ".join(f"{len(v)} {k}" for k, v in extra.items()))


if __name__ == "__main__":
    main()
