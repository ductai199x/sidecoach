"""Record what python-chess makes of PGN texts, for the Kotlin port (chess/Pgn.kt).

For each text: every game `read_game` gives, one after another -- its headers in order, its
main line's moves (UCI) and each move's clock, whether it has errors, and `str(game)` (what the
library keeps and keys a pasted game by). The texts: the awkward ones by hand (a BOM, `%`
lines, a comment over lines, side lines within side lines, NAGs and move marks, a result in
the movetext and not the tags, a move that cannot be played, a set-up position, a variant, no
headers, several games at once), three of Tai's own chess.com games as the archive gives them
(from a copy of his data), and forty random games written with clocks, NAGs, comments and
side lines.

    cd D:/chess-coaching-companion
    uv run --no-sync python android/tools/make_pgn_fixtures.py
"""

from __future__ import annotations

import io
import json
import logging
import random
import sqlite3
import sys
from pathlib import Path

import chess
import chess.pgn

OUT = (Path(__file__).resolve().parents[1] / "app/src/androidTest/assets/chess/pgn.json")
COPY = Path(
    "C:/Users/Tai/AppData/Local/Temp/claude/d--chess-coaching-companion/"
    "5a849ff1-467c-42d3-861a-c579f032804a/scratchpad/appcopy/profile.sqlite"
)
logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)

BY_HAND = {
    "bom and percent lines": "\ufeff% a comment line\n[Event \"x\"]\n[White \"A\"]\n[Black \"B\"]\n\n1. e4 e5 2. Nf3 *\n",
    "comment over lines": '[White "A"]\n\n1. e4 { a comment\nthat runs on } e5 {[%clk 0:09:58]} 2. Nf3 {\n[%clk 0:09:50] } Nc6 1-0\n',
    "side lines in side lines": "1. e4 e5 (1... c5 2. Nf3 (2. c3 d5) 2... d6) 2. Nf3 Nc6 (2... d6 3. d4) 3. Bb5 *\n",
    "nags and marks": "1. e4! e5? 2. Nf3!! Nc6?? 3. Bb5!? a6?! 4. Ba4 $1 Nf6 $146 *\n",
    "result in the movetext": '[Event "?"]\n\n1. f3 e5 2. g4 Qh4# 0-1\n',
    "an illegal move": "1. e4 e5 2. Ke3 Nc6 *\n",
    "an ambiguous move": '[FEN "4k3/8/8/8/8/8/8/N3K2N w - - 0 1"]\n[SetUp "1"]\n\n1. Nf2 *\n',
    "a set-up position": '[FEN "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"]\n[SetUp "1"]\n\n1. e4 Kd7 2. e5 Ke6 1/2-1/2\n',
    "a variant": '[Variant "Atomic"]\n\n1. e4 e5 *\n',
    "chess960 named": '[Variant "Chess960"]\n\n1. e4 e5 *\n',
    "no headers": "1. d4 d5 2. c4 e6 3. Nc3 Nf6 *\n",
    "no moves": '[Event "empty"]\n\n*\n',
    "several games": '[White "A"]\n\n1. e4 *\n\n[White "B"]\n\n1. d4 d5 *\n\n\n1. c4 *\n',
    "a comment before the moves": '{ the start } 1. e4 { and after } e5 *\n',
    "semicolon comments": "1. e4 ; to the end of the line\ne5 2. Nf3 *\n",
    "castling and promotion": "1. e4 d5 2. exd5 c6 3. dxc6 Nf6 4. cxb7 Nbd7 5. bxa8=Q e6 6. Nf3 Bd6 7. Bc4 O-O 8. O-O *\n",
    "castling with zeros": "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. 0-0 Nf6 *\n",
    "tags with escapes": '[Event "the \\"big\\" one"]\n[Site "a\\\\b"]\n\n1. e4 *\n',
    "lichess style": (
        '[Event "Rated Blitz game"]\n[Site "https://lichess.org/AbCdEfGh"]\n[Date "2026.09.28"]\n'
        '[White "nosegazm"]\n[Black "someone"]\n[Result "1-0"]\n[UTCDate "2026.09.28"]\n'
        '[UTCTime "18:02:11"]\n[WhiteElo "1500"]\n[BlackElo "1480"]\n[Variant "Standard"]\n'
        '[TimeControl "180+2"]\n[ECO "C20"]\n[Termination "Normal"]\n\n'
        "1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } 2. Qh5 { [%clk 0:02:59] } "
        "2... Nc6 { [%clk 0:02:58] } 3. Bc4 { [%clk 0:02:57] } 3... Nf6?? { [%clk 0:02:50] } "
        "4. Qxf7# { [%clk 0:02:55] } 1-0\n"
    ),
}


def record(text: str) -> list[dict]:
    handle = io.StringIO(text)
    games = []
    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            return games
        games.append(
            {
                "headers": [[k, v] for k, v in game.headers.items()],
                "moves": [m.uci() for m in game.mainline_moves()],
                "clocks": [n.clock() for n in game.mainline()],
                "errors": bool(game.errors),
                "text": str(game),
            }
        )


def random_game(rng: random.Random) -> str:
    """A random game written by python-chess with clocks, NAGs, comments and side lines."""
    game = chess.pgn.Game()
    game.headers["White"] = "W" + str(rng.randrange(1000))
    game.headers["TimeControl"] = rng.choice(["600", "180+2", "60", "1/86400"])
    board = chess.Board()
    node: chess.pgn.GameNode = game
    clock = 600.0
    for _ply in range(rng.randrange(8, 70)):
        moves = list(board.legal_moves)
        if not moves:
            break
        move = rng.choice(moves)
        if rng.random() < 0.15 and len(moves) > 1:
            other = rng.choice([m for m in moves if m != move])
            side = node.add_variation(other, comment="a side line" if rng.random() < 0.5 else "")
            b = board.copy()
            b.push(other)
            for _ in range(rng.randrange(0, 3)):
                ms = list(b.legal_moves)
                if not ms:
                    break
                m = rng.choice(ms)
                side = side.add_variation(m)
                b.push(m)
        node = node.add_variation(move)
        if rng.random() < 0.1:
            node.nags.add(rng.choice([1, 2, 3, 4, 5, 6, 10, 146]))
        clock = max(0.0, clock - rng.random() * 10)
        node.set_clock(round(clock, rng.choice([0, 1])))
        if rng.random() < 0.05:
            node.comment += " note {with} brace" if rng.random() < 0.5 else " note"
        board.push(move)
    game.headers["Result"] = board.result() if board.is_game_over() else rng.choice(["*", "1-0", "0-1", "1/2-1/2"])
    return str(game) + "\n"


def main() -> None:
    cases = [{"name": name, "text": text, "games": record(text)} for name, text in BY_HAND.items()]
    if COPY.exists():
        db = sqlite3.connect(COPY)
        for (key, pgn) in db.execute("SELECT key, pgn FROM library WHERE source = 'chesscom' ORDER BY key LIMIT 3"):
            cases.append({"name": f"chess.com {key}", "text": pgn, "games": record(pgn)})
    else:
        print("no copy of Tai's data: the chess.com cases are left out", file=sys.stderr)
    rng = random.Random(20260930)
    for i in range(40):
        text = random_game(rng)
        cases.append({"name": f"random {i}", "text": text, "games": record(text)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"cases": cases}, indent=1), encoding="utf-8")
    games = sum(len(c["games"]) for c in cases)
    print(f"{OUT}: {len(cases)} texts, {games} games")


if __name__ == "__main__":
    main()
