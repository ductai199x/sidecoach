"""Record what the desktop's opening book answers, for the Kotlin port to be held to.

The book is pure -- moves in, an opening name, whether they are in theory, and where they
left it out -- so it can be compared exactly. The inputs, over the real Lichess tables:

- every named line, whole, and a prefix of it a few plies short;
- every named line with a legal move off the end, which usually leaves the book;
- each of those with one move swapped mid-line for another legal move: the deviation;
- a few hundred seeded random games from the start, most of which leave theory at once;
- transpositions the tables resolve by position (1.d4 Nf6 2.c4 e6 3.Nc3 d5 and friends);
- moves that do not belong to a game from the start, which name nothing.

The tables themselves are copied beside the fixture, so the test replays the same data the
Python read; and the compiled sizes (named lines, positions) are recorded too.

    cd D:/chess-coaching-companion
    uv run --no-sync python android/tools/make_openings_fixtures.py
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402

from sidecoach.analysis.openings import OpeningBook, _compile  # noqa: E402

TABLES = DESKTOP / "data" / "openings"
OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "openings"


def line_moves(pgn: str) -> list[chess.Move]:
    board = chess.Board()
    out: list[chess.Move] = []
    for token in pgn.split():
        if token[0].isdigit():
            continue
        out.append(board.push_san(token))
    return out


def sequences(rng: random.Random) -> list[list[chess.Move]]:
    lines: list[list[chess.Move]] = []
    for table in sorted(TABLES.glob("*.tsv")):
        for row in table.read_text(encoding="utf-8").splitlines():
            fields = row.split("\t")
            if len(fields) != 3 or fields[0] == "eco":
                continue
            try:
                lines.append(line_moves(fields[2]))
            except ValueError:
                continue
    out: list[list[chess.Move]] = []
    for moves in lines:
        out.append(moves)
        if len(moves) > 3:
            out.append(moves[: len(moves) - 3])
        board = chess.Board()
        for move in moves:
            board.push(move)
        legal = list(board.legal_moves)
        if legal:
            out.append([*moves, rng.choice(legal)])
        if len(moves) > 2:
            cut = rng.randrange(1, len(moves))
            board = chess.Board()
            for move in moves[:cut]:
                board.push(move)
            others = [m for m in board.legal_moves if m != moves[cut]]
            if others:
                out.append([*moves[:cut], rng.choice(others)])
    for _ in range(300):
        board = chess.Board()
        played: list[chess.Move] = []
        for _ in range(rng.randrange(1, 16)):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = rng.choice(legal)
            board.push(move)
            played.append(move)
        out.append(played)
    for pgn in (
        "1. d4 Nf6 2. c4 e6 3. Nc3 d5",
        "1. d4 d5 2. c4 e6 3. Nc3 Nf6",
        "1. Nf3 d5 2. d4 Nf6 3. c4 e6",
        "1. c4 e5 2. Nc3 Nf6 3. Nf3 Nc6",
        "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6",
    ):
        out.append(line_moves(pgn))
    # The name is Lichess's (Tai, 2026-10-02): the last named position in the first 40 plies,
    # a transposition back into theory included -- out of the tables at 2.Bc4 and back in the
    # Italian; a named position at ply 39 and another at 41; and long random games, of their
    # own seed so the cases above stay as they were.
    out.append(line_moves("1. e4 e5 2. Bc4 Nc6 3. Nf3"))
    out.append(line_moves(" ".join(["Nf3 Nf6 Ng1 Ng8"] * 9) + " e4 e5 Nf3 Nc6 Bc4"))
    long = random.Random(20261002)
    for _ in range(40):
        board = chess.Board()
        played = []
        for _ in range(long.randrange(30, 61)):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = long.choice(legal)
            board.push(move)
            played.append(move)
        out.append(played)
    # Not a game from the start position: e7e5 first is not legal for White.
    out.append([chess.Move.from_uci("e7e5"), chess.Move.from_uci("e2e4")])
    out.append([])
    return out


def main() -> None:
    book = OpeningBook.load(TABLES)
    assert book is not None, f"no tables in {TABLES}"
    tables = sorted(TABLES.glob("*.tsv"))
    names, positions = _compile(tables)
    rng = random.Random(20260922)
    cases = []
    for moves in sequences(rng):
        found = book.identify(moves)
        cases.append({
            "moves": [m.uci() for m in moves],
            "identify": None if found is None else {"eco": found.eco, "name": found.name, "ply": found.ply},
            "in_book": book.in_book(moves),
            "deviation": book.deviation(moves),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    for table in tables:
        shutil.copy(table, OUT / table.name)
    (OUT / "book.json").write_text(
        json.dumps({"names": len(names), "positions": len(positions), "cases": cases}),
        encoding="utf-8",
    )
    named = sum(1 for c in cases if c["identify"])
    print(f"{len(cases)} sequences ({named} named, {sum(c['in_book'] for c in cases)} in book), "
          f"{len(names)} named positions of {len(positions)} -> {OUT}")


if __name__ == "__main__":
    main()
