"""Record what the desktop's mating-pattern namer says, for the Kotlin port to be held to.

`analysis/mates.py` names which textbook mate a checkmate is -- the Anastasia, the Boden, the
nineteen Lichess patterns -- and a wrong name teaches the wrong picture, so the port has to
name exactly what the desktop names, with the same evidence squares.

The positions are every checkmate the puzzle sets end in: the per-theme set
(`data/puzzles/themes-7-120.csv`, a hundred and twenty puzzles of each Lichess theme, so every
mating pattern is there many times over) and the sample the rest of the tagger is validated
on, each puzzle's solution played out to its final position; plus the desktop's own positives
and negatives from `tests/test_mates.py`, the shapes each pattern must and must not be given.

    cd D:/chess-coaching-companion
    uv run --no-sync python android/tools/make_mates_fixtures.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402

from sidecoach.analysis.mates import MATE_PATTERNS, mating_pattern_tags  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"
SETS = [
    DESKTOP / "data" / "puzzles" / "themes-7-120.csv",
    DESKTOP / "data" / "puzzles" / "sample.csv",
]


def case(after: chess.Board, move: chess.Move, source: str) -> dict:
    return {
        "source": source,
        "fen": after.fen(),
        "move": move.uci(),
        "tags": [
            {"name": t.name, "squares": list(t.squares)} for t in mating_pattern_tags(after, move)
        ],
    }


def desktop_cases() -> list[dict]:
    """The desktop test's positives and negatives, played to the mate."""
    sys.path.insert(0, str(DESKTOP / "tests"))
    import test_mates  # noqa: PLC0415

    out = []
    seen = set()
    for row in [*test_mates.POSITIVES, *test_mates.NEGATIVES]:
        fen, san = row[1], row[2]
        if (fen, san) in seen:
            continue
        seen.add((fen, san))
        board = chess.Board(fen)
        move = board.parse_san(san)
        board.push(move)
        out.append(case(board, move, f"test_mates {row[0]}"))
    # The double check whose second checker is needed: no Arabian mate.
    after = chess.Board("r4rRk/pp5p/4n2N/1b6/3B4/8/P4PPP/6K1 b - - 0 1")
    out.append(case(after, chess.Move.from_uci("g1g8"), "test_mates double check"))
    return out


def main() -> None:
    cases = desktop_cases()
    seen = {c["fen"] for c in cases}
    for path in SETS:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                board = chess.Board(row["FEN"])
                last = None
                for uci in row["Moves"].split():
                    move = chess.Move.from_uci(uci)
                    if move not in board.legal_moves:
                        last = None
                        break
                    board.push(move)
                    last = move
                if last is None or not board.is_checkmate() or board.fen() in seen:
                    continue
                seen.add(board.fen())
                cases.append(case(board, last, row["PuzzleId"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mates.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")
    counts = {name: 0 for name in MATE_PATTERNS}
    for c in cases:
        for tag in c["tags"]:
            counts[tag["name"]] += 1
    unnamed = sum(1 for c in cases if not c["tags"])
    print(f"{len(cases)} mates ({unnamed} with no pattern) -> {OUT / 'mates.json'}")
    print("patterns: " + ", ".join(f"{n}x{c}" for n, c in counts.items()))


if __name__ == "__main__":
    main()
