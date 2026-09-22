"""Record what the desktop's motif tagger says, for the Kotlin port to be held to.

The tagger is what gives the coach its theme, its focus squares and the sentence it says
about a tactic, so a port that disagrees anywhere changes what the coach teaches. It is pure
-- board, move, principal variation in, tags out -- so it can be compared exactly.

The positions are the Lichess puzzle sample the desktop validates against
(`data/puzzles/sample.csv`): real tactics, which is where a tagger does its work. Each
puzzle's solution is walked and every position where the solver is to move is recorded, with
the move, the continuation and the mate distance -- the same thing the live pipeline tags as
a game unfolds.

The whole sample gives ten thousand positions and nine megabytes, most of it the same half
dozen tags over and over. What is kept is chosen for coverage: every case that carries a tag
still short of its quota, so the rare patterns -- the under-promotion, the two castlings, the
kill-box mate -- are all in, and the common ones stop at a few dozen. That is a tenth of the
size and tests strictly more per byte.

    cd D:/chess-coaching-companion
    uv run --no-sync python D:/sidecoach-android/tools/make_motif_fixtures.py [puzzles]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402

from sidecoach.analysis.motifs import tags_for_move, tags_for_position  # noqa: E402
from sidecoach.analysis.phase import phase_of  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"
PUZZLES = DESKTOP / "data" / "puzzles" / "sample.csv"


def tag_json(tag) -> dict:
    return {"name": tag.name, "squares": list(tag.squares), "move": tag.move}


def mate_distance(board: chess.Board, moves: list[chess.Move]) -> int | None:
    """The mover's own moves to mate, counting the first, or None if the line does not mate."""
    after = board.copy(stack=False)
    played = 0
    for move in moves:
        if move not in after.legal_moves:
            return None
        after.push(move)
        played += 1
        if after.is_checkmate():
            # The mover moves on the even plies of the line (0, 2, 4 ...).
            return (played + 1) // 2 if played % 2 == 1 else None
    return None


# Cases kept per tag name. Enough that a detector's branches are exercised, few enough
# that `capture` does not fill the file.
QUOTA = 40


def choose(cases: list[dict]) -> list[dict]:
    """The cases worth keeping: each one that carries a tag still under its quota."""
    counts: dict[str, int] = {}
    kept: list[dict] = []
    for case in cases:
        names = {tag["name"] for tag in case["for_move"] + case["for_position"]}
        if not names:
            names = {"(none)"}
        if any(counts.get(name, 0) < QUOTA for name in names):
            kept.append(case)
            for name in names:
                counts[name] = counts.get(name, 0) + 1
    return kept


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 4571
    if not PUZZLES.exists():
        raise SystemExit(f"no puzzle sample at {PUZZLES}")
    cases: list[dict] = []
    phases: list[dict] = []
    with PUZZLES.open(newline="", encoding="utf-8") as handle:
        for row in list(csv.DictReader(handle))[:limit]:
            board = chess.Board(row["FEN"])
            moves = [chess.Move.from_uci(u) for u in row["Moves"].split()]
            # The first move is the opponent's, played to reach the position the solver faces.
            for index, move in enumerate(moves):
                if move not in board.legal_moves:
                    break
                if index % 2 == 1:  # the solver is to move here
                    pv = moves[index + 1 :]
                    mate_in = mate_distance(board, moves[index:])
                    cases.append(
                        {
                            "puzzle": row["PuzzleId"],
                            "fen": board.fen(),
                            "move": move.uci(),
                            "pv": [m.uci() for m in pv],
                            "mate_in": mate_in,
                            "for_move": [
                                tag_json(t)
                                for t in tags_for_move(board, move, pv=pv, mate_in=mate_in)
                            ],
                            "for_position": [
                                tag_json(t)
                                for t in tags_for_position(
                                    board, pv=[move, *pv], mate_in=mate_in
                                )
                            ],
                        }
                    )
                    phases.append(
                        {
                            "fen": board.fen(),
                            "in_book": False,
                            "phase": phase_of(board, False),
                        }
                    )
                board.push(move)

    # A few opening positions too, for the phase table's book branch.
    opening = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6"):
        opening.push_san(san)
        phases.append({"fen": opening.fen(), "in_book": True, "phase": phase_of(opening, True)})
        phases.append({"fen": opening.fen(), "in_book": False, "phase": phase_of(opening, False)})

    seen_all = len(cases)
    cases = choose(cases)
    phases = phases[: len(cases)] + phases[-12:]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "motifs.json").write_text(
        json.dumps({"cases": cases, "phases": phases}), encoding="utf-8"
    )
    names: dict[str, int] = {}
    for case in cases:
        for tag in case["for_move"] + case["for_position"]:
            names[tag["name"]] = names.get(tag["name"], 0) + 1
    print(f"{len(cases)} of {seen_all} positions from {limit} puzzles -> {OUT / 'motifs.json'}")
    print("tags seen: " + ", ".join(f"{n}x{c}" for n, c in sorted(names.items())))


if __name__ == "__main__":
    main()
