"""Record what the desktop's motif tagger says, for the Kotlin port to be held to.

The tagger is what gives the coach its theme, its focus squares and the sentence it says
about a tactic, so a port that disagrees anywhere changes what the coach teaches. It is pure
-- board, move, principal variation in, tags out -- so it can be compared exactly.

The positions are the Lichess puzzle sample the desktop validates against
(`data/puzzles/sample.csv`) and the per-theme set (`data/puzzles/themes-7-120.csv`, a hundred
and twenty puzzles of each Lichess theme, so every mating pattern and every line motif is
there many times): real tactics, which is where a tagger does its work. Each puzzle's
solution is walked and every position where the solver is to move is recorded, with the move,
the continuation and the mate distance -- the same thing the live pipeline tags as a game
unfolds.

The two sets give some thirty thousand positions, most of it the same half dozen tags over
and over. What is kept is chosen for coverage: every case that carries a tag still short of
its quota, so the rare patterns -- the under-promotion, the two castlings, the balestra mate --
are all in, and the common ones stop at a few dozen; and every STRIDE-th case whatever it
carries, because since 2026-09-23 most detectors reject what the geometry alone would name (a
pin that wins nothing, a fork the enemy takes), and those rejections are in the cases no quota
picks.

Each case also carries `pays_off` (`motifs.pays_off`, 2026-09-24): for every tag of a pattern
aimed at a piece (`motifs.TARGETED`) among the move's and the position's tags, whether the line
[move, *pv] cashes it in -- what the annotator keeps a move's tactic by, and the idea reader its
blow.

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

from sidecoach.analysis.motifs import (  # noqa: E402
    TARGETED,
    pays_off,
    tags_for_move,
    tags_for_position,
)
from sidecoach.analysis.phase import phase_of  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"
PUZZLES = DESKTOP / "data" / "puzzles" / "sample.csv"
THEMES = DESKTOP / "data" / "puzzles" / "themes-7-120.csv"


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


def payoffs(board: chess.Board, move: chess.Move, pv: list[chess.Move], tags: list) -> list[dict]:
    """`pays_off` at ply 0 of the line [move, *pv], walked to its first illegal move as the
    annotator walks it, for each distinct targeted tag."""
    line = [move, *pv]
    boards = [board.copy(stack=False)]
    for m in line:
        if m not in boards[-1].legal_moves:
            break
        nxt = boards[-1].copy(stack=False)
        nxt.push(m)
        boards.append(nxt)
    line = line[: len(boards) - 1]
    out: list[dict] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for tag in tags:
        key = (tag.name, tuple(tag.squares))
        if tag.name not in TARGETED or key in seen:
            continue
        seen.add(key)
        out.append({**tag_json(tag), "pays": pays_off(boards, line, 0, tag)})
    return out


# Cases kept per tag name. Enough that a detector's branches are exercised, few enough
# that `capture` does not fill the file.
QUOTA = 40
# And one case in this many whatever it carries: the rejections no quota picks.
STRIDE = 16


def choose(cases: list[dict]) -> list[dict]:
    """The cases worth keeping: each one that carries a tag still under its quota, and
    every STRIDE-th."""
    counts: dict[str, int] = {}
    kept: list[dict] = []
    for index, case in enumerate(cases):
        names = {tag["name"] for tag in case["for_move"] + case["for_position"]}
        if not names:
            names = {"(none)"}
        if any(counts.get(name, 0) < QUOTA for name in names) or index % STRIDE == 0:
            kept.append(case)
            for name in names:
                counts[name] = counts.get(name, 0) + 1
    return kept


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    for path in (PUZZLES, THEMES):
        if not path.exists():
            raise SystemExit(f"no puzzles at {path}")
    cases: list[dict] = []
    phases: list[dict] = []
    rows: list[dict] = []
    for path in (PUZZLES, THEMES):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(list(csv.DictReader(handle))[:limit])
    for row in rows:
        board = chess.Board(row["FEN"])
        moves = [chess.Move.from_uci(u) for u in row["Moves"].split()]
        # The first move is the opponent's, played to reach the position the solver faces.
        for index, move in enumerate(moves):
            if move not in board.legal_moves:
                break
            if index % 2 == 1:  # the solver is to move here
                pv = moves[index + 1 :]
                mate_in = mate_distance(board, moves[index:])
                for_move = tags_for_move(board, move, pv=pv, mate_in=mate_in)
                for_position = tags_for_position(board, pv=[move, *pv], mate_in=mate_in)
                cases.append(
                    {
                        "puzzle": row["PuzzleId"],
                        "fen": board.fen(),
                        "move": move.uci(),
                        "pv": [m.uci() for m in pv],
                        "mate_in": mate_in,
                        "for_move": [tag_json(t) for t in for_move],
                        "for_position": [tag_json(t) for t in for_position],
                        "pays_off": payoffs(board, move, pv, [*for_move, *for_position]),
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
    print(f"{len(cases)} of {seen_all} positions from {len(rows)} puzzles -> {OUT / 'motifs.json'}")
    print("tags seen: " + ", ".join(f"{n}x{c}" for n, c in sorted(names.items())))
    paid = [p["pays"] for case in cases for p in case["pays_off"]]
    print(f"pays_off: {len(paid)} targeted tags, {sum(paid)} paid off")


if __name__ == "__main__":
    main()
