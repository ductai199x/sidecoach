"""Validate our tactics against the Lichess puzzle sample (ground truth).

Each puzzle is a FEN plus a solution line in UCI; the first move is the opponent's, played
to reach the position the solver faces, and the rest is the solution. Lichess assigns a
theme if it occurs anywhere in the solution, so we tag every position along the line where
the solver is to move -- exactly what our live pipeline tags as the game unfolds -- and
union the result. For each puzzle we then:

  * check, per theme our tagger implements, whether we recover the theme Lichess assigned;
  * feed each geometric tactic to describe_tactic and check it renders a true sentence.

Only themes our tagger can emit are scored; the most common themes we do not implement are
listed separately as coverage gaps.

Run:  uv run python tools/puzzles/validate.py [limit]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import chess

from sidecoach.analysis import motifs
from sidecoach.coach.facts import describe_tactic

SAMPLE = Path("data/puzzles/sample.csv")
ALIAS = {"absolutePin": "pin"}
DESCRIBABLE = {"pin", "absolutePin", "fork", "skewer", "hangingPiece", "trappedPiece"}
# Every theme our tagger can emit, in the Lichess spelling (absolutePin folds into pin).
OURS = {
    "pin",
    "advancedPawn",
    "anastasiaMate",
    "arabianMate",
    "attackingF2F7",
    "attraction",
    "backRankMate",
    "bodenMate",
    "capturingDefender",
    "clearance",
    "deflection",
    "discoveredAttack",
    "doubleBishopMate",
    "doubleCheck",
    "dovetailMate",
    "exposedKing",
    "fork",
    "hangingPiece",
    "hookMate",
    "interference",
    "killBoxMate",
    "mate",
    "mateIn1",
    "mateIn2",
    "mateIn3",
    "mateIn4",
    "mateIn5",
    "promotion",
    "skewer",
    "smotheredMate",
    "trappedPiece",
    "underPromotion",
    "vukovicMate",
    "xRayAttack",
}


def walk(
    board: chess.Board, solution: list[chess.Move]
) -> list[tuple[chess.Board, chess.Move, list[motifs.MotifTag]]]:
    """At each solver-to-move step along the line: the position, its best move, its tags.

    This mirrors the live pipeline, which tags each position the player faces with the
    static features plus the tags of the engine's best move.
    """
    steps: list[tuple[chess.Board, chess.Move, list[motifs.MotifTag]]] = []
    view = board.copy()
    for i in range(0, len(solution), 2):
        remaining = solution[i:]
        if remaining[0] not in view.legal_moves:
            break
        mate_in = _mate_in(view, remaining)
        tags = motifs.tags_for_position(view, pv=remaining, mate_in=mate_in)
        steps.append((view.copy(), remaining[0], tags))
        view.push(remaining[0])  # solver move
        if len(remaining) > 1:
            view.push(remaining[1])  # opponent reply
    return steps


def _mate_in(board: chess.Board, remaining: list[chess.Move]) -> int | None:
    view = board.copy()
    solver = 0
    for i, move in enumerate(remaining):
        if i % 2 == 0:
            solver += 1
        view.push(move)
    return solver if view.is_checkmate() else None


def main() -> int:
    if not SAMPLE.exists():
        print(f"no sample at {SAMPLE}; run tools/puzzles/fetch.py first")
        return 1
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    have: defaultdict[str, int] = defaultdict(int)
    got: defaultdict[str, int] = defaultdict(int)
    unimpl: defaultdict[str, int] = defaultdict(int)
    d_have: defaultdict[str, int] = defaultdict(int)
    d_ok: defaultdict[str, int] = defaultdict(int)
    tagged_any = total = 0

    with SAMPLE.open(encoding="utf-8", newline="") as fh:
        for n, row in enumerate(csv.DictReader(fh)):
            if limit and n >= limit:
                break
            moves = row["Moves"].split()
            if len(moves) < 2:
                continue
            try:
                board = chess.Board(row["FEN"])
                board.push(chess.Move.from_uci(moves[0]))
                solution = [chess.Move.from_uci(m) for m in moves[1:]]
            except (ValueError, AssertionError):
                continue
            if not solution or solution[0] not in board.legal_moves:
                continue
            total += 1
            lichess = set(row["Themes"].split())
            steps = walk(board, solution)
            names = {ALIAS.get(t.name, t.name) for _, _, tags in steps for t in tags}
            if names & lichess:
                tagged_any += 1
            for theme in lichess:
                if theme in OURS:
                    have[theme] += 1
                    if theme in names:
                        got[theme] += 1
                else:
                    unimpl[theme] += 1
            # describe_tactic is checked on the very position and best move the tag came from,
            # deduped per theme so one long puzzle cannot dominate the rate.
            described: set[str] = set()
            for pos, best, tags in steps:
                for tag in tags:
                    canon = ALIAS.get(tag.name, tag.name)
                    if tag.name not in DESCRIBABLE or canon not in lichess or canon in described:
                        continue
                    described.add(canon)
                    d_have[canon] += 1
                    if describe_tactic(pos, tag.name, tag.squares, best) is not None:
                        d_ok[canon] += 1

    print(f"puzzles checked: {total}")
    print(f"recovered >=1 Lichess theme in {tagged_any}/{total} ({tagged_any / total:.0%})\n")
    print("per-theme recall on themes our tagger implements:")
    for theme in sorted(have, key=lambda t: -have[t]):
        if have[theme] < 20:
            continue
        rate = got[theme] / have[theme]
        flag = "  <-- weak" if rate < 0.5 else ""
        print(f"  {theme:18s} {got[theme]:4d}/{have[theme]:4d}  {rate:5.0%}{flag}")
    print("\ncommon Lichess themes we do not implement (coverage gaps, informational):")
    for theme in sorted(unimpl, key=lambda t: -unimpl[t])[:12]:
        print(f"  {theme:18s} {unimpl[theme]:4d}")
    print("\ndescribe_tactic renders a true sentence when we and Lichess agree on the tactic:")
    for theme in sorted(d_have):
        h = d_have[theme]
        print(f"  {theme:18s} {d_ok[theme]:4d}/{h:4d}  {d_ok[theme] / h:5.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
