"""Ground-truth accuracy of verify_claims on puzzle positions (no coach LLM).

For each puzzle we know the solution, so a BEST claim on it should verify true and one on a
non-candidate move false; python-chess tells us which enemy pieces truly hang, so a HANGING
claim on a loose piece should verify true and on a guarded one false.

Run:  uv run python tools/eval/coach_qa_verify.py [n_puzzles]
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import chess

from sidecoach.analysis.annotator import GameAnnotator
from sidecoach.analysis.engine import Engine
from sidecoach.analysis.score import Thresholds
from sidecoach.cli import _load_book, _load_tagger
from sidecoach.coach.claims import Claim, ClaimKind, verify_claims
from sidecoach.coach.facts import facts_from_analysis
from sidecoach.config import load_settings, require_stockfish

SAMPLE = Path("data/puzzles/sample.csv")


def hanging_square(board: chess.Board, us: chess.Color) -> chess.Square | None:
    """An enemy piece attacked by us and defended by nothing."""
    for sq, pc in board.piece_map().items():
        if (
            pc.color != us
            and pc.piece_type not in (chess.PAWN, chess.KING)
            and board.attackers(us, sq)
            and not board.attackers(not us, sq)
        ):
            return sq
    return None


def safe_square(board: chess.Board, us: chess.Color) -> chess.Square | None:
    """An enemy piece we do not attack at all."""
    for sq, pc in board.piece_map().items():
        if (
            pc.color != us
            and pc.piece_type not in (chess.PAWN, chess.KING)
            and not board.attackers(us, sq)
        ):
            return sq
    return None


def main() -> int:
    n_puzzles = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    s = load_settings()
    eng = Engine(path=require_stockfish(s), threads=8, hash_mb=512, multipv=4, max_seconds=2.0)
    ann = GameAnnotator(
        engine=eng,
        depth=12,
        multipv=4,
        thresholds=Thresholds.from_settings(s.coach),
        tagger=_load_tagger(),
        book=_load_book(s),
        with_threats=True,
    )
    with SAMPLE.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    random.Random(11).shuffle(rows)

    tally = {k: [0, 0] for k in ("best_true", "best_false", "hang_true", "hang_false")}

    def count(key: str, verdict, want: str) -> None:
        tally[key][0] += 1
        tally[key][1] += verdict.result == want

    done = 0
    with eng:
        for row in rows:
            if done >= n_puzzles:
                break
            mv = row["Moves"].split()
            try:
                b = chess.Board(row["FEN"])
                b.push(chess.Move.from_uci(mv[0]))
                sol = chess.Move.from_uci(mv[1])
            except (ValueError, AssertionError, IndexError):
                continue
            if sol not in b.legal_moves:
                continue
            a = ann.analysis_event("g", b.copy(), 0)
            f = facts_from_analysis(b.copy(), a)
            if not f.candidates:
                continue
            done += 1
            us = b.turn
            best = Claim(ClaimKind.BEST, None, None, b.san(sol), "")
            count("best_true", verify_claims([best], b, facts=f)[0], "true")
            duds = [m for m in b.legal_moves if b.san(m) not in f.candidates]
            if duds:
                dud = Claim(ClaimKind.BEST, None, None, b.san(random.choice(duds)), "")
                count("best_false", verify_claims([dud], b, facts=f)[0], "false")
            hs = hanging_square(b, us)
            if hs is not None:
                hang = Claim(ClaimKind.HANGING, None, chess.square_name(hs), None, "")
                count("hang_true", verify_claims([hang], b, facts=f)[0], "true")
            ss = safe_square(b, us)
            if ss is not None:
                safe = Claim(ClaimKind.HANGING, None, chess.square_name(ss), None, "")
                count("hang_false", verify_claims([safe], b, facts=f)[0], "false")

    print(f"verify_claims ground-truth accuracy over {done} puzzle positions:")
    for k, (n, ok) in tally.items():
        if n:
            print(f"  {k:12s} {ok}/{n}  {ok / n:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
