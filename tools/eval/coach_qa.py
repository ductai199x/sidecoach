"""Layer 2: grade the coach's actual reply to a user's claim, using puzzle ground truth.

For each puzzle we know the solution and the board, so a claim's truth is known: "is the
solution best?" (yes), "is this dud best?" (no), "is this loose piece hanging?" (yes), "is
this guarded piece hanging?" (no). We run the whole coach -- parse the claim, verify it, reply
-- and a stance grader (agree / disagree / unclear) checks the reply against the known truth.
Every exchange is printed so the grades can be read back.

Run:  uv run python tools/eval/coach_qa.py [n_puzzles]
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
from sidecoach.cli import _claims_check, _draft_check, _load_book, _load_tagger
from sidecoach.coach.coach import Coach
from sidecoach.coach.facts import facts_from_analysis
from sidecoach.coach.policy import InterjectionPolicy, PolicySettings
from sidecoach.config import load_settings, require_stockfish
from sidecoach.events import GameStarted, Side
from sidecoach.llm.factory import make_provider

_PIECE = {chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook", chess.QUEEN: "queen"}


def grade(provider, claim: str, truth: bool, reply: str) -> str:
    """AGREE / DISAGREE / UNCLEAR: what stance the coach's reply took on the claim."""
    system = (
        "You judge whether a chess coach's reply agrees with, disagrees with (corrects), or is "
        "unclear about a student's claim. Answer with exactly one word: AGREE, DISAGREE, "
        "or UNCLEAR."
    )
    user = f'Student claim: "{claim}"\n\nCoach reply: "{reply}"\n\nOne word:'
    try:
        out = provider.chat(
            system, [{"role": "user", "content": user}], max_tokens=8, temperature=0.0
        )
    except Exception:  # noqa: BLE001
        return "UNCLEAR"
    text = out.text.upper()
    for word in ("DISAGREE", "AGREE", "UNCLEAR"):
        if word in text:
            return word
    return "UNCLEAR"


def hanging_and_safe(board: chess.Board, us: chess.Color):
    hang = safe = None
    for sq, pc in board.piece_map().items():
        if pc.color == us or pc.piece_type in (chess.PAWN, chess.KING):
            continue
        if board.attackers(us, sq) and not board.attackers(not us, sq):
            hang = hang or sq
        elif not board.attackers(us, sq):
            safe = safe or sq
    return hang, safe


def cases(board: chess.Board, solution: chess.Move, facts):
    """User claims with known truth: (claim_text, is_true)."""
    us = board.turn
    out = []
    out.append((f"I want to play {board.san(solution)}, is that a strong move here?", True))
    duds = [
        m
        for m in board.legal_moves
        if board.san(m) not in facts.candidates and not board.is_capture(m)
    ]
    if duds:
        out.append(
            (
                f"I'm thinking about playing {board.san(random.choice(duds))}. "
                "Is that a good move?",
                False,
            )
        )
    hang, safe = hanging_and_safe(board, us)
    if hang is not None:
        p = _PIECE[board.piece_type_at(hang) or chess.PAWN]
        out.append((f"I think their {p} on {chess.square_name(hang)} is hanging, right?", True))
    elif safe is not None:
        p = _PIECE[board.piece_type_at(safe) or chess.PAWN]
        out.append(
            (f"Their {p} on {chess.square_name(safe)} looks like it's hanging to me.", False)
        )
    return out


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    s = load_settings()
    provider = make_provider(s.llm)
    eng = Engine(path=require_stockfish(s), threads=8, hash_mb=512, multipv=4, max_seconds=2.5)
    ann = GameAnnotator(
        engine=eng,
        depth=14,
        multipv=4,
        thresholds=Thresholds.from_settings(s.coach),
        tagger=_load_tagger(),
        book=_load_book(s),
        with_threats=True,
    )
    with Path("data/puzzles/sample.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    random.Random(3).shuffle(rows)

    correct = total = 0
    with eng:
        done = 0
        for row in rows:
            if done >= n:
                break
            mv = row["Moves"].split()
            try:
                b = chess.Board(row["FEN"])
                b.push(chess.Move.from_uci(mv[0]))
                sol = chess.Move.from_uci(mv[1])
            except Exception:  # noqa: BLE001
                continue
            if sol not in b.legal_moves:
                continue
            a = ann.analysis_event("g", b.copy(), 0)
            f = facts_from_analysis(b.copy(), a)
            if not f.candidates:
                continue
            done += 1
            us = Side.WHITE if b.turn == chess.WHITE else Side.BLACK
            print("#" * 74)
            print(
                f"# puzzle {row['PuzzleId']} | solver {us.value} | solution {b.san(sol)}"
                f" | themes {row['Themes'][:50]}"
            )
            for claim, truth in cases(b, sol, f):
                coach = Coach(
                    provider,
                    policy=InterjectionPolicy(PolicySettings(stuck_seconds=30)),
                    draft_check=_draft_check(),
                    claims=_claims_check(provider),
                )
                coach.on_event(GameStarted(game_id="g", user_side=us, start_fen=b.fen()), 0.0)
                coach.on_event(a, 0.0)
                coach.ask(claim, 10.0)
                coach.speaker.run_pending()
                reply = " ".join(m.text for m in coach.poll())
                stance = grade(provider, claim, truth, reply)
                want = "AGREE" if truth else "DISAGREE"
                ok = stance == want
                correct += ok
                total += 1
                print(f"\n  USER: {claim}   [truth={'TRUE' if truth else 'FALSE'}]")
                print(f"  COACH: {reply}")
                print(f"  -> stance {stance} (want {want}) {'OK' if ok else 'MISS'}")
    print("\n" + "=" * 40)
    print(
        f"reply matches ground truth: {correct}/{total} ({correct / total:.0%})"
        if total
        else "no cases"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
