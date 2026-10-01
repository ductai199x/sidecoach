"""Record what the desktop's positional layer says, for the Kotlin port to be held to.

This is the half of the coach's knowledge that is not tactics: the open file, the passed
pawn, the outpost, the bad bishop, the Lucena and the Philidor, and -- the highest-value
fact of all -- what the engine's best move is *for*. The coach teaches from these sentences
word for word, so the port has to produce them word for word.

Everything here is pure, so it is compared exactly. The positions come from three places:
the Lichess puzzle sample (middlegames and endgames with something going on), random games
(openings, development, uncastled kings), and a handful of built endings for the textbook
detectors that a random game almost never reaches. The move ideas are taken over every legal
move of a position, which covers the idea table densely without an engine. Two threats are
tried for the prophylaxis idea: the opponent's first legal move after a pass, and the first
capture of one of our pieces that wins it by the rules (`exchange.exchange` > 0), over every
legal move -- the capture is where "a guard that still loses the piece stops nothing" lives.

    cd D:/chess-coaching-companion
    uv run --no-sync python D:/sidecoach-android/tools/make_positional_fixtures.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402

from sidecoach.analysis import exchange, positional  # noqa: E402
from sidecoach.analysis.phase import phase_of  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"
PUZZLES = DESKTOP / "data" / "puzzles" / "sample.csv"

# The textbook endings a random game does not reach, so every detector is exercised.
BUILT = [
    "8/8/8/8/8/1k6/1P6/1K6 w - - 0 1",            # opposition, key squares
    # Key squares, distant opposition: three squares between the kings. (The position here
    # before the parity audit, 2026-09-29, had the kings two squares apart: plain opposition.)
    "8/3k4/8/8/8/3K4/3P4/8 w - - 0 1",
    "1K6/1P6/8/8/8/8/r7/2k4R w - - 0 1",          # Lucena
    "8/4k3/r7/4P3/4K3/8/8/7R b - - 0 1",          # Philidor: rook on the third rank
    # The detectors' negatives (the parity audit, 2026-09-29): the rook off the third rank, so
    # the Philidor is not set up; a rook pawn, which has no key squares; the defending king
    # beside the pawn, not cut off, so no Lucena.
    "8/4k3/8/4P3/4K3/8/8/r6R b - - 0 1",
    "4k3/8/2K5/8/P7/8/8/8 w - - 0 1",
    "3K4/3Pk3/8/8/8/8/8/R6r w - - 0 1",
    "8/8/3K4/8/3P4/8/8/7k w - - 0 1",             # the king on a key square
    "8/8/8/8/1k6/8/1p6/1K1R4 b - - 0 1",          # rook and pawn v rook
    "8/8/1R6/8/8/1k6/3r4/1K6 w - - 0 1",
    "3k4/8/8/3K4/3P4/8/8/8 w - - 0 1",
    "8/5p2/8/8/8/8/5P2/8 w - - 0 1",              # bare pawn race
    "8/8/8/8/k7/8/5P2/6K1 w - - 0 1",             # square of the pawn
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "4k3/8/8/8/8/8/4P3/4K2R w K - 0 1",           # uncastled with rights
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",  # Sicilian structure
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9",  # IQP shapes
    "r1bq1rk1/pp3ppp/2n1pn2/2bp4/8/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 10",
    "rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 6",
    "r1bqk2r/pppn1ppp/4pn2/3p4/1bPP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 6",
    "8/pp3ppp/8/2p5/2P5/8/PP3PPP/8 w - - 0 20",   # majorities
    "8/1p3p2/p7/8/8/P7/1P3P2/8 w - - 0 30",
    "r5k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 20",     # open file
    "3r2k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 20",
    "8/3k4/3p4/3P4/3K4/8/8/8 w - - 0 40",         # closed, pawn pointing shapes
    "rnbqk2r/pp2ppbp/3p1np1/2pP4/4P3/2N2N2/PP2BPPP/R1BQK2R b KQkq - 0 7",  # King's Indian
    "rnbqkbnr/pp3ppp/4p3/2ppP3/3P4/8/PPP2PPP/RNBQKBNR w KQkq - 0 4",       # French advance
    # The desktop's rescue test: a knight a pawn attacks with a pawn-guarded square to run to.
    # (Its knight guarded beside the enemy king and its knight attacked only by a pinned knight
    # are in BUILT_THREATS, with the threat each is about.)
    "4k3/p7/8/3p4/2N5/8/8/4K3 w - - 0 1",
    # Desktop 3d9568d's gates: a bare king before a queen and a knight (no king's march), a
    # king that captures, a king in check, h4 with a queen against a bishop (no storm in an
    # ending with pieces), an even trade of rooks while ahead and a loose bishop taken.
    "8/8/8/3k4/8/8/1QN5/4K3 b - - 0 1",
    "8/8/8/3k4/3p4/4K3/8/8 w - - 0 1",
    "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
    "4b1k1/5pp1/8/8/8/8/5PPP/4Q1K1 w - - 0 1",
    "3rk3/8/8/8/8/8/8/Q2RK3 w - - 0 1",
    "4k3/8/8/3b4/8/8/8/Q2RK3 w - - 0 1",
]

# The desktop's prophylaxis tests, with the threat each is asked about: a guard that still
# loses the knight to the pawn, and a guard pinned to its own king. Then desktop 3d9568d's
# rescue tests (a rescue is of the piece their threat takes, never a pawn): the bishop on b5
# with their threat on it and elsewhere, the pawn on e4 that steps on, the knight on c4.
BUILT_THREATS = [
    ("4k3/8/8/8/1p6/2N5/8/2B1K3 w - - 0 1", "b4c3"),
    ("4k3/8/8/3b4/5n2/P6B/6P1/7K w - - 0 1", "f4h3"),
    ("rnbqkbnr/1ppp1ppp/p7/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 3", "a6b5"),
    ("rnbqkbnr/1ppp1ppp/p7/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 3", "d8h4"),
    ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "d5e4"),
    ("4k3/p7/8/3p4/2N5/8/8/4K3 w - - 0 1", "d5c4"),
    # Moved here from BUILT with their threats (the parity audit, 2026-09-29): a knight guarded
    # beside the enemy king, which the king cannot take, and a knight attacked only by a
    # knight pinned to its king -- neither is in danger, so moving it is no rescue.
    ("8/8/5k2/4N3/3P4/8/8/4K3 w - - 0 1", "f6e5"),
    ("4k3/8/2n5/1B6/3N4/8/8/4K3 w - - 0 1", "c6d4"),
]


def feature_json(fact) -> dict:
    return {"kind": fact.kind, "text": fact.text, "squares": list(fact.squares)}


def capture_threat(board: chess.Board) -> chess.Move | None:
    """The opponent's first capture, after a pass, that wins one of our pieces by the rules."""
    if board.is_check():
        return None
    passed = board.copy(stack=False)
    passed.push(chess.Move.null())
    for move in passed.legal_moves:
        victim = passed.piece_at(move.to_square)
        if victim is not None and victim.color == board.turn and exchange.exchange(passed, move) > 0:
            return move
    return None


def case_for(board: chess.Board, phase: str, threat_uci: str | None = None) -> dict:
    us = board.turn
    them = not us
    worst = positional.worst_piece_square(board, us)
    ideas = []
    for move in board.legal_moves:
        idea = positional.best_move_idea(board, move, us, phase=phase, worst=worst, threat=None)
        ideas.append({"move": move.uci(), "idea": None if idea is None else feature_json(idea)})
    # The prophylaxis branch needs a threat: the opponent's first legal move after a pass.
    threat = None
    if not board.is_check():
        passed = board.copy(stack=False)
        passed.push(chess.Move.null())
        threat = next(iter(passed.legal_moves), None)
    # And a threat that wins something, tried against every legal move.
    capture = chess.Move.from_uci(threat_uci) if threat_uci else capture_threat(board)
    capture_ideas = []
    if capture is not None:
        for move in board.legal_moves:
            idea = positional.best_move_idea(
                board, move, us, phase=phase, worst=worst, threat=capture
            )
            capture_ideas.append(
                {"move": move.uci(), "idea": None if idea is None else feature_json(idea)}
            )
    threat_ideas = []
    if threat is not None:
        for move in list(board.legal_moves)[:12]:
            idea = positional.best_move_idea(
                board, move, us, phase=phase, worst=worst, threat=threat
            )
            threat_ideas.append(
                {"move": move.uci(), "idea": None if idea is None else feature_json(idea)}
            )
    return {
        "fen": board.fen(),
        "phase": phase,
        "worst": None if worst is None else chess.square_name(worst),
        "features_us": [feature_json(f) for f in positional.features(board, us, phase=phase)],
        "features_them": [feature_json(f) for f in positional.features(board, them, phase=phase)],
        "top_us": [feature_json(f) for f in positional.top_features(board, us, phase=phase)],
        "top_them": [feature_json(f) for f in positional.top_features(board, them, phase=phase)],
        "ideas": ideas,
        "threat": None if threat is None else threat.uci(),
        "threat_ideas": threat_ideas,
        "capture_threat": None if capture is None else capture.uci(),
        "capture_threat_ideas": capture_ideas,
    }


QUOTA = 25


def kinds_of(case: dict) -> set[str]:
    out = {f["kind"] for f in case["features_us"] + case["features_them"]}
    out |= {
        i["idea"]["kind"]
        for i in case["ideas"] + case["threat_ideas"] + case["capture_threat_ideas"]
        if i["idea"]
    }
    return out or {"(none)"}


def choose(cases: list[dict], keep: int = 0) -> list[dict]:
    """The cases that bring a kind still under its quota; the first `keep` (the built ones,
    each there for a detector or a negative of its own) whatever their kinds -- the quota
    dropped one phase of a built position once."""
    counts: dict[str, int] = {}
    kept: list[dict] = []
    for index, case in enumerate(cases):
        kinds = kinds_of(case)
        if index < keep or any(counts.get(kind, 0) < QUOTA for kind in kinds):
            kept.append(case)
            for kind in kinds:
                counts[kind] = counts.get(kind, 0) + 1
    return kept


def main() -> None:
    rng = random.Random(20260922)
    cases: list[dict] = []

    for fen in BUILT:
        board = chess.Board(fen)
        for phase in ("opening", "middlegame", "endgame"):
            cases.append(case_for(board, phase))
    for fen, threat in BUILT_THREATS:
        board = chess.Board(fen)
        for phase in ("opening", "middlegame", "endgame"):
            cases.append(case_for(board, phase, threat))
    built = len(cases)

    # Random games: openings and middlegames, where development and the king's safety live.
    for _ in range(80):
        board = chess.Board()
        for ply in range(rng.randint(2, 60)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if ply % 5 == 0:
                cases.append(case_for(board, phase_of(board)))

    # The puzzle sample: real middlegames and endgames.
    if PUZZLES.exists():
        with PUZZLES.open(newline="", encoding="utf-8") as handle:
            for row in list(csv.DictReader(handle))[:900]:
                board = chess.Board(row["FEN"])
                moves = [chess.Move.from_uci(u) for u in row["Moves"].split()]
                for index, move in enumerate(moves):
                    if move not in board.legal_moves:
                        break
                    if index % 2 == 1:
                        cases.append(case_for(board, phase_of(board)))
                    board.push(move)

    seen_all = len(cases)
    cases = choose(cases, keep=built)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "positional.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")
    counts: dict[str, int] = {}
    for case in cases:
        for kind in kinds_of(case):
            counts[kind] = counts.get(kind, 0) + 1
    print(f"{len(cases)} of {seen_all} positions -> {OUT / 'positional.json'}")
    print("kinds: " + ", ".join(f"{k}x{v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
