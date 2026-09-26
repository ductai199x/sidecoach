"""How often real players meet and miss what the simulated students' habits are about: threats
and tactics in the Nova authors' Lichess rapid sample (`calibrate.py`'s), by rating band, read
exactly as the students' habits read them (tools/eval/habits.py: `Situation.threat`,
`Situation.tactic`, `Habit.misses`). The students' targets (habits.HUMAN_MISS) come from here.

A position is counted only when it is live and when the moves that led to it are known: the
player's previous position and move, and the opponent's reply between them (rebuilt from the
same game's rows), so an exchange in progress is read as the coach reads it.

A measure of "missed" must pass three checks before its numbers are used: misses fall with the
rating; misses are commoner where a threat or a tactic is on the board than elsewhere; and
examples from the lowest and highest bands read right by hand (`--review` writes them out).

    uv run --no-sync python tools/bots/habit_rates.py --per-band 2500 --workers 20
    uv run --no-sync python tools/bots/habit_rates.py --review
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "eval"))

import chess  # noqa: E402

SAMPLE = ROOT / "data" / "models" / "nova" / "unified_sample_600k.jsonl"
OUT = ROOT / "data" / "bots" / "calibration" / "habit_rates.jsonl"
REVIEW = ROOT / "data" / "bots" / "calibration" / "habit_review.txt"

_evaluator = None


def _init(depth: int | None = None) -> None:
    global _evaluator
    from habits import DEPTH, Evaluator  # pyright: ignore[reportMissingImports]

    _evaluator = Evaluator(depth=depth or DEPTH)


# How many of the player's earlier positions are rebuilt into the board's moves: two plies
# read ...Nxc3 bxc3 Bxf3 as a bishop to win back from after bxc3 (the depth-18 hand review,
# 2026-09-26); six read the run of captures whole.
HISTORY = 3


def _board(job: dict) -> chess.Board:
    """The position with the moves that led to it on the board."""
    board = chess.Board(job["start_fen"])
    for uci in job["moves"]:
        board.push_uci(uci)
    return board


def _measure(job: dict) -> dict | None:
    from habits import (  # pyright: ignore[reportMissingImports]
        HABITS,
        Situation,
        idea_of,
        swing,
        tactic_of,
    )

    assert _evaluator is not None
    board = _board(job)
    human = chess.Move.from_uci(job["actual"])
    if human not in board.legal_moves or board.is_game_over():
        return None
    _evaluator.fresh()
    top = _evaluator.top(board)
    if not top:
        return None
    played = _evaluator.lines(board, [human])[human]
    best = max(top[0][0], played[0])
    win, mates, tactic_line = tactic_of(board, [*top, played], best)
    passed = _evaluator.passed(board)
    owed = idea_of.worth(idea_of.exchange_owed(board))
    pv = top[0][1]
    threat_line: list[chess.Move] = passed[2] if passed else []
    s = Situation(
        best=best,
        probs={human: 1.0},
        scores={human: played[0]},
        candidates=[human],
        swings={human: swing(board, played[1], played[2])},
        threat_win=None if passed is None else passed[0],
        threat_mates=bool(passed and passed[1]),
        tactic_win=win,
        tactic_mates=mates,
        owed=owed,
    )
    if not s.live:
        return None
    passed_board = board.copy(stack=False)
    if not board.is_check():
        passed_board.push(chess.Move.null())
        passed_board = chess.Board(passed_board.fen())
    return {
        "band": job["band"],
        "rating": job["rating"],
        "fen": board.fen(),
        "start_fen": job["start_fen"],
        "moves": job["moves"],
        "played": human.uci(),
        "best_move": pv[0].uci(),
        "best": round(best, 4),
        "loss": round(s.loss(human), 4),
        "swing": s.swings[human],
        "threat_win": s.threat_win,
        "threat_mates": s.threat_mates,
        "tactic_win": s.tactic_win,
        "tactic_mates": s.tactic_mates,
        "owed": owed,
        "threat": s.threat,
        "tactic": s.tactic,
        "missed_threat": bool(HABITS["threat_blind"].misses(s)),
        "missed_tactic": bool(HABITS["tactic_blind"].misses(s)),
        "mistake": s.loss(human) >= 0.10,
        # The lines the labels were read from, and the move played with its line.
        "threat_line": _san(passed_board, threat_line) if passed else "",
        "tactic_line": _san(board, tactic_line),
        "played_line": _san(board, played[1]),
    }


def _san(board: chess.Board, line: list[chess.Move], n: int = 10) -> str:
    """A line in SAN, the side to move's material after each ply in brackets."""
    from habits import material  # pyright: ignore[reportMissingImports]

    b = board.copy(stack=False)
    who = b.turn
    base = material(b, who)
    out = []
    for m in line[:n]:
        if m not in b.legal_moves:
            break
        san = b.san(m)
        b.push(m)
        out.append(f"{san}[{material(b, who) - base:+d}]")
    return " ".join(out)


def _last_move(prev: dict, row: dict) -> str | None:
    """The opponent's move between the player's previous position and this one."""
    b = chess.Board(prev["fen"])
    mine = chess.Move.from_uci(prev["actual"])
    if mine not in b.legal_moves:
        return None
    b.push(mine)
    target = chess.Board(row["fen"])
    for m in b.legal_moves:
        b.push(m)
        same = b.board_fen() == target.board_fen() and b.turn == target.turn
        b.pop()
        if same:
            return m.uci()
    return None


def _report(out: list[dict]) -> None:
    bands = sorted({r["band"] for r in out}, key=lambda b: int(b.split("-")[0]))
    print(
        f"\n{'band':>10} {'live':>5} | {'mistake':>7} | {'threat on':>9} {'missed':>7}"
        f" | {'tactic on':>9} {'missed':>7} | per 40 live: threats, tactics"
    )
    for band in bands:
        rs = [r for r in out if r["band"] == band]
        th = [r for r in rs if r["threat"]]
        ta = [r for r in rs if r["tactic"]]

        def share(xs: list[dict], key: str) -> float:
            return sum(x[key] for x in xs) / len(xs) if xs else float("nan")

        print(
            f"{band:>10} {len(rs):>5} | {share(rs, 'mistake'):>7.1%}"
            f" | {len(th) / len(rs):>9.1%} {share(th, 'missed_threat'):>7.1%}"
            f" | {len(ta) / len(rs):>9.1%} {share(ta, 'missed_tactic'):>7.1%}"
            f" | {40 * len(th) / len(rs):.1f}, {40 * len(ta) / len(rs):.1f}"
        )


def _review(per_group: int, seed: int) -> None:
    """Positions from the lowest and highest bands, missed and not, written out whole: the
    board, the moves that led to it, and the very lines the labels were read from -- the
    opponent's if the player passed, the student's tactic, the move played -- with the
    material after every ply."""
    rows = [json.loads(x) for x in OUT.read_text("utf-8").splitlines() if x]
    rng = random.Random(seed)
    groups = []
    for band in ("800-1100", "2300-4000"):
        for kind, key in (("threat", "missed_threat"), ("tactic", "missed_tactic")):
            for missed in (True, False):
                pool = [r for r in rows if r["band"] == band and r[kind] and r[key] == missed]
                rng.shuffle(pool)
                title = f"{band}: {kind}, {'missed' if missed else 'met'}"
                groups.append((title, pool[:per_group]))
    near = [r for r in rows if r["band"] == "800-1100" and not r["threat"] and not r["tactic"]]
    near = [r for r in near if (r["threat_win"] or 0) >= 1 or r["tactic_win"] >= 1]
    rng.shuffle(near)
    groups.append(("800-1100: neither, but something is at stake", near[:per_group]))

    text = []
    for title, rs in groups:
        text += ["#" * 78, f"# {title}"]
        for r in rs:
            board = _board(r)
            us = board.turn
            prev = chess.Board(r["start_fen"])
            sans = []
            for uci in r["moves"]:
                sans.append(prev.san(chess.Move.from_uci(uci)))
                prev.push_uci(uci)
            moves = " ".join(sans)
            text += [
                "-" * 78,
                f"{r['band']} rated {r['rating']}; {'White' if us else 'Black'} to move after"
                f" {moves}",
                str(board),
                f"FEN {r['fen']}",
                f"standing {r['best']}; owed by the exchange in progress: {r['owed']}",
                f"if they passed (their material): {r['threat_line']}  (wins"
                f" {r['threat_win']}, mates {r['threat_mates']})",
                f"tactic line: {r['tactic_line']}  (wins {r['tactic_win']} net,"
                f" mates {r['tactic_mates']})",
                f"played: {r['played_line']}  (loss {r['loss']}, comes out {r['swing']:+d})",
                f"read as: threat={r['threat']} missed={r['missed_threat']}; tactic={r['tactic']}"
                f" missed={r['missed_tactic']}",
            ]
    REVIEW.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"wrote {REVIEW.relative_to(ROOT)}: {sum(len(rs) for _, rs in groups)} positions")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-band", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--review", action="store_true", help="write positions out for reading")
    ap.add_argument("--per-group", type=int, default=4)
    args = ap.parse_args()
    if args.review:
        _review(args.per_group, args.seed)
        return
    rows: dict[tuple, dict] = {}
    by_band: dict[str, list[dict]] = defaultdict(list)
    with SAMPLE.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows[(r["player_id"], r["ply"], r["band"])] = r
            if r["ply"] >= 12:
                by_band[r["band"]].append(r)
    rng = random.Random(args.seed)
    jobs = []
    for rs in by_band.values():
        rng.shuffle(rs)
        taken = 0
        for r in rs:
            if taken >= args.per_band:
                break
            # Back through the player's earlier positions in the same game, the opponent's
            # reply between each two rebuilt; at least one is needed.
            moves: list[str] = []
            at = r
            start = None
            for _ in range(HISTORY):
                prev = rows.get((at["player_id"], at["ply"] - 2, at["band"]))
                reply = _last_move(prev, at) if prev is not None else None
                if prev is None or reply is None:
                    break
                moves = [prev["actual"], reply, *moves]
                start, at = prev["fen"], prev
            if start is None:
                continue
            jobs.append(
                {
                    "band": r["band"],
                    "rating": r["rating"],
                    "start_fen": start,
                    "moves": moves,
                    "actual": r["actual"],
                }
            )
            taken += 1
    print(f"{len(jobs)} positions with the moves that led to them", flush=True)
    t0 = time.time()
    out = []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        for i, res in enumerate(pool.imap_unordered(_measure, jobs, chunksize=8), 1):
            if res is not None:
                out.append(res)
            if i % 250 == 0 or i == len(jobs):
                spent = time.time() - t0
                left = spent / i * (len(jobs) - i)
                print(
                    f"  {i}/{len(jobs)} ({i / len(jobs):.0%}) {spent:.0f} s, about"
                    f" {left / 60:.0f} min left",
                    flush=True,
                )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    _report(out)
    print(f"({time.time() - t0:.0f} s) -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
