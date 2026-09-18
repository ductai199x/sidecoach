"""Check the evaluation data against itself and against the board.

  masters     each ply's SAN and UCI apply legally from its fen_before and lead to the next
              ply's fen_before; the game's result matches its headers; annotations sit only
              on aligned plies and look like prose, not the next game
  facts       every scripted move applies legally; assessment and analysis events belong to
              their ply (ply, uci, fen); by_user agrees with the user's side; a takeback's
              fen is a position the game was at; the start analysis is of the start position
  split       every game of split.json has a facts file; train and held-out are disjoint
  judgments   every record parses; scores are consistent with the verdict; no pass with a
              hard fail (every run under data/eval/judgments, or --run one)
  ledger      every line has its fields; the totals by model

Run:  uv run python tools/eval/validate.py [--run <judgments run>]
Exit code 1 when anything is wrong; every problem is printed with its file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


EVAL = ROOT / "data" / "eval"
problems: list[str] = []
counts: Counter = Counter()


def bad(where: str, what: str) -> None:
    problems.append(f"{where}: {what}")


# ---- masters ----------------------------------------------------------------------------


def check_masters() -> None:
    for path in sorted((EVAL / "masters").glob("*/*.json")):
        g = json.loads(path.read_text(encoding="utf-8"))
        where = f"masters/{path.parent.name}/{path.name}"
        counts["masters games"] += 1
        board = chess.Board()
        plies = g["plies"]
        if not plies:
            bad(where, "no plies")
            continue
        for i, p in enumerate(plies):
            if p["ply"] != i + 1:
                bad(where, f"ply numbering off at {i}")
            if board.fen() != p["fen_before"]:
                bad(where, f"ply {p['ply']}: fen_before is not the position reached")
                board = chess.Board(p["fen_before"])
            try:
                move = board.parse_san(p["san"])
            except ValueError:
                bad(where, f"ply {p['ply']}: SAN {p['san']} illegal")
                break
            if move.uci() != p["uci"]:
                bad(where, f"ply {p['ply']}: SAN {p['san']} is {move.uci()}, not {p['uci']}")
            board.push(move)
            counts["masters plies"] += 1
            note = p.get("annotation") or ""
            if note:
                counts["masters annotated"] += 1
                if len(note) > 6000:
                    bad(
                        where,
                        f"ply {p['ply']}: annotation {len(note)} chars -- swallowed the next game?",
                    )
                if re.search(r"^#{1,3} Game \d+", note, re.M) or "Lessons from this game" in note:
                    bad(where, f"ply {p['ply']}: annotation runs into the next game")
        result = g.get("result", "*")
        if result not in ("1-0", "0-1", "1/2-1/2", "*"):
            bad(where, f"result {result!r}")
        if board.is_checkmate() and result == "*":
            bad(where, "checkmate on the board but no result")
        if g["aligned"] > len(plies) + 2:
            bad(where, f"aligned {g['aligned']} > plies {len(plies)}")


# ---- facts ------------------------------------------------------------------------------


def check_facts() -> dict[str, dict]:
    facts_by_name: dict[str, dict] = {}
    for path in sorted((EVAL / "facts").glob("*/*.json")):
        f = json.loads(path.read_text(encoding="utf-8"))
        where = f"facts/{path.parent.name}/{path.name}"
        counts["facts games"] += 1
        facts_by_name[f"{path.parent.name}/{path.stem}"] = f
        user_white = f["user_side"] == "white"
        if f["start_analysis"]["fen"] != f["start_fen"]:
            bad(where, "start_analysis is not of start_fen")
        board = chess.Board(f["start_fen"])
        seen_fens = {board.fen()}
        for step in f["script"]:
            kind = step["kind"]
            if kind == "takeback":
                if step["fen"] not in seen_fens:
                    bad(where, f"takeback to a position the game never reached: {step['fen'][:30]}")
                board = chess.Board(step["fen"])
                continue
            if kind == "ended":
                continue
            counts["facts plies"] += 1
            if board.fen() != step["fen_before"]:
                bad(where, f"ply {step['ply']}: fen_before is not the position reached")
                board = chess.Board(step["fen_before"])
            move = chess.Move.from_uci(step["uci"])
            if move not in board.legal_moves:
                bad(where, f"ply {step['ply']}: {step['uci']} illegal")
                continue
            if board.san(move) != step["san"]:
                bad(where, f"ply {step['ply']}: san {step['san']} != {board.san(move)}")
            if step["by_user"] != ((board.turn == chess.WHITE) == user_white):
                bad(where, f"ply {step['ply']}: by_user disagrees with the user's side")
            board.push(move)
            seen_fens.add(board.fen())
            if board.fen() != step["fen_after"]:
                bad(where, f"ply {step['ply']}: fen_after wrong")
            a, an = step["assessment"], step["analysis"]
            if a["ply"] != step["ply"] or a["uci"] != step["uci"]:
                bad(
                    where, f"ply {step['ply']}: assessment for another move ({a['ply']} {a['uci']})"
                )
            if an["fen"] != step["fen_after"] or an["ply"] != step["ply"]:
                bad(where, f"ply {step['ply']}: analysis for another position")
            if not (0.0 <= a["expected_loss"] <= 1.0):
                bad(where, f"ply {step['ply']}: expected_loss {a['expected_loss']}")
            if a.get("label") is None and a["severity"] == "none":
                bad(where, f"ply {step['ply']}: no label")
            for line in an["lines"]:
                b2 = chess.Board(an["fen"])
                for u in line["moves"][:3]:
                    m = chess.Move.from_uci(u)
                    if m not in b2.legal_moves:
                        bad(where, f"ply {step['ply']}: engine line {line['moves'][:3]} illegal")
                        break
                    b2.push(m)
    return facts_by_name


# ---- split, judgments ---------------------------------------------------------------


def check_split() -> None:
    path = EVAL / "split.json"
    if not path.exists():
        bad("split.json", "missing")
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    held, train = d.get("heldout_games", []), d.get("train_games", [])
    for name in held + train:
        source, _, stem = name.partition("/")
        if not (EVAL / "facts" / source / f"{stem}.json").exists():
            bad("split.json", f"{name} has no facts file")
    if set(held) & set(train):
        bad("split.json", f"{len(set(held) & set(train))} games both held out and in training")
    counts["split held-out games"] = len(held)
    counts["split training games"] = len(train)


def check_judgments(run: str | None) -> None:
    runs = [run] if run else sorted(p.name for p in (EVAL / "judgments").iterdir() if p.is_dir())
    for name in runs:
        for path in sorted((EVAL / "judgments" / name).glob("*.jsonl")):
            where = f"judgments/{name}/{path.name}"
            with path.open(encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    try:
                        j = json.loads(line)
                    except json.JSONDecodeError:
                        bad(where, f"line {n} malformed")
                        continue
                    counts["judgments"] += 1
                    s = j["scores"]
                    v = j.get("verdict")
                    if v:
                        ruled_false = any(
                            isinstance(c, dict) and c.get("supported") is False
                            for c in v.get("claims", [])
                        )
                        if s["fabrication"] != ruled_false:
                            bad(where, f"{j['id']}: fabrication flag disagrees with the verdict")
                    if s["pass"] and (
                        s["fabrication"] or s["leak"] or s["board"] or not s["form_ok"]
                    ):
                        bad(where, f"{j['id']}: pass with a hard fail")


def check_ledger() -> None:
    """The ledger: the original file plus one file per process (parallel runs append to
    their own, so lines never interleave)."""
    paths = [EVAL / "usage.jsonl"] + sorted((EVAL / "usage").glob("usage-*.jsonl"))
    total = Counter()
    for path in paths:
        if not path.exists():
            continue
        where = path.name if path.parent == EVAL else f"usage/{path.name}"
        for n, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            try:
                u = json.loads(line)
            except json.JSONDecodeError:
                bad(where, f"line {n} malformed")
                continue
            for k in ("run", "purpose", "model", "item", "prompt_tokens", "completion_tokens"):
                if k not in u:
                    bad(where, f"line {n} lacks {k}")
            total[u.get("model", "?")] += u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
            counts["ledger calls"] += 1
    for k, v in total.items():
        counts[f"ledger tokens {k}"] = v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="one judgments run to check (default: every run)")
    args = ap.parse_args()
    check_masters()
    check_facts()
    check_split()
    check_judgments(args.run)
    check_ledger()
    for k, v in sorted(counts.items()):
        print(f"{k}: {v:,}")
    if problems:
        kinds: Counter = Counter()
        for prob in problems:
            where, _, what = prob.partition(": ")
            what = re.sub(r"^ply \d+: ", "", what)
            what = re.sub(r"\d[\d,]*", "N", what)
            kinds[(where.split("/")[0], what[:100])] += 1
        print(f"\n{len(problems)} problems, by kind:")
        for (area, what), n in kinds.most_common():
            print(f"  {n:5d}  {area}: {what}")
        print("\nfirst 40:")
        for prob in problems[:40]:
            print("  " + prob)
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
