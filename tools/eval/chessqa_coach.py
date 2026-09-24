"""The offline coach against ChessQA: what the coach says, checked against answers it never
sees -- ChessQA gives the right answer among its options, so a coach that is right says
what that answer says without being shown it (Tai, 2026-09-23).

No model is called. Three of ChessQA's families have an answer the coach's own facts can be
read against:

  Short Tactics       the coach's move, goal, leak, grounding and claims, checked as the
                      puzzle bench checks a Lichess puzzle (`coach_bench.score_board`)
  Position Judgement  who stands better, in the coach's words (`facts.standing_words`), against
                      the band of Stockfish's number ChessQA names (-400 .. 400)
  Motifs              the pins, skewers, forks and batteries the coach lists as on the board
                      (`coach/inventory.py`), and the discovered and double checks the tagger
                      finds, against ChessQA's complete lists

The pins and skewers the coach lists follow Tai's rule -- the piece behind is the king, or
can be won -- where ChessQA's are geometric; a ChessQA pin the coach leaves out because the
piece behind is defended is counted apart, as agreed. Semantic (commentary) needs words
matched to words and is not here.

  uv run python tools/eval/chessqa_coach.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools/puzzles"))

import chess  # noqa: E402

ITEMS = ROOT / "data/eval/chessqa/items.jsonl"


def load() -> list[dict]:
    with ITEMS.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _fen(item: dict) -> str:
    return item["input"].split(" moves ")[0].strip()


# ---- Short Tactics --------------------------------------------------------------------


def tactics(items: list[dict], analyses: dict) -> list[dict]:
    from coach_bench import score_board

    out = []
    for item in items:
        meta = json.loads(item["metadata_json"])
        board = chess.Board(_fen(item))
        pv = meta["pv"].split()
        solution = pv[1:] if pv and pv[0] != item["correct_answer"] else pv
        if not solution or solution[0] != item["correct_answer"]:
            solution = [item["correct_answer"], *solution[1:]]
        record = score_board(
            board,
            solution,
            set(meta.get("themes", [])),
            analyses[board.fen()],
            item["task_id"],
            int(meta.get("rating", 0)),
        )
        record["task"] = item["task_type"]
        out.append(record)
    return out


# ---- Position Judgement -----------------------------------------------------------------

_BANDS = {  # ChessQA's option, from White's side, and the coach's word for it
    400: "winning",
    200: "better",
    0: "about equal",
    -200: "worse",
    -400: "losing",
}


def judgement(items: list[dict], analyses: dict) -> list[dict]:
    from sidecoach.coach.facts import standing_words

    out = []
    for item in items:
        board = chess.Board(_fen(item))
        analysis = analyses[board.fen()]
        top = analysis.lines[0] if analysis.lines else None
        if top is None:
            continue
        white = top.expected_score if board.turn == chess.WHITE else 1 - top.expected_score
        words = standing_words(white)  # "you are winning" read as White's side
        said = next(
            w for w in ("winning", "better", "about equal", "worse", "losing") if w in words
        )
        want = _BANDS[int(item["correct_answer"])]
        # A neighbouring band is a near miss: 300 centipawns sits between "better" and
        # "winning", and the coach's bands are cut by expected score, not by pawns.
        order = ["losing", "worse", "about equal", "better", "winning"]
        near = abs(order.index(said) - order.index(want)) <= 1
        out.append(
            {
                "id": item["task_id"],
                "want": want,
                "said": said,
                "exact": said == want,
                "near": near,
                "cp": json.loads(item["metadata_json"]).get("actual_evaluation"),
            }
        )
    return out


# ---- Motifs -----------------------------------------------------------------------------


def _sq(s: chess.Square) -> str:
    return chess.square_name(s)


def _alpha(squares: list[chess.Square]) -> list[str]:
    return sorted(_sq(s) for s in squares)


def coach_motifs(board: chess.Board, kind: str) -> set[str]:
    """What the coach lists as on the board, in ChessQA's notation."""
    from sidecoach.analysis import motifs
    from sidecoach.coach.inventory import batteries, forks_in_place, pin_target, skewer_target

    out: set[str] = set()
    if kind == "pin":
        for s in chess.SQUARES:
            hit = pin_target(board, s)
            if hit is not None:
                out.add(f"{_sq(hit[0])}>{_sq(s)}>{_sq(hit[1])}")
    elif kind == "skewer":
        for s in chess.SQUARES:
            hit = skewer_target(board, s)
            if hit is not None:
                out.add(f"{_sq(s)}>{_sq(hit[0])}>{_sq(hit[1])}")
    elif kind == "fork":
        for colour in (chess.WHITE, chess.BLACK):
            for forker, targets in forks_in_place(board, colour):
                out.add(f"{_sq(forker)}>" + "-".join(_alpha(targets)))
    elif kind == "battery":
        for colour in (chess.WHITE, chess.BLACK):
            for group in batteries(board, colour):
                out.add(">".join(_alpha(group)))
    elif kind in ("discovered_check", "double_check"):
        want = "doubleCheck" if kind == "double_check" else "discoveredAttack"
        for move in board.legal_moves:
            if not board.gives_check(move):
                continue
            for tag in motifs.tags_for_move(board, move):
                # The tag lists the vacated square, then each slider and what it now
                # bears on: a discovered check has the king among them.
                king = board.king(not board.turn)
                king_hit = (
                    tag.name == "discoveredAttack"
                    and king is not None
                    and chess.square_name(king) in tag.squares[2:]
                )
                if tag.name == want and (want == "doubleCheck" or king_hit):
                    out.add(move.uci())
                    break
    return out


def _theirs(answer: str) -> set[str]:
    return {a.strip() for a in answer.split(",") if a.strip()}


def normalise(kind: str, found: set[str]) -> set[str]:
    """Both lists in one notation: a promotion once, as ChessQA writes it ("d7d8", not
    four pieces); a battery as its whole line, squares sorted (ChessQA writes c8>d8>f8 and
    e2>d3 where the coach has two pairs and d3>e2)."""
    if kind in ("discovered_check", "double_check"):
        return {m[:4] for m in found}
    if kind == "battery":
        groups = [set(g.split(">")) for g in found]
        merged = True
        while merged:
            merged = False
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if groups[i] & groups[j] and _collinear(groups[i] | groups[j]):
                        groups[i] |= groups.pop(j)
                        merged = True
                        break
                if merged:
                    break
        return {">".join(sorted(g)) for g in groups}
    return found


def _collinear(squares: set[str]) -> bool:
    pts = [(ord(s[0]) - 97, int(s[1])) for s in squares]
    (x0, y0), (x1, y1) = pts[0], pts[1]
    return all((x1 - x0) * (y - y0) == (y1 - y0) * (x - x0) for x, y in pts[2:])


def _why_skewer_left_out(board: chess.Board, skewer: str) -> str:
    from sidecoach.analysis import motifs

    a, b, c = (chess.parse_square(x) for x in skewer.split(">"))
    slider = board.piece_at(a)
    if slider is None or board.piece_at(c) is None:
        return "other"
    if not motifs.winnable(board, c, motifs._value(slider.piece_type), ignore=b):
        return "the piece behind is defended and worth no more (Tai's rule)"
    return "other"


def _why_left_out(board: chess.Board, pin: str) -> str:
    """Why the coach does not list a ChessQA pin: the rule that drops it."""
    from sidecoach.analysis import motifs

    a, b, c = (chess.parse_square(x) for x in pin.split(">"))
    slider = board.piece_at(a)
    behind = board.piece_at(c)
    if slider is None or behind is None:
        return "other"
    if behind.piece_type != chess.KING and not motifs.winnable(
        board, c, motifs._value(slider.piece_type), ignore=b
    ):
        return "the piece behind is defended and worth no more (Tai's rule)"
    front = board.piece_at(b)
    if front is not None and motifs._value(behind.piece_type) <= motifs._value(front.piece_type):
        return "the piece behind is worth no more than the front one"
    return "other"


def motif_check(items: list[dict]) -> dict:
    results: dict[str, Counter] = defaultdict(Counter)
    reasons: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for item in items:
        kind = item["task_type"].removeprefix("motifs_")
        board = chess.Board(_fen(item))
        ours = normalise(kind, coach_motifs(board, kind))
        theirs = normalise(kind, _theirs(str(item["correct_answer"])))
        if kind == "pin":
            # ChessQA lists pins to the king; the coach's relative pins (a piece that can be
            # won behind) are counted apart.
            relative = {
                p
                for p in ours - theirs
                if (k := board.piece_at(chess.parse_square(p.split(">")[2]))) is not None
                and k.piece_type != chess.KING
            }
            results["pin"]["relative_extra"] += len(relative)
            ours = ours - relative
        r = results[kind]
        r["items"] += 1
        r["exact"] += ours == theirs
        r["both"] += len(ours & theirs)
        r["ours_only"] += len(ours - theirs)
        r["theirs_only"] += len(theirs - ours)
        if kind == "pin":
            for pin in theirs - ours:
                reasons["pin: " + _why_left_out(board, pin)] += 1
        if kind == "skewer":
            for skewer in theirs - ours:
                reasons["skewer: " + _why_skewer_left_out(board, skewer)] += 1
        if ours != theirs and len(examples[kind]) < 3:
            examples[kind].append(f"{board.fen()}  ours {sorted(ours)}  theirs {sorted(theirs)}")
    return {"results": results, "pin_reasons": reasons, "examples": examples}


# ---- the run ----------------------------------------------------------------------------


def main() -> None:
    from analysis_cache import analyse_all

    items = load()
    tactic_items = [i for i in items if i["task_category"] == "Short Tactics"]
    judge_items = [i for i in items if i["task_category"] == "Position Judgement"]
    motif_items = [i for i in items if i["task_category"] == "Motifs"]

    motif = motif_check(motif_items)
    print("Motifs (the coach's lists against ChessQA's):")
    for kind, r in sorted(motif["results"].items()):
        n = r["items"]
        precision = r["both"] / max(r["both"] + r["ours_only"], 1)
        recall = r["both"] / max(r["both"] + r["theirs_only"], 1)
        print(
            f"  {kind:17} exact {r['exact'] / n:6.1%}  precision {precision:6.1%}  "
            f"recall {recall:6.1%}  (n={n})"
        )
    extra = motif["results"]["pin"]["relative_extra"]
    print(
        f"  (and {extra} relative pins the coach lists that ChessQA, pins to the king only, "
        "has not)"
    )
    if motif["pin_reasons"]:
        print("  ChessQA pins and skewers the coach leaves out, by why:")
        for why, k in motif["pin_reasons"].most_common():
            print(f"    {k:4}  {why}")
    for kind, lines in motif["examples"].items():
        for line in lines[:2]:
            print(f"    [{kind}] {line}")

    fens = [chess.Board(_fen(i)).fen() for i in tactic_items + judge_items]
    analyses = analyse_all(fens)

    records = tactics(tactic_items, analyses)
    n = len(records)
    print(f"\nShort Tactics: {sum(r['correct'] for r in records) / n:.2%} correct of {n}")
    for name in ("move", "goal", "amount", "leak", "grounded", "claims", "text"):
        scored = [r["checks"][name] for r in records if name in r["checks"]]
        if scored:
            print(f"  {name:9} {sum(scored) / len(scored):6.2%}  (n={len(scored)})")
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r["correct"])
    for task, oks in sorted(by_task.items()):
        print(f"  {task:40} {sum(oks) / len(oks):6.1%}  n={len(oks)}")
    for r in [r for r in records if not r["correct"]][:6]:
        bad = [k for k, v in r["checks"].items() if not v]
        print(f"    {r['id']} {bad}: {r['notes']}  | {r['texts']['idea'][:140]}")

    judged = judgement(judge_items, analyses)
    n = len(judged)
    exact = sum(j["exact"] for j in judged) / n
    near = sum(j["near"] for j in judged) / n
    print(
        f"\nPosition Judgement: the coach's words in ChessQA's band {exact:.1%}, "
        f"within one band {near:.1%} (n={n})"
    )
    confusion = Counter((j["want"], j["said"]) for j in judged if not j["exact"])
    for (want, said), k in confusion.most_common(8):
        print(f"    {k:4}  ChessQA {want:12} coach {said}")


if __name__ == "__main__":
    main()
