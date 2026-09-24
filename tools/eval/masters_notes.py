"""The offline coach against the grandmasters' notes: at every annotated move of the master
games, does the coach see what the annotator saw?

The notes (data/eval/masters/, the books' commentary aligned to the games; the OCR lost the
piece letters from the moves, not the words) say what mattered: an attack on the king, the
open file, a weak pawn, a threat to meet, material to win, a won ending. They are read here
as concepts, by a word list, and the coach's own reading of the same moment is mapped to the
same concepts: its idea for the position the move was played from (goal, pattern, plan,
threats, positional features) and its word or lesson on the move itself. Per concept: how
often the coach sees what the note names (recall), and how often the coach names a tactic
at a moment whose note names none (the "pointless" signal: a true tactic that is not the
point). Heuristic by nature -- a note discusses variations and history too -- so the numbers
are read beside a side-by-side sample, never alone.

The facts come from data/eval/facts/masters_v2/ (the current engine settings and tagger).

  uv run python tools/eval/masters_notes.py --sample 40 --out notes_sample.md
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sidecoach.analysis import motifs  # noqa: E402
from sidecoach.coach import idea as idea_of  # noqa: E402
from sidecoach.coach import plans  # noqa: E402
from sidecoach.coach.coach import COST_YOU, idea_template, lesson_template  # noqa: E402
from sidecoach.coach.facts import facts_from_analysis  # noqa: E402
from sidecoach.coach.verdicts import verdict  # noqa: E402
from sidecoach.events import Analysis, MotifTag  # noqa: E402

MASTERS = ROOT / "data" / "eval" / "masters"
FACTS = ROOT / "data" / "eval" / "facts" / "masters_v2"

# What a note is about, by its words. Each concept is a chess idea a coach should name.
NOTE_CONCEPTS: dict[str, str] = {
    "king_attack": r"\battack(s|ing)? (on|against) (the|his|black's|white's) king|kingside attack|"
    r"king-side attack|mating attack|attack on the king|storm|h-file|pry open|"
    r"sac(rifice)?s? .{0,40}mate|attack\b.{0,30}\bking",
    "mate": r"\bmate\b|\bmates\b|mating|checkmate",
    "win_material": r"\bwins? (a|the|two|an|his|her) (pawn|piece|exchange|queen|rook|bishop|"
    r"knight)|winning (a|the) (pawn|piece|exchange)|wins material|loses? (a|the) (pawn|piece|"
    r"exchange)|drops? (a|the) (pawn|piece)",
    "threat": r"\bthreat|threaten|\bdefen[cs]e|\bdefend|parry|counterattack|counter-attack",
    "open_file": r"open file|half-open|\b[a-h]-file\b|doubl(e|ing) (the )?rooks|seventh rank|"
    r"7th rank",
    "passed_pawn": r"passed pawn|passer|queening|\bqueens\b|promot",
    "weak_pawns": r"isolated|backward|doubled pawn|weak pawn|weakness|hanging pawns|"
    r"pawn majority|minority attack|pawn structure",
    "squares": r"outpost|weak square|\bhole\b|strong point|weak (dark|light)|dark squares|"
    r"light squares",
    "endgame": r"\bending\b|\bendgame\b|king and pawn|opposition|zugzwang|simplif|"
    r"exchang(e|ing) (the )?(queens|rooks|pieces)|trade",
    "centre": r"\bcent(re|er)\b|pawn break|central",
    "development": r"develop|castl|king safety|in the middle|uncastled",
    "bishops": r"bishop pair|two bishops|bad bishop|good bishop|fianchetto",
    "sacrifice": r"\bsac(rifice)?\b|sacrifices|gives up",
    "fork": r"\bfork",
    "pin": r"\bpin(s|ned|ning)?\b",
    "draw": r"\bdraw|perpetual|stalemate|repetition",
}

# What the coach's reading is about: its goal, pattern and plan, mapped to the same ideas.
GOAL_CONCEPTS = {
    "mate": {"mate", "king_attack"},
    "attack": {"king_attack"},
    "win": {"win_material"},
    "combination": {"win_material", "sacrifice"},
    "promote": {"passed_pawn"},
    "convert": {"endgame"},
    "king": {"endgame"},
    "breakthrough": {"endgame", "passed_pawn"},
    "draw": {"draw"},
    "defend": {"threat"},
}
PLAN_CONCEPTS = {
    "takeFile": {"open_file"},
    "rookBehindPasser": {"passed_pawn"},
    "pushPasser": {"passed_pawn"},
    "passerPawnEnding": {"passed_pawn", "endgame"},
    "pawnStorm": {"king_attack"},
    "outpostMove": {"squares"},
    "centre": {"centre"},
    "break": {"centre"},
    "centralise": {"centre"},
    "develop": {"development"},
    "castle": {"development"},
    "prophylaxis": {"threat"},
    "rescue": {"threat"},
    "simplify": {"endgame"},
    "activateKing": {"endgame"},
    "space": {"centre"},
}
# The plan the best line carries out (`coach/plans.py`).
LINE_PLAN_CONCEPTS = {
    "attack": {"king_attack"},
    "storm": {"king_attack"},
    "double": {"open_file"},
    "outpost": {"squares"},
    "passer": {"passed_pawn"},
    "break": {"centre"},
    "wing": {"weak_pawns"},
}
PATTERN_CONCEPTS = {
    "fork": {"fork"},
    "pin": {"pin"},
    "absolutePin": {"pin"},
    "sacrifice": {"sacrifice"},
}
TACTIC_WORDS = r"\bforks?\b|\bpinned\b|\bskewers?\b|\bnow bears on\b|\bis loose\b|\btrapped\b"


def note_concepts(text: str) -> set[str]:
    low = text.lower()
    return {name for name, rx in NOTE_CONCEPTS.items() if re.search(rx, low)}


def coach_concepts(facts, idea, texts: list[str]) -> set[str]:  # noqa: ANN001
    out: set[str] = set()
    out |= GOAL_CONCEPTS.get(idea.goal, set())
    if facts.plan:
        kind = next((k for k, v in _PLAN_TEXT.items() if v == facts.plan), None)
        out |= PLAN_CONCEPTS.get(kind or "", set())
        line_kind = next(
            (k for k, v in plans.PLAN_NAME.items() if v.split("{")[0] in facts.plan), None
        )
        out |= LINE_PLAN_CONCEPTS.get(line_kind or "", set())
    if idea.blow is not None:
        out |= PATTERN_CONCEPTS.get(idea.blow.name, set())
    if idea.sacrifices:
        out.add("sacrifice")
    for text in [*texts, *(facts.ideas or [])]:
        out |= note_concepts(text)
    return out


from sidecoach.coach.facts import _PLAN_THEME as _PLAN_TEXT  # noqa: E402


def _paid(board: chess.Board, move: chess.Move, step: dict, tags: list[MotifTag]) -> list[MotifTag]:
    """The move's tags as the annotator now keeps them: a tactic only when the line after
    the move cashes it in (facts computed before that rule carry every tag)."""
    lines = step["analysis"].get("lines") or []
    pv = [chess.Move.from_uci(u) for u in (lines[0]["moves"] if lines else [])]
    line = [move, *pv]
    boards = [board.copy(stack=False)]
    for m in line:
        if m not in boards[-1].legal_moves:
            break
        nxt = boards[-1].copy(stack=False)
        nxt.push(m)
        boards.append(nxt)
    line = line[: len(boards) - 1]
    return [t for t in tags if t.name not in motifs.TARGETED or motifs.pays_off(boards, line, 0, t)]


def games() -> list[tuple[Path, Path]]:
    out = []
    for f in sorted(FACTS.glob("*.json")):
        book, game = f.stem.split("__", 1)
        m = MASTERS / book / f"{game}.json"
        if m.exists():
            out.append((f, m))
    return out


def readings(facts_path: Path, master_path: Path) -> list[dict]:
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    master = json.loads(master_path.read_text(encoding="utf-8"))
    notes = {p["ply"]: p.get("annotation") for p in master["plies"]}
    before_analysis = Analysis.model_validate(facts["start_analysis"])
    out = []
    moves = [s for s in facts["script"] if s["kind"] == "move"]
    for step in moves:
        board = chess.Board(step["fen_before"])
        move = chess.Move.from_uci(step["uci"])
        note = notes.get(step["ply"])
        assessment = step["assessment"]
        if (
            note
            and before_analysis.lines
            and before_analysis.fen.split()[:4] == board.fen().split()[:4]
        ):
            us = board.turn
            f = facts_from_analysis(board, before_analysis, user=us)
            idea = idea_of.read(board, before_analysis)
            label = assessment.get("label")
            tags = _paid(
                board, move, step, [MotifTag.model_validate(t) for t in assessment.get("tags", [])]
            )
            word, _ = verdict(
                board, move, label, tags, threats=before_analysis.threats, rng=random.Random(0)
            )
            best = f.candidates[0] if f.candidates else None
            lesson = ""
            if label in ("inaccuracy", "mistake", "blunder", "miss") and best:
                lesson = lesson_template(f, best, COST_YOU[0].format(best=best))
            said = idea_template(f)
            texts = [said, word, lesson]
            out.append(
                {
                    "game": facts_path.stem,
                    "ply": step["ply"],
                    "san": step["san"],
                    "label": label,
                    "note": note,
                    "note_concepts": sorted(note_concepts(note)),
                    "coach_concepts": sorted(coach_concepts(f, idea, texts)),
                    "goal": idea.goal,
                    "plan": f.plan,
                    "idea": said,
                    "word": word,
                    "lesson": lesson,
                    "tactic_named": bool(re.search(TACTIC_WORDS, word)),
                }
            )
        before_analysis = Analysis.model_validate(step["analysis"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    pairs = games()
    if args.limit:
        pairs = pairs[: args.limit]
    rows: list[dict] = []
    for fp, mp in pairs:
        rows.extend(readings(fp, mp))
    print(f"{len(pairs)} games, {len(rows)} annotated moves")
    seen: Counter[str] = Counter()
    hit: Counter[str] = Counter()
    for r in rows:
        for c in r["note_concepts"]:
            seen[c] += 1
            hit[c] += c in r["coach_concepts"]
    print("concept          note  coach sees it")
    for c, n in seen.most_common():
        print(f"  {c:15} {n:5}  {hit[c] / n:6.1%}")
    named = [r for r in rows if r["tactic_named"]]
    tactical = {"fork", "pin", "sacrifice", "win_material", "mate"}
    quiet = [r for r in named if not set(r["note_concepts"]) & tactical]
    print(f"words naming a tactic: {len(named)}; where the note names none: {len(quiet)}")
    goals = Counter(r["goal"] for r in rows)
    print(f"coach goals at annotated moves: {dict(goals)}")
    if args.out:
        rng = random.Random(args.seed)
        pick = rng.sample(rows, min(args.sample, len(rows)))
        lines = []
        for r in pick:
            lines.append(f"### {r['game']} ply {r['ply']} {r['san']} ({r['label']})")
            lines.append(f"- NOTE: {r['note'][:600]}")
            lines.append(f"- note concepts: {', '.join(r['note_concepts']) or '-'}")
            lines.append(f"- coach idea (before the move): {r['idea']}")
            lines.append(f"- coach word on the move: {r['word']}")
            if r["lesson"]:
                lines.append(f"- coach lesson: {r['lesson']}")
            lines.append(f"- coach concepts: {', '.join(r['coach_concepts']) or '-'}")
            lines.append("")
        Path(args.out).write_text("\n".join(lines), encoding="utf-8")
        print(f"sample -> {args.out}")
    dump = ROOT / "data" / "eval" / "masters_notes.jsonl"
    with dump.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"rows -> {dump}")


if __name__ == "__main__":
    main()
