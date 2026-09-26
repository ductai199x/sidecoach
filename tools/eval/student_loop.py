"""The coach improved round by round on simulated students (students.py).

A round:
  1. The candidate coach (this working tree) and the baseline (the last round's candidate,
     its src/ exported from git) each coach the same fresh batch of student games -- games no
     correction was made from -- with the same students.
  2. Each notebook sits the exam (the frozen positions where the habit bites).
  3. The two coaches are compared session by session: errors the notebook fixed, good
     instincts it kept (paired). This is the held-out test of the last round's corrections.
  4. The candidate's sessions are read into a digest: the habit's moves and what the coach
     said at each, what the student wrote and found unclear, what its notebook tells it to
     check, how it did in the exam. The corrections for the next round are made from the
     digest (by hand: code, templates, the coach's directive) and committed; the next round
     tests them on a new batch.
The loop stops when two rounds running bring no improvement on their held-out batch.

  uv run --no-sync python tools/eval/student_loop.py prepare --batch 1 2
  uv run --no-sync python tools/eval/student_loop.py round --round 1 --batch 1 --coach none
      [--baseline <commit>] [--budget-usd 1.0]
  uv run --no-sync python tools/eval/student_loop.py compare r1-base r1-cand
  uv run --no-sync python tools/eval/student_loop.py digest r1-cand
  uv run --no-sync python tools/eval/student_loop.py status

State: data/eval/students/loop/loop.json (rounds, commits, results, spend); versions of the
coach under data/eval/students/versions/<commit>/src.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "eval"
sys.path.insert(0, str(TOOLS))
OUT = ROOT / "data" / "eval" / "students"
LOOP = OUT / "loop"
VERSIONS = OUT / "versions"
STATE = LOOP / "loop.json"
PY = sys.executable
STUDENT = "openai:gpt-5.6-luna@none"
GAMES_PER_CELL = 18
KEEP = 6  # three sessions a cell
# The check that cures each habit (students.CHECKS); for a threat, a check of the move's own
# safety is the other half (the piece moved off the threat to a square where it is lost).
CURE = {"threat_blind": "threats", "tactic_blind": "tactics", "greedy": "safety"}


def run(cmd: list[str], log: Path, env: dict | None = None) -> None:
    """A step, its output appended to the round's log as it comes."""
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})},
            check=False,
        )
    if proc.returncode:
        raise SystemExit(f"step failed ({proc.returncode}): {' '.join(cmd)} -- see {log}")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def export_src(commit: str) -> Path:
    """The coach's src/ at a commit, unpacked under versions/ (read from git, nothing in the
    working tree touched)."""
    full = git("rev-parse", commit)
    dest = VERSIONS / full[:12]
    if (dest / "src" / "sidecoach" / "__init__.py").exists():
        return dest / "src"
    dest.mkdir(parents=True, exist_ok=True)
    tar = subprocess.run(
        ["git", "archive", "--format=tar", full, "src"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as archive:
        archive.extractall(dest, filter="data")
    return dest / "src"


def load_state() -> dict:
    return json.loads(STATE.read_text("utf-8")) if STATE.exists() else {"rounds": []}


def save_state(state: dict) -> None:
    LOOP.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1), "utf-8")


# ---- prepare -----------------------------------------------------------------------------------


def cmd_prepare(args: argparse.Namespace) -> None:
    """Batches of fresh student games and their engine facts."""
    LOOP.mkdir(parents=True, exist_ok=True)
    for batch in args.batch:
        log = LOOP / f"prepare-b{batch}.log"
        t0 = time.time()
        games = [
            PY,
            str(TOOLS / "students.py"),
            "games",
            "--batch",
            str(batch),
            "--games-per-cell",
            str(args.games_per_cell),
            "--keep",
            str(args.keep),
            "--workers",
            str(args.workers),
        ]
        run(games, log)
        facts = [PY, str(TOOLS / "facts_parallel.py"), "students", "--batch", str(batch)]
        run([*facts, "--workers", str(args.workers)], log)
        print(f"batch {batch} ready ({time.time() - t0:.0f} s)", flush=True)


# ---- reading a run -----------------------------------------------------------------------------


def sessions(run_name: str) -> dict[str, dict]:
    return {
        p.stem: json.loads(p.read_text("utf-8"))
        for p in sorted((OUT / "runs" / run_name).glob("*.json"))
    }


def exam_rows(run_name: str, student: str) -> list[dict]:
    import students

    path = students.EXAM_RESULTS / students.tag(student) / f"{run_name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x]


def reads(run_name: str, student: str) -> dict[str, dict]:
    import students

    path = students.EXAM_RESULTS / students.tag(student) / f"{run_name}.reads.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def per_session(rows: list[dict]) -> dict[str, dict]:
    """Each session's exam: errors fixed and controls kept, as (k, n)."""
    out: dict[str, dict] = defaultdict(lambda: {"errors": [0, 0], "controls": [0, 0]})
    for r in rows:
        sid = r["notebook"].split(":", 1)[1]
        kind = "controls" if r["kind"] == "control" else "errors"
        out[sid][kind][0] += r["good"]
        out[sid][kind][1] += 1
    return out


def cured(book: dict | None, habit: str) -> bool:
    """The notebook tells its player to make the check that cures the habit, in the
    middlegame (where most of the habit's positions are)."""
    return bool(book) and CURE[habit] in book["checks"].get("middlegame", [])


def signals(s: dict, book: dict | None) -> dict:
    """What a session's transcript and notebook say about the coaching, beside the exam:
    how many notes carry the habit's cure (focus), whether the coach named the habit's
    pattern, what the student found unclear, and how often a piece's moves were read out."""
    import students
    from habits import HABITS

    from sidecoach.profile.causes import HABIT

    notes = s["notes"]
    per = students.per_note(book["said"], len(notes)) if book else []
    focus = sum(CURE[s["habit"]] in checks for checks, _ in per)
    words = HABIT[HABITS[s["habit"]].cause][0]
    coach = [t["text"] for t in s["transcript"] if t["who"] == "coach"]
    closed, carried = question_threads(s["transcript"])
    return {
        "notes": len(notes),
        "focus": focus,
        "pattern": any(words in text for text in coach),
        "unclear": len(s.get("unclear", [])),
        "dumps": sum(" can go to " in text for text in coach),
        "own": len(s.get("own_notes", [])),
        "closed": closed,
        "carried": carried,
    }


def question_threads(transcript: list[dict]) -> tuple[int, int]:
    """(closed, carried on): each answer the student gave to a question of the coach's, and
    whether the coach's reply to it closed the question or asked another (the AI coach asked
    a new question after every answer and resolved none, 2026-09-26)."""
    closed = carried = 0
    said = [t for t in transcript if t["who"] in ("coach", "you")]
    for k, e in enumerate(said):
        if e["who"] != "you" or "?" in e["text"] or k == 0:
            continue
        asked = said[k - 1]
        if asked["who"] != "coach" or not asked["text"].rstrip().endswith("?"):
            continue
        reply = []
        for x in said[k + 1 :]:
            if x["who"] != "coach" or x["ply"] != e["ply"]:
                break
            reply.append(x["text"])
        if reply:
            carried += reply[-1].rstrip().endswith("?")
            closed += not reply[-1].rstrip().endswith("?")
    return closed, carried


def signal_lines(name: str, runs: dict, books: dict, common: list[str]) -> list[str]:
    rows = [signals(runs[s], books.get("\n".join(runs[s]["notes"]))) for s in common]
    n = max(len(rows), 1)
    return [
        f"  {name:<14} notes carrying the cure {sum(r['focus'] for r in rows)}/"
        f"{sum(r['notes'] for r in rows)}; pattern named in {sum(r['pattern'] for r in rows)}/"
        f"{len(rows)} sessions; unclear {sum(r['unclear'] for r in rows) / n:.1f} a session;"
        f" pieces read out {sum(r['dumps'] for r in rows) / n:.1f} a session; own notes"
        f" {sum(r['own'] for r in rows)}",
        f"  {name:<14} answers to the coach's questions: closed"
        f" {sum(r['closed'] for r in rows)}, answered with another question"
        f" {sum(r['carried'] for r in rows)}",
    ]


# ---- compare -----------------------------------------------------------------------------------


def compare(base: str, cand: str, student: str) -> str:
    """The two runs on the same batch, session by session and position by position."""
    from bench import mcnemar, wilson

    lines = [f"{cand} against {base} (student {student})"]
    rows = {name: exam_rows(name, student) for name in (base, cand)}
    books = {name: reads(name, student) for name in (base, cand)}
    runs = {name: sessions(name) for name in (base, cand)}
    common = sorted(set(runs[base]) & set(runs[cand]))
    lines.append(f"sessions: {len(common)} in both ({len(runs[base])} / {len(runs[cand])})")
    if not common:
        return "\n".join(lines)
    for name in (base, cand):
        here = [r for r in rows[name] if r["notebook"].split(":", 1)[1] in common]
        for kind in ("errors", "controls"):
            sel = [r for r in here if (r["kind"] == "control") == (kind == "controls")]
            k, n = sum(r["good"] for r in sel), len(sel)
            lo, hi = wilson(k, n) if n else (0.0, 0.0)
            what = "fixed" if kind == "errors" else "kept"
            lines.append(
                f"  {name:<14} {kind} {what} {k}/{n} = {k / max(n, 1):.3f} [{lo:.2f}, {hi:.2f}]"
            )
        cure = sum(
            cured(books[name].get("\n".join(runs[name][s]["notes"])), runs[name][s]["habit"])
            for s in common
        )
        lines.append(f"  {name:<14} notebooks with the habit's check: {cure}/{len(common)}")
        lines += signal_lines(name, runs[name], books[name], common)
    by = {
        name: {(r["item"], r["notebook"].split(":", 1)[1]): r for r in rows[name]}
        for name in (base, cand)
    }
    for kind in ("errors", "controls"):
        pairs = [
            (by[base][key], by[cand][key])
            for key in by[base]
            if key in by[cand]
            and key[1] in common
            and (by[base][key]["kind"] == "control") == (kind == "controls")
        ]
        only_base = sum(a["good"] and not b["good"] for a, b in pairs)
        only_cand = sum(b["good"] and not a["good"] for a, b in pairs)
        lines.append(
            f"  paired {kind}: {len(pairs)} pairs; only {base} good {only_base}, only {cand} good"
            f" {only_cand}; McNemar p={mcnemar(only_base, only_cand):.3f}"
        )
    s_base, s_cand = per_session(rows[base]), per_session(rows[cand])
    better = worse = same = 0
    for sid in common:
        a = s_base[sid]["errors"][0]
        b = s_cand[sid]["errors"][0]
        better += b > a
        worse += b < a
        same += a == b
    lines.append(
        f"  sessions (errors fixed): {cand} better {better}, worse {worse}, the same {same};"
        f" sign test p={mcnemar(worse, better):.3f}"
    )
    # The positions of one notebook stand or fall together (the notebook either has the
    # cure or not): the honest test is on the sessions' differences, their signs flipped.
    for kind in ("errors", "controls"):
        diffs = [
            s_cand[sid][kind][0] - s_base[sid][kind][0]
            for sid in common
            if s_cand[sid][kind][1] == s_base[sid][kind][1]
        ]
        mean, p = sign_flip(diffs)
        lines.append(
            f"  sessions paired ({kind}): mean difference {mean:+.2f} positions a session,"
            f" sign-flip p={p:.3f} (n={len(diffs)})"
        )
    focus_better = focus_worse = 0
    for sid in common:
        a = signals(runs[base][sid], books[base].get("\n".join(runs[base][sid]["notes"])))
        b = signals(runs[cand][sid], books[cand].get("\n".join(runs[cand][sid]["notes"])))
        focus_better += b["focus"] > a["focus"]
        focus_worse += b["focus"] < a["focus"]
    lines.append(
        f"  sessions (notes carrying the cure): {cand} more {focus_better}, fewer {focus_worse};"
        f" sign test p={mcnemar(focus_worse, focus_better):.3f}"
    )
    return "\n".join(lines)


def sign_flip(diffs: list[int], draws: int = 20000) -> tuple[float, float]:
    """The mean of paired differences and its two-sided p by random sign flips (exact
    enumeration for up to 16 non-zero differences)."""
    import itertools
    import random

    if not diffs:
        return 0.0, 1.0
    observed = abs(sum(diffs))
    nonzero = [abs(d) for d in diffs if d]
    if len(nonzero) <= 16:
        sums = [
            abs(sum(s * d for s, d in zip(signs, nonzero, strict=True)))
            for signs in itertools.product((1, -1), repeat=len(nonzero))
        ]
        return sum(diffs) / len(diffs), sum(s >= observed for s in sums) / len(sums)
    rng = random.Random(0)
    hits = sum(
        abs(sum(d if rng.random() < 0.5 else -d for d in nonzero)) >= observed for _ in range(draws)
    )
    return sum(diffs) / len(diffs), (hits + 1) / (draws + 1)


def cmd_compare(args: argparse.Namespace) -> None:
    print(compare(args.base, args.cand, args.student))


# ---- digest ------------------------------------------------------------------------------------


def games_of(transcript: list[dict]) -> list[list[dict]]:
    """A session's transcript cut at each new game's marker."""
    out: list[list[dict]] = []
    for t in transcript:
        if t["who"] == "note" and t["text"].startswith("A new game"):
            out.append([])
        if out:
            out[-1].append(t)
    return out


def digest(run_name: str, student: str) -> str:
    """The run read for what to correct: for each session, its exam and notebook, what the
    notebook tells the player to check, what the student found unclear, and what the coach
    said at each of the habit's moves and in the review."""
    runs = sessions(run_name)
    books = reads(run_name, student)
    exam = per_session(exam_rows(run_name, student))
    out = [f"# Digest: {run_name} (student {student})", ""]
    silent = spoke = cure = 0
    for sid, s in runs.items():
        book = books.get("\n".join(s["notes"]))
        cure += cured(book, s["habit"])
        e, c = exam[sid]["errors"], exam[sid]["controls"]
        out.append(f"## {sid} -- {s['habit']} at {s['level']}; results {' '.join(s['results'])}")
        out.append(
            f"exam: errors fixed {e[0]}/{e[1]}, controls kept {c[0]}/{c[1]}; checks"
            f" {book['checks'] if book else '-'}; temperament: {s['temperament']}"
        )
        out.append("notebook:")
        out += [f"  - {n}" for n in s["notes"]] or ["  (empty)"]
        for key in ("unclear", "already_knew"):
            if s.get(key):
                out.append(f"{key}: " + " | ".join(s[key]))
        out.append(f"helpful: {s.get('helpful')}")
        played = zip(games_of(s["transcript"]), s["habit_plies"], strict=True)
        for i, (game, plies) in enumerate(played, 1):
            out.append(f"### game {i}: the habit's moves at plies {plies or 'none'}")
            for ply in plies:
                said = [t for t in game if t["ply"] == ply and t["who"] in ("coach", "you")]
                if not any(t["who"] == "coach" for t in said):
                    silent += 1
                    out.append(f"  ply {ply}: the coach said nothing")
                    continue
                spoke += 1
                for t in said:
                    out.append(f"  ply {ply} {t['who']}: {t['text']}")
            review = [t["text"] for t in game if t.get("trigger") == "review"]
            if review:
                out.append("  review: " + " ".join(review))
        out.append("")
    out.insert(
        2,
        f"The habit's moves the coach spoke to: {spoke} of {spoke + silent}. Notebooks with the"
        f" habit's check: {cure} of {len(runs)}.\n",
    )
    return "\n".join(out)


def cmd_digest(args: argparse.Namespace) -> None:
    text = digest(args.run, args.student)
    path = LOOP / f"{args.run}-digest.md"
    LOOP.mkdir(parents=True, exist_ok=True)
    path.write_text(text, "utf-8")
    print(text.splitlines()[2])
    print(f"-> {path}")


# ---- a round -----------------------------------------------------------------------------------


def session_cmd(run_name: str, args: argparse.Namespace, budget: float) -> list[str]:
    return [
        PY,
        str(TOOLS / "students.py"),
        "session",
        "--run",
        run_name,
        "--coach",
        args.coach,
        "--student",
        args.student,
        "--batch",
        str(args.batch),
        "--replicates",
        str(args.replicates),
        *(["--only", *args.only] if args.only else []),
        "--workers",
        str(args.workers),
        "--budget-usd",
        f"{budget:.2f}",
    ]


def cmd_round(args: argparse.Namespace) -> None:
    state = load_state()
    log = LOOP / f"r{args.round}.log"
    LOOP.mkdir(parents=True, exist_ok=True)
    cand, base = f"r{args.round}-cand", f"r{args.round}-base"
    dirty = bool(git("status", "--porcelain", "src"))
    record = {
        "round": args.round,
        "batch": args.batch,
        "coach": args.coach,
        "student": args.student,
        "candidate": git("rev-parse", "HEAD") + (" (src dirty)" if dirty else ""),
        "baseline": args.baseline,
        "started": time.strftime("%Y-%m-%d %H:%M"),
    }
    print(f"round {args.round}: batch {args.batch}, coach {args.coach} -> {log}", flush=True)
    t0 = time.time()
    run(session_cmd(cand, args, args.budget_usd), log)
    print(f"  {cand} sessions done ({time.time() - t0:.0f} s)", flush=True)
    runs = [cand]
    if args.baseline:
        src = export_src(args.baseline)
        run(session_cmd(base, args, args.budget_usd), log, env={"SIDECOACH_SRC": str(src)})
        print(f"  {base} sessions done ({time.time() - t0:.0f} s)", flush=True)
        runs.append(base)
    exam = [PY, str(TOOLS / "students.py"), "exam", "--student", args.student]
    run([*exam, "--arms", ",".join(runs), "--workers", "16", "--budget-usd", "2.00"], log)
    print(f"  exams done ({time.time() - t0:.0f} s)", flush=True)
    text = digest(cand, args.student)
    (LOOP / f"{cand}-digest.md").write_text(text, "utf-8")
    record["digest"] = text.splitlines()[2]
    if args.baseline:
        record["compare"] = compare(base, cand, args.student)
        (LOOP / f"r{args.round}-compare.txt").write_text(record["compare"], "utf-8")
        print(record["compare"])
    record["seconds"] = round(time.time() - t0)
    state["rounds"] = [r for r in state["rounds"] if r["round"] != args.round] + [record]
    save_state(state)
    print(record["digest"])
    print(f"digest: {LOOP / f'{cand}-digest.md'}")


def cmd_status(args: argparse.Namespace) -> None:
    for r in load_state()["rounds"]:
        print(f"round {r['round']} (batch {r['batch']}, {r['coach']}): {r['candidate'][:12]}")
        print(f"  {r.get('digest', '')}")
        for line in r.get("compare", "").splitlines()[1:]:
            print(f"  {line}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--batch", type=int, nargs="+", required=True)
    p.add_argument("--games-per-cell", type=int, default=GAMES_PER_CELL)
    p.add_argument("--keep", type=int, default=KEEP)
    p.add_argument("--workers", type=int, default=20)
    r = sub.add_parser("round")
    r.add_argument("--round", type=int, required=True)
    r.add_argument("--batch", type=int, required=True)
    r.add_argument("--coach", default="none")
    r.add_argument("--student", default=STUDENT)
    r.add_argument("--baseline", help="the commit whose coach the candidate is tested against")
    r.add_argument("--budget-usd", type=float, default=0.5, help="for each run of sessions")
    r.add_argument("--workers", type=int, default=10)
    r.add_argument("--replicates", type=int, default=2, help="students for each session")
    r.add_argument("--only", nargs="*", help="only the sessions whose names hold one of these")
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("cand")
    c.add_argument("--student", default=STUDENT)
    d = sub.add_parser("digest")
    d.add_argument("run")
    d.add_argument("--student", default=STUDENT)
    sub.add_parser("status")
    args = ap.parse_args()
    commands = {
        "prepare": cmd_prepare,
        "round": cmd_round,
        "compare": cmd_compare,
        "digest": cmd_digest,
        "status": cmd_status,
    }
    commands[args.cmd](args)


if __name__ == "__main__":
    main()
