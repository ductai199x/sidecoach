"""The excellence rubric: how good a finished run's replies are beyond true and complete.

The criteria are in tools/eval/rubric.toml, for the owner to edit: each one yes/no question
about a reply against the record of its moment, asked for the classes it applies to (and,
for `only_if = "ends_with_question"`, only when the reply ends with a question -- decided in
code). The judge sees the record as its evidence already is (judge.evidence_block: the
message the coach was given, the coach's own earlier words marked as evidence only of what
it said -- which is what the "connection" criterion's link to a lesson already given needs
-- with the facts beside it), then the reply, then the one question; effort none, up to
three votes that stop once two
agree, every call budget-checked on the ledger. A call that fails or an answer that is not
yes or no is not a verdict: the criterion is counted as failed and not scored.

It judges an EXISTING run: no coach calls. The AI arm is the run's judged moments where the
model's own reply was spoken; with --floor, the templates of the run's paired moments too.
Answers are kept under data/eval/judgments/<run>-rubric (and <run>-rubric-floor), per
criterion and per question's wording: a rerun pays only for what it has not asked, and an
edited question is asked again. Like every judgment they carry the code's version, and
answers from other code are refused unless --reuse-stale.

  uv run python tools/eval/rubric.py --run effort-medium-47138dd-r1 --floor --budget-usd 1 \
      [--dry-run]

The report (data/eval/rubric/<run>.md and .json): per criterion the yes-rate of each arm;
per class, and overall, the excellence score -- per reply the mean yes-rate over the
criteria that apply to it, then the mean over replies; the paired comparison where both
arms were judged; and the lowest-scoring AI replies with the criteria they failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import tomllib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import estimate  # noqa: E402
import judge as judge_module  # noqa: E402
import stamp  # noqa: E402
from judge import (  # noqa: E402
    COACH_WORDS_RULE,
    BudgetExceeded,
    JudgeFailed,
    Ledger,
    _chat_with_backoff,
    binary_judge,
    evidence_block,
    reply_status,
    settings_prices,
)
from parity import is_quick  # noqa: E402
from providers import judge_provider  # noqa: E402

from sidecoach.coach.coach import _names  # noqa: E402
from sidecoach.config import load_settings  # noqa: E402

RUBRIC = Path(__file__).with_name("rubric.toml")
EVAL = ROOT / "data" / "eval"
REPORTS = EVAL / "rubric"
VOTES = 3  # at most; the asking stops once two agree
# When a criterion is asked at all (decided in code): the reply ends with a question; the
# coach's request listed earlier moments of the same pattern to tie to.
ONLY_IF = ("ends_with_question", "has_links", "effect_given", "rule_asked")
# The words every coach request carries when it tells the model to leave its rule of thumb
# out, the rule cooling down (sidecoach.coach.coach.RULE_HELD_MARK, repeated here so the
# rubric reads no coach module).
RULE_HELD_MARK = "no rule of thumb this time"
_ELICIT = re.compile(
    r"\bwhat (?:were|was) you (?:expecting|hoping|trying)|\bwhat did you expect", re.IGNORECASE
)

# The section of the coach's request that lists earlier moments of the same pattern, one a
# line: "- At Bg4 (their move): ...".
EARLIER = "# Earlier in this game"
_AT = re.compile(r"^-\s+At\s+([^\s(,:;]+)")


def request_of(item: dict) -> str:
    """The request the coach was given for an item: its last message."""
    return str(item["messages"][-1]["content"]) if item.get("messages") else ""


def earlier_moments(request: str) -> list[str]:
    """The moves of the moments the request's "# Earlier in this game" lists (SAN, marks
    kept as written); empty when it has no such section."""
    if EARLIER not in request:
        return []
    section = request.split(EARLIER, 1)[1]
    moves: list[str] = []
    for line in section.splitlines()[1:]:
        if line.startswith("#"):
            break
        found = _AT.match(line.strip())
        if found:
            moves.append(found.group(1).rstrip(".,"))
    return moves


# A pawn's step reads as a square: named as a move only where no word before it makes it
# one ("your pawn on e4", "to e4", "the e4-pawn") -- but "as at e5", "like after e5" are the
# link (the pre-spend audit, 2026-09-25).
_AS_SQUARE = re.compile(
    r"(?:\b(?:on|to|from|at|onto|into|of|the|pawn|knight|bishop|rook|queen|king)\s+)$", re.I
)
_AS_LINK = re.compile(r"\b(?:as|like|same as)\s+(?:at|with|after|on|in)\s+$", re.I)
# Castling said in words: "when you castled".
_CASTLED = re.compile(r"\bcastl(?:e|ed|es|ing)\b", re.I)


def _names_move(reply: str, move: str) -> bool:
    """Whether `reply` names `move` as a move: a piece move, capture or castling as its SAN
    token; a pawn's step only where it stands as a move, not as a square."""
    core = move.rstrip("+#")
    if core.startswith("O-O") and _CASTLED.search(reply):
        return True
    if not re.fullmatch(r"[a-h][1-8](?:=[QRBN])?", core):
        return _names(reply, core)
    for found in re.finditer(rf"(?<![A-Za-z0-9]){re.escape(core)}(?![A-Za-z0-9-])", reply):
        before = reply[: found.start()]
        if _AS_LINK.search(before) or not _AS_SQUARE.search(before):
            return True
    return False


# What the move a question is about does, as the request gives it: the threat's effect in
# brackets on the sheet ("h5 (attacks the knight on g4)"), the last move's "What Ra8 does:".
_THREAT_EFFECT = re.compile(
    r"(?:what the other side threatens|threats, if the (?:player|opponent) passed)[^\n]*:\s*"
    r"(?:the (?:player|opponent)'s )?\S+ \("
)
_MOVE_EFFECT = re.compile(r"^What \S+ does: ", re.M)


def effect_given(cls: str, request: str) -> bool:
    """Whether the request says what the move the answer is about does: for a threat, its
    effect on the sheet; for the opponent's last move, its "What ... does" line. Any other
    class: yes."""
    if cls == "ask_threat":
        return bool(_THREAT_EFFECT.search(request))
    if cls == "ask_opponent_move":
        now = request.split("# Position now", 1)[-1]
        return bool(_MOVE_EFFECT.search(now.split("## Standing", 1)[0]))
    return True


def names_a_link(reply: str, request: str) -> bool:
    """Connection, decided in code: the reply names the move of at least one earlier moment
    the request listed, as a move (`_names_move`)."""
    return any(_names_move(reply, move) for move in earlier_moments(request))


# The criteria decided in code, by the name a rubric file gives them (`code = "..."`): the
# judge is never called for them.
CODE_RULES = {"names_a_link": names_a_link}

RUBRIC_SYSTEM = (
    "You judge one reply a chess coach gave a player at one moment of their game, against "
    "the record of that moment. Answer the question with one word, yes or no, and nothing "
    "else. " + COACH_WORDS_RULE
)


@dataclass(frozen=True)
class Criterion:
    key: str
    question: str
    applies_to: tuple[str, ...]
    only_if: str | None = None
    # The code rule that decides it (CODE_RULES), or None: the judge is asked.
    code: str | None = None

    @property
    def qhash(self) -> str:
        """The question's wording (and its code rule), short: an answer is kept for this
        wording only."""
        return hashlib.sha1((self.question + "|" + (self.code or "")).encode()).hexdigest()[:10]

    def applies(self, cls: str, reply: str, request: str = "") -> bool:
        """Whether the criterion is asked of this reply (decided in code); `request` is the
        coach's request, for `has_links`."""
        if "*" not in self.applies_to and cls not in self.applies_to:
            return False
        if self.only_if == "ends_with_question":
            # "What were you expecting that move to do?" asks what the player thought, for the
            # profile: the coach listening, not a question that teaches.
            last = re.split(r"(?<=[.!])\s+", reply.strip())[-1]
            return reply.rstrip().endswith("?") and not _ELICIT.search(last)
        if self.only_if == "has_links":
            return bool(earlier_moments(request))
        if self.only_if == "effect_given":
            return effect_given(cls, request)
        if self.only_if == "rule_asked":
            # Told to leave the rule out while it cools down (2026-09-27): no rule is no fault.
            return RULE_HELD_MARK not in request.lower()
        return True

    def decide(self, reply: str, request: str) -> bool:
        """The answer of a criterion decided in code."""
        assert self.code is not None
        return CODE_RULES[self.code](reply, request)


def load_rubric(path: Path = RUBRIC, rating: tuple[int, int] | None = None) -> list[Criterion]:
    """The criteria of a rubric file, the rating band filled in ({low}, {high})."""
    low, high = rating or _rating()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    out: list[Criterion] = []
    for raw in data.get("criterion") or []:
        key = str(raw["key"])
        only_if = raw.get("only_if")
        if only_if is not None and only_if not in ONLY_IF:
            raise ValueError(f"{path.name}: {key}: only_if must be one of {ONLY_IF}")
        code = raw.get("code")
        if code is not None and code not in CODE_RULES:
            raise ValueError(f"{path.name}: {key}: code must be one of {sorted(CODE_RULES)}")
        applies = tuple(str(x) for x in raw.get("applies_to") or ["*"])
        question = str(raw["question"]).replace("{low}", str(low)).replace("{high}", str(high))
        out.append(Criterion(key, question, applies, only_if, code))
    keys = [c.key for c in out]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path.name}: a key is used twice: {keys}")
    if not out:
        raise ValueError(f"{path.name}: no [[criterion]]")
    return out


def _rating() -> tuple[int, int]:
    profile = load_settings().profile
    return profile.rating_low, profile.rating_high


# ---- what a run holds -------------------------------------------------------------------

# "<name>-<version>-r<round>", the version a short hash with, for uncommitted code, a digest.
_RUN = re.compile(r"^(?P<name>.+?)-(?P<version>[0-9a-f]{7,}(?:-d[0-9a-f]{7,})?)-r(?P<round>\d+)$")


def items_dir(run: str) -> Path:
    """Where icr_play.py kept a run's items: data/eval/<name>/items-<version>-r<round>."""
    found = _RUN.match(run)
    if found is None:
        raise SystemExit(f"{run}: not a run name of icr_play.py; give --items")
    return EVAL / found["name"] / f"items-{found['version']}-r{found['round']}"


def load_items(folder: Path) -> dict[str, dict]:
    """A finished run's items, as they are: its replies are judged as that run's, whatever
    the code now (the report names the run)."""
    out: dict[str, dict] = {}
    for path in sorted(folder.glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            item = json.loads(line)
            out[item["id"]] = item
    if not out:
        raise SystemExit(f"no items in {folder}")
    return out


def judged_ids(run: str) -> set[str]:
    folder = judge_module.JUDGMENTS / run
    return {
        json.loads(line)["id"]
        for path in folder.glob("*.jsonl")
        for line in path.open(encoding="utf-8")
        if line.strip()
    }


@dataclass
class Task:
    arm: str  # "ai" | "floor"
    item: dict
    reply: str

    @property
    def id(self) -> str:
        return self.item["id"]


def tasks_of(
    run: str, items: dict[str, dict], *, floor: bool, unjudged: bool = False
) -> list[Task]:
    """The AI arm: the run's judged moments where the model's own reply was spoken -- with
    `unjudged`, every moment of its items where it was (a run the parity judge has not read:
    the locked test set, scored on the rubric alone within Tai's $3 cap, 2026-09-26). The
    floor arm: the templates of the run's paired moments (its floor judgments), or, when
    the run judged no floor, of its AI moments that have a template of their own."""
    ai_ids = set(items) if unjudged else judged_ids(run)
    out = [
        Task("ai", items[i], items[i]["reply"]["reply"])
        for i in sorted(ai_ids)
        if i in items and items[i].get("reply") and reply_status(items[i]["reply"]) == "model"
    ]
    if floor:
        floor_ids = judged_ids(f"{run}-floor") or {t.id for t in out if not is_quick(t.item)}
        out += [
            Task("floor", items[i], items[i]["fallback"])
            for i in sorted(floor_ids)
            if i in items and (items[i].get("fallback") or "").strip()
        ]
    return out


# ---- asking -------------------------------------------------------------------------------


def record_of(item: dict, reply: str) -> str:
    """What the judge reads before each question: the record, then the reply."""
    return (
        RUBRIC_SYSTEM
        + "\n\nThe record:\n"
        + evidence_block(item, with_reference=False)
        + '\n\nThe coach\'s reply:\n"""\n'
        + reply
        + '\n"""'
    )


def answer_of(text: str) -> bool | None:
    word = re.sub(r"[^a-z]", " ", text.lower()).split()
    return {"yes": True, "no": False}.get(word[0]) if word else None


def ask(binary, ledger: Ledger, item_id: str, system: str, question: str) -> bool:
    """One criterion: up to three votes, stopping once two agree. An answer that is not yes
    or no raises JudgeFailed: the criterion is not scored."""
    votes: list[bool] = []
    while len(votes) < VOTES and votes.count(True) < 2 and votes.count(False) < 2:
        completion = _chat_with_backoff(
            binary,
            system,
            [{"role": "user", "content": "Question: " + question + "\nAnswer yes or no."}],
            ledger,
            item_id,
            max_tokens=5,
            temperature=0.0,
            thinking=False,
        )
        vote = answer_of(completion.text or "")
        if vote is None:
            raise JudgeFailed(f"{item_id}: not yes or no: {completion.text!r}")
        votes.append(vote)
    return votes.count(True) >= 2


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def store(run: str, arm: str) -> Path:
    return judge_module.JUDGMENTS / (f"{run}-rubric" + ("-floor" if arm == "floor" else ""))


def load_answers(run: str, arm: str, *, reuse_stale: bool) -> dict[str, dict[str, dict]]:
    """The answers kept for an arm: {item id: {criterion key: {"q", "yes", "reply"}}}."""
    version = stamp.code_version()
    out: dict[str, dict[str, dict]] = {}
    for path in store(run, arm).glob("*.jsonl"):
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            stamp.check(
                rec.get("commit"),
                f"the rubric answers on {rec['id']}",
                reuse_stale=reuse_stale,
                current=version,
            )
            out.setdefault(rec["id"], {}).update(rec["answers"])
    return out


def todo_of(task: Task, criteria: list[Criterion], kept: dict[str, dict]) -> list[Criterion]:
    """The criteria still to ask of a task: those that apply and have no answer for this
    wording of the question and this reply."""
    reply = _sha(task.reply)
    request = request_of(task.item)
    return [
        c
        for c in criteria
        if c.applies(task.item["class"], task.reply, request)
        and not (kept.get(c.key, {}).get("q") == c.qhash and kept[c.key].get("reply") == reply)
    ]


def run_rubric(
    run: str,
    tasks: list[Task],
    criteria: list[Criterion],
    judge,
    ledger: Ledger,
    *,
    workers: int = 8,
    reuse_stale: bool = False,
) -> tuple[dict[str, dict[str, dict[str, dict]]], Counter]:
    """Ask every criterion not yet answered; the answers of both arms and the failures."""
    version = stamp.code_version()
    answers = {arm: load_answers(run, arm, reuse_stale=reuse_stale) for arm in ("ai", "floor")}
    binary = binary_judge(judge)
    lock = threading.Lock()
    failed: Counter = Counter()

    def one(task: Task) -> dict[str, dict]:
        kept = answers[task.arm].get(task.id, {})
        system = record_of(task.item, task.reply)
        new: dict[str, dict] = {}
        for c in todo_of(task, criteria, kept):
            try:
                yes = (
                    c.decide(task.reply, request_of(task.item))
                    if c.code
                    else ask(binary, ledger, task.id, system, c.question)
                )
            except JudgeFailed as exc:
                with lock:
                    failed[c.key] += 1
                print(f"  {task.arm} {c.key} failed: {exc}", flush=True)
                continue
            new[c.key] = {"q": c.qhash, "yes": yes, "reply": _sha(task.reply)}
        if new:
            rec = {
                "id": task.id,
                "class": task.item["class"],
                "arm": task.arm,
                "answers": new,
                "commit": version,
            }
            path = store(run, task.arm) / f"{task.item['game'].replace('/', '__')}.jsonl"
            with lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                answers[task.arm].setdefault(task.id, {}).update(new)
        return new

    todo = [t for t in tasks if todo_of(t, criteria, answers[t.arm].get(t.id, {}))]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, t) for t in todo]
        try:
            for fut in as_completed(futures):
                fut.result()
        except BudgetExceeded:
            for fut in futures:
                fut.cancel()
            raise
    return answers, failed


# ---- the report ---------------------------------------------------------------------------


def scores_of(
    tasks: list[Task], criteria: list[Criterion], answers: dict[str, dict[str, dict[str, dict]]]
) -> list[dict]:
    """Per reply: the criteria it was asked (with this wording), what it passed and failed,
    and its excellence -- the mean yes-rate over them (None when none was answered)."""
    by_key = {c.key: c for c in criteria}
    out = []
    for t in tasks:
        kept = answers.get(t.arm, {}).get(t.id, {})
        reply = _sha(t.reply)
        asked = {
            k: v["yes"]
            for k, v in kept.items()
            if k in by_key
            and v.get("q") == by_key[k].qhash
            and v.get("reply") == reply
            and by_key[k].applies(t.item["class"], t.reply, request_of(t.item))
        }
        out.append(
            {
                "id": t.id,
                "arm": t.arm,
                "class": t.item["class"],
                "reply": t.reply,
                "answers": asked,
                "failed": sorted(k for k, yes in asked.items() if not yes),
                "excellence": (sum(asked.values()) / len(asked)) if asked else None,
            }
        )
    return out


def _rate(xs: list[bool]) -> str:
    return f"{sum(xs) / len(xs):.2f} ({len(xs)})" if xs else "-"


def _mean(xs: list[float]) -> str:
    return f"{sum(xs) / len(xs):.2f}" if xs else "-"


def report(run: str, rows: list[dict], criteria: list[Criterion], failed: Counter) -> str:
    lines = [f"# Excellence rubric: {run}", ""]
    arms = ("ai", "floor")
    names = {"ai": "AI", "floor": "template"}
    ex = {
        a: [r["excellence"] for r in rows if r["arm"] == a and r["excellence"] is not None]
        for a in arms
    }
    lines.append(
        "Excellence (per reply the mean yes-rate over the criteria that apply, then the mean"
        " over replies): "
        + "; ".join(f"{names[a]} {_mean(ex[a])} over {len(ex[a])}" for a in arms if ex[a])
        + "."
    )
    ai = {r["id"]: r for r in rows if r["arm"] == "ai" and r["excellence"] is not None}
    fl = {r["id"]: r for r in rows if r["arm"] == "floor" and r["excellence"] is not None}
    both = sorted(set(ai) & set(fl))
    if both:
        better = sum(ai[i]["excellence"] > fl[i]["excellence"] for i in both)
        worse = sum(ai[i]["excellence"] < fl[i]["excellence"] for i in both)
        ai_mean = _mean([ai[i]["excellence"] for i in both])
        fl_mean = _mean([fl[i]["excellence"] for i in both])
        lines.append(
            f"Paired ({len(both)} moments judged in both arms): AI {ai_mean}, template"
            f" {fl_mean}; the AI higher in {better}, lower in {worse}, the same in"
            f" {len(both) - better - worse}."
        )
    if failed:
        lines.append(
            "Not scored (the call failed or the answer was not yes or no): "
            + ", ".join(f"{k} {v}" for k, v in sorted(failed.items()))
            + "."
        )
    lines += ["", "| criterion | AI yes-rate (n) | template yes-rate (n) |", "|---|---|---|"]
    for c in criteria:
        cells = [
            _rate([r["answers"][c.key] for r in rows if r["arm"] == a and c.key in r["answers"]])
            for a in arms
        ]
        lines.append(f"| {c.key} | {cells[0]} | {cells[1]} |")
    lines += [
        "",
        "| class | AI n | AI excellence | template n | template excellence |",
        "|---|---|---|---|---|",
    ]
    classes = sorted({r["class"] for r in rows})
    for cls in classes:
        cells = []
        for a in arms:
            xs = [
                r["excellence"]
                for r in rows
                if r["arm"] == a and r["class"] == cls and r["excellence"] is not None
            ]
            cells += [str(len(xs)), _mean(xs)]
        lines.append(f"| {cls} | " + " | ".join(cells) + " |")
    lowest = sorted(ai.values(), key=lambda r: (r["excellence"], r["id"]))[:10]
    lowest = [r for r in lowest if r["failed"]]
    if lowest:
        lines += ["", "## The lowest-scoring AI replies", ""]
        for r in lowest:
            failed_keys = ", ".join(r["failed"])
            lines += [
                f"- {r['id']} ({r['class']}): {r['excellence']:.2f}, failed {failed_keys}",
                f"  - AI: {r['reply']}",
            ]
            if r["id"] in fl:
                lines.append(f"  - template: {fl[r['id']]['reply']}")
    return "\n".join(lines) + "\n"


def write_report(run: str, rows: list[dict], criteria: list[Criterion], failed: Counter) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    md = REPORTS / f"{run}.md"
    md.write_text(report(run, rows, criteria, failed), encoding="utf-8")
    (REPORTS / f"{run}.json").write_text(
        json.dumps(
            {
                "run": run,
                "criteria": [c.__dict__ for c in criteria],
                "rows": rows,
                "failed": dict(failed),
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return md


def estimate_of(
    tasks: list[Task],
    criteria: list[Criterion],
    answers: dict[str, dict[str, dict[str, dict]]],
    prices: tuple[float, float, float],
) -> tuple[estimate.Estimate, int]:
    """What the questions not yet answered would cost, and how many there are."""
    work: list[tuple[str, int]] = []
    for t in tasks:
        # The criteria decided in code cost nothing.
        n = len(
            [c for c in todo_of(t, criteria, answers.get(t.arm, {}).get(t.id, {})) if not c.code]
        )
        if n:
            work.append((record_of(t.item, t.reply), n))
    return estimate.repeated_record_estimate(work, prices), sum(n for _, n in work)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--run", required=True, help="an icr_play.py run, e.g. effort-medium-47138dd-r1"
    )
    ap.add_argument("--items", type=Path, help="the run's items directory, when not the usual")
    ap.add_argument("--floor", action="store_true", help="judge the templates of its moments too")
    ap.add_argument("--rubric", type=Path, default=RUBRIC)
    ap.add_argument("--judge", default="openai:gpt-5.6-luna@medium")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--dry-run", action="store_true", help="estimate the cost; call nothing")
    ap.add_argument("--reuse-stale", action="store_true")
    ap.add_argument(
        "--unjudged",
        action="store_true",
        help="every moment of the items the model answered, not only the parity-judged ones",
    )
    args = ap.parse_args()

    criteria = load_rubric(args.rubric)
    items = load_items(args.items or items_dir(args.run))
    tasks = tasks_of(args.run, items, floor=args.floor, unjudged=args.unjudged)
    if not tasks:
        raise SystemExit(f"{args.run}: no judged moments with the model's own reply")
    arms = Counter(t.arm for t in tasks)
    print(
        f"{args.run}: {arms.get('ai', 0)} AI replies"
        + (f", {arms.get('floor', 0)} templates" if args.floor else "")
        + f"; {len(criteria)} criteria from {args.rubric.name}",
        flush=True,
    )
    if args.dry_run:
        kept = {a: load_answers(args.run, a, reuse_stale=True) for a in ("ai", "floor")}
        cost, questions = estimate_of(tasks, criteria, kept, settings_prices())
        print(
            f"dry run: {questions} questions not yet answered, up to {VOTES} votes each"
            f" (~{estimate.EXPECTED_VOTES} expected): {cost.line()}"
            f" (budget ${args.budget_usd:.2f})"
        )
        return
    judge, _ = judge_provider(args.judge)
    ledger = Ledger(
        f"rubric-{args.run}",
        "rubric",
        judge.model,
        budget_usd=args.budget_usd,
        resume=True,
        version=stamp.code_version(),
    )
    if ledger.before:
        print(f"this rubric run spent ${ledger.before:.4f} before; it counts against the budget")
    try:
        answers, failed = run_rubric(
            args.run,
            tasks,
            criteria,
            judge,
            ledger,
            workers=args.workers,
            reuse_stale=args.reuse_stale,
        )
    except BudgetExceeded as exc:
        print(f"stopped: {exc}", flush=True)
        raise SystemExit(2) from None
    rows = scores_of(tasks, criteria, answers)
    path = write_report(args.run, rows, criteria, failed)
    print(report(args.run, rows, criteria, failed))
    print(f"wrote {path}; {ledger.summary()}", flush=True)


if __name__ == "__main__":
    main()
