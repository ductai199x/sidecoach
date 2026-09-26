"""The judge against hand labels -- and a run's replies judged again by the judge as it is now.

A gold file is written by hand, one line per reply read:

  data/eval/gold/<name>.jsonl
    {"id": "<item id>", "arm": "ai" | "floor",
     "false_claims": ["<a claim of the reply that is false or unsupported>", ...],
     "facts": {"better_move": true, "idea": false, ...},
     "rubric": {"why": true, "transfer": false, "no_hedging": null, ...}}

`facts` says, for each required fact of the item (its `required` keys), whether the reply
states it: the judge's completeness is a vote too, and checked like the rest.

`false_claims` empty means the reply states nothing false; a rubric value of null is not
labelled. The comparison reads a run's judgments (<run>, <run>-floor) and its rubric answers
(<run>-rubric, <run>-rubric-floor) and prints, per reply and per criterion, agreement, a
judge false fail (the judge faulted what the label passes) or a judge false pass (the judge
passed what the label faults), each disagreement with the reply and the claim or criterion.

  uv run python tools/eval/gold.py --gold round3 --run quality-r3-97ebbba-r1

--rejudge judges a run's existing replies again with the judge's code as it is now, under a
new run name (no coach calls): the AI arm's moments, the templates of its paired moments
(unless --no-floor) and the rubric (unless --no-rubric). Every call is on the ledger,
checked against --budget-usd; --dry-run estimates it.

  uv run python tools/eval/gold.py --rejudge --run quality-r3-97ebbba-r1 --budget-usd 2 \
      [--as quality-r3-97ebbba-r1-v2] [--gold round3] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import estimate  # noqa: E402
import parity  # noqa: E402
import rubric  # noqa: E402
import stamp  # noqa: E402
from judge import (  # noqa: E402
    BINARY_SYSTEM,
    BudgetExceeded,
    Ledger,
    evidence_block,
    judge_all,
    settings_prices,
)
from providers import judge_provider  # noqa: E402
from report import load  # noqa: E402

GOLD = ROOT / "data" / "eval" / "gold"
ARMS = ("ai", "floor")


def load_gold(path: Path) -> list[dict]:
    """The labelled replies of a gold file."""
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("arm", "ai") not in ARMS:
            raise ValueError(f"{path.name}:{n}: arm must be ai or floor")
        row.setdefault("arm", "ai")
        row.setdefault("false_claims", [])
        row.setdefault("facts", {})
        row.setdefault("rubric", {})
        rows.append(row)
    return rows


_WORD = re.compile(r"[a-z0-9+#=-]{3,}")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def same_claim(label: str, judged: str) -> bool:
    """Whether a claim the judge faulted is the one a label names: half the label's words
    (moves, squares and words of three letters or more) are in it."""
    want = _words(label)
    return bool(want) and len(want & _words(judged)) / len(want) >= 0.5


@dataclass
class Comparison:
    claims: Counter = field(default_factory=Counter)
    facts: Counter = field(default_factory=Counter)
    rubric: dict[str, Counter] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)


def compare(gold: list[dict], run: str) -> Comparison:
    """The run's verdicts against the labels."""
    judged = {
        arm: {j["id"]: j for j in load(f"{run}-floor" if arm == "floor" else run, None)}
        for arm in ARMS
    }
    answers = {arm: rubric.load_answers(run, arm, reuse_stale=True) for arm in ARMS}
    out = Comparison()
    for row in gold:
        arm, rid = row["arm"], row["id"]
        j = judged[arm].get(rid)
        if j is None:
            out.claims["not judged"] += 1
            out.lines.append(f"- {rid} ({arm}): not judged under {run}")
            continue
        faulted = [u.split(" -- ", 1)[0] for u in j["scores"].get("unsupported") or []]
        labelled = list(row["false_claims"])
        head = f"- {rid} ({arm}, {j['class']}): {j['reply']}"
        if faulted and not labelled:
            out.claims["judge false fail"] += 1
            out.lines += [head, *[f"  - judge faulted, label passes: {c}" for c in faulted]]
        elif labelled and not faulted:
            out.claims["judge false pass"] += 1
            out.lines += [head, *[f"  - label faults, judge passed: {c}" for c in labelled]]
        else:
            out.claims["agree"] += 1
            missed = [c for c in labelled if not any(same_claim(c, f) for f in faulted)]
            extra = [f for f in faulted if not any(same_claim(c, f) for c in labelled)]
            if missed or extra:
                out.claims["agree, other claims"] += 1
                out.lines += [head]
                out.lines += [f"  - label faults, judge passed: {c}" for c in missed]
                out.lines += [f"  - judge faulted, label passes: {c}" for c in extra]
        stated = {
            x["key"]: x.get("mentioned")
            for x in (j.get("verdict") or {}).get("required", [])
            if isinstance(x, dict) and "key" in x
        }
        for key, want in (row.get("facts") or {}).items():
            if want is None:
                continue
            got = stated.get(key)
            if got is None:
                out.facts["not judged"] += 1
            elif bool(got) == want:
                out.facts["agree"] += 1
            else:
                kind = "judge false fail" if want else "judge false pass"
                out.facts[kind] += 1
                out.lines.append(f"- {rid} ({arm}) fact {key}: {kind}: {j['reply']}")
        kept = answers[arm].get(rid, {})
        for key, want in (row.get("rubric") or {}).items():
            if want is None:
                continue
            counts = out.rubric.setdefault(key, Counter())
            got = kept.get(key, {}).get("yes")
            if got is None:
                counts["not judged"] += 1
            elif got == want:
                counts["agree"] += 1
            else:
                kind = "judge false fail" if want else "judge false pass"
                counts[kind] += 1
                said = "yes" if got else "no"
                out.lines += [
                    f"- {rid} ({arm}) {key}: {kind} -- the judge said {said}: {j['reply']}"
                ]
    return out


def summary(comp: Comparison, run: str, gold: str) -> str:
    lines = [f"# The judge against {gold}: {run}", ""]
    c = comp.claims
    lines.append(
        "Claims (per reply): "
        + ", ".join(f"{k} {c.get(k, 0)}" for k in ("agree", "judge false fail", "judge false pass"))
        + (f"; not judged {c['not judged']}" if c.get("not judged") else "")
        + (
            f"; agreeing on the verdict but not the claim {c['agree, other claims']}"
            if c.get("agree, other claims")
            else ""
        )
        + "."
    )
    if comp.facts:
        f = comp.facts
        lines.append(
            "Required facts: "
            + ", ".join(
                f"{k} {f.get(k, 0)}"
                for k in ("agree", "judge false fail", "judge false pass", "not judged")
            )
            + "."
        )
    if comp.rubric:
        lines += [
            "",
            "| criterion | agree | judge false fail | judge false pass | not judged |",
            "|---|---|---|---|---|",
        ]
        for key, k in sorted(comp.rubric.items()):
            lines.append(
                f"| {key} | {k.get('agree', 0)} | {k.get('judge false fail', 0)} |"
                f" {k.get('judge false pass', 0)} | {k.get('not judged', 0)} |"
            )
    if comp.lines:
        lines += ["", "## Where they differ", "", *comp.lines]
    return "\n".join(lines) + "\n"


# ---- the packet a labeller reads ------------------------------------------------------------


def packets(items: dict[str, dict], n_ai: int, n_floor: int, seed: int = 2509) -> list[dict]:
    """The replies to label, for a blind labeller: `n_ai` of the model's replies, spread
    evenly over the benchmark's levels and families (by class when an item has no bench
    tag), and the templates of `n_floor` of the same moments. Each with what the labeller
    needs and nothing of the judge's: the whole request, the required facts, and the rubric
    questions that apply to that reply. Two thirds are marked "tune" (the judge is fixed
    against them) and one third "confirm" (held out: the judge's accuracy is measured there).
    """
    import random

    criteria = rubric.load_rubric()
    rng = random.Random(seed)
    groups: dict[tuple, list[dict]] = {}
    for it in sorted(items.values(), key=lambda x: x["id"]):
        reply = (it.get("reply") or {}).get("reply")
        if not reply or (it.get("reply") or {}).get("fallback_used"):
            continue
        tag = it.get("bench") or {}
        key = (tag.get("level"), tag.get("family")) if tag else (None, it["class"])
        groups.setdefault(key, []).append(it)
    for members in groups.values():
        rng.shuffle(members)
    chosen: list[dict] = []
    order = sorted(groups, key=str)
    while len(chosen) < n_ai and any(groups[k] for k in order):
        for k in order:
            if groups[k] and len(chosen) < n_ai:
                chosen.append(groups[k].pop())
    with_template = [it for it in chosen if (it.get("fallback") or "").strip()]
    floor = rng.sample(with_template, min(n_floor, len(with_template)))
    out: list[dict] = []
    for arm, picked in (("ai", chosen), ("floor", floor)):
        for n, it in enumerate(picked):
            reply = it["reply"]["reply"] if arm == "ai" else it["fallback"]
            request = rubric.request_of(it)
            out.append(
                {
                    "id": it["id"],
                    "arm": arm,
                    "part": "confirm" if n % 3 == 2 else "tune",
                    "class": it["class"],
                    "level": (it.get("bench") or {}).get("level"),
                    "question": it.get("question"),
                    "reply": reply,
                    "required": it.get("required") or [],
                    "criteria": [
                        {"key": c.key, "question": c.question}
                        for c in criteria
                        if c.code is None and c.applies(it["class"], reply, request)
                    ],
                    "fen": it.get("fen"),
                    "request": request,
                }
            )
    return out


# ---- judging a run again --------------------------------------------------------------------


@dataclass
class Plan:
    items: dict[str, dict]
    ai: list[dict]
    floor: list[str]
    tasks: list[rubric.Task]


def plan(run: str, items_dir: Path | None, *, floor: bool, with_rubric: bool) -> Plan:
    """What a re-judging of `run` covers: the replies its AI arm judged, the templates its
    floor arm judged, and the rubric's tasks on the same moments."""
    items = rubric.load_items(items_dir or rubric.items_dir(run))
    ai_ids = rubric.judged_ids(run)
    replies = [items[i]["reply"] for i in sorted(ai_ids) if i in items and items[i].get("reply")]
    floor_ids = sorted(i for i in rubric.judged_ids(f"{run}-floor") if i in items) if floor else []
    tasks = rubric.tasks_of(run, items, floor=floor) if with_rubric else []
    return Plan(items, replies, floor_ids, tasks)


def estimate_of(p: Plan, criteria: list[rubric.Criterion]) -> dict[str, estimate.Estimate]:
    prices = settings_prices()

    def record(it: dict) -> str:
        return BINARY_SYSTEM + evidence_block(it, with_reference=False) + (it.get("fallback") or "")

    ai_items = [p.items[r["id"]] for r in p.ai]
    out = {"AI arm": estimate.judge_estimate(ai_items, prices, record_of=record)}
    if p.floor:
        out["floor arm"] = estimate.judge_estimate(
            [p.items[i] for i in p.floor], prices, record_of=record
        )
    if p.tasks:
        out["rubric"] = rubric.estimate_of(p.tasks, criteria, {}, prices)[0]
    return out


def rejudge(
    run: str,
    new: str,
    p: Plan,
    criteria: list[rubric.Criterion],
    judge,
    ledger: Ledger,
    *,
    workers: int,
) -> None:
    """The run's replies judged again under `new`: the AI arm, the floor, the rubric."""
    judge_all(p.items, p.ai, new, judge, ledger, thinking=False, quiet=True, workers=workers)
    if p.floor:
        parity.judge_floor(p.items, p.floor, new, judge, ledger, thinking=False, workers=workers)
    if p.tasks:
        tasks = rubric.tasks_of(new, p.items, floor=bool(p.floor))
        answers, failed = rubric.run_rubric(new, tasks, criteria, judge, ledger, workers=workers)
        rows = rubric.scores_of(tasks, criteria, answers)
        path = rubric.write_report(new, rows, criteria, failed)
        print(f"rubric report: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", required=True, help="the run to compare, or to judge again")
    ap.add_argument("--gold", help="the gold file's name under data/eval/gold/ (or a path)")
    ap.add_argument("--rejudge", action="store_true", help="judge the run's replies again first")
    ap.add_argument("--as", dest="new", help="the re-judged run's name")
    ap.add_argument("--items", type=Path, help="the run's items directory, when not the usual")
    ap.add_argument("--no-floor", dest="floor", action="store_false")
    ap.add_argument("--no-rubric", dest="with_rubric", action="store_false")
    ap.add_argument("--judge", default="openai:gpt-5.6-luna@medium")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget-usd", type=float)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--packet",
        type=Path,
        help="write the replies to label (blind: nothing of the judge's) to this JSONL",
    )
    ap.add_argument("--n-ai", type=int, default=60)
    ap.add_argument("--n-floor", type=int, default=20)
    ap.add_argument("--part", choices=["tune", "confirm"], help="compare on these labels only")
    args = ap.parse_args()

    run = args.run
    if args.packet:
        items = rubric.load_items(args.items or rubric.items_dir(run))
        rows = packets(items, args.n_ai, args.n_floor)
        args.packet.parent.mkdir(parents=True, exist_ok=True)
        with args.packet.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        parts = Counter((r["arm"], r["part"]) for r in rows)
        print(f"wrote {len(rows)} replies to label to {args.packet}: {dict(parts)}")
        return
    if args.rejudge:
        if args.budget_usd is None:
            raise SystemExit("--rejudge calls the judge: give --budget-usd")
        new = args.new or f"{run}-rejudge-{stamp.code_version()}"
        criteria = rubric.load_rubric()
        p = plan(run, args.items, floor=args.floor, with_rubric=args.with_rubric)
        print(
            f"{run} -> {new}: {len(p.ai)} AI replies, {len(p.floor)} templates,"
            f" {len(p.tasks)} rubric tasks",
            flush=True,
        )
        if args.dry_run:
            parts = estimate_of(p, criteria)
            for name, est in parts.items():
                print(f"  {name}: {est.line()}")
            total = sum(e.usd for e in parts.values())
            print(f"  in all: ~${total:.2f} (budget ${args.budget_usd:.2f}); see estimate.py")
            return
        judge, _ = judge_provider(args.judge)
        ledger = Ledger(
            f"gold-{new}",
            "gold",
            judge.model,
            budget_usd=args.budget_usd,
            resume=True,
            version=stamp.code_version(),
        )
        try:
            rejudge(run, new, p, criteria, judge, ledger, workers=args.workers)
        except BudgetExceeded as exc:
            raise SystemExit(f"stopped: {exc}") from None
        print(ledger.summary())
        run = new
    if args.gold:
        path = Path(args.gold)
        if not path.exists():
            path = GOLD / f"{args.gold}.jsonl"
        gold = load_gold(path)
        if args.part:
            gold = [g for g in gold if g.get("part", "tune") == args.part]
        print(summary(compare(gold, run), run, f"{path.stem} ({args.part or 'all'})"))


if __name__ == "__main__":
    main()
