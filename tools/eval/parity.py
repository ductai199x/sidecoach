"""The floor test: is the AI coach at least as good as the free template coach, moment by
moment? The template is the floor (Tai, 2026-09-25: the AI coach says at least what the
free coach says).

The same items are judged twice by the same judge: the AI arm (each item's reply, under
the run name) and the floor arm (each item's template, `fallback`, under `<run>-floor`, no
coach calls, the model's form not asked of it). Then, per item:

  both            both pass
  ai_only         the AI reply passes, the template does not
  template_only   the template passes, the AI reply does not -- the moments to read
  neither         neither passes

with each arm's completeness (the share of the required facts it names) side by side, and
whether the AI reply names fewer of them than the template, more, or the same; and the
floor's coverage, decided in code: whether the AI reply names what the coach required of it
(`must_name`, the coach's own `omitted`), and the share of the moves and squares the
template names that the reply names too.

Not paired, and counted: a reply the checks refused, where the template spoke -- a win for
the template, in the headline; a reply the model never wrote (its call failed, or no call
was made); and a quick follow-up, whose template was already spoken as the answer, so there
is no template to set beside it.

The moments are drawn as icr_play.py draws a round's (`sample_moments`, the seed
`round_seed(seed, round)`), and the report says which seed. The latency of the coach's
calls, by trigger, is in the report too (the whole reply: without streaming the first
sentence is not measured).

  uv run python tools/eval/parity.py --items data/eval/<name>/items-<version>-r1 \
      --run <name>-<version>-r1 --budget-usd 2 [--dry-run]

Every judge call is on the ledger, checked against --budget-usd before it is made; the
budget is the run's, across restarts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import estimate  # noqa: E402
import judge as judge_module  # noqa: E402
import stamp  # noqa: E402
from judge import (  # noqa: E402
    BINARY_SYSTEM,
    COACH_WORDS_RULE,
    BudgetExceeded,
    JudgeFailed,
    Ledger,
    _chat_with_backoff,
    binary_judge,
    evidence_block,
    judge_all,
    mark_coach_words,
    reply_status,
    settings_prices,
)
from providers import judge_provider  # noqa: E402

from sidecoach.coach.coach import _names, omitted  # noqa: E402
from sidecoach.config import load_settings  # noqa: E402

PARITY = ROOT / "data" / "eval" / "parity"

_NAMED = re.compile(r"O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?")

# Why a drawn moment is not paired.
QUICK = "quick_followup"
LEFT_OUT_WORDS = {
    "rejected": "the checks refused the model and the template spoke (a template win)",
    "model_failed": "the model's call failed or came back empty",
    "template_only": "no call was made (a fast game)",
    QUICK: "a quick follow-up: its template was already spoken as the answer",
}


# ---- the preference: which reply helps the player more ---------------------------------

PREFER_SYSTEM = (
    "You compare two replies a chess coach could give at one moment of a game. Answer with "
    "one letter, A or B, and nothing else. " + COACH_WORDS_RULE
)


def prefer_question(rating: tuple[int, int] | None = None) -> str:
    low, high = rating or _rating()
    return (
        f"A chess coach said one of these two replies to a player rated about {low} to "
        f"{high}, in this moment of their game. Which would help that player more -- "
        "understand the position and play better -- while staying true to the record? "
        "Answer A or B."
    )


def _rating() -> tuple[int, int]:
    profile = load_settings().profile
    return profile.rating_low, profile.rating_high


def ai_first(item_id: str, seed: int) -> bool:
    """Whether the AI reply is shown as A for this item: a seeded coin per item, so the
    order bias cancels over the moments and each order is known."""
    return random.Random(f"{seed}:prefer:{item_id}").random() < 0.5


def prefer_prompt(
    item: dict, ai_reply: str, template: str, *, seed: int, rating: tuple[int, int] | None = None
) -> tuple[str, str, bool]:
    """(system, user message, whether the AI is A): the record -- the message the coach was
    given, its own earlier words marked (judge.COACH_WORDS_RULE) -- in the system prompt,
    the two replies and the question in the user message."""
    shown = str(item["messages"][-1]["content"]) if item.get("messages") else ""
    system = PREFER_SYSTEM + "\n\nThe record:\n<<<\n" + mark_coach_words(shown) + "\n>>>"
    first = ai_first(item["id"], seed)
    a, b = (ai_reply, template) if first else (template, ai_reply)
    user = (
        f'Reply A:\n"""\n{a}\n"""\n\nReply B:\n"""\n{b}\n"""\n\nQuestion: {prefer_question(rating)}'
    )
    return system, user, first


def judge_prefer(
    by_id: dict[str, dict],
    rows: list[dict],
    run: str,
    judge,
    ledger: Ledger,
    *,
    seed: int,
    workers: int = 8,
    reuse_stale: bool = False,
) -> list[dict]:
    """One A-or-B question per paired moment, at effort none, one vote: which reply helps
    the player more while staying true to the record. The order is a seeded coin per item.
    Kept under `<run>-prefer` (a moment asked once is never asked again); a call that fails
    or an answer that is neither A nor B is no preference -- it is counted, never scored.
    Every call is budget-checked on the ledger."""
    version = stamp.code_version()
    out_dir = judge_module.JUDGMENTS / f"{run}-prefer"
    out_dir.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    for path in out_dir.glob("*.jsonl"):
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            stamp.check(
                rec.get("commit"),
                f"the preference on {rec['id']}",
                reuse_stale=reuse_stale,
                current=version,
            )
            done[rec["id"]] = rec
    binary = binary_judge(judge)
    rating = _rating()
    lock = threading.Lock()
    failed = 0

    def ask(row: dict) -> dict:
        item = by_id[row["id"]]
        system, user, first = prefer_prompt(
            item, row["ai"], row["template"], seed=seed, rating=rating
        )
        completion = _chat_with_backoff(
            binary,
            system,
            [{"role": "user", "content": user}],
            ledger,
            row["id"],
            max_tokens=5,
            temperature=0.0,
            thinking=False,
        )
        answer = prefer_answer(completion.text or "")
        if answer is None:
            raise JudgeFailed(f"{row['id']}: no A or B in {completion.text!r}")
        return {
            "id": row["id"],
            "class": row["class"],
            "ai_is": "A" if first else "B",
            "answer": answer,
            "ai_preferred": (answer == "A") == first,
            "commit": version,
        }

    todo = [r for r in rows if r["id"] not in done]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(ask, row): row for row in todo}
        try:
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                except Exception as exc:  # noqa: BLE001 -- BudgetExceeded is not an Exception
                    failed += 1
                    if failed <= 5:
                        print(f"  preference failed: {type(exc).__name__}: {exc}", flush=True)
                    continue
                with lock:
                    game = by_id[rec["id"]]["game"].replace("/", "__")
                    with (out_dir / f"{game}.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done[rec["id"]] = rec
        except BudgetExceeded:
            for fut in futures:
                fut.cancel()
            raise
    if failed:
        print(f"  {failed} preferences not given (the call failed or no A/B)", flush=True)
    return [done[r["id"]] for r in rows if r["id"] in done]


def prefer_answer(text: str) -> str | None:
    """ "A" or "B" when the answer is that letter ("A", "B.", "Reply B"); None otherwise --
    "Both." is not B."""
    words = re.sub(r"[^A-Za-z]", " ", text).upper().split()
    if words and words[0] == "REPLY":
        words = words[1:]
    return words[0] if words and words[0] in ("A", "B") else None


def prefer_summary(prefs: list[dict]) -> list[str]:
    """The preference's lines for the report: the AI preferred in N of M, by class, and
    by the order the replies were shown in (so a position bias shows)."""
    if not prefs:
        return []
    n = len(prefs)
    won = sum(p["ai_preferred"] for p in prefs)
    lines = [
        "",
        f"Preferred (one A/B question a moment, effort none): the AI in {won} of {n}"
        f" ({won / n:.2f}).",
    ]
    for order, label in (("A", "shown first (A)"), ("B", "shown second (B)")):
        ps = [p for p in prefs if p["ai_is"] == order]
        if ps:
            k = sum(p["ai_preferred"] for p in ps)
            lines.append(f"- the AI {label}: preferred in {k} of {len(ps)}")
    answered_a = sum(p["answer"] == "A" for p in prefs)
    lines.append(f"- the judge answered A in {answered_a} of {n} (position bias shows here)")
    lines += ["", "| class | n | AI preferred |", "|---|---|---|"]
    by_class: dict[str, list[dict]] = {}
    for p in prefs:
        by_class.setdefault(p["class"], []).append(p)
    for cls in sorted(by_class):
        ps = by_class[cls]
        lines.append(f"| {cls} | {len(ps)} | {sum(p['ai_preferred'] for p in ps)} |")
    return lines


def round_seed(seed: int, round_no: int = 1) -> int:
    """The seed a round's moments are drawn with, in icr_play.py and here alike."""
    return seed * 1000 + round_no


def floor_record(item: dict) -> dict:
    """The template as a reply record, for the floor arm."""
    return {
        "id": item["id"],
        "class": item["class"],
        "reply": item.get("fallback") or "",
        "fallback_used": False,
        "problems": [],
    }


def is_quick(item: dict) -> bool:
    """A quick follow-up, or any moment with no template of its own to set beside the reply."""
    return bool(item.get("quick_followup")) or not (item.get("fallback") or "").strip()


def sort_out(items: list[dict]) -> tuple[list[dict], Counter]:
    """The drawn items that can be paired (the model's own reply beside a template of the
    same moment), and the rest counted by why not (LEFT_OUT_WORDS)."""
    pairable: list[dict] = []
    left_out: Counter = Counter()
    for it in items:
        status = reply_status(it["reply"])
        if status in ("model_failed", "template_only"):
            left_out[status] += 1
        elif is_quick(it):
            left_out[QUICK] += 1
        elif status == "rejected":
            left_out["rejected"] += 1
        else:
            pairable.append(it)
    return pairable, left_out


def template_names(template: str) -> list[str]:
    """The moves and squares the template names, each once, in order."""
    return list(dict.fromkeys(m.group(0) for m in _NAMED.finditer(template)))


def coverage(item: dict, reply: str) -> dict:
    """How much of the floor the reply covers, decided in code: what the coach required it
    to name (must_name), and the share of the template's moves and squares it names."""
    must = list(item.get("must_name") or [])
    named = template_names(item.get("fallback") or "")
    present = [x for x in named if _names(reply, x)]
    return {
        "must_name": must,
        "must_name_ok": omitted(reply, must) is None,
        "template_names": named,
        "present": present,
        "share": len(present) / len(named) if named else None,
    }


OUTCOMES = ("both", "ai_only", "template_only", "neither")
FACTS_OUTCOMES = ("fewer", "same", "more")


def _facts_outcome(ai: float | None, floor: float | None) -> str | None:
    """Whether the AI reply names fewer of the required facts than the template, as many,
    or more (the same facts are required of both)."""
    if ai is None or floor is None:
        return None
    if math.isclose(ai, floor):
        return "same"
    return "fewer" if ai < floor else "more"


def pair(items: dict[str, dict], ai: list[dict], floor: list[dict]) -> list[dict]:
    """One row per item judged in both arms."""
    floor_by_id = {j["id"]: j for j in floor}
    rows = []
    for a in ai:
        f = floor_by_id.get(a["id"])
        if f is None:
            continue
        item = items[a["id"]]
        ai_pass, floor_pass = bool(a["scores"]["pass"]), bool(f["scores"]["pass"])
        outcome = (
            "both"
            if ai_pass and floor_pass
            else "ai_only"
            if ai_pass
            else "template_only"
            if floor_pass
            else "neither"
        )
        ai_facts, floor_facts = a["scores"].get("completeness"), f["scores"].get("completeness")
        rows.append(
            {
                "id": a["id"],
                "class": a["class"],
                "level": a["level"],
                "outcome": outcome,
                "ai_pass": ai_pass,
                "floor_pass": floor_pass,
                "ai_completeness": ai_facts,
                "floor_completeness": floor_facts,
                "facts": _facts_outcome(ai_facts, floor_facts),
                "ai": a["reply"],
                "template": f["reply"],
                "coverage": coverage(item, a["reply"]),
                "ai_fails": _fails(a),
                "floor_fails": _fails(f),
                # The claims code left unjudged, by why: counted in the report, so a drop
                # rule that eats statements shows (the pre-spend audit, 2026-09-25).
                "ai_dropped": [d.get("why", "") for d in a.get("dropped_claims") or ()],
                "floor_dropped": [d.get("why", "") for d in f.get("dropped_claims") or ()],
            }
        )
    return rows


def _fails(j: dict) -> list[str]:
    """Why a judgment is not a pass, in short."""
    s, det = j["scores"], j.get("deterministic") or {}
    out = [f"leak: {x}" for x in det.get("leak") or []]
    out += [f"board: {x}" for x in det.get("board") or []]
    out += [f"rule: {x}" for x in det.get("rules") or []]
    out += [f"verdict: {x}" for x in det.get("verdict") or []]
    out += [f"floor: {x}" for x in det.get("floor") or []]
    out += [f"form: {x}" for x in (det.get("form") or {}).get("problems") or []]
    out += [f"unsupported: {x}" for x in s.get("unsupported") or []]
    if s.get("relevance") is not None and s["relevance"] < 0.5:
        out.append("answered with questions only")
    return out


def _mean(xs: list[float]) -> str:
    return f"{sum(xs) / len(xs):.2f}" if xs else "-"


def _quantile(values: list[float], q: float) -> float:
    """The nearest-rank quantile."""
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))]


def latency_by_trigger(items: list[dict]) -> list[tuple[str, int, float, float]]:
    """(trigger, moments, p50, p90) of the seconds the coach's calls took for a moment (all
    its calls: a reply sent back once takes two), over every moment the model answered."""
    by: dict[str, list[float]] = {}
    for it in items:
        reply = it.get("reply")
        if not reply or reply_status(reply) in ("model_failed", "template_only"):
            continue
        seconds = reply.get("elapsed_s")
        if seconds:
            by.setdefault(it["trigger"], []).append(float(seconds))
    return [
        (trigger, len(v), _quantile(v, 0.5), _quantile(v, 0.9)) for trigger, v in sorted(by.items())
    ]


def summary(
    rows: list[dict],
    left_out: Counter,
    *,
    seed: int | None = None,
    per_game: int | None = None,
    latency: list[tuple[str, int, float, float]] | None = None,
    prefer: list[dict] | None = None,
) -> str:
    """The pairing report, as markdown: the headline -- pass and completeness side by side,
    the wins each way -- then by class, the preference, the latency, and the moments to
    read."""
    lines = ["# Parity: the AI coach against its template", ""]
    n = len(rows)
    counts = Counter(r["outcome"] for r in rows)
    drawn = (
        f"drawn with seed {seed}" + (f", {per_game} a game" if per_game else "")
        if seed is not None
        else "drawn as the AI arm's judgments were"
    )
    lines.append(f"{n} moments judged in both arms ({drawn}).")
    for arm in ("ai", "floor"):
        dropped = Counter(why for r in rows for why in r.get(f"{arm}_dropped") or ())
        if dropped:
            said = ", ".join(f"{k} {v}" for k, v in dropped.most_common())
            lines.append(
                f"Claims left unjudged in the {arm} arm: {sum(dropped.values())} ({said})."
            )
    lines += [
        "",
        "| arm | pass | completeness |",
        "|---|---|---|",
        f"| AI | {_mean([float(r['ai_pass']) for r in rows])} |"
        f" {_mean([r['ai_completeness'] for r in rows if r['ai_completeness'] is not None])} |",
        f"| template | {_mean([float(r['floor_pass']) for r in rows])} |"
        f" {_mean([r['floor_completeness'] for r in rows if r['floor_completeness'] is not None])}"
        " |",
        "",
    ]
    rejected = left_out.get("rejected", 0)
    lines.append(
        "Outcomes: " + ", ".join(f"{k} {counts.get(k, 0)}" for k in OUTCOMES) + "."
        f" Wins: the AI {counts.get('ai_only', 0)}; the template"
        f" {counts.get('template_only', 0) + rejected} ({counts.get('template_only', 0)} judged,"
        f" {rejected} where the checks refused the model and the template spoke)."
    )
    facts = Counter(r["facts"] for r in rows if r["facts"] is not None)
    lines.append(
        "Required facts: the AI names fewer than the template in "
        f"{facts.get('fewer', 0)}, the same in {facts.get('same', 0)}, more in "
        f"{facts.get('more', 0)}."
    )
    if left_out:
        lines.append(
            "Not paired: "
            + "; ".join(f"{v} {LEFT_OUT_WORDS.get(k, k)}" for k, v in sorted(left_out.items()))
            + "."
        )
    shares = [r["coverage"]["share"] for r in rows if r["coverage"]["share"] is not None]
    must = [r for r in rows if r["coverage"]["must_name"]]
    if shares:
        lines.append(
            f"Floor coverage: the reply names {sum(shares) / len(shares):.2f} of the moves and"
            f" squares its template names (over {len(shares)} templates that name any)."
        )
    if must:
        ok = sum(r["coverage"]["must_name_ok"] for r in must)
        lines.append(f"What the coach required it to name: named in {ok} of {len(must)}.")
    lines += [
        "",
        "| class | n | both | ai_only | template_only | neither | fewer facts | more facts |"
        " coverage |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    by_class: dict[str, list[dict]] = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(r)
    for cls in sorted(by_class):
        rs = by_class[cls]
        c = Counter(r["outcome"] for r in rs)
        f = Counter(r["facts"] for r in rs)
        cov = [r["coverage"]["share"] for r in rs if r["coverage"]["share"] is not None]
        outcomes = " | ".join(str(c.get(k, 0)) for k in OUTCOMES)
        lines.append(
            f"| {cls} | {len(rs)} | {outcomes} | {f.get('fewer', 0)} | {f.get('more', 0)} |"
            f" {_mean(cov)} |"
        )
    lines += prefer_summary(prefer or [])
    if latency:
        lines += [
            "",
            "Latency of the coach's calls per moment, seconds (the whole reply; the first"
            " sentence is not measured without streaming):",
            "",
            "| trigger | moments | p50 | p90 |",
            "|---|---|---|---|",
        ]
        lines += [f"| {t} | {k} | {p50:.1f} | {p90:.1f} |" for t, k, p50, p90 in latency]
    worse = [r for r in rows if r["outcome"] == "template_only" or r["facts"] == "fewer"]
    if worse:
        lines += ["", "## Where the template did better", ""]
        for r in worse:
            lines += [
                f"- {r['id']} ({r['class']}, {r['level']}): {r['outcome']}, the AI names"
                f" {r['facts'] or 'n/a'} required facts",
                f"  - AI: {r['ai']}",
                f"  - fails: {'; '.join(r['ai_fails']) or 'none'}",
                f"  - template: {r['template']}",
            ]
    return "\n".join(lines) + "\n"


def judge_floor(
    by_id: dict[str, dict],
    ids: list[str],
    run: str,
    judge,
    ledger: Ledger,
    *,
    thinking: bool,
    workers: int = 8,
    reuse_stale: bool = False,
) -> list[dict]:
    """The floor arm: the templates of `ids`, judged under `<run>-floor`."""
    return judge_all(
        by_id,
        [floor_record(by_id[i]) for i in ids],
        f"{run}-floor",
        judge,
        ledger,
        thinking=thinking,
        quiet=True,
        workers=workers,
        floor=True,
        reuse_stale=reuse_stale,
    )


def write_report(
    run: str,
    rows: list[dict],
    left_out: Counter,
    *,
    seed: int | None = None,
    per_game: int | None = None,
    latency: list[tuple[str, int, float, float]] | None = None,
    prefer: list[dict] | None = None,
) -> Path:
    PARITY.mkdir(parents=True, exist_ok=True)
    md = PARITY / f"{run}.md"
    md.write_text(
        summary(rows, left_out, seed=seed, per_game=per_game, latency=latency, prefer=prefer),
        encoding="utf-8",
    )
    (PARITY / f"{run}.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "per_game": per_game,
                "rows": rows,
                "left_out": dict(left_out),
                "latency": latency,
                "prefer": prefer or [],
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return md


def load_items(folder: Path, *, reuse_stale: bool) -> dict[str, dict]:
    """The items of a play-through directory (one jsonl per game), refused when another
    version of the code made them unless `reuse_stale`."""
    out: dict[str, dict] = {}
    version = stamp.code_version()
    for path in sorted(folder.glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            item = json.loads(line)
            stamp.check(
                item.get("commit"), f"{path.name}", reuse_stale=reuse_stale, current=version
            )
            out[item["id"]] = item
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--items", type=Path, required=True, help="a play-through's items directory")
    ap.add_argument("--run", required=True, help="the AI arm's judgments run name")
    ap.add_argument("--judge", default="openai:gpt-5.6-luna@medium")
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument(
        "--moments-per-game",
        type=int,
        default=8,
        help="when the AI arm is not judged yet: this many moments a game (0 = all)",
    )
    ap.add_argument("--seed", type=int, default=7, help="icr_play.py's --seed")
    ap.add_argument("--round", type=int, default=1, help="the round the items are of")
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--dry-run", action="store_true", help="estimate the cost; call nothing")
    ap.add_argument("--reuse-stale", action="store_true")
    ap.add_argument(
        "--no-prefer",
        dest="prefer",
        action="store_false",
        help="skip the A-or-B preference between the AI reply and the template",
    )
    args = ap.parse_args()

    from icr_play import sample_moments

    items = load_items(args.items, reuse_stale=args.reuse_stale)
    with_reply = [it for it in items.values() if it.get("reply")]
    ai_dir = ROOT / "data" / "eval" / "judgments" / args.run
    judged_ids = (
        {
            json.loads(line)["id"]
            for p in ai_dir.glob("*.jsonl")
            for line in p.open(encoding="utf-8")
        }
        if ai_dir.exists()
        else set()
    )
    seed = round_seed(args.seed, args.round)
    if judged_ids:
        chosen = [it for it in with_reply if it["id"] in judged_ids]
    else:
        chosen = sample_moments(with_reply, args.moments_per_game, seed)
    pairable, left_out = sort_out(chosen)
    print(
        f"{len(chosen)} moments (seed {seed}), {len(pairable)} to pair; not paired:"
        f" {dict(left_out) or 'none'}",
        flush=True,
    )
    if args.dry_run:
        prices = settings_prices()

        def record(it: dict) -> str:
            return (
                BINARY_SYSTEM
                + evidence_block(it, with_reference=False)
                + (it.get("fallback") or "")
            )

        todo_ai = [it for it in pairable if it["id"] not in judged_ids]
        cost = estimate.judge_estimate(todo_ai, prices, record_of=record).add(
            estimate.judge_estimate(pairable, prices, record_of=record)
        )
        print(f"dry run: AI arm {len(todo_ai)} + floor arm {len(pairable)} replies: {cost.line()}")
        if args.prefer:
            asked = prefer_estimate(pairable, prices, seed)
            print(f"  preference, {len(pairable)} questions: {asked.line()}")
        return
    judge, thinking = judge_provider(args.judge)
    ledger = Ledger(
        f"parity-{args.run}",
        "parity",
        judge.model,
        budget_usd=args.budget_usd,
        resume=True,
        version=stamp.code_version(),
    )
    if ledger.before:
        print(f"this run spent ${ledger.before:.4f} before; it counts against the budget")
    try:
        # Only the model's own replies: a template the checks put in its place is the floor
        # arm's text already.
        ai = judge_all(
            items,
            [it["reply"] for it in pairable],
            args.run,
            judge,
            ledger,
            thinking=thinking,
            quiet=True,
            workers=args.judge_workers,
            reuse_stale=args.reuse_stale,
        )
        ai = [j for j in ai if j.get("status", "model") == "model"]
        floor = judge_floor(
            items,
            [j["id"] for j in ai],
            args.run,
            judge,
            ledger,
            thinking=thinking,
            workers=args.judge_workers,
            reuse_stale=args.reuse_stale,
        )
        rows = pair(items, ai, floor)
        prefs = (
            judge_prefer(
                items,
                rows,
                args.run,
                judge,
                ledger,
                seed=seed,
                workers=args.judge_workers,
                reuse_stale=args.reuse_stale,
            )
            if args.prefer
            else []
        )
    except BudgetExceeded as exc:
        print(f"stopped: {exc}", flush=True)
        raise SystemExit(2) from None
    latency = latency_by_trigger(with_reply)
    # icr_play.py draws a round's moments with the same seed (round_seed): judged there or
    # drawn here, they are the same moments.
    per_game = None if judged_ids else args.moments_per_game
    path = write_report(
        args.run, rows, left_out, seed=seed, per_game=per_game, latency=latency, prefer=prefs
    )
    print(summary(rows, left_out, seed=seed, per_game=per_game, latency=latency, prefer=prefs))
    print(f"wrote {path}; {ledger.summary()}", flush=True)


def prefer_estimate(items: list[dict], prices: tuple[float, float, float], seed: int):
    """The preference's cost: one uncached call a paired moment, its prompt the record and
    the two replies (the template's length stands in for the AI reply's)."""

    def prompt(it: dict) -> str:
        template = it.get("fallback") or ""
        system, user, _ = prefer_prompt(it, template, template, seed=seed, rating=(0, 0))
        return system + user

    return estimate.single_call_estimate([prompt(it) for it in items], prices)


if __name__ == "__main__":
    main()
