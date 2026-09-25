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
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import estimate  # noqa: E402
import stamp  # noqa: E402
from judge import (  # noqa: E402
    BINARY_SYSTEM,
    BudgetExceeded,
    Ledger,
    evidence_block,
    judge_all,
    reply_status,
    settings_prices,
)
from providers import judge_provider  # noqa: E402

from sidecoach.coach.coach import _names, omitted  # noqa: E402

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
) -> str:
    """The pairing report, as markdown: the headline -- pass and completeness side by side,
    the wins each way -- then by class, the moments to read, and the latency."""
    lines = ["# Parity: the AI coach against its template", ""]
    n = len(rows)
    counts = Counter(r["outcome"] for r in rows)
    drawn = (
        f"drawn with seed {seed}" + (f", {per_game} a game" if per_game else "")
        if seed is not None
        else "drawn as the AI arm's judgments were"
    )
    lines.append(f"{n} moments judged in both arms ({drawn}).")
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
) -> Path:
    PARITY.mkdir(parents=True, exist_ok=True)
    md = PARITY / f"{run}.md"
    md.write_text(
        summary(rows, left_out, seed=seed, per_game=per_game, latency=latency), encoding="utf-8"
    )
    (PARITY / f"{run}.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "per_game": per_game,
                "rows": rows,
                "left_out": dict(left_out),
                "latency": latency,
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
    except BudgetExceeded as exc:
        print(f"stopped: {exc}", flush=True)
        raise SystemExit(2) from None
    rows = pair(items, ai, floor)
    latency = latency_by_trigger(with_reply)
    # icr_play.py draws a round's moments with the same seed (round_seed): judged there or
    # drawn here, they are the same moments.
    per_game = None if judged_ids else args.moments_per_game
    path = write_report(args.run, rows, left_out, seed=seed, per_game=per_game, latency=latency)
    print(summary(rows, left_out, seed=seed, per_game=per_game, latency=latency))
    print(f"wrote {path}; {ledger.summary()}", flush=True)


if __name__ == "__main__":
    main()
