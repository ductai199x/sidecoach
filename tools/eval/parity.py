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

and the floor's coverage, decided in code: whether the AI reply names what the coach
required of it (`must_name`, the coach's own `omitted`), and the share of the moves and
squares the template names that the reply names too. Items whose reply the model never
wrote (its call failed, or no call was made) are left out and counted; so are those where
the checks refused the model and the template spoke (the two arms are then the same text).

  uv run python tools/eval/parity.py --items data/eval/<name>/items-<version>-r1 \
      --run <name>-<version>-r1 --budget-usd 2 [--dry-run]

Every judge call is on the ledger, checked against --budget-usd before it is made.
"""

from __future__ import annotations

import argparse
import json
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


def floor_record(item: dict) -> dict:
    """The template as a reply record, for the floor arm."""
    return {
        "id": item["id"],
        "class": item["class"],
        "reply": item.get("fallback") or "",
        "fallback_used": False,
        "problems": [],
    }


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
        rows.append(
            {
                "id": a["id"],
                "class": a["class"],
                "level": a["level"],
                "outcome": outcome,
                "ai_pass": ai_pass,
                "floor_pass": floor_pass,
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


def excluded(replies: list[dict]) -> Counter:
    """The replies left out of the pairing, by why: the model did not write them, or the
    checks refused it and the template spoke."""
    return Counter(status for r in replies if (status := reply_status(r)) != "model")


def summary(rows: list[dict], left_out: Counter) -> str:
    """The pairing report, as markdown."""
    lines = ["# Parity: the AI coach against its template", ""]
    n = len(rows)
    counts = Counter(r["outcome"] for r in rows)
    rates = (
        f" (AI pass {sum(r['ai_pass'] for r in rows) / n:.2f}, template pass"
        f" {sum(r['floor_pass'] for r in rows) / n:.2f})"
        if n
        else ""
    )
    lines.append(
        f"{n} moments judged in both arms: "
        + ", ".join(f"{k} {counts.get(k, 0)}" for k in OUTCOMES)
        + rates
    )
    if left_out:
        lines.append(
            "Left out: "
            + ", ".join(f"{v} {k}" for k, v in sorted(left_out.items()))
            + " (model_failed and template_only: the model wrote nothing; rejected: the"
            " checks refused it and the template spoke)."
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
        "| class | n | both | ai_only | template_only | neither | coverage |",
        "|---|---|---|---|---|---|---|",
    ]
    by_class: dict[str, list[dict]] = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(r)
    for cls in sorted(by_class):
        rs = by_class[cls]
        c = Counter(r["outcome"] for r in rs)
        cov = [r["coverage"]["share"] for r in rs if r["coverage"]["share"] is not None]
        cov_text = f"{sum(cov) / len(cov):.2f}" if cov else "-"
        outcomes = " | ".join(str(c.get(k, 0)) for k in OUTCOMES)
        lines.append(f"| {cls} | {len(rs)} | {outcomes} | {cov_text} |")
    worse = [r for r in rows if r["outcome"] == "template_only"]
    if worse:
        lines += ["", "## Where the template passed and the AI did not", ""]
        for r in worse:
            lines += [
                f"- {r['id']} ({r['class']}, {r['level']})",
                f"  - AI: {r['ai']}",
                f"  - fails: {'; '.join(r['ai_fails'])}",
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


def write_report(run: str, rows: list[dict], left_out: Counter) -> Path:
    PARITY.mkdir(parents=True, exist_ok=True)
    md = PARITY / f"{run}.md"
    md.write_text(summary(rows, left_out), encoding="utf-8")
    (PARITY / f"{run}.json").write_text(
        json.dumps({"rows": rows, "left_out": dict(left_out)}, indent=1, ensure_ascii=False),
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
        default=5,
        help="when the AI arm is not judged yet: this many moments a game (0 = all)",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--dry-run", action="store_true", help="estimate the cost; call nothing")
    ap.add_argument("--reuse-stale", action="store_true")
    args = ap.parse_args()

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
    if judged_ids:
        chosen = [it for it in with_reply if it["id"] in judged_ids]
    else:
        from icr_play import sample_moments

        chosen = sample_moments(with_reply, args.moments_per_game, args.seed)
    replies = [it["reply"] for it in chosen]
    left_out = excluded(replies)
    pairable = [it for it in chosen if reply_status(it["reply"]) == "model"]
    print(
        f"{len(chosen)} moments ({len(pairable)} with the model's own reply);"
        f" left out {dict(left_out) or 'none'}",
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
    ledger = Ledger(f"parity-{args.run}", "parity", judge.model, budget_usd=args.budget_usd)
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
    path = write_report(args.run, rows, left_out)
    print(summary(rows, left_out))
    print(f"wrote {path}; {ledger.summary()}", flush=True)


if __name__ == "__main__":
    main()
