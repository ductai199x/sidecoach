"""Scores per class and per source for one or more runs, side by side.

Run:  uv run python tools/eval/report.py --run <judgments run> [--run <another>]
      [--ids split.json:heldout_games]
Prints the tables; --out <file.md> writes them too.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JUDGMENTS = ROOT / "data" / "eval" / "judgments"


def load(run: str, ids: set[str] | None) -> list[dict]:
    out = []
    for path in sorted((JUDGMENTS / run).glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            if ids is None or rec["id"] in ids:
                out.append(rec)
    return out


def _mean(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def aggregate(recs: list[dict]) -> dict:
    s = [r["scores"] for r in recs]
    return {
        "n": len(recs),
        "pass": _mean([1.0 if x["pass"] else 0.0 for x in s]),
        "completeness": _mean([x["completeness"] for x in s]),
        "correctness": _mean([x["correctness"] for x in s]),
        "claim_support": _mean([x.get("claim_support") for x in s]),
        "fabrication": _mean([1.0 if x["fabrication"] else 0.0 for x in s]),
        "leak": _mean([1.0 if x["leak"] else 0.0 for x in s]),
        "board": _mean([1.0 if x["board"] else 0.0 for x in s]),
        "form": _mean([1.0 if x["form_ok"] else 0.0 for x in s]),
        "relevance": _mean([x["relevance"] for x in s]),
        # Over the items with a master's note: the point made; the point in the facts; the
        # point made where the facts had it (what the coach could have done).
        "idea": _mean(
            [1.0 if x["idea_match"] else 0.0 for x in s if x.get("idea_match") is not None]
        ),
        "idea_in_facts": _mean(
            [1.0 if x.get("idea_in_facts") else 0.0 for x in s if x.get("idea_match") is not None]
        ),
        "idea_reachable": _mean(
            [
                1.0 if x["idea_match"] else 0.0
                for x in s
                if x.get("idea_match") is not None and x.get("idea_in_facts")
            ]
        ),
        "fallback": _mean([1.0 if r.get("fallback_used") else 0.0 for r in recs]),
    }


def fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}"


def table(title: str, groups: dict[str, dict[str, dict]], runs: list[str]) -> list[str]:
    cols = [
        "n",
        "pass",
        "completeness",
        "correctness",
        "claim_support",
        "fabrication",
        "leak",
        "form",
        "relevance",
        "idea",
        "idea_in_facts",
        "idea_reachable",
        "fallback",
    ]
    lines = [
        f"### {title}",
        "",
        "| group | run | " + " | ".join(cols) + " |",
        "|---|---|" + "---|" * len(cols),
    ]
    for group in sorted(groups):
        for run in runs:
            a = groups[group].get(run)
            if not a:
                continue
            lines.append(
                f"| {group} | {run} | "
                + " | ".join(str(a["n"]) if c == "n" else fmt(a[c]) for c in cols)
                + " |"
            )
    return lines + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--ids")
    ap.add_argument("--out", help="also write the tables to this markdown file")
    args = ap.parse_args()
    ids = None
    if args.ids:
        path, _, key = args.ids.rpartition(":")
        ids = set(json.loads(Path(path).read_text(encoding="utf-8"))[key])
    by_class: dict[str, dict[str, dict]] = {}
    by_source: dict[str, dict[str, dict]] = {}
    overall: dict[str, dict[str, dict]] = {"all": {}}
    for run in args.run:
        recs = load(run, ids)
        overall["all"][run] = aggregate(recs)
        for key, group in (("class", by_class), ("source", by_source)):
            buckets: dict[str, list[dict]] = {}
            for r in recs:
                buckets.setdefault(r[key], []).append(r)
            for name, rs in buckets.items():
                group.setdefault(name, {})[run] = aggregate(rs)
    lines = [
        f"# Coach evaluation: {', '.join(args.run)}" + (f" ({args.ids})" if args.ids else ""),
        "",
    ]
    lines += table("Overall", overall, args.run)
    lines += table("By source", by_source, args.run)
    lines += table("By class", by_class, args.run)
    print("\n".join(lines))
    if args.out:
        Path(args.out).write_text("\n".join(lines), encoding="utf-8")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
