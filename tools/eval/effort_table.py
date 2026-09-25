"""The effort bench: the same drawn moments with the coach at each reasoning effort, side
by side.

Each arm is an icr_play.py run of its own -- its --name, its ledger and budget, its
judgments -- with the coach at one effort (the routine effort stays the app's, none) and
everything else the same: the games (--source, --games, --seed), the code, and so the
catalogue and the draw (parity.round_seed): the same moments in every arm, the history
before each played at that arm's effort. --floor judges the templates and asks the A-or-B
preference; this table reads what the arms wrote:

  for e in none low medium; do
    uv run python tools/eval/icr_play.py --source bots --games 5 --rounds 1 \
        --moments-per-game 8 --seed 7 --floor --coach openai:gpt-5.6-luna@$e \
        --name effort-$e --budget-usd 1 [--dry-run]
  done
  uv run python tools/eval/effort_table.py effort-none effort-low effort-medium

Per arm: pass, completeness, fabrication (the AI arm's judged moments), the share of the
paired moments where the judge preferred the AI reply to the template, the coach's latency
by trigger (p50/p90 seconds a moment), and the dollars on the arm's ledger (the coach's
calls and in all). The table says when the arms did not judge the same moments or ran
different code: then they are not comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge import usage_records  # noqa: E402
from report import aggregate, load  # noqa: E402

EVAL = ROOT / "data" / "eval"
PARITY = EVAL / "parity"
# The ledger's labels for the coach's own calls (providers.Metered); a judge call is labelled
# with the moment it judged.
COACH_LABELS = ("coach", "coach-routine", "claims", "summariser", "verifier")


def arm(name: str, round_no: int = 1) -> dict:
    """What one arm wrote: its round's record, its judgments, its parity report, its spend."""
    record_path = EVAL / name / f"round-{round_no}.json"
    if not record_path.exists():
        raise SystemExit(f"{name}: no {record_path.name}; run the arm first")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    run = record["run"]
    judged = load(run, None)
    parity_path = PARITY / f"{run}.json"
    parity = json.loads(parity_path.read_text(encoding="utf-8")) if parity_path.exists() else {}
    calls = usage_records(name)
    usd = sum(float(c.get("usd") or 0.0) for c in calls)
    coach_usd = sum(float(c.get("usd") or 0.0) for c in calls if c.get("item") in COACH_LABELS)
    coach = str(record.get("coach") or "")
    prefs = parity.get("prefer") or []
    return {
        "name": name,
        "run": run,
        "effort": coach.rpartition("@")[2] if "@" in coach else "?",
        "commit": record.get("commit"),
        "seed": record.get("seed"),
        "drawn": sorted(record.get("drawn") or []),
        "ids": sorted(j["id"] for j in judged),
        "scores": aggregate(judged) if judged else {},
        "preferred": (sum(p["ai_preferred"] for p in prefs) / len(prefs)) if prefs else None,
        "prefs": len(prefs),
        "latency": {t: (n, p50, p90) for t, n, p50, p90 in parity.get("latency") or []},
        "usd": usd,
        "coach_usd": coach_usd,
    }


def _f(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}"


def table(arms: list[dict]) -> str:
    lines = ["# Effort bench", ""]
    drawn = {tuple(a["drawn"]) for a in arms}
    commits = {a["commit"] for a in arms}
    seeds = {a["seed"] for a in arms}
    if len(drawn) > 1 or len(commits) > 1 or len(seeds) > 1:
        lines.append(
            "NOT COMPARABLE: the arms drew different moments, ran different code or drew with"
            f" different seeds (commits {sorted(map(str, commits))},"
            f" seeds {sorted(map(str, seeds))})."
        )
        lines.append("")
    # A moment drawn in every arm but judged in only some: its call failed there.
    judged = [set(a["ids"]) for a in arms]
    if judged and len({frozenset(s) for s in judged}) > 1:
        common = set.intersection(*judged)
        lines.append(
            f"Judged in every arm: {len(common)}; the rest failed or were not called in some"
            " arm (see each arm's round log)."
        )
        lines.append("")
    lines += [
        "| arm | effort | moments | pass | completeness | fabrication | preferred over the"
        " template | $ coach | $ in all |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a in arms:
        s = a["scores"]
        preferred = f"{_f(a['preferred'])} of {a['prefs']}" if a["prefs"] else "-"
        lines.append(
            f"| {a['name']} | {a['effort']} | {len(a['ids'])} | {_f(s.get('pass'))} |"
            f" {_f(s.get('completeness'))} | {_f(s.get('fabrication'))} | {preferred} |"
            f" {a['coach_usd']:.2f} | {a['usd']:.2f} |"
        )
    triggers = sorted({t for a in arms for t in a["latency"]})
    if triggers:
        lines += [
            "",
            "Coach seconds a moment, p50/p90 (the whole reply, all its calls):",
            "",
            "| trigger | " + " | ".join(a["name"] for a in arms) + " |",
            "|---|" + "---|" * len(arms),
        ]
        for t in triggers:
            cells = []
            for a in arms:
                n, p50, p90 = a["latency"].get(t, (0, None, None))
                cells.append(f"{p50:.1f}/{p90:.1f} (n={n})" if n else "-")
            lines.append(f"| {t} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("names", nargs="+", help="the arms' --name, e.g. effort-none effort-low")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--out", type=Path, help="also write the table to this markdown file")
    args = ap.parse_args()
    text = table([arm(n, args.round) for n in args.names])
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
