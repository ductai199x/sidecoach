"""Inputs match the judge (Tai, 2026-09-25): every demand the judge makes of a reply must be
asked for in the coach's request, class by class, and every check the coach's code runs on a
reply must be stated to the model. A round scored against demands its directive never made
measures the wrong thing, and costs money doing it.

For each drawn moment (the dry run's `--dump-items`), the demands the judge will make:

  facts     the required facts (taxonomy.required), each by its key
  rubric    the criteria of rubric.toml that apply to the moment's class
  checks    the coach's own checks that apply (figures, the earlier moments, the move just
            played, what the reply must name)

and, for each, whether the request asks for it -- a phrase of the directive, the value on the
sheet. A key with no rule here is a gap too: a new fact, criterion or check gets its rule in
the same change. Free: no calls.

  uv run --no-sync python tools/eval/icr_play.py --source bots --games 5 --rounds 1 \\
      --moments-per-game 8 --seed 11 --dry-run --dump-items data/eval/align/drawn.jsonl
  uv run --no-sync python tools/eval/align.py data/eval/align/drawn.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rubric as rubric_module  # noqa: E402

from sidecoach.coach import coach as coach_module  # noqa: E402
from sidecoach.coach.person import third_person  # noqa: E402


def directive_of(item: dict) -> str:
    content = str(item["messages"][-1]["content"])
    return content.rsplit("# Task\n", 1)[-1]


def request_of(item: dict) -> str:
    return str(item["messages"][-1]["content"])


def _has(*phrases: str) -> Callable[[dict, dict], bool]:
    """Asked when the directive says one of `phrases` (case-insensitive)."""

    def rule(item: dict, fact: dict) -> bool:
        low = directive_of(item).lower()
        return any(p.lower() in low for p in phrases)

    return rule


def _value_in_directive(item: dict, fact: dict) -> bool:
    return fact["value"].lower() in directive_of(item).lower()


def _on_sheet(item: dict, value: str) -> bool:
    """The value is in the request, as written or as the sheet puts it in the third person
    ("against their king" is "against the opponent's king" on the sheet)."""
    request = request_of(item).lower()
    return value.lower() in request or third_person(value, side="player").lower() in request


def _on_sheet_and(*phrases: str) -> Callable[[dict, dict], bool]:
    """Asked when the value is in the request and the directive points at it."""

    def rule(item: dict, fact: dict) -> bool:
        return _on_sheet(item, fact["value"]) and _has(*phrases)(item, fact)

    return rule


# Each required fact's key: how the request asks for it.
FACT_RULES: dict[str, Callable[[dict, dict], bool]] = {
    "label": _value_in_directive,
    "better_move": lambda it, f: (
        _value_in_directive(it, f) or _on_sheet_and("better move", "the move")(it, f)
    ),
    "cost": _has("cost"),
    "idea": _on_sheet_and("pattern", "the idea"),
    "tactic": _on_sheet_and("tactic"),
    "stakes": _has("stakes"),
    "key_fact": _has("key fact"),
    "trap": _has("trap"),
    "goal": _has("goal"),
    "pattern": _has("pattern"),
    "nature": _has("forcing", "nature"),
    "target": _has("target"),
    "piece": _has("piece"),
    "candidate": lambda it, f: _value_in_directive(it, f) or _has("give the move")(it, f),
    "line": _has("line"),
    "plan": _has("plan", "aim for"),
    "eval": _has("who stands better"),
    "material": _has("material"),
    "threat": _has("nothing is threatened"),
    "named_verdict": _has("holds up"),
    "move": _value_in_directive,
    "strong_moves": _has("strong"),
    "lost_worst": _has("ground was lost"),
    "opening": _has("opening"),
    "result": _has("result"),
}


def fact_rule(key: str) -> Callable[[dict, dict], bool] | None:
    if key.startswith("threat_"):
        return _on_sheet_and("threat")
    if key.startswith("count_"):
        return _has("slips")
    return FACT_RULES.get(key)


# Each rubric criterion: how the request asks for it (the directive, or the system prompt for
# what holds of every reply).
RUBRIC_RULES: dict[str, Callable[[dict], bool]] = {
    "why": lambda it: any(
        p in directive_of(it).lower()
        for p in (
            "what it does",
            "why the better move works",
            "what it goes after",
            "why, concretely",
            "threatens or allows",
            "does or allows",
            "why it works",
            "what makes it the move",
        )
    ),
    "transfer": lambda it: any(
        p in directive_of(it).lower()
        for p in (
            "pattern to remember",
            "lesson to keep",
            "principle behind it",
            "pattern to keep",
            "idea to keep",
        )
    ),
    "actionable": lambda it: any(
        p in directive_of(it).lower()
        for p in (
            "question",
            "one thing to work on",
            "lesson to keep",
            "pattern to keep",
            "what to look for",
        )
    ),
    "question_board": lambda it: "a piece, a square or a move of this position" in it["system"],
    "question_idea": lambda it: "sets them looking the right way" in it["system"],
    "question_open": lambda it: "never to repeat back what you just told them" in it["system"],
    "connection": lambda it: coach_module.TIE_LISTED in directive_of(it),
    "no_gushing": lambda it: "never gushing" in it["system"],
    "no_scolding": lambda it: "never with blame or harsh words" in it["system"],
    "no_hedging": lambda it: "no hedging" in it["system"],
}


def gaps(item: dict, criteria: list) -> list[str]:
    """What the judge will demand of this moment's reply that its request does not ask."""
    out: list[str] = []
    directive = directive_of(item)
    request = request_of(item)
    for fact in item.get("required") or []:
        rule = fact_rule(fact["key"])
        if rule is None:
            out.append(f"fact {fact['key']}: no rule in align.py")
        elif not rule(item, fact):
            out.append(f"fact {fact['key']} ({fact['value'][:60]}): not asked")
    cls = item["class"]
    for criterion in criteria:
        # A closing question is judged when there is one: the directive must allow it.
        if not criterion.applies(cls, "?" if criterion.only_if else "", request):
            continue
        if criterion.only_if == "ends_with_question":
            _, wants = coach_module.directive_limits(directive)
            if wants is False:
                continue
        rule = RUBRIC_RULES.get(criterion.key)
        if rule is None:
            out.append(f"rubric {criterion.key}: no rule in align.py")
        elif not rule(item):
            out.append(f"rubric {criterion.key}: not asked")
    # The coach's checks, stated: a figure where none was asked for, the earlier moments, the
    # move just played.
    trigger = item["trigger"]
    # The figure check is keyed on the constant itself (`Speaker._notes`), so a remark that
    # carries no figures unasked must carry it.
    if trigger in ("decisive_question", "stuck", "post_move", "post_mistake") and (
        coach_module.NO_FIGURES not in directive
    ):
        out.append("check figures: applied, never stated")
    if item.get("unlisted") is not None and not (
        coach_module.TIE_LISTED in directive or coach_module.NO_EARLIER in directive
    ):
        out.append("check earlier moments: applied, never stated")
    if item.get("unlisted") is not None and "number" not in directive:
        out.append("check move numbers: applied, never stated")
    if trigger in ("post_move", "post_mistake") and not (
        coach_module.NOT_THE_MOVE in directive or coach_module.NOT_THE_MOVE_HELD in directive
    ):
        out.append("check the move just played: applied, never stated")
    if item.get("must_name") and not any(
        p in directive
        for p in (
            "the better move",
            "was better",
            "named as the sheet names it",
            "give the move",
            "the move and what it does",
            "as the fact sheet names them",
        )
    ):
        out.append("check must name: applied, never stated")
    if coach_module.TIE_TO_THE_GAME in directive:
        out.append("the link marker reached the model unreplaced")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items", type=Path, help="the drawn moments (icr_play --dump-items)")
    args = ap.parse_args()
    criteria = rubric_module.load_rubric()
    items = [
        json.loads(line)
        for line in args.items.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    found = Counter()
    bad = 0
    for item in items:
        problems = gaps(item, criteria)
        if problems:
            bad += 1
            print(f"--- {item['id']} [{item['class']}, {item['level']}]")
            for p in problems:
                print(f"    {p}")
                found[p.split(":")[0]] += 1
    print(f"\n{len(items)} moments, {bad} with gaps")
    for key, n in found.most_common():
        print(f"  {n:3d}  {key}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
