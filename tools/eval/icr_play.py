"""ICR on play-throughs (Nguyen & Stamm, arXiv 2606.23917, section 5), the way the coach is
used: each round, every master game is played through the real coach from move 1 to the
end, coached from the side the facts were built for. At each moment the policy fires --
the verdict after the player's move, a question at a decisive moment, a hint when they
are stuck, the answers to what they ask -- the model answers from the harness as it
stands, and its answer goes forward into the harness of the next moment, as it does live.
Every (harness, directive, answer) is judged; the K worst are paired with the reply that
was right (written from the facts by the writer model, checked and judged) and appended
to C in the system prompt; repeat.

  data/eval/<name>/pool.json         C, one demonstration per entry
  data/eval/<name>/C-round-N.md      C as it stood after round N
  data/eval/<name>/round-N.json      the round's scores, the demos added, the failures
  data/eval/<name>/items-rN/         every moment of the round with its reply (jsonl per game)
  judgments under run <name>-rN

  uv run python tools/eval/icr_play.py --coach openai:gpt-5.6-luna@medium
      --judge openai:gpt-5.6-luna@medium
      --writer openai:gpt-5.6-luna@medium --games 6 --rounds 4 --demos-per-round 5 --name icr-play
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import ENGINE_CACHE, FACTS, MASTERS, CachedEngine, build_game  # noqa: E402
from judge import (  # noqa: E402
    JudgeFailed,
    Ledger,
    deterministic,
    evidence_block,
    judge_all,
    judge_one,
    parse_judgment,
)  # noqa: E402
from pool import C_HEADER, JARGON, Pool, facts_gap, failure_score  # noqa: E402
from providers import Metered, coach_provider, judge_provider  # noqa: E402
from report import aggregate  # noqa: E402

from sidecoach.cli import _draft_check  # noqa: E402
from sidecoach.coach import prompt as coach_prompt  # noqa: E402

WRITER_SYSTEM = (
    "You write the reply a chess coach should have given, beside a club player, in the "
    "middle of their game. You are given: the message the coach saw at that moment (the "
    "game so far, the position, the facts it may use, the disclosure level in force, and "
    "the task); the evidence the judge holds it to, with the required facts and, when there "
    "is one, the master's own note on the move; what the coach actually wrote and what was "
    "wrong with it; and the form required.\n\n"
    "Write the reply the best coach would give: every required fact, said naturally; every "
    "statement a fact the message or the evidence states or a direct consequence of the "
    "move itself -- nothing of your own, however plausible; nothing named above the "
    "disclosure level; the master's note used only where the facts carry it. Warm, direct, "
    "specific to this moment; no filler, no praise for its own sake. Address the player as "
    "'you' and the other side as 'your opponent'; never mix up whose move is whose. A "
    "yes-or-no question gets its yes or no first, then the fact behind it; never answer a "
    "question only with a question, and never ask the player to find a move you have just "
    "named. Plain standard chess words (King, rook, file, fork, pin); never 'engine', "
    "'evidence', 'verdict', 'expected score', 'level' or any hint that anything is hidden. "
    "The coach's rules, which the reply must keep:\n\n"
    + coach_prompt.RULES
    + "\n\nGive three candidate replies that differ in angle, each complete on its own. "
    'Answer in JSON only: {"replies": ["<reply 1>", "<reply 2>", "<reply 3>"]}'
)


def _form_words(item: dict) -> str:
    fm = item.get("form") or {}
    form = []
    if fm.get("max_sentences"):
        form.append(f"at most {fm['max_sentences']} sentences")
    if fm.get("ends_with_question") is True:
        form.append("ending with one question")
    if fm.get("ends_with_question") is False:
        form.append("no question")
    return ", ".join(form) or "two to four sentences"


_QUESTION_WORDS = (
    "Was",
    "Is",
    "Did",
    "Does",
    "Do",
    "Can",
    "Could",
    "Would",
    "Should",
    "What",
    "Which",
    "How",
    "Why",
    "Where",
)


def _check(item: dict, reply: str, draft_check) -> list[str]:
    """The checks a coach draft gets, plus the writer's own: no sheet jargon, no question
    in place of an answer."""
    det = deterministic(item, reply, draft_check)
    problems = det["leak"] + det["board"] + det["form"]["problems"]
    if jargon := JARGON.findall(reply):
        problems.append("the sheet's own words: " + ", ".join(sorted({j.lower() for j in jargon})))
    first = reply.strip().split(" ")[0].rstrip(",") if reply.strip() else ""
    if (
        item.get("question")
        and reply.count("?") == 1
        and reply.rstrip().endswith("?")
        and first in _QUESTION_WORDS
    ):
        problems.append("a question in place of an answer")
    return problems


def write_demo(
    item: dict, judgment: dict, judge, ledger: Ledger, draft_check, *, thinking: bool, writer
) -> tuple[str | None, list[str]]:
    """The reply that was right for this moment, or None with why. Three candidates from
    the writer, each checked and judged; the best that passes with every required fact
    and no unsupported claim; one retry with the problems; else nothing."""
    wrong = list(judgment["scores"].get("unsupported") or [])
    wrong += [
        f"left out {x['key']}"
        for x in (judgment.get("verdict") or {}).get("required", [])
        if isinstance(x, dict) and not x.get("mentioned")
    ]
    det = judgment.get("deterministic") or {}
    wrong += list(det.get("leak") or []) + list(det.get("board") or [])
    brief = (
        'The message the coach saw:\n"""\n'
        + demo_input(item)
        + '\n"""\n\n'
        + "The judge's evidence and required facts:\n"
        + evidence_block(item, with_reference=True)
        + "\n\n"
        + 'What the coach wrote:\n"'
        + item["reply"]["reply"]
        + '"\n'
        + "What was wrong with it: "
        + ("; ".join(str(w) for w in wrong) or "it did not pass")
        + "\n\n"
        + f"Form required: {_form_words(item)}."
    )
    note = ""
    problems_seen: list[str] = []
    for _attempt in range(2):
        t0 = time.time()
        try:
            completion = writer.chat(
                WRITER_SYSTEM,
                [
                    {
                        "role": "user",
                        "content": brief + note + "\n\nWrite the three replies. JSON only.",
                    }
                ],
                max_tokens=4000,
                temperature=0.4,
                thinking=False,
            )
            ledger.record(item["id"], completion.stats, time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            return None, [f"writer failed: {exc}"]
        parsed = parse_judgment(completion.text)
        candidates = (
            [str(r).strip() for r in (parsed or {}).get("replies", []) if str(r).strip()]
            if parsed
            else []
        )
        if not candidates:
            problems_seen = ["no replies in the writer's output"]
            note = "\n\nYour previous output had no replies."
            continue
        best: tuple[tuple, str] | None = None
        problems_seen = []
        for reply in candidates[:3]:
            problems = _check(item, reply, draft_check)
            if problems:
                problems_seen += problems
                continue
            try:
                verdict = judge_one(
                    item,
                    {"id": item["id"], "reply": reply},
                    judge,
                    ledger,
                    draft_check,
                    thinking=thinking,
                )
            except JudgeFailed as exc:
                problems_seen.append(f"judge failed: {exc}")
                continue
            sc = verdict["scores"]
            missing = [
                x["key"]
                for x in (verdict.get("verdict") or {}).get("required", [])
                if isinstance(x, dict) and not x.get("mentioned")
            ]
            if not sc["pass"] or sc.get("unsupported") or missing:
                problems_seen += (
                    list(sc.get("unsupported") or [])
                    + [f"left out {k}" for k in missing]
                    + ([] if sc["pass"] else ["did not pass the judge"])
                )
                continue
            key = (
                sc.get("completeness") or 0,
                sc.get("claim_support") or 0,
                sc.get("relevance") or 0,
                -len(reply),
            )
            if best is None or key > best[0]:
                best = (key, reply)
        if best is not None:
            return best[1], []
        withheld = sorted(
            {str(x).split(" ", 2)[-1] for x in problems_seen if str(x).startswith("withheld ")}
        )
        note = (
            "\n\nYour previous candidates were rejected: "
            + "; ".join(dict.fromkeys(str(x) for x in problems_seen))
            + ". Fix that, saying only what the message and the evidence state."
        )
        if withheld:
            note += " Do not use these words at all: " + ", ".join(f"'{w}'" for w in withheld) + "."
    return None, problems_seen


# A demonstration's input is the harness as the coach saw it, with the record cut to the
# last few plies: the older plies are one line each, but their dialogue is what a long
# game's harness is mostly made of, and a demonstration is about the moment.
DEMO_RECORD_PLIES = 6


SPLIT = ROOT / "data" / "eval" / "split.json"


def heldout_games() -> set[str]:
    """The games held out of training (split.json), as facts file stems."""
    if not SPLIT.exists():
        return set()
    return {
        g.removeprefix("masters/")
        for g in json.loads(SPLIT.read_text(encoding="utf-8")).get("heldout_games", [])
    }


def masters(
    *, heldout: bool = False, only: set[str] | None = None
) -> list[tuple[Path, dict[int, str]]]:
    """Every master game with the book's notes by ply: the training games, or the held-out;
    `only` restricts to those stems."""
    out = []
    held = heldout_games()
    for path in sorted((FACTS / "masters").glob("*.json")):
        if (path.stem in held) != heldout:
            continue
        if only is not None and path.stem not in only:
            continue
        book, stem = path.stem.split("__", 1)
        notes = MASTERS / book / f"{stem}.json"
        annotations = {}
        if notes.exists():
            m = json.loads(notes.read_text(encoding="utf-8"))
            annotations = {p["ply"]: p["annotation"] for p in m["plies"] if p.get("annotation")}
        out.append((path, annotations))
    return out


def run_games(
    icr: Path, total: int, heldout_share: float, seed: int
) -> tuple[list[str], list[str]]:
    """The run's games, chosen once and kept in games.json: `total` games, a `heldout_share`
    of them from the held-out split, the rest from the training split, both drawn at
    random with `seed`."""
    path = icr / "games.json"
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        return d["train"], d["heldout"]
    rng = random.Random(f"{seed}:games")
    train_all = [p.stem for p, _ in masters()]
    held_all = [p.stem for p, _ in masters(heldout=True)]
    rng.shuffle(train_all)
    rng.shuffle(held_all)
    n_held = max(1, round(total * heldout_share))
    held = sorted(held_all[:n_held])
    train = sorted(train_all[: total - len(held)])
    path.write_text(json.dumps({"train": train, "heldout": held}, indent=1), encoding="utf-8")
    return train, held


def demo_input(item: dict) -> str:
    """The harness at the moment, the record cut to its last plies, the task kept."""
    text = str(item["messages"][-1]["content"])
    if "# Game so far\n" not in text or "# Position now" not in text:
        return text
    head, rest = text.split("# Game so far\n", 1)
    record, tail = rest.split("# Position now", 1)
    lines = record.rstrip("\n").split("\n")
    starts = [i for i, line in enumerate(lines) if line[:1].isdigit()]
    if len(starts) > DEMO_RECORD_PLIES:
        cut = starts[-DEMO_RECORD_PLIES]
        kept = (
            lines[: starts[0]]
            + [f"(... {len(starts) - DEMO_RECORD_PLIES} earlier moves ...)"]
            + lines[cut:]
        )
        record = "\n".join(kept) + "\n\n"
    return f"{head}# Game so far\n{record}# Position now{tail}"


def condense(
    text: str, *, plies: int | None = None, drop: tuple[str, ...] = (), task_only: bool = False
) -> str:
    """A demonstration's input made leaner: the record cut to its last `plies`; the
    sections named in `drop` (by their '## ' title's first word, e.g. 'On', 'Engine',
    'The', 'Facts') removed from the position block; or, `task_only`, the setting, the
    last move, what was said here and the task alone."""
    if "# Game so far\n" not in text or "# Position now" not in text:
        return text
    head, rest = text.split("# Game so far\n", 1)
    record, tail = rest.split("# Position now", 1)
    position, _, task = tail.partition("# Task")
    if plies is not None:
        lines = record.rstrip("\n").split("\n")
        starts = [i for i, line in enumerate(lines) if line[:1].isdigit()]
        if len(starts) > plies:
            cut = starts[-plies]
            earlier = len(starts) - plies
            kept = (
                [x for x in lines[: starts[0]] if not x.startswith("(...")]
                + [f"(... {earlier} earlier moves ...)"]
                + lines[cut:]
            )
            record = "\n".join(kept) + "\n\n"
    sections = [x for x in position.split("\n## ") if x.strip()]
    titled = []
    for k, sec in enumerate(sections):
        title = sec.split("\n", 1)[0].strip()
        if k == 0 and not sec.startswith("## "):
            titled.append(("", sec))  # text before the first section, if any
            continue
        titled.append((title.removeprefix("## "), sec))
    keep_titles = {"Last move", "Said here"} if task_only else None
    out_sections = []
    for title, sec in titled:
        first = title.split(" ")[0] if title else ""
        if keep_titles is not None and title not in keep_titles:
            continue
        if first in drop:
            continue
        out_sections.append(sec if sec.startswith("## ") or not title else "## " + sec)
    position = "\n" + "\n## ".join(x.removeprefix("## ") for x in out_sections).strip("\n")
    position = "\n## " + position.lstrip("\n") if out_sections else ""
    if task_only:
        record = ""
        return f"{head.rstrip()}\n\n# Position now{position}\n\n# Task{task}"
    return f"{head}# Game so far\n{record}# Position now{position}\n\n# Task{task}"


DISTIL_SYSTEM = (
    "You distil what a chess coach should learn from corrected examples of its own replies. "
    "Each example gives the situation (the kind of question or moment, the disclosure level "
    "in force, the task the coach was given), what the coach wrote, what was wrong with it, "
    "and the reply that was right. Write the lessons, one per kind of situation, each in "
    "this shape on one line: 'When <situation>: <what the reply must state, taken from "
    'which part of the message, and what it must never do>. For example: "<a generic '
    "reply>\"'. The rule and the example must be GENERAL: they hold in any game, for any "
    "player, side, opening or position. In the example use placeholders in angle brackets "
    "-- <move>, <the opponent's move>, <piece>, <square>, <label>, <cost>, <opening name>, "
    "<count> -- never a real move, square, piece placement, opening name, player, side, "
    "score or position from the examples. At most one lesson per kind of situation, each "
    "under forty words. No preamble, no numbering of the examples: a plain list."
)

LESSONS_HEADER = (
    "Lessons from earlier games -- what a reply must do in each situation. Every fact in "
    "your reply comes from the game in front of you.\n"
)


def distil(pool, writer, ledger: Ledger) -> str:
    """The lessons the demonstrations teach, written once by the writer model."""
    parts = []
    for k, d in enumerate(pool.demos, start=1):
        situation = d["input"].rsplit("# Task", 1)[-1].strip()[:700]
        parts.append(
            f"Example {k} -- situation: {d['class']} at level {d['level']}. "
            f"Task given: {situation}\n"
            f'What the coach wrote: "{d.get("wrote", "")}"\n'
            f"What was wrong: "
            f"{'; '.join(str(x)[:120] for x in d.get('problems', [])) or 'it did not pass'}\n"
            f'The right reply: "{d["reply"]}"'
        )
    content = "\n\n".join(parts) + "\n\nWrite the lessons."
    t0 = time.time()
    completion = writer.chat(
        DISTIL_SYSTEM,
        [{"role": "user", "content": content}],
        max_tokens=1500,
        temperature=0.2,
        thinking=False,
    )
    ledger.record("distil", completion.stats, time.time() - t0)
    return (completion.text or "").strip()


def apply_condense(pool, spec: str) -> None:
    """`spec`: comma-separated knobs -- plies=N, drop=On+Engine+The, task -- rendered into
    each demonstration's lean_input, which render_demo prefers."""
    plies = None
    drop: tuple[str, ...] = ()
    task_only = False
    for knob in [x.strip() for x in spec.split(",") if x.strip()]:
        if knob.startswith("plies="):
            plies = int(knob.split("=", 1)[1])
        elif knob.startswith("drop="):
            drop = tuple(knob.split("=", 1)[1].split("+"))
        elif knob == "task":
            task_only = True
    for d in pool.demos:
        d["lean_input"] = condense(d["input"], plies=plies, drop=drop, task_only=task_only)


# The classes worth judging first, most instructive first: the ones where the coach most
# often fabricates or omits. Plain verdicts on sound moves come last and are sampled.
_JUDGE_FIRST = (
    "lesson",
    "ask_last_move",
    "ask_opponent_move",
    "ask_game_so_far",
    "ask_line",
    "ask_move",
    "ask_eval",
    "ask_threat",
    "ask_plan",
    "ask_named_move",
    "ask_opening",
    "ask_material",
    "ask_hint",
    "ask_free",
    "verdict_weaker",
    "stuck_hint",
    "decisive_question",
)


# A plain verdict on a sound move is drawn with this weight against 1 for every other
# class: it is half of all moments and the least often wrong.
VERDICT_WEIGHT = 0.3


def catalogue_of(items: list[dict]) -> dict[str, list[dict]]:
    """Every moment of every game, as (id, ply, class) by facts stem: what a round can
    sample from without playing the game first."""
    out: dict[str, list[dict]] = {}
    for it in items:
        stem = it["id"].split("/")[1] + "__" + it["id"].split("/")[2]
        out.setdefault(stem, []).append(
            {"id": it["id"], "ply": it["ply"], "class": it["class"], "game": it["game"]}
        )
    return out


def sample_moments(
    items: list[dict], per_game: int, seed: int, *, exclude: set[str] = frozenset()
) -> list[dict]:
    """`per_game` moments of every game to judge, drawn at random with `seed` (a different
    seed each round draws a different batch), weighted toward the classes that teach;
    never a moment already in C (`exclude`)."""
    if per_game <= 0:
        return [it for it in items if it["id"] not in exclude]
    by_game: dict[str, list[dict]] = {}
    for it in items:
        if it["id"] not in exclude:
            by_game.setdefault(it["game"], []).append(it)
    chosen: list[dict] = []
    for game, its in by_game.items():
        rng = random.Random(f"{seed}:sample:{game}")
        last = max(it["ply"] for it in its) or 1
        # One early, two in the middle, two late (for five): the game's plies in thirds.
        early = [it for it in its if it["ply"] <= last / 3]
        mid = [it for it in its if last / 3 < it["ply"] <= 2 * last / 3]
        late = [it for it in its if it["ply"] > 2 * last / 3]
        wanted = [round(per_game * 0.2), round(per_game * 0.4), 0]
        wanted[2] = per_game - wanted[0] - wanted[1]
        taken: list[dict] = []
        for bucket, n in zip((early, mid, late), wanted, strict=True):
            pool = [it for it in bucket if it not in taken]
            weights = [VERDICT_WEIGHT if it["class"] == "verdict" else 1.0 for it in pool]
            for _ in range(min(n, len(pool))):
                pick = rng.choices(range(len(pool)), weights=weights, k=1)[0]
                taken.append(pool.pop(pick))
                weights.pop(pick)
        # A thin bucket: fill from the rest of the game.
        rest = [it for it in its if it not in taken]
        weights = [VERDICT_WEIGHT if it["class"] == "verdict" else 1.0 for it in rest]
        while len(taken) < min(per_game, len(its)) and rest:
            pick = rng.choices(range(len(rest)), weights=weights, k=1)[0]
            taken.append(rest.pop(pick))
            weights.pop(pick)
        chosen += taken
    chosen.sort(key=lambda it: (it["game"], it["ply"], it["id"]))
    return chosen


def play_round(
    games: list[tuple[Path, dict[int, str]]],
    seed: int,
    system: str,
    provider,
    workers: int,
    out_dir: Path,
    *,
    summarise_every: int,
    summariser=None,
    stop_after: dict[str, int] | None = None,
    compact_record: bool = False,
) -> list[dict]:
    """Every game through the coach with the model answering; the items with their replies."""
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_cache: dict[str, list[dict]] = (
        json.loads(ENGINE_CACHE.read_text(encoding="utf-8")) if ENGINE_CACHE.exists() else {}
    )
    draft_check = _draft_check()
    items: list[dict] = []
    bar = tqdm(total=len(games), desc="play", unit="game", file=sys.stderr, mininterval=5)

    def one(path: Path, annotations: dict[int, str]) -> list[dict]:
        done = out_dir / f"{path.stem}.jsonl"
        if done.exists():
            return [json.loads(line) for line in done.open(encoding="utf-8")]
        engine = CachedEngine(engine_cache)
        t0 = time.time()
        try:
            rng = random.Random(f"{seed}:masters/{path.stem}")
            built = build_game(
                "masters",
                path,
                rng,
                annotations,
                None,
                engine,
                provider=provider,
                system=system,
                draft_check=draft_check,
                summarise_every=summarise_every,
                summariser=summariser,
                stop_after_ply=None if stop_after is None else stop_after.get(path.stem),
                compact_record=compact_record,
            )
        finally:
            engine.close()
        wall = round(time.time() - t0, 1)
        for it in built:
            it["game_wall_s"] = wall
        with done.open("w", encoding="utf-8") as f:
            for it in built:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        return built

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, path, ann): path for path, ann in games}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                items += fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  {path.stem} failed: {type(exc).__name__}: {exc}", flush=True)
            bar.update(1)
            bar.set_postfix(game=path.stem[:24])
    bar.close()
    ENGINE_CACHE.write_text(json.dumps(engine_cache, indent=0), encoding="utf-8")
    items.sort(key=lambda it: (it["game"], it["ply"], it["id"]))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="icr-play")
    ap.add_argument("--coach", default="openai:gpt-5.6-luna@medium")
    ap.add_argument("--judge", default="openai:gpt-5.6-luna@medium")
    ap.add_argument(
        "--writer",
        default="openai:gpt-5.6-luna@high",
        help="the model that writes the one right reply a round: one call, so high effort",
    )
    ap.add_argument(
        "--games",
        type=int,
        default=100,
        help="how many master games in the run, training and held-out together",
    )
    ap.add_argument(
        "--heldout-share",
        type=float,
        default=0.2,
        help="the share of --games held out for evaluation",
    )
    ap.add_argument(
        "--heldout",
        action="store_true",
        help=(
            "play the run's held-out games instead of its training games (an evaluation: "
            "no demonstrations are added)"
        ),
    )
    ap.add_argument("--game", action="append", help="only these games (facts file stems)")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument(
        "--moments-per-game",
        type=int,
        default=5,
        help=(
            "judge this many moments a game each round -- one early, two mid, two late for "
            "five -- a fresh draw every round (0 = every moment)"
        ),
    )
    ap.add_argument("--start-round", type=int, default=1)
    ap.add_argument(
        "--demos-per-round",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4, 5],
        help="the worst moments that become demonstrations each round, at most one per class",
    )
    ap.add_argument("--epsilon", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=5, help="games played at once")
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--budget-tokens", type=int, default=60_000_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--summarise-every", type=int, default=20)
    ap.add_argument(
        "--summariser",
        default="openai:gpt-5.6-luna@none",
        help="the model that folds the dialogue (a rewrite: low effort)",
    )
    ap.add_argument("--reset", action="store_true", help="empty C before round 1")
    ap.add_argument(
        "--pool-from",
        help="start from another run's C (its pool.json), e.g. to evaluate it on held-out games",
    )
    ap.add_argument(
        "--condense",
        default="",
        help="render C leaner: 'plies=3', 'drop=On+Engine+The', 'task' (comma-separated)",
    )
    ap.add_argument(
        "--one-per-class",
        action="store_true",
        help="keep only the latest demonstration of each class",
    )
    ap.add_argument(
        "--distil",
        action="store_true",
        help=(
            "C as lessons distilled from the demonstrations (plus --keep-demos of them, "
            "smallest form)"
        ),
    )
    ap.add_argument(
        "--keep-demos",
        type=int,
        default=0,
        help="with --distil: how many demonstrations to keep beside the lessons, in task-only form",
    )
    ap.add_argument(
        "--compact-record",
        action="store_true",
        help="the harness's record compacted: sound moves nobody spoke after as a move list",
    )
    ap.add_argument(
        "--tag", default="", help="a name for this evaluation's directory (held-out runs)"
    )
    args = ap.parse_args()

    icr = ROOT / "data" / "eval" / args.name
    icr.mkdir(parents=True, exist_ok=True)
    train_stems, held_stems = run_games(icr, args.games, args.heldout_share, args.seed)
    games = masters(heldout=args.heldout, only=set(held_stems if args.heldout else train_stems))
    if args.game:
        games = [g for g in games if g[0].stem in set(args.game)]
    print(
        f"{len(games)} {'held-out' if args.heldout else 'training'} games of {len(train_stems)}"
        f" + {len(held_stems)}: {', '.join(p.stem for p, _ in games)}",
        flush=True,
    )
    if args.heldout:
        # An evaluation: its own directory under the run, its own run names, the C it is
        # given (--pool-from <run>[:C-round-N] or the run's own C) and never a change to
        # the training pool.
        args.demos_per_round = 0
        tag = (
            (args.pool_from or "noC").replace("/", "_").replace(":", "-")
            + (("-" + re.sub(r"[^A-Za-z0-9]+", "_", args.condense)) if args.condense else "")
            + ("-1pc" if args.one_per_class else "")
            + (f"-distil{args.keep_demos}" if args.distil else "")
            + (("-" + args.tag) if args.tag else "")
        )
        out = icr / "heldout" / tag
        out.mkdir(parents=True, exist_ok=True)
        demos = []
        if args.pool_from:
            src, _, snapshot = args.pool_from.partition(":")
            src_dir = ROOT / "data" / "eval" / src
            demos = Pool.load(src_dir / "pool.json").demos
            if snapshot:  # C as it stood after round N: the demos of rounds <= N
                n = int(snapshot.rsplit("-", 1)[-1])
                demos = [d for d in demos if d["round"] <= n]
        if args.one_per_class:
            latest = {d["class"]: d for d in demos}
            demos = list(latest.values())
        pool = Pool(demos, out / "pool.json")
        if args.condense:
            apply_condense(pool, args.condense)
        pool.save()
        run_prefix = f"{args.name}-heldout-{tag}"
        icr = out
    else:
        pool = Pool.load(icr / "pool.json")
        if args.pool_from:
            pool = Pool(
                Pool.load(ROOT / "data" / "eval" / args.pool_from / "pool.json").demos,
                icr / "pool.json",
            )
            pool.save()
        elif args.reset:
            pool.demos = []
            pool.save()
        run_prefix = args.name
    judge, thinking = judge_provider(args.judge)
    ledger = Ledger(args.name, "icr", judge.model, args.budget_tokens)
    writer = (
        judge
        if args.writer == args.judge
        else Metered(judge_provider(args.writer)[0], ledger, "writer")
    )
    provider = coach_provider(args.coach, ledger)
    summariser = coach_provider(args.summariser, ledger) if args.summarise_every else None
    draft_check = _draft_check()
    demonstrated = {d["id"] for d in pool.demos}
    previous: dict[str, dict] | None = None
    lessons = ""
    if args.heldout and args.distil:
        lessons_path = icr / "lessons.md"
        if lessons_path.exists():
            lessons = lessons_path.read_text(encoding="utf-8")
        else:
            lessons = distil(pool, writer, ledger)
            lessons_path.write_text(lessons, encoding="utf-8")
        print(f"lessons (~{len(lessons) // 4} tokens):\n{lessons}\n", flush=True)
        keep = pool.demos[-args.keep_demos :] if args.keep_demos else []
        for d in keep:
            d["lean_input"] = condense(d["input"], task_only=True)
        pool.demos = keep
    for r in range(args.start_round, args.rounds + 1):
        run = f"{run_prefix}-r{r}"
        system = pool.system_text()
        if lessons:
            base = coach_prompt.system_message(with_corrections=False)
            system = (
                base
                + "\n\n"
                + LESSONS_HEADER
                + lessons
                + (("\n\n" + C_HEADER + pool.text()) if pool.demos else "")
            )
        print(
            f"\n== round {r}: C has {len(pool.demos)} demonstrations (~{pool.tokens()} tokens);"
            f" run {run}",
            flush=True,
        )
        t0 = time.time()
        catalogue_path = icr / "catalogue.json"
        catalogue = (
            json.loads(catalogue_path.read_text(encoding="utf-8"))
            if catalogue_path.exists()
            else None
        )
        if catalogue is None or args.heldout:
            # The first round plays every game in full: the baseline, and the catalogue of
            # every game's moments that later rounds sample from before playing.
            items = play_round(
                games,
                args.seed,
                system,
                provider,
                args.workers,
                icr / f"items-r{r}",
                summarise_every=args.summarise_every,
                summariser=summariser,
                compact_record=args.compact_record,
            )
            if not args.heldout:
                catalogue_path.write_text(json.dumps(catalogue_of(items)), encoding="utf-8")
            judged_items = sample_moments(
                [it for it in items if it.get("reply")],
                args.moments_per_game,
                args.seed * 1000 + r,
                exclude=demonstrated,
            )
        else:
            # A fresh draw of moments from every game, then each game played only as far
            # as its last drawn moment.
            wanted = sample_moments(
                [m for ms in catalogue.values() for m in ms],
                args.moments_per_game,
                args.seed * 1000 + r,
                exclude=demonstrated,
            )
            wanted_ids = {m["id"] for m in wanted}
            stop_after = {
                stem: max(m["ply"] for m in ms if m["id"] in wanted_ids)
                for stem, ms in catalogue.items()
                if any(m["id"] in wanted_ids for m in ms)
            }
            games_now = [g for g in games if g[0].stem in stop_after]
            items = play_round(
                games_now,
                args.seed,
                system,
                provider,
                args.workers,
                icr / f"items-r{r}",
                summarise_every=args.summarise_every,
                summariser=summariser,
                stop_after=stop_after,
                compact_record=args.compact_record,
            )
            judged_items = [it for it in items if it["id"] in wanted_ids and it.get("reply")]
            if len(judged_items) < len(wanted_ids):
                print(
                    f"  {len(wanted_ids) - len(judged_items)} drawn moments did not recur in"
                    " the replay",
                    flush=True,
                )
        replies = [it["reply"] for it in judged_items]
        fallbacks = sum(1 for it in items if it.get("reply") and it["reply"]["fallback_used"])
        print(
            f"  {len(items)} moments, {len(replies)} to judge, in {time.time() - t0:.0f}s;"
            f" fallbacks {fallbacks}",
            flush=True,
        )
        by_id = {it["id"]: it for it in items}
        judgments = judge_all(
            by_id,
            replies,
            run,
            judge,
            ledger,
            thinking=thinking,
            quiet=True,
            workers=args.judge_workers,
        )
        agg = aggregate(judgments)
        print(
            f"  scores: pass {agg['pass']:.2f} completeness {agg['completeness']:.2f}"
            f" correctness {agg['correctness']:.2f}"
            f" claim_support {(agg.get('claim_support') or 0):.2f}"
            f" fabrication {agg['fabrication']:.2f} leak {agg['leak']:.2f}"
            f" idea {agg['idea'] if agg['idea'] is not None else float('nan'):.2f}"
            f" fallback {agg['fallback']:.2f}",
            flush=True,
        )
        print(f"  ledger: {ledger.summary()}", flush=True)
        scores = {j["id"]: j["scores"] for j in judgments}
        converged = False
        if previous is not None:
            deltas = [
                abs((scores[k].get("correctness") or 0) - (previous[k].get("correctness") or 0))
                + abs((scores[k].get("completeness") or 0) - (previous[k].get("completeness") or 0))
                for k in scores
                if k in previous
            ]
            delta = sum(deltas) / len(deltas) if deltas else 0.0
            print(f"  change since last round (Eq. 5): {delta:.4f}", flush=True)
            converged = delta < args.epsilon
        previous = scores

        # The worst moments become demonstrations: the input as the coach saw it, the
        # reply that was right.
        # The worst first: the failure score (a fabrication, a leak or a false board claim
        # outranks an omission outranks a wrong fact), ties broken by the lowest average of
        # the judge's scores.
        def mean_score(j: dict) -> float:
            s = j["scores"]
            parts = [
                s.get("completeness"),
                s.get("correctness"),
                s.get("claim_support"),
                s.get("relevance"),
            ]
            parts = [x for x in parts if x is not None]
            return sum(parts) / len(parts) if parts else 1.0

        # Completeness first: the moment that left out the most of what it had to say is
        # the worst; the failure score (fabrications, leaks, form) breaks ties. Ranked by
        # the failure score alone, one unsupported claim outranked a reply that stated
        # nothing it should have, and completeness stayed at .82 while fabrication fell.
        def completeness_key(j: dict) -> tuple[float, float, float]:
            s = j["scores"]
            compl = s.get("completeness")
            return (1.0 if compl is None else compl, -failure_score(j), mean_score(j))

        ranked = sorted((j for j in judgments if j["id"] not in demonstrated), key=completeness_key)
        added: list[dict] = []
        unwritten: list[dict] = []
        classes_added: set[str] = set()
        for j in ranked:
            if len(added) >= args.demos_per_round:
                break
            if failure_score(j) <= 0:
                break
            if j["class"] in classes_added:
                continue  # one demonstration per class a round: three behaviours, not one
            item = by_id[j["id"]]
            reply, problems = write_demo(
                item, j, judge, ledger, draft_check, thinking=thinking, writer=writer
            )
            if reply is None:
                unwritten.append({"id": j["id"], "why": problems})
                continue
            demo = {
                "id": item["id"],
                "class": item["class"],
                "level": item["level"],
                "round": r,
                "score": failure_score(j),
                "input": demo_input(item),
                "wrote": item["reply"]["reply"],
                "reply": reply,
                "problems": [
                    *j["scores"].get("unsupported", []),
                    *[
                        f"left out {x['key']}"
                        for x in (j.get("verdict") or {}).get("required", [])
                        if isinstance(x, dict) and not x.get("mentioned")
                    ],
                ],
            }
            pool.demos.append(demo)
            demonstrated.add(item["id"])
            classes_added.add(item["class"])
            added.append(demo)
        pool.save()
        (icr / f"C-round-{r}.md").write_text(pool.text(), encoding="utf-8")
        gaps = [j["id"] for j in judgments if facts_gap(j)]
        (icr / f"round-{r}.json").write_text(
            json.dumps(
                {
                    "round": r,
                    "scores": agg,
                    "added": [d["id"] for d in added],
                    "unwritten": unwritten,
                    "facts_gaps": gaps,
                    "ledger": ledger.summary(),
                },
                indent=1,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(
            f"  C: +{len(added)} demonstrations ({len(unwritten)} could not be written)"
            f" -> {len(pool.demos)}; facts gaps {len(gaps)}",
            flush=True,
        )
        if converged:
            print("  converged", flush=True)
            break
        if ledger.over_budget():
            print("  over budget", flush=True)
            break


if __name__ == "__main__":
    main()
