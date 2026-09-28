"""ChessQA (Wen, Tang & Anderson, 2025; MIT, wieeii/ChessQA-Benchmark) on the coach's model,
with the position given in a chosen encoding, so one factor changes at a time:

  --context fen,pieces         the FEN line of the question, plus their piece list
  --context fen,pieces,diagram ... plus the diagram as the object draws it
  --context pieces,diagram     the FEN line removed from the question
  --context fen,pieces,legal   the paper's own prompt (their context: piece list + legal moves)
  --context fen,diagram,facts  the object as the coach sees it: the diagram, the engine's
                               lines and scores, what each move does, the threats, the facts;
                               for the Semantic tier the one-ply object with the verdict on
                               the move -- then the question

Blocks: fen (keep the question's FEN line), pieces (their piece arrangement line), legal
(their legal-move line), diagram (the object's board + whose move), facts (the object's
messages before the question). Semantic items carry no context placeholder in the dataset;
the blocks go in front of the question there.

Five tiers: Structural (read the board), Motifs, Short Tactics, Position Judgment, Semantic
(four human commentaries on a position and move, one true). The engine makes Short Tactics
and Position Judgment trivial with `facts`; they are reported, not claimed.

  uv run python tools/eval/chessqa.py facts  --per-tier 50           # Stockfish, cached
  uv run python tools/eval/chessqa.py run    --context fen,pieces --per-tier 50
      [--tiers Structural,Motifs]
  uv run python tools/eval/chessqa.py report

Items come from data/eval/chessqa/items.jsonl (the five parquet files, fetched row by row
from the Hugging Face datasets server; 3,500 rows).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chess
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chessqa_prompt_utils import PromptConfig, format_prompt, get_context  # noqa: E402
from judge import JudgeFailed, Ledger, _chat_with_backoff  # noqa: E402
from providers import provider_for  # noqa: E402

from sidecoach.analysis.annotator import GameAnnotator  # noqa: E402
from sidecoach.analysis.engine import Engine  # noqa: E402
from sidecoach.analysis.score import Thresholds  # noqa: E402
from sidecoach.cli import _load_book, _load_tagger  # noqa: E402
from sidecoach.coach.conversation import Conversation, Entry, PlyRecord  # noqa: E402
from sidecoach.coach.facts import ascii_board, facts_from_analysis  # noqa: E402
from sidecoach.coach.named import analyse_named  # noqa: E402
from sidecoach.config import load_settings, require_stockfish  # noqa: E402
from sidecoach.events import (  # noqa: E402
    Analysis,
    DisclosureLevel,
    GameStarted,
    MoveAssessment,
    MovePlayed,
    Side,
)

DIR = ROOT / "data" / "eval" / "chessqa"
ITEMS = DIR / "items.jsonl"
FACTS = DIR / "facts.jsonl"
RUNS = DIR / "runs"
BLOCKS = ("fen", "pieces", "legal", "diagram", "facts", "direct")
TIERS = ("Structural", "Motifs", "Short Tactics", "Position Judgement", "Semantic")

BENCH_SYSTEM = (
    "You are a strong chess player answering benchmark questions. Reason briefly, then "
    "finish with one line 'FINAL ANSWER: <answer>' in exactly the format the question asks "
    "for, and nothing after it."
)
HARNESS_SYSTEM = (
    BENCH_SYSTEM
    + " The messages before the question carry the position and the engine's word on it, "
    "computed on the board: the board, the moves, the standing (Stockfish's evaluation), "
    "the engine's lines with their scores, what each move does, the threats, the facts of "
    "the position, and 'On the board now', a complete list of what hangs, what is pinned, "
    "the double attacks, skewers and batteries, and the checks and captures on offer. They "
    "are true and complete. Answer from them wherever they bear on the question: a pin, "
    "fork, battery or check not listed there is not on the board, and the evaluation is "
    "the standing's, not your own."
)
# The question without its call for step-by-step analysis: the coach's operating point.
_STEP_BY_STEP = "Analyze step by step and explain your reasoning.\n"

_MOVE_IN_QUESTION = re.compile(r"makes the move: (\S+)")
# "in FEN: <placement> <side> <castling> <ep>[ <half> <full>]" on the question's first line.
_FEN_IN_QUESTION = re.compile(r" in FEN: \S+ [wb] \S+ \S+(?: \d+ \d+)?")
_FINAL = re.compile(r"FINAL ANSWER:\s*(.+?)\s*(?:\n|$)", re.IGNORECASE)


# ---- items --------------------------------------------------------------------------


def load_items() -> list[dict[str, Any]]:
    return [json.loads(line) for line in ITEMS.read_text(encoding="utf-8").splitlines() if line]


def fen_of(item: dict[str, Any]) -> str:
    """The position; state-tracking items carry '<fen> | <moves>'. Some FENs lack the
    clocks, which python-chess supplies."""
    return chess.Board(item["input"].split("|", 1)[0].strip()).fen()


def move_of(item: dict[str, Any]) -> str | None:
    """The move a Semantic item is about (UCI), or None."""
    m = _MOVE_IN_QUESTION.search(item["question"])
    return m.group(1) if m else None


def sample(items: list[dict[str, Any]], per_tier: int | None, seed: int) -> list[dict[str, Any]]:
    """`per_tier` items from each tier, spread evenly over its task types, in a fixed order."""
    if per_tier is None:
        return items
    rng = random.Random(seed)
    by_type: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for it in items:
        by_type[it["task_category"]][it["task_type"]].append(it)
    out: list[dict[str, Any]] = []
    for tier in TIERS:
        pools = [list(v) for _, v in sorted(by_type[tier].items())]
        for pool in pools:
            rng.shuffle(pool)
        picked: list[dict[str, Any]] = []
        while len(picked) < per_tier and any(pools):
            for pool in pools:
                if pool and len(picked) < per_tier:
                    picked.append(pool.pop())
        out.extend(picked)
    return out


# ---- facts: Stockfish on each position, cached -------------------------------------------


def load_facts() -> dict[str, dict[str, Any]]:
    if not FACTS.exists():
        return {}
    out = {}
    for line in FACTS.read_text(encoding="utf-8").splitlines():
        if line:
            rec = json.loads(line)
            out[rec["task_id"]] = rec
    return out


def compute_facts(items: list[dict[str, Any]], *, depth: int, seconds: float, threads: int) -> None:
    have = load_facts()
    todo = [it for it in items if it["task_id"] not in have]
    print(f"{len(todo)} positions to analyse ({len(have)} cached)", flush=True)
    if not todo:
        return
    settings = load_settings()
    engine = Engine(
        path=require_stockfish(settings),
        threads=threads,
        hash_mb=1024,
        multipv=6,
        max_seconds=seconds,
    )
    ann = GameAnnotator(
        engine=engine,
        depth=depth,
        multipv=6,
        thresholds=Thresholds.from_settings(settings.coach),
        tagger=_load_tagger(),
        book=_load_book(settings),
        with_threats=True,
    )
    t0 = time.time()
    try:
        with FACTS.open("a", encoding="utf-8") as f:
            for n, it in enumerate(todo, 1):
                ann.reset()
                board = chess.Board(fen_of(it))
                gid = it["task_id"]
                rec: dict[str, Any] = {
                    "task_id": gid,
                    "fen": board.fen(),
                    "analysis_before": json.loads(
                        ann.analysis_event(gid, board, 0).model_dump_json()
                    ),
                }
                uci = move_of(it)
                if uci:
                    move = chess.Move.from_uci(uci)
                    assessment, after = ann.assess(gid, board.copy(), move, 1)
                    rec["move"] = {"uci": uci, "san": board.san(move)}
                    rec["assessment"] = json.loads(assessment.model_dump_json())
                    rec["analysis_after"] = json.loads(after.model_dump_json())
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if n % 25 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    finally:
        engine.close()


# ---- the three prompts ---------------------------------------------------------------------


def context_block(item: dict[str, Any], blocks: set[str]) -> str:
    """What replaces the question's context placeholder: the chosen blocks, in a fixed
    order (diagram, piece list, legal moves), each as its source writes it."""
    board = chess.Board(fen_of(item))
    parts: list[str] = []
    if "diagram" in blocks:
        side = "White" if board.turn == chess.WHITE else "Black"
        parts.append(f"{ascii_board(board)}\n{side} to move.")
    theirs = get_context(item["input"]).strip().splitlines() if blocks & {"pieces", "legal"} else []
    for line in theirs:
        if line.startswith("Piece arrangement:") and "pieces" in blocks:
            parts.append(line)
        if line.startswith("Legal moves:") and "legal" in blocks:
            parts.append(line)
    return ("\n".join(parts) + "\n\n") if parts else ""


def question_text(item: dict[str, Any], blocks: set[str]) -> str:
    """The dataset's question with its placeholders filled: the context as `blocks` say,
    the FEN line kept or removed (a state-tracking item keeps its initial FEN: the moves
    need it), and their format example."""
    context = context_block(item, blocks - {"facts"})
    if "CONTEXT_PLACEHOLDER" in item["question"]:
        head, _, tail = item["question"].partition("CONTEXT_PLACEHOLDER")
        q = format_prompt(
            {**item, "question": head + context + tail},
            PromptConfig(add_context=False, format_example_index=0),
        )
    else:
        q = context + format_prompt(item, PromptConfig(add_context=False, format_example_index=0))
    if "fen" not in blocks:
        q = _FEN_IN_QUESTION.sub("", q, count=1)
    return q


def messages_for(
    blocks: set[str], item: dict[str, Any], facts: dict[str, Any] | None
) -> tuple[str, list[dict[str, str]]]:
    question = question_text(item, blocks - {"direct"})
    if "direct" in blocks:
        question = question.replace(_STEP_BY_STEP, "Answer from the facts given, briefly.\n")
    if "facts" not in blocks:
        return BENCH_SYSTEM, [{"role": "user", "content": question}]
    assert facts is not None
    # The coach's examination of what the question names -- a piece on a square -- goes
    # ahead of the question, as the directive carries it live.
    board = chess.Board(facts["fen"])
    examined = analyse_named(
        item["question"],
        board,
        None,
        user_colour=board.turn,
        level=DisclosureLevel.LINE,
        static=True,
    )
    head = ("\n".join(examined) + "\n\n") if examined else ""
    return HARNESS_SYSTEM, [
        *harness_messages(item, facts),
        {"role": "user", "content": head + question},
    ]


def harness_messages(item: dict[str, Any], facts: dict[str, Any]) -> list[dict[str, Any]]:
    """The object for the item: a game joined at the position, the side to move as the
    player; for a Semantic item, the move played and the engine's verdict on it, as the
    coach's conversation renders a ply. Everything at LINE: a benchmark withholds nothing.
    The diagrams the object draws are replaced by FEN lines: the position is given in the
    question in the encoding under test, and the ladder showed the diagram costs this model."""
    out = _harness_messages(item, facts)
    for fen in {facts["fen"], *([_after_fen(facts)] if facts.get("move") else [])}:
        drawn = ascii_board(chess.Board(fen))
        out = [{**m, "content": m["content"].replace(drawn, f"Position (FEN): {fen}")} for m in out]
    return out


def _after_fen(facts: dict[str, Any]) -> str:
    after = chess.Board(facts["fen"])
    after.push_uci(facts["move"]["uci"])
    return after.fen()


def _harness_messages(item: dict[str, Any], facts: dict[str, Any]) -> list[dict[str, Any]]:
    board = chess.Board(facts["fen"])
    user_side = Side.WHITE if board.turn == chess.WHITE else Side.BLACK
    user_colour = board.turn
    gid = item["task_id"]
    convo = Conversation()
    convo.new_game(GameStarted(game_id=gid, user_side=user_side, start_fen=board.fen()))
    convo.level = DisclosureLevel.LINE
    if facts.get("move"):
        after = board.copy()
        after.push_uci(facts["move"]["uci"])
        played = MovePlayed(
            game_id=gid,
            ply=1,
            san=facts["move"]["san"],
            uci=facts["move"]["uci"],
            fen_before=board.fen(),
            fen_after=after.fen(),
            by_user=True,
        )
        convo.move(played)
        convo.assess(MoveAssessment.model_validate(facts["assessment"]))
        analysis = Analysis.model_validate(facts["analysis_after"])
        convo.analysed(
            analysis, facts_from_analysis(after, analysis, last_move=played.san, user=user_colour)
        )
        return [dict(m) for m in convo.messages()]
    # No move: the facts of the position the player faces, rendered by the same pieces of
    # the ply message (the lines block and the grounding line), under a placeholder record.
    analysis = Analysis.model_validate(facts["analysis_before"])
    record = PlyRecord(
        move=MovePlayed(
            game_id=gid,
            ply=0,
            san="-",
            uci="0000",
            fen_before=board.fen(),
            fen_after=board.fen(),
            by_user=False,
        )
    )
    record.analysis = analysis
    record.facts = facts_from_analysis(board, analysis, user=user_colour)
    lines = [
        f"Moves so far: (none)\n{ascii_board(board)}",
        convo._turn_text(board).capitalize() + ".",
        *(
            f"## {title}\n{body}"
            for title, body in convo._position_sections(
                record, board, user_colour, DisclosureLevel.LINE
            )
        ),
    ]
    convo.entries.append(Entry("user", "game", "\n".join(x for x in lines if x)))
    return [dict(m) for m in convo.messages()]


# ---- answers and scoring --------------------------------------------------------------------


def final_answer(text: str) -> str | None:
    found = _FINAL.findall(text or "")
    if not found:
        return None
    return found[-1].strip().strip("*").strip("`").strip()


def _tokens(answer: str) -> set[str]:
    return {t.strip().strip("'\"").lower() for t in answer.split(",") if t.strip()}


def correct(item: dict[str, Any], answer: str | None) -> bool:
    if answer is None:
        return False
    truth = str(item["correct_answer"]).strip()
    if item["answer_type"] == "multi":
        return _tokens(answer) == _tokens(truth)
    if item["task_category"] == "Semantic":
        m = re.match(r"\(?([A-Da-d])\)?", answer.strip())
        return bool(m) and m.group(1).upper() == truth.upper()
    if item["task_category"] == "Position Judgement":
        try:
            return int(answer.replace("+", "").replace(",", "").strip()) == int(truth)
        except ValueError:
            return False
    return " ".join(answer.split()) == " ".join(truth.split())


# ---- run -----------------------------------------------------------------------------------


def run(
    blocks: set[str],
    items: list[dict[str, Any]],
    *,
    model: str,
    workers: int,
    budget: int,
    max_tokens: int,
    force: bool,
    thinking: bool = False,
    local_model: str | None = None,
    tag: str | None = None,
    effort: str | None = None,
    openai_model: str | None = None,
) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    config = "+".join(b for b in BLOCKS if b in blocks)
    out_path = RUNS / (
        f"{config}-{tag or model}{'-thinking' if thinking else ''}"
        f"{'-' + effort if effort else ''}.jsonl"
    )
    done: set[str] = set()
    if out_path.exists() and not force:
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line:
                done.add(json.loads(line)["task_id"])
    todo = [it for it in items if it["task_id"] not in done]
    facts = load_facts() if "facts" in blocks else {}
    if "facts" in blocks:
        missing = [it["task_id"] for it in todo if it["task_id"] not in facts]
        if missing:
            raise SystemExit(
                f"{len(missing)} items without facts; run `facts` first (e.g. {missing[0]})"
            )
    print(f"{config}: {len(todo)} items to answer on {model} ({len(done)} done)", flush=True)
    if not todo:
        return
    ledger = Ledger(f"chessqa-{config}", "chessqa", tag or model, budget)
    # Raw providers: every call below goes through _chat_with_backoff, which checks the
    # ledger's budget first and records the call after.
    if model == "openai":
        spec = f"openai:{openai_model or 'gpt-5.6-luna'}@{effort or ''}"
        provider = provider_for(spec, timeout_s=300)
    elif model == "local" and local_model:
        from sidecoach.config import LLMSettings
        from sidecoach.llm.factory import make_provider

        provider = make_provider(LLMSettings(provider="local", model=local_model), timeout_s=600)
    else:
        provider = provider_for(model, timeout_s=300)
        if effort and hasattr(provider, "extra_body"):
            # DashScope passes reasoning_effort into the Qwen3.8 template (low / medium / xhigh).
            setattr(  # noqa: B010 -- a provider attribute the type does not declare
                provider,
                "extra_body",
                {**getattr(provider, "extra_body", {}), "reasoning_effort": effort},
            )
    lock = threading.Lock()
    n_ok = n_done = 0

    def one(it: dict[str, Any]) -> dict[str, Any]:
        system, messages = messages_for(blocks, it, facts.get(it["task_id"]))
        completion = _chat_with_backoff(
            provider,
            system,
            messages,
            ledger,
            it["task_id"],
            max_tokens=max_tokens,
            temperature=0.1,
            thinking=thinking,
        )
        text = completion.text or ""
        answer = final_answer(text)
        return {
            "task_id": it["task_id"],
            "task_category": it["task_category"],
            "task_type": it["task_type"],
            "config": config,
            "model": model,
            "thinking": thinking,
            "answer": answer,
            "correct": correct(it, answer),
            "prompt_tokens": int((completion.stats or {}).get("prompt_tokens") or 0),
            "completion_tokens": int((completion.stats or {}).get("completion_tokens") or 0),
            "raw": text,
        }

    bar = tqdm(
        total=len(todo),
        desc=f"chessqa {config} {tag or model}",
        unit="item",
        file=sys.stderr,
        mininterval=5,
    )
    with out_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, it): it for it in todo}
        for fut in as_completed(futures):
            bar.update(1)
            try:
                rec = fut.result()
            except JudgeFailed as exc:
                print(f"  failed: {exc}", flush=True)
                continue
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_done += 1
                n_ok += rec["correct"]
                bar.set_postfix(acc=f"{n_ok / n_done:.2f}", tokens=f"{ledger.total:,}")
    bar.close()
    print(
        f"{config}: {n_ok}/{n_done} correct; {ledger.calls} calls, {ledger.prompt:,} prompt"
        f" + {ledger.completion:,} completion tokens",
        flush=True,
    )


# ---- report --------------------------------------------------------------------------------


def report() -> None:
    runs: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(RUNS.glob("*.jsonl")):
        runs[path.stem] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
    if not runs:
        print("no runs")
        return
    names = list(runs)
    lines = [
        "# ChessQA on the coach's model",
        "",
        "Accuracy (exact match; set match for multi-answer items).",
        "",
    ]

    def table(title: str, key: str, order: list[str]) -> None:
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| " + key + " | " + " | ".join(names) + " |")
        lines.append("|---|" + "---|" * len(names))
        groups: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
        for name, recs in runs.items():
            by: dict[str, list[bool]] = defaultdict(list)
            for r in recs:
                by[r[key]].append(bool(r["correct"]))
            for g, vals in by.items():
                groups[g][name] = (sum(vals), len(vals))
        for g in order or sorted(groups):
            if g not in groups:
                continue
            cells = []
            for name in names:
                if name in groups[g]:
                    ok, n = groups[g][name]
                    cells.append(f"{ok / n:.2f} ({n})")
                else:
                    cells.append("-")
            lines.append(f"| {g} | " + " | ".join(cells) + " |")
        lines.append("")

    # Overall first, as its own row.
    lines.append("### Overall")
    lines.append("")
    lines.append("| | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    cells = []
    for name in names:
        recs = runs[name]
        cells.append(
            f"{sum(bool(r['correct']) for r in recs) / len(recs):.2f} ({len(recs)})"
            if recs
            else "-"
        )
    lines.append("| all | " + " | ".join(cells) + " |")
    cells = []
    for name in names:
        recs = runs[name]
        cells.append(
            f"{sum(r['prompt_tokens'] for r in recs) / max(1, len(recs)):,.0f}"
            f" + {sum(r['completion_tokens'] for r in recs) / max(1, len(recs)):,.0f}"
        )
    lines.append("| tokens per item (prompt + completion) | " + " | ".join(cells) + " |")
    cells = []
    for name in names:
        recs = runs[name]
        cells.append(f"{sum(r['answer'] is None for r in recs) / max(1, len(recs)):.2f}")
    lines.append("| no FINAL ANSWER | " + " | ".join(cells) + " |")
    lines.append("")
    table("By tier", "task_category", list(TIERS))
    table("By task", "task_type", [])
    text = "\n".join(lines)
    (DIR / "report.md").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("facts")
    f.add_argument("--per-tier", type=int)
    f.add_argument("--seed", type=int, default=7)
    f.add_argument("--depth", type=int, default=14)
    f.add_argument("--seconds", type=float, default=2.0)
    f.add_argument("--threads", type=int, default=10)
    r = sub.add_parser("run")
    r.add_argument("--context", required=True, help="comma-separated blocks: " + ",".join(BLOCKS))
    r.add_argument("--tiers", help="comma-separated tiers to run (default all)")
    r.add_argument("--per-tier", type=int)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--model", choices=["flash", "dashscope", "local", "openai"], default="flash")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--budget-tokens", type=int, default=30_000_000)
    r.add_argument("--max-tokens", type=int, default=4096)
    r.add_argument("--force", action="store_true")
    r.add_argument(
        "--thinking",
        action="store_true",
        help="the model's thinking mode on (DashScope enable_thinking)",
    )
    r.add_argument(
        "--local-model", help="a model id on the local server, instead of the configured one"
    )
    r.add_argument("--tag", help="a short name for the run file, instead of the model kind")
    r.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh"],
        help="reasoning effort for thinking mode (DashScope / OpenAI)",
    )
    r.add_argument(
        "--openai-model", help="the OpenAI model id for --model openai (default gpt-5.6-luna)"
    )
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "report":
        report()
        return
    items = sample(load_items(), args.per_tier, args.seed)
    if args.cmd == "facts":
        compute_facts(items, depth=args.depth, seconds=args.seconds, threads=args.threads)
    else:
        blocks = {b.strip() for b in args.context.split(",") if b.strip()}
        unknown = blocks - set(BLOCKS)
        if unknown:
            raise SystemExit(f"unknown blocks: {sorted(unknown)}")
        if args.tiers:
            wanted = {t.strip() for t in args.tiers.split(",")}
            items = [it for it in items if it["task_category"] in wanted]
        run(
            blocks,
            items,
            model=args.model,
            workers=args.workers,
            budget=args.budget_tokens,
            max_tokens=args.max_tokens,
            force=args.force,
            thinking=args.thinking,
            local_model=args.local_model,
            tag=args.tag,
            effort=args.effort,
            openai_model=args.openai_model,
        )


if __name__ == "__main__":
    main()
