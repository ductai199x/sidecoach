"""The model's review of games' key moves (coach/review_words.py), for reading by hand: every key
move with its facts' gist and the model's paragraphs, the dropped ones with why, and what each
call cost -- the words made right before they are made short.

    uv run --no-sync python tools/review/ai_review.py DB [--game KEY | --games N] \
        [--model local] [--budget USD] [--out FILE]

DB: a profile.sqlite with reviews in it; the games whose player is known, the latest first.
`--model`: a provider spec (llm.factory.provider_from_spec). `local` is free; any other spends,
through `judge.Ledger` under the run `ai-review` with `--budget` dollars (the spend already on
the record counting against it).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import chess

from sidecoach.coach import review_words
from sidecoach.config import load_settings
from sidecoach.events import ReviewedGame
from sidecoach.llm.factory import provider_from_spec
from sidecoach.llm.usage import KNOWN_PRICES

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from judge import BudgetExceeded, Ledger  # noqa: E402

PRICES = KNOWN_PRICES["gpt-5.6-luna"]


def games(db: Path, key: str | None, count: int) -> list[tuple[str, ReviewedGame, str | None]]:
    """The reviews to write about: the one named, else the latest `count` whose player is known."""
    con = sqlite3.connect(db)
    query = (
        "SELECT r.key, r.report, l.user_side FROM reviews r JOIN library l ON l.key = r.key"
        + (" WHERE r.key = ?" if key else " WHERE l.user_side IS NOT NULL")
        + " ORDER BY l.played_at DESC"
    )
    rows = con.execute(query, (key,) if key else ()).fetchall()[: 1 if key else count]
    return [(k, ReviewedGame.model_validate_json(report), user) for k, report, user in rows]


def reading(game: ReviewedGame, user: str | None, words: review_words.ReviewWords) -> list[str]:
    """The review as lines to read: each key move's gist, then its paragraphs as the student
    reads them (in the model's colours where not shown)."""
    lines = []
    for ply in review_words.key_plies(game):
        move = game.moves[ply - 1]
        facts = review_words.move_facts(game, ply, user)
        number = f"{(ply + 1) // 2}{'.' if ply % 2 else '...'}"
        chances = next(v for k, v in facts.items() if k.endswith("winning chances"))
        head = f"{number} {move.san} ({facts['kind']}; {chances}; {facts['played by']})"
        best = move.assessment.best[0] if move.assessment.best else None
        if best:
            head += f"  best {chess.Board(move.fen_before).san(chess.Move.from_uci(best))}"
        reply = next((v for k, v in facts.items() if k.endswith("best reply")), None)
        if reply:
            head += f"  reply {reply}"
        lines.append(head)
        said = review_words.parse(words.answers.get(ply, ""))
        if ply in words.dropped:
            lines.append(f"  [dropped: {words.dropped[ply]}]")
        elif ply not in words.texts:
            lines.append("  [no paragraph]")
        for part in review_words.PARTS:
            if part in said:
                shown = words.texts.get(ply, {}).get(part)
                note = "" if shown else " (not shown)"
                if shown and part == "why" and ply in words.reveals:
                    note = " (shown with the answer)"
                lines.append(f"  {part}{note}: {shown or said[part]}")
        lines.append("")
    tokens = {
        key: sum(int(call.get(key) or 0) for call in words.calls)
        for key in ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens")
    }
    slowest = max((call.get("elapsed_s") or 0 for call in words.calls), default=0)
    lines.append(
        f"model {words.model}; {len(words.calls)} calls, the slowest {slowest:.0f} s; "
        f"tokens {json.dumps(tokens)}"
    )
    if words.cost is not None:
        lines.append(f"at gpt-5.6-luna's prices: ${words.cost:.4f}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=Path)
    parser.add_argument("--game", help="one review's key")
    parser.add_argument("--games", type=int, default=1, help="else the latest so many")
    parser.add_argument("--model", default="local")
    parser.add_argument("--budget", type=float, default=0.0, help="dollars, for a paid model")
    parser.add_argument("--think", action="store_true", help="let the model reason first")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    load_settings()  # the keys, from .env or the credential store
    provider = provider_from_spec(args.model, timeout_s=900)
    paid = args.model != "local"
    ledger = (
        Ledger(
            "ai-review",
            "AI review of key moves, read by hand",
            provider.model,
            budget_usd=args.budget,
            prices=(PRICES.prompt or 0.0, PRICES.cached or 0.0, PRICES.completion or 0.0),
            resume=True,
        )
        if paid
        else None
    )
    out: list[str] = []
    answers: list[str] = []  # the model's own words, in colours: FILE's .answers.txt
    for key, game, user in games(args.db, args.game, args.games):
        opening = next((m.after.opening for m in reversed(game.moves) if m.after.opening), None)
        out.append(f"==== {key}: {len(game.moves)} plies, you play {user}")
        try:
            if ledger is not None:
                ledger.check()
            began = time.monotonic()
            words = review_words.write(
                game, provider, user=user, opening=opening, prices=PRICES, thinking=args.think
            )
            if ledger is not None:
                for call in words.calls:
                    ledger.record(key, call, time.monotonic() - began, model=provider.model)
        except BudgetExceeded as stop:
            out.append(f"stopped: {stop}")
            break
        out.extend(reading(game, user, words))
        out.append("")
        answers.extend(f"==== {key} ply {ply}\n{text}\n" for ply, text in words.answers.items())
        print("\n".join(out[-3:]), flush=True)
    if ledger is not None:
        out.append(f"ledger: {ledger.summary()}")
    text = "\n".join(out)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        args.out.with_suffix(".answers.txt").write_text("\n".join(answers), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
