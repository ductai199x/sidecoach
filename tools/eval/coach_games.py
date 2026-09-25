"""Coach every recorded game from its PGN, faithfully, and log every message for judging.

Drives the real Coach with the real LLM, engine and tagger off the truth.pgn (no vision),
emitting the same events the live pipeline does. The user is White (nosegazm) in every
recording. Output: for each coach message, the ply, the move in play, the trigger, the
disclosure level, the text, and whether any check flagged the draft.

The coach is the app's (coach_build.py; no engine for the moves a player names). A model
other than none is only called through the ledger, checked against --budget-usd first.

Run:  uv run python tools/eval/coach_games.py [game-label] --budget-usd 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coach_build import build_coach  # noqa: E402
from judge import BudgetExceeded, Ledger  # noqa: E402
from providers import Metered, spec_of  # noqa: E402

from sidecoach.analysis.annotator import GameAnnotator  # noqa: E402
from sidecoach.analysis.engine import Engine  # noqa: E402
from sidecoach.analysis.score import Thresholds  # noqa: E402
from sidecoach.cli import _load_book, _load_tagger  # noqa: E402
from sidecoach.config import load_settings, require_stockfish  # noqa: E402
from sidecoach.events import (  # noqa: E402
    GameStarted,
    MovePlayed,
    Side,
)
from sidecoach.llm.factory import make_provider  # noqa: E402

DEPTH = 12
CAP = 2.0


def metered_provider(settings, budget_usd: float | None, run: str):
    """The app's model on a ledger (None when it is the offline coach)."""
    provider = make_provider(settings.llm)
    if getattr(provider, "offline", False):
        return provider, None
    if budget_usd is None:
        raise SystemExit(f"{provider.name} costs money: give --budget-usd")
    ledger = Ledger(f"{run}-{int(time.time())}", run, spec_of(provider), budget_usd=budget_usd)
    return Metered(provider, ledger), ledger


def coach_one(path: Path, engine: Engine, provider, settings, label: str) -> None:
    ann = GameAnnotator(
        engine=engine,
        depth=DEPTH,
        multipv=settings.engine.multipv,
        thresholds=Thresholds.from_settings(settings.coach),
        tagger=_load_tagger(),
        book=_load_book(settings),
        with_threats=True,
    )
    with path.open(encoding="utf-8") as fh:
        game = chess.pgn.read_game(fh)
    assert game is not None, f"{path}: no game in the PGN"
    board = game.board()
    gid = label
    coach = build_coach(provider, settings=settings)
    # facts per ply, for context in the report
    facts_by_ply: dict[int, object] = {}

    def drain() -> None:
        coach.speaker.run_pending()
        for msg in coach.poll():
            rec = coach.spoken[-1]
            if msg.text.endswith(": good.") and not rec.problems:
                continue  # a terse nod, not a teaching moment; counted in the summary
            f = facts_by_ply.get(msg.ply or 0)
            plan = getattr(f, "plan", None)
            theme = getattr(f, "theme", None)
            ideas = getattr(f, "ideas", None)
            flags = ""
            if rec.problems:
                flags = f"  [CHECK CAUGHT: {rec.problems} | attempts={rec.attempts}]"
            print(f"\n[{label} ply {msg.ply} | {msg.trigger.value} | {msg.level.name}]{flags}")
            if theme or plan:
                print(f"    facts: theme={theme!r} plan={plan!r}")
            if ideas:
                print(f"    ideas: {ideas}")
            print(f"    COACH: {msg.text}")

    coach.on_event(GameStarted(game_id=gid, user_side=Side.WHITE, start_fen=board.fen()), 0.0)
    a0 = ann.analysis_event(gid, board.copy(), 0)
    facts_by_ply[0] = coach.positions[0].facts if 0 in coach.positions else None
    coach.on_event(a0, 0.0)
    facts_by_ply[0] = coach.positions[0].facts
    drain()

    ts = 0.0
    for ply, move in enumerate(game.mainline_moves()):
        by_user = board.turn == chess.WHITE
        san = board.san(move)
        uci = move.uci()
        fen_before = board.fen()
        assessment, analysis = ann.assess(gid, board.copy(), move, ply + 1)
        board.push(move)
        ts += 20.0
        coach.on_event(
            MovePlayed(
                game_id=gid,
                ply=ply + 1,
                san=san,
                uci=uci,
                fen_before=fen_before,
                fen_after=board.fen(),
                by_user=by_user,
            ),
            ts,
        )
        # The live pipeline's order: the verdict before the analysis of the position after.
        if by_user:
            coach.on_event(assessment, ts)
        coach.on_event(analysis, ts)
        if (ply + 1) in coach.positions:
            facts_by_ply[ply + 1] = coach.positions[ply + 1].facts
        drain()

    s = coach.speaker
    print(
        f"\n=== {label}: {len(coach.spoken)} messages, "
        f"{sum(1 for x in coach.spoken if x.problems)} drafts caught, "
        f"{s.fallbacks} templates used ==="
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="?", help="only the recordings whose label has this")
    ap.add_argument("--budget-usd", type=float, help="required for any model but none")
    args = ap.parse_args()
    settings = load_settings()
    provider, ledger = metered_provider(settings, args.budget_usd, "coach_games")
    engine = Engine(
        path=require_stockfish(settings), threads=8, hash_mb=512, multipv=4, max_seconds=CAP
    )
    games = sorted(Path("data/recordings").glob("g*/truth.pgn"))
    try:
        with engine:
            for pgn in games:
                label = pgn.parent.name
                if args.only and args.only not in label:
                    continue
                print("#" * 78)
                print(f"# {label}")
                print("#" * 78)
                coach_one(pgn, engine, provider, settings, label)
                sys.stdout.flush()
    except BudgetExceeded as exc:
        print(f"stopped: {exc}")
        return 2
    finally:
        if ledger is not None:
            print(ledger.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
