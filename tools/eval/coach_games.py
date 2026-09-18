"""Coach every recorded game from its PGN, faithfully, and log every message for judging.

Drives the real Coach with the real LLM, engine and tagger off the truth.pgn (no vision),
emitting the same events the live pipeline does. The user is White (nosegazm) in every
recording. Output: for each coach message, the ply, the move in play, the trigger, the
disclosure level, the text, and whether any check flagged the draft.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import chess.pgn

from sidecoach.analysis.annotator import GameAnnotator
from sidecoach.analysis.engine import Engine
from sidecoach.analysis.score import Thresholds
from sidecoach.cli import _draft_check, _load_book, _load_tagger
from sidecoach.coach.coach import Coach
from sidecoach.coach.policy import InterjectionPolicy, PolicySettings
from sidecoach.config import find_stockfish, load_settings
from sidecoach.events import (
    GameStarted,
    MovePlayed,
    Side,
)
from sidecoach.llm.factory import make_provider

DEPTH = 12
CAP = 2.0


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
    game = chess.pgn.read_game(path.open(encoding="utf-8"))
    board = game.board()
    gid = label
    coach = Coach(
        provider,
        policy=InterjectionPolicy(PolicySettings(stuck_seconds=30)),
        draft_check=_draft_check(),
    )
    # facts per ply, for context in the report
    facts_by_ply: dict[int, object] = {}

    def drain() -> None:
        coach.speaker.run_pending()
        for msg in coach.poll():
            rec = coach.spoken[-1]
            if msg.text.endswith(": good.") and not rec.problems:
                continue  # a terse nod, not a teaching moment; counted in the summary
            f = facts_by_ply.get(msg.ply)
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
        coach.on_event(analysis, ts)
        if (ply + 1) in coach.positions:
            facts_by_ply[ply + 1] = coach.positions[ply + 1].facts
        if by_user:
            coach.on_event(assessment, ts)
        drain()

    s = coach.speaker
    print(
        f"\n=== {label}: {len(coach.spoken)} messages, "
        f"{sum(1 for x in coach.spoken if x.problems)} drafts caught, "
        f"{s.fallbacks} templates used ==="
    )


def main() -> int:
    settings = load_settings()
    provider = make_provider(settings.llm)
    engine = Engine(
        path=find_stockfish(settings), threads=8, hash_mb=512, multipv=4, max_seconds=CAP
    )
    games = sorted(Path("data/recordings").glob("g*/truth.pgn"))
    only = sys.argv[1] if len(sys.argv) > 1 else None
    with engine:
        for pgn in games:
            label = pgn.parent.name
            if only and only not in label:
                continue
            print("#" * 78)
            print(f"# {label}")
            print("#" * 78)
            coach_one(pgn, engine, provider, settings, label)
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
