"""A few Lichess puzzles through the live coach, at the moments a player meets them: what it
says when the puzzle is joined, when the player stays stuck (twice), when they ask for a
hint, and after a wrong move -- beside the puzzle's themes and solution, to see whether it
names the main idea (a mate, a fork, a pin) or talks around it (Tai, 2026-09-23: "look at
the whole board", "what is your opponent's most dangerous idea" on a mate in two). The
Lichess themes are for reading the output against, never an input: a real game has none.

The coach is built as the panel builds it (`cli._Coaching`'s helpers), the engine at the
app's settings; the model is the one named, every call on the ledger with a token cap.

  uv run python tools/eval/puzzle_coach.py --provider none
  uv run python tools/eval/puzzle_coach.py --provider openai --budget-tokens 400000
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge import BudgetExceeded, Ledger  # noqa: E402
from providers import Metered  # noqa: E402

from sidecoach import cli  # noqa: E402
from sidecoach.analysis.annotator import GameAnnotator  # noqa: E402
from sidecoach.analysis.computed import Computed  # noqa: E402
from sidecoach.analysis.engine import Engine  # noqa: E402
from sidecoach.coach.coach import Coach  # noqa: E402
from sidecoach.coach.policy import InterjectionPolicy, PolicySettings  # noqa: E402
from sidecoach.config import find_stockfish, load_settings  # noqa: E402
from sidecoach.events import GameStarted, MovePlayed, Side  # noqa: E402
from sidecoach.llm.factory import make_provider  # noqa: E402

THEMES = [
    "mateIn1",
    "mateIn2",
    "mateIn3",
    "fork",
    "pin",
    "skewer",
    "discoveredAttack",
    "hangingPiece",
]


def pick(seed: int, low: int, high: int) -> list[dict]:
    """One puzzle per theme, rated between `low` and `high`, the same ones for a seed."""
    rows = list(csv.DictReader((ROOT / "data/puzzles/sample.csv").open(encoding="utf-8")))
    random.Random(seed).shuffle(rows)
    chosen: list[dict] = []
    for theme in THEMES:
        for row in rows:
            fits = theme in row["Themes"].split() and low <= int(row["Rating"]) <= high
            if fits and row not in chosen:
                chosen.append({**row, "theme": theme})
                break
    return chosen


def wrong_move(board: chess.Board, solution: chess.Move) -> chess.Move:
    """A move a player who missed the idea might play: a check or a capture that is not
    the solution, else the first other legal move -- the same one every run."""
    others = [m for m in board.legal_moves if m != solution]
    forcing = [m for m in others if board.gives_check(m) or board.is_capture(m)]
    return sorted(forcing or others, key=lambda m: m.uci())[0]


def build_coach(provider_name: str, ledger: Ledger | None, computed: Computed) -> Coach:
    """The coach the panel runs, on the model named (`none`: the offline templates)."""
    settings = load_settings()
    settings.llm.provider = provider_name
    if provider_name == "openai":
        settings.llm.model = "gpt-5.6-luna"
        settings.llm.routine_effort = "medium"  # Tai: luna at medium, everywhere
    provider = make_provider(settings.llm)
    if ledger is not None:
        provider = Metered(provider, ledger)
    offline = bool(getattr(provider, "offline", False))
    policy = InterjectionPolicy(
        PolicySettings(stuck_seconds=settings.coach.stuck_seconds, quiet=settings.coach.quiet)
    )
    routine = cli._routine_provider(settings, getattr(provider, "inner", provider))
    if ledger is not None and routine is not getattr(provider, "inner", provider):
        routine = Metered(routine, ledger)
    return Coach(
        provider,
        policy=policy,
        draft_check=cli._draft_check(),
        claims=None if offline else cli._claims_check(provider),
        verifier=None if offline else cli._verifier(settings, provider),  # type: ignore[arg-type]
        engine=cli._named_engine(settings, computed),
        summarise_every=settings.coach.summarise_every,
        compact_record=settings.coach.compact_record,
        routine_provider=routine,  # type: ignore[arg-type]
        verbalized_sampling=settings.coach.verbalized_sampling,
        local_verdicts=settings.coach.local_verdicts,
        persona=settings.coach.persona,
        rating=(settings.profile.rating_low, settings.profile.rating_high),
    )


def run(provider_name: str, seed: int, budget: int, show_record: int) -> None:
    settings = load_settings()
    path = find_stockfish(settings)
    assert path is not None, "no Stockfish"
    ledger = (
        Ledger(f"puzzle-coach-{int(time.time())}", "puzzle_coach", "gpt-5.6-luna", budget)
        if provider_name != "none"
        else None
    )
    engine = Engine(
        path=path,
        threads=settings.engine.threads,
        hash_mb=settings.engine.hash_mb,
        multipv=settings.engine.multipv,
        pv_plies=settings.engine.pv_plies,
        score_model=settings.engine.score_model,
        max_seconds=settings.engine.max_seconds,
    )
    engine.open()
    # As the app builds it (`cli`'s analysis): the motif tagger, the book, the thresholds.
    from sidecoach.analysis.score import Thresholds

    annotator = GameAnnotator(
        engine=engine,
        depth=settings.engine.depth,
        multipv=settings.engine.multipv,
        thresholds=Thresholds.from_settings(settings.coach),
        tagger=cli._load_tagger(),
        book=cli._load_book(settings),
    )
    for n, puzzle in enumerate(pick(seed, 1200, 1800)):
        if ledger is not None and ledger.over_budget():
            print("-- budget reached; stopped")
            break
        coach = build_coach(provider_name, ledger, annotator.computed)
        board = chess.Board(puzzle["FEN"])
        moves = puzzle["Moves"].split()
        board.push_uci(moves[0])  # the opponent's move that sets the puzzle
        solver = Side.WHITE if board.turn == chess.WHITE else Side.BLACK
        line = board.copy()
        solution = []
        for uci in moves[1:]:
            solution.append(line.san(chess.Move.from_uci(uci)))
            line.push_uci(uci)
        print(f"\n## {n + 1}. {puzzle['theme']} ({puzzle['PuzzleId']}, {puzzle['Rating']})")
        print(f"themes: {puzzle['Themes']}")
        print(f"{solver.value} to move: {board.fen()}")
        print(f"solution: {' '.join(solution)}")
        gid = f"pz{n}"
        ts = 1000.0

        def said(label: str, coach: Coach = coach) -> None:
            coach.speaker.run_pending()
            for message in coach.poll():
                print(f"  [{label}:{message.level.name.lower()}] {message.text}")

        coach.on_event(
            GameStarted(game_id=gid, user_side=solver, start_fen=board.fen(), kind="position"),
            ts,
        )
        coach.on_event(annotator.analysis_event(gid, board, 0), ts)
        said("joined")
        for spell in ("stuck", "stuck again"):
            ts += coach.policy.settings.stuck_seconds + 5
            coach.tick(ts)
            said(spell)
        ts += 3
        coach.ask("hint", ts)
        said("hint")
        # A wrong move: the lesson after it.
        move = wrong_move(board, chess.Move.from_uci(moves[1]))
        after = board.copy()
        after.push(move)
        ts += 5
        coach.on_event(
            MovePlayed(
                game_id=gid,
                ply=1,
                san=board.san(move),
                uci=move.uci(),
                fen_before=board.fen(),
                fen_after=after.fen(),
                by_user=True,
                think_ms=30_000,
            ),
            ts,
        )
        assessment, analysis = annotator.assess(gid, board, move, 1)
        coach.on_event(assessment, ts)
        coach.on_event(analysis, ts)
        cost = assessment.expected_loss
        print(f"  (played {board.san(move)}: {assessment.label}, cost {cost:.2f})")
        said("lesson")
        if n < show_record:
            print("  --- what the model was given ---")
            print("  " + coach.conversation.render().replace("\n", "\n  "))
    engine.close()
    if ledger is not None:
        print("\n" + ledger.summary())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="none", help="none (templates) or openai (luna)")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--budget-tokens", type=int, default=400_000)
    parser.add_argument(
        "--show-record", type=int, default=1, help="print the record for the first N"
    )
    args = parser.parse_args()
    try:
        run(args.provider, args.seed, args.budget_tokens, args.show_record)
    except BudgetExceeded as exc:  # checked before every call, through the coach's own
        raise SystemExit(f"stopped: {exc}") from None


if __name__ == "__main__":
    main()
