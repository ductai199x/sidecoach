"""Record what the desktop's profile stage makes of real games, for the Kotlin port.

The profile turns a game's events into the record (every move with its facts), the evidence
(each mistake placed on the two axes, cause and motif), the review, the practice and the
snapshot of the player model -- and answers questions from the record with the coach's own
SQL. It is pure given its inputs, so it is compared exactly: the same events in, the same
events out, the same rows in every table.

The events are real. Four games are played on one board each, every move judged by the
desktop's annotator on the real Stockfish (one thread, so the recording is the same on every
run) with the motif tagger and the opening book, and fed to the profiler the way the panel
feeds it -- a clock before each move, the move, its verdict, the analysis of where it led:

- the Blackburne Shilling trap, the player White: a queen lost to a mate, a move made with
  eight seconds on the clock, a lesson from the coach on the mistake and the player's own
  words about it ("I didn't see the queen"), which become the self-report and the cause;
- Legal's mate, the player Black: the queen taken, a mate in two missed;
- a Ruy Lopez, the player White, with a blunder taken back and another move played, ended
  by resignation;
- a puzzle from the store, joined on the screen and solved move by move.

Then the session ends (its summary), a second session reads the first's summary back, and
the ask side is recorded against the database the games left: the system message, SQL found
in replies, the nine worked queries run and rendered, three refused, and two conversations
with a scripted model -- one answered, one that never writes prose.

The puzzle draw is fixed (no shuffle) and the session id given, so nothing random is left.

    cd D:/chess-coaching-companion
    uv run --no-sync python D:/sidecoach-android/tools/make_profile_fixtures.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402

from sidecoach.analysis import motifs  # noqa: E402
from sidecoach.analysis.annotator import GameAnnotator  # noqa: E402
from sidecoach.analysis.engine import Engine  # noqa: E402
from sidecoach.analysis.openings import OpeningBook  # noqa: E402
from sidecoach.config import find_stockfish, load_settings  # noqa: E402
from sidecoach.events import (  # noqa: E402
    ClockObservation,
    CoachMessage,
    DisclosureLevel,
    GameEnded,
    GameStarted,
    MovePlayed,
    Side,
    Takeback,
    Trigger,
    UserUtterance,
)
from sidecoach.llm.base import Completion  # noqa: E402
from sidecoach.profile import ask  # noqa: E402
from sidecoach.profile.profiler import Profiler  # noqa: E402
from sidecoach.profile.puzzles import PuzzleStore  # noqa: E402
from sidecoach.profile.store import QueryRefused, Store  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "profile"
DEPTH = 12
MULTIPV = 4
PUZZLE_ROWS = 800


def dump(event: Any) -> dict:
    data = json.loads(event.model_dump_json(exclude={"ts", "source_frame_id"}))
    data["type"] = type(event).__name__
    return data


class Run:
    """The events fed, and what the profiler said back to each."""

    def __init__(self, profiler: Profiler) -> None:
        self.profiler = profiler
        self.steps: list[dict] = []
        self.ts = 1000.0

    def feed(self, event: Any, *, advance: float = 5.0) -> list[Any]:
        self.ts += advance
        out = self.profiler.on_event(event, self.ts)
        self.steps.append({"ts": self.ts, "event": dump(event), "out": [dump(e) for e in out]})
        return out

    def call(self, name: str, **kwargs: Any) -> None:
        """A method call on the profiler (tick, flush), recorded like an event."""
        self.ts += kwargs.pop("advance", 5.0)
        out = getattr(self.profiler, name)(self.ts)
        self.steps.append({"ts": self.ts, "call": name, "out": [dump(e) for e in out]})


def play(
    run: Run,
    annotator: GameAnnotator,
    game_id: str,
    user: Side,
    sans: list[str],
    *,
    clock_ms: int = 180_000,
    fast: dict[int, int] | None = None,
    lesson: tuple[int, str] | None = None,
    takeback: tuple[int, str] | None = None,
) -> chess.Board:
    """One game from the start: a clock, the move, its verdict and analysis, for each ply."""
    run.feed(GameStarted(game_id=game_id, user_side=user, start_fen=chess.STARTING_FEN))
    board = chess.Board()
    white, black = clock_ms, clock_ms
    ply = 0
    queue = list(sans)
    while queue:
        san = queue.pop(0)
        ply += 1
        mover = board.turn
        if fast and ply in fast:
            if mover == chess.WHITE:
                white = fast[ply]
            else:
                black = fast[ply]
        run.feed(ClockObservation(white_ms=white, black_ms=black, running=Side.WHITE if mover else Side.BLACK))
        before = board.fen()
        move = board.parse_san(san)
        assessment, analysis = annotator.assess(game_id, board, move, ply)
        board.push(move)
        by_user = (mover == chess.WHITE) == (user is Side.WHITE)
        run.feed(MovePlayed(game_id=game_id, ply=ply, san=san, uci=move.uci(), fen_before=before,
                            fen_after=board.fen(), by_user=by_user, think_ms=3000 + 100 * ply))
        run.feed(assessment)
        run.feed(analysis)
        if mover == chess.WHITE:
            white -= 4000
        else:
            black -= 4000
        if lesson and lesson[0] == ply:
            run.feed(CoachMessage(text="That knight grab let the queen in.", level=DisclosureLevel(3),
                                  trigger=Trigger.POST_MISTAKE, game_id=game_id, ply=ply - 1))
            run.feed(UserUtterance(text=lesson[1], source="text"), advance=10.0)
        if takeback and takeback[0] == ply:
            board.pop()
            ply -= 1
            run.feed(Takeback(game_id=game_id, plies=1, fen=board.fen()))
            queue.insert(0, takeback[1])
            takeback = None  # once: the move played instead stands
    return board


def puzzle_solved(run: Run, store: PuzzleStore) -> dict:
    """A puzzle of the store joined on the screen and solved move by move."""
    for motif in ("fork", "hangingPiece", "pin"):
        found = store.pick(motif, 1100, count=1, rng=_Fixed())
        if found:
            puzzle = found[0]
            break
    else:
        raise SystemExit("no puzzle in the sample to solve")
    board = chess.Board(puzzle.fen)
    board.push_uci(puzzle.moves[0])
    user = Side.WHITE if board.turn == chess.WHITE else Side.BLACK
    run.feed(GameStarted(game_id="puzzle-game", user_side=user, start_fen=board.fen(), kind="position"))
    for i, uci in enumerate(puzzle.moves[1:], start=1):
        before = board.fen()
        move = chess.Move.from_uci(uci)
        san = board.san(move)
        board.push(move)
        run.feed(MovePlayed(game_id="puzzle-game", ply=i, san=san, uci=uci, fen_before=before,
                            fen_after=board.fen(), by_user=i % 2 == 1))
    return {"id": puzzle.id}


class _Fixed(random.Random):
    """No shuffle: the draw among a recommendation's puzzles, fixed for the comparison."""

    def shuffle(self, x: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        return None


class Scripted:
    """A model that says what it is told to, in order."""

    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def healthcheck(self) -> tuple[bool, str]:
        return True, "scripted"

    def chat(self, system: str, messages: list, **_: Any) -> Completion:
        self.seen.append([dict(m) for m in messages])
        return Completion(text=self.replies.pop(0) if self.replies else "```sql\nSELECT 1\n```")


def tables(store: Store) -> dict:
    out = {}
    for table in ("games", "moves", "evidence", "dialogue", "sessions", "recommendations",
                  "puzzle_attempts", "settings"):
        cursor = store._db.execute(f"SELECT * FROM {table} ORDER BY rowid")
        names = [d[0] for d in cursor.description]
        out[table] = {"columns": names, "rows": [list(r) for r in cursor.fetchall()]}
    return out


def main() -> None:
    random.Random.shuffle = _Fixed.shuffle  # type: ignore[method-assign]  # the profiler's draw
    path = find_stockfish(load_settings())
    assert path is not None, "no Stockfish for the desktop"
    book = OpeningBook.load(DESKTOP / "data" / "openings")
    assert book is not None
    OUT.mkdir(parents=True, exist_ok=True)

    # The puzzles: the first rows of the sample, written beside the fixture for the phone.
    sample = DESKTOP / "data" / "puzzles" / "sample.csv"
    with sample.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))[: PUZZLE_ROWS + 1]
    puzzles_csv = OUT / "puzzles.csv"
    with puzzles_csv.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    puzzles = PuzzleStore(None)
    imported = puzzles.import_csv(puzzles_csv)

    store = Store(None)
    profiler = Profiler(store, session_id="session-one", ts=1000.0, puzzles=puzzles)
    run = Run(profiler)
    first_snapshot = dump(profiler.snapshot())
    with Engine(path=path, threads=1, hash_mb=64, multipv=MULTIPV) as engine:
        def annotator() -> GameAnnotator:
            return GameAnnotator(engine=engine, depth=DEPTH, multipv=MULTIPV, tagger=motifs, book=book)

        board = play(
            run, annotator(), "shilling", Side.WHITE,
            "e4 e5 Nf3 Nc6 Bc4 Nd4 Nxe5 Qg5 Nxf7 Qxg2 Rf1 Qxe4+ Be2 Nf3#".split(),
            fast={11: 8000},
            lesson=(9, "I didn't see the queen coming to g5 at all"),
        )
        run.feed(GameEnded(game_id="shilling", result="0-1", reason="game over on board",
                           pgn=None, final_fen=board.fen()))
        board = play(
            run, annotator(), "legal", Side.BLACK,
            "e4 e5 Nf3 d6 Bc4 Bg4 Nc3 g6 Nxe5 Bxd1 Bxf7+ Ke7 Nd5#".split(),
        )
        run.feed(GameEnded(game_id="legal", result="1-0", reason="game over on board",
                           pgn=None, final_fen=board.fen()))
        board = play(
            run, annotator(), "ruy", Side.WHITE,
            ("e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Na5 Bc2 c5 "
             "d4 Qc7 Nbd2 cxd4 cxd4 Nc6 Nb3 a5 Be3 a4 Nbd2 Bd7 Qe2").split(),
            takeback=(33, "Rc1"),
        )
        run.feed(GameEnded(game_id="ruy", result="0-1", reason="user resigned or lost on time",
                           pgn=None, final_fen=board.fen()))
    puzzle = puzzle_solved(run, puzzles)
    run.call("tick")
    summary = profiler.end_session(run.ts + 5.0)
    second = Profiler(store, session_id="session-two", ts=run.ts + 10.0, puzzles=puzzles)
    second_snapshot = dump(second.snapshot())

    # The ask side, against the record the games left.
    queries = []
    for question, sql in ask.EXAMPLES:
        columns, rows_ = store.query(sql)
        queries.append({"question": question, "sql": sql, "columns": columns,
                        "rows": [list(r) for r in rows_], "rendered": ask.render_rows(columns, rows_)})
    refused = []
    for sql in ("DELETE FROM moves", "SELECT 1; SELECT 2", "PRAGMA table_info(moves)", ""):
        try:
            store.query(sql)
            refused.append({"sql": sql, "refused": False})
        except QueryRefused:
            refused.append({"sql": sql, "refused": True})
    extracts = [
        {"text": t, "sql": ask.extract_sql(t)}
        for t in (
            "Here:\n```sql\nSELECT COUNT(*) FROM games\n```",
            "```\nSELECT 1\n```",
            "```SQL\n  SELECT 2  \n```\n```sql\nSELECT 3\n```",
            "You have played three games.",
        )
    ]
    conversations = []
    for replies in (
        ["Let me look.\n```sql\nSELECT COUNT(*) AS games, ROUND(AVG(final_score), 2) AS score FROM games"
         " WHERE result IS NOT NULL\n```",
         "```sql\nDELETE FROM games\n```",
         "You have finished three games and scored poorly in them."],
        [],
    ):
        provider = Scripted(replies)
        answer = ask.answer("How am I doing?", store, provider)  # type: ignore[arg-type]
        conversations.append({
            "replies": replies,
            "text": answer.text,
            "calls": answer.calls,
            "failed": answer.failed,
            "exchanges": [{"sql": e.sql, "columns": e.columns, "rows": [list(r) for r in e.rows],
                           "refused": e.error is not None} for e in answer.exchanges],
            "messages": provider.seen,
        })

    fixture = {
        "puzzles_imported": imported,
        "session": "session-one",
        "first_snapshot": first_snapshot,
        "steps": run.steps,
        "puzzle": puzzle,
        "summary": summary,
        "second_snapshot": second_snapshot,
        "tables": tables(store),
        "system": ask.system_message(),
        "queries": queries,
        "refused": refused,
        "extracts": extracts,
        "conversations": conversations,
        "cells": [c.__dict__ for c in store.cells()],
    }
    (OUT / "profile.json").write_text(json.dumps(fixture), encoding="utf-8")
    emitted = [e["type"] for s in run.steps for e in s["out"]]
    print(f"{len(run.steps)} steps; emitted {sorted(set(emitted))} "
          f"({emitted.count('ProfileEvidence')} evidence, {emitted.count('GameReview')} reviews); "
          f"{imported} puzzles; summary: {summary}")
    for line in second_snapshot["insights"]:
        print("  insight:", line)


if __name__ == "__main__":
    main()
