"""A finished game reviewed whole (docs/game-review-plan.md): every position of it measured by
several engines at once, then every move judged in order from what they found -- the numbers
behind the review's summary, its step-through, and the coach's words on every move.

The live analyser judges a move as the game goes, one engine, one position at a time. A
review has the whole game at once, and a game's positions are independent searches, so a few
engines measure them side by side -- whole, with the threats and traps the live coach's facts
are made of. One annotator then walks the moves in order from those measurements, which costs
no search: a move's label needs the move before it (a miss is the opponent's mistake not
taken), so the walk is sequential, and it is quick.
"""

from __future__ import annotations

import contextlib
import os
import queue
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import chess
import chess.pgn
from pydantic import TypeAdapter

from sidecoach.analysis.annotator import Book, GameAnnotator, Measured, Tagger
from sidecoach.analysis.engine import Engine
from sidecoach.analysis.score import Thresholds
from sidecoach.events import ReviewedGame, ReviewedMove, Side


@dataclass(frozen=True)
class Strength:
    """How hard the engines look at every position: chess.com's Fast, Standard and Deep."""

    name: str
    depth: int
    multipv: int


STRENGTHS: dict[str, Strength] = {
    s.name: s
    for s in (Strength("fast", 14, 3), Strength("standard", 18, 6), Strength("deep", 22, 6))
}

# Each engine's threads; engines side by side take up to two thirds of the cores between them,
# the rest left to everything else the machine does (a game, the capture, the page).
THREADS = 2
MAX_WORKERS = 8


def workers_for(cores: int | None = None) -> int:
    """How many engines measure side by side on a machine with `cores` cores."""
    cores = cores or os.cpu_count() or 4
    return max(1, min(MAX_WORKERS, cores * 2 // 3 // THREADS))


Progress = Callable[[int, int], None]


def positions(game: chess.pgn.Game) -> list[chess.Board]:
    """Every position of the game's main line, the first before any move and the last after
    the final one, each carrying the moves that led to it (the opening book reads them)."""
    board = game.board()
    boards = [board.copy()]
    for move in game.mainline_moves():
        board.push(move)
        boards.append(board.copy())
    return boards


_MEASURED: TypeAdapter[Measured] = TypeAdapter(Measured)


def measured_json(measured: Measured) -> str:
    """A measured position as it is kept while a review is under way (Store.add_progress)."""
    return _MEASURED.dump_json(measured).decode("utf-8")


def measured_from_json(text: str) -> Measured:
    return _MEASURED.validate_json(text)


def measure_all(
    boards: list[chess.Board],
    engines: Sequence[Engine],
    *,
    depth: int,
    multipv: int,
    thresholds: Thresholds,
    on_progress: Progress | None = None,
    stop: threading.Event | None = None,
    known: dict[str, Measured] | None = None,
    on_measured: Callable[[Measured], None] | None = None,
) -> list[Measured]:
    """Every position measured whole, each engine taking the next position left, in the
    game's order so a caller's progress reads from the first move. A search that fails ends
    its engine's share; the others finish the rest, and the walk searches whatever is left.
    `known`: positions measured before (by FEN; a review that was stopped part way), not
    searched again and counted as done from the start; `on_measured` is handed each position
    as it is measured, to keep it."""
    known = known or {}
    todo: queue.SimpleQueue[chess.Board] = queue.SimpleQueue()
    found: list[Measured] = []
    for board in boards:
        if board.fen() in known:
            found.append(known[board.fen()])
        else:
            todo.put(board)
    if found and on_progress is not None:
        on_progress(len(found), len(boards))
    lock = threading.Lock()

    def work(engine: Engine) -> None:
        annotator = GameAnnotator(
            engine=engine, depth=depth, multipv=multipv, thresholds=thresholds
        )
        while stop is None or not stop.is_set():
            try:
                board = todo.get_nowait()
            except queue.Empty:
                return
            try:
                measured = annotator.measure(board)
            except Exception:  # noqa: BLE001 - the walk measures what is missing itself
                return
            if on_measured is not None:
                # Not kept is only a resume the slower.
                with contextlib.suppress(Exception):
                    on_measured(measured)
            with lock:
                found.append(measured)
                done = len(found)
            if on_progress is not None:
                on_progress(done, len(boards))

    threads = [
        threading.Thread(target=work, args=(engine,), name=f"sidecoach-review-{i}", daemon=True)
        for i, engine in enumerate(engines)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return found


def review_game(
    game: chess.pgn.Game,
    engines: Sequence[Engine],
    *,
    depth: int,
    multipv: int,
    thresholds: Thresholds,
    tagger: Tagger | None = None,
    book: Book | None = None,
    strength: str = "standard",
    game_id: str = "review",
    on_progress: Progress | None = None,
    stop: threading.Event | None = None,
    known: dict[str, Measured] | None = None,
    on_measured: Callable[[Measured], None] | None = None,
) -> ReviewedGame:
    """The game reviewed whole with `engines` (open, at least one; the first also walks the
    moves). `on_progress(done, total)` counts the positions measured; `known` and
    `on_measured` carry a review over a stop (`measure_all`)."""
    started = time.perf_counter()
    boards = positions(game)
    measured = measure_all(
        boards,
        engines,
        depth=depth,
        multipv=multipv,
        thresholds=thresholds,
        on_progress=on_progress,
        stop=stop,
        known=known,
        on_measured=on_measured,
    )
    annotator = GameAnnotator(
        engine=engines[0],
        depth=depth,
        multipv=multipv,
        thresholds=thresholds,
        tagger=tagger,
        book=book,
    )
    annotator.adopt(measured)
    start = annotator.analysis_event(game_id, boards[0], 0)
    times = clocks(game)
    moves: list[ReviewedMove] = []
    for ply, move in enumerate(game.mainline_moves(), start=1):
        if stop is not None and stop.is_set():
            break
        before = boards[ply - 1]
        assessment, after = annotator.assess(game_id, before, move, ply)
        clock_ms, spent_ms = times[ply - 1]
        moves.append(
            ReviewedMove(
                ply=ply,
                san=before.san(move),
                side=Side.WHITE if before.turn == chess.WHITE else Side.BLACK,
                fen_before=before.fen(),
                assessment=assessment,
                after=after,
                clock_ms=clock_ms,
                spent_ms=spent_ms,
            )
        )
    return ReviewedGame(
        headers={k: str(v) for k, v in game.headers.items()},
        start=start,
        moves=moves,
        strength=strength,
        depth=depth,
        seconds=round(time.perf_counter() - started, 2),
    )


_CONTROL = re.compile(r"^(\d+)(?:\+(\d+))?$")


def clocks(game: chess.pgn.Game) -> list[tuple[int | None, int | None]]:
    """Each move's clock after it and the time it took, in ms, from the PGN's `[%clk]`
    comments and its `TimeControl` ("600+5": 600 s a side, 5 s added a move). A
    correspondence game's clock ("1/86400") is days, not a move's thinking: none there."""
    control = _CONTROL.match(game.headers.get("TimeControl", ""))
    initial = int(control.group(1)) * 1000 if control else None
    increment = int(control.group(2) or 0) * 1000 if control else 0
    last: dict[bool, int | None] = {chess.WHITE: initial, chess.BLACK: initial}
    out: list[tuple[int | None, int | None]] = []
    for node in game.mainline():
        mover = not node.board().turn
        seconds = node.clock()
        if seconds is None:
            out.append((None, None))
            continue
        clock = round(seconds * 1000)
        before = last[mover]
        spent = before - clock + increment if before is not None and control else None
        out.append((clock, max(0, spent) if spent is not None else None))
        last[mover] = clock
    return out
