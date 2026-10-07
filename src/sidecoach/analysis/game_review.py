"""A finished game reviewed whole (docs/game-review.md): every position of it measured by
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

from sidecoach.analysis.annotator import (
    Book,
    GameAnnotator,
    Measured,
    Tagger,
    cache_key,
    move_tags,
    opening_name,
    own_line,
    position_tags,
    verdict_lines,
)
from sidecoach.analysis.engine import Engine, Interrupted
from sidecoach.analysis.score import (
    Previous,
    Thresholds,
    is_decisive,
    is_only_move,
    position_score,
    severity,
    verdict,
)
from sidecoach.events import Analysis, ReviewedGame, ReviewedMove, Side


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

# Each engine's threads: one. At a fixed depth more threads widen the search instead of speeding
# it -- four took 7.7 times the processor time of one, and twice as long, for an answer no better
# (the engine audit, 2026-10-04). Engines side by side take up to two thirds of the cores between
# them, the rest left to everything else the machine does (a game, the capture, the page).
THREADS = 1
MAX_WORKERS = 8
# How often a review running with a stop looks at it: the searches under way are cut at most
# this long after the stop.
STOP_LOOK_S = 0.2


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
    failed: list[Engine] | None = None,
) -> list[Measured]:
    """Every position measured whole (`annotator`'s docstring), each engine taking the game's
    next position left and finishing it -- its best line, its candidates at EARLY_DEPTH, each
    candidate's own position, its side searches -- in the game's order, so a caller's progress
    reads from the first move and a review stopped part way keeps every position it finished.
    A best line is searched once for the review: the game's next position is mostly a candidate
    of the one before, and an engine wanting one another is searching waits for it. A search
    that fails ends its engine's share; the others finish the rest, and the walk searches
    whatever is left. `known`: positions measured whole before (by FEN: a review stopped part
    way, or the live game's own), searched no more and counted done; `on_measured` is handed
    each position as it is measured whole, to keep it."""
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
    cond = threading.Condition()
    best: dict[str, Measured] = {}
    searching: set[str] = set()  # the best lines an engine is on

    def best_of(annotator: GameAnnotator, board: chess.Board) -> Measured:
        """`board`'s best line: known, found, waited for, or searched here."""
        whole = known.get(board.fen())
        if whole is not None:
            return whole
        key = cache_key(board)
        with cond:
            while key in searching:
                cond.wait()
            if key in best:
                return best[key]
            searching.add(key)
        try:
            line = annotator.best_line(board)
        except BaseException:
            with cond:  # a waiter searches it itself
                searching.discard(key)
                cond.notify_all()
            raise
        with cond:
            best[key] = line
            searching.discard(key)
            cond.notify_all()
        return line

    def measure(annotator: GameAnnotator, board: chess.Board) -> Measured:
        first = best_of(annotator, board)
        hint = annotator.hint_lines(board)
        children = []
        for move in annotator.candidates(board, first, hint):
            child = board.copy()
            child.push(move)
            children.append((move, best_of(annotator, child)))
        return annotator.with_side(board, annotator.whole(board, first, children))

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
                with engine.interruptible():  # a stop reaches the search under way
                    measured = measure(annotator, board)
            except Interrupted:
                return  # stopped: the position is not done, and nothing failed
            except Exception:  # noqa: BLE001 - the walk measures what is missing itself
                if failed is not None:
                    with cond:
                        failed.append(engine)  # named: the walk searches on another
                return
            if on_measured is not None:
                # Not kept is only a resume the slower.
                with contextlib.suppress(Exception):
                    on_measured(measured)
            with cond:
                found.append(measured)
                if on_progress is not None:
                    on_progress(len(found), len(boards))  # in order: under the lock

    threads = [
        threading.Thread(target=work, args=(engine,), name=f"sidecoach-review-{i}", daemon=True)
        for i, engine in enumerate(engines)
    ]
    for thread in threads:
        thread.start()
    if stop is not None:
        # A stop (the room's Pause) cuts the searches under way, not only the positions after
        # them: each engine's position whole is its best line, its hint, up to six candidates
        # and its side searches. Told again while any runs: a stop landing between two
        # positions is cleared as the next search begins (`Engine.interruptible`).
        while any(t.is_alive() for t in threads):
            if stop.wait(STOP_LOOK_S):
                for engine in engines:
                    engine.interrupt()
                for thread in threads:
                    thread.join(STOP_LOOK_S)
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
    """The game reviewed whole with `engines` (open, at least one; the first whose share did
    not fail also walks the moves). `on_progress(done, total)` counts the positions measured;
    `known` and `on_measured` carry a review over a stop (`measure_all`)."""
    started = time.perf_counter()
    boards = positions(game)
    failed: list[Engine] = []
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
        failed=failed,
    )
    # The walk measures what is missing on an engine whose share did not end in a failure.
    walker = next((e for e in engines if not any(e is f for f in failed)), engines[0])
    annotator = GameAnnotator(
        engine=walker,
        depth=depth,
        multipv=multipv,
        thresholds=thresholds,
        tagger=tagger,
        book=book,
    )
    annotator.adopt(measured)
    if stop is not None and stop.is_set():
        # Stopped: what is kept is what `on_measured` was handed; nothing more is searched.
        empty = annotator.analysis_from(
            game_id, boards[0], 0, annotator.measured_lines(boards[0], [])
        )
        return ReviewedGame(
            headers={k: str(v) for k, v in game.headers.items()},
            start=empty,
            moves=[],
            strength=strength,
            depth=depth,
            seconds=round(time.perf_counter() - started, 2),
        )
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


def relabel(
    reviewed: ReviewedGame,
    thresholds: Thresholds,
    book: Book | None = None,
    tagger: Tagger | None = None,
) -> ReviewedGame:
    """A saved review made again from what it kept -- each position's lines and whether it was
    in book, each move's loss -- as a review made today with `thresholds` makes it, no engine:
    each position's decisive and only-move calls, each move's severity and label (`score.verdict`,
    read off the lines `annotator.verdict_lines` gives, as the annotator reads them), and with
    `tagger` each move's and position's motifs read again (a sacrifice is what makes a move
    brilliant: the rule of 2026-10-06 reaches reviews saved before it); without, the saved tags
    stand. With `book`, each position's opening is named again. A review saved before the rules
    changed keeps its measurements and is labelled as one made today (Tai, 2026-10-02)."""
    start = chess.Board(reviewed.start.fen)
    named = book is not None and start.fen() == chess.STARTING_FEN
    first = _recalled(start, reviewed.start, thresholds, tagger)
    before_lines = first.lines
    before_decisive = first.decisive
    previous: Previous | None = None
    played: list[chess.Move] = []
    moves: list[ReviewedMove] = []
    for item in reviewed.moves:
        board = chess.Board(item.fen_before)
        move = chess.Move.from_uci(item.assessment.uci)
        loss = item.assessment.expected_loss
        after_lines = verdict_lines(own_line(before_lines, move), item.after.lines)
        tags = (
            move_tags(tagger, board, move, after_lines)
            if tagger is not None
            else item.assessment.tags
        )
        label = verdict(
            board,
            move,
            before_lines,
            after_lines,
            loss=loss,
            before_score=position_score(board, before_lines),
            decisive=before_decisive,
            in_book=item.after.in_book,
            sacrifice=any(tag.name == "sacrifice" for tag in tags),
            previous=previous,
            thresholds=thresholds,
        )
        after_board = board.copy(stack=False)
        after_board.push(move)
        after = _recalled(after_board, item.after, thresholds, tagger)
        played.append(move)
        if named:
            assert book is not None
            after = after.model_copy(update={"opening": opening_name(book, played)})
        assessment = item.assessment.model_copy(
            update={"label": label, "severity": severity(loss, thresholds), "tags": tags}
        )
        moves.append(item.model_copy(update={"assessment": assessment, "after": after}))
        previous = Previous(loss, move.to_square, board.is_capture(move))
        before_lines = after.lines
        before_decisive = after.decisive
    return reviewed.model_copy(update={"start": first, "moves": moves})


def _recalled(
    board: chess.Board, analysis: Analysis, thresholds: Thresholds, tagger: Tagger | None
) -> Analysis:
    """A saved position's calls made again from its saved lines, as `GameAnnotator.whole` makes
    them with `thresholds`, and with `tagger` its motifs (`annotator.position_tags`)."""
    update: dict[str, object] = {
        "decisive": is_decisive(board, analysis.lines, thresholds),
        "only_move": is_only_move(board, analysis.lines, thresholds),
    }
    if tagger is not None:
        update["tags"] = position_tags(tagger, board, analysis.lines)
    return analysis.model_copy(update=update)


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
