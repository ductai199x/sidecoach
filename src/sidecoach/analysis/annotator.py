"""From moves to assessments: what a move cost, what the position holds, what is threatened.

`GameAnnotator` does the work for one game and is synchronous, which is what a post-game
review wants. `Analyser` wraps it in a worker thread, which is what a live session needs:
the engine takes about a third of a second per position and the capture loop must not wait
for it.

Every position of a game is measured once, to the same depth, and cached, so the position
after one move is the position before the next and a game costs one analysis per ply. The
matching depth is what makes the loss of a move meaningful: two searches of different
lengths would differ by more than most mistakes do.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import chess

from sidecoach.analysis.engine import Engine
from sidecoach.analysis.score import (
    Thresholds,
    eval_loss,
    is_decisive,
    is_only_move,
    position_score,
    severity,
)
from sidecoach.analysis.threats import threat_moves
from sidecoach.events import Analysis, Event, Line, MotifTag, MoveAssessment


class Tagger(Protocol):
    """What `sidecoach.analysis.motifs` provides; a module satisfies this."""

    def tags_for_move(
        self,
        before: chess.Board,
        move: chess.Move,
        *,
        pv: Sequence[chess.Move] = (),
        mate_in: int | None = None,
    ) -> list[MotifTag]: ...

    def tags_for_position(
        self,
        board: chess.Board,
        *,
        pv: Sequence[chess.Move] = (),
        mate_in: int | None = None,
    ) -> list[MotifTag]: ...


class Book(Protocol):
    """What `sidecoach.analysis.openings.OpeningBook` provides."""

    def identify(self, moves: Sequence[chess.Move]) -> object | None: ...


@dataclass
class Measured:
    """One position, measured once: its lines and what they mean."""

    fen: str
    lines: list[Line]
    score: float
    decisive: bool
    only_move: bool
    threats: list[str] = field(default_factory=list)

    @property
    def depth(self) -> int:
        """How deep the search actually got, which the time cap can cut short."""
        return self.lines[0].depth if self.lines else 0


@dataclass
class GameAnnotator:
    """Assessments for the moves of one game. Not thread-safe; `Analyser` owns one."""

    engine: Engine
    depth: int = 18
    multipv: int = 4
    thresholds: Thresholds = field(default_factory=Thresholds)
    tagger: Tagger | None = None
    book: Book | None = None
    with_threats: bool = True
    _cache: dict[str, Measured] = field(default_factory=dict, init=False, repr=False)

    def reset(self) -> None:
        """Forget the cache; a new game shares no positions with the last one."""
        self._cache.clear()

    def measure(self, board: chess.Board) -> Measured:
        key = board.fen()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        lines = self.engine.analyse(board, depth=self.depth, multipv=self.multipv)
        measured = Measured(
            fen=key,
            lines=lines,
            score=position_score(board, lines),
            decisive=is_decisive(board, lines, self.thresholds),
            only_move=is_only_move(board, lines, self.thresholds),
            threats=(
                threat_moves(self.engine, board, depth=self.depth) if self.with_threats else []
            ),
        )
        self._cache[key] = measured
        return measured

    def analysis_event(self, game_id: str, board: chess.Board, ply: int) -> Analysis:
        """What the position holds, for the side to move."""
        measured = self.measure(board)
        return Analysis(
            game_id=game_id,
            fen=measured.fen,
            ply=ply,
            lines=measured.lines,
            decisive=measured.decisive,
            only_move=measured.only_move,
            threats=measured.threats,
            tags=self._position_tags(board, measured),
            opening=self._opening(board),
        )

    def assess(
        self, game_id: str, before: chess.Board, move: chess.Move, ply: int
    ) -> tuple[MoveAssessment, Analysis]:
        """What the move cost its mover, and what the position it led to holds."""
        if move not in before.legal_moves:
            raise ValueError(f"{move.uci()} is not legal in {before.fen()}")
        measured_before = self.measure(before)
        # The history comes along: the opening a game is in is a fact about its moves, not
        # about the position, and the position after this move is the next move's history.
        after = before.copy()
        after.push(move)
        measured_after = self.measure(after)
        loss = eval_loss(measured_before.lines, move.uci(), measured_after.score)
        assessment = MoveAssessment(
            game_id=game_id,
            ply=ply,
            uci=move.uci(),
            expected_loss=loss,
            severity=severity(loss, self.thresholds),
            best=measured_before.lines[0].moves if measured_before.lines else [],
            tags=self._move_tags(before, move, measured_after),
            depth=min(measured_before.depth, measured_after.depth or measured_before.depth),
        )
        return assessment, self.analysis_event(game_id, after, ply)

    # Tagging and openings are optional collaborators, so a bare engine still works.

    def _position_tags(self, board: chess.Board, measured: Measured) -> list[MotifTag]:
        """What the position holds for the side to move; their mate distance counts their
        own next move, which is the same convention a move's tags use."""
        if self.tagger is None or not measured.lines:
            return []
        return self.tagger.tags_for_position(
            board, pv=_moves(measured.lines[0]), mate_in=measured.lines[0].mate
        )

    def _move_tags(self, before: chess.Board, move: chess.Move, after: Measured) -> list[MotifTag]:
        if self.tagger is None:
            return []
        pv = _moves(after.lines[0]) if after.lines else []
        return self.tagger.tags_for_move(
            before, move, pv=pv, mate_in=_mate_in(before, move, after.lines)
        )

    def _opening(self, board: chess.Board) -> str | None:
        """The opening this game is in, when the board carries the moves that got here.

        A position read off the screen, or one adopted by a resync, has no history to look
        up, and a fragment of a history would name the wrong opening.
        """
        if self.book is None or not board.move_stack:
            return None
        if board.root().fen() != chess.STARTING_FEN:
            return None
        found = self.book.identify(list(board.move_stack))
        if found is None:
            return None
        eco = getattr(found, "eco", None)
        name = getattr(found, "name", None)
        if name is None:
            return None
        return f"{eco} {name}" if eco else str(name)


def _moves(line: Line) -> list[chess.Move]:
    out: list[chess.Move] = []
    for uci in line.moves:
        try:
            out.append(chess.Move.from_uci(uci))
        except ValueError:
            break
    return out


def _mate_in(before: chess.Board, move: chess.Move, after_lines: Sequence[Line]) -> int | None:
    """Moves the mover needs to mate, counting this one; negative when the mover gets mated.

    The engine reports the mate distance in the position after the move, where the opponent
    is to move and counts their own moves: "mated in two" there means two more moves by the
    mover, three counting the one just played.
    """
    after = before.copy(stack=False)
    after.push(move)
    if after.is_checkmate():
        return 1
    if not after_lines or after_lines[0].mate is None:
        return None
    mate = after_lines[0].mate
    return -mate if mate > 0 else abs(mate) + 1


@dataclass(frozen=True)
class _Request:
    game_id: str
    ply: int
    board: chess.Board
    move: chess.Move | None


class Analyser:
    """A `GameAnnotator` on a worker thread: submit moves, poll events, never block.

    The engine is slow by design and the capture loop runs at ten frames a second, so
    nothing here waits for a result. Assessments arrive a moment after the move, which is
    when a coach would speak anyway.
    """

    def __init__(self, annotator: GameAnnotator, *, max_pending: int = 64) -> None:
        self.annotator = annotator
        self.max_pending = max_pending
        self.dropped = 0
        self.failures = 0
        self.last_error: str | None = None
        self._requests: queue.Queue[_Request | None] = queue.Queue()
        self._results: queue.SimpleQueue[Event] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._busy = threading.Event()

    def __enter__(self) -> Analyser:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def pending(self) -> int:
        return self._requests.qsize() + (1 if self._busy.is_set() else 0)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="sidecoach-analyser", daemon=True)
        self._thread.start()

    def close(self, timeout: float = 5.0) -> None:
        thread, self._thread = self._thread, None
        if thread is not None:
            self._requests.put(None)
            thread.join(timeout)
        self.annotator.engine.close()

    def submit_move(self, game_id: str, before: chess.Board, move: chess.Move, ply: int) -> None:
        self._submit(_Request(game_id, ply, before.copy(stack=False), move))

    def submit_position(self, game_id: str, board: chess.Board, ply: int) -> None:
        self._submit(_Request(game_id, ply, board.copy(stack=False), None))

    def reset(self, game_id: str | None = None) -> None:
        """A new game: drop what is queued, since none of it is about this game."""
        while True:
            try:
                self._requests.get_nowait()
            except queue.Empty:
                break
        self._requests.put(_Request(game_id or "", 0, chess.Board(), None))
        self._requests.get()  # keep the queue's unfinished count honest after the drain

    def poll(self) -> list[Event]:
        out: list[Event] = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                break
        return out

    def _submit(self, request: _Request) -> None:
        if self._requests.qsize() >= self.max_pending:
            # The engine is further behind than a game can get; the newest position still
            # matters more than an old one, so the oldest goes.
            try:
                self._requests.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
        self._requests.put(request)

    def _run(self) -> None:
        while True:
            request = self._requests.get()
            if request is None:
                return
            self._busy.set()
            try:
                if request.move is None:
                    self._results.put(
                        self.annotator.analysis_event(request.game_id, request.board, request.ply)
                    )
                else:
                    assessment, analysis = self.annotator.assess(
                        request.game_id, request.board, request.move, request.ply
                    )
                    self._results.put(assessment)
                    self._results.put(analysis)
            except Exception as exc:  # noqa: BLE001
                # A failed analysis must never end a session; the game goes on unassessed.
                self.failures += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._busy.clear()


def open_analyser(
    engine_path: object,
    *,
    depth: int,
    multipv: int,
    threads: int,
    hash_mb: int,
    pv_plies: int,
    thresholds: Thresholds,
    score_model: str = "human",
    max_seconds: float | None = 5.0,
    tagger: Tagger | None = None,
    book: Book | None = None,
    make_engine: Callable[..., Engine] = Engine,
) -> Analyser:
    """An analyser ready to take moves, with its engine already started."""
    from pathlib import Path

    engine = make_engine(
        path=Path(str(engine_path)),
        threads=threads,
        hash_mb=hash_mb,
        multipv=multipv,
        pv_plies=pv_plies,
        score_model=score_model,
        max_seconds=max_seconds,
    )
    engine.open()
    annotator = GameAnnotator(
        engine=engine,
        depth=depth,
        multipv=multipv,
        thresholds=thresholds,
        tagger=tagger,
        book=book,
    )
    analyser = Analyser(annotator)
    analyser.start()
    return analyser
