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
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import chess

from sidecoach.analysis import exchange, motifs
from sidecoach.analysis.defences import defence_lines
from sidecoach.analysis.engine import Engine, Interrupted
from sidecoach.analysis.score import (
    Thresholds,
    eval_loss,
    is_decisive,
    is_only_move,
    label_move,
    move_accuracy,
    position_score,
    severity,
)
from sidecoach.analysis.threats import aim_lines, threat_moves
from sidecoach.analysis.traps import trap_lines
from sidecoach.events import Analysis, Event, Line, MotifTag, MoveAssessment

# A whole measurement while the analyser is behind a fast game, or was within
# BEHIND_S: capped at HURRY_S and without the threat search (each could take max_seconds).
HURRY_S = 1.0
BEHIND_S = 10.0
# While no move waits -- the player thinking, or the opponent -- the engine looks ahead:
# the positions after the LOOK_AHEAD best moves from the position on the board, each then
# after the best reply to it, measured whole and cached. A game that goes there has its
# verdict and its facts at once (Tai, 2026-09-27: "cache ... the possible legal moves from
# both sides"); a move of its own stops the look ahead within the millisecond.
LOOK_AHEAD = 2


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

    def in_book(self, moves: Sequence[chess.Move]) -> bool: ...


@dataclass
class Measured:
    """One position, measured once: its lines and what they mean."""

    fen: str
    lines: list[Line]
    score: float
    decisive: bool
    only_move: bool
    threats: list[str] = field(default_factory=list)
    traps: list[Line] = field(default_factory=list)
    defences: list[Line] = field(default_factory=list)
    threatens: list[Line] = field(default_factory=list)

    @property
    def depth(self) -> int:
        """How deep the search actually got, which the time cap can cut short."""
        return self.lines[0].depth if self.lines else 0


@dataclass
class GameAnnotator:
    """Assessments for the moves of one game. Not thread-safe; `Analyser` owns one."""

    engine: Engine
    depth: int = 18
    multipv: int = 6
    thresholds: Thresholds = field(default_factory=Thresholds)
    tagger: Tagger | None = None
    book: Book | None = None
    with_threats: bool = True
    _cache: dict[str, Measured] = field(default_factory=dict, init=False, repr=False)
    # Positions measured for a verdict only: the best line at the full depth, no
    # candidates, no threats. Kept apart from `_cache`, so a later whole measurement of
    # the same position is not answered with the thin one.
    _verdict_cache: dict[str, Measured] = field(default_factory=dict, init=False, repr=False)
    # The last move assessed: game, ply, loss, where it landed and whether it captured, so
    # the next move knows whether the opponent just erred and whether it is merely taking
    # back.
    _previous: tuple[str, int, float, chess.Square, bool] | None = field(
        default=None, init=False, repr=False
    )
    # Positions measured ahead of the game (`look_ahead`) and not yet reached, and how many
    # the game has reached: what looking ahead saved.
    _ahead: set[str] = field(default_factory=set, init=False, repr=False)
    ahead_hits: int = field(default=0, init=False)

    def reset(self) -> None:
        """Forget the cache; a new game shares no positions with the last one."""
        self._cache.clear()
        self._verdict_cache.clear()
        self._ahead.clear()
        self._previous = None

    def look_ahead(self, board: chess.Board) -> Measured:
        """A position the game may reach, measured whole before it does, as the game's own
        positions are, and cached with them. Inside `Engine.interruptible` a search stopped
        raises `Interrupted`, and nothing is cached."""
        key = board.fen()
        if key in self._cache:
            return self._cache[key]
        measured = self.measure(board)
        self._ahead.add(key)
        return measured

    def _cached(self, key: str) -> Measured | None:
        """The position measured whole, counting a position measured ahead reached."""
        measured = self._cache.get(key)
        if measured is not None and key in self._ahead:
            self._ahead.discard(key)
            self.ahead_hits += 1
        return measured

    def measure_for_verdict(self, board: chess.Board) -> Measured:
        """The position measured for a verdict: whole if it already is, else its best
        line alone at the full depth -- what `eval_loss` needs, at a fraction of the
        cost. When the engine is behind a fast game, this is all a stale ply gets
        (2026-09-22: a queen ending played at two seconds a move ran the analyser forty
        plies behind, and every word on a move came after the board had moved on)."""
        key = board.fen()
        whole = self._cached(key)
        if whole is not None:
            return whole
        thin = self._verdict_cache.get(key)
        if thin is not None:
            return thin
        lines = self.engine.analyse(board, depth=self.depth, multipv=1)
        thin = Measured(
            fen=key,
            lines=lines,
            score=position_score(board, lines),
            decisive=False,
            only_move=False,
        )
        self._verdict_cache[key] = thin
        return thin

    def measure(self, board: chess.Board, *, hurry: bool = False) -> Measured:
        """The position whole: the candidates, the threats, the decisive-moment and
        only-move calls. `hurry` caps the search at HURRY_S and skips the threats: the
        analyser is (or was just) behind a fast game, and the player is not waiting ten
        seconds for the candidates of a position they will have left."""
        key = board.fen()
        cached = self._cached(key)
        if cached is not None:
            return cached
        lines = self.engine.analyse(
            board,
            depth=self.depth,
            multipv=self.multipv,
            time_s=HURRY_S if hurry else None,
            carry_best=not hurry,
        )
        measured = Measured(
            fen=key,
            lines=lines,
            score=position_score(board, lines),
            decisive=is_decisive(board, lines, self.thresholds),
            only_move=is_only_move(board, lines, self.thresholds),
            threats=(
                threat_moves(self.engine, board, depth=self.depth)
                if self.with_threats and not hurry
                else []
            ),
            # The captures that look free and are not, beside the threats: what the first
            # hint warns of ("the knight on g4 is bait"), searched only when not hurried.
            traps=trap_lines(self.engine, board, lines) if self.with_threats and not hurry else [],
            # And the recaptures the best line declines, with what refutes them: why the
            # obvious defence fails is the lesson ("the pawn on f2 cannot take back").
            defences=(
                defence_lines(self.engine, board, lines) if self.with_threats and not hurry else []
            ),
            # And what a quiet best move threatens: how it wins ("it threatens mate").
            threatens=(
                aim_lines(self.engine, board, lines) if self.with_threats and not hurry else []
            ),
        )
        self._cache[key] = measured
        return measured

    def analysis_event(
        self, game_id: str, board: chess.Board, ply: int, *, hurry: bool = False, thin: bool = False
    ) -> Analysis:
        """What the position holds, for the side to move. `thin`: only what the verdict
        already measured (the best line; no threats, no decisive call) -- the analysis
        of a position the game has left, so the coach still has the facts of the
        position each move was played from and the review has its score."""
        measured = self.measure_for_verdict(board) if thin else self.measure(board, hurry=hurry)
        return Analysis(
            game_id=game_id,
            fen=measured.fen,
            ply=ply,
            lines=measured.lines,
            decisive=measured.decisive,
            only_move=measured.only_move,
            threats=measured.threats,
            traps=measured.traps,
            defences=measured.defences,
            threatens=measured.threatens,
            tags=self._position_tags(board, measured),
            opening=self._opening(board),
            in_book=self._in_book(board),
        )

    def assess(
        self, game_id: str, before: chess.Board, move: chess.Move, ply: int
    ) -> tuple[MoveAssessment, Analysis]:
        """What the move cost its mover, and what the position it led to holds."""
        assessment = self.assess_move(game_id, before, move, ply)
        after = before.copy()
        after.push(move)
        return assessment, self.analysis_event(game_id, after, ply)

    def assess_move(
        self, game_id: str, before: chess.Board, move: chess.Move, ply: int
    ) -> MoveAssessment:
        """What the move cost its mover -- from the best line of the position it led to,
        at the full depth but that one line: a sixth of the time of the six the coach's
        candidates take (0.1 s against 0.7 at depth 18), so the verdict, and the word
        on the move, come that much sooner. `analysis_event` then measures the position
        whole; a position already measured whole is not searched again."""
        if move not in before.legal_moves:
            raise ValueError(f"{move.uci()} is not legal in {before.fen()}")
        measured_before = self.measure_for_verdict(before)
        # The history comes along: the opening a game is in is a fact about its moves, not
        # about the position, and the position after this move is the next move's history.
        after = before.copy()
        after.push(move)
        measured_after = self.measure_for_verdict(after)
        loss = eval_loss(measured_before.lines, move.uci(), measured_after.score)
        tags = self._move_tags(before, move, measured_after)
        best = measured_before.lines[0].moves if measured_before.lines else []
        previous = self._previous
        if previous is not None and (previous[0] != game_id or previous[1] != ply - 1):
            previous = None  # another game, or a gap: the move before this one is unknown
        opponent_erred = previous is not None and previous[2] >= self.thresholds.mistake
        # Taking back on the square the opponent just captured on. Taking a piece that
        # merely moved there is not obvious: that is how a blunder gets punished.
        recapture = (
            previous is not None
            and previous[4]
            and before.is_capture(move)
            and move.to_square == previous[3]
        )
        # A capture nobody can take back is a grab, not a find; a king's capture always is.
        # Taking back is read by the rules: a piece guarded only by one pinned to its own king
        # is taken for nothing.
        free_grab = before.is_capture(move) and not exchange.can_retake(before, move)
        obvious = recapture or free_grab or before.legal_moves.count() == 1
        label = label_move(
            loss,
            self.thresholds,
            is_best=bool(best) and best[0] == move.uci(),
            in_book=self._in_book(after),
            only_move=measured_before.only_move,
            sacrifice=any(tag.name == "sacrifice" for tag in tags),
            before_score=measured_before.score,
            opponent_erred=opponent_erred,
            obvious=obvious,
        )
        self._previous = (game_id, ply, loss, move.to_square, before.is_capture(move))
        assessment = MoveAssessment(
            game_id=game_id,
            ply=ply,
            uci=move.uci(),
            expected_loss=loss,
            severity=severity(loss, self.thresholds),
            best=best,
            tags=tags,
            depth=min(measured_before.depth, measured_after.depth or measured_before.depth),
            label=label,
            accuracy=move_accuracy(loss),
        )
        return assessment

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
        tags = self.tagger.tags_for_move(
            before, move, pv=pv, mate_in=_mate_in(before, move, after.lines)
        )
        # A tactic the move makes is only its point when the line after it cashes it in:
        # "the queen on e7 is loose" was said of a queen that simply steps away, 98 times
        # in the master games at moves whose note names no tactic (2026-09-24).
        line = [move, *pv]
        boards = [before.copy(stack=False)]
        for m in line:
            if m not in boards[-1].legal_moves:
                break
            nxt = boards[-1].copy(stack=False)
            nxt.push(m)
            boards.append(nxt)
        line = line[: len(boards) - 1]
        return [
            t for t in tags if t.name not in motifs.TARGETED or motifs.pays_off(boards, line, 0, t)
        ]

    def _in_book(self, board: chess.Board) -> bool:
        """Whether the game is still in theory after this move; needs the moves that got here."""
        if self.book is None or not board.move_stack:
            return False
        if board.root().fen() != chess.STARTING_FEN:
            return False
        return self.book.in_book(list(board.move_stack))

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


@dataclass
class _Request:
    game_id: str
    ply: int
    board: chess.Board
    move: chess.Move | None
    # Set when the game takes the move back while it is being measured (`Analyser.withdraw`):
    # nothing about it is delivered, and the position after it is not measured.
    withdrawn: bool = field(default=False, compare=False)


class Analyser:
    """A `GameAnnotator` on a worker thread: submit moves, poll events, never block.

    The engine is slow by design and the capture loop runs at ten frames a second, so
    nothing here waits for a result. Assessments arrive a moment after the move, which is
    when a coach would speak anyway.
    """

    def __init__(
        self, annotator: GameAnnotator, *, max_pending: int = 64, look_ahead: int = 0
    ) -> None:
        self.annotator = annotator
        self.max_pending = max_pending
        # How many of the best moves from the position on the board are looked ahead at
        # (LOOK_AHEAD in a live session; none unless asked).
        self.look_ahead = look_ahead
        self.dropped = 0
        self.failures = 0
        self.skipped = 0  # whole analyses not run for plies the game had already left
        self.withdrawn = 0  # requests let go because the game took their move back
        self._behind_until = 0.0  # the whole analysis is hurried until then
        self.last_error: str | None = None
        self._requests: queue.Queue[_Request | None] = queue.Queue()
        self._results: queue.SimpleQueue[Event] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        # Requests submitted and not yet answered (or dropped): counted, not read off
        # the queue, so `pending` is never 0 in the instant between a request leaving
        # the queue and the worker starting on it -- a review taken then would miss
        # the last verdict.
        self._outstanding = 0
        self._count_lock = threading.Lock()
        # The request the worker is on, so a takeback can let go of it.
        self._current: _Request | None = None
        # The positions to look ahead at, with how many moves ahead each is, and whether
        # the worker is looking ahead now (a request then interrupts it).
        self._ahead: list[tuple[chess.Board, int]] = []
        self._looking = False
        self.looked = 0  # positions measured ahead

    def __enter__(self) -> Analyser:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def pending(self) -> int:
        with self._count_lock:
            return self._outstanding

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="sidecoach-analyser", daemon=True)
        self._thread.start()

    def close(self, timeout: float = 5.0) -> None:
        thread, self._thread = self._thread, None
        if thread is not None:
            self._requests.put(None)
            if self._looking:
                self.annotator.engine.interrupt()
            thread.join(timeout)
        self.annotator.engine.close()

    def submit_move(self, game_id: str, before: chess.Board, move: chess.Move, ply: int) -> None:
        # With its moves: the book is asked with the moves that led here (`_in_book`,
        # `_opening`), and a copy without them was in the book on the first move alone.
        self._submit(_Request(game_id, ply, before.copy(), move))

    def submit_position(self, game_id: str, board: chess.Board, ply: int) -> None:
        self._submit(_Request(game_id, ply, board.copy(stack=False), None))

    def reset(self) -> None:
        """A new game: drop what is queued and forget the measured positions, since none
        of it is about this game. What the worker is busy with finishes and is delivered;
        the pipeline ignores an event for a game that is over."""
        while True:
            try:
                self._requests.get_nowait()
            except queue.Empty:
                break
            self._settle()
        self._ahead = []
        self.annotator.reset()

    def withdraw(self, game_id: str, ply: int) -> None:
        """The game's moves after `ply` taken back: what is queued for them is dropped, and one
        being measured is let go once its search ends -- nothing about it is delivered, and
        the position after it is not measured. The engine gets to the replacement move at
        once, where it used to measure every withdrawn move first (Tai, 2026-09-23:
        "takebacks make the program lag behind by a lot"; seconds on the phone)."""
        kept: list[_Request | None] = []
        while True:
            try:
                request = self._requests.get_nowait()
            except queue.Empty:
                break
            if request is not None and request.game_id == game_id and request.ply > ply:
                self.withdrawn += 1
                self._settle()
            else:
                kept.append(request)
        for request in kept:
            self._requests.put(request)
        self._ahead = []  # ahead of moves no longer played
        current = self._current
        if current is not None and current.game_id == game_id and current.ply > ply:
            current.withdrawn = True
            self.withdrawn += 1

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
                self._settle()
            except queue.Empty:
                pass
        with self._count_lock:
            self._outstanding += 1
        self._requests.put(request)
        # After the put: a worker that starts looking ahead after this sees the request.
        if self._looking:
            self.annotator.engine.interrupt()

    def _settle(self) -> None:
        with self._count_lock:
            self._outstanding = max(0, self._outstanding - 1)

    def _run(self) -> None:
        while True:
            request = self._next()
            if request is None:
                return
            self._current = request
            try:
                if request.move is None:
                    event = self.annotator.analysis_event(
                        request.game_id, request.board, request.ply
                    )
                    if not request.withdrawn:
                        self._results.put(event)
                        self._plan_ahead(request.board, event.lines)
                else:
                    # The verdict first, from the best line alone; the position whole after
                    # -- unless newer moves are waiting: then this ply is history, its
                    # verdict is all the coach and the review need, and the whole analysis
                    # (candidates, threats, the decisive-moment call) goes to the position
                    # the player is actually looking at.
                    verdict = self.annotator.assess_move(
                        request.game_id, request.board, request.move, request.ply
                    )
                    if request.withdrawn:
                        continue
                    self._results.put(verdict)
                    after = request.board.copy()
                    after.push(request.move)
                    behind = not self._requests.empty()
                    if behind:
                        self.skipped += 1
                        self._behind_until = time.monotonic() + BEHIND_S
                    hurry = time.monotonic() < self._behind_until
                    event = self.annotator.analysis_event(
                        request.game_id, after, request.ply, thin=behind, hurry=hurry
                    )
                    self._results.put(event)
                    if not (behind or hurry):
                        self._plan_ahead(after, event.lines)
            except Exception as exc:  # noqa: BLE001
                # A failed analysis must never end a session; the game goes on unassessed.
                self.failures += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._current = None
                self._settle()

    def _next(self) -> _Request | None:
        """The next request, and while none waits, the positions ahead measured one by one
        (`_look_ahead`), a request interrupting the one being measured."""
        while True:
            try:
                return self._requests.get_nowait()
            except queue.Empty:
                pass
            ahead = self._ahead
            if not ahead:
                return self._requests.get()
            self._look_ahead(*ahead.pop(0), ahead)

    def _plan_ahead(self, board: chess.Board, lines: list[Line]) -> None:
        """The positions to look ahead at from `board`, the position the game is at: after
        each of its `look_ahead` best moves first; after the best reply to each follows as
        each is measured (`_look_ahead`)."""
        ahead: list[tuple[chess.Board, int]] = []
        for line in lines[: self.look_ahead]:
            move = _moves(line)[:1]
            if move and move[0] in board.legal_moves:
                after = board.copy(stack=False)
                after.push(move[0])
                ahead.append((after, 1))
        self._ahead = ahead

    def _look_ahead(
        self, board: chess.Board, moves: int, ahead: list[tuple[chess.Board, int]]
    ) -> None:
        """One position ahead measured, unless a request is waiting or comes meanwhile; one
        move ahead, the best reply to it goes next in `ahead`, the plan it came from --
        unless the game has planned anew meanwhile (a new game, a takeback)."""
        engine = self.annotator.engine
        self._looking = True
        try:
            # Inside the interruptible searches, so a request put before this is seen here
            # and one put after it interrupts (`_submit`).
            with engine.interruptible():
                if not self._requests.empty():
                    return
                measured = self.annotator.look_ahead(board)
        except Interrupted:
            return  # the game moved: its own position comes first, and plans anew
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            return
        finally:
            self._looking = False
        self.looked += 1
        reply = _moves(measured.lines[0])[:1] if measured.lines else []
        if moves == 1 and reply and reply[0] in board.legal_moves and ahead is self._ahead:
            after = board.copy(stack=False)
            after.push(reply[0])
            ahead.insert(0, (after, 2))


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
    analyser = Analyser(annotator, look_ahead=LOOK_AHEAD)
    analyser.start()
    return analyser
