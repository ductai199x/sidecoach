"""The engine service for a game followed live: every position measured once, on a pool of
single-thread Stockfish workers, by one-line searches only (`annotator`'s docstring).

What the measurements decided (tools/engine/, the engine audit of 2026-10-04/05):
- one thread a search: more threads never answered a fixed depth sooner, at up to 17 times
  the processor time;
- one line a search: several lines to depth 18 explode on decided and forcing positions (up
  to 53 s), one line does not (p50 0.19 s, p99 1.1 s);
- no look-ahead beyond a position's candidates: with searches this cheap, searching further
  ahead bought hundredths of a second and threw away a third of its work. The candidates --
  each of the position's best moves searched as its own position -- are the positions the
  game goes to next.

A game position is measured in stages, each told to the readers as it comes, the next
replacing it: its first answer (its candidates at EARLY_DEPTH, or its best line while they are
not in yet), at once; the position whole -- its best line first, its other candidates from
their own positions' searches -- provisional; then with its side searches, final.

The workers take, first to last: what the next verdict needs (the position a move was played
from, whole; the position it led to, its best line); what the coach asks for (`request`); the
position the game is on; the game's other positions (their review is then a search of none of
them). A move is judged once the position it was played from is whole -- its candidates were
searched while its mover thought -- and the position it led to has its best line, or for a
search running past DECIDED_AFTER_S, a decided score at the deepest depth it finished (Tai,
2026-10-05: the severities this flips are as few as two depth-18 searches' own disagreement).
Verdicts go out in move order.

A position with one legal move is never searched: its best line is its move's, read from the
position that move leads to (`GameAnnotator.from_only_move`).

A search fails alone: what it held back is judged or told without it, never waited on; a move
it leaves no way to judge is told as not judged (`VerdictFailed`), in its place among the
verdicts, never skipped in silence; and nothing a worker does after a search can end the worker.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess

from sidecoach.analysis.annotator import (
    EARLY_DEPTH,
    GameAnnotator,
    Measured,
    cache_key,
    own_line,
)
from sidecoach.analysis.engine import Engine, Interrupted
from sidecoach.analysis.score import position_score, terminal_expected_score
from sidecoach.events import Event, Line, VerdictFailed

log = logging.getLogger(__name__)

# How many workers search side by side, one thread each: three answer as fast as four or six
# (tools/engine/workers_compare.py; `config.EngineSettings.workers`).
WORKERS = 3
# A search still running this long whose deepest finished depth -- at least EARLY_DEPTH --
# shows the position decided (an expected score at most DECIDED[0] or at least DECIDED[1]) has
# that depth's score stand in for the full depth's in a verdict (the module's docstring).
DECIDED_AFTER_S = 1.0
DECIDED = (0.15, 0.85)
# The positions kept, searched, for the game and any game after it (a puzzle tried again is
# answered at once); the oldest go past this.
KEEP_NODES = 20000
# How long the coach waits on the pool for a position (`PoolSearcher`): a whole position takes
# about a second, two on a phone; past this the coach answers from what is known.
REQUEST_TIMEOUT_S = 8.0

BEST = "best"  # one line to the full depth
HINT = "hint"  # the candidates at EARLY_DEPTH
SIDE = "side"  # the side searches of a whole position

# What a ply has been told, in order; a stage told is not told again for that position.
FIRST, FIRST_BOTH, WHOLE, FINAL = 1, 2, 3, 4


@dataclass(eq=False)
class _Node:
    """One position: its board (with the moves that led there, which the engine reads for
    repetitions), and what has been found of it."""

    key: str
    board: chess.Board
    # Its one legal move, when it has only one: its best line is that move's (never searched).
    only: chess.Move | None = None
    best: Measured | None = None
    hint: list[Line] | None = None
    # Its candidates, once its best line and its hint are in, and their positions' keys.
    candidates: list[tuple[chess.Move, str]] | None = None
    parents: set[str] = field(default_factory=set)  # the positions it is a candidate of
    whole: Measured | None = None  # its candidates' searches, no side searches
    final: Measured | None = None  # with its side searches
    stand_in: Measured | None = None  # a decided score while its best line is searched
    deepest: Line | None = None  # its best line's search: the deepest depth finished so far
    failed: set[str] = field(default_factory=set)

    @property
    def value(self) -> Measured | None:
        """What a verdict may read of it: its best line, or the decided stand-in."""
        return self.best or self.stand_in

    @property
    def first_done(self) -> bool:
        """Its best line and its hint are in (a failed one counts as in, with nothing)."""
        return (self.best is not None or BEST in self.failed) and (
            self.hint is not None or HINT in self.failed
        )

    @property
    def lost(self) -> bool:
        """Its best line and its hint both failed: nothing will be known of it."""
        return BEST in self.failed and HINT in self.failed


@dataclass
class _Move:
    ply: int
    before: str
    after: str
    move: chess.Move


@dataclass(frozen=True)
class _Job:
    kind: str  # BEST, HINT or SIDE
    key: str
    era: int


@dataclass(eq=False)
class _Request:
    """A position the coach asked for (`request`), the caller waiting on it, and the answer
    it was given."""

    key: str
    whole: bool  # its candidates too, or its best line alone
    done: threading.Event = field(default_factory=threading.Event)
    answer: Measured | None = None


class Analyser:
    """A game's positions measured on a pool of workers: submit positions and moves, poll
    events, never block (`request` alone waits, for the coach's questions). `engines`: one per
    worker (open or not); the annotator's own engine alone when not given. `clock` and `timer`
    time the decided rule (a test hands in its own)."""

    def __init__(
        self,
        annotator: GameAnnotator,
        *,
        engines: Sequence[Engine] | None = None,
        clock: Callable[[], float] = time.monotonic,
        timer: Callable[..., Any] = threading.Timer,
    ) -> None:
        self.annotator = annotator
        self.engines: list[Engine] = list(engines) if engines else [annotator.engine]
        self._clock = clock
        self._timer = timer
        self._cond = threading.Condition()
        self._nodes: dict[str, _Node] = {}
        self._game_id: str | None = None
        self._path: dict[int, str] = {}  # ply -> the game's position at it
        self._boards: dict[int, chess.Board] = {}  # ply -> the game's board, its moves with it
        self._ply_of: dict[str, int] = {}
        self._moves: dict[int, _Move] = {}
        self._current = 0
        self._next_verdict: int | None = None
        self._told: dict[int, tuple[str, int]] = {}  # ply -> (position, stage told)
        self._requests: list[_Request] = []
        self._running: dict[int, _Job] = {}  # by worker: the job a stop can reach
        self._since: dict[int, float] = {}  # by worker: when its running search began
        self._stopping: set[int] = set()  # workers told to stop, not yet back
        self._claimed: dict[int, _Job] = {}  # by worker: the job it took, running or about to
        self._era = 0
        self._results: queue.SimpleQueue[Event] = queue.SimpleQueue()
        self._threads: list[threading.Thread] = []
        self._closed = False
        # What it did, for the logs and the measures.
        self.searches = 0  # one-line searches to the full depth
        self.hints = 0
        self.side_searches = 0
        self.stopped = 0  # searches let go because nothing would read them
        self.withdrawn = 0
        self.failures = 0
        # The engine's own searches that failed (Stockfish stopped answering, a fault in it):
        # what the player is told of, once (the panel's `_engine_trouble`).
        self.search_failures = 0
        self.last_error: str | None = None

    # ---- the caller's side ----------------------------------------------------------------

    def start(self) -> None:
        if self._threads:
            return
        for index, engine in enumerate(self.engines):
            thread = threading.Thread(
                target=self._work,
                args=(index, engine),
                name=f"sidecoach-analyser-{index}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def __enter__(self) -> Analyser:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self, timeout: float = 5.0) -> None:
        with self._cond:
            self._closed = True
            for index in list(self._running):
                self.engines[index].interrupt()
            for request in self._requests:
                request.done.set()  # the coach waits no more: it has what is known
            self._requests.clear()
            self._cond.notify_all()
        for thread in self._threads:
            thread.join(timeout)
        self._threads = []
        for engine in self._unique_engines():
            engine.close()

    @property
    def pending(self) -> int:
        """What is still owed to the game: its positions not told all they will be, and its
        moves not judged."""
        with self._cond:
            owed = sum(1 for ply, key in self._path.items() if not self._settled(ply, key))
            if self._next_verdict is not None:
                owed += sum(1 for ply in self._moves if ply >= self._next_verdict)
            return owed

    def poll(self) -> list[Event]:
        out: list[Event] = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                return out

    def submit_position(self, game_id: str, board: chess.Board, ply: int) -> None:
        """A position the game stands on with no move to it: its start, or a resync."""
        with self._cond:
            self._game_id = game_id
            self._drop_after(ply)
            key = self._node(board).key
            move = self._moves.get(ply)
            if move is not None and move.after != key:
                # The move to this ply led elsewhere: it is not this game's any more.
                del self._moves[ply]
                self.withdrawn += 1
                if not self._moves:
                    self._next_verdict = None
            self._set_path(ply, key, board)
            self._current = ply
            # Told again: a position submitted is one the player faces anew (a takeback to the
            # start, a re-read), whatever was told of it before.
            self._told.pop(ply, None)
            self._tell(ply)
            self._settle()

    def submit_move(self, game_id: str, before: chess.Board, move: chess.Move, ply: int) -> None:
        with self._cond:
            self._game_id = game_id
            if move not in before.legal_moves:
                self.failures += 1
                self.last_error = f"{move.uci()} is not legal in {before.fen()}"
                return
            # What stood at this ply and after it is the game's no longer: a move played here
            # again replaces them (the glue withdraws them first; this holds without it).
            self._drop_after(ply - 1)
            source = self._node(before.copy())
            if self._path.get(ply - 1) != source.key:
                # The game is on another position than the one the move before led to: that
                # move is not this game's, and is not judged.
                gone = self._moves.pop(ply - 1, None)
                self.withdrawn += gone is not None
                self._set_path(ply - 1, source.key, before)
                self._tell(ply - 1)
            after = before.copy()
            after.push(move)
            target = self._node(after)
            self._set_path(ply, target.key, after)
            self._moves[ply] = _Move(ply, source.key, target.key, move)
            if self._next_verdict is None or ply < self._next_verdict:
                self._next_verdict = ply
            self._current = ply
            # The verdict first: the word on the move is the news, the position after it next.
            self._judge()
            self._tell(ply)
            self._stop_unwanted()
            self._settle()

    def withdraw(self, game_id: str, ply: int) -> None:
        """The game's moves after `ply` taken back: nothing more is said of them, and a search
        made only for them stops."""
        with self._cond:
            if game_id != self._game_id:
                return
            self.withdrawn += sum(1 for p in self._moves if p > ply)
            self._drop_after(ply)
            self._current = ply
            self._stop_unwanted()
            self._settle()

    def reset(self) -> None:
        """A new game, or the board re-read: the game's path let go, and what is under way for
        it stops; what was measured is kept, for a position seen again."""
        with self._cond:
            self._game_id = None
            self._path.clear()
            self._boards.clear()
            self._ply_of.clear()
            self._moves.clear()
            self._told.clear()
            self._next_verdict = None
            self._current = 0
            for node in self._nodes.values():
                node.failed.clear()  # a search that failed is tried again for a new game
            self._stop_unwanted()
            self.annotator.reset()
            self._settle()

    def request(
        self, board: chess.Board, *, whole: bool = True, timeout: float | None = None
    ) -> Measured | None:
        """The coach's question on a position (a move it or the player asks about, a position
        its words are about): the position whole -- its candidates at the full depth -- or its
        best line alone, waited for. Taken after the verdicts and before the game's own work.
        What is known by `timeout`, or when the analyser closes, is the answer then (None for
        nothing); asked again, what is in comes at once."""
        with self._cond:
            key = cache_key(board)
            if self._closed:
                return self._answer(key, whole)
            node = self._node(board)
            if self._answered(node, whole):
                return self._answer(key, whole)
            request = _Request(key, whole)
            self._requests.append(request)
            self._settle()
        try:
            request.done.wait(timeout)
        finally:
            with self._cond:
                if request in self._requests:
                    self._requests.remove(request)  # given up: nothing more is searched for it
                    self._stop_unwanted()
        with self._cond:
            return request.answer or self._answer(key, whole)

    def measured_path(self, game_id: str) -> list[Measured]:
        """The game's positions measured whole with their side searches, in the order of the
        game: what its review need not search again (`_Assessing.keep_for_review`). Nothing
        for a game this is not on."""
        with self._cond:
            if game_id != self._game_id:
                return []
            out = []
            for ply in sorted(self._path):
                node = self._nodes.get(self._path[ply])
                if node is not None and node.final is not None:
                    out.append(node.final)
            return out

    def forget_measured(self) -> None:
        """The settings a search depends on changed: nothing measured before counts, and a
        search under way is not kept (`GameAnnotator.forget_measured`). The game's positions
        are measured again; what was told of the ones it has left stands -- they are past, and
        telling them again would only repeat them to the readers."""
        with self._cond:
            self._era += 1
            # The game's positions and the coach's are kept, unmeasured; the rest go.
            kept = set(self._ply_of) | {r.key for r in self._requests}
            self._nodes = {k: self._fresh(k, n.board) for k, n in self._nodes.items() if k in kept}
            self._told = {p: (k, FINAL) for p, (k, _) in self._told.items() if p != self._current}
            for index in list(self._running):
                self.engines[index].interrupt()
            self.annotator.forget_measured()
            self._settle()

    def reconfigure(self, **options: Any) -> None:
        """Hash, the lines' length or the scoring saved mid-session: on every worker's engine."""
        for engine in self._unique_engines():
            for name, value in options.items():
                if name == "hash_mb":
                    engine.reconfigure(hash_mb=value)
                else:
                    setattr(engine, name, value)

    # ---- the positions --------------------------------------------------------------------

    def _fresh(self, key: str, board: chess.Board) -> _Node:
        """A node for `board`, nothing found of it yet: a finished position or one with more
        than one legal move as it is; one with a single legal move has that move, and nothing to
        hint (the move is the answer)."""
        node = _Node(key, board.copy())
        if terminal_expected_score(board) is None:
            legal = list(board.legal_moves)
            if len(legal) == 1:
                node.only = legal[0]
                node.hint = []
        return node

    def _node(self, board: chess.Board) -> _Node:
        key = cache_key(board)
        node = self._nodes.get(key)
        if node is None:
            node = self._nodes[key] = self._fresh(key, board)
        return node

    def _child(self, node: _Node, move: chess.Move, key: str) -> _Node:
        """The candidate `move`'s position, made again if it was let go (`_trim`)."""
        child = self._nodes.get(key)
        if child is None:
            board = node.board.copy()
            board.push(move)
            child = self._node(board)
        child.parents.add(node.key)
        return child

    def _trim(self) -> None:
        """Past KEEP_NODES, the oldest tenth of the positions let go -- but for the game's, the
        coach's, those a worker is on, their candidates and the forced moves' positions they
        take their best lines from. Run between the steps that choose
        candidates, never inside one; a candidate let go is made again when it is wanted."""
        if len(self._nodes) < KEEP_NODES:
            return
        roots = [
            *self._path.values(),
            *(k for m in self._moves.values() for k in (m.before, m.after)),
            *(r.key for r in self._requests),
            *(j.key for j in self._claimed.values()),
        ]
        keep = set(roots)
        for key in roots:
            node = self._nodes.get(key)
            if node is not None:
                for wanted in [key, *(child for _, child in node.candidates or [])]:
                    keep.update(self._forced(wanted, make=False))
        for key in list(self._nodes)[: KEEP_NODES // 10]:
            if key in keep:
                continue
            node = self._nodes.pop(key)
            for _, child in node.candidates or []:
                if child in self._nodes:
                    self._nodes[child].parents.discard(key)

    def _set_path(self, ply: int, key: str, board: chess.Board) -> None:
        """The game's position at `ply` is `key`, `board` as the game reached it: the one it
        held before, if another, is no longer the game's -- nothing more is told of it here."""
        old = self._path.get(ply)
        if old is not None and old != key:
            if self._ply_of.get(old) == ply:
                del self._ply_of[old]
            self._told.pop(ply, None)
        self._path[ply] = key
        self._boards[ply] = board.copy()
        self._ply_of[key] = ply

    def _drop_after(self, ply: int) -> None:
        """The game's plies after `ply` let go: their positions, their moves, what was told."""
        for p in [p for p in self._path if p > ply]:
            key = self._path.pop(p)
            self._boards.pop(p, None)
            if self._ply_of.get(key) == p:
                del self._ply_of[key]
            self._told.pop(p, None)
        for p in [p for p in self._moves if p > ply]:
            del self._moves[p]
            self.annotator.forget_assessed(p)
        if self._next_verdict is not None and self._next_verdict > ply + 1:
            self._next_verdict = ply + 1
        if not self._moves:
            self._next_verdict = None

    def _candidates_of(self, node: _Node) -> list[tuple[chess.Move, str]]:
        """The position's candidates and their positions' keys: chosen once -- its one legal
        move at once, else when its best line and its hint are in -- each candidate's position
        made a node that knows it."""
        if node.candidates is None and (node.only is not None or node.first_done):
            if node.only is not None:
                moves = [node.only]
            else:
                best = node.best or self._empty(node)
                moves = self.annotator.candidates(node.board, best, node.hint or [])
            chosen = []
            for move in moves:
                child = node.board.copy()
                child.push(move)
                child_node = self._node(child)
                child_node.parents.add(node.key)
                chosen.append((move, child_node.key))
            node.candidates = chosen
        return node.candidates or []

    @staticmethod
    def _empty(node: _Node) -> Measured:
        return Measured(node.board.fen(), [], 0.5, False, False, key=node.key)

    def _forced(self, key: str, *, make: bool = True) -> list[str]:
        """`key`, and the positions a chain of single legal moves from it leads to: a position
        with one legal move takes its best line from the next (`_jobs_for`), so the searches it
        waits on are theirs, however many forced moves down. `make`: the chain's positions made
        as `_jobs_for` makes them; without it (`_trim`), only those already made."""
        out = [key]
        node = self._nodes.get(key)
        while node is not None and node.only is not None:
            candidates = self._candidates_of(node) if make else node.candidates or []
            if not candidates or candidates[0][1] in out:
                break
            key = candidates[0][1]
            out.append(key)
            node = self._nodes.get(key)
        return out

    def _wanted(self) -> set[str]:
        """Every position a search is wanted for: the game's, what the coach asked for, their
        candidates, and the forced moves' positions each of those takes its best line from."""
        out: set[str] = set()
        for key in [*self._path.values(), *(r.key for r in self._requests)]:
            node = self._nodes.get(key)
            if node is None:
                continue
            for wanted in [key, *(child for _, child in self._candidates_of(node))]:
                out.update(self._forced(wanted))
        return out

    # ---- what the workers take ---------------------------------------------------------------

    def _jobs_for(self, node: _Node, *, whole: bool, side: bool) -> list[_Job]:
        """The searches `node` still needs, in order: its best line (a single legal move's,
        from its position's); for `whole`, its hint and its candidates' best lines; for `side`,
        its side searches once it is whole."""
        jobs = []
        if node.only is not None:
            # Its best line is its move's position's: that search is its own.
            for move, key in self._candidates_of(node):
                child = self._child(node, move, key)
                if node.best is None:
                    jobs += self._jobs_for(child, whole=False, side=False)
        elif node.best is None and BEST not in node.failed:
            jobs.append(_Job(BEST, node.key, self._era))
        if not whole:
            return jobs
        if node.hint is None and HINT not in node.failed:
            jobs.append(_Job(HINT, node.key, self._era))
        for move, key in self._candidates_of(node):
            child = self._child(node, move, key)
            if child.best is None and BEST not in child.failed:
                jobs += self._jobs_for(child, whole=False, side=False)
        if side and node.whole is not None and node.final is None and SIDE not in node.failed:
            jobs.append(_Job(SIDE, node.key, self._era))
        return jobs

    def _agenda(self) -> list[_Job]:
        """Every search wanted, first to last (the module's docstring): the verdicts' needs,
        the coach's requests, the position the game is on, the game's other positions."""
        out: list[_Job] = []
        if self._next_verdict is not None:
            for ply in sorted(p for p in self._moves if p >= self._next_verdict):
                move = self._moves[ply]
                out += self._jobs_for(self._nodes[move.before], whole=True, side=False)
                out += self._jobs_for(self._nodes[move.after], whole=False, side=False)
        for request in self._requests:
            out += self._jobs_for(self._nodes[request.key], whole=request.whole, side=False)
        current = self._path.get(self._current)
        if current is not None:
            out += self._jobs_for(self._nodes[current], whole=True, side=True)
        for ply in sorted(self._path, reverse=True):
            out += self._jobs_for(self._nodes[self._path[ply]], whole=True, side=True)
        return out

    def _pick(self) -> _Job | None:
        busy = {(j.kind, j.key) for j in self._claimed.values()}
        return next((job for job in self._agenda() if (job.kind, job.key) not in busy), None)

    def _work(self, index: int, engine: Engine) -> None:
        while True:
            with self._cond:
                job = None
                while not self._closed:
                    try:
                        job = self._pick()
                    except Exception as exc:  # noqa: BLE001 - a bug here must not end the worker
                        self._failed("choosing a search", exc)
                        self._cond.wait(1.0)  # not a hot loop on a fault that stays
                        continue
                    if job is not None:
                        break
                    self._cond.wait()
                if self._closed or job is None:
                    return
                # Taken the moment it is picked: no other worker picks it meanwhile.
                self._claimed[index] = job
                node = self._nodes[job.key]
                board = node.board.copy()
                whole = node.whole
            found: Measured | list[Line] | None = None
            try:
                found = self._run(index, engine, job, board, whole)
            except Interrupted:
                pass
            except Exception as exc:  # noqa: BLE001 - a failed search must never end a session
                with self._cond:
                    self._failed(f"the {job.kind} search of {board.fen()}", exc)
                    self.search_failures += 1
                    failed = self._nodes.get(job.key)
                    if failed is not None and job.era == self._era:
                        failed.failed.add(job.kind)
                        self._guarded(self._advance, failed)
            with self._cond:
                self._running.pop(index, None)
                self._since.pop(index, None)
                self._stopping.discard(index)
                self._claimed.pop(index, None)
                if found is not None and job.era == self._era and job.key in self._nodes:
                    self._guarded(self._found, job, found)
                self._guarded(self._trim)
                self._cond.notify_all()

    def _guarded(self, step: Callable[..., None], *args: Any) -> None:
        """`step` run under the lock; a fault in it counted and logged, never the worker's end."""
        try:
            step(*args)
        except Exception as exc:  # noqa: BLE001 - nothing after a search may end the worker
            self._failed(getattr(step, "__name__", "a step"), exc)

    def _failed(self, what: str, exc: BaseException) -> None:
        log.warning("analyser: %s failed: %r", what, exc)
        self.failures += 1
        self.last_error = f"{type(exc).__name__}: {exc}"

    def _run(
        self,
        index: int,
        engine: Engine,
        job: _Job,
        board: chess.Board,
        whole: Measured | None,
    ) -> Measured | list[Line] | None:
        """One job's search, stoppable from the moment it is running (`_stop_unwanted`);
        None when it was no longer wanted by then."""
        with engine.interruptible():
            with self._cond:
                if self._closed or job.era != self._era or not self._still_wanted(job):
                    return None
                self._running[index] = job  # inside the stoppable block: a stop reaches it
                self._since[index] = self._clock()
            if job.kind == HINT:
                return self.annotator.hint_lines(board, engine=engine)
            if job.kind == SIDE:
                assert whole is not None
                return self.annotator.with_side(board, whole, engine=engine)
            timer = self._timer(DECIDED_AFTER_S, self._decide, args=(job, index))
            timer.daemon = True
            timer.start()
            try:
                return self.annotator.best_line(
                    board,
                    engine=engine,
                    on_depth=lambda lines, job=job, index=index: self._deepened(job, index, lines),
                )
            finally:
                timer.cancel()

    def _still_wanted(self, job: _Job) -> bool:
        node = self._nodes.get(job.key)
        if node is None:
            return False
        done = {BEST: node.best, HINT: node.hint, SIDE: node.final}[job.kind]
        return done is None and job.key in self._wanted()

    def _stop_unwanted(self) -> None:
        wanted = self._wanted()
        for index, job in list(self._running.items()):
            if job.key not in wanted and index not in self._stopping:
                self._stopping.add(index)
                self.stopped += 1
                self.engines[index].interrupt()

    def _settle(self) -> None:
        """After a change from outside: the positions trimmed, the workers woken."""
        self._guarded(self._trim)
        self._cond.notify_all()

    # ---- what comes back ----------------------------------------------------------------

    def _deepened(self, job: _Job, index: int, lines: list[Line]) -> None:
        """A depth of a best line's search finished, the search going on: kept, and past
        DECIDED_AFTER_S read for a decided score."""
        with self._cond:
            node = self._nodes.get(job.key)
            if node is None or job.era != self._era or not lines:
                return
            node.deepest = lines[0]
            since = self._since.get(index)
            if since is not None and self._clock() - since >= DECIDED_AFTER_S:
                self._guarded(self._stand_in, node)

    def _decide(self, job: _Job, index: int) -> None:
        """DECIDED_AFTER_S into a best line's search (the timer): its deepest finished depth
        read for a decided score, as each depth after it is (`_deepened`)."""
        with self._cond:
            node = self._nodes.get(job.key)
            if node is not None and job.era == self._era and self._running.get(index) == job:
                self._guarded(self._stand_in, node)

    def _stand_in(self, node: _Node) -> None:
        """A decided score at the deepest depth finished stands in for the full depth's in a
        verdict (the module's docstring); nothing for a position still open, or too shallow."""
        line = node.deepest
        if node.best is not None or line is None or line.depth < EARLY_DEPTH:
            return
        if DECIDED[0] < line.expected_score < DECIDED[1]:
            return
        node.stand_in = Measured(
            fen=node.board.fen(),
            lines=[line],
            score=position_score(node.board, [line]),
            decisive=False,
            only_move=False,
            key=node.key,
        )
        self._carry_stand_in(node)
        self._judge()
        self._cond.notify_all()

    def _carry_stand_in(self, node: _Node) -> None:
        """`node`'s decided stand-in, carried as its best line is (`_carry_only`): a position
        whose one legal move leads here, and so on back up a forced sequence, stands in with
        it -- a verdict on the move that led into the forced sequence reads that position."""
        if node.stand_in is None:
            return
        for key in list(node.parents):
            parent = self._nodes.get(key)
            if parent is None or parent.only is None or parent.best is not None:
                continue
            parent.stand_in = self.annotator.from_only_move(
                parent.board, parent.only, node.stand_in
            )
            self._carry_stand_in(parent)

    def _found(self, job: _Job, found: Measured | list[Line]) -> None:
        node = self._nodes[job.key]
        if job.kind == BEST:
            assert isinstance(found, Measured)
            node.best = found
            self.searches += 1
        elif job.kind == HINT:
            assert isinstance(found, list)
            node.hint = found
            self.hints += 1
        else:
            assert isinstance(found, Measured)
            node.final = found
            self.side_searches += 1
        self._advance(node)

    def _advance(self, node: _Node) -> None:
        """A search of `node` done (or failed): a single legal move's position it is the
        candidate of given its best line, it and every position it is a candidate of made whole
        if they can be, and what waited on them told, judged and answered."""
        # Every position given a best line here -- this one, and those up a forced sequence --
        # and the positions each is a candidate of.
        for changed in [node, *self._carry_only(node)]:
            for key in [changed.key, *changed.parents]:
                whole = self._nodes.get(key)
                if whole is not None and whole.whole is None:
                    self._complete(whole)
        if SIDE in node.failed and node.whole is not None and node.final is None:
            node.final = node.whole
        # The positions moves are judged from told before the verdicts (a verdict's reader finds
        # the position it was judged from told), the position a move led to after its verdict
        # (the word on the move is the news).
        waiting = self._next_verdict
        for ply in sorted(self._path):
            if waiting is None or ply < waiting:
                self._tell(ply)
        self._judge()
        for ply in sorted(self._path):
            self._tell(ply)
        self._answer_requests()

    def _carry_only(self, node: _Node) -> list[_Node]:
        """`node`'s best line in (or failed): each position whose one legal move leads here
        takes its best line from it, and so on back up a forced sequence. The positions it
        reached."""
        reached: list[_Node] = []
        for key in list(node.parents):
            parent = self._nodes.get(key)
            if parent is None or parent.only is None or parent.best is not None:
                continue
            if node.best is not None:
                parent.best = self.annotator.from_only_move(parent.board, parent.only, node.best)
            elif BEST in node.failed:
                parent.failed.add(BEST)
            else:
                continue
            reached += [parent, *self._carry_only(parent)]
        return reached

    def _complete(self, node: _Node) -> None:
        """`node` made whole once its candidates are all in (a failed one left out) -- but for
        its best line's own move: that line is its own search's (`GameAnnotator.whole`), and
        its move's position, searched all the same as the game's likeliest next, is not waited
        for."""
        candidates = self._candidates_of(node)
        if not node.first_done or node.lost:
            return
        own = node.best.lines[0].moves[:1] if node.best is not None and node.best.lines else []
        children = [
            (move, self._child(node, move, key)) for move, key in candidates if [move.uci()] != own
        ]
        if any(c.best is None and BEST not in c.failed for _, c in children):
            return
        node.whole = self.annotator.whole(
            node.board,
            node.best or self._empty(node),
            [(move, c.best) for move, c in children if c.best is not None],
        )
        self.annotator.computed.add(node.board, node.whole.lines)
        if not self.annotator.with_threats or terminal_expected_score(node.board) is not None:
            node.final = node.whole

    def _judge(self) -> None:
        """Each move judged as soon as it can be, in the order of the moves: the move before
        is what a verdict reads ("the opponent just erred", "a recapture")."""
        while self._next_verdict is not None:
            if self._next_verdict not in self._moves:
                later = [p for p in self._moves if p > self._next_verdict]
                if not later:
                    return
                self._next_verdict = min(later)  # a gap in the numbering: the next move's
            move = self._moves[self._next_verdict]
            before, after = self._nodes[move.before], self._nodes[move.after]
            if before.whole is None:
                if before.lost:
                    # Nothing to judge it from: told as not judged, never skipped in silence.
                    self._unjudged(move, "the position it was played from could not be searched")
                    continue
                return
            own = own_line(before.whole.lines, move.move) is not None
            if not own and after.value is None:
                if BEST in after.failed:
                    self._unjudged(move, "the position it led to could not be searched")
                    continue
                return
            board = self._boards.get(move.ply - 1, before.board)
            try:
                verdict = self.annotator.assess_from(
                    self._game_id or "",
                    board,
                    move.move,
                    move.ply,
                    before.whole,
                    None if own else after.value,
                )
            except Exception as exc:  # noqa: BLE001 - a move that cannot be judged is told so
                self._failed(f"judging {move.move.uci()} at ply {move.ply}", exc)
                self._unjudged(move, f"judging it failed: {type(exc).__name__}: {exc}")
                continue
            self._results.put(verdict)
            self._next_verdict += 1

    def _unjudged(self, move: _Move, why: str) -> None:
        """`move` told as not judged (`VerdictFailed`), in its place among the verdicts, and the
        verdicts carried on from the next."""
        reason = why + (f" ({self.last_error})" if self.last_error else "")
        self._results.put(
            VerdictFailed(
                game_id=self._game_id or "", ply=move.ply, uci=move.move.uci(), reason=reason
            )
        )
        self._next_verdict = move.ply + 1

    def _answered(self, node: _Node, whole: bool) -> bool:
        if whole:
            return node.whole is not None or node.lost
        return node.best is not None or BEST in node.failed

    def _answer(self, key: str, whole: bool) -> Measured | None:
        node = self._nodes.get(key)
        if node is None:
            return None
        if whole and node.whole is not None:
            return node.final or node.whole
        return node.best

    def _answer_requests(self) -> None:
        left = []
        for request in self._requests:
            node = self._nodes.get(request.key)
            if node is None or self._answered(node, request.whole):
                request.answer = self._answer(request.key, request.whole)
                request.done.set()
            else:
                left.append(request)
        self._requests = left

    def _settled(self, ply: int, key: str) -> bool:
        """Whether the game's position at `ply` has been told all it will be."""
        told = self._told.get(ply)
        return told is not None and told[0] == key and told[1] == FINAL

    def _tell(self, ply: int) -> None:
        """The game's position at `ply` told as far as it has come (the module's docstring),
        each stage once. The event is made before the stage is marked told: a fault making it
        leaves the stage to be told again, never marked and never said."""
        key = self._path.get(ply)
        if key is None or self._game_id is None:
            return
        node = self._nodes[key]
        stage = self._stage(node)
        told = self._told.get(ply)
        if stage == 0 or (told is not None and told[0] == key and told[1] >= stage):
            return
        if stage == FINAL:
            measured = node.final
        elif stage == WHOLE:
            measured = node.whole
        else:
            measured = self.annotator.first_answer(node.board, node.best, node.hint or [])
        board = self._boards.get(ply, node.board)
        event = None
        if measured is not None:
            event = self.annotator.analysis_from(
                self._game_id, board, ply, measured, provisional=stage < FINAL
            )
        self._told[ply] = (key, stage)
        if event is not None:
            self._results.put(event)

    @staticmethod
    def _stage(node: _Node) -> int:
        """How far `node` has come (a position whose searches all failed is settled, with
        nothing to tell)."""
        if node.final is not None or node.lost:
            return FINAL
        if node.whole is not None:
            return WHOLE
        if node.best is not None and node.hint:
            return FIRST_BOTH
        return FIRST if node.best is not None or node.hint else 0

    def _unique_engines(self) -> list[Engine]:
        seen: list[Engine] = []
        for engine in self.engines:
            if not any(engine is e for e in seen):
                seen.append(engine)
        return seen


class PoolSearcher:
    """The coach's engine (`computed.ComputedFirst`): its questions answered by the analyser's
    pool -- the position whole, its candidates at the full depth -- never by a Stockfish of
    its own (the engine audit, 2026-10-05: the coach's second engine searched three lines,
    which explode as six do). The depth is the analyser's; the wait is bounded by `timeout`.
    What it answers the analyser has kept already (`fills_computed`)."""

    fills_computed = True

    def __init__(self, analyser: Analyser, *, timeout: float | None = REQUEST_TIMEOUT_S) -> None:
        self.analyser = analyser
        self.timeout = timeout

    def analyse(
        self, board: chess.Board, *, depth: int | None = None, best_only: bool = False
    ) -> list[Line]:
        """`board`'s lines, whole -- or with `best_only` its best line, not waiting for the
        whole; `depth` is the analyser's own, whatever is asked."""
        measured = self.analyser.request(board, whole=not best_only, timeout=self.timeout)
        return list(measured.lines) if measured is not None else []

    def close(self) -> None:
        """The pool is the analyser's, and closes with it."""


def open_analyser(
    engine_path: object,
    *,
    depth: int,
    multipv: int,
    workers: int = WORKERS,
    hash_mb: int,
    pv_plies: int,
    thresholds: Any,
    score_model: str = "human",
    tagger: Any = None,
    book: Any = None,
    make_engine: Callable[..., Engine] = Engine,
) -> Analyser:
    """An analyser ready to take moves, its workers' engines started side by side (each loads
    its network: one after another, the first move waited for all of them), one thread each."""
    engines = [
        make_engine(
            path=Path(str(engine_path)),
            threads=1,
            hash_mb=hash_mb,
            multipv=multipv,
            pv_plies=pv_plies,
            score_model=score_model,
        )
        for _ in range(max(1, workers))
    ]
    errors: list[BaseException] = []

    def open_one(engine: Engine) -> None:
        try:
            engine.open()
        except Exception as exc:  # noqa: BLE001 - said once all have tried, below
            errors.append(exc)

    openers = [threading.Thread(target=open_one, args=(e,), daemon=True) for e in engines]
    for opener in openers:
        opener.start()
    for opener in openers:
        opener.join()
    if errors:
        for engine in engines:
            engine.close()
        raise errors[0]
    annotator = GameAnnotator(
        engine=engines[0],
        depth=depth,
        multipv=multipv,
        thresholds=thresholds,
        tagger=tagger,
        book=book,
    )
    analyser = Analyser(annotator, engines=engines)
    analyser.start()
    return analyser


__all__ = [
    "DECIDED_AFTER_S",
    "REQUEST_TIMEOUT_S",
    "WORKERS",
    "Analyser",
    "PoolSearcher",
    "open_analyser",
]
