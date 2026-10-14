"""Stockfish over UCI: one persistent process, MultiPV lines, real win-draw-loss numbers.

The engine is the only part of the program that is allowed to be slow, so it never runs on
the capture thread; `sidecoach.analysis.annotator` owns the worker that drives it.

Every line comes back from the mover's point of view, which is what the coach reasons in:
`expected_score` is (wins + draws / 2) / 1000, so 0.5 is a dead draw and 1.0 is a win,
whoever is to move. Centipawns are carried alongside for display only. The design forbids
thresholds in centipawns, because 50 centipawns at +8 is nothing and at 0.0 is a game.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import chess.engine

from sidecoach.analysis.score import expected_score, game_over
from sidecoach.events import Line

log = logging.getLogger(__name__)

# A principal variation can run thirty plies; nobody reads that far and events stay small.
DEFAULT_PV_PLIES = 12
# The best line is read for what it is for -- the mate, the piece won, the pawn queened --
# and a puzzle's point can come seven moves in (a queen won on ply 11, a pawn queening on
# ply 15, in a sample of Lichess puzzles). Stockfish's own line often stops short (a hash
# hit cuts it to three plies), so the best line is kept to this length, and a shorter one
# carried on by searching from its end, whatever the other lines are cut to (Tai,
# 2026-09-23: "at least 8 to 10 ply since some puzzles and hard positions are really deep").
MIN_BEST_PLIES = 16
# How deep, and for how long at most, each search that carries the line on looks.
EXTEND_DEPTH = 14
EXTEND_SECONDS = 0.5
# How many searches may carry one line on.
EXTEND_TRIES = 3
# python-chess's model for turning centipawns into win-draw-loss when an engine will not.
WDL_MODEL = "sf"
DEFAULT_SCORE_MODEL = "human"


def _lower_priority(engine: chess.engine.SimpleEngine) -> None:
    """The engine's process below the player's own work (POSIX), once it has started: set in
    the child before it ran (`preexec_fn`) is unsafe in a process with threads, as the sidecar
    is -- the child can deadlock on a lock another thread held (the Linux audit, 2026-10-13)."""
    if sys.platform == "win32":
        return
    try:
        pid = engine.transport.get_pid()
        os.setpriority(os.PRIO_PROCESS, pid, 5)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - the engine runs all the same, at normal priority
        log.warning("the engine's priority was not lowered: %s", exc)


def _quit(engine: chess.engine.SimpleEngine) -> None:
    """The process ended: asked to quit, and killed if that fails -- a process wedged without
    reading its input would otherwise outlive its engine, and the app."""
    try:
        engine.quit()
    except Exception:  # noqa: BLE001 - a dead engine needs no goodbye
        # By its PID, as `Engine._end_process` does: the transport is its event loop's, not
        # this thread's, and the process's handle is held until `close`, so the PID is its own.
        with contextlib.suppress(Exception):
            kill = signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
            os.kill(engine.transport.get_pid(), kill)
        with contextlib.suppress(Exception):
            engine.close()


class EngineUnavailable(RuntimeError):
    """The engine binary is missing, or died and could not be restarted."""


class Interrupted(RuntimeError):
    """A search inside `Engine.interruptible` was stopped by `Engine.interrupt`."""


@dataclass
class Engine:
    """A Stockfish process that outlives the positions it is asked about.

    A crashed engine is restarted once per call: a live coaching session must survive an
    engine that dies, and the position it was asked about is still on the board.
    """

    path: Path
    threads: int = 4
    hash_mb: int = 256
    multipv: int = 6
    pv_plies: int = DEFAULT_PV_PLIES
    # How a line's score becomes an expected score; see `analysis.score.expected_score`.
    score_model: str = DEFAULT_SCORE_MODEL
    # A depth asked for is a depth reached: no time cap cuts a search short. A cap of five
    # seconds answered 63 of 1,642 searches of a session below depth 18, some at depth 11,
    # and they were read as depth 18 (the engine audit, 2026-10-04; Tai: no lower depths).
    # Work the player is not waiting on (the background review of imported games) yields
    # the processor to everything else: the process runs below normal priority.
    low_priority: bool = False
    # An engine that for this long neither says anything nor uses the processor is hung, not
    # thinking: its process is ended and the search made again (`_watched`). Silence alone is
    # not enough: a search reports each depth and each root move it takes up, but one root
    # move can take most of a search -- a real position (three moves, in check) went 48 s
    # without a word at depth 18, still searching, and was killed (the engine audit,
    # 2026-10-04). A search still working, however long it takes, is never cut.
    watchdog_s: float = 30.0
    # How the watchdog waits: until the search is done (True) or the patience has run (False).
    # A test drives it without waiting on the clock.
    watchdog_wait: Callable[[threading.Event, float], bool] = field(
        default=lambda done, seconds: done.wait(seconds), repr=False
    )
    # The engine process's processor time so far, in seconds (None when it cannot be read):
    # what tells a search working in silence from a hung one. A test gives its own.
    cpu_seconds: Callable[[], float | None] | None = field(default=None, repr=False)
    _engine: chess.engine.SimpleEngine | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _wdl_from_engine: bool = field(default=False, init=False, repr=False)
    # Inside `interruptible`: the search running, which `interrupt` stops, and whether it has.
    _interruptible: bool = field(default=False, init=False, repr=False)
    _stopped: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _search: chess.engine.SimpleAnalysisResult | None = field(default=None, init=False, repr=False)
    # Closed by its owner: a search under way (a request's next one, a crash's restart)
    # does not start another process nobody would close. `open` opens it again.
    _closed: bool = field(default=False, init=False, repr=False)
    # Taken to set `_engine` or `_closed`, never across a search or a process's start: a
    # process started as the engine closes is quit, not kept by a closed engine for good.
    _life: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    # Threads or hash saved while a search ran: set on the process before the next search.
    _options_owed: bool = field(default=False, init=False, repr=False)
    # How many reports the engine has made: the watchdog reads it to tell a search that is
    # talking from one that has gone silent.
    _heard: int = field(default=0, init=False, repr=False)

    def __enter__(self) -> Engine:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def name(self) -> str:
        """What the engine calls itself, for the doctor and the logs."""
        self._start()
        assert self._engine is not None
        return str(self._engine.id.get("name", self.path.name))

    def open(self) -> None:
        with self._life:
            self._closed = False
        self._start()

    def _start(self) -> None:
        if self._closed:
            raise EngineUnavailable("the engine was closed")
        if self._engine is not None:
            return
        popen: dict[str, Any] = {}
        if self.low_priority and sys.platform == "win32":
            popen["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
        try:
            engine = chess.engine.SimpleEngine.popen_uci(str(self.path), **popen)
        except Exception as exc:  # noqa: BLE001
            raise EngineUnavailable(f"cannot start {self.path}: {exc}") from exc
        if self.low_priority:
            _lower_priority(engine)
        options: dict[str, str | int | bool | None] = {}
        if "Threads" in engine.options:
            options["Threads"] = self.threads
        if "Hash" in engine.options:
            options["Hash"] = self.hash_mb
        # Real win-draw-loss numbers beat a model fitted to centipawns; ask for them when the
        # engine has them, and fall back to python-chess's model when it does not.
        self._wdl_from_engine = "UCI_ShowWDL" in engine.options
        if self._wdl_from_engine:
            options["UCI_ShowWDL"] = True
        if options:
            engine.configure(options)
        with self._life:
            closed = self._closed
            if not closed:
                self._engine = engine
        if closed:
            # Closed while it started (the panel stopped as a crashed engine restarted): no
            # one would ever quit it.
            _quit(engine)
            raise EngineUnavailable("the engine was closed")

    def reconfigure(self, *, threads: int | None = None, hash_mb: int | None = None) -> None:
        """Threads and hash on the running engine (the settings saved mid-session). Never
        during a search: a `setoption` stops the search under way, which was then kept as if
        it had finished (0.6 s and depth 14 of a 4 s search). With a search running they are
        set before the next one, and the save does not wait for it."""
        if threads is not None:
            self.threads = threads
        if hash_mb is not None:
            self.hash_mb = hash_mb
        if not self._lock.acquire(blocking=False):
            self._options_owed = True
            return
        try:
            self._configure()
        finally:
            self._lock.release()

    def _configure(self) -> None:
        """Threads and hash set on the process; under the lock, between searches."""
        self._options_owed = False
        if self._engine is None:
            return
        options: dict[str, str | int | bool | None] = {}
        if "Threads" in self._engine.options:
            options["Threads"] = self.threads
        if "Hash" in self._engine.options:
            options["Hash"] = self.hash_mb
        if options:
            with contextlib.suppress(Exception):
                self._engine.configure(options)

    def close(self) -> None:
        with self._life:
            self._closed = True
        self._drop()

    def _drop(self) -> None:
        """The process gone (a crash's restart starts another; `close` does not)."""
        with self._life:
            engine, self._engine = self._engine, None
        if engine is not None:
            _quit(engine)

    def analyse(
        self,
        board: chess.Board,
        *,
        depth: int | None = None,
        time_s: float | None = None,
        multipv: int | None = None,
        pv_plies: int | None = None,
        carry_best: bool = False,
        on_depth: Callable[[list[Line]], None] | None = None,
    ) -> list[Line]:
        """The best lines for `board`, best first, from the side to move's point of view.
        `carry_best`: a best line shorter than MIN_BEST_PLIES is carried on to it -- for
        the coach, who reads the line for its point; a verdict needs only the score.
        `on_depth`: handed each depth's lines as the search finishes it, the search going on.

        A finished position has no lines: there is nothing to play. Callers read its value
        from the result instead, with `sidecoach.analysis.score.terminal_expected_score`.
        """
        if game_over(board):
            return []
        if depth is None and time_s is None:
            raise ValueError("analyse needs a depth or a time limit")
        limit = chess.engine.Limit(depth=depth, time=time_s)
        wanted = multipv if multipv is not None else self.multipv
        wanted = max(1, min(wanted, board.legal_moves.count()))
        plies = pv_plies if pv_plies is not None else self.pv_plies

        def as_lines(infos: list[chess.engine.InfoDict], carry: bool) -> list[Line]:
            lines = [
                _line(
                    info,
                    board,
                    max(plies, MIN_BEST_PLIES) if carry and i == 0 else plies,
                    self.score_model,
                )
                for i, info in enumerate(infos)
            ]
            return [line for line in lines if line.moves]

        depths = None if on_depth is None else (lambda infos: on_depth(as_lines(infos, False)))
        with self._lock:
            if self._options_owed:
                self._configure()
            with self._watched():
                infos = self._analyse_with_restart(board, limit, wanted, depths)
        lines = as_lines(infos, carry_best)
        if carry_best and lines and len(lines[0].moves) < MIN_BEST_PLIES:
            lines[0] = self._carried_on(board, lines[0], MIN_BEST_PLIES)
        return lines

    def _carried_on(self, board: chess.Board, line: Line, plies: int) -> Line:
        """`line` carried on by searches from its end until it is `plies` long or the game
        is over there; its score and depth stay the first search's."""
        moves = list(line.moves)
        end = board.copy()  # with its moves: a repetition on the way ends the line
        for uci in moves:
            end.push(chess.Move.from_uci(uci))
        for _ in range(EXTEND_TRIES):
            if len(moves) >= plies or game_over(end):
                break
            limit = chess.engine.Limit(depth=EXTEND_DEPTH, time=EXTEND_SECONDS)
            with self._lock:
                if self._options_owed:
                    self._configure()
                with self._watched():
                    infos = self._analyse_with_restart(end, limit, 1)
            more = (infos[0].get("pv") or []) if infos else []
            if not more:
                break
            for move in more[: plies - len(moves)]:
                moves.append(move.uci())
                end.push(move)
        return line.model_copy(update={"moves": moves})

    @contextlib.contextmanager
    def _watched(self) -> Iterator[None]:
        """A search watched: one that for a whole `watchdog_s` reports nothing and uses
        (almost) no processor time is an engine that stopped answering without dying -- the
        search would wait on it for good, and its worker with it -- so its process is ended,
        and the search is made again on a new one (`_analyse_with_restart`), watched the same
        way. A search still reporting or still working, however long it takes, is left to
        finish: the depth asked for is the depth answered. A window it cannot judge is never
        an engine's end: the processor time unreadable, the window's start a process since
        replaced (its baseline another's), or a window the machine slept through."""
        done = threading.Event()
        # Working: at least a tenth of the patience spent searching.
        busy = self.watchdog_s / 10

        def watch() -> None:
            heard, spent = self._heard, self._cpu_mark()
            while True:
                began = time.monotonic()
                if self.watchdog_wait(done, self.watchdog_s):
                    return
                now = self._cpu_mark()
                slept = time.monotonic() - began > 2 * self.watchdog_s
                working = True  # unless judged idle below
                if spent is not None and now is not None and spent[0] == now[0]:
                    working = now[1] - spent[1] >= busy
                if self._heard == heard and not working and not slept:
                    self._end_process()
                heard, spent = self._heard, now

        watcher = threading.Thread(target=watch, name="sidecoach-engine-watchdog", daemon=True)
        watcher.start()
        try:
            yield
        finally:
            done.set()

    def _cpu_mark(self) -> tuple[object, float] | None:
        """The engine's process and its processor time so far: a window's baseline, judged
        only against the same process's."""
        engine = self._engine
        cpu = self._cpu()
        if cpu is None:
            return None
        return (id(engine) if self.cpu_seconds is not None else self._pid(), cpu)

    def _pid(self) -> int | None:
        engine = self._engine
        if engine is None:
            return None
        try:
            return engine.transport.get_pid()
        except Exception:  # noqa: BLE001 - gone
            return None

    def _cpu(self) -> float | None:
        """The engine process's processor time so far (`cpu_seconds`), or None."""
        if self.cpu_seconds is not None:
            return self.cpu_seconds()
        engine = self._engine
        if engine is None:
            return None
        try:
            import psutil

            times = psutil.Process(engine.transport.get_pid()).cpu_times()
        except Exception:  # noqa: BLE001 - gone, or unreadable: silence decides alone
            return None
        return times.user + times.system

    def _end_process(self) -> None:
        """The engine's process ended from outside, the way a crash ends it."""
        engine = self._engine
        if engine is None:
            return
        with contextlib.suppress(Exception):  # gone already
            kill = signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
            os.kill(engine.transport.get_pid(), kill)

    @contextlib.contextmanager
    def interruptible(self) -> Iterator[None]:
        """Searches that another thread may stop (`interrupt`): each raises `Interrupted`
        once stopped, and so does every search after it here -- a look ahead the game has
        overtaken is let go at once, not searched to its end first. Nested, the inner block
        keeps the outer's stop: one that came before it holds inside it too."""
        outer = self._interruptible
        if not outer:
            self._stopped.clear()
        self._interruptible = True
        try:
            yield
        finally:
            self._interruptible = outer

    def interrupt(self) -> None:
        """Stop the search running inside `interruptible`, and the ones after it there."""
        self._stopped.set()
        search = self._search
        if search is not None:
            with contextlib.suppress(Exception):  # it may have ended just now
                search.stop()

    def _analyse_with_restart(
        self,
        board: chess.Board,
        limit: chess.engine.Limit,
        multipv: int,
        on_depth: Callable[[list[chess.engine.InfoDict]], None] | None = None,
    ) -> list[chess.engine.InfoDict]:
        for attempt in (1, 2):
            self._start()
            assert self._engine is not None
            try:
                # Read as it goes, always: a search the time cap cuts is answered with the
                # last depth finished whole (`_streamed`).
                return self._streamed(board, limit, multipv, on_depth)
            except (chess.engine.EngineTerminatedError, BrokenPipeError) as exc:
                self._drop()
                if attempt == 2:
                    raise EngineUnavailable(f"engine died twice on {board.fen()}: {exc}") from exc
        raise AssertionError("unreachable")

    def _streamed(
        self,
        board: chess.Board,
        limit: chess.engine.Limit,
        multipv: int,
        on_depth: Callable[[list[chess.engine.InfoDict]], None] | None,
    ) -> list[chess.engine.InfoDict]:
        """`analyse`'s search, read as it goes: each depth handed to `on_depth` once its
        last line is in (Stockfish reports a depth's lines best first), and, inside
        `interruptible`, stopped by `interrupt`. A search cut mid-depth (the time cap) has
        its first lines a depth further than the rest, and the gap between them set two
        depths against each other: it is answered with the last depth finished whole."""
        stoppable = self._interruptible
        if stoppable and self._stopped.is_set():
            raise Interrupted
        assert self._engine is not None
        with self._engine.analysis(board, limit, multipv=multipv) as search:
            self._search = search if stoppable else None
            try:
                if stoppable and self._stopped.is_set():  # interrupted before it was in reach
                    search.stop()
                latest: dict[int, chess.engine.InfoDict] = {}
                done = 0
                whole: list[chess.engine.InfoDict] | None = None
                for info in search:
                    self._heard += 1  # talking: not hung (`_watched`)
                    rank, depth = info.get("multipv", 1), info.get("depth")
                    if depth is None or not info.get("pv"):
                        continue
                    latest[rank] = info
                    if rank == multipv and depth > done:
                        done = depth
                        whole = [latest[k] for k in sorted(latest)]
                        if on_depth is not None:
                            on_depth(whole)
                infos = list(search.multipv)
                if whole is not None and len({i.get("depth") for i in infos}) > 1:
                    infos = whole
            finally:
                self._search = None
        if stoppable and self._stopped.is_set():
            raise Interrupted
        return infos


def _line(info: chess.engine.InfoDict, board: chess.Board, pv_plies: int, model: str) -> Line:
    score = info.get("score")
    if score is None:
        return Line(moves=[], depth=0, expected_score=0.5)
    relative = score.relative
    pov_wdl = info.get("wdl")
    if pov_wdl is None:
        pov_wdl = score.wdl(model=WDL_MODEL, ply=board.ply())
    wdl = pov_wdl.pov(board.turn)
    pv = info.get("pv") or []
    numbers = (wdl.wins, wdl.draws, wdl.losses)
    return Line(
        moves=[move.uci() for move in pv[:pv_plies]],
        depth=int(info.get("depth", 0)),
        expected_score=expected_score(model, relative.score(), relative.mate(), numbers),
        score_cp=relative.score(),
        mate=relative.mate(),
        wdl=numbers,
    )


def san_line(
    board: chess.Board, moves: list[str], limit: int | None = None, *, numbered: bool = True
) -> str:
    """A line of UCI moves as a person reads it: `24. Nxe5 Bxe5 25. Qh5`; without
    `numbered`, the bare moves `Nxe5 Bxe5 Qh5` (the position says whose move it is).

    Anything the engine cannot play from this position stops the line; a line that has
    drifted out of step with the board is a bug worth seeing rather than hiding.
    """
    replay = board.copy(stack=False)
    parts: list[str] = []
    for i, uci in enumerate(moves[: limit if limit is not None else len(moves)]):
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in replay.legal_moves:
            break
        if numbered and replay.turn == chess.WHITE:
            parts.append(f"{replay.fullmove_number}.")
        elif numbered and i == 0:
            parts.append(f"{replay.fullmove_number}...")
        parts.append(replay.san(move))
        replay.push(move)
    return " ".join(parts)
