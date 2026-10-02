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
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import chess.engine

from sidecoach.analysis.score import expected_score, game_over
from sidecoach.events import Line

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


def _lower_priority() -> None:
    """In the engine's process before it starts (POSIX): below the player's own work."""
    if sys.platform != "win32":
        os.nice(5)


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
    # No single position may hold up a game. Some positions are pathologically slow to
    # reach a given depth: a long forced mate with several lines to separate can take
    # minutes, and one of those in a post-game review stalls the whole run. Whichever
    # limit comes first wins, and the depth actually reached is reported on every line.
    max_seconds: float | None = 5.0
    # Work the player is not waiting on (the background review of imported games) yields
    # the processor to everything else: the process runs below normal priority.
    low_priority: bool = False
    _engine: chess.engine.SimpleEngine | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _wdl_from_engine: bool = field(default=False, init=False, repr=False)
    # Inside `interruptible`: the search running, which `interrupt` stops, and whether it has.
    _interruptible: bool = field(default=False, init=False, repr=False)
    _stopped: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _search: chess.engine.SimpleAnalysisResult | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> Engine:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def name(self) -> str:
        """What the engine calls itself, for the doctor and the logs."""
        self.open()
        assert self._engine is not None
        return str(self._engine.id.get("name", self.path.name))

    def open(self) -> None:
        if self._engine is not None:
            return
        popen: dict[str, Any] = {}
        if self.low_priority and sys.platform == "win32":
            popen["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
        elif self.low_priority:
            popen["preexec_fn"] = _lower_priority
        try:
            engine = chess.engine.SimpleEngine.popen_uci(str(self.path), **popen)
        except Exception as exc:  # noqa: BLE001
            raise EngineUnavailable(f"cannot start {self.path}: {exc}") from exc
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
        self._engine = engine

    def reconfigure(self, *, threads: int | None = None, hash_mb: int | None = None) -> None:
        """Threads and hash on the running engine (the settings saved mid-session)."""
        if threads is not None:
            self.threads = threads
        if hash_mb is not None:
            self.hash_mb = hash_mb
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
        engine, self._engine = self._engine, None
        if engine is not None:
            with contextlib.suppress(Exception):  # a dead engine needs no goodbye
                engine.quit()

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
        capped = time_s
        if depth is not None and self.max_seconds is not None:
            capped = self.max_seconds if time_s is None else min(time_s, self.max_seconds)
        limit = chess.engine.Limit(depth=depth, time=capped)
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
                infos = self._analyse_with_restart(end, limit, 1)
            more = (infos[0].get("pv") or []) if infos else []
            if not more:
                break
            for move in more[: plies - len(moves)]:
                moves.append(move.uci())
                end.push(move)
        return line.model_copy(update={"moves": moves})

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
            self.open()
            assert self._engine is not None
            try:
                if self._interruptible or on_depth is not None:
                    return self._streamed(board, limit, multipv, on_depth)
                return list(self._engine.analyse(board, limit, multipv=multipv))
            except (chess.engine.EngineTerminatedError, BrokenPipeError) as exc:
                self.close()
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
        `interruptible`, stopped by `interrupt`."""
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
                for info in search:
                    rank, depth = info.get("multipv", 1), info.get("depth")
                    if depth is None or not info.get("pv"):
                        continue
                    latest[rank] = info
                    if on_depth is not None and rank == multipv and depth > done:
                        done = depth
                        on_depth([latest[k] for k in sorted(latest)])
                infos = list(search.multipv)
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
