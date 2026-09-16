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
import threading
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.engine

from sidecoach.analysis.score import expected_score
from sidecoach.events import Line

# A principal variation can run thirty plies; nobody reads that far and events stay small.
DEFAULT_PV_PLIES = 12
# python-chess's model for turning centipawns into win-draw-loss when an engine will not.
WDL_MODEL = "sf"
DEFAULT_SCORE_MODEL = "human"


class EngineUnavailable(RuntimeError):
    """The engine binary is missing, or died and could not be restarted."""


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
    _engine: chess.engine.SimpleEngine | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _wdl_from_engine: bool = field(default=False, init=False, repr=False)

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
        try:
            engine = chess.engine.SimpleEngine.popen_uci(str(self.path))
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
    ) -> list[Line]:
        """The best lines for `board`, best first, from the side to move's point of view.

        A finished position has no lines: there is nothing to play. Callers read its value
        from the result instead, with `sidecoach.analysis.score.terminal_expected_score`.
        """
        if board.is_game_over(claim_draw=True):
            return []
        if depth is None and time_s is None:
            raise ValueError("analyse needs a depth or a time limit")
        capped = time_s
        if depth is not None and self.max_seconds is not None:
            capped = self.max_seconds if time_s is None else min(time_s, self.max_seconds)
        limit = chess.engine.Limit(depth=depth, time=capped)
        wanted = multipv if multipv is not None else self.multipv
        wanted = max(1, min(wanted, board.legal_moves.count()))
        with self._lock:
            infos = self._analyse_with_restart(board, limit, wanted)
        plies = pv_plies if pv_plies is not None else self.pv_plies
        lines = [_line(info, board, plies, self.score_model) for info in infos]
        return [line for line in lines if line.moves]

    def _analyse_with_restart(
        self, board: chess.Board, limit: chess.engine.Limit, multipv: int
    ) -> list[chess.engine.InfoDict]:
        for attempt in (1, 2):
            self.open()
            assert self._engine is not None
            try:
                return list(self._engine.analyse(board, limit, multipv=multipv))
            except (chess.engine.EngineTerminatedError, BrokenPipeError) as exc:
                self.close()
                if attempt == 2:
                    raise EngineUnavailable(f"engine died twice on {board.fen()}: {exc}") from exc
        raise AssertionError("unreachable")


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


def san_line(board: chess.Board, moves: list[str], limit: int | None = None) -> str:
    """A line of UCI moves as a person reads it: `24. Nxe5 Bxe5 25. Qh5`.

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
        if replay.turn == chess.WHITE:
            parts.append(f"{replay.fullmove_number}.")
        elif i == 0:
            parts.append(f"{replay.fullmove_number}...")
        parts.append(replay.san(move))
        replay.push(move)
    return " ".join(parts)
