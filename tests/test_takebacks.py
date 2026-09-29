"""Takebacks end to end: the board rendered, the pipeline reading it, the engine glue
(`cli._Assessing`) and the coach behind it, all held against each other.

Tai's game on the Pixel, 2026-09-23: he tried one 17th move after another with takebacks,
and the program lagged further behind with each one -- the engine measured every withdrawn
move before the replacement, a verdict on a withdrawn move was taken for the one that
replaced it, and a board left unexplained for a while was sent downstream as a resync, so
the coach and the engine restarted their numbering while the reconciler kept its own and
the next takeback ("10 plies") meant nothing to them. Here the same shapes: a stretch the
reconciler cannot explain and a takeback of eleven plies after it, and takebacks pressed
again and again, two plies a press, then a different move. (A takeback of two plies, the
Pixel game's, is the reconciler's in test_reconcile; through the engine it is the same as
twelve.)
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import chess
import numpy as np
import pytest

from sidecoach.analysis.annotator import Analyser, GameAnnotator
from sidecoach.cli import _Assessing
from sidecoach.config import Settings
from sidecoach.events import (
    Analysis,
    GameStarted,
    MoveAssessment,
    MovePlayed,
    Resync,
    Takeback,
    Unexplained,
)
from sidecoach.state.pipeline import GamePipeline
from tests.coaching import scripted_coach
from tests.fakes import FakeEngine
from tests.synth import synthetic_board

pytestmark = pytest.mark.vision

WIDTH, HEIGHT = 1280, 800


class _SlowEngine(FakeEngine):
    """A fake engine that takes a moment per search, as a real one does, and remembers the
    positions it began searching while `watching`."""

    def __init__(self, seconds: float) -> None:
        super().__init__({})
        self.seconds = seconds
        # `armed` by a test, `watching` from the moment the analyser is told of a takeback
        # (`Chain`): what it searched before then was the game's to search.
        self.armed = False
        self.watching = False
        self.asked: list[str] = []
        self._lock = threading.Lock()

    def analyse(self, board: chess.Board, **kwargs: object) -> list:
        from sidecoach.analysis.engine import Interrupted

        # Stopped as a real engine is (`Engine.interrupt`): none begins once stopped, and one
        # under way ends within a moment. Noted as the search starts.
        if self._interruptible and self._stopped.is_set():
            raise Interrupted
        with self._lock:
            if self.watching:
                self.asked.append(board.board_fen())
        end = time.monotonic() + self.seconds
        while time.monotonic() < end:
            if self._interruptible and self._stopped.is_set():
                raise Interrupted
            time.sleep(0.005)
        return super().analyse(board, **kwargs)


class Chain:
    """Frames in; the pipeline's events to the engine glue and the coach; the engine's
    words drained to the coach on every frame, as the panel's loop does."""

    def __init__(self, tmp: Path, engine_seconds: float = 0.03) -> None:
        self.pipe = GamePipeline()
        self.board = chess.Board()
        self.ts = 0.0
        self.engine = _SlowEngine(engine_seconds)
        self.assessing = _Assessing(Settings(data_dir=tmp), enabled=False)
        self.assessing.analyser = Analyser(
            GameAnnotator(engine=self.engine, depth=6, with_threats=False)
        )
        withdraw = self.assessing.analyser.withdraw

        def told(game_id: str, ply: int) -> None:
            withdraw(game_id, ply)
            if self.engine.armed:
                self.engine.watching = True

        self.assessing.analyser.withdraw = told  # type: ignore[method-assign]
        self.assessing.analyser.start()
        self.coach, _ = scripted_coach(["never used"] * 50)
        self.events: list[object] = []
        self.drained: list[object] = []
        self.told_at = 0  # where `drained` stood when the last takeback was told (`feed`)

    def frame(
        self,
        highlight: chess.Move | None = None,
        board: chess.Board | None = None,
        animating: tuple[chess.Square, chess.Square, float] | None = None,
    ) -> np.ndarray:
        return synthetic_board(
            board=board or self.board,
            x0=300,
            y0=120,
            cell=64,
            highlight=highlight,
            animating=animating,
            width=WIDTH,
            height=HEIGHT,
        )

    def feed(self, frame: np.ndarray, times: int = 1) -> None:
        for _ in range(times):
            self.ts += 0.1
            events = self.pipe.process(frame, self.ts, frame_id=int(self.ts * 10))
            for event in events:
                self.events.append(event)
                if isinstance(event, Takeback):
                    # What the engine said before this was said of the game as it stood.
                    self.told_at = len(self.drained)
                if isinstance(event, GameStarted | MovePlayed | Takeback | Resync):
                    self.assessing.submit(event)
                self.coach.on_event(event, self.ts)
            for said in self.assessing.drain():
                self.drained.append(said)
                self.coach.on_event(said, self.ts)

    def play(self, *sans: str) -> None:
        for san in sans:
            move = self.board.push_san(san)
            self.feed(self.frame(highlight=move), 3)

    def take_back(self, plies: int) -> None:
        """chess.com's takeback: the board goes back, and quiet frames show it there."""
        for _ in range(plies):
            self.board.pop()
        self.feed(self.frame(), 4)

    def wait_for_engine(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        assert self.assessing.analyser is not None
        while self.assessing.analyser.pending and time.monotonic() < deadline:
            self.feed(self.frame(), 1)
            time.sleep(0.01)
        self.feed(self.frame(), 2)

    def close(self) -> None:
        self.assessing.close()

    @property
    def rec(self):  # noqa: ANN201
        assert self.pipe.reconciler is not None
        return self.pipe.reconciler


def _line(board: chess.Board) -> dict[int, str]:
    return {i: m.uci() for i, m in enumerate(board.move_stack, start=1)}


def _fens(board: chess.Board) -> dict[int, str]:
    walked = chess.Board()
    out: dict[int, str] = {}
    for i, move in enumerate(board.move_stack, start=1):
        walked.push(move)
        out[i] = walked.board_fen()
    return out


def _agree(c: Chain, since: int) -> None:
    """The game, the reconciler, the engine glue and the coach count the same plies, and
    nothing drained since `since` and the takeback was told is about a move or a position
    the game has left: what came before it was said of the game as it then stood -- the
    engine at work on the moves the takeback was to withdraw, as it should be."""
    final = _line(c.board)
    assert _line(c.rec.board) == final, "the reconciler's game"
    assert c.assessing._line == final, "the engine glue's game"
    assert {p: m.uci for p, m in c.coach.moves.items()} == final, "the coach's game"
    fens = _fens(c.board)
    for said in c.drained[max(since, c.told_at) :]:
        if isinstance(said, MoveAssessment):
            assert final.get(said.ply) == said.uci, f"a verdict on a withdrawn move: {said}"
        elif isinstance(said, Analysis) and said.ply > 0:
            assert fens.get(said.ply) == said.fen.split()[0], (
                f"a withdrawn position: ply {said.ply}"
            )


def test_an_unexplained_stretch_keeps_the_numbering_for_a_long_takeback(tmp_path: Path) -> None:
    """The board unexplained for a long while, then back eleven plies with a different move:
    the reconciler, the engine glue and the coach count the same plies throughout."""
    c = Chain(tmp_path)
    try:
        c.feed(c.frame(), 3)
        c.play("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4", "cxd4", "Bb4+")
        c.wait_for_engine()
        c.rec.stale_after = 2
        c.rec.resync_after = 6
        # A board that keeps changing and never settles -- the takeback button pressed again
        # and again, pieces sliding back -- with cells the game cannot account for: nothing
        # explains it, and no still, complete position is there to adopt.
        frames = []
        for knight in (chess.H3, chess.H4):
            odd = c.board.copy()
            for square in (chess.A2, chess.B2, chess.G2):
                odd.remove_piece_at(square)
            odd.set_piece_at(knight, chess.Piece(chess.KNIGHT, chess.BLACK))
            frames.append(c.frame(board=odd))
        for i in range(16):
            c.feed(frames[i % 2], 1)
        assert any(isinstance(e, Unexplained) for e in c.events), "the stretch was noted"
        assert not any(isinstance(e, Resync) for e in c.events), "and not sent as a resync"
        assert len(c.coach.moves) == 12, "the coach kept the game"

        # Back eleven plies -- to the position after 1.e4 -- and a different move from there.
        c.take_back(11)
        since = len(c.drained)
        c.play("Nf6", "e5")
        c.wait_for_engine()
        assert [t.plies for t in c.events if isinstance(t, Takeback)] == [11]
        _agree(c, since)
    finally:
        c.close()


GAME = [
    "e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4",
    "cxd4", "Bb4+", "Bd2", "Bxd2+", "Nbxd2", "d5", "exd5", "Nxd5", "Qb3", "Nce7",
]  # fmt: skip


@pytest.mark.parametrize(
    ("per_position", "animate"),
    [(2, True), (1, False), (0, False)],
    ids=["stepwise-sliding", "a-frame-each", "one-jump"],
)
def test_ten_plies_taken_back_quickly(tmp_path: Path, per_position: int, animate: bool) -> None:
    """Takeback pressed again and again, fast (Tai, 2026-09-23: "try to test farther
    takebacks (that are done quickly) too"): the board steps back two plies at a time, each
    earlier position up for a frame or two -- the pieces caught sliding back in between --
    or lands ten plies back at once; then a different move. One takeback of ten plies, no
    resync, the plies counted alike everywhere, and the replacement's verdict prompt with
    an engine a phone's speed behind: the rest of the search under way when the takeback
    came, then its own, where the withdrawn plies' own searches came first (the takeback of
    two plies on the Pixel: 0.66 s, where they made it 1.05 s); and nothing about the
    withdrawn moves asked or said once the takeback is known
    (test_a_takeback_s_withdrawn_moves_are_let_go_and_never_said)."""
    c = Chain(tmp_path)
    try:
        c.feed(c.frame(), 3)
        c.play(*GAME)
        c.wait_for_engine()
        search = 0.4
        c.engine.seconds = search
        c.play("Qb5+", "c6")  # the engine is on these when the takebacks start
        withdrawn = c.board.board_fen()
        for _step in range(6):  # 22 plies back to 20, then to 10: 2 plies a press
            for _ply in range(2):
                move = c.board.peek()
                if animate:
                    c.feed(c.frame(animating=(move.to_square, move.from_square, 0.5)), 1)
                c.board.pop()
            if per_position:
                c.feed(c.frame(), per_position)
        c.feed(c.frame(), 3)  # the player looks at the position before playing on
        since = len(c.drained)
        c.engine.armed = True  # recorded from when the takeback is told
        move = c.board.push_san("e5")  # 6.e5 in place of 6.cxd4
        began = time.monotonic()
        c.feed(c.frame(highlight=move), 3)
        while time.monotonic() - began < 10 and not any(
            isinstance(e, MoveAssessment) and e.ply == 11 and e.uci == "e4e5" for e in c.drained
        ):
            c.feed(c.frame(highlight=move), 1)
            time.sleep(0.01)
        waited = time.monotonic() - began
        c.play("d5")
        c.wait_for_engine()
        c.engine.armed = c.engine.watching = False

        took = [e.plies for e in c.events if isinstance(e, Takeback)]
        assert took == [12], took  # from ply 22 back to ply 10
        assert not any(isinstance(e, Resync) for e in c.events), "no resync"
        assert waited < 2.2 * search, f"the replacement's verdict took {waited:.2f} s"
        assert withdrawn not in c.engine.asked
        _agree(c, since)
    finally:
        c.close()
