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
from pathlib import Path

import chess
import numpy as np
import pytest

from sidecoach.analysis.analyser import Analyser
from sidecoach.analysis.annotator import GameAnnotator
from sidecoach.cli import _Assessing
from sidecoach.config import Settings
from sidecoach.events import (
    Analysis,
    GameStarted,
    MoveAssessment,
    MovePlayed,
    Resync,
    Side,
    Takeback,
    Unexplained,
)
from sidecoach.state.pipeline import GamePipeline
from tests.coaching import drain, scripted_coach
from tests.fakes import HANG_S, AnalyserWatch, FakeEngine
from tests.synth import synthetic_board

pytestmark = pytest.mark.vision

WIDTH, HEIGHT = 1280, 800


class _GatedEngine(FakeEngine):
    """A fake engine whose searches run as the test lets them, never on a clock: answered at
    once while the gate is open; after `hold`, each search begun waits at the gate until
    `release` -- or until it is interrupted, and then it stops, as a real one does
    (`Engine.interrupt`): none begins once stopped, and one under way ends.

    Every search begun is noted in `log`, in order with what the analyser said (`Chain`); and
    the positions begun while `watching` in `asked`."""

    def __init__(self, log: list[tuple[str, object]]) -> None:
        super().__init__({})
        self.log = log
        # `armed` by a test, `watching` from the moment the analyser is told of a takeback
        # (`Chain`): what it searched before then was the game's to search.
        self.armed = False
        self.watching = False
        self.asked: list[str] = []
        self.ends: list[tuple[str, str]] = []  # each held search's position, and how it ended
        self._gate = threading.Condition()
        self._held = False
        self.at_gate = 0  # searches waiting at the gate now

    def hold(self) -> None:
        with self._gate:
            self._held = True

    def release(self) -> None:
        with self._gate:
            self._held = False
            self._gate.notify_all()

    def interrupt(self) -> None:
        super().interrupt()
        with self._gate:
            self._gate.notify_all()

    def wait_at_gate(self) -> None:
        """Until a search is held at the gate: the engine is busy on it."""
        with self._gate:
            assert self._gate.wait_for(lambda: self.at_gate > 0, HANG_S), "no search came"

    def wait_ended(self, count: int) -> list[tuple[str, str]]:
        """Until `count` held searches have ended (let go or interrupted); how they ended."""
        with self._gate:
            assert self._gate.wait_for(lambda: len(self.ends) >= count, HANG_S), self.ends
            return list(self.ends)

    def analyse(self, board: chess.Board, **kwargs: object) -> list:
        from sidecoach.analysis.engine import Interrupted

        def stopped() -> bool:
            return self._interruptible and self._stopped.is_set()

        if stopped():
            raise Interrupted
        with self._gate:
            self.log.append(("search", board.board_fen()))
            if self.watching:
                self.asked.append(board.board_fen())
            if self._held:
                self.at_gate += 1
                self._gate.notify_all()
                try:
                    let_go = self._gate.wait_for(lambda: not self._held or stopped(), HANG_S)
                finally:
                    self.at_gate -= 1
                assert let_go, "a held search never let go"
                self.ends.append((board.board_fen(), "interrupted" if stopped() else "answered"))
                self._gate.notify_all()
            if stopped():
                raise Interrupted
        return super().analyse(board, **kwargs)


class _Said:
    """The analyser's results queue, noting each word in the engine's log as it is put."""

    def __init__(self, inner: object, log: list[tuple[str, object]]) -> None:
        self._inner, self._log = inner, log

    def put(self, item: object) -> None:
        self._log.append(("said", item))
        self._inner.put(item)  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


class Chain:
    """Frames in; the pipeline's events to the engine glue and the coach; the engine's
    words drained to the coach on every frame, as the panel's loop does. The engine keeps up
    with the game -- each move's work settled before the next frame -- unless the test holds
    it (`_GatedEngine.hold`): then it is behind exactly as far as the test says."""

    def __init__(self, tmp: Path, replies: tuple[str, ...] = ()) -> None:
        self.pipe = GamePipeline()
        self.board = chess.Board()
        self.ts = 0.0
        # In order: every search begun, every word the analyser said, and each takeback the
        # analyser was told of.
        self.log: list[tuple[str, object]] = []
        self.engine = _GatedEngine(self.log)
        self.assessing = _Assessing(Settings(data_dir=tmp), enabled=False)
        analyser = Analyser(GameAnnotator(engine=self.engine, depth=6, with_threats=False))
        self.assessing.analyser = analyser
        analyser._results = _Said(analyser._results, self.log)  # type: ignore[assignment]
        self.watch = AnalyserWatch(analyser)
        withdraw = analyser.withdraw

        def told(game_id: str, ply: int) -> None:
            withdraw(game_id, ply)
            # After the analyser has let go: whatever begins after this is not of a withdrawn
            # move (a search under way had begun before).
            self.log.append(("takeback", ply))
            if self.engine.armed:
                self.engine.watching = True

        analyser.withdraw = told  # type: ignore[method-assign]
        analyser.start()
        self.coach, self.provider = scripted_coach(list(replies) or ["never used"] * 50)
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
            if not self.engine._held:
                self.watch.idle()  # the engine keeps up: this move's work done

    def take_back(self, plies: int) -> None:
        """chess.com's takeback: the board goes back, and quiet frames show it there."""
        for _ in range(plies):
            self.board.pop()
        self.feed(self.frame(), 4)

    def wait_for_engine(self) -> None:
        """Until everything owed is answered (a signal, not a clock), then the frames that
        hand what the engine said to the coach."""
        self.watch.idle()
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
    an engine behind: the search under way when the takeback came let go, then its own,
    where the withdrawn plies' own searches came first (the takeback of two plies on the
    Pixel: 0.66 s, where they made it 1.05 s); and nothing about the withdrawn moves asked
    or said once the takeback is known
    (test_a_takeback_s_withdrawn_moves_are_let_go_and_never_said).

    "Prompt" is counted, not timed: the engine is held, so it is on the withdrawn moves when
    the takebacks start, and every search it begins is logged in order with what the
    analyser says. Between the takeback and the replacement's verdict it searches only the
    replacement's own positions, at most the two a verdict needs."""
    c = Chain(tmp_path)
    try:
        c.feed(c.frame(), 3)
        c.play(*GAME)
        c.wait_for_engine()
        c.engine.hold()
        c.play("Qb5+", "c6")  # the engine is on these when the takebacks start
        c.engine.wait_at_gate()
        # Owed: the two positions and the two verdicts.
        assert c.assessing.analyser is not None and c.assessing.analyser.pending == 4
        withdrawn = c.board.board_fen()
        old_line = _fens(c.board)
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
        c.feed(c.frame(highlight=move), 3)
        told = [i for i, (kind, _) in enumerate(c.log) if kind == "takeback"]
        assert len(told) == 1, "the takeback told once"
        # The search under way when the takeback came was let go, not run to its end: it was
        # of a withdrawn position.
        first = c.engine.wait_ended(1)[0]
        assert first[1] == "interrupted" and first[0] in old_line.values(), first
        c.engine.release()
        c.wait_for_engine()
        verdicts = [  # at ply 11 since the takeback (6.cxd4's own came long before)
            i
            for i, (kind, said) in enumerate(c.log)
            if i > told[0]
            and kind == "said"
            and isinstance(said, MoveAssessment)
            and said.ply == 11
        ]
        assert verdicts, "the replacement was never judged"
        replacement = c.log[verdicts[0]][1]
        assert isinstance(replacement, MoveAssessment) and replacement.uci == "e4e5"
        assert any(
            isinstance(e, MoveAssessment) and e.ply == 11 and e.uci == "e4e5" for e in c.drained
        ), "the replacement's verdict reached the coach"
        # Between the takeback and the replacement's verdict: the replacement's own searches
        # alone -- the position it was played from and the one it made, at most -- and nothing
        # of the withdrawn plies said.
        between = c.log[told[0] + 1 : verdicts[0]]
        searched = [fen for kind, fen in between if kind == "search"]
        own = {_fens(c.board)[10], _fens(c.board)[11]}
        assert 1 <= len(searched) <= 2 and set(searched) <= own, searched
        assert not [s for kind, s in between if kind == "said"], "a word before the verdict"
        c.play("d5")
        c.wait_for_engine()
        c.engine.armed = c.engine.watching = False

        took = [e.plies for e in c.events if isinstance(e, Takeback)]
        assert took == [12], took  # from ply 22 back to ply 10
        assert not any(isinstance(e, Resync) for e in c.events), "no resync"
        assert c.engine.asked, "the takeback was told once the test watched for it"
        assert withdrawn not in c.engine.asked
        # Nothing withdrawn searched from the takeback on, to the end of the game.
        gone = {fen for ply, fen in old_line.items() if ply > 10}
        after = {fen for kind, fen in c.log[told[0] + 1 :] if kind == "search"}
        assert not gone & after, "a withdrawn position searched after the takeback"
        _agree(c, since)
    finally:
        c.close()


HINT = "Think about which piece you want in the centre first."


def test_a_hint_asked_before_the_first_move_is_answered(tmp_path: Path) -> None:
    """The start position was never sent to the engine -- only a position begun from the
    middle was -- so a hint asked before the first move was "One moment -- still looking"
    and nothing after it, and no stuck hint came there (the state review, 2026-10-01). The
    board read, the engine glue and the coach on a model: the model is asked, and answers."""
    c = Chain(tmp_path, replies=(HINT,) * 3)
    try:
        c.feed(c.frame(), 3)
        assert any(isinstance(e, GameStarted) for e in c.events)
        c.coach.ask("hint", c.ts)
        c.wait_for_engine()
        assert any(isinstance(e, Analysis) and e.ply == 0 for e in c.drained), "never measured"
        said = [m.text for m in drain(c.coach)]
        assert c.provider.calls, f"the model was never asked: {said}"
        assert HINT in said, said
    finally:
        c.close()


def test_a_takeback_to_where_the_game_began_measures_it_again(tmp_path: Path) -> None:
    """Every move of a position taken back: the position the game began from is the one the
    player faces again, and the engine is asked for it again (at once, it is measured), so
    the look ahead goes on from there and not from the moves withdrawn."""
    assessing = _Assessing(Settings(data_dir=tmp_path), enabled=False)
    engine = FakeEngine({})
    assessing.analyser = Analyser(GameAnnotator(engine=engine, depth=6, with_threats=False))
    watch = AnalyserWatch(assessing.analyser)
    assessing.analyser.start()
    try:
        start = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
        assessing.submit(
            GameStarted(game_id="g", user_side=Side.WHITE, start_fen=start.fen(), kind="position")
        )
        board = start.copy()
        for ply, san in enumerate(["Bc4", "Bc5"], start=1):
            before = board.fen()
            move = board.push_san(san)
            assessing.submit(
                MovePlayed(
                    game_id="g",
                    ply=ply,
                    san=san,
                    uci=move.uci(),
                    fen_before=before,
                    fen_after=board.fen(),
                    by_user=ply == 1,
                )
            )
        watch.idle()  # both moves answered (signalled by the worker, not polled)
        assessing.drain()
        assessing.submit(Takeback(game_id="g", plies=2, fen=start.fen()))
        watch.idle()
        again = assessing.drain()
        assert [e.fen for e in again if isinstance(e, Analysis) and e.ply == 0] == [start.fen()]
    finally:
        assessing.close()
