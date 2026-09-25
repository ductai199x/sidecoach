"""Moves that slide (Tai's chess.com puzzles, 2026-09-24/25: "I played Rxc1+ but it
misunderstood as Rc7"). Every site slides a moved piece in a straight line from its square to
the next; chess.com's phone app always does, speeding up as it goes, with a faint trail. A
frame that catches the piece near the centre of a square on its way reads as a piece standing
there, and the next frame -- the piece further on, the site's reply already sliding -- used to
explain as that piece moving on: Rc7, an invented Rf1, then Rxc1 twice. `reconcile.in_flight`
reads a piece that stops partway along its line and goes on the same way as one move.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

from sidecoach.state.pipeline import GamePipeline
from sidecoach.state.reconcile import in_flight
from sidecoach.vision.pieces import Side
from tests.synth import synthetic_board


def test_a_piece_that_stops_partway_and_goes_on_is_in_flight() -> None:
    m = chess.Move.from_uci
    assert in_flight([m("c8c7"), m("h1f1"), m("c7c1")])  # the rook on its way to c1
    assert in_flight([m("d8d7"), m("a5c3"), m("c3e1")])  # the reply's queen, a5 to e1
    assert not in_flight([m("c8c7"), m("h1f1"), m("c7c8")])  # back where it came from
    assert not in_flight([m("g1f3"), m("e7e5"), m("f3g1")])
    assert not in_flight([m("c8c7"), m("h1f1"), m("c7a7")])  # on, but another way
    assert not in_flight([m("e2e4"), m("e7e5"), m("g1f3")])


def _frame(board: chess.Board, bottom: chess.Color, **kw: object):  # noqa: ANN202
    return synthetic_board(
        board=board,
        x0=300,
        y0=120,
        cell=64,
        width=1280,
        height=800,
        bottom=Side.WHITE if bottom == chess.WHITE else Side.BLACK,
        coordinates=True,
        **kw,  # type: ignore[arg-type]
    )


def _played(
    start: str, setup: str, moves: list[str], slide: float, power: float, phase: float
) -> list[str]:
    """The puzzle played through the pipeline at the phone's ~9.6 frames a second, every move
    sliding for `slide` seconds with progress `u ** power` and a faint trail, the site
    replying the moment the solver's piece lands."""
    board = chess.Board(start)
    solver = not board.turn
    events = [(0.0, chess.Move.from_uci(setup))]
    t = 3.0
    for i, uci in enumerate(moves):
        events.append((t, chess.Move.from_uci(uci)))
        t += slide if i % 2 == 0 else 2.0
    end = t + 1.5

    def scene(time: float):  # noqa: ANN202
        b = board.copy()
        last = None
        for t0, move in events:
            if time < t0:
                break
            u = (time - t0) / slide
            if u < 1.0:
                p = u**power
                sq = (move.from_square, move.to_square)
                img = _frame(b, solver, highlight=move, animating=(*sq, p))
                if p > 0.05:
                    g1 = _frame(b, solver, highlight=move, animating=(*sq, max(0.0, p - 0.08)))
                    g2 = _frame(b, solver, highlight=move, animating=(*sq, max(0.0, p - 0.16)))
                    img = (
                        0.70 * img.astype(np.float32)
                        + 0.20 * g1.astype(np.float32)
                        + 0.10 * g2.astype(np.float32)
                    ).astype(np.uint8)
                return img
            b.push(move)
            last = move
        return _frame(b, solver, highlight=last)

    pipe = GamePipeline()
    ts = 0.0
    period = 0.104
    for _ in range(3):
        ts += period
        pipe.process(_frame(chess.Board(), chess.WHITE), ts, frame_id=int(ts * 1000))
    k = 0
    while (time := phase + k * period) <= end:
        ts += period
        pipe.process(scene(time), ts, frame_id=int(ts * 1000))
        k += 1
    assert pipe.reconciler is not None
    return [m.uci() for m in pipe.reconciler.board.move_stack]


@pytest.mark.vision
@pytest.mark.parametrize(
    ("start", "setup", "moves", "slide", "power", "phase"),
    [
        # Tai's phone puzzle: Rc1+ Rxc1 Rxc1+ Rxc1 Qxd5; Rxc1+ was recorded as Rc7.
        (
            "2r3k1/pQ3p2/6p1/q6p/2r5/P4P2/1P4PP/1K1R3R w - - 0 1",
            "b7d5",
            ["c4c1", "d1c1", "c8c1", "h1c1", "a5d5"],
            0.4,
            2.0,
            0.02,
        ),
        # The desktop dump's: Qd1+ Rxd1 Rxd1+ Qe1; Rxd1 was recorded as Rd3 and Rxd1.
        (
            "2kr2nr/2p5/2p1p1pp/3QPp2/3R4/P3P3/1Pq2PPP/R1B3K1 w - - 0 1",
            "d5a5",
            ["c2d1", "d4d1", "d8d1", "a5e1"],
            0.3,
            2.0,
            0.04,
        ),
    ],
)
def test_a_move_seen_sliding_across_squares_is_one_move(
    start: str, setup: str, moves: list[str], slide: float, power: float, phase: float
) -> None:
    played = _played(start, setup, moves, slide, power, phase)
    assert played[-len(moves) :] == moves, played
