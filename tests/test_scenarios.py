"""What happens to the game window during a real session, end to end through the pipeline.

Each scenario renders the same synthetic game and disturbs the window the way a player
would: moving it, resizing it, covering it, minimising it, scrolling it off screen. The
pipeline must never crash, and the game must come out right once the board is back.
"""

import chess
import numpy as np
import pytest

from sidecoach.events import GameStarted, MovePlayed, Resync
from sidecoach.state.pipeline import GamePipeline
from tests.synth import busy_background, synthetic_board

pytestmark = pytest.mark.vision  # the synthetic vision pipeline: run with -m vision

FRAME = dict(width=1280, height=800)


def _played(events) -> list[str]:
    return [e.san for e in events if isinstance(e, MovePlayed)]


class Session:
    """Feed frames to a pipeline while keeping the game's true position."""

    def __init__(self, x0: int = 300, y0: int = 120, cell: int = 64) -> None:
        self.pipe = GamePipeline()
        self.board = chess.Board()
        self.ts = 0.0
        self.x0, self.y0, self.cell = x0, y0, cell
        self.played: list[str] = []
        self.events = []

    def frame(self, highlight=None, **kw) -> np.ndarray:
        return synthetic_board(
            board=self.board,
            x0=kw.pop("x0", self.x0),
            y0=kw.pop("y0", self.y0),
            cell=kw.pop("cell", self.cell),
            highlight=highlight,
            **FRAME,
            **kw,
        )

    def feed(self, frame: np.ndarray, times: int = 1) -> list[str]:
        out: list[str] = []
        for _ in range(times):
            self.ts += 0.1
            events = self.pipe.process(frame, self.ts, frame_id=int(self.ts * 10))
            self.events += events
            out += _played(events)
        self.played += out
        return out

    def start(self) -> None:
        self.feed(self.frame(), 2)
        assert any(isinstance(e, GameStarted) for e in self.events)

    def play(self, *sans: str, frames: int = 2) -> list[str]:
        out: list[str] = []
        for san in sans:
            move = self.board.push_san(san)
            out += self.feed(self.frame(highlight=move), frames)
        return out

    def settle(self, frames: int = 3) -> list[str]:
        return self.feed(self.frame(), frames)


def test_window_moved_mid_game() -> None:
    s = Session()
    s.start()
    assert s.play("e4", "e5") == ["e4", "e5"]
    s.x0, s.y0 = 520, 210  # the window was dragged elsewhere
    s.settle(6)  # the tracker loses the old lock and re-locks; no move is invented
    assert s.played == ["e4", "e5"]
    assert s.play("Nf3", "Nc6") == ["Nf3", "Nc6"]
    assert s.pipe.reconciler is not None and s.pipe.reconciler.board.fen() == s.board.fen()


def test_window_resized_mid_game() -> None:
    s = Session()
    s.start()
    assert s.play("d4", "d5") == ["d4", "d5"]
    s.cell = 48  # the window was resized: the board is three quarters as big
    s.settle(6)
    assert s.played == ["d4", "d5"]
    assert s.play("c4", "e6") == ["c4", "e6"]


def test_board_fully_covered_while_moves_happen() -> None:
    """Another window over the whole board for a while; two plies happen underneath."""
    s = Session()
    s.start()
    assert s.play("e4", "e5") == ["e4", "e5"]
    covered = s.frame().copy()
    covered[60:720, 240:880] = (60, 60, 60)
    s.feed(covered, 25)  # 2.5 seconds under another window
    assert s.played == ["e4", "e5"]
    s.board.push_san("Nf3")
    s.board.push_san("Nc6")
    out = s.settle(8)  # uncovered: both plies are caught up
    assert out == ["Nf3", "Nc6"], s.events[-8:]


def test_board_minimised_then_restored() -> None:
    s = Session()
    s.start()
    assert s.play("e4", "e5", "Nf3") == ["e4", "e5", "Nf3"]
    desktop = busy_background(FRAME["width"], FRAME["height"], seed=7)
    s.feed(desktop, 40)  # minimised for four seconds; the tracker searches and backs off
    assert s.pipe.board is None
    assert s.played == ["e4", "e5", "Nf3"]
    s.board.push_san("Nc6")
    out = s.settle(30)  # restored; the tracker's backoff must not hide the board for long
    assert out == ["Nc6"]
    assert s.play("Bb5", "a6") == ["Bb5", "a6"]


def test_board_partly_covered_by_another_window() -> None:
    """A window over the right third of the board: moves on the left still commit."""
    s = Session()
    s.start()
    assert s.play("e4", "e5") == ["e4", "e5"]

    def cover(frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        out[80:760, 300 + 5 * 64 : 900] = (240, 240, 240)  # columns f to h and beyond
        return out

    move = s.board.push_san("d4")  # on the visible side
    assert s.feed(cover(s.frame(highlight=move)), 4) == ["d4"]
    move = s.board.push_san("Nf6")  # lands on the covered side: unknown for now
    assert s.feed(cover(s.frame(highlight=move)), 4) == []
    assert not any(isinstance(e, Resync) for e in s.events)
    assert s.settle(4) == ["Nf6"]  # the window is gone


def test_toast_over_a_corner_does_not_stall_the_game() -> None:
    s = Session()
    s.start()

    def toast(frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        out[130:230, 640:900] = (30, 30, 30)  # a notification over the top-right corner
        return out

    for san in ("e4", "c5", "Nf3", "d6"):
        move = s.board.push_san(san)
        assert s.feed(toast(s.frame(highlight=move)), 3) == [san]


def test_board_scrolled_partly_off_screen_and_back() -> None:
    s = Session()
    s.start()
    assert s.play("e4", "e5") == ["e4", "e5"]
    s.y0 = -200  # the page scrolled: the top rows are off the screen
    s.settle(8)  # no crash, nothing invented
    assert s.played == ["e4", "e5"]
    s.y0 = 120
    s.settle(6)
    assert s.play("Nf3") == ["Nf3"]


@pytest.mark.parametrize("plies", [2, 4])
def test_catch_up_after_a_gap_of_a_few_plies(plies: int) -> None:
    """The board was not visible for a moment; up to four plies are searched for."""
    s = Session()
    s.start()
    assert s.play("e4", "e5") == ["e4", "e5"]
    line = ["Nf3", "Nc6", "Bb5", "a6"][:plies]
    for san in line:
        s.board.push_san(san)
    assert s.settle(6) == line


@pytest.mark.parametrize(
    ("cell", "dx", "dy"),
    [
        (50, 0.0, 2.4),
        (50, -2.4, 0.0),
        (50, 2.0, 0.0),
        (50, 0.0, -0.8),
        (48, 3.0, 3.0),
        (64, -1.4, -1.4),
        # Its Nc6 did not land on the highlighted square until the landing compared the
        # reference and the piece at the detail both have (cfaa416).
        (48, -1.4, -1.4),
    ],
)
def test_a_board_moved_any_fraction_of_a_cell_is_found_again(
    cell: int, dx: float, dy: float
) -> None:
    """Moves that kept the old lock: two cells and a fraction (the cells it left behind still
    fit), 0.8 of a cell (the palette re-read inverted), two whole cells (a window part board
    and part background ranked first), the diagonals. The lock follows within a few frames
    and the game goes on (the parity audit's sweep, 2026-09-26)."""
    s = Session(x0=440, y0=200, cell=cell)
    s.start()
    assert s.play("e4", "e5") == ["e4", "e5"]
    s.x0, s.y0 = int(440 + dx * cell), int(200 + dy * cell)
    s.settle(8)
    b = s.pipe.board
    assert b is not None and abs(b.region.x - s.x0) <= 3 and abs(b.region.y - s.y0) <= 3
    s.play("Nf3", "Nc6")
    s.settle(3)
    assert s.played == ["e4", "e5", "Nf3", "Nc6"]
