"""The site's result badges: the discs chess.com stamps on the two kings when a game ends,
read for the result so that a resignation, a flag or an agreed draw -- which leave no
trace on the squares -- end the game with its result instead of the End game menu.

The real badges come from two live games as crops under tests/fixtures/badges (43 and
80 px squares, the winner's crown, the loser's fallen king in either colour, the badge
pushed inward on the board's edge); the rest is rendered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chess
import cv2
import numpy as np
import pytest

from sidecoach.events import Event, GameEnded, MovePlayed, Region, ScreenState, Side
from sidecoach.state.pipeline import BADGE_FRAMES, GamePipeline
from sidecoach.vision.result_badges import Badge, badge_at, badge_on, result_of
from sidecoach.vision.squares import BoardGeometry
from tests.synth import THEMES, draw_result_badge, synthetic_board

pytestmark = pytest.mark.vision

FIXTURES = Path(__file__).parent / "fixtures" / "badges"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
# The end of the sixth live game (white resigned in effect: mated on g1), and a middlegame.
LOST = "6k1/4rnp1/6Q1/p1p4P/2P2R2/6P1/1r4q1/5RK1 w - - 3 45"
MIDDLE = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8"
X0, Y0 = 200, 100


def _geometry(cell: int) -> BoardGeometry:
    return BoardGeometry(Region(X0, Y0, 8 * cell, 8 * cell), float(cell))


def _frame(board: chess.Board, cell: int = 64, **kw: Any) -> np.ndarray:
    return synthetic_board(board=board, cell=cell, x0=X0, y0=Y0, width=1400, height=1200, **kw)


# ---- the real badges ------------------------------------------------------------------


@pytest.mark.parametrize("entry", MANIFEST, ids=[e["file"] for e in MANIFEST])
def test_the_live_games_badges_read_as_recorded(entry: dict[str, object]) -> None:
    crop = cv2.imread(str(FIXTURES / str(entry["file"])))
    assert crop is not None
    cell = float(entry["cell"])  # type: ignore[arg-type]
    corner_x = float(entry["x0"]) + cell  # type: ignore[arg-type]
    corner_y = float(entry["y0"])  # type: ignore[arg-type]
    want = {"win": Badge.WIN, "loss": Badge.LOSS, "none": None}[str(entry["kind"])]
    assert badge_at(crop, corner_x, corner_y, cell) is want


def read_result(frame, geometry, board: chess.Board, bottom: Side) -> str | None:  # noqa: ANN001
    """The result the badges on the two kings say (`result_of`), as the pipeline pairs them."""
    white, black = board.king(chess.WHITE), board.king(chess.BLACK)
    if white is None or black is None:
        return None
    return result_of(
        badge_on(frame, geometry, white, bottom), badge_on(frame, geometry, black, bottom)
    )


# ---- rendered: every theme, size and orientation ----------------------------------------


@pytest.mark.parametrize("theme", sorted(THEMES))
@pytest.mark.parametrize("cell", [44, 72, 128])
@pytest.mark.parametrize("bottom", [Side.WHITE, Side.BLACK])
def test_every_theme_size_and_orientation_reads_the_result(
    theme: str, cell: int, bottom: Side
) -> None:
    board = chess.Board(LOST)
    for result in ("1-0", "0-1", None):
        frame = _frame(board, cell, theme=theme, bottom=bottom, result=result)
        assert read_result(frame, _geometry(cell), board, bottom) == result, (result, theme)
    # The draw badge is unseen; a rendered guess at it reads as nothing.
    frame = _frame(board, cell, theme=theme, bottom=bottom, result="1/2-1/2")
    assert read_result(frame, _geometry(cell), board, bottom) is None


def test_kings_on_the_boards_edges_are_read_where_the_site_pushes_the_badge_in() -> None:
    # White's king on h1 and black's on a8: the badge is pushed left on the right column
    # and down on the top row, and mirrored for the other orientation.
    board = chess.Board("k7/8/8/8/8/8/8/7K w - - 0 1")
    for bottom in (Side.WHITE, Side.BLACK):
        for result in ("1-0", "0-1"):
            frame = _frame(board, 56, bottom=bottom, result=result)
            assert read_result(frame, _geometry(56), board, bottom) == result


# ---- what is not a badge ----------------------------------------------------------------


def test_one_badge_alone_or_two_that_disagree_is_no_result() -> None:
    board = chess.Board(LOST)
    white_king, black_king = board.king(chess.WHITE), board.king(chess.BLACK)
    assert white_king is not None and black_king is not None
    frame = _frame(board, 64, result="1-0")
    geometry = _geometry(64)
    assert badge_on(frame, geometry, white_king, Side.WHITE) is Badge.WIN
    # A lone badge is nothing, and so is the same badge on both kings.
    white_only = _frame(board, 64)
    draw_result_badge(white_only, X0, Y0, 64, Side.WHITE, white_king, "win")
    assert read_result(white_only, geometry, board, Side.WHITE) is None
    both_win = white_only.copy()
    draw_result_badge(both_win, X0, Y0, 64, Side.WHITE, black_king, "win")
    assert read_result(both_win, geometry, board, Side.WHITE) is None


def test_the_boards_own_decorations_are_not_badges() -> None:
    """A check's red glow under the king, a highlight on its square, an arrow into it, a
    move-classification badge on it, the hover outline, a premove's red tint: none is a
    disc of one colour with a ring of others around it at the square's corner."""
    board = chess.Board(LOST)  # white to move, in check from the queen on g2
    assert board.is_check()
    king = board.king(chess.WHITE)
    geometry = _geometry(64)
    frames = {
        "check": _frame(board, check=True),
        "highlight": _frame(board, highlight=chess.Move.from_uci("h1g1")),
        "red mark": _frame(board, red_marks=(king,)),
        "green arrow": _frame(board, arrows=(chess.Move.from_uci("h4g1"),), arrow_colour="green"),
        "red arrow": _frame(board, arrows=(chess.Move.from_uci("h4g1"),), arrow_colour="red"),
        "best badge": _frame(board, badge=(king, "best")),
        "blunder badge": _frame(board, badge=(king, "blunder")),
        "hover": _frame(board, dragging=(king, king, 0.0, 0.0)),
        "premove": _frame(board, premoves=((chess.Move.from_uci("g1h2"),),)),
        "selected": _frame(board, selected=king),
    }
    for name, frame in frames.items():
        assert read_result(frame, geometry, board, Side.WHITE) is None, name


@pytest.mark.parametrize("bottom", [Side.WHITE, Side.BLACK])
def test_a_bulky_piece_walling_off_its_corner_is_not_a_badge(bottom: Side) -> None:
    """On the grey theme a bold king leaves a bare grey corner that a ring around it
    cannot see past: a disc of the square's own colour is the square (stress seed 60)."""
    board = chess.Board("rnbqkbnr/pp1Q1p1p/2p4p/4P3/8/2P2P2/PP2P1PP/RN2KBNR b KQkq - 0 6")
    for cell in (44, 56, 72):
        for check in (False, True):
            frame = _frame(board, cell, theme="grey", style="bold", bottom=bottom, check=check)
            assert read_result(frame, _geometry(cell), board, bottom) is None, (cell, check)


def test_no_king_no_result() -> None:
    board = chess.Board(None)
    board.set_piece_at(chess.E4, chess.Piece(chess.QUEEN, chess.WHITE))
    frame = _frame(board, 64)
    assert read_result(frame, _geometry(64), board, Side.WHITE) is None


# ---- the pipeline ends the game on them ---------------------------------------------------


class Game:
    """A pipeline fed rendered frames of a game played on `board`."""

    def __init__(self, bottom: Side = Side.WHITE) -> None:
        self.bottom = bottom
        self.pipe = GamePipeline()
        self.board = chess.Board()
        self.ts = 0.0
        self.events: list[Event] = []
        self.show(2)
        assert self.pipe.reconciler is not None and self.pipe.reconciler.started

    def show(self, frames: int = 2, **kw: Any) -> list[Event]:
        out: list[Event] = []
        for _ in range(frames):
            out += self.pipe.process(
                synthetic_board(board=self.board, bottom=self.bottom, **kw), self.ts
            )
            self.ts += 0.1
        self.events += out
        return out

    def play(self, *sans: str) -> None:
        for san in sans:
            move = self.board.push_san(san)
            self.show(2, highlight=move)

    def ended(self) -> list[GameEnded]:
        return [e for e in self.events if isinstance(e, GameEnded)]


@pytest.mark.parametrize("bottom", [Side.WHITE, Side.BLACK])
def test_a_resignation_ends_the_game_with_the_sites_result(bottom: Side) -> None:
    g = Game(bottom)
    g.play("e4", "e5", "Nf3", "Nc6", "Bb5")
    assert not g.ended()
    # Black resigns: the badges appear, the board unchanged.
    new = g.show(BADGE_FRAMES, highlight=chess.Move.from_uci("f1b5"), result="1-0")
    ended = [e for e in new if isinstance(e, GameEnded)]
    assert len(ended) == 1
    assert ended[0].result == "1-0"
    assert ended[0].reason == "resignation"
    assert ended[0].pgn is not None and ended[0].pgn.endswith("Bb5")
    assert g.pipe._screen_state is ScreenState.GAME_OVER
    # The badges stay on screen; the game does not end twice.
    again = g.show(3, highlight=chess.Move.from_uci("f1b5"), result="1-0")
    assert not any(isinstance(e, GameEnded) for e in again)


def test_a_draw_badge_is_not_read_until_one_is_on_record() -> None:
    """The rendered guess at a draw badge (grey discs) ends nothing: a draw ends through
    the dialog's cover or the End game menu, which carries the reason."""
    g = Game()
    g.play("e4", "e5", "Nf3", "Nc6")
    new = g.show(BADGE_FRAMES + 2, highlight=chess.Move.from_uci("b8c6"), result="1/2-1/2")
    assert not any(isinstance(e, GameEnded) for e in new)


def test_a_flag_is_a_loss_on_time() -> None:
    g = Game()
    g.play("e4", "e5")
    g.pipe._last_clock = (0, 240000, None)  # white's clock at zero
    new = g.show(BADGE_FRAMES, highlight=chess.Move.from_uci("e7e5"), result="0-1")
    ended = [e for e in new if isinstance(e, GameEnded)]
    assert [(e.result, e.reason) for e in ended] == [("0-1", "lost on time")]


def test_one_frame_of_badges_is_not_enough() -> None:
    g = Game()
    g.play("e4", "e5")
    new = g.show(1, highlight=chess.Move.from_uci("e7e5"), result="1-0")
    assert not any(isinstance(e, GameEnded) for e in new)
    # Gone again: a mark that flashed past.
    new = g.show(2, highlight=chess.Move.from_uci("e7e5"))
    assert not any(isinstance(e, GameEnded) for e in new)
    assert g.pipe._badges is None


def test_a_mate_on_the_board_ends_the_game_once_whatever_the_badges_say() -> None:
    g = Game()
    g.play("f3", "e5", "g4")
    move = g.board.push_san("Qh4#")
    # The mating move and the badges arrive together, and stay.
    new = g.show(4, highlight=move, result="0-1")
    ended = [e for e in new if isinstance(e, GameEnded)]
    assert len(ended) == 1
    assert ended[0].result == "0-1"
    assert "Qh4#" in [e.san for e in new if isinstance(e, MovePlayed)]


def test_the_reason_comes_from_the_board() -> None:
    pipe = GamePipeline()
    mate = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert pipe._end_reason(mate, "0-1") == "checkmate"
    assert pipe._end_reason(chess.Board(), "1-0") == "resignation"
    pipe._last_clock = (5000, 0, None)
    assert pipe._end_reason(chess.Board(), "1-0") == "lost on time"
    assert pipe._end_reason(chess.Board(), "0-1") == "resignation"
