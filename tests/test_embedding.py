"""The embedding: pieces named by the nearest cell of the start position in a frozen
pretrained net's embedding space. Its reading of sets of every style from a start
position of their own, the pipeline following a game with it, the start frame of a
shaded set on a board of its own tone, and no recogniser at all without the net."""

from __future__ import annotations

import random

import chess
import pytest

from sidecoach.events import GameStarted, MovePlayed, Region, Side
from sidecoach.state.pipeline import GamePipeline
from sidecoach.vision.embedding import CellEmbedder
from sidecoach.vision.pieces import ThemeModel
from sidecoach.vision.squares import BoardGeometry, square_at
from tests.synth import synthetic_board


def _frame(board: chess.Board, style: str, theme: str, cell: int, **kw: object):
    return synthetic_board(
        width=8 * cell + 400,
        height=8 * cell + 200,
        board=board,
        theme=theme,
        cell=cell,
        style=style,
        coordinates=True,
        **kw,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("style", ["kosal", "pixel", "riohacha", "chess7", "cburnett-wood"])
@pytest.mark.parametrize("theme", ["green", "dark"])
def test_reads_a_set_from_its_own_start_position(style: str, theme: str) -> None:
    """Every piece of a random middlegame named, every empty square empty, at a small
    and a large cell, with the last move highlighted and the king's check glowing."""
    rng = random.Random(7)
    board = chess.Board()
    for _ in range(30):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    wrong = []
    for cell in (48, 112):
        geometry = BoardGeometry(Region(300, 120, 8 * cell, 8 * cell), float(cell))
        model = ThemeModel.from_start_position(
            _frame(chess.Board(), style, theme, cell), geometry, Side.WHITE
        )
        assert model is not None
        readings = model.classify_frame(
            _frame(board, style, theme, cell, highlight=board.move_stack[-1], check=True),
            geometry,
        )
        for r in range(8):
            for c in range(8):
                piece = board.piece_at(square_at(r, c, Side.WHITE))
                got = readings[r][c]
                want = piece.symbol() if piece else None
                if want is None and got.kind != "empty":
                    wrong.append((cell, r, c, "empty", got.kind, got.symbol))
                elif want is not None and (got.kind != "piece" or got.symbol != want):
                    wrong.append((cell, r, c, want, got.kind, got.symbol, got.guess))
    assert len(wrong) <= 1, wrong


def test_the_pipeline_follows_a_game_with_the_embedding() -> None:
    pipe = GamePipeline()
    board = chess.Board()
    cell = 80
    ts = 0.0
    got: list[str] = []

    def show(frames: int, **kw: object) -> None:
        nonlocal ts
        for _ in range(frames):
            for event in pipe.process(_frame(board, "tatiana", "brown", cell, **kw), ts):
                if isinstance(event, MovePlayed):
                    got.append(event.san)
                if isinstance(event, GameStarted):
                    got.append("start")
            ts += 0.1

    show(3)
    assert pipe.model is not None
    assert got == ["start"]
    line = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Bxc6", "dxc6", "O-O", "Bg4"]
    for san in line:
        move = board.push_san(san)
        show(3, highlight=move)
    assert got[1:] == line


def test_a_shaded_set_on_its_own_tone_bootstraps() -> None:
    """The shaded facet of the dark pieces is the square's colour: their silhouettes are
    half a piece, off centre. The embedding reads the start position all the same."""
    cell = 64
    frame = _frame(chess.Board(), "riohacha", "dark", cell)
    geometry = BoardGeometry(Region(300, 120, 8 * cell, 8 * cell), float(cell))
    model = ThemeModel.from_start_position(frame, geometry, Side.WHITE)
    assert model is not None
    readings = model.classify_frame(frame, geometry)
    start = chess.Board()
    wrong = [
        (r, c, readings[r][c].kind, readings[r][c].symbol)
        for r in range(8)
        for c in range(8)
        if (piece := start.piece_at(square_at(r, c, Side.WHITE))) is not None
        and readings[r][c].symbol != piece.symbol()
    ]
    assert len(wrong) <= 1, wrong


def test_without_the_net_nothing_is_learned(monkeypatch: pytest.MonkeyPatch) -> None:
    """The net ships in the package; a broken install without it learns no set, rather
    than read pieces some other way."""
    monkeypatch.setattr(CellEmbedder, "load", staticmethod(lambda path=None: None))
    cell = 64
    frame = _frame(chess.Board(), "cburnett", "green", cell)
    geometry = BoardGeometry(Region(300, 120, 8 * cell, 8 * cell), float(cell))
    assert ThemeModel.from_start_position(frame, geometry, Side.WHITE) is None
