"""The embedding: pieces named by the nearest cell of the start position in a frozen
pretrained net's embedding space. Its reading of sets of every style from a start
position of their own, and no recogniser at all without the net. (The pipeline follows
games with it everywhere: tests/test_stress.py, tests/test_board_sizes.py.)"""

from __future__ import annotations

import random

import chess
import pytest

from sidecoach.events import Region, Side
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
@pytest.mark.parametrize(("theme", "cell"), [("green", 48), ("dark", 112)])
def test_reads_a_set_from_its_own_start_position(style: str, theme: str, cell: int) -> None:
    """Every piece of a random middlegame named, every empty square empty, with the last
    move highlighted and the king's check glowing: each set at a small cell on the green
    board and a large one on the dark. Not a square wrong: on the dark board the side
    the embedding alone would vote is wrong for a pixel queen and a riohacha pawn, and
    the brightness and the fill outvote it (pieces._vote_side)."""
    rng = random.Random(7)
    board = chess.Board()
    for _ in range(30):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    geometry = BoardGeometry(Region(300, 120, 8 * cell, 8 * cell), float(cell))
    model = ThemeModel.from_start_position(
        _frame(chess.Board(), style, theme, cell), geometry, Side.WHITE
    )
    assert model is not None
    readings = model.classify_frame(
        _frame(board, style, theme, cell, highlight=board.move_stack[-1], check=True),
        geometry,
    )
    wrong = []
    for r in range(8):
        for c in range(8):
            piece = board.piece_at(square_at(r, c, Side.WHITE))
            got = readings[r][c]
            want = piece.symbol() if piece else None
            if want is None and got.kind != "empty":
                wrong.append((r, c, "empty", got.kind, got.symbol))
            elif want is not None and (got.kind != "piece" or got.symbol != want):
                wrong.append((r, c, want, got.kind, got.symbol, got.guess))
    assert not wrong, wrong


def test_without_the_net_nothing_is_learned(monkeypatch: pytest.MonkeyPatch) -> None:
    """The net ships in the package; a broken install without it learns no set, rather
    than read pieces some other way."""
    monkeypatch.setattr(CellEmbedder, "load", staticmethod(lambda path=None: None))
    cell = 64
    frame = _frame(chess.Board(), "cburnett", "green", cell)
    geometry = BoardGeometry(Region(300, 120, 8 * cell, 8 * cell), float(cell))
    assert ThemeModel.from_start_position(frame, geometry, Side.WHITE) is None
