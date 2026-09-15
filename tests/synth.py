"""Synthetic frames for vision tests: a rendered position on a busy background.

Piece types get distinct silhouettes so recognition can be tested: pawn circle, knight
triangle, bishop diamond, rook square, queen star, king cross. White pieces are light with a
dark outline, black pieces dark with a light outline, like every real piece set.
"""

from __future__ import annotations

import chess
import cv2
import numpy as np

from sidecoach.events import Side
from sidecoach.vision.squares import row_col

LIGHT = (208, 236, 238)
DARK = (86, 150, 118)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
TINT_ALPHA = 0.5


def tint(base: tuple[int, int, int], overlay: tuple[int, int, int]) -> tuple[int, int, int]:
    """Translucent overlay blended over a square colour, as chess.com draws highlights."""
    return tuple(
        int(round((1 - TINT_ALPHA) * b + TINT_ALPHA * o))
        for b, o in zip(base, overlay, strict=True)
    )  # type: ignore[return-value]


def _polygon(cx: float, cy: float, radius: float, points: list[tuple[float, float]]) -> np.ndarray:
    return np.array(
        [[int(round(cx + radius * px)), int(round(cy + radius * py))] for px, py in points],
        dtype=np.int32,
    )


def draw_piece(frame: np.ndarray, cx: int, cy: int, cell: int, symbol: str) -> None:
    white = symbol.isupper()
    fill = (245, 245, 245) if white else (35, 35, 35)
    edge = (0, 0, 0) if white else (210, 210, 210)
    kind = symbol.lower()
    radius = cell * 0.36
    if kind == "p":
        cv2.circle(frame, (cx, cy), int(cell * 0.24), fill, -1)
        cv2.circle(frame, (cx, cy), int(cell * 0.24), edge, 2)
        return
    if kind == "n":
        pts = _polygon(cx, cy, radius, [(-0.9, 0.8), (0.9, 0.8), (0.1, -0.9)])
    elif kind == "b":
        pts = _polygon(cx, cy, radius, [(0, -1), (0.7, 0), (0, 1), (-0.7, 0)])
    elif kind == "r":
        pts = _polygon(cx, cy, radius, [(-0.7, -0.8), (0.7, -0.8), (0.7, 0.8), (-0.7, 0.8)])
    elif kind == "q":
        star = []
        for i in range(10):
            angle = -np.pi / 2 + i * np.pi / 5
            r = 1.0 if i % 2 == 0 else 0.45
            star.append((r * np.cos(angle), r * np.sin(angle)))
        pts = _polygon(cx, cy, radius, star)
    else:  # king: cross
        pts = _polygon(
            cx,
            cy,
            radius,
            [
                (-0.3, -1),
                (0.3, -1),
                (0.3, -0.3),
                (1, -0.3),
                (1, 0.3),
                (0.3, 0.3),
                (0.3, 1),
                (-0.3, 1),
                (-0.3, 0.3),
                (-1, 0.3),
                (-1, -0.3),
                (-0.3, -0.3),
            ],
        )
    cv2.fillPoly(frame, [pts], fill)
    cv2.polylines(frame, [pts], True, edge, 2)


def render_board(
    frame: np.ndarray,
    x0: int,
    y0: int,
    cell: int,
    board: chess.Board,
    bottom: Side = Side.WHITE,
    *,
    light: tuple[int, int, int] = LIGHT,
    dark: tuple[int, int, int] = DARK,
    highlight: chess.Move | None = None,
    red_marks: tuple[chess.Square, ...] = (),
    dragging: tuple[chess.Square, chess.Square, float, float] | None = None,
) -> None:
    """Render `board`. `dragging` = (origin, hovered square, dx, dy): the origin's piece is
    drawn over the hovered square displaced by (dx, dy) cells, under a hover outline."""
    for r in range(8):
        for c in range(8):
            colour = light if (r + c) % 2 == 0 else dark
            cv2.rectangle(
                frame,
                (x0 + c * cell, y0 + r * cell),
                (x0 + (c + 1) * cell - 1, y0 + (r + 1) * cell - 1),
                colour,
                -1,
            )
    tinted: list[tuple[chess.Square, tuple[int, int, int]]] = []
    if highlight is not None:
        tinted += [(highlight.from_square, YELLOW), (highlight.to_square, YELLOW)]
    tinted += [(square, RED) for square in red_marks]
    for square, overlay in tinted:
        r, c = row_col(square, bottom)
        base = light if (r + c) % 2 == 0 else dark
        cv2.rectangle(
            frame,
            (x0 + c * cell, y0 + r * cell),
            (x0 + (c + 1) * cell - 1, y0 + (r + 1) * cell - 1),
            tint(base, overlay),
            -1,
        )
    lifted = dragging[0] if dragging else None
    for square, piece in board.piece_map().items():
        if square == lifted:
            continue
        r, c = row_col(square, bottom)
        draw_piece(
            frame, x0 + c * cell + cell // 2, y0 + r * cell + cell // 2, cell, piece.symbol()
        )
    if dragging is not None:
        origin, over, dx, dy = dragging
        piece = board.piece_at(origin)
        r, c = row_col(over, bottom)
        inset = max(1, cell // 32)
        cv2.rectangle(
            frame,
            (x0 + c * cell + inset, y0 + r * cell + inset),
            (x0 + (c + 1) * cell - 1 - inset, y0 + (r + 1) * cell - 1 - inset),
            (255, 255, 255),
            max(2, cell // 20),
        )
        if piece is not None:
            draw_piece(
                frame,
                x0 + c * cell + cell // 2 + int(round(dx * cell)),
                y0 + r * cell + cell // 2 + int(round(dy * cell)),
                cell,
                piece.symbol(),
            )


def busy_background(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse = rng.integers(40, 220, (height // 40 + 1, width // 40 + 1, 3), dtype=np.uint8)
    frame = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    for _ in range(30):
        x, y = rng.integers(0, width - 60), rng.integers(0, height - 60)
        w, h = rng.integers(10, 60), rng.integers(10, 60)
        colour = tuple(int(v) for v in rng.integers(0, 256, 3))
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), colour, -1)
    return frame


def synthetic_board(
    *,
    width: int = 1280,
    height: int = 800,
    x0: int = 300,
    y0: int = 120,
    cell: int = 64,
    light: tuple[int, int, int] = LIGHT,
    dark: tuple[int, int, int] = DARK,
    pieces: bool = True,
    board: chess.Board | None = None,
    bottom: Side = Side.WHITE,
    highlight: chess.Move | None = None,
    red_marks: tuple[chess.Square, ...] = (),
    dragging: tuple[chess.Square, chess.Square, float, float] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """A board on a busy background. `pieces=False` renders an empty board."""
    frame = busy_background(width, height, seed)
    position = board if board is not None else (chess.Board() if pieces else chess.Board(None))
    render_board(
        frame,
        x0,
        y0,
        cell,
        position,
        bottom,
        light=light,
        dark=dark,
        highlight=highlight,
        red_marks=red_marks,
        dragging=dragging,
    )
    return frame


def with_overlays(frame: np.ndarray, x0: int, y0: int, cell: int) -> np.ndarray:
    """Arrows, circles, a move-quality badge, extra highlights, and a cursor over the board."""
    out = frame.copy()
    overlay = out.copy()
    cv2.arrowedLine(
        overlay,
        (x0 + cell // 2, y0 + 7 * cell + cell // 2),
        (x0 + 4 * cell + cell // 2, y0 + 3 * cell + cell // 2),
        (60, 160, 60),
        max(3, cell // 6),
        tipLength=0.25,
    )
    cv2.arrowedLine(
        overlay,
        (x0 + 6 * cell + cell // 2, y0 + cell // 2),
        (x0 + 2 * cell + cell // 2, y0 + 4 * cell + cell // 2),
        (40, 40, 220),
        max(3, cell // 6),
        tipLength=0.25,
    )
    cv2.circle(
        overlay,
        (x0 + 5 * cell + cell // 2, y0 + 5 * cell + cell // 2),
        cell // 2 - 2,
        (40, 40, 220),
        3,
    )
    for r, c in ((2, 3), (3, 3), (5, 6)):
        cv2.rectangle(
            overlay,
            (x0 + c * cell, y0 + r * cell),
            (x0 + (c + 1) * cell - 1, y0 + (r + 1) * cell - 1),
            (90, 200, 250),
            -1,
        )
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
    badge = (x0 + 4 * cell - cell // 4, y0 + 6 * cell + cell // 4)
    cv2.circle(out, badge, cell // 5, (0, 200, 0), -1)
    cv2.putText(
        out,
        "!!",
        (badge[0] - cell // 8, badge[1] + cell // 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        cell / 120,
        (255, 255, 255),
        1,
    )
    cursor = np.array(
        [[0, 0], [0, 18], [5, 14], [9, 22], [12, 20], [8, 12], [14, 12]], dtype=np.int32
    )
    cursor += np.array([x0 + 3 * cell + 10, y0 + 2 * cell + 10])
    cv2.fillPoly(out, [cursor], (255, 255, 255))
    cv2.polylines(out, [cursor], True, (0, 0, 0), 1)
    return out
