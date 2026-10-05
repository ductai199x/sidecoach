"""Synthetic frames for vision tests: a rendered position on a busy background.

Piece types get distinct silhouettes so recognition can be tested: pawn circle, knight
triangle, bishop diamond, rook square, queen star, king cross. White pieces are light with a
dark outline, black pieces dark with a light outline, like every real piece set, and the
edges are anti-aliased, as a site draws them: the blend of piece and square along an edge
is what a naive overlay measure took for an arrow.

What real use put on the board, and what this renders: the site's themes (`THEMES`: the
green default, chess.com's brown, a blue and a grey), the last-move and selection
highlights, right-click red marks, premove queues, a piece in flight under the hover
outline, translucent arrows with a head that lands on a piece (`draw_arrow`), the
move-classification badge in a cell's corner (`draw_badge`), the coordinate labels the
orientation is read from (`coordinates=True`), clocks, and dialogs (test_occlusion).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass
from typing import Any

import chess
import cv2
import numpy as np

from sidecoach.events import Region, Side
from sidecoach.vision.board_detect import BoardCandidate
from sidecoach.vision.clock import GlyphTemplates
from sidecoach.vision.squares import BoardGeometry, row_col
from tests.pieceset import WOODEN_SET, draw_sprite, sheet_sets

LIGHT = (208, 236, 238)
DARK = (86, 150, 118)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
TINT_ALPHA = 0.5
# Square colours (light, dark) in BGR, as the sites draw them. Brown is chess.com's default
# (the trial games were played on it); green is chess.com's green and the default here;
# "dark" is a dark board (light squares darker than a white piece's outline is bright);
# "purple" and "ice" are lichess-like; "wood" is brown with a grain texture (see
# `TEXTURED`), the way chess.com's wood themes and lichess's "maple" draw their squares.
THEMES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "green": (LIGHT, DARK),
    "brown": ((181, 217, 240), (99, 136, 181)),
    "blue": ((222, 227, 234), (166, 130, 93)),
    "grey": ((190, 190, 190), (120, 120, 120)),
    "dark": ((110, 110, 110), (62, 62, 62)),
    "purple": ((224, 214, 240), (168, 132, 180)),
    "ice": ((232, 226, 214), (172, 152, 124)),
    "wood": ((150, 200, 230), (80, 120, 170)),
}
TEXTURED = frozenset({"wood"})
# A grain: low-amplitude noise stretched along the rows, added to every square.
TEXTURE_AMPLITUDE = 12


@dataclass(frozen=True)
class PieceStyle:
    """How a set draws its pieces: the fills and outlines of the two sides, the outline's
    width as a share of the cell, and the size of the pieces in their squares. A style
    with `real` set draws the cburnett set (tests/pieceset.py) instead of the polygons,
    in the colours `colouring` swaps in; `scale` then sizes the sprite in the cell."""

    white_fill: tuple[int, int, int] = (245, 245, 245)
    white_edge: tuple[int, int, int] = (0, 0, 0)
    black_fill: tuple[int, int, int] = (35, 35, 35)
    black_edge: tuple[int, int, int] = (210, 210, 210)
    outline: float = 1 / 32  # of the cell
    scale: float = 1.0  # 1.0 is the default radius; sites draw pieces at 0.85 to 1.15
    # Detail lines inside the piece in the outline's colour, as most real sets draw them
    # (a queen's crown, a rook's crenellations): the interior is then two tones.
    detail: bool = False
    real: bool = False
    colouring: tuple[tuple[str, str], ...] = ()
    sheet: str = ""  # a set from tests/fixtures/pieces, drawn instead of the vectors
    # How far below the square's centre the pieces rest, as a share of the cell: chess.com's
    # puzzles page draws its small start positions' pieces larger and lower than its boards.
    drop: float = 0.0


PIECE_STYLES: dict[str, PieceStyle] = {
    "classic": PieceStyle(),
    "thin": PieceStyle(outline=1 / 48, scale=0.9),
    "bold": PieceStyle(outline=1 / 20, scale=1.1),
    # A coloured set: cream against dark brown, both sides chromatic.
    "wooden": PieceStyle(
        white_fill=(190, 225, 245),
        white_edge=(40, 70, 100),
        black_fill=(30, 55, 90),
        black_edge=(150, 190, 215),
    ),
    # A set whose outlines are faint: white on white, black on black, as some flat
    # sets draw them.
    "flat": PieceStyle(white_edge=(160, 160, 160), black_edge=(90, 90, 90), outline=1 / 40),
    # A set with lines drawn inside every piece.
    "detailed": PieceStyle(detail=True),
    # The real thing: lichess's default set, drawn at the size of the cell as sites draw
    # their vectors; then the same set in a wooden colouring, and drawn small.
    "cburnett": PieceStyle(real=True),
    "cburnett-wood": PieceStyle(
        real=True,
        colouring=tuple(sorted(WOODEN_SET.items())),
        white_fill=(190, 225, 245),
        white_edge=(45, 70, 90),
        black_fill=(45, 70, 90),
        black_edge=(143, 188, 215),
    ),
    "cburnett-small": PieceStyle(real=True, scale=0.85),
}
# Every set lichess draws, from its sprite sheets (tests/fixtures/pieces/README.md).
PIECE_STYLES.update({name: PieceStyle(real=True, sheet=name) for name in sheet_sets()})
# chess.com's legal-move hints when a piece is selected: a grey translucent disc on an
# empty destination, a ring around a capturable piece.
HINT = (0, 0, 0)
HINT_ALPHA = 0.14
# The check glow: a red radial gradient under the checked king, opaque at the centre.
CHECK = (0, 0, 255)
# Highlight colours other sites use for the last move: chess.com's yellow, lichess's green,
# a blue.
HIGHLIGHTS: dict[str, tuple[int, int, int]] = {
    "yellow": YELLOW,
    "green": (60, 220, 140),
    "blue": (230, 160, 60),
}
# chess.com's arrow: orange, drawn translucent over squares and pieces alike.
ARROW = (36, 179, 252)
ARROW_ALPHA = 0.8
# Arrow colours a site or a user draws: chess.com's orange, and green, blue and red.
ARROWS: dict[str, tuple[int, int, int]] = {
    "orange": ARROW,
    "green": (80, 170, 40),
    "blue": (230, 130, 40),
    "red": (50, 50, 220),
}
# The stress sweep's first find (tests/test_stress.py): a theme whose light squares lie
# within BG_TOLERANCE (25 grey levels) of the black pieces' light outline -- (214, 214,
# 214) against an outline of 210 -- hides that outline on light squares, so a black piece
# had one silhouette on light squares and another on dark ones. Matching on the piece's
# own tone (pieces.Silhouette.tone) made it readable; kept as the regression case.
LOW_CONTRAST_GREY = ((214, 214, 214), (140, 140, 140))
# The classification badges: the fill colour of each, white glyph on top.
BADGES: dict[str, tuple[int, int, int]] = {
    "brilliant": (170, 165, 38),
    "great": (216, 156, 91),
    "best": (86, 172, 129),
    "excellent": (86, 172, 129),
    "good": (150, 160, 120),
    "book": (118, 106, 90),
    "inaccuracy": (66, 187, 245),
    "mistake": (52, 141, 255),
    "miss": (73, 88, 255),
    "blunder": (72, 62, 250),
}


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


def draw_piece(
    frame: np.ndarray,
    cx: int,
    cy: int,
    cell: int,
    symbol: str,
    style: PieceStyle = PIECE_STYLES["classic"],
) -> None:
    cy += int(round(cell * style.drop))
    if style.real:
        size = max(1, int(round(cell * style.scale)))
        draw_sprite(
            frame,
            cx - size // 2,
            cy - size // 2,
            size,
            symbol,
            dict(style.colouring),
            sheet=style.sheet or None,
        )
        return
    white = symbol.isupper()
    fill = style.white_fill if white else style.black_fill
    edge = style.white_edge if white else style.black_edge
    kind = symbol.lower()
    radius = cell * 0.36 * style.scale
    thickness = max(1, int(round(cell * style.outline)))
    if kind == "p":
        r = int(cell * 0.24 * style.scale)
        cv2.circle(frame, (cx, cy), r, fill, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r, edge, thickness, lineType=cv2.LINE_AA)
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
    cv2.fillPoly(frame, [pts], fill, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [pts], True, edge, thickness, lineType=cv2.LINE_AA)
    if style.detail:
        # A few strokes inside the shape, thinner than the outline as sets draw them:
        # chords across it at a third and two thirds of its height, and one down the
        # middle for the taller pieces.
        thickness = max(1, thickness // 2)
        span = radius * 0.5
        for dy in (-radius * 0.3, radius * 0.3):
            y = int(round(cy + dy))
            cv2.line(
                frame,
                (int(round(cx - span)), y),
                (int(round(cx + span)), y),
                edge,
                thickness,
                lineType=cv2.LINE_AA,
            )
        if kind in "kq":
            cv2.line(
                frame,
                (cx, int(round(cy - radius * 0.6))),
                (cx, int(round(cy + radius * 0.6))),
                edge,
                thickness,
                lineType=cv2.LINE_AA,
            )


def draw_hints(
    frame: np.ndarray,
    x0: int,
    y0: int,
    cell: int,
    bottom: Side,
    board: chess.Board,
    square: chess.Square,
) -> None:
    """The legal-move hints chess.com shows for the selected piece on `square`: a small
    translucent disc on every empty destination, a translucent ring round every capture."""
    overlay = frame.copy()
    for move in board.legal_moves:
        if move.from_square != square:
            continue
        r, c = row_col(move.to_square, bottom)
        centre = (x0 + c * cell + cell // 2, y0 + r * cell + cell // 2)
        if board.is_capture(move):
            cv2.circle(
                overlay, centre, cell // 2 - 1, HINT, max(2, cell // 12), lineType=cv2.LINE_AA
            )
        else:
            cv2.circle(overlay, centre, max(3, int(cell * 0.17)), HINT, -1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, HINT_ALPHA, frame, 1.0 - HINT_ALPHA, 0.0, frame)


def draw_check(
    frame: np.ndarray, x0: int, y0: int, cell: int, bottom: Side, square: chess.Square
) -> None:
    """chess.com's check mark: a red radial glow under the king, solid at the centre and
    fading to the square's colour at the edge."""
    r, c = row_col(square, bottom)
    patch = frame[y0 + r * cell : y0 + (r + 1) * cell, x0 + c * cell : x0 + (c + 1) * cell]
    ys, xs = np.mgrid[0:cell, 0:cell]
    distance = np.hypot(xs - (cell - 1) / 2, ys - (cell - 1) / 2) / (cell / 2)
    alpha = np.clip(1.0 - distance, 0.0, 1.0)[..., None].astype(np.float32)
    red = np.array(CHECK, dtype=np.float32)
    patch[:] = np.clip(patch.astype(np.float32) * (1 - alpha) + red * alpha, 0, 255).astype(
        np.uint8
    )


def draw_arrow(
    frame: np.ndarray,
    x0: int,
    y0: int,
    cell: int,
    bottom: Side,
    move: chess.Move,
    *,
    colour: tuple[int, int, int] = ARROW,
    alpha: float = ARROW_ALPHA,
    width: float = 0.2,
) -> None:
    """An arrow from one square's centre to another's, the way chess.com draws it: a shaft
    `width` of a cell wide and a triangular head half a cell across, translucent, with
    anti-aliased edges, over whatever is on the squares -- the head lands on the piece. A
    knight's move is drawn as chess.com draws it, an L: two cells along the long leg,
    then the turn, so the arrow crosses different squares than the straight line would."""
    r0, c0 = row_col(move.from_square, bottom)
    r1, c1 = row_col(move.to_square, bottom)
    start = np.array([x0 + c0 * cell + cell / 2, y0 + r0 * cell + cell / 2])
    end = np.array([x0 + c1 * cell + cell / 2, y0 + r1 * cell + cell / 2])
    knight = {abs(r1 - r0), abs(c1 - c0)} == {1, 2}
    if knight:
        corner = np.array([end[0], start[1]]) if abs(c1 - c0) == 2 else np.array([start[0], end[1]])
        legs = [(start, corner), (corner, end)]
    else:
        legs = [(start, end)]
    shaft_width = cell * width
    overlay = frame.copy()
    for i, (a, b) in enumerate(legs):
        direction = b - a
        length = float(np.linalg.norm(direction))
        if length < 1:
            continue
        unit = direction / length
        normal = np.array([-unit[1], unit[0]])
        last = i == len(legs) - 1
        head_length = cell * 0.5 if last else 0.0
        base = b - unit * head_length
        # The first leg of an L runs on past the corner by half the shaft, to meet the
        # second leg squarely.
        far = base if last else base + unit * shaft_width / 2
        shaft = np.array(
            [
                a + normal * shaft_width / 2,
                far + normal * shaft_width / 2,
                far - normal * shaft_width / 2,
                a - normal * shaft_width / 2,
            ]
        )
        polygons = [shaft]
        if last:
            head_width = cell * 0.5
            polygons.append(
                np.array([base + normal * head_width / 2, b, base - normal * head_width / 2])
            )
        for polygon in polygons:
            cv2.fillPoly(
                overlay, [np.round(polygon).astype(np.int32)], colour, lineType=cv2.LINE_AA
            )
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)


def draw_badge(
    frame: np.ndarray,
    x0: int,
    y0: int,
    cell: int,
    bottom: Side,
    square: chess.Square,
    label: str = "best",
) -> None:
    """A move-classification badge on the square's top-right corner, overlapping the cell's
    edge as chess.com's do: a filled disc with a white glyph."""
    r, c = row_col(square, bottom)
    centre = (x0 + (c + 1) * cell - cell // 6, y0 + r * cell + cell // 6)
    radius = max(4, cell // 5)
    cv2.circle(frame, centre, radius, BADGES.get(label, BADGES["best"]), -1, lineType=cv2.LINE_AA)
    glyph = {"blunder": "??", "mistake": "?", "brilliant": "!!", "great": "!"}.get(label, "*")
    cv2.putText(
        frame,
        glyph,
        (centre[0] - radius // 2, centre[1] + radius // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        cell / 130,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )


# chess.com's result badges, as measured on the live games: the winner's green, the
# loser's red, a grey for a draw (BGR).
RESULT_COLOURS = {"win": (96, 187, 141), "loss": (38, 38, 211), "draw": (150, 150, 150)}


def draw_result_badge(
    frame: np.ndarray,
    x0: int,
    y0: int,
    cell: int,
    bottom: Side,
    square: chess.Square,
    kind: str,
    icon: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """The site's end-of-game badge on a king's square: a disc about a third of a square
    across just inside the top-right corner on screen, pushed inward on the board's top
    row and right column, with an icon at its centre (a crown for the winner, the fallen
    king in the loser's colour, a half for a draw)."""
    r, c = row_col(square, bottom)
    radius = max(4, int(round(0.17 * cell)))
    cx = x0 + (c + 1) * cell - (0.21 if c == 7 else 0.04) * cell
    cy = y0 + r * cell + (0.21 if r == 0 else 0.08) * cell
    centre = (int(round(cx)), int(round(cy)))
    cv2.circle(frame, centre, radius, RESULT_COLOURS[kind], -1, lineType=cv2.LINE_AA)
    h = radius * 0.55
    if kind == "win":
        pts = [(-0.9, 0.4), (-0.9, -0.3), (-0.45, 0.1), (0, -0.7), (0.45, 0.1), (0.9, -0.3)]
        pts.append((0.9, 0.4))
        poly = np.array([[cx + px * h, cy + py * h] for px, py in pts], dtype=np.int32)
        cv2.fillPoly(frame, [poly], icon, lineType=cv2.LINE_AA)
    elif kind == "loss":
        pts = [(-0.9, -0.2), (0.3, 0.7), (0.8, 0.2), (-0.1, -0.7), (-0.6, -0.9), (-0.9, -0.6)]
        poly = np.array([[cx + px * h, cy + py * h] for px, py in pts], dtype=np.int32)
        cv2.fillPoly(frame, [poly], icon, lineType=cv2.LINE_AA)
    else:
        cv2.putText(
            frame,
            "=",
            (int(cx - h * 0.7), int(cy + h * 0.6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            radius / 14,
            icon,
            max(1, radius // 5),
            lineType=cv2.LINE_AA,
        )


def draw_result_label(
    frame: np.ndarray,
    x0: int,
    y0: int,
    cell: int,
    bottom: Side,
    square: chess.Square,
    kind: str,
    alpha: float = 1.0,
) -> None:
    """The label a site shows on a king the moment the game ends, before its dialog: a
    light pill straddling the square's top edge, the word in the result's colour
    ("Checkmate" in red, "Winner" in green), fading in with `alpha`. It runs over the
    neighbouring squares, to the left when the square is on the right edge."""
    r, c = row_col(square, bottom)
    width = int(round(1.7 * cell))
    height = max(8, int(round(0.34 * cell)))
    top = y0 + r * cell - height // 2
    left = x0 + c * cell + int(round(0.25 * cell))
    if c >= 6:
        left = x0 + (c + 1) * cell - int(round(0.25 * cell)) - width
    overlay = frame.copy()
    radius = height // 2
    cv2.rectangle(
        overlay, (left + radius, top), (left + width - radius, top + height), (255, 255, 255), -1
    )
    cv2.circle(
        overlay, (left + radius, top + radius), radius, (255, 255, 255), -1, lineType=cv2.LINE_AA
    )
    cv2.circle(
        overlay,
        (left + width - radius, top + radius),
        radius,
        (255, 255, 255),
        -1,
        lineType=cv2.LINE_AA,
    )
    word = {"win": "Winner", "loss": "Checkmate", "draw": "Draw"}[kind]
    colour = {"win": (70, 170, 110), "loss": (53, 59, 225), "draw": (120, 120, 120)}[kind]
    scale = height / 28.0
    cv2.putText(
        overlay,
        word,
        (left + radius, top + height - max(2, int(round(0.28 * height)))),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        max(1, int(round(scale * 2))),
        lineType=cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


def draw_dialog(frame: np.ndarray, x0: int, y0: int, cell: int, alpha: float = 1.0) -> None:
    """A site's game-over dialog over the middle of the board, fading in with `alpha`: a
    dark rounded panel about five squares across, a darker header with a title, a
    round avatar, a wide green button and two grey ones below it -- a body of several
    colours, as the real ones have."""
    width, height = int(5.2 * cell), int(5.6 * cell)
    left = x0 + 4 * cell - width // 2
    top = y0 + 4 * cell - height // 2 - int(0.3 * cell)
    overlay = frame.copy()
    panel = (41, 38, 43)
    cv2.rectangle(overlay, (left, top), (left + width, top + height), panel, -1)
    cv2.rectangle(overlay, (left, top), (left + width, top + int(1.3 * cell)), (33, 31, 35), -1)
    title = max(0.3, cell / 90.0)
    cv2.putText(
        overlay,
        "You Won!",
        (left + int(1.2 * cell), top + int(0.8 * cell)),
        cv2.FONT_HERSHEY_SIMPLEX,
        title,
        (235, 235, 235),
        max(1, int(title * 2)),
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "by checkmate",
        (left + int(1.6 * cell), top + int(1.15 * cell)),
        cv2.FONT_HERSHEY_SIMPLEX,
        title * 0.55,
        (170, 170, 170),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.circle(
        overlay,
        (left + int(0.9 * cell), top + int(2.2 * cell)),
        int(0.45 * cell),
        (120, 110, 100),
        -1,
    )
    cv2.rectangle(
        overlay,
        (left + int(1.6 * cell), top + int(1.8 * cell)),
        (left + width - int(0.4 * cell), top + int(2.7 * cell)),
        (60, 58, 62),
        -1,
    )
    cv2.rectangle(
        overlay,
        (left + int(0.4 * cell), top + int(3.5 * cell)),
        (left + width - int(0.4 * cell), top + int(4.3 * cell)),
        (76, 182, 129),
        -1,
    )
    cv2.putText(
        overlay,
        "Game Review",
        (left + int(1.3 * cell), top + int(4.05 * cell)),
        cv2.FONT_HERSHEY_SIMPLEX,
        title * 0.8,
        (255, 255, 255),
        max(1, int(title * 2)),
        lineType=cv2.LINE_AA,
    )
    for i in range(2):
        bx = left + int(0.4 * cell) + i * int(2.3 * cell)
        cv2.rectangle(
            overlay,
            (bx, top + int(4.6 * cell)),
            (bx + int(2.1 * cell), top + int(5.2 * cell)),
            (58, 56, 60),
            -1,
        )
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


def draw_coordinates(
    frame: np.ndarray,
    x0: int,
    y0: int,
    cell: int,
    bottom: Side,
    light: tuple[int, int, int],
    dark: tuple[int, int, int],
) -> None:
    """Rank digits in the top-left corner of the left-edge cells and file letters in the
    bottom-right corner of the bottom-edge cells, each in the other square colour, as
    chess.com draws them; the orientation reader learns the digits from a start frame."""
    scale = cell / 110
    for row in range(8):
        rank = (8 - row) if bottom == Side.WHITE else (row + 1)
        ink = dark if row % 2 == 0 else light  # the top-left cell is light
        cv2.putText(
            frame,
            str(rank),
            (x0 + max(2, cell // 20), y0 + row * cell + int(cell * 0.28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            ink,
            max(1, cell // 40),
            lineType=cv2.LINE_AA,
        )
    for col in range(8):
        letter = "abcdefgh"[col] if bottom == Side.WHITE else "hgfedcba"[col]
        ink = light if (7 + col) % 2 == 0 else dark
        cv2.putText(
            frame,
            letter,
            (x0 + (col + 1) * cell - int(cell * 0.22), y0 + 8 * cell - max(2, cell // 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            ink,
            max(1, cell // 40),
            lineType=cv2.LINE_AA,
        )


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
    premoves: tuple[tuple[chess.Move, ...], ...] = (),
    arrows: tuple[chess.Move, ...] = (),
    badge: tuple[chess.Square, str] | None = None,
    coordinates: bool = False,
    style: PieceStyle = PIECE_STYLES["classic"],
    highlight_colour: tuple[int, int, int] = YELLOW,
    arrow_colour: tuple[int, int, int] = ARROW,
    arrow_width: float = 0.2,
    texture: bool = False,
    seed: int = 0,
    selected: chess.Square | None = None,
    check: bool = False,
    animating: tuple[chess.Square, chess.Square, float] | None = None,
    result: str | None = None,
    labels: float | None = None,
    dialog: float | None = None,
    badges: bool = True,
    drag_outline: bool = True,
) -> None:
    """Render `board`. `dragging` = (origin, hovered square, dx, dy): the origin's piece is
    drawn over the hovered square displaced by (dx, dy) cells, under a hover outline.
    `animating` = (origin, destination, progress): the site animating a move it has
    already made -- the origin's piece drawn that far along the straight line to the
    destination, no outline, whatever is on the destination still there under it.
    `arrows` are drawn over the pieces; `badge` is (square, label); `coordinates` draws
    the rank and file labels; `style` is the piece set; `texture` gives the squares a
    wood grain; `selected` is a piece the user has clicked, its square highlighted and
    its legal moves hinted; `check` draws the red glow under a king in check.

    `drag_outline=False` draws a drag the way chess.com's phone app does: the piece under
    the finger and no outline round the square it is over.

    `premoves` are queues the user has set up while the opponent thinks, drawn the way
    chess.com draws them: every square of a queue is tinted red, the piece is drawn at
    the end of its queue and its real square is left empty. A queue is a tuple of moves
    of one piece, each from the last one's destination; several queues are several
    pieces.

    `result` ("1-0", "0-1", "1/2-1/2") stamps the site's badges on the two kings;
    `labels` (an alpha) shows the result labels on them instead of, or as well as, the
    badges; `dialog` (an alpha) draws the game-over dialog over the middle of the board.
    """
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
    if texture:
        _grain(frame, x0, y0, 8 * cell, seed)
    tinted: list[tuple[chess.Square, tuple[int, int, int]]] = []
    if highlight is not None:
        tinted += [
            (highlight.from_square, highlight_colour),
            (highlight.to_square, highlight_colour),
        ]
    if selected is not None:
        tinted.append((selected, highlight_colour))
    tinted += [(square, RED) for square in red_marks]
    queued_from: dict[chess.Square, chess.Square] = {}  # the piece's real square -> queue end
    # A pawn queued to promotion is drawn as the piece it becomes, from then on (chess.com's
    # queen: Tai's game, 2026-09-28, b6-b1=Q premoved and drawn as a queen on b1).
    promoted: dict[chess.Square, int] = {}
    for queue in premoves:
        for move in queue:
            tinted += [(move.from_square, RED), (move.to_square, RED)]
            if move.promotion:
                promoted[queue[0].from_square] = move.promotion
        if queue:
            queued_from[queue[0].from_square] = queue[-1].to_square
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
    king = board.king(board.turn)
    if check and king is not None and board.is_check():
        draw_check(frame, x0, y0, cell, bottom, king)
    lifted = dragging[0] if dragging else None
    flying = animating[0] if animating else None
    for square, piece in board.piece_map().items():
        if square in (lifted, flying):
            continue
        if square in queued_from:
            continue  # drawn at the end of its queue instead
        if square in queued_from.values():
            continue  # a queued capture: the premoved piece is drawn over it
        r, c = row_col(square, bottom)
        draw_piece(
            frame, x0 + c * cell + cell // 2, y0 + r * cell + cell // 2, cell, piece.symbol(), style
        )
    for origin, end in queued_from.items():
        piece = board.piece_at(origin)
        if piece is None:
            continue
        if origin in promoted:
            piece = chess.Piece(promoted[origin], piece.color)
        r, c = row_col(end, bottom)
        draw_piece(
            frame, x0 + c * cell + cell // 2, y0 + r * cell + cell // 2, cell, piece.symbol(), style
        )
    if coordinates:
        draw_coordinates(frame, x0, y0, cell, bottom, light, dark)
    if selected is not None:
        draw_hints(frame, x0, y0, cell, bottom, board, selected)
    for arrow in arrows:
        draw_arrow(frame, x0, y0, cell, bottom, arrow, colour=arrow_colour, width=arrow_width)
    if badge is not None:
        draw_badge(frame, x0, y0, cell, bottom, badge[0], badge[1])
    if result is not None and badges:
        kinds = {"1-0": ("win", "loss"), "0-1": ("loss", "win"), "1/2-1/2": ("draw", "draw")}
        for colour, kind in zip((chess.WHITE, chess.BLACK), kinds[result], strict=True):
            king = board.king(colour)
            if king is not None:
                icon = (255, 255, 255) if colour == chess.WHITE or kind != "loss" else (20, 20, 20)
                draw_result_badge(frame, x0, y0, cell, bottom, king, kind, icon)
    if labels is not None and result is not None:
        kinds = {"1-0": ("win", "loss"), "0-1": ("loss", "win"), "1/2-1/2": ("draw", "draw")}
        for colour, kind in zip((chess.WHITE, chess.BLACK), kinds[result], strict=True):
            king = board.king(colour)
            if king is not None:
                draw_result_label(frame, x0, y0, cell, bottom, king, kind, alpha=labels)
    if dialog is not None:
        draw_dialog(frame, x0, y0, cell, alpha=dialog)
    if animating is not None:
        origin, destination, progress = animating
        piece = board.piece_at(origin)
        r0, c0 = row_col(origin, bottom)
        r1, c1 = row_col(destination, bottom)
        if piece is not None:
            draw_piece(
                frame,
                x0 + int(round((c0 + (c1 - c0) * progress) * cell)) + cell // 2,
                y0 + int(round((r0 + (r1 - r0) * progress) * cell)) + cell // 2,
                cell,
                piece.symbol(),
                style,
            )
    if dragging is not None:
        origin, over, dx, dy = dragging
        piece = board.piece_at(origin)
        r, c = row_col(over, bottom)
        inset = max(1, cell // 32)
        if drag_outline:
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
                style,
            )


def _grain(frame: np.ndarray, x0: int, y0: int, side: int, seed: int) -> None:
    """A wood grain over the board: noise stretched along the rows, a few levels deep. A
    board partly off the frame keeps the grain of the part on it."""
    rng = np.random.default_rng(seed + 17)
    rows = rng.normal(0.0, 1.0, (side // 6 + 1, side // 2 + 1)).astype(np.float32)
    grain = cv2.resize(rows, (side, side), interpolation=cv2.INTER_CUBIC)
    grain = np.clip(grain * TEXTURE_AMPLITUDE / 2, -TEXTURE_AMPLITUDE, TEXTURE_AMPLITUDE)
    top, left = max(0, y0), max(0, x0)
    bottom, right = min(frame.shape[0], y0 + side), min(frame.shape[1], x0 + side)
    if bottom <= top or right <= left:
        return
    grain = grain[top - y0 : bottom - y0, left - x0 : right - x0]
    patch = frame[top:bottom, left:right].astype(np.float32)
    frame[top:bottom, left:right] = np.clip(patch + grain[..., None], 0, 255).astype(np.uint8)


def bulky_pieces(
    frame: np.ndarray,
    x0: int,
    y0: int,
    cell: int,
    board: chess.Board,
    bottom: Side = Side.WHITE,
    *,
    radius: float = 0.47,
) -> None:
    """Redraw every piece of `board` as a disc of `radius` cells, so it reaches the cell's
    border ring: the coarse checker score of such a board drops well below an empty one's."""
    for square, piece in board.piece_map().items():
        r, c = row_col(square, bottom)
        white = piece.symbol().isupper()
        centre = (x0 + c * cell + cell // 2, y0 + r * cell + cell // 2)
        cv2.circle(
            frame, centre, int(cell * radius), (245, 245, 245) if white else (35, 35, 35), -1
        )
        cv2.circle(frame, centre, int(cell * radius), (0, 0, 0) if white else (210, 210, 210), 2)


def blend_rect(
    frame: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    colour: tuple[int, int, int],
    alpha: float,
) -> None:
    """Blend `colour` at `alpha` over the rectangle [x0, x1) x [y0, y1), in place."""
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(frame.shape[1], x1), min(frame.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    region = frame[y0:y1, x0:x1].astype(np.float32)
    overlay = np.array(colour, dtype=np.float32)
    frame[y0:y1, x0:x1] = np.clip((1.0 - alpha) * region + alpha * overlay, 0, 255).astype(
        frame.dtype
    )


def blend_outline(
    frame: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    colour: tuple[int, int, int],
    alpha: float,
    thickness: int,
    inset: int = 0,
) -> None:
    """Blend a square outline `thickness` pixels wide, `inset` pixels inside the rectangle
    [x0, x1) x [y0, y1), in place: the way a site draws the outline of a hovered square."""
    a, b, c, d = x0 + inset, y0 + inset, x1 - inset, y1 - inset
    blend_rect(frame, a, b, c, b + thickness, colour, alpha)
    blend_rect(frame, a, d - thickness, c, d, colour, alpha)
    blend_rect(frame, a, b + thickness, a + thickness, d - thickness, colour, alpha)
    blend_rect(frame, c - thickness, b + thickness, c, d - thickness, colour, alpha)


class Kept:
    """The frames a renderer drew last, up to `max_bytes`, by the arguments that drew them: the
    renderers here are pure (every random number comes from the `seed` argument), so the
    same arguments are the same pixels, and a frame asked for again is copied, not drawn
    again. Measured 2026-10-02: 57% of the suite's 4,840 boards and 98% of its 5,034
    backgrounds were drawn before, in the same process, mostly a test or two earlier (a
    frame shown for several ticks; a start position at every learning). Each caller gets
    its own copy to draw on; the kept one is read-only. `SIDECOACH_TEST_CACHE=0` (read at
    import, or `Kept.enabled` set by a test) draws every frame afresh."""

    enabled = os.environ.get("SIDECOACH_TEST_CACHE", "1") != "0"

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.bytes = 0
        # A frame kept whole, or as its difference from a background: (top, left, crop).
        self.frames: OrderedDict[Hashable, tuple[Any, ...]] = OrderedDict()
        self.hits = self.misses = 0

    def get(
        self,
        key: Hashable | None,
        draw: Callable[[], np.ndarray],
        base: Callable[[], np.ndarray] | None = None,
    ) -> np.ndarray:
        """The frame for `key`, drawn by `draw` the first time. With `base` (the background
        the frame was drawn on, itself kept), only the rectangle where the frame differs
        from it is kept, and the frame is put back together from the two -- the same pixels
        exactly (everything outside the rectangle IS the background) in a fraction of the
        memory: a 1920x1200 geometry frame is 6.9 MB whole, its board a few hundred kB."""
        if key is None or not Kept.enabled:
            return draw()
        kept = self.frames.get(key)
        if kept is not None:
            self.hits += 1
            self.frames.move_to_end(key)
            if len(kept) == 1:
                return kept[0].copy()
            top, left, crop = kept
            assert base is not None
            frame = base()
            frame[top : top + crop.shape[0], left : left + crop.shape[1]] = crop
            return frame
        self.misses += 1
        frame = draw()
        if base is None:
            stored: tuple[Any, ...] = (frame.copy(),)
        else:
            # Where the frame differs from its background, in any channel: the bounding box
            # of the non-zero pixels of the largest channel difference (OpenCV, in C).
            diff = cv2.absdiff(frame, base())
            left, top, w, h = cv2.boundingRect(
                cv2.max(cv2.max(diff[..., 0], diff[..., 1]), diff[..., 2])
            )
            stored = (top, left, frame[top : top + h, left : left + w].copy())
        for part in stored:
            if isinstance(part, np.ndarray):
                part.flags.writeable = False
                self.bytes += part.nbytes
        self.frames[key] = stored
        while self.bytes > self.max_bytes and len(self.frames) > 1:
            _, old = self.frames.popitem(last=False)
            self.bytes -= sum(p.nbytes for p in old if isinstance(p, np.ndarray))
        return frame


# A test process keeps at most 256 MB of frames (Tai, 2026-10-02: "max 256 MB x 8"): the
# backgrounds whole (a 1920x1200 one is 6.9 MB, and a test draws on a few), the boards as their
# difference from them.
MB = 1 << 20
_BACKGROUNDS = Kept(64 * MB)
_BOARDS = Kept(192 * MB)


def busy_background(width: int, height: int, seed: int) -> np.ndarray:
    """A desktop's clutter: coarse colour noise and thirty rectangles, from `seed`."""
    return _BACKGROUNDS.get((width, height, seed), lambda: _draw_background(width, height, seed))


def _draw_background(width: int, height: int, seed: int) -> np.ndarray:
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
    light: tuple[int, int, int] | None = None,
    dark: tuple[int, int, int] | None = None,
    theme: str = "green",
    pieces: bool = True,
    board: chess.Board | None = None,
    bottom: Side = Side.WHITE,
    highlight: chess.Move | None = None,
    red_marks: tuple[chess.Square, ...] = (),
    dragging: tuple[chess.Square, chess.Square, float, float] | None = None,
    premoves: tuple[tuple[chess.Move, ...], ...] = (),
    arrows: tuple[chess.Move, ...] = (),
    badge: tuple[chess.Square, str] | None = None,
    coordinates: bool = False,
    style: str | PieceStyle = "classic",
    highlight_colour: str = "yellow",
    arrow_colour: str = "orange",
    arrow_width: float = 0.2,
    seed: int = 0,
    selected: chess.Square | None = None,
    check: bool = False,
    animating: tuple[chess.Square, chess.Square, float] | None = None,
    result: str | None = None,
    labels: float | None = None,
    dialog: float | None = None,
    badges: bool = True,
    drag_outline: bool = True,
) -> np.ndarray:
    """A board on a busy background. `pieces=False` renders an empty board; `theme` names
    the square colours (see THEMES) unless `light` and `dark` are given; `style` the piece
    set (PIECE_STYLES); `highlight_colour` and `arrow_colour` name HIGHLIGHTS and ARROWS."""
    theme_light, theme_dark = THEMES[theme]
    light = theme_light if light is None else light
    dark = theme_dark if dark is None else dark
    position = board if board is not None else (chess.Board() if pieces else chess.Board(None))
    # Everything the pixels depend on, as drawn: the position by its FEN (the pieces, and
    # the side to move and the rights a check glow or the hint dots read), the colours and
    # the set as looked up now.
    drawn = {
        "light": light,
        "dark": dark,
        "highlight": highlight,
        "red_marks": red_marks,
        "dragging": dragging,
        "premoves": premoves,
        "arrows": arrows,
        "badge": badge,
        "coordinates": coordinates,
        "style": PIECE_STYLES[style] if isinstance(style, str) else style,
        "highlight_colour": HIGHLIGHTS[highlight_colour],
        "arrow_colour": ARROWS[arrow_colour],
        "arrow_width": arrow_width,
        "texture": theme in TEXTURED,
        "seed": seed,
        "selected": selected,
        "check": check,
        "animating": animating,
        "result": result,
        "labels": labels,
        "dialog": dialog,
        "badges": badges,
        "drag_outline": drag_outline,
    }

    def draw() -> np.ndarray:
        frame = busy_background(width, height, seed)
        render_board(frame, x0, y0, cell, position, bottom, **drawn)  # type: ignore[arg-type]
        return frame

    key: Hashable | None = (width, height, x0, y0, cell, position.fen(), bottom, *drawn.items())
    try:
        hash(key)
    except TypeError:  # a list where a tuple was meant: drawn, not kept
        key = None
    return _BOARDS.get(key, draw, base=lambda: busy_background(width, height, seed))


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


def _trim(canvas: np.ndarray) -> np.ndarray:
    """A template canvas cropped to the columns that hold ink, so scaling it to a glyph
    width scales the glyph and not its padding."""
    columns = np.flatnonzero(canvas.max(axis=0) > 0.02)
    if len(columns) == 0:
        return canvas
    return canvas[:, columns[0] : columns[-1] + 1]


def draw_clock_box(
    frame: np.ndarray,
    x_right: int,
    y_centre: int,
    cell: float,
    text: str,
    *,
    light: bool = True,
    icon_angle: float | None = None,
    colours: tuple[int, int] | None = None,
    templates: GlyphTemplates | None = None,
) -> None:
    """Draw a chess.com clock box, right-aligned at `x_right`, the way the site draws one.

    The glyphs are the learned templates, scaled to the digit height a board of this cell
    size implies, so a test reads back what the reader was trained on. `light` picks the
    polarity: white's box is light with dark text, black's dark with light text.
    `icon_angle` is the hand of the running clock's face in radians; None draws the idle
    dash instead. `colours` overrides the (background, text) greys of the theme.
    """
    templates = templates or GlyphTemplates.load()
    height = max(6, int(round(0.47 * cell)))
    line = max(4, int(round(0.23 * cell)))
    pad = max(1, int(round(0.12 * cell)))
    background, ink = colours if colours is not None else ((255, 40) if light else (36, 250))
    glyphs = [_trim(templates.canvas_for(ch, light)[0]) for ch in text]
    widths = [max(1, int(round(templates.canvas_for(ch, light)[1] * line))) for ch in text]
    gap = max(1, line // 10)
    icon = int(round(0.55 * height))
    # chess.com's box is a fixed width with the digits right-aligned in it, so the icon
    # sits well clear of them; a box only as wide as its content would not.
    content = sum(widths) + gap * (len(glyphs) - 1) + 3 * pad + icon
    width = max(content, int(round(1.56 * cell)))
    x0, y0 = x_right - width, y_centre - height // 2
    cv2.rectangle(frame, (x0, y0), (x_right - 1, y0 + height - 1), (background,) * 3, -1)
    x = x_right - pad - sum(widths) - gap * (len(glyphs) - 1)
    top = y_centre - line // 2
    for canvas, w in zip(glyphs, widths, strict=True):
        patch = cv2.resize(canvas, (w, line), interpolation=cv2.INTER_AREA)
        block = frame[top : top + line, x : x + w].astype(np.float32)
        blend = background + patch[:, :, None] * (ink - background)
        frame[top : top + line, x : x + w] = np.clip(
            np.where(patch[:, :, None] > 0.02, blend, block), 0, 255
        ).astype(np.uint8)
        x += w + gap
    centre = (x0 + pad + icon // 2, y_centre)
    radius = max(2, icon // 2 - 1)
    cv2.circle(frame, centre, radius, (ink,) * 3, 1)
    if icon_angle is None:
        cv2.line(
            frame,
            (centre[0] - radius // 2, centre[1]),
            (centre[0] + radius // 2, centre[1]),
            (ink,) * 3,
            1,
        )
    else:
        tip = (
            centre[0] + int(round(0.7 * radius * np.cos(icon_angle))),
            centre[1] + int(round(0.7 * radius * np.sin(icon_angle))),
        )
        cv2.line(frame, centre, tip, (ink,) * 3, 1)


def board_with_clocks(
    top: str,
    bottom: str,
    *,
    cell: int = 43,
    x0: int = 300,
    y0: int = 120,
    width: int = 1280,
    height: int = 720,
    top_light: bool = False,
    icon_angles: tuple[float | None, float | None] = (None, None),
    colours: tuple[tuple[int, int], tuple[int, int]] | None = None,
    page: tuple[int, int, int] = (46, 46, 46),
    seed: int = 3,
    templates: GlyphTemplates | None = None,
) -> tuple[np.ndarray, BoardCandidate]:
    """A board with a clock box above and below it, and the candidate a tracker would give.

    The strips the clocks sit in are painted the page colour first, as a site's page is
    behind them; a box on random noise would be a harder problem than the real one.
    """
    frame = synthetic_board(
        width=width, height=height, x0=x0, y0=y0, cell=cell, pieces=True, seed=seed
    )
    side = 8 * cell
    band = int(round(1.4 * cell))
    for y in (y0 - band, y0 + side):
        cv2.rectangle(
            frame,
            (max(0, x0 - cell), max(0, y)),
            (min(width - 1, x0 + side + cell), min(height - 1, y + band)),
            page,
            -1,
        )
    templates = templates or GlyphTemplates.load()
    draw_clock_box(
        frame,
        x0 + side,
        y0 - int(round(0.37 * cell)),
        cell,
        top,
        light=top_light,
        icon_angle=icon_angles[0],
        colours=colours[0] if colours else None,
        templates=templates,
    )
    draw_clock_box(
        frame,
        x0 + side,
        y0 + side + int(round(0.37 * cell)),
        cell,
        bottom,
        light=not top_light,
        icon_angle=icon_angles[1],
        colours=colours[1] if colours else None,
        templates=templates,
    )
    board = BoardCandidate(
        region=Region(x0, y0, side, side), cell_px=float(cell), score=1.0, contrast=60.0, fit=1.0
    )
    return frame, board


def game_frames(
    sans: tuple[str, ...] = ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"),
    *,
    frames_per_state: int = 3,
    fps: float = 10.0,
    cell: int = 64,
    theme: str = "green",
    coordinates: bool = True,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """The frames of a short game as a capture loop would hand them over: (frame id,
    timestamp, frame). The start position sits for a few frames, then every move for a
    few, with the last-move highlight. What a whole-game recording replay stood in for."""
    board = chess.Board()
    frame_id = 0

    def emit(**kw: object) -> Iterator[tuple[int, float, np.ndarray]]:
        nonlocal frame_id
        for _ in range(frames_per_state):
            yield (
                frame_id,
                frame_id / fps,
                synthetic_board(
                    board=board,
                    cell=cell,
                    theme=theme,
                    coordinates=coordinates,
                    **kw,  # type: ignore[arg-type]
                ),
            )
            frame_id += 1

    yield from emit()
    for san in sans:
        move = board.push_san(san)
        yield from emit(highlight=move)


# ---- the board as the tests show it ---------------------------------------------------

# Where `synthetic_board` draws the board by default: at (300, 120), 64 px squares.
GEOM = BoardGeometry(Region(300, 120, 512, 512), 64.0)


def puzzle_frame(board: chess.Board, bottom: chess.Color, **kw: object) -> np.ndarray:
    """A puzzle as the site shows it: the solver's side at the bottom, the coordinates on."""
    return synthetic_board(
        board=board,
        bottom=Side.WHITE if bottom == chess.WHITE else Side.BLACK,
        coordinates=True,
        **kw,  # type: ignore[arg-type]
    )


def bulky_start(frame: np.ndarray, x0: int, y0: int, cell: int) -> None:
    """A start position drawn with bulky pieces (`bulky_pieces`), whose coarse checker score
    is well under an empty board's."""
    render_board(frame, x0, y0, cell, chess.Board())
    bulky_pieces(frame, x0, y0, cell, chess.Board())


# A game-over dialog over the default board: columns a to e, rows 2 to 6.
DIALOG_X0, DIALOG_Y0 = 300 - 40, 120 + 140
DIALOG_X1, DIALOG_Y1 = 300 + 330, 120 + 420


def _dialog_rect(bottom: Side) -> tuple[int, int, int, int]:
    """The dialog's rectangle, mirrored through the board's centre when black sits at the
    bottom so that it covers the same squares in either orientation."""
    if bottom == Side.WHITE:
        return DIALOG_X0, DIALOG_Y0, DIALOG_X1, DIALOG_Y1
    cx, cy = 300 + 256, 120 + 256
    return 2 * cx - DIALOG_X1, 2 * cy - DIALOG_Y1, 2 * cx - DIALOG_X0, 2 * cy - DIALOG_Y0


def with_dialog(
    frame: np.ndarray, dark: bool = False, alpha: float = 1.0, bottom: Side = Side.WHITE
) -> np.ndarray:
    """A game-over dialog with text covering the left half and middle of the default board
    (test_occlusion).

    `alpha` below 1 blends the dialog over the board, as chess.com does while it fades in.
    """
    if alpha < 1.0:
        full = with_dialog(frame, dark=dark, bottom=bottom)
        return cv2.addWeighted(full, alpha, frame, 1.0 - alpha, 0.0)
    fill = (50, 50, 50) if dark else (245, 245, 245)
    ink = (230, 230, 230) if dark else (40, 40, 40)
    width, height = DIALOG_X1 - DIALOG_X0 + 1, DIALOG_Y1 - DIALOG_Y0 + 1
    dialog = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(dialog, (0, 0), (width - 1, height - 1), fill, -1)
    cv2.rectangle(dialog, (0, 0), (width - 1, height - 1), (120, 120, 120), 2)
    for i in range(6):
        cv2.putText(
            dialog, "You won by checkmate", (15, 40 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ink, 2
        )
    if bottom == Side.BLACK:
        # The same pixels over the same squares: the dialog turned with the board.
        dialog = cv2.rotate(dialog, cv2.ROTATE_180)
    out = frame.copy()
    x0, y0, x1, y1 = _dialog_rect(bottom)
    out[y0 : y1 + 1, x0 : x1 + 1] = dialog[: out.shape[0] - y0, : out.shape[1] - x0]
    return out
