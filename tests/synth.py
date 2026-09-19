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

import chess
import cv2
import numpy as np

from sidecoach.events import Region, Side
from sidecoach.vision.board_detect import BoardCandidate
from sidecoach.vision.clock import GlyphTemplates
from sidecoach.vision.squares import row_col

LIGHT = (208, 236, 238)
DARK = (86, 150, 118)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
TINT_ALPHA = 0.5
# Square colours (light, dark) in BGR, as the sites draw them. Brown is chess.com's default
# (the trial games were played on it); green is chess.com's green and the default here.
THEMES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "green": (LIGHT, DARK),
    "brown": ((181, 217, 240), (99, 136, 181)),
    "blue": ((222, 227, 234), (166, 130, 93)),
    "grey": ((214, 214, 214), (140, 140, 140)),
}
# chess.com's arrow: orange, drawn translucent over squares and pieces alike.
ARROW = (36, 179, 252)
ARROW_ALPHA = 0.8
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


def draw_piece(frame: np.ndarray, cx: int, cy: int, cell: int, symbol: str) -> None:
    white = symbol.isupper()
    fill = (245, 245, 245) if white else (35, 35, 35)
    edge = (0, 0, 0) if white else (210, 210, 210)
    kind = symbol.lower()
    radius = cell * 0.36
    thickness = max(2, int(round(cell / 32)))
    if kind == "p":
        cv2.circle(frame, (cx, cy), int(cell * 0.24), fill, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), int(cell * 0.24), edge, thickness, lineType=cv2.LINE_AA)
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
) -> None:
    """An arrow from one square's centre to another's, the way chess.com draws it: a shaft
    a fifth of a cell wide and a triangular head half a cell across, translucent, with
    anti-aliased edges, over whatever is on the squares -- the head lands on the piece."""
    r0, c0 = row_col(move.from_square, bottom)
    r1, c1 = row_col(move.to_square, bottom)
    start = np.array([x0 + c0 * cell + cell / 2, y0 + r0 * cell + cell / 2])
    end = np.array([x0 + c1 * cell + cell / 2, y0 + r1 * cell + cell / 2])
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1:
        return
    unit = direction / length
    normal = np.array([-unit[1], unit[0]])
    head_length, head_width, shaft_width = cell * 0.5, cell * 0.5, cell * 0.2
    base = end - unit * head_length
    shaft = np.array(
        [
            start + normal * shaft_width / 2,
            base + normal * shaft_width / 2,
            base - normal * shaft_width / 2,
            start - normal * shaft_width / 2,
        ]
    )
    head = np.array([base + normal * head_width / 2, end, base - normal * head_width / 2])
    overlay = frame.copy()
    for polygon in (shaft, head):
        cv2.fillPoly(overlay, [np.round(polygon).astype(np.int32)], colour, lineType=cv2.LINE_AA)
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
) -> None:
    """Render `board`. `dragging` = (origin, hovered square, dx, dy): the origin's piece is
    drawn over the hovered square displaced by (dx, dy) cells, under a hover outline.
    `arrows` are drawn over the pieces; `badge` is (square, label); `coordinates` draws
    the rank and file labels.

    `premoves` are queues the user has set up while the opponent thinks, drawn the way
    chess.com draws them: every square of a queue is tinted red, the piece is drawn at
    the end of its queue and its real square is left empty. A queue is a tuple of moves
    of one piece, each from the last one's destination; several queues are several
    pieces.
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
    tinted: list[tuple[chess.Square, tuple[int, int, int]]] = []
    if highlight is not None:
        tinted += [(highlight.from_square, YELLOW), (highlight.to_square, YELLOW)]
    tinted += [(square, RED) for square in red_marks]
    queued_from: dict[chess.Square, chess.Square] = {}  # the piece's real square -> queue end
    for queue in premoves:
        for move in queue:
            tinted += [(move.from_square, RED), (move.to_square, RED)]
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
    lifted = dragging[0] if dragging else None
    for square, piece in board.piece_map().items():
        if square == lifted:
            continue
        if square in queued_from:
            continue  # drawn at the end of its queue instead
        if square in queued_from.values():
            continue  # a queued capture: the premoved piece is drawn over it
        r, c = row_col(square, bottom)
        draw_piece(
            frame, x0 + c * cell + cell // 2, y0 + r * cell + cell // 2, cell, piece.symbol()
        )
    for origin, end in queued_from.items():
        piece = board.piece_at(origin)
        if piece is None:
            continue
        r, c = row_col(end, bottom)
        draw_piece(
            frame, x0 + c * cell + cell // 2, y0 + r * cell + cell // 2, cell, piece.symbol()
        )
    if coordinates:
        draw_coordinates(frame, x0, y0, cell, bottom, light, dark)
    for arrow in arrows:
        draw_arrow(frame, x0, y0, cell, bottom, arrow)
    if badge is not None:
        draw_badge(frame, x0, y0, cell, bottom, badge[0], badge[1])
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
    seed: int = 0,
) -> np.ndarray:
    """A board on a busy background. `pieces=False` renders an empty board; `theme` names
    the square colours (see THEMES) unless `light` and `dark` are given."""
    theme_light, theme_dark = THEMES[theme]
    light = theme_light if light is None else light
    dark = theme_dark if dark is None else dark
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
        premoves=premoves,
        arrows=arrows,
        badge=badge,
        coordinates=coordinates,
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
