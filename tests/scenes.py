"""Stand-ins for the crops of real games the vision tests once read from tests/fixtures.

Each scene draws the situation one crop showed -- the same position, the same square size
and place on the frame, the same marks over the board, the same page around it -- with
lichess's open board colours and the cburnett pieces (lichess's default set; GPLv2+, read
from python-chess by tests/pieceset.py), through the renderer of tests/synth.py. The public
repo carries no site's artwork (Tai, 2026-10-13); the crops themselves stay on the
developer's disk, under data/fixtures/real/, and `tools/vision/real_crops.py` runs the old
real-pixel checks on them there.

- `cluttered_page`: a whole site page around a board of 86 px squares at (290, 65) -- a
  sidebar, an evaluation bar beside the board, avatars, captured pieces, a chat panel, a
  striped move list, buttons -- for the board detector.
- `arrow_scenes`: the trial game of 2026-09-16 on the brown board, cropped to the board with
  a 24 px margin at 80 px squares: its start, the clean frame after 22...e5, and the frames
  with two and three arrows over it (the heads on the black pawn on e5, red-marked).
- `joined_scenes`: the recorded game g01 at 43 px squares: its start, a settled middlegame
  frame, and the same frame with the c1 bishop in flight over f4.
- `puzzle_black_84`: a puzzle on the phone, Black at the bottom, 84 px squares filling the
  frame.
- `badge_crops`: the result badges of two games' ends, at 80.1176 and 43.0233 px squares (a
  page zoom: drawn at whole pixels and scaled), cropped round the kings as the live crops
  were.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import chess
import cv2
import numpy as np

from sidecoach.events import Side
from tests.pieceset import draw_sprite
from tests.synth import ARROW, PIECE_STYLES, draw_badge, render_board

# lichess's board colours, (light, dark) in BGR: brown #f0d9b5/#b58863, green
# #ffffdd/#86a666, blue #dee3e6/#8ca2ad.
LICHESS_BROWN = ((181, 217, 240), (99, 136, 181))
LICHESS_GREEN = ((221, 255, 255), (102, 166, 134))
LICHESS_BLUE = ((230, 227, 222), (173, 162, 140))
# lichess's last-move highlight, rgb(155, 199, 0), in BGR. The arrows keep the orange
# measured on the trial game (`synth.ARROW`) over its red-marked square: lichess's own
# darker orange (#e68f00) lies close to that red mark's tint, a pairing neither site draws.
LAST_MOVE = (0, 199, 155)
# A dark site page: the background, a panel on it, its text and its quieter text.
PAGE = (33, 36, 38)
PANEL = (44, 46, 48)
ROW = (38, 40, 42)
TEXT = (235, 235, 235)
MUTED = (150, 150, 150)
CBURNETT = PIECE_STYLES["cburnett"]

# The page's board, where the detector must find it.
PAGE_X0, PAGE_Y0, PAGE_CELL = 290, 65, 86
PAGE_W, PAGE_H = 1520, 820
PAGE_POSITION = "Q7/2N3k1/1K6/4P3/8/8/8/8 b - - 0 58"

# The trial game after 22...e5: the d4 pawn faces the e5 pawn.
ARROW_CELL, MARGIN = 80, 24
ARROW_POSITION = "3rr3/4k3/p2q1n1p/1p2p1p1/3PP3/5P2/PP3BPP/2RQ1RK1 w - - 0 23"
ARROW_LAST = chess.Move.from_uci("e6e5")

# g01's middlegame as its tracker had it, White to move, the last move (h7 to g6) lit: the
# c1 bishop is the piece lifted.
JOINED_CELL = 43
JOINED_PLACEMENT = "rnbqkb1r/pppp4/4p1p1/5n2/3P4/2N2N2/PPP2PPP/R1BQKB1R"
JOINED_LAST = chess.Move.from_uci("h7g6")

# The phone's puzzle, Black to move and at the bottom, after the bishop's d3-b1.
PUZZLE_CELL = 84
PUZZLE_POSITION = "7k/4PK1P/1p6/8/b1Pp4/p7/8/1B6 b - - 0 1"
PUZZLE_LAST = chess.Move.from_uci("d3b1")


def _text(
    frame: np.ndarray,
    words: str,
    x: int,
    y: int,
    scale: float = 0.5,
    colour: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame, words, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA
    )


def _rounded(
    frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, colour: tuple[int, int, int], r: int
) -> None:
    """A filled rectangle with rounded corners."""
    cv2.rectangle(frame, (x0 + r, y0), (x1 - r, y1), colour, -1)
    cv2.rectangle(frame, (x0, y0 + r), (x1, y1 - r), colour, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(frame, (cx, cy), r, colour, -1, lineType=cv2.LINE_AA)


def _avatar(frame: np.ndarray, x: int, y: int, size: int) -> None:
    """A player's picture: a face on a coloured square."""
    cv2.rectangle(frame, (x, y), (x + size, y + size), (120, 110, 100), -1)
    cx, cy = x + size // 2, y + size // 2
    cv2.circle(frame, (cx, cy + size // 3), size // 3, (60, 90, 160), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (cx, cy - size // 10), size // 5, (150, 175, 215), -1, lineType=cv2.LINE_AA)


def _captured(frame: np.ndarray, x: int, y: int, symbols: str, size: int = 18) -> int:
    """Captured pieces in a row, overlapping as the sites draw them; the x after them."""
    for symbol in symbols:
        if symbol == " ":
            x += size // 2
            continue
        draw_sprite(frame, x, y, size, symbol)
        x += size * 2 // 3
    return x + size // 2


def _flag(frame: np.ndarray, x: int, y: int) -> None:
    """A country's flag beside a name: red and white stripes."""
    cv2.rectangle(frame, (x, y), (x + 15, y + 10), (255, 255, 255), -1)
    cv2.rectangle(frame, (x, y + 5), (x + 15, y + 10), (40, 40, 220), -1)


def _clock(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, words: str) -> None:
    """A clock box: light, with the time in dark figures."""
    _rounded(frame, x0, y0, x1, y1, (232, 232, 232), 3)
    _text(frame, words, x0 + 10, y1 - 6, 0.6, (40, 40, 40), 2)


@cache
def cluttered_page() -> np.ndarray:
    """A site's page with a bot game near its end, as a 1440p screen cropped to the page shows
    it: the board of 86 px squares at (290, 65) with its coordinates, the last move
    highlighted and a classification badge on the king's square; a white evaluation bar
    beside it; the players' pictures, names and captured pieces above and below; a sidebar of
    menu entries; a chat panel with an avatar and two white bubbles; a move list in striped
    rows with piece glyphs; buttons. Read-only: copy it to draw on it."""
    frame = np.full((PAGE_H, PAGE_W, 3), PAGE, dtype=np.uint8)
    # The sidebar: a logo, the menu, search and the user at the foot.
    frame[:, :180] = (27, 29, 31)
    draw_sprite(frame, 26, 14, 32, "N")
    _text(frame, "Boards", 62, 40, 0.75, TEXT, 2)
    for i, (word, colour) in enumerate(
        (("Play", (60, 140, 230)), ("Puzzles", (40, 110, 240)), ("Learn", (200, 140, 40)))
    ):
        y = 80 + 40 * i
        cv2.circle(frame, (39, y), 11, colour, -1, lineType=cv2.LINE_AA)
        _text(frame, word, 62, y + 6, 0.55, TEXT, 1)
    _text(frame, "v   More", 32, 204, 0.5, MUTED, 1)
    cv2.circle(frame, (37, 709), 8, MUTED, 2, lineType=cv2.LINE_AA)
    _text(frame, "Search", 62, 715, 0.5, MUTED, 1)
    _avatar(frame, 26, 736, 24)
    _text(frame, "player", 62, 754, 0.5, TEXT, 1)
    for i in range(4):
        cv2.circle(frame, (39 + 38 * i, 792), 7, (170, 170, 170), -1, lineType=cv2.LINE_AA)
    # The evaluation bar beside the board, white: the side to move mates.
    cv2.rectangle(frame, (250, PAGE_Y0), (280, PAGE_Y0 + 8 * PAGE_CELL - 1), (240, 240, 240), -1)
    _text(frame, "M4", 256, 745, 0.35, (60, 60, 60), 1)
    # The board.
    board = chess.Board(PAGE_POSITION)
    render_board(
        frame,
        PAGE_X0,
        PAGE_Y0,
        PAGE_CELL,
        board,
        Side.WHITE,
        light=LICHESS_GREEN[0],
        dark=LICHESS_GREEN[1],
        highlight=chess.Move.from_uci("f8g7"),
        highlight_colour=LAST_MOVE,
        coordinates=True,
        style=CBURNETT,
    )
    draw_badge(frame, PAGE_X0, PAGE_Y0, PAGE_CELL, Side.WHITE, chess.G7, "best")
    # The players: the bot above, the user below -- a picture, a bold name, a flag, and a
    # row of what they have taken, as dense in edges as the site's bars.
    _avatar(frame, 290, 15, 40)
    _text(frame, "Coach (2000)", 340, 31, 0.55, TEXT, 2)
    _flag(frame, 468, 19)
    _captured(frame, 340, 34, "PPPPPPP BB N RR", 20)
    cv2.circle(frame, (993, 25), 7, MUTED, 2, lineType=cv2.LINE_AA)
    cv2.rectangle(frame, (290, 763), (330, 803), (225, 225, 225), -1)
    draw_sprite(frame, 296, 767, 30, "P")
    _text(frame, "player", 360, 778, 0.55, TEXT, 2)
    _flag(frame, 428, 766)
    x = _captured(frame, 340, 783, "ppppppp bb nn rr q", 20)
    _text(frame, "+13", x, 798, 0.45, MUTED, 1)
    # The panel: a header, the coach's picture and two bubbles, the opening's name, the
    # moves in striped rows, three buttons and a row of icons.
    _rounded(frame, 1010, 15, 1510, 805, PANEL, 6)
    _text(frame, "Play Coach", 1213, 45, 0.65, TEXT, 2)
    _avatar(frame, 1023, 92, 90)
    for top, words in ((120, "Time for my king to move."), (192, "Do you see the mate?")):
        _rounded(frame, 1115, top, 1493, top + 62, (255, 255, 255), 10)
        _text(frame, words, 1130, top + 37, 0.5, (40, 40, 40), 1)
    cv2.line(frame, (1010, 296), (1510, 296), (60, 62, 64), 1)
    _text(frame, "French Defense: Normal Variation", 1026, 320, 0.45, MUTED, 1)
    cv2.line(frame, (1010, 334), (1510, 334), (60, 62, 64), 1)
    moves = [
        ("Nd5", "Kd7"), ("Kd3", "Kc6"), ("Kc4", "Kd6"), ("a5", "Kd7"), ("Kxc5", "Kc8"),
        ("Kb6", "Kb8"), ("e5", "Kc8"), ("Nc7", "Kd8"), ("a6", "Ke7"), ("a7", "Kf8"),
        ("a8=Q+", "Kg7"),
    ]  # fmt: skip
    for i, (white, black) in enumerate(moves):
        y = 345 + 30 * i
        frame[y : y + 30, 1012:1500] = ROW if i % 2 == 0 else PANEL
        _text(frame, f"{48 + i}.", 1025, y + 21, 0.45, MUTED, 1)
        for x, san in ((1075, white), (1173, black)):
            if san[0] in "KQRBN":
                draw_sprite(frame, x - 2, y + 6, 18, san[0])
                san = san[1:]
                x += 17
            _text(frame, san, x, y + 21, 0.5, TEXT, 1)
    _rounded(frame, 1168, 648, 1214, 673, (85, 85, 85), 3)
    _text(frame, "g7", 1190, 666, 0.5, TEXT, 1)
    for left in (1022, 1183, 1345):
        _rounded(frame, left, 700, left + 152, 746, (64, 66, 70), 5)
    flag = np.array([[1092, 712], [1110, 716], [1092, 724]], dtype=np.int32)
    cv2.fillPoly(frame, [flag], (220, 220, 220), lineType=cv2.LINE_AA)
    cv2.line(frame, (1092, 712), (1092, 734), (220, 220, 220), 2, lineType=cv2.LINE_AA)
    cv2.circle(frame, (1259, 720), 8, (220, 220, 220), -1, lineType=cv2.LINE_AA)
    cv2.arrowedLine(frame, (1430, 728), (1410, 718), (220, 220, 220), 3, tipLength=0.5)
    for i in range(3):
        cv2.circle(frame, (1038 + 32 * i, 776), 7, MUTED, 2, lineType=cv2.LINE_AA)
    frame.flags.writeable = False
    return frame


def _board_frame(
    board: chess.Board,
    cell: int,
    colours: tuple[tuple[int, int, int], tuple[int, int, int]],
    *,
    margin: int = MARGIN,
    bottom: Side = Side.WHITE,
    clocks: tuple[str, str] = ("15:53", "14:46"),
    **render: object,
) -> np.ndarray:
    """A board as a recording cropped to it shows it: the page's dark margin round the board,
    the players' pictures, names and clocks cut by the crop above and below, and the board
    with its coordinates and the cburnett pieces."""
    side = 8 * cell
    size = side + 2 * margin
    frame = np.full((size, size, 3), PAGE, dtype=np.uint8)
    for y_box, clock in ((-8, clocks[0]), (margin + side + 4, clocks[1])):
        _avatar(frame, margin, y_box, margin + 4)
        _text(frame, "player", margin + margin + 12, y_box + margin - 4, 0.4, TEXT, 1)
        _clock(frame, margin + side - 120, y_box, margin + side, y_box + margin + 4, clock)
    render_board(
        frame,
        margin,
        margin,
        cell,
        board,
        bottom,
        light=colours[0],
        dark=colours[1],
        highlight_colour=LAST_MOVE,
        coordinates=True,
        style=CBURNETT,
        **render,  # type: ignore[arg-type]
    )
    frame.flags.writeable = False
    return frame


@cache
def arrow_scenes() -> dict[str, np.ndarray]:
    """The trial game on the brown board at 80 px squares, cropped with a 24 px margin: its
    start ("start"), the clean frame after 22...e5 ("clean"), then the e5 pawn marked red with
    arrows from d6 and d4 whose heads land on it ("two_arrows"), and a third from f2 to c5
    across d4 ("three_arrows"). Read-only."""
    position = chess.Board(ARROW_POSITION)
    two = (chess.Move.from_uci("d6e5"), chess.Move.from_uci("d4e5"))
    marked = {
        "board": position,
        "highlight": ARROW_LAST,
        "red_marks": (chess.E5,),
        "arrow_colour": ARROW,
        "clocks": ("16:15", "0:24"),
    }
    return {
        "start": _board_frame(chess.Board(), ARROW_CELL, LICHESS_BROWN, clocks=("15:00", "14:55")),
        "clean": _board_frame(
            position, ARROW_CELL, LICHESS_BROWN, highlight=ARROW_LAST, clocks=("16:15", "0:14")
        ),
        "two_arrows": _scene(marked, arrows=two),
        "three_arrows": _scene(marked, arrows=(*two, chess.Move.from_uci("f2c5"))),
    }


def _scene(marked: dict, **more: object) -> np.ndarray:
    kw = {**marked, **more}
    return _board_frame(kw.pop("board"), ARROW_CELL, LICHESS_BROWN, **kw)  # type: ignore[arg-type]


@cache
def joined_scenes() -> dict[str, np.ndarray]:
    """g01 on the green board at 43 px squares, cropped with a 24 px margin: its start
    ("start"); the middlegame, quiet, the last move highlighted ("settled");
    and the same board with the c1 bishop picked up -- its square lit, the dots on its four
    empty destinations, the piece over f4 a little off its centre under the hover outline
    ("in_flight"). Read-only."""
    position = chess.Board(JOINED_PLACEMENT + " w - - 0 10")
    return {
        "start": _board_frame(chess.Board(), JOINED_CELL, LICHESS_GREEN, clocks=("15:00", "14:36")),
        "settled": _board_frame(position, JOINED_CELL, LICHESS_GREEN, highlight=JOINED_LAST),
        "in_flight": _board_frame(
            position,
            JOINED_CELL,
            LICHESS_GREEN,
            highlight=JOINED_LAST,
            selected=chess.C1,
            dragging=(chess.C1, chess.F4, 0.08, -0.12),
            clocks=("15:53", "14:48"),
        ),
    }


@cache
def puzzle_black_84() -> np.ndarray:
    """The phone's puzzle: the blue board of 84 px squares filling a 672 px frame, Black at
    the bottom (rank 1 at the top, the 8 at the foot of the left edge), the bishop's last
    move highlighted. Read-only."""
    frame = np.zeros((8 * PUZZLE_CELL, 8 * PUZZLE_CELL, 3), dtype=np.uint8)
    render_board(
        frame,
        0,
        0,
        PUZZLE_CELL,
        chess.Board(PUZZLE_POSITION),
        Side.BLACK,
        light=LICHESS_BLUE[0],
        dark=LICHESS_BLUE[1],
        highlight=PUZZLE_LAST,
        highlight_colour=LAST_MOVE,
        coordinates=True,
        style=CBURNETT,
    )
    frame.flags.writeable = False
    return frame


@dataclass(frozen=True)
class BadgeCrop:
    """A crop round a king's square: the square's top-left corner at (x0, y0) in the crop,
    its side `cell`, and the badge on it ("win", "loss" or "none")."""

    name: str
    kind: str
    cell: float
    x0: float
    y0: float
    crop: np.ndarray


# The two games' ends: the first won by White at 80.1176 px squares on the brown board (the
# kings on g1 and h5), the second lost by White at 43.0233 px on the green (g1 and g8). Each
# is seen the moment before the end ("none") and with the site's badges ("win", "loss").
BADGE_GAMES = (
    (80, 80.1176, LICHESS_BROWN, "8/8/6p1/6Pk/7Q/7P/5PP1/6K1 b - - 0 50", "g7g6", "f4h4", "1-0"),
    (43, 43.0233, LICHESS_GREEN, "6k1/4rnp1/6Q1/p1p4P/2P2R2/6P1/1r4q1/5RK1 w - - 3 45", "h4h5",
     "g4g2", "0-1"),
)  # fmt: skip
BADGE_CROPS = (
    "win_g1_80px", "loss_h5_80px", "none_g1_80px", "none_h5_80px",
    "loss_g1_43px", "win_g8_43px", "none_g1_43px", "none_g8_43px",
)  # fmt: skip


@cache
def badge_crops() -> dict[str, BadgeCrop]:
    """The eight crops of `BADGE_CROPS`: each king of each game, before and at its end, two
    squares across with the king's square in the middle -- the page beside or below the board
    included where the king stands on its edge. The frames are drawn at whole-pixel squares
    with a wide margin and scaled to the games' squares, as a zoomed page draws them."""
    out: dict[str, BadgeCrop] = {}
    for drawn, cell, colours, fen, before, last, result in BADGE_GAMES:
        board = chess.Board(fen)
        margin = 2 * drawn
        for when, highlight, shown in (("none", before, None), ("end", last, result)):
            frame = _board_frame(
                board,
                drawn,
                colours,
                margin=margin,
                highlight=chess.Move.from_uci(highlight),
                result=shown,
                clocks=("0:31", "0:09"),
            )
            scale = cell / drawn
            size = int(round(frame.shape[0] * scale))
            scaled = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
            winner = {"1-0": chess.WHITE, "0-1": chess.BLACK}[result]
            for colour in (chess.WHITE, chess.BLACK):
                king = board.king(colour)
                assert king is not None
                kind = "none" if when == "none" else ("win" if colour == winner else "loss")
                name = f"{kind}_{chess.square_name(king)}_{drawn}px"
                col, row = chess.square_file(king), 7 - chess.square_rank(king)
                left, top = (margin + col * drawn) * scale, (margin + row * drawn) * scale
                x, y = math.floor(left - cell / 2), math.floor(top - cell / 2)
                span = int(math.ceil(2 * cell))
                crop = scaled[y : y + span, x : x + span].copy()
                crop.flags.writeable = False
                out[name] = BadgeCrop(name, kind, cell, left - x, top - y, crop)
    assert sorted(out) == sorted(BADGE_CROPS), sorted(out)
    return out
