"""Record what the desktop's recogniser makes of every cell, over every piece set it knows.

The parity audit of 2026-09-26 found the phone's recogniser off the desktop's in places no
fixture reached: the palette a drawing is told from (the phone clustered nothing: it used the
silhouettes' two tones, and read three cells of every anarcandy board as drawn), the grey the
Otsu split reads, the erosion under a piece's fill, half-precision exemplars, Python's
rounding. This holds the whole recogniser to the Python's answers at once, compactly: each of
the 44 sets of `tests/synth.py` (the 35 lichess sprite sheets and the synthetic ones), on the
green or the brown board in turn, learned from its start position and read on

- a middlegame dressed the way a game dresses it: the last move highlighted, the king in
  check with its glow, a piece dragged under the hover outline, a piece selected with its
  legal-move dots, and arrows over squares and pieces, in one of the four arrow colours (the
  colour turns with the set, so each colour meets eleven sets);
- for a few sets, the game-over dialog over the middle of the board, faded in or opaque.

For each model: the square colours, the set's fills and outlines, the palette (k-means,
seeded: `cv2.setRNGSeed(SEED)` just before the start position is learned, as the Kotlin
test seeds OpenCV's generator on its own thread), and each exemplar's brightness, mask and
tone areas, landing centroid and landing crop. For each cell: every field of the
`CellReading`, the silhouette's overlay share, and which side the fill under the true
piece's mask says (`_fill_side`). And what the rest of the vision makes of the same boards:
the theme cache's key for the square colours and the colours it samples from the dressed
board's corners (`theme_cache`), and the rank labels learned from the start position and
the side they read at the bottom of the dressed board (`coordinates`).

The boards are cut to the board and a few pixels round it, at the phone's cell (83 px: a
board across a 1344-pixel screen mirrored at half size).

    cd D:/chess-coaching-companion
    uv run --no-sync python <worktree>/tools/make_cell_fixtures.py

It writes next to itself: <worktree>/app/src/androidTest/assets/cells/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# The net on the CPU, as the phone runs it: the GPU's numbers are a hair further off.
os.environ["SIDECOACH_EMBED_DEVICE"] = "cpu"

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from sidecoach.events import Region, Side  # noqa: E402
from sidecoach.vision import pieces  # noqa: E402
from sidecoach.vision.coordinates import learn_rank_labels, read_bottom_side  # noqa: E402
from sidecoach.vision.theme_cache import palette_key, sample_palette  # noqa: E402
from sidecoach.vision.pieces import CELL, ThemeModel, silhouette  # noqa: E402
from sidecoach.vision.squares import BoardGeometry, square_at  # noqa: E402
from tests.synth import ARROWS, PIECE_STYLES, synthetic_board  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "app" / "src" / "androidTest" / "assets" / "cells"
SEED = 20260926
CELL_PX = 83
MARGIN = 4
THEMES = ("green", "brown")
COLOURS = tuple(ARROWS)  # orange, green, blue, red
# The sets the dialog is drawn over, faded in for the first two, opaque for the others.
DIALOG = {"anarcandy": 0.6, "cburnett": 0.6, "merida": 1.0, "pixel": 1.0}


def dressed() -> tuple[chess.Board, dict]:
    """The middlegame and what is drawn over it: 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.Ng5 d5
    5.exd5 Na5 6.Bb5+ -- Black in check, the bishop's move highlighted."""
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Ng5", "d5", "exd5", "Na5"):
        board.push_san(san)
    last = board.parse_san("Bb5+")
    board.push(last)
    return board, {
        "highlight": last,
        "check": True,
        # Black's queen dragged from d8 over d6 (empty), a little off the square's middle.
        "dragging": (chess.D8, chess.D6, 0.12, 0.06),
        "arrows": (chess.Move.from_uci("g5f7"), chess.Move.from_uci("a5c4"), chess.Move.from_uci("h2h4")),
    }


def dialog_board() -> chess.Board:
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O", "Nf6"):
        board.push_san(san)
    return board


def frame_of(style: str, theme: str, board: chess.Board, **drawn) -> np.ndarray:
    side = 8 * CELL_PX + 2 * MARGIN
    return synthetic_board(
        width=side,
        height=side,
        x0=MARGIN,
        y0=MARGIN,
        cell=CELL_PX,
        theme=theme,
        style=style,
        board=board,
        coordinates=True,
        seed=7,
        **drawn,
    )


def r6(value: float) -> float:
    return round(float(value), 6)


def colours_of(model: ThemeModel) -> list:
    return [
        [[r6(v) for v in fill], None if edge is None else [r6(v) for v in edge]]
        for fill, edge in model.piece_colours
    ]


def cells_of(
    model: ThemeModel, frame: np.ndarray, geometry: BoardGeometry, board: chess.Board
) -> list:
    readings = model.classify_frame(frame, geometry)
    out = []
    for r in range(8):
        for c in range(8):
            reading = readings[r][c]
            crop = geometry.crop(frame, r, c, size=2 * CELL)
            sil = silhouette(crop, colours=model.piece_colours, palette=model.palette)
            piece = board.piece_at(square_at(r, c, Side.WHITE))
            fill = None
            if piece is not None:
                side = model._fill_side(sil, piece.symbol())
                fill = [piece.symbol(), None if side is None else side.value]
            out.append(
                [
                    reading.kind,
                    reading.symbol,
                    None if reading.colour is None else reading.colour.value,
                    r6(reading.confidence),
                    r6(reading.fg_fraction),
                    r6(reading.luminance),
                    reading.highlighted,
                    [r6(v) for v in reading.background],
                    r6(reading.brightness),
                    reading.settled,
                    reading.hover,
                    r6(reading.offset),
                    reading.guess,
                    reading.covered,
                    reading.tint,
                    reading.drawn,
                    r6(sil.overlay),
                    fill,
                ]
            )
    return out


def write(name: str, frame: np.ndarray) -> str:
    cv2.imwrite(str(OUT / name), frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return name


def main() -> None:
    assert pieces.MATCHER == "embed" and pieces.embedder() is not None, "the desktop's matcher"
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    geometry = BoardGeometry(Region(MARGIN, MARGIN, 8 * CELL_PX, 8 * CELL_PX), float(CELL_PX))
    middlegame, drawn = dressed()
    cases = []
    for i, style in enumerate(sorted(PIECE_STYLES)):
        theme = THEMES[i % 2]
        colour = COLOURS[i % len(COLOURS)]
        start_board = chess.Board()
        start = frame_of(style, theme, start_board)
        cv2.setRNGSeed(SEED)
        model = ThemeModel.from_start_position(start, geometry, Side.WHITE)
        assert model is not None, f"{style}/{theme}: no model"
        write(f"{style}-start.png", start)
        dressed_frame = frame_of(style, theme, middlegame, arrow_colour=colour, **drawn)
        # The board as drawn: the dragged queen has left d8 and hangs over d6.
        shown = middlegame.copy(stack=False)
        queen = shown.remove_piece_at(chess.D8)
        shown.set_piece_at(chess.D6, queen)
        boards = [
            {
                "file": write(f"{style}-dressed.png", dressed_frame),
                "what": f"highlight, check, drag, {colour} arrows",
                "cells": cells_of(model, dressed_frame, geometry, shown),
            }
        ]
        if style in DIALOG:
            under = dialog_board()
            alpha = DIALOG[style]
            covered = frame_of(style, theme, under, dialog=alpha, selected=chess.F3)
            boards.append(
                {
                    "file": write(f"{style}-dialog.png", covered),
                    "what": f"dialog at {alpha}, knight selected",
                    "cells": cells_of(model, covered, geometry, under),
                }
            )
        labels = learn_rank_labels(start, geometry, Side.WHITE)
        bottom = read_bottom_side(dressed_frame, geometry, labels)
        sampled = sample_palette(dressed_frame, geometry)
        cases.append(
            {
                "style": style,
                "theme": theme,
                "start": f"{style}-start.png",
                "dressed": boards[0]["file"],
                "palette_key": palette_key(model.light_bg, model.dark_bg),
                "sampled": None if sampled is None else [[r6(v) for v in c] for c in sampled],
                "rank_labels": {str(rank): int(glyph.sum()) for rank, glyph in sorted(labels.items())},
                "bottom_read": None if bottom is None else bottom.value,
                "model": {
                    "light_bg": [r6(v) for v in model.light_bg],
                    "dark_bg": [r6(v) for v in model.dark_bg],
                    "piece_colours": colours_of(model),
                    "palette": [[r6(v) for v in colour] for colour in model.palette],
                    "exemplars": {
                        e.symbol: [
                            r6(e.brightness),
                            int(e.mask.sum()),
                            int(e.tone.sum()),
                            r6(e.landing_centroid[0]),
                            r6(e.landing_centroid[1]),
                            int(model.landing_crops[e.symbol].astype(np.int64).sum()),
                        ]
                        for e in model.exemplars
                    },
                },
                "boards": boards,
            }
        )
        drawn_cells = sum(1 for b in boards for cell in b["cells"] if cell[15])
        print(f"{style:15s} {theme:5s} {colour:6s} palette {len(model.palette):2d}, drawn {drawn_cells}")
    index = {
        "seed": SEED,
        "cell": CELL_PX,
        "margin": MARGIN,
        "fields": [
            "kind", "symbol", "colour", "confidence", "fg_fraction", "luminance", "highlighted",
            "background", "brightness", "settled", "hover", "offset", "guess", "covered", "tint",
            "drawn", "overlay", "fill_side",
        ],
        "cases": cases,
    }
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":")) + "\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"{len(cases)} sets, {sum(len(c['boards']) for c in cases)} boards, {size / 1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
