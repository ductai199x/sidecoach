"""Record where the desktop's recogniser says each piece sits: the phase landing, cell for cell.

Desktop de58ca7 decides whether a piece has landed by the phase correlation of the cell's grey
crop against that piece kind's start crop (`pieces._phase_shift`). The phone
does the same with its own OpenCV, a different build of it; this holds its numbers to the
Python's. A model is learned from a start frame, as the pipeline learns one, and frames are
read one after another, as a game is: pieces at rest on the phone-sized boards, and pieces
caught mid-slide with a trail on the slide scenarios' frames. For every cell the desktop's
`classify_frame` gives its kind, symbol, settled flag and offset -- the phase shift, a share of
the cell, for a piece the net named.

The frames are the ones already in the assets (the boards fixtures and the pipeline fixtures'
slide frames), so nothing new is committed for them but this file's numbers. Beside them, the
desktop's own piece tests of its cfa8544 vision work (aac94ac, cfaa416), on their pages
(`boards/resized-*.png`): a set learned at one place and cell size and read at another, each
geometry the desktop's detector's (see `resized_cases`).

    cd D:/chess-coaching-companion
    uv run --no-sync python <checkout>/tools/make_landing_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from sidecoach.events import Region, Side  # noqa: E402
from sidecoach.vision.board_detect import detect_boards  # noqa: E402
from sidecoach.vision.pieces import ThemeModel  # noqa: E402
from sidecoach.vision.squares import BoardGeometry, row_col  # noqa: E402
from tests.synth import synthetic_board  # noqa: E402

# Next to this file, so a worktree reads and writes its own assets.
ASSETS = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets"
OUT = ASSETS / "boards" / "landing.json"
# The slide scenarios' board: `make_pipeline_fixtures.SLIDE_BOARD`.
SLIDE = (300, 120, 64)


def read(path: str):  # noqa: ANN201
    frame = cv2.imread(str(ASSETS / path), cv2.IMREAD_COLOR)
    assert frame is not None, path
    return frame


def case(name: str, start: str, frames: list[str], x: int, y: int, cell: int) -> dict:
    geometry = BoardGeometry(Region(x, y, 8 * cell, 8 * cell), float(cell))
    model = ThemeModel.from_start_position(read(start), geometry, Side.WHITE)
    assert model is not None, f"{name}: no model from {start}"
    assert model.landing_crops, f"{name}: no landing crops"
    out = []
    for path in frames:
        readings = model.classify_frame(read(path), geometry)
        out.append(
            {
                "file": path,
                "cells": [
                    [r.kind, r.symbol, r.settled, r.offset]
                    for row in readings
                    for r in row
                ],
            }
        )
    return {"name": name, "start": start, "x": x, "y": y, "cell": cell, "frames": out}


# ---- the desktop's piece tests of its cfa8544 vision work --------------------------------------
#
# Desktop aac94ac and cfaa416 test the recogniser on a set learned at one place and cell size and
# read at another, each locked by the detector as the pipeline locks them (tests/test_pieces.py):
# a king or a queen on the square colour the start position never showed it on, pieces at rest
# on a board resized since its set was learned, a pawn in flight on one, and a rook on its
# labelled corner at the learned size. The pages are theirs but for the background more than
# RESIZED_PAD cells from the board, flattened, and each case passes the desktop's assertion on
# them here or the generator stops. The phone learns and reads at the geometries recorded.

RESIZED_PAD = 1.0
FLAT = (32, 32, 32)


def flatten_around(frame: np.ndarray, x0: int, y0: int, cell: int) -> np.ndarray:
    pad = int(round(RESIZED_PAD * cell))
    keep = frame[max(0, y0 - pad) : y0 + 8 * cell + pad, max(0, x0 - pad) : x0 + 8 * cell + pad].copy()
    out = np.full_like(frame, FLAT)
    out[max(0, y0 - pad) : y0 + 8 * cell + pad, max(0, x0 - pad) : x0 + 8 * cell + pad] = keep
    return out


def write(name: str, frame: np.ndarray) -> str:
    path = f"boards/{name}.png"
    cv2.imwrite(str(ASSETS / path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return path


def resized_case(
    name: str,
    square: str,
    learned: tuple[int, int, int],
    read: tuple[int, int, int],
    sans: tuple[str, ...],
    size: dict,
    bottom: Side,
    want,  # noqa: ANN001  the desktop's assertion on the reading
    **decorations: object,
) -> dict:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    kw = dict(size, bottom=bottom)
    start = flatten_around(synthetic_board(x0=learned[0], y0=learned[1], cell=learned[2], **kw), *learned)
    frame = flatten_around(
        synthetic_board(x0=read[0], y0=read[1], cell=read[2], board=board, **decorations, **kw), *read
    )
    first, now = detect_boards(start)[0], detect_boards(frame)[0]
    model = ThemeModel.from_start_position(start, BoardGeometry(first.region, first.cell_px), bottom)
    assert model is not None, f"{name}: no model"
    readings = model.classify_frame(frame, BoardGeometry(now.region, now.cell_px))
    r, c = row_col(chess.parse_square(square), bottom)
    piece = board.piece_at(chess.parse_square(square))
    if not want(readings[r][c], piece):
        raise SystemExit(f"{name}: the desktop read {square} as {readings[r][c]}")
    return {
        "name": name,
        "start": write(f"resized-{name}-start", start),
        "start_region": [first.region.x, first.region.y, first.region.w, first.region.h],
        "start_cell": first.cell_px,
        "white_at_bottom": bottom == Side.WHITE,
        "square": [r, c],
        "frames": [
            {
                "file": write(f"resized-{name}", frame),
                "region": [now.region.x, now.region.y, now.region.w, now.region.h],
                "cell": now.cell_px,
                "cells": [[x.kind, x.symbol, x.settled, x.offset] for row in readings for x in row],
            }
        ],
    }


def named(reading, piece) -> bool:  # noqa: ANN001
    return reading.kind == "piece" and reading.symbol == piece.symbol()


def landed(reading, piece) -> bool:  # noqa: ANN001
    return named(reading, piece) and reading.settled


def resized_cases() -> list[dict]:
    cases = []
    # test_a_king_or_queen_is_named_on_the_other_square_colour (desktop aac94ac)
    for name, style, theme, cell, bottom, sans, square in (
        ("king-dark", "wooden", "wood", 64, Side.BLACK, ("d3", "e6", "Nd2", "Ke7"), "e7"),
        (
            "queen-light", "flat", "dark", 96, Side.WHITE,
            ("h3", "g5", "a4", "b6", "d4", "Bg7", "Bxg5", "d5", "c4", "Bf6", "cxd5", "Qxd5", "a5"),
            "d5",
        ),
    ):  # fmt: skip
        size = {"width": 8 * cell + 600, "height": 8 * cell + 240, "theme": theme, "style": style,
                "coordinates": True}  # fmt: skip
        cases.append(
            resized_case(name, square, (300, 120, cell), (300, 120, cell), sans, size, bottom, named)
        )
    # test_a_piece_at_rest_lands_on_a_board_resized_since_its_set_was_learned (desktop cfaa416)
    page = {"width": 1920, "height": 1200, "coordinates": False}
    for name, style, theme, learned, read, sans, square, lit in (
        (
            "rook-40-34", "wooden", "brown", (1124, 191, 40), (1648, 911, 34),
            ("b3", "h6", "c3", "Nf6", "Qc2", "g5", "Qe4", "Rh7"), "h7", "h8h7",
        ),
        (
            "king-48-132", "classic", "ice", (559, 284, 48), (607, 44, 132),
            ("a4", "e5", "Ra2", "Ba3", "Ra1", "f5", "Nh3", "Ke7", "d4", "Kd6"), "d6", "e7d6",
        ),
    ):  # fmt: skip
        cases.append(
            resized_case(
                name, square, learned, read, sans, dict(page, theme=theme, style=style), Side.WHITE,
                landed, highlight=chess.Move.from_uci(lit),
            )
        )
    # test_a_piece_in_flight_on_a_resized_board_does_not_land_on_the_square_it_passes
    move = chess.Move.from_uci("b2b4")
    cases.append(
        resized_case(
            "pawn-in-flight-40-36", "b3", (1062, 299, 40), (600, 300, 36), ("h3", "f6"),
            dict(page, theme="dark", style="spatial"), Side.WHITE,
            lambda reading, _: not (reading.kind == "piece" and reading.settled),
            highlight=move, animating=(move.from_square, move.to_square, 0.75),
        )
    )  # fmt: skip
    # test_a_rook_on_a_labelled_corner_lands_at_the_size_its_set_was_learned_at
    sans = (
        "Nh3", "d6", "Ng5", "e6", "Nc3", "Ne7", "Na4", "Nd7",
        "d3", "Nb6", "Kd2", "h6", "Ke3", "Rh7", "Nf3", "a5",
    )  # fmt: skip
    cases.append(
        resized_case(
            "rook-corner-64", "h1", (300, 120, 64), (300, 120, 64), sans,
            dict(page, theme="green", style="flat", coordinates=True), Side.BLACK,
            lambda reading, _: reading.kind == "piece" and reading.symbol == "R" and reading.settled,
        )
    )  # fmt: skip
    return cases


def main() -> None:
    for old in (ASSETS / "boards").glob("resized-*.png"):
        old.unlink()
    cases = []
    boards = json.loads((ASSETS / "boards" / "index.json").read_text(encoding="utf-8"))
    by_name = {b["name"]: b for b in boards}
    for board in boards:
        start = board.get("classified_from")
        if not start or by_name[start]["bottom"] != "white":
            continue
        s = by_name[start]
        cases.append(
            case(
                f"boards {board['name']} from {start}",
                f"boards/{s['file']}",
                [f"boards/{s['file']}", f"boards/{board['file']}"],
                s["x"],
                s["y"],
                s["cell"],
            )
        )
    index = json.loads((ASSETS / "pipeline" / "index.json").read_text(encoding="utf-8"))
    for run in index["runs"]:
        if not run["name"].startswith("slide-"):
            continue
        files: list[str] = []
        for frame in run["frames"]:
            if not files or files[-1] != frame["file"]:
                files.append(frame["file"])
        paths = [f"pipeline/{f}" for f in files]
        cases.append(case(run["name"], paths[0], paths, *SLIDE))
    cases += resized_cases()
    OUT.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    pieces_read = sum(
        1 for c in cases for f in c["frames"] for cell in f["cells"] if cell[0] == "piece"
    )
    landing = sum(
        1
        for c in cases
        for f in c["frames"]
        for cell in f["cells"]
        if cell[0] == "piece" and not cell[2]
    )
    print(f"{len(cases)} cases, {pieces_read} piece readings, {landing} not settled -> {OUT}")


if __name__ == "__main__":
    main()
