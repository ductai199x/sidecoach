"""Render the boards the Android vision tests read, using the desktop's own renderer.

    cd D:\\chess-coaching-companion
    uv run --no-sync python D:/sidecoach-android/tools/make_fixtures.py

The desktop repo is the source of the renderer (tests/synth.py: real chess.com and
lichess themes, the sprite sheets of 35 piece sets) and of the ground truth, so the
Kotlin port is checked against the same boards the Python is. Each fixture is a PNG
plus a JSON line saying where the board is and what stands on it; both go into the
instrumented tests' assets.

The sizes are a phone's, not a desktop's: the capture service mirrors the screen at half
its width, so a 1344 x 2992 Pixel becomes 672 x 1496, and a board that fills the width is
about 664 px across -- a cell of 83.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402
import cv2  # noqa: E402

from sidecoach.events import Region  # noqa: E402

from sidecoach.events import Side  # noqa: E402
from sidecoach.vision.pieces import (  # noqa: E402
    occupancy,
    orientation_from_start,
    looks_like_start_position,
)
from sidecoach.vision.squares import BoardGeometry  # noqa: E402
from tests.synth import synthetic_board  # noqa: E402

OUT = Path("D:/sidecoach-android/app/src/androidTest/assets/boards")

# A phone's screen, mirrored at half size, with the board across the width the way a
# chess app draws it: a little inset, the clocks and the move list above and below.
PHONE_W, PHONE_H = 672, 1496


def games() -> list[dict]:
    start = chess.Board()
    italian = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"):
        italian.push_san(san)
    endgame = chess.Board("8/5pk1/6p1/7p/7P/6P1/5PK1/3r4 w - - 0 1")
    return [
        {"name": "start-green", "board": start, "theme": "green", "style": "cburnett"},
        {"name": "start-brown", "board": start, "theme": "brown", "style": "cburnett"},
        {"name": "italian-green", "board": italian, "theme": "green", "style": "cburnett"},
        {"name": "italian-flipped", "board": italian, "theme": "green", "style": "cburnett",
         "bottom": Side.BLACK},
        {"name": "endgame-blue", "board": endgame, "theme": "blue", "style": "cburnett"},
        {"name": "start-merida", "board": start, "theme": "green", "style": "merida"},
        {"name": "italian-alpha", "board": italian, "theme": "brown", "style": "alpha"},
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cell = 83
    x0 = (PHONE_W - 8 * cell) // 2
    y0 = 420  # a board sitting under the opponent's clock, as the apps place it
    index = []
    for spec in games():
        frame = synthetic_board(
            width=PHONE_W,
            height=PHONE_H,
            x0=x0,
            y0=y0,
            cell=cell,
            theme=spec["theme"],
            style=spec["style"],
            board=spec["board"],
            bottom=spec.get("bottom", Side.WHITE),
            coordinates=True,
            seed=7,
        )
        path = OUT / f"{spec['name']}.png"
        cv2.imwrite(str(path), frame)
        # The Python's own numbers for this frame, so the Kotlin port is held to them
        # rather than to my reading of the code: the foreground fraction of every cell,
        # whether that occupancy has the shape of a start position, and the orientation.
        geometry = BoardGeometry(Region(x0, y0, 8 * cell, 8 * cell), float(cell))
        fractions = occupancy(frame, geometry)
        index.append(
            {
                "name": spec["name"],
                "fractions": [round(float(v), 5) for v in fractions.ravel()],
                "start_shaped": bool(looks_like_start_position(frame, geometry)),
                "orientation": orientation_from_start(frame, geometry).value,
                "file": path.name,
                "x": x0,
                "y": y0,
                "cell": cell,
                "side": 8 * cell,
                "fen": spec["board"].fen(),
                "bottom": spec.get("bottom", Side.WHITE).value,
                "width": PHONE_W,
                "height": PHONE_H,
            }
        )
        print(f"{path.name}: board at ({x0},{y0}) {8 * cell}px, cell {cell}")
    (OUT / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(index)} fixtures to {OUT}")


if __name__ == "__main__":
    main()
