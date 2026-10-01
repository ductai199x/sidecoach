"""Render the boards the Android vision tests read, using the desktop's own renderer.

    cd D:\\chess-coaching-companion
    uv run --no-sync python <worktree>/tools/make_fixtures.py

The desktop repo is the source of the renderer (tests/synth.py: real chess.com and
lichess themes, the sprite sheets of 35 piece sets) and of the ground truth, so the
Kotlin port is checked against the same boards the Python is. Each fixture is a PNG
plus a JSON line saying where the board is and what stands on it; both go into the
instrumented tests' assets (next to this file: <worktree>/app/src/androidTest/assets/boards).

The sizes are a phone's, not a desktop's: the capture service mirrors the screen at half
its width, so a 1344 x 2992 Pixel becomes 672 x 1496, and a board that fills the width is
about 664 px across -- a cell of 83.

A board with `classified_from` is also read by the desktop's recogniser, learned from that
start board the way the pipeline learns one (its orientation read from the start frame):
`symbols` and `kinds` are what `classify_frame` named in each cell, in screen order, for
ThemeStoreTest to hold the phone to (LandingTest reads the same pairs from landing.json). The k-means palette is seeded
(`cv2.setRNGSeed`) so the file comes out the same every time.
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

from sidecoach.events import Region, Side  # noqa: E402
from sidecoach.vision.pieces import (  # noqa: E402
    ThemeModel,
    looks_like_start_position,
    occupancy,
    orientation_from_start,
)
from sidecoach.vision.squares import BoardGeometry  # noqa: E402
from tests.synth import synthetic_board  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "app" / "src" / "androidTest" / "assets" / "boards"
SEED = 20260926

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
        {"name": "start-green", "board": start, "theme": "green", "style": "cburnett",
         "classified_from": "start-green"},
        {"name": "start-brown", "board": start, "theme": "brown", "style": "cburnett",
         "classified_from": "start-brown"},
        {"name": "italian-green", "board": italian, "theme": "green", "style": "cburnett",
         "classified_from": "start-green"},
        {"name": "italian-flipped", "board": italian, "theme": "green", "style": "cburnett",
         "bottom": Side.BLACK, "classified_from": "start-green"},
        {"name": "endgame-blue", "board": endgame, "theme": "blue", "style": "cburnett"},
        {"name": "start-merida", "board": start, "theme": "green", "style": "merida",
         "classified_from": "start-merida"},
        {"name": "italian-alpha", "board": italian, "theme": "brown", "style": "alpha"},
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cell = 83
    x0 = (PHONE_W - 8 * cell) // 2
    y0 = 420  # a board sitting under the opponent's clock, as the apps place it
    geometry = BoardGeometry(Region(x0, y0, 8 * cell, 8 * cell), float(cell))
    index = []
    frames = {}
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
        frames[spec["name"]] = frame
        path = OUT / f"{spec['name']}.png"
        cv2.imwrite(str(path), frame)
        # The Python's own numbers for this frame, so the Kotlin port is held to them
        # rather than to my reading of the code: the foreground fraction of every cell,
        # whether that occupancy has the shape of a start position, and the orientation.
        fractions = occupancy(frame, geometry)
        entry = {
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
        start = spec.get("classified_from")
        if start is not None:
            learned_from = frames[start]
            cv2.setRNGSeed(SEED)
            model = ThemeModel.from_start_position(
                learned_from, geometry, orientation_from_start(learned_from, geometry)
            )
            assert model is not None, f"{spec['name']}: no model from {start}"
            readings = model.classify_frame(frame, geometry)
            entry["classified_from"] = start
            entry["kinds"] = [r.kind for row in readings for r in row]
            entry["symbols"] = [r.symbol or "" for row in readings for r in row]
        index.append(entry)
        print(f"{path.name}: board at ({x0},{y0}) {8 * cell}px, cell {cell}")
    (OUT / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(index)} fixtures to {OUT}")


if __name__ == "__main__":
    main()
