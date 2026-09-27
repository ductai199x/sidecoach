"""Record where the desktop's recogniser says each piece sits: the phase landing, cell for cell.

Desktop de58ca7 decides whether a piece has landed by the phase correlation of the cell's grey
crop against that piece kind's start crop (`pieces.LANDING`, "phase", `_phase_shift`). The phone
does the same with its own OpenCV, a different build of it; this holds its numbers to the
Python's. A model is learned from a start frame, as the pipeline learns one, and frames are
read one after another, as a game is: pieces at rest on the phone-sized boards, and pieces
caught mid-slide with a trail on the slide scenarios' frames. For every cell the desktop's
`classify_frame` gives its kind, symbol, settled flag and offset -- the phase shift, a share of
the cell, for a piece the net named.

The frames are the ones already in the assets (the boards fixtures and the pipeline fixtures'
slide frames), so nothing new is committed but this file's numbers.

    cd D:/chess-coaching-companion
    uv run --no-sync python <checkout>/tools/make_landing_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import cv2  # noqa: E402

from sidecoach.events import Region, Side  # noqa: E402
from sidecoach.vision import pieces  # noqa: E402
from sidecoach.vision.pieces import ThemeModel  # noqa: E402
from sidecoach.vision.squares import BoardGeometry  # noqa: E402

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


def main() -> None:
    assert pieces.LANDING == "phase", "the desktop's own landing rule"
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
