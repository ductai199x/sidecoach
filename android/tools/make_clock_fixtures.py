"""Record what the desktop reads off a chess.com clock, for the Kotlin to be held to.

`vision/clock.py` finds each clock box beside the board, cuts the digits out by column
projection, classifies them against templates learned from the recordings, and decides which
clock is running from the digits falling and the icon's hand sweeping. All of it is a pure
function of the pixels, so the frames are rendered here by the desktop's own synthetic
renderer (`tests/synth.py`, which draws the boxes with those same templates), written out as
PNGs, and every reading recorded beside them.

The templates themselves ship with the app: the desktop's `clock_templates.npz` is written
out as `app/src/main/assets/vision/clock_templates.json`.

    cd D:/chess-coaching-companion
    uv run --no-sync python <checkout>/tools/make_clock_fixtures.py

It writes next to itself, into that checkout's assets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

DESKTOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from sidecoach.events import Side  # noqa: E402
from sidecoach.state.pipeline import clock_observation  # noqa: E402
from sidecoach.vision.clock import (  # noqa: E402
    BOX_WIDTH_CELLS,
    SEARCH_CELLS,
    ClockReader,
    GlyphTemplates,
    parse_clock,
)
from tests.synth import board_with_clocks  # noqa: E402

ANDROID = Path(__file__).resolve().parent.parent
TEST_ASSETS = ANDROID / "app" / "src" / "androidTest" / "assets"
APP_ASSETS = ANDROID / "app" / "src" / "main" / "assets"
FRAMES = TEST_ASSETS / "clocks"
OUT = TEST_ASSETS / "chess" / "clock.json"

# Clock strings the parser must agree about, readable and not.
PARSE_CASES = [
    "15:00",
    "3:07",
    "0:09",
    "0:59.9",
    "1:02:03",
    "12:34",
    "9:60",
    "1:60:00",
    "10:2:3",
    "1:2:3",
    "",
    "::",
    "5",
    "5:",
    ":30",
    "100:00",
    "0:00",
    "0:00.0",
    "59:59",
    "1:23:45",
    "abc",
    "1.2",
]

# One case is one frame: the two clock texts, the geometry, the polarity, the icon hands.
CASES: list[dict[str, Any]] = [
    {"label": "start-dark-page", "top": "15:00", "bottom": "15:00", "cell": 43,
     "top_light": False, "angles": (None, None)},
    {"label": "bottom-running", "top": "14:58", "bottom": "14:52", "cell": 43,
     "top_light": False, "angles": (None, 0.0)},
    {"label": "top-running", "top": "9:31", "bottom": "10:04", "cell": 43,
     "top_light": True, "angles": (1.2, None)},
    {"label": "small-cell", "top": "3:07", "bottom": "2:41", "cell": 28,
     "top_light": False, "angles": (None, None)},
    {"label": "large-cell", "top": "25:00", "bottom": "24:13", "cell": 72,
     "top_light": True, "angles": (None, None),
     "x0": 300, "y0": 60, "width": 1400, "height": 800},
    {"label": "tenths", "top": "0:59.9", "bottom": "1:02", "cell": 43,
     "top_light": False, "angles": (0.3, None)},
    {"label": "hours", "top": "1:02:03", "bottom": "59:59", "cell": 43,
     "top_light": False, "angles": (None, None)},
    {"label": "light-page", "top": "5:00", "bottom": "5:00", "cell": 43,
     "top_light": True, "angles": (None, None), "page": (238, 238, 238)},
    {"label": "own-colours", "top": "7:30", "bottom": "7:30", "cell": 43,
     "top_light": False, "angles": (None, None),
     "colours": ((30, 235), (245, 30))},
    {"label": "offset-board", "top": "2:00", "bottom": "1:15", "cell": 51,
     "top_light": False, "angles": (None, None), "x0": 120, "y0": 200},
]

# A short sequence on one board, so the running-clock decision has motion to go on: the
# bottom clock falling and its icon sweeping, then the handover.
SEQUENCE = [
    {"top": "10:00", "bottom": "10:00", "angles": (None, None), "ts": 0.0},
    {"top": "10:00", "bottom": "9:59", "angles": (None, 0.0), "ts": 1.0},
    {"top": "10:00", "bottom": "9:58", "angles": (None, 1.4), "ts": 2.0},
    {"top": "10:00", "bottom": "9:57", "angles": (None, 2.8), "ts": 3.0},
    # The bottom player moves: their clock takes the increment and the top one starts.
    {"top": "10:00", "bottom": "10:02", "angles": (None, None), "ts": 4.0},
    {"top": "9:59", "bottom": "10:02", "angles": (0.0, None), "ts": 5.0},
    {"top": "9:58", "bottom": "10:02", "angles": (1.4, None), "ts": 6.0},
    # Then nothing moves for long enough that the evidence goes stale.
    {"top": "9:58", "bottom": "10:02", "angles": (1.4, None), "ts": 12.0},
]


def reading_json(reading: Any) -> dict[str, Any] | None:
    if reading is None:
        return None
    return {
        "top_ms": reading.top_ms,
        "bottom_ms": reading.bottom_ms,
        "running": reading.running,
        "confidence": reading.confidence,
        "light": reading.light,
    }


def box_json(reading: Any) -> dict[str, Any] | None:
    """What the reader made of one box, for a failure that says where it went wrong."""
    if reading is None:
        return None
    return {
        "text": reading.text,
        "ms": reading.ms,
        "confidence": reading.confidence,
        "box": {
            "top": reading.box.top,
            "bottom": reading.box.bottom,
            "left": reading.box.left,
            "right": reading.box.right,
            "background": reading.box.background,
            "text": reading.box.text,
            "light": reading.box.light,
        },
        "glyphs": [[label, score, margin] for label, score, margin in reading.glyphs],
        "icon": None
        if reading.icon is None
        else [reading.icon.x, reading.icon.y, reading.icon.w, reading.icon.h],
    }


def observation_json(observation: Any) -> dict[str, Any] | None:
    if observation is None:
        return None
    return {
        "white_ms": observation.white_ms,
        "black_ms": observation.black_ms,
        "running": str(observation.running) if observation.running else None,
    }


def cropped(frame: np.ndarray, board: Any) -> tuple[np.ndarray, Any]:
    """The frame cut down to the window the reader reads, with the board moved with it.

    The reader only looks at the two bands above and below the board, within 3.5 cells of
    its right edge; everything else in a 1280x720 board frame is a megabyte of PNG nobody
    reads.
    """
    from sidecoach.events import Region  # noqa: PLC0415
    from sidecoach.vision.board_detect import BoardCandidate  # noqa: PLC0415

    height, width = frame.shape[:2]
    cell = board.cell_px
    x, y, w, h = board.region
    reach = int(round(SEARCH_CELLS * cell)) + 2
    x0 = max(0, int(x + w - BOX_WIDTH_CELLS[1] * cell) - 4)
    x1 = min(width, x + w + 2)
    y0 = max(0, y - reach)
    y1 = min(height, y + h + reach)
    cut = np.ascontiguousarray(frame[y0:y1, x0:x1])
    moved = BoardCandidate(
        region=Region(x - x0, y - y0, w, h),
        cell_px=board.cell_px,
        score=board.score,
        contrast=board.contrast,
        fit=board.fit,
    )
    return cut, moved


def render(case: dict[str, Any], templates: GlyphTemplates) -> tuple[np.ndarray, Any]:
    kwargs: dict[str, Any] = {
        "cell": case["cell"],
        "top_light": case["top_light"],
        "icon_angles": case["angles"],
        "templates": templates,
    }
    for key in ("page", "colours", "x0", "y0", "width", "height"):
        if key in case:
            kwargs[key] = case[key]
    return cropped(*board_with_clocks(case["top"], case["bottom"], **kwargs))


def board_dict(board: Any) -> dict[str, Any]:
    cell = board.cell_px
    return {
        "x": board.region.x,
        "y": board.region.y,
        "w": board.region.w,
        "h": board.region.h,
        # JSON has no NaN; the Kotlin reads the word back as one.
        "cell": "NaN" if cell != cell else cell,
    }


def edge_cases(templates: GlyphTemplates) -> list[dict[str, Any]]:
    """The frame's edges (`tests/test_clock.py`): a page scrolled until the top clock is off
    the frame, a frame cut through both boxes, and boards no reader can use -- none of
    them raises, and none reads a clock that is not wholly there."""
    from sidecoach.events import Region  # noqa: PLC0415
    from sidecoach.vision.board_detect import BoardCandidate  # noqa: PLC0415

    out: list[dict[str, Any]] = []

    def record(label: str, frame: np.ndarray, board: Any, name: str | None = None) -> None:
        if name is None:
            name = f"edge-{label}.png"
            cv2.imwrite(str(FRAMES / name), frame)
            frame = cv2.imread(str(FRAMES / name), cv2.IMREAD_COLOR)
        reading = ClockReader(templates).read(frame, board, 1.0)
        out.append({"label": label, "frame": name, "board": board_dict(board), "reading": reading_json(reading)})

    frame, board = board_with_clocks("16:18", "15:34", cell=43, y0=60, height=800, templates=templates)
    # Scrolled up until the board's top edge is off screen: the clock above it has gone with
    # it, the one below is still there.
    scrolled = frame[200:, :]
    moved = BoardCandidate(
        Region(board.region.x, board.region.y - 200, board.region.w, board.region.h),
        board.cell_px, 1.0, 60.0, 1.0,
    )
    record("scrolled-off", *cropped(scrolled, moved))
    # Cut just left of the clocks' right edge, so both boxes are half there.
    edge = board.region.x + board.region.w - 20
    record("cut-off", *cropped(frame[:, :edge], board))
    # Boards no reader can use, on the first case's frame.
    first = cv2.imread(str(FRAMES / f"{CASES[0]['label']}.png"), cv2.IMREAD_COLOR)
    for label, region, cell in (
        ("empty-region", Region(300, 120, 0, 0), 43.0),
        ("no-cell", Region(300, 120, 344, 344), 0.0),
        ("above-left", Region(-500, -500, 344, 344), 43.0),
        ("far-off", Region(9000, 9000, 344, 344), 43.0),
        ("nan-cell", Region(300, 120, 344, 344), float("nan")),
    ):
        record(label, first, BoardCandidate(region, cell, 1.0, 60.0, 1.0), f"{CASES[0]['label']}.png")
    return out


def main() -> None:
    templates = GlyphTemplates.load()

    APP_ASSETS.joinpath("vision").mkdir(parents=True, exist_ok=True)
    canvases = np.clip(np.rint(templates.canvases * 255), 0, 255).astype(np.uint8)
    (APP_ASSETS / "vision" / "clock_templates.json").write_text(
        json.dumps(
            {
                "labels": list(templates.labels),
                "canvases": [one.reshape(-1).tolist() for one in canvases],
                "advances": [float(a) for a in templates.advances],
            }
        ),
        encoding="utf-8",
    )

    FRAMES.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for case in CASES:
        frame, board = render(case, templates)
        name = f"{case['label']}.png"
        cv2.imwrite(str(FRAMES / name), frame)
        # Read back what was written, so the Kotlin and the Python see the same pixels.
        frame = cv2.imread(str(FRAMES / name), cv2.IMREAD_COLOR)
        reader = ClockReader(templates)
        reading = reader.read(frame, board, 1.0)
        cases.append(
            {
                "label": case["label"],
                "frame": name,
                "board": {
                    "x": board.region.x,
                    "y": board.region.y,
                    "w": board.region.w,
                    "h": board.region.h,
                    "cell": board.cell_px,
                },
                "ts": 1.0,
                "reading": reading_json(reading),
                "boxes": {
                    "top": box_json(reader.last["top"]),
                    "bottom": box_json(reader.last["bottom"]),
                },
                "observation_white_bottom": observation_json(
                    clock_observation(reading, Side.WHITE)
                ),
                "observation_black_bottom": observation_json(
                    clock_observation(reading, Side.BLACK)
                ),
            }
        )

    # The sequence, one reader across every frame.
    reader = ClockReader(templates)
    steps: list[dict[str, Any]] = []
    for index, step in enumerate(SEQUENCE):
        frame, board = render(
            {
                "top": step["top"],
                "bottom": step["bottom"],
                "cell": 43,
                "top_light": False,
                "angles": step["angles"],
            },
            templates,
        )
        name = f"sequence-{index:02d}.png"
        cv2.imwrite(str(FRAMES / name), frame)
        frame = cv2.imread(str(FRAMES / name), cv2.IMREAD_COLOR)
        reading = reader.read(frame, board, step["ts"])
        steps.append(
            {
                "frame": name,
                "ts": step["ts"],
                "board": {
                    "x": board.region.x,
                    "y": board.region.y,
                    "w": board.region.w,
                    "h": board.region.h,
                    "cell": board.cell_px,
                },
                "reading": reading_json(reading),
                "observation_white_bottom": observation_json(
                    clock_observation(reading, Side.WHITE)
                ),
            }
        )

    # A board with no clocks drawn anywhere: the reader must answer nothing.
    from tests.synth import synthetic_board  # noqa: PLC0415

    from sidecoach.vision.board_detect import BoardCandidate  # noqa: PLC0415
    from sidecoach.events import Region  # noqa: PLC0415

    bare = synthetic_board(width=1280, height=720, x0=300, y0=120, cell=43, pieces=True, seed=3)
    bare, bare_board = cropped(
        bare,
        BoardCandidate(
            region=Region(300, 120, 8 * 43, 8 * 43),
            cell_px=43.0,
            score=1.0,
            contrast=60.0,
            fit=1.0,
        ),
    )
    cv2.imwrite(str(FRAMES / "no-clocks.png"), bare)
    bare = cv2.imread(str(FRAMES / "no-clocks.png"), cv2.IMREAD_COLOR)
    bare_reading = ClockReader(templates).read(bare, bare_board, 1.0)

    payload = {
        "parse": [{"text": text, "ms": parse_clock(text)} for text in PARSE_CASES],
        "cases": cases,
        "sequence": steps,
        "bare": {
            "frame": "no-clocks.png",
            "board": {
                "x": bare_board.region.x,
                "y": bare_board.region.y,
                "w": bare_board.region.w,
                "h": bare_board.region.h,
                "cell": bare_board.cell_px,
            },
            "reading": reading_json(bare_reading),
        },
        "edges": edge_cases(templates),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload), encoding="utf-8")
    pngs = sum(one.stat().st_size for one in FRAMES.glob("*.png"))
    print(
        f"{len(cases)} frames + {len(steps)} in sequence, {len(PARSE_CASES)} parses "
        f"-> {OUT} and {FRAMES} ({pngs / 1e6:.1f} MB of PNG)"
    )


if __name__ == "__main__":
    main()
