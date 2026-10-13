"""The vision checks on the crops of real games, which stay on the developer's disk.

The public repo carries no site's artwork (Tai, 2026-10-13): the crops of chess.com screens the
tests once read from tests/fixtures are kept locally, under data/fixtures/real/ (data/ is never
committed), and the tests read stand-ins drawn with lichess's open boards and pieces instead
(tests/scenes.py). This script runs the old real-pixel checks, assertion for assertion, on the
local copies:

- page: the board found on a chess.com bot game's page (chesscom_page.png), 86 px squares at
  (290, 65) -- tests/test_board_detect.py's `test_a_cluttered_page_yields_its_board`.
- arrows: the trial game's arrows over the e5 pawn (chesscom_start.png, chesscom_arrow_clean.png,
  chesscom_arrow_640.png, chesscom_arrow_662.png, frames.json) --
  tests/test_overlay_marks.py's `test_the_trial_games_arrow_frames_read_the_same`.
- eights: the side at the bottom read from where the 8 is, nothing learned
  (chesscom_puzzle_black_84.png, chesscom_start.png) -- tests/test_coordinates.py's
  `test_the_8_reads_the_side_with_nothing_learned`.
- cold: g01 read cold, a settled frame adopted and a frame in flight refused (g01_start.png,
  g01_settled.png, g01_in_flight.png, frames.json) -- tests/test_cold_start.py's
  `test_cold_start_reads_the_recorded_games_frames_the_same`.
- badges: the result badges of two live games (badges/manifest.json and its eight crops) --
  tests/test_result_badges.py's `test_the_live_games_badges_read_as_recorded`.

What only these files have: the site's own piece set, fonts, arrow and badge artwork, and the
blur of a real screen capture.

    uv run --no-sync python tools/vision/real_crops.py
    uv run --no-sync python tools/vision/real_crops.py --dir D:/elsewhere/real badges cold

Exit status: 0 every check passed, 1 a check failed, 2 files are missing (named).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REAL = REPO / "data" / "fixtures" / "real"
# As the test suite runs it (tests/conftest.py): the piece net on the CPU, no voice models.
os.environ.setdefault("SIDECOACH_EMBED_DEVICE", "cpu")
os.environ.setdefault("SIDECOACH_VOICE_MODELS", "0")

import chess  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from sidecoach.events import GameStarted, Region, Side  # noqa: E402
from sidecoach.state.pipeline import GamePipeline  # noqa: E402
from sidecoach.state.reconcile import Readings  # noqa: E402
from sidecoach.vision.board_detect import detect_boards  # noqa: E402
from sidecoach.vision.coordinates import GLYPH, read_bottom_side  # noqa: E402
from sidecoach.vision.pieces import CellReading  # noqa: E402
from sidecoach.vision.result_badges import Badge, badge_at  # noqa: E402
from sidecoach.vision.squares import BoardGeometry, square_at  # noqa: E402
from sidecoach.vision.theme_cache import ThemeCache  # noqa: E402

BADGE_FILES = [
    "loss_g1_43px_1300.png",
    "loss_h5_80px_1915.png",
    "none_g1_43px_1200.png",
    "none_g1_80px_1649.png",
    "none_g8_43px_1200.png",
    "none_h5_80px_1649.png",
    "win_g1_80px_1915.png",
    "win_g8_43px_1300.png",
]
FILES: dict[str, list[str]] = {
    "page": ["chesscom_page.png"],
    "arrows": [
        "frames.json",
        "chesscom_start.png",
        "chesscom_arrow_clean.png",
        "chesscom_arrow_640.png",
        "chesscom_arrow_662.png",
    ],
    "eights": ["chesscom_puzzle_black_84.png", "chesscom_start.png"],
    "cold": ["frames.json", "g01_start.png", "g01_settled.png", "g01_in_flight.png"],
    "badges": ["badges/manifest.json", *(f"badges/{name}" for name in BADGE_FILES)],
}


def _image(root: Path, name: str) -> np.ndarray:
    frame = cv2.imread(str(root / name))
    assert frame is not None, f"{name} does not read as an image"
    return frame


def _meta(root: Path) -> dict:
    return json.loads((root / "frames.json").read_text(encoding="utf-8"))


def _cell(readings: Readings, name: str, bottom: Side) -> CellReading:
    target = chess.parse_square(name)
    for r in range(8):
        for c in range(8):
            if square_at(r, c, bottom) == target:
                return readings[r][c]
    raise AssertionError(name)


def check_page(root: Path) -> None:
    boards = detect_boards(_image(root, "chesscom_page.png"))
    assert boards, "no board found on the real page"
    best = boards[0]
    assert best.fit >= 0.85, best
    assert abs(best.cell_px - 86) <= 1, best
    assert abs(best.region.x - 290) <= 2 and abs(best.region.y - 65) <= 2, best


def check_arrows(root: Path) -> None:
    meta = _meta(root)
    margin = meta["margin"]
    start = _image(root, "chesscom_start.png")
    pipe = GamePipeline()
    for i in range(4):
        pipe.process(start, 0.1 * i)
    assert pipe.model is not None and pipe.board is not None and pipe.bottom is not None
    (white_fill, _), (black_fill, _) = pipe.model.piece_colours
    assert white_fill.min() > 200 and black_fill.max() < 80, "the pieces are white and black"
    side = meta["chesscom_start"]["region"][2]
    geometry = BoardGeometry(Region(margin, margin, side, side), side / 8)
    clean = pipe.model.classify_frame(_image(root, "chesscom_arrow_clean.png"), geometry)
    assert _cell(clean, "d4", pipe.bottom).symbol == "P"
    assert _cell(clean, "e5", pipe.bottom).symbol == "p"
    for name in ("chesscom_arrow_640.png", "chesscom_arrow_662.png"):
        readings = pipe.model.classify_frame(_image(root, name), geometry)
        head = _cell(readings, "e5", pipe.bottom)
        assert head.kind == "unknown" and head.drawn and not head.hover, (
            f"{name}: e5 read as {head.kind} {head.symbol}"
        )
        tail = _cell(readings, "d4", pipe.bottom)
        assert tail.symbol in (None, "P"), f"{name}: d4 read as {tail.symbol}"


def check_eights(root: Path) -> None:
    solid = {rank: np.ones((GLYPH, GLYPH), dtype=bool) for rank in range(1, 9)}
    puzzle = _image(root, "chesscom_puzzle_black_84.png")
    on_phone = BoardGeometry(Region(0, 0, 672, 672), 84.0)
    assert read_bottom_side(puzzle, on_phone, solid) is Side.BLACK, "the puzzle's side"
    start = _image(root, "chesscom_start.png")
    [board, *_] = detect_boards(start)
    side = read_bottom_side(start, BoardGeometry(board.region, board.cell_px), solid)
    assert side is Side.WHITE, "the start's side"


def check_cold(root: Path) -> None:
    meta = _meta(root)
    margin = meta["margin"]
    with tempfile.TemporaryDirectory() as folder:
        cache = ThemeCache(Path(folder))
        warm = GamePipeline(cache=cache)
        start = _image(root, "g01_start.png")
        for i in range(4):
            warm.process(start, 0.1 * i)
        assert warm.model is not None and warm.model.rank_labels, "no theme learned from g01"
        side = meta["g01_settled"]["region"][2]
        geometry = BoardGeometry(Region(margin, margin, side, side), side / 8)
        cold = GamePipeline(cache=cache)
        events = cold.cold_start(_image(root, "g01_settled.png"), geometry)
        assert any(isinstance(e, GameStarted) for e in events), "cold start did not begin a game"
        assert cold.bottom is not None and cold.bottom.value == meta["g01_settled"]["bottom"]
        assert cold.reconciler is not None
        truth = chess.Board(meta["g01_settled"]["placement"] + " w - - 0 1")
        off = [
            chess.square_name(sq)
            for sq in chess.SQUARES
            if cold.reconciler.board.piece_at(sq) != truth.piece_at(sq)
        ]
        assert not off, f"cold read disagrees with the tracked position on {off}"
        refused = GamePipeline(cache=cache)
        assert refused.cold_start(_image(root, "g01_in_flight.png"), geometry) == [], (
            "the frame in flight was adopted"
        )


def check_badges(root: Path) -> None:
    manifest = json.loads((root / "badges" / "manifest.json").read_text(encoding="utf-8"))
    wrong = []
    for entry in manifest:
        crop = _image(root / "badges", entry["file"])
        cell = float(entry["cell"])
        want = {"win": Badge.WIN, "loss": Badge.LOSS, "none": None}[entry["kind"]]
        got = badge_at(crop, float(entry["x0"]) + cell, float(entry["y0"]), cell)
        if got is not want:
            wrong.append(f"{entry['file']}: {got}, want {want}")
    assert not wrong, "; ".join(wrong)


CHECKS: dict[str, Callable[[Path], None]] = {
    "page": check_page,
    "arrows": check_arrows,
    "eights": check_eights,
    "cold": check_cold,
    "badges": check_badges,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="The vision checks on the crops of real games, kept locally."
    )
    parser.add_argument("checks", nargs="*", help=f"any of {', '.join(CHECKS)} (default: all)")
    parser.add_argument("--dir", type=Path, default=REAL, help=f"the crops (default {REAL})")
    args = parser.parse_args()
    names = args.checks or list(CHECKS)
    unknown = [name for name in names if name not in CHECKS]
    if unknown:
        parser.error(f"no such check: {', '.join(unknown)}")
    missing = sorted({f for name in names for f in FILES[name] if not (args.dir / f).is_file()})
    if missing:
        print(f"Missing under {args.dir}:")
        for name in missing:
            print(f"  {name}")
        print("Copy the real crops there (they are never committed) and run again.")
        return 2
    failed = 0
    for name in names:
        try:
            CHECKS[name](args.dir)
        except AssertionError as error:
            failed += 1
            print(f"FAIL {name}: {error or traceback.format_exc(limit=-1).strip()}")
        else:
            print(f"pass {name}")
    print(f"{len(names) - failed} of {len(names)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
