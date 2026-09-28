"""Record the desktop's answers for the rest of the vision: the change gate, the result
badges, the board detector and the tracker.

The parity audit of 2026-09-26 found these ported faithfully but held to nothing the
desktop computed: the phone's tests checked a board was found within a pixel or two of where
it was drawn, not that it was found where the Python finds it, and Python's rounding (halves
to even) was the phone's (halves up) in every place a board's pixels are counted -- which
decides a cell's box when the pitch is a half pixel, and the gate's margin when the cell is
odd. This records, compactly, one table per part:

- gate: frames built from a flat grey and rectangles moved by some levels (the Kotlin builds
  the same), each scenario a run of frames and the `changed` answers of the desktop's
  `ChangeGate`; and `board_gate` for boards whose half cell is a half pixel, or off the frame.
- badges: the live games' crops (tests/fixtures/badges) and a few rendered boards (both
  orientations, two sizes, the kings on the edges, a bulky piece walling off its corner, no
  result), with `badge_at`/`find_disc`, and the pair `result_of` makes of the kings' badges.
- boards: small frames with a board on a busy page -- in place, nudged, slid, resized,
  covered by the game-over dialog, flush with the frame's corner at a half-pixel pitch, two
  boards, a larger checkerboard, an inverted and a low-contrast theme, nothing -- with what
  `detect_boards` returns for each, the fit at the board and a cell to either side, and the
  tracker's lock after each of a run of updates. Then the moves of desktop 17c45a6's
  `test_a_board_moved_any_fraction_of_a_cell_is_found_again` on a larger page -- two cells and
  a fraction, two whole cells, 0.8 of a cell up, the diagonals -- with the cells readable at
  the old lock and the new place (`readable_cells`), and the tracker following each (its
  `probes` beside its `detections`).
- edges: the desktop's detector and tracker tests of its cfa8544 vision work (80b1194, 44b112f,
  1271cf5, 0c957b5, 2669d9c) on their own 1920 by 1200 pages, the background well away from
  the board flattened: boards flush with the right or bottom edge found, a candidate a
  fraction of a cell off refused though its outer row is cut, boards flush with an edge or a
  cell from it followed through a nudge, a board resized to near the bottom locked where it
  is, a board gone and back at another size found within the full search's first wait, and a
  tracker asking for 0.95 searching on the first frame and the sixth; and desktop dcbbeaf's: a
  wood board near the bottom nudged an eighth of a cell, re-fitted before its colours are read
  again, and a game board moved past a re-fit's reach with an empty board beside it. Each
  passes the desktop's own assertions here, or the generator stops.

    cd D:/chess-coaching-companion
    uv run --no-sync python <worktree>/tools/make_vision_fixtures.py

It writes next to itself: <worktree>/app/src/androidTest/assets/vision/.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from sidecoach.events import Region, Side  # noqa: E402
from sidecoach.replay.recording import ChangeGate, board_gate  # noqa: E402
from sidecoach.vision.board_detect import (  # noqa: E402
    BoardCandidate,
    detect_boards,
    fit_score,
    readable_cells,
    square_colours,
    to_gray,
)
from sidecoach.vision.result_badges import (  # noqa: E402
    Badge,
    badge_at,
    badge_on,
    find_disc,
    result_of,
    square_colour,
)
from sidecoach.vision.squares import BoardGeometry  # noqa: E402
from sidecoach.vision.tracker import BoardTracker  # noqa: E402
from tests.synth import (  # noqa: E402
    DARK,
    bulky_pieces,
    busy_background,
    draw_dialog,
    render_board,
    synthetic_board,
)

OUT = Path(__file__).resolve().parents[1] / "app" / "src" / "androidTest" / "assets" / "vision"
BADGES = DESKTOP / "tests" / "fixtures" / "badges"
W, H = 560, 440


def r6(value: float) -> float:
    return round(float(value), 6)


def png(name: str, frame: np.ndarray) -> str:
    cv2.imwrite(str(OUT / name), frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return name


# ---- the gate ----------------------------------------------------------------------------

GATE_W, GATE_H = 120, 90


def gate_frame(rects: list[list[int]]) -> np.ndarray:
    """A flat grey frame with each rectangle (x, y, w, h, levels) raised by some levels."""
    frame = np.full((GATE_H, GATE_W, 3), 100, dtype=np.uint8)
    for x, y, w, h, levels in rects:
        frame[y : y + h, x : x + w] = np.clip(100 + levels, 0, 255)
    return frame


def gate_cases() -> dict:
    area = GATE_W * GATE_H  # 10800: 0.0005 of it is 5.4 pixels
    scenarios = [
        {
            "what": "whole frame: first, same, a speck, a block, a small step, the keyframe",
            "region": None,
            "keyframe": 1.0,
            "steps": [
                [0.0, []],
                [0.1, []],
                [0.2, [[10, 10, 2, 2, 40]]],  # 4 pixels: under the share
                [0.3, [[10, 10, 3, 2, 40]]],  # 6 pixels: over it
                [0.4, [[10, 10, 3, 2, 40], [50, 50, 20, 20, 12]]],  # 12 levels is not over 12
                [0.5, [[10, 10, 3, 2, 40], [50, 50, 20, 20, 13]]],
                [1.4, [[10, 10, 3, 2, 40], [50, 50, 20, 20, 13]]],  # nothing new: not yet a second
                [1.5, [[10, 10, 3, 2, 40], [50, 50, 20, 20, 13]]],  # a second since the last change
            ],
        },
        {
            "what": "gated on a region: a change outside it is none, inside it is",
            "region": [30, 20, 40, 30],
            "keyframe": None,
            "steps": [
                [0.0, []],
                [0.1, [[0, 0, 25, 15, 60]]],
                [0.2, [[0, 0, 25, 15, 60], [35, 25, 5, 5, 60]]],
                [0.3, [[0, 0, 25, 15, 60], [35, 25, 5, 5, 60], [68, 48, 4, 4, 60]]],
            ],
        },
        {
            "what": "a region off the frame compares the whole frame",
            "region": [2000, 2000, 10, 10],
            "keyframe": None,
            "steps": [[0.0, []], [0.1, [[100, 80, 10, 5, 30]]], [0.2, [[100, 80, 10, 5, 30]]]],
        },
    ]
    for scenario in scenarios:
        region = scenario["region"]
        gate = ChangeGate(
            region=None if region is None else Region(*region),
            keyframe_interval_s=scenario["keyframe"],
        )
        scenario["changed"] = [bool(gate.changed(gate_frame(rects), ts)) for ts, rects in scenario["steps"]]
    boards = []
    for x, y, cell, width, height in (
        (4, 420, 83.0, 672, 1496),  # half a cell is 41.5: to even, 42
        (4, 420, 85.0, 672, 1496),  # 42.5: to even, 42 (halves up would say 43)
        (0, 0, 81.0, 672, 1496),  # 40.5: 40, and clamped at the frame's corner
        (0, 0, 43.5, 348, 348),  # 21.75
        (700, 1500, 83.0, 672, 1496),  # off the frame: no gate
    ):
        side = int(round(8 * cell))
        candidate = BoardCandidate(Region(x, y, side, side), cell, 1.0, 50.0)
        gate = board_gate(candidate, width, height)
        boards.append(
            {
                "region": [x, y, side, side],
                "cell": cell,
                "frame": [width, height],
                "gate": None if gate is None else [gate.x, gate.y, gate.w, gate.h],
            }
        )
    return {"frame": [GATE_W, GATE_H], "area": area, "scenarios": scenarios, "boards": boards}


# ---- the badges ----------------------------------------------------------------------------

LOST = "6k1/4rnp1/6Q1/p1p4P/2P2R2/6P1/1r4q1/5RK1 w - - 3 45"
EDGES = "k7/8/8/8/8/8/8/7K w - - 0 1"
BULKY = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8"


def disc_of(found) -> list | None:  # noqa: ANN001
    if found is None:
        return None
    colour, rim, ring = found
    return [[r6(v) for v in colour], r6(rim), r6(ring)]


def badge_cases() -> dict:
    manifest = json.loads((BADGES / "manifest.json").read_text(encoding="utf-8"))
    crops = []
    for entry in manifest:
        shutil.copyfile(BADGES / entry["file"], OUT / f"badge-{entry['file']}")
        crop = cv2.imread(str(BADGES / entry["file"]))
        cell = float(entry["cell"])
        corner_x = float(entry["x0"]) + cell
        corner_y = float(entry["y0"])
        background = square_colour(crop, corner_x, corner_y, cell)
        badge = badge_at(crop, corner_x, corner_y, cell)
        crops.append(
            {
                "file": f"badge-{entry['file']}",
                "corner": [corner_x, corner_y],
                "cell": cell,
                "kind": entry["kind"],
                "background": [r6(v) for v in background],
                "disc": disc_of(find_disc(crop, corner_x, corner_y, cell, background)),
                "badge": None if badge is None else badge.value,
            }
        )
    boards = []
    margin = 6
    for name, fen, cell, theme, bottom, result, bulky in (
        ("lost-white-won", LOST, 44, "green", Side.WHITE, "1-0", False),
        ("lost-black-won-flipped", LOST, 72, "brown", Side.BLACK, "0-1", False),
        ("edges-white-won", EDGES, 44, "blue", Side.WHITE, "1-0", False),
        ("edges-black-won-flipped", EDGES, 44, "brown", Side.BLACK, "0-1", False),
        ("draw-unseen", LOST, 44, "green", Side.WHITE, "1/2-1/2", False),
        ("no-result-bulky", BULKY, 44, "green", Side.WHITE, None, True),
    ):
        board = chess.Board(fen)
        side = 8 * cell + 2 * margin
        frame = synthetic_board(
            board=board, cell=cell, x0=margin, y0=margin, width=side, height=side,
            theme=theme, bottom=bottom, result=result, coordinates=True, seed=3,
        )
        if bulky:
            bulky_pieces(frame, margin, margin, cell, board, bottom)
        geometry = BoardGeometry(Region(margin, margin, 8 * cell, 8 * cell), float(cell))
        kings = {}
        for colour, key in ((chess.WHITE, "white"), (chess.BLACK, "black")):
            square = board.king(colour)
            found = badge_on(frame, geometry, square, bottom)
            kings[key] = None if found is None else found.value
        boards.append(
            {
                "file": png(f"badge-{name}.png", frame),
                "fen": fen,
                "cell": cell,
                "margin": margin,
                "bottom": bottom.value,
                "kings": kings,
                # The pair of the two kings' badges (`result_badges.result_of`, desktop e0ea564).
                "result": result_of(*(
                    None if kings[key] is None else Badge(kings[key]) for key in ("white", "black")
                )),
            }
        )
    return {"crops": crops, "boards": boards}


# ---- the boards ----------------------------------------------------------------------------


def candidate_of(board: BoardCandidate | None) -> list | None:
    if board is None:
        return None
    r = board.region
    return [
        r.x, r.y, r.w, r.h, r6(board.cell_px), r6(board.fit), r6(board.contrast), board.occupied,
        board.readable,
    ]  # fmt: skip


def page(x0: int, y0: int, cell: int, **kw) -> np.ndarray:  # noqa: ANN003
    return synthetic_board(width=W, height=H, x0=x0, y0=y0, cell=cell, seed=5, **kw)


# The moves of desktop 17c45a6's scenario test, in cells, on a page with room for them: the
# board of 40 px cells at (240, 160) on 800 by 640, after 1. e4 e5.
MOVE_W, MOVE_H = 800, 640
MOVE_X0, MOVE_Y0, MOVE_CELL = 240, 160, 40
MOVES = [(0.0, 2.4), (-2.4, 0.0), (2.0, 0.0), (0.0, -0.8), (3.0, 3.0), (-1.4, -1.4)]


def moved_page(dx: float, dy: float) -> np.ndarray:
    board = chess.Board()
    board.push_san("e4")
    move = board.push_san("e5")
    return synthetic_board(
        width=MOVE_W, height=MOVE_H, x0=int(MOVE_X0 + dx * MOVE_CELL), y0=int(MOVE_Y0 + dy * MOVE_CELL),
        cell=MOVE_CELL, board=board, highlight=move, coordinates=True, seed=5,
    )  # fmt: skip


def board_cases() -> dict:
    frames: dict[str, np.ndarray] = {}
    start = chess.Board()
    frames["board"] = page(100, 40, 43, board=start, coordinates=True)
    frames["half-cell-on"] = page(121, 40, 43, board=start, coordinates=True)
    frames["cell-on"] = page(143, 40, 43, board=start, coordinates=True)
    frames["nudged"] = page(113, 40, 43, board=start, coordinates=True)
    frames["resized"] = page(100, 40, 47, board=start, coordinates=True)
    covered = frames["board"].copy()
    draw_dialog(covered, 100, 40, 43, alpha=1.0)
    frames["dialog"] = covered
    frames["noise"] = np.random.default_rng(11).integers(0, 256, (H, W, 3), dtype=np.uint8)
    # Flush with the frame's corner at a half-pixel pitch: drawn at twice the size, halved,
    # as the capture halves the phone's screen.
    big = synthetic_board(width=2 * W, height=2 * H, x0=0, y0=0, cell=87, board=start, coordinates=True, seed=5)
    frames["corner-half-pixel"] = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    two = np.full((H, W, 3), 90, dtype=np.uint8)
    render_board(two, 10, 20, 36, start)
    render_board(two, 330, 60, 26, chess.Board(None))
    frames["two-boards"] = two
    checker = np.full((H, W, 3), 90, dtype=np.uint8)
    for r in range(10):
        for c in range(12):
            colour = (208, 236, 238) if (r + c) % 2 == 0 else (86, 150, 118)
            cv2.rectangle(checker, (20 + c * 40, 20 + r * 40), (20 + (c + 1) * 40 - 1, 20 + (r + 1) * 40 - 1), colour, -1)
    frames["larger-checkerboard"] = checker
    frames["inverted"] = page(90, 30, 45, board=start, light=(60, 60, 200), dark=(200, 200, 60))
    frames["low-contrast"] = page(90, 30, 45, board=start, light=(214, 214, 214), dark=(140, 140, 140))
    frames["moved-home"] = moved_page(0.0, 0.0)
    moved = {}
    for dx, dy in MOVES:
        name = f"moved-{dx:+.1f}-{dy:+.1f}"
        frames[name] = moved_page(dx, dy)
        moved[name] = (int(MOVE_X0 + dx * MOVE_CELL), int(MOVE_Y0 + dy * MOVE_CELL))

    files = {name: png(f"board-{name}.png", frame) for name, frame in frames.items()}
    detections = []
    for name, frame in frames.items():
        gray = to_gray(frame)
        found = detect_boards(frame)
        fits = []
        if found:
            top = found[0]
            for dx in (-1, 0, 1):
                region = Region(int(round(top.region.x + dx * top.cell_px)), top.region.y, top.region.w, top.region.h)
                fit, contrast = fit_score(gray, region, top.cell_px)
                fits.append([region.x, region.y, region.w, region.h, r6(top.cell_px), r6(fit), r6(contrast)])
        detections.append(
            {"file": files[name], "boards": [candidate_of(b) for b in found], "fits": fits}
        )

    runs = []
    for what, settings, sequence in (
        ("locks once and holds", {}, ["board"] * 6),
        # Desktop dcbbeaf: the frame that fails at the lock is re-fitted at once, so a half-cell
        # move and a whole-cell slide are followed on that frame, without a detection.
        ("moved half a cell: followed without a detection", {"max_misses": 2}, ["board"] + ["half-cell-on"] * 4),
        ("slid a whole cell: re-fitted on the frame it fails", {"refine_every": 10, "max_misses": 3}, ["board"] + ["cell-on"] * 3),
        ("nudged under a third of a cell: the re-fit", {}, ["board"] + ["nudged"] * 12),
        ("resized in place", {}, ["board"] + ["resized"] * 4),
        ("the dialog over it: held, then clear", {}, ["board", "dialog", "dialog", "board", "board"]),
        ("flush with the corner at a half-pixel pitch", {}, ["corner-half-pixel"] * 12),
        ("nothing, then a board", {}, ["noise"] * 4 + ["board"] * 2),
    ):
        runs.append(track(what, settings, sequence, frames, files))
    for run, x in ((runs[1], 121), (runs[2], 143)):
        followed = run["states"][1]
        if not (followed[1] and followed[2] == 1 and followed[3] and abs(followed[3][0] - x) <= 2):
            raise SystemExit(f"{run['what']}: the desktop's tracker {run['states'][:2]}")

    # The board moved as the desktop's scenario test moves it: the cells the lock read at home
    # and at the new place, in the lock's colours, and the tracker following it -- to within
    # three pixels of where it was drawn, as the test asserts, or this is no fixture of it.
    home_gray = to_gray(frames["moved-home"])
    home = Region(MOVE_X0, MOVE_Y0, 8 * MOVE_CELL, 8 * MOVE_CELL)
    colours = square_colours(home_gray, home, float(MOVE_CELL))
    assert colours is not None
    readable = []
    for name, (x0, y0) in moved.items():
        gray = to_gray(frames[name])
        for region in (home, Region(x0, y0, 8 * MOVE_CELL, 8 * MOVE_CELL)):
            readable.append(
                {
                    "file": files[name],
                    "region": [region.x, region.y, region.w, region.h],
                    "cell": float(MOVE_CELL),
                    "colours": [r6(v) for v in colours],
                    "readable": readable_cells(gray, region, float(MOVE_CELL), colours),
                }
            )
    for name, (x0, y0) in moved.items():
        run = track(f"moved {name[6:]} cells: found again", {}, ["moved-home"] * 3 + [name] * 12, frames, files)
        last = run["states"][-1][3]
        if last is None or abs(last[0] - x0) > 3 or abs(last[1] - y0) > 3:
            raise SystemExit(f"{name}: the desktop's tracker did not follow the board: {run['states']}")
        runs.append(run)
    return {"detections": detections, "runs": runs, "readable": readable}


def track(what: str, settings: dict, sequence: list[str], frames: dict, files: dict) -> dict:
    """The tracker's lock after each update of a run: locked, verified, its detections, the
    board, its probes."""
    tracker = BoardTracker(**settings)
    states = []
    for name in sequence:
        board = tracker.update(frames[name])
        states.append(
            [tracker.locked, tracker.verified, tracker.detections, candidate_of(board), tracker.probes]
        )
    return {"what": what, "settings": settings, "frames": [files[n] for n in sequence], "states": states}


# ---- boards at the frame's edge: the desktop's tests of cfa8544's vision work ----------------
#
# Desktop 80b1194, 44b112f, 1271cf5, 0c957b5 and 2669d9c test the detector and the tracker on
# full 1920 x 1200 pages (tests/test_board_detect.py, tests/test_tracker.py): boards flush with
# an edge, nudged along it, a cell from it, resized near it, gone and back at another size. The
# frames are theirs, but for the busy background well away from the board, which is flattened
# (EDGE_PAD cells from it and more: past the reach of the tracker's re-fit, 1.7 cells), so a
# page costs a fraction of its megabyte; each case is checked here to pass the desktop's own
# assertions on the flattened page, or it is no fixture of that test.

EDGE_W, EDGE_H = 1920, 1200
EDGE_PAD = 2.0
FLAT = (32, 32, 32)


def flatten_around(frame: np.ndarray, boards: list[tuple[int, int, int]]) -> np.ndarray:
    """Everything more than EDGE_PAD cells from every board in `boards` (x0, y0, cell) set
    to one colour; the boards and what lies around them untouched. No boards: all of it."""
    keep = np.zeros(frame.shape[:2], dtype=bool)
    for x0, y0, cell in boards:
        pad = int(round(EDGE_PAD * cell))
        keep[max(0, y0 - pad) : max(0, y0 + 8 * cell + pad), max(0, x0 - pad) : max(0, x0 + 8 * cell + pad)] = True
    out = frame.copy()
    out[~keep] = FLAT
    return out


def edge_page(x0: int, y0: int, cell: int, **kw) -> np.ndarray:  # noqa: ANN003
    kw.setdefault("width", EDGE_W)
    kw.setdefault("height", EDGE_H)
    return flatten_around(synthetic_board(x0=x0, y0=y0, cell=cell, **kw), [(x0, y0, cell)])


def near(board: BoardCandidate | None, x: int, y: int, cell: float, px: float, pitch: float) -> bool:
    return (
        board is not None
        and abs(board.region.x - x) <= px
        and abs(board.region.y - y) <= px
        and abs(board.cell_px - cell) <= pitch
    )


def edge_cases() -> dict:
    frames: dict[str, np.ndarray] = {}
    detections = []
    runs = []
    files: dict[str, str] = {}

    def keep(name: str, frame: np.ndarray) -> str:
        frames[name] = frame
        files[name] = png(f"edge-{name}.png", frame)
        return name

    # test_board_detect: a board flush with the right or bottom edge is found where it is.
    for style, x, y in (
        ("chess7", EDGE_W - 8 * 136, 80),
        ("pixel", EDGE_W - 8 * 136, 80),
        ("cardinal", 10, EDGE_H - 8 * 136),
        ("governor", 10, EDGE_H - 8 * 136),
    ):
        name = keep(f"flush-{style}", edge_page(x, y, 136, theme="ice", style=style))
        found = detect_boards(frames[name])
        if not found or not near(found[0], x, y, 136, 2, 1.0):
            raise SystemExit(f"{name}: the desktop did not find the board at ({x}, {y}): {found[:1]}")
        detections.append({"file": files[name], "boards": [candidate_of(b) for b in found], "fits": []})

    # test_board_detect: a candidate a fraction of a cell off is refused though its outer row is
    # cut; and test_tracker: the board resized near the frame's bottom is locked where it is.
    board = chess.Board()
    for san in (
        "f3", "Nc6", "g4", "e5", "b4", "b6", "Bh3", "Na5",
        "Kf2", "Bxb4", "c4", "Nxc4", "Bf1", "Nb2", "a3", "d5",
    ):  # fmt: skip
        board.push_san(san)
    kw = {"theme": "wood", "style": "pixel", "bottom": Side.BLACK, "board": board}
    resized = keep("resized-near-bottom", edge_page(583, 110, 136, **kw))
    before = keep("resized-from", edge_page(833, 360, 105, **kw))
    gray = to_gray(frames[resized])
    fits = []
    for region, cell in ((Region(583, 110, 1088, 1088), 136.0), (Region(582, 24, 1090, 1090), 136.2)):
        fit, contrast = fit_score(gray, region, cell)
        fits.append([region.x, region.y, region.w, region.h, r6(cell), r6(fit), r6(contrast)])
    if not (fits[0][5] >= 0.95 and fits[1][5] == 0.0):
        raise SystemExit(f"resized-near-bottom: the desktop's fits {fits}")
    detections.append(
        {"file": files[resized], "boards": [candidate_of(b) for b in detect_boards(frames[resized])], "fits": fits}
    )
    run = track(
        "resized to near the frame's bottom: locked where it is", {},
        [before] + [resized] * (BoardTracker().max_misses + 3), frames, files,
    )  # fmt: skip
    last = run["states"][-1]
    if not (last[1] and last[3] and abs(last[3][0] - 583) <= 2 and abs(last[3][1] - 110) <= 2):
        raise SystemExit(f"resized-near-bottom: the desktop's tracker ended at {last}")
    runs.append(run)

    # test_tracker: a board flush with an edge is followed when nudged.
    for cell, (bx, by), (ax, ay) in (
        (136, (0, 0), (0, 5)),
        (88, (985, 0), (993, 0)),
        (64, (0, 300), (0, 309)),
        (97, (EDGE_W - 776, 318), (EDGE_W - 776, 291)),
        (64, (500, EDGE_H - 512), (509, EDGE_H - 512)),
        (80, (4, 16), (0, 0)),
        (80, (734, 76), (726, 81)),
    ):
        first = keep(f"nudge-{cell}-{bx}-{by}", edge_page(bx, by, cell))
        moved = keep(f"nudge-{cell}-{ax}-{ay}", edge_page(ax, ay, cell))
        run = track(
            f"flush, {cell} px, ({bx}, {by}) nudged to ({ax}, {ay})", {},
            [first] + [moved] * (BoardTracker().refine_every + 2), frames, files,
        )  # fmt: skip
        last = run["states"][-1]
        if not (last[1] and last[3] and abs(last[3][0] - ax) <= 2 and abs(last[3][1] - ay) <= 2
                and abs(last[3][4] - cell) <= 0.5):  # fmt: skip
            raise SystemExit(f"nudge {cell} ({ax}, {ay}): the desktop's tracker ended at {last}")
        runs.append(run)

    # test_tracker (desktop dcbbeaf): a wood board five pixels above the frame's bottom, nudged
    # an eighth of a cell, is re-fitted before its colours are read again.
    board = chess.Board()
    board.set_board_fen("1r1qk1nr/pbpp4/1pn4p/4Ppp1/P7/bP3PP1/1BPNP2P/2Q1KBNR")
    kw = {"board": board, "theme": "wood", "style": "cburnett-wood"}
    first = keep("wood-near-bottom", edge_page(155, 683, 64, **kw))
    nudged = keep("wood-near-bottom-nudged", edge_page(147, 677, 64, **kw))
    run = track("a wood board near the bottom, nudged: re-fitted before its colours", {}, [first] + [nudged] * 3, frames, files)
    if not (run["states"][0][3] and abs(run["states"][0][3][0] - 155) <= 1):
        raise SystemExit(f"wood near the bottom: the desktop's lock {run['states'][0]}")
    last = run["states"][-1]
    if not (last[1] and last[3] and abs(last[3][0] - 147) <= 1 and abs(last[3][1] - 677) <= 1):
        raise SystemExit(f"wood near the bottom: the desktop's tracker ended at {last}")
    runs.append(run)

    # test_tracker (desktop dcbbeaf): the game board moves further than a re-fit reaches, with
    # an empty board of the same size on screen; the search puts the game board first.
    def game_and_empty(game_x: int) -> np.ndarray:
        frame = np.full((900, 1400, 3), 90, dtype=np.uint8)
        render_board(frame, game_x, 100, 48, chess.Board())
        bulky_pieces(frame, game_x, 100, 48, chess.Board())
        render_board(frame, 800, 100, 48, chess.Board(None))
        return frame

    home = keep("game-and-empty-60", game_and_empty(60))
    moved = keep("game-and-empty-380", game_and_empty(380))
    run = track("the game board moved past a re-fit's reach, an empty board beside it", {"max_misses": 2}, [home] + [moved] * 4, frames, files)
    last = run["states"][-1]
    if not (last[3] and abs(last[3][0] - 380) <= 2 and last[2] == 2):
        raise SystemExit(f"game and empty: the desktop's tracker ended at {last}")
    runs.append(run)

    # test_tracker: a board back at another size after a moment is found soon.
    small = {"width": 1400, "height": 1000}
    home = keep("back-home", edge_page(300, 120, 64, **small))
    gone = keep("back-gone", flatten_around(busy_background(1400, 1000, 0), []))
    back = keep("back-again", edge_page(200, 60, 100, **small))
    tracker = BoardTracker()
    sequence = [home] + [gone] * (tracker.max_misses + 1)
    for name in sequence:
        tracker.update(frames[name])
    if tracker.locked:
        raise SystemExit("back: the desktop's tracker kept its lock on the empty page")
    count = 0
    while count < 30:
        count += 1
        sequence.append(back)
        board_back = tracker.update(frames[back])
        if board_back is not None and tracker.verified:
            break
    if not (near(board_back, 200, 60, 100, 2, 1.0) and count <= tracker._full.first):
        raise SystemExit(f"back: the desktop found {board_back} after {count} frames")
    runs.append(track("gone, then back at another size: found soon", {}, sequence, frames, files))

    # test_tracker: the thresholds reach detection -- a fit of 58/64, and a tracker asking
    # for 0.95 searches on the first frame and the sixth.
    spoiled = synthetic_board(x0=300, y0=120, cell=64, pieces=False)
    for r, c in ((0, 0), (0, 2), (2, 0), (4, 4), (6, 2), (2, 6)):
        spoiled[120 + r * 64 : 184 + r * 64, 300 + c * 64 : 364 + c * 64] = DARK
    name = keep("spoiled", flatten_around(spoiled, [(300, 120, 64)]))
    runs.append(track("six light cells painted dark: locked", {}, [name], frames, files))
    run = track("six light cells painted dark, asking 0.95", {"min_fit": 0.95}, [name] * 12, frames, files)
    if runs[-1]["states"][0][3] is None or run["states"][-1][2] != 2 or any(s[3] for s in run["states"]):
        raise SystemExit(f"spoiled: the desktop's trackers {runs[-1]['states']} {run['states'][-1]}")
    runs.append(run)
    return {"detections": detections, "runs": runs}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    out = {"gate": gate_cases(), "badges": badge_cases(), "boards": board_cases(), "edges": edge_cases()}
    (OUT / "index.json").write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT.iterdir())
    gate = out["gate"]
    print("gate:", [s["changed"] for s in gate["scenarios"]], [b["gate"] for b in gate["boards"]])
    print("badges:", [(c["file"], c["badge"]) for c in out["badges"]["crops"]])
    print("badge boards:", [(b["file"], b["kings"], b["result"]) for b in out["badges"]["boards"]])
    for d in out["boards"]["detections"]:
        print("detect", d["file"], d["boards"][:1], [f[5] for f in d["fits"]])
    for run in out["boards"]["runs"]:
        last = run["states"][-1]
        print("track", run["what"], last[3][:5] if last[3] else None, "detections", last[2], "probes", last[4])
    for r in out["boards"]["readable"]:
        print("readable", r["file"], r["region"][:2], r["readable"])
    for d in out["edges"]["detections"]:
        print("edge detect", d["file"], d["boards"][:1], [f[5] for f in d["fits"]])
    for run in out["edges"]["runs"]:
        last = run["states"][-1]
        print("edge track", run["what"], last[3][:5] if last[3] else None, "detections", last[2])
    edge_bytes = sum(p.stat().st_size for p in OUT.glob("edge-*.png"))
    print(f"edge pages: {edge_bytes / 1e6:.1f} MB")
    print(f"{size / 1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
