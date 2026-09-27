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
  result), with `badge_at`/`find_disc` and `read_result` for each.
- boards: small frames with a board on a busy page -- in place, nudged, slid, resized,
  covered by the game-over dialog, flush with the frame's corner at a half-pixel pitch, two
  boards, a larger checkerboard, an inverted and a low-contrast theme, nothing -- with what
  `detect_boards` returns for each, the fit at the board and a cell to either side, and the
  tracker's lock after each of a run of updates.

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
    to_gray,
)
from sidecoach.vision.result_badges import (  # noqa: E402
    badge_at,
    badge_on,
    find_disc,
    read_result,
    square_colour,
)
from sidecoach.vision.squares import BoardGeometry  # noqa: E402
from sidecoach.vision.tracker import BoardTracker  # noqa: E402
from tests.synth import bulky_pieces, draw_dialog, render_board, synthetic_board  # noqa: E402

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
                "result": read_result(frame, geometry, board, bottom),
            }
        )
    return {"crops": crops, "boards": boards}


# ---- the boards ----------------------------------------------------------------------------


def candidate_of(board: BoardCandidate | None) -> list | None:
    if board is None:
        return None
    r = board.region
    return [r.x, r.y, r.w, r.h, r6(board.cell_px), r6(board.fit), r6(board.contrast), board.occupied]


def page(x0: int, y0: int, cell: int, **kw) -> np.ndarray:  # noqa: ANN003
    return synthetic_board(width=W, height=H, x0=x0, y0=y0, cell=cell, seed=5, **kw)


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
        ("moved half a cell: re-detected", {"max_misses": 2}, ["board"] + ["half-cell-on"] * 4),
        ("slid a whole cell: the re-fit moves the lock", {"refine_every": 3, "max_misses": 3}, ["board"] + ["cell-on"] * 3),
        ("nudged under a third of a cell: the re-fit", {}, ["board"] + ["nudged"] * 12),
        ("resized in place", {}, ["board"] + ["resized"] * 4),
        ("the dialog over it: held, then clear", {}, ["board", "dialog", "dialog", "board", "board"]),
        ("flush with the corner at a half-pixel pitch", {}, ["corner-half-pixel"] * 12),
        ("nothing, then a board", {}, ["noise"] * 4 + ["board"] * 2),
    ):
        tracker = BoardTracker(**settings)
        states = []
        for name in sequence:
            board = tracker.update(frames[name])
            states.append(
                [tracker.locked, tracker.verified, tracker.detections, candidate_of(board)]
            )
        runs.append(
            {"what": what, "settings": settings, "frames": [files[n] for n in sequence], "states": states}
        )
    return {"detections": detections, "runs": runs}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    out = {"gate": gate_cases(), "badges": badge_cases(), "boards": board_cases()}
    (OUT / "index.json").write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT.iterdir())
    gate = out["gate"]
    print("gate:", [s["changed"] for s in gate["scenarios"]], [b["gate"] for b in gate["boards"]])
    print("badges:", [(c["file"], c["badge"]) for c in out["badges"]["crops"]])
    print("badge boards:", [(b["file"], b["kings"], b["result"]) for b in out["badges"]["boards"]])
    for d in out["boards"]["detections"]:
        print("detect", d["file"], d["boards"][:1], [f[5] for f in d["fits"]])
    for run in out["boards"]["runs"]:
        print("track", run["what"], [s[3][:5] if s[3] else None for s in run["states"]][-1], run["states"][-1][2])
    print(f"{size / 1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
