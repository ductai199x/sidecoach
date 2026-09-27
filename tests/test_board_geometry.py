"""The board moved, resized, scrolled, dragged partly off the screen and covered, mid-game, every
way a window can be (Tai, 2026-09-26: "make sure you can move the board wherever and resize it
however and it must still work without any problem whatsoever").

Each seed takes a random game from the stress sweep (a theme, a piece style, a size, an
orientation, the moves and their decorations) and puts geometry events around the moves: a
nudge of a fraction of a cell, a shift of a few cells and a fraction, a jump across the screen,
a resize, a drag that moves and resizes at once, any of those animated over several frames, the
board dragged partly off the screen and back, a dialog over part of it (sometimes with a move
played under it), a resize below the size the detector can see and back. Each lands just before
a move or right after it, with the move's highlight still on. The pipeline must commit exactly
the moves played, invent none, and end locked on the board where it is with the game's
position. A few seeds run in the suite; `--seeds` sweeps:

    uv run pytest tests/test_board_geometry.py -q -n auto --seeds 300
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import chess
import cv2
import numpy as np
import pytest

from sidecoach.events import GameEnded, GameStarted, MovePlayed, Resync, Side, Unexplained
from sidecoach.state.pipeline import GamePipeline
from tests.synth import synthetic_board
from tests.test_stress import _drag_path, scenario

pytestmark = pytest.mark.vision

W, H = 1920, 1200
CELLS = (32, 40, 48, 56, 64, 80, 96, 112)
MIN_CELL = 28  # a resize stays at or above this; "tiny" goes below the detector's 24 px floor
MAX_CELL = 136  # the board fits the frame's height with room
EVENTS = {
    "none": 30,
    "nudge": 8,
    "shift": 12,
    "jump": 6,
    "resize": 12,
    "drag_resize": 8,
    "animated": 10,
    "offscreen": 5,
    "cover": 5,
    "tiny": 3,
}


@dataclass
class Geo:
    x: float
    y: float
    cell: float

    def fits(self) -> bool:
        return (
            self.x >= 0
            and self.y >= 0
            and self.x + 8 * self.cell <= W
            and self.y + 8 * self.cell <= H
        )


def _clamp(g: Geo) -> Geo:
    """On the frame, whole: at the cell size the renderer draws (whole pixels), with the corner
    rounded as it is drawn -- a board clamped at 70.6 px cells was drawn at 71 and ran 3 px off
    the frame, which is the off-screen scenario, not this one."""
    cell = float(round(min(MAX_CELL, max(MIN_CELL, g.cell))))
    x = min(max(0, round(g.x)), W - 8 * int(cell))
    y = min(max(0, round(g.y)), H - 8 * int(cell))
    return Geo(float(x), float(y), cell)


@dataclass(frozen=True)
class Outcome:
    got: list[str]
    played: list[str]
    same_board: bool
    locked_right: bool
    log: list[str]


def play(seed: int) -> Outcome:
    sc = scenario(seed)
    rng = random.Random(seed * 7919 + 1)
    pipe = GamePipeline()
    board = chess.Board()
    ts = 0.0
    got: list[str] = []
    log: list[str] = []
    geo = _clamp(Geo(rng.uniform(0, W), rng.uniform(0, H), rng.choice(CELLS)))
    geo = _clamp(Geo(rng.uniform(0, W - 8 * geo.cell), rng.uniform(0, H - 8 * geo.cell), geo.cell))

    def frame_for(
        g: Geo, cover: tuple[float, float, float, float] | None = None, **kw: object
    ) -> np.ndarray:
        img = synthetic_board(
            width=W,
            height=H,
            board=board,
            theme=sc.theme,
            cell=int(round(g.cell)),
            x0=int(round(g.x)),
            y0=int(round(g.y)),
            bottom=sc.bottom,
            coordinates=sc.coordinates,
            style=sc.style,
            highlight_colour=sc.highlight_colour,
            arrow_colour=sc.arrow_colour,
            arrow_width=sc.arrow_width,
            **kw,  # type: ignore[arg-type]
        )
        if cover is not None:
            fx, fy, fw, fh = cover
            size = 8 * g.cell
            x0, y0 = int(g.x + fx * size), int(g.y + fy * size)
            cv2.rectangle(
                img, (x0, y0), (int(x0 + fw * size), int(y0 + fh * size)), (38, 38, 42), -1
            )
        return img

    def show(
        frames: int,
        g: Geo | None = None,
        cover: tuple[float, float, float, float] | None = None,
        **kw: object,
    ) -> None:
        nonlocal ts
        for _ in range(frames):
            for event in pipe.process(frame_for(g or geo, cover=cover, **kw), ts):
                if isinstance(event, MovePlayed):
                    got.append(event.san)
                    log.append(event.san)
                elif isinstance(event, Resync | Unexplained | GameStarted | GameEnded):
                    log.append(f"{type(event).__name__}({getattr(event, 'reason', '')})")
            ts += 0.1

    def target(kind: str) -> Geo:
        """Where a moving event takes the board (for the "during" timing)."""
        if kind == "nudge":
            return _clamp(
                Geo(
                    geo.x + rng.uniform(-0.5, 0.5) * geo.cell,
                    geo.y + rng.uniform(-0.5, 0.5) * geo.cell,
                    geo.cell,
                )
            )
        if kind == "shift":
            dx, dy = rng.choice([(1, 0), (0, 1), (1, 1), (1, -1)])
            d = rng.uniform(0.5, 3.5) * rng.choice([-1, 1])
            return _clamp(Geo(geo.x + dx * d * geo.cell, geo.y + dy * d * geo.cell, geo.cell))
        if kind == "jump":
            return _clamp(Geo(rng.uniform(0, W), rng.uniform(0, H), geo.cell))
        cell = geo.cell * rng.uniform(0.6, 1.6)
        if kind == "resize":
            cx, cy = geo.x + 4 * geo.cell, geo.y + 4 * geo.cell
            return _clamp(Geo(cx - 4 * cell, cy - 4 * cell, cell))
        return _clamp(
            Geo(geo.x + rng.uniform(-3, 3) * geo.cell, geo.y + rng.uniform(-3, 3) * geo.cell, cell)
        )

    def event(kind: str, last: chess.Move | None) -> None:
        nonlocal geo
        lit: dict[str, object] = {"highlight": last} if last is not None else {}
        if kind == "nudge":
            geo = _clamp(
                Geo(
                    geo.x + rng.uniform(-0.5, 0.5) * geo.cell,
                    geo.y + rng.uniform(-0.5, 0.5) * geo.cell,
                    geo.cell,
                )
            )
        elif kind == "shift":
            dx, dy = rng.choice([(1, 0), (0, 1), (1, 1), (1, -1)])
            d = rng.uniform(0.5, 3.5) * rng.choice([-1, 1])
            geo = _clamp(Geo(geo.x + dx * d * geo.cell, geo.y + dy * d * geo.cell, geo.cell))
        elif kind == "jump":
            geo = _clamp(Geo(rng.uniform(0, W), rng.uniform(0, H), geo.cell))
        elif kind in ("resize", "drag_resize"):
            cell = geo.cell * rng.uniform(0.6, 1.6)
            if kind == "resize":
                anchor = rng.choice(["corner", "centre"])
                if anchor == "centre":
                    cx, cy = geo.x + 4 * geo.cell, geo.y + 4 * geo.cell
                    geo = _clamp(Geo(cx - 4 * cell, cy - 4 * cell, cell))
                else:
                    geo = _clamp(Geo(geo.x, geo.y, cell))
            else:
                geo = _clamp(
                    Geo(
                        geo.x + rng.uniform(-3, 3) * geo.cell,
                        geo.y + rng.uniform(-3, 3) * geo.cell,
                        cell,
                    )
                )
        elif kind == "animated":
            target = _clamp(
                Geo(
                    geo.x + rng.uniform(-3, 3) * geo.cell,
                    geo.y + rng.uniform(-3, 3) * geo.cell,
                    geo.cell * rng.choice([1.0, 1.0, rng.uniform(0.7, 1.4)]),
                )
            )
            steps = rng.choice([3, 5, 8])
            for i in range(1, steps):
                t = i / steps
                mid = _clamp(
                    Geo(
                        geo.x + (target.x - geo.x) * t,
                        geo.y + (target.y - geo.y) * t,
                        geo.cell + (target.cell - geo.cell) * t,
                    )
                )
                show(1, mid, **lit)
            geo = target
        elif kind == "offscreen":
            edge = rng.choice(["left", "right", "top", "bottom"])
            off = rng.uniform(1.0, 4.0) * geo.cell
            size = 8 * geo.cell
            gx = -off if edge == "left" else (W - size + off if edge == "right" else geo.x)
            gy = -off if edge == "top" else (H - size + off if edge == "bottom" else geo.y)
            show(rng.randint(3, 6), Geo(gx, gy, geo.cell), **lit)
            geo = _clamp(Geo(rng.uniform(0, W), rng.uniform(0, H), geo.cell))
        elif kind == "tiny":
            small = Geo(geo.x, geo.y, rng.uniform(14, 22))
            show(rng.randint(2, 4), small, **lit)
            geo = _clamp(Geo(geo.x, geo.y, max(MIN_CELL, geo.cell * rng.uniform(0.8, 1.2))))
        log.append(f"{kind}->({geo.x:.0f},{geo.y:.0f},{geo.cell:.1f})")

    show(3)
    assert pipe.reconciler is not None and pipe.reconciler.started, sc.where
    user_colour = chess.WHITE if sc.bottom == Side.WHITE else chess.BLACK
    last: chess.Move | None = None
    for san, decoration in zip(sc.sans, sc.decorations, strict=True):
        log.append(f"| {san}:{decoration}")
        kind = rng.choices(list(EVENTS), list(EVENTS.values()))[0]
        when = rng.choice(["before", "after", "during"])
        move = board.parse_san(san)
        mover = board.turn
        drag = (
            _drag_path(move, sc.bottom, rng)
            if decoration == "drag" and mover == user_colour
            else []
        )
        slide = rng.choice([1, 2, 3, 5]) if decoration == "animate" else 0
        frames = len(drag) + slide + 3
        # The geometry of each of the move's frames: where the board is while it is played.
        during: list[Geo] = [geo] * frames
        covered: list[tuple[float, float, float, float] | None] = [None] * frames
        if kind != "none" and when == "during":
            if kind in ("nudge", "shift", "jump", "resize", "drag_resize", "animated"):
                goal = target("drag_resize" if kind == "animated" else kind)
                during = [
                    _clamp(
                        Geo(
                            geo.x + (goal.x - geo.x) * (i + 1) / frames,
                            geo.y + (goal.y - geo.y) * (i + 1) / frames,
                            geo.cell + (goal.cell - geo.cell) * (i + 1) / frames,
                        )
                    )
                    for i in range(frames)
                ]
                geo = goal
            elif kind == "cover":
                box = (
                    rng.uniform(0, 0.5),
                    rng.uniform(0, 0.5),
                    rng.uniform(0.2, 0.5),
                    rng.uniform(0.2, 0.5),
                )
                covered = [box] * frames
            elif kind == "offscreen":
                edge = rng.choice(["left", "right", "top", "bottom"])
                off = rng.uniform(1.0, 3.0) * geo.cell
                size = 8 * geo.cell
                gx = -off if edge == "left" else (W - size + off if edge == "right" else geo.x)
                gy = -off if edge == "top" else (H - size + off if edge == "bottom" else geo.y)
                during = [Geo(gx, gy, geo.cell)] * frames
            elif kind == "tiny":
                during = [Geo(geo.x, geo.y, rng.uniform(14, 22))] * frames
            log.append(f"during {kind}->({geo.x:.0f},{geo.y:.0f},{geo.cell:.1f})")
        step = iter(range(frames))
        if kind != "none" and when == "before":
            if kind == "cover":
                cover = (
                    rng.uniform(0, 0.5),
                    rng.uniform(0, 0.5),
                    rng.uniform(0.2, 0.5),
                    rng.uniform(0.2, 0.5),
                )
                show(rng.randint(3, 8), cover=cover, **({"highlight": last} if last else {}))
                log.append("cover")
            else:
                event(kind, last)
        for drag_step in drag:
            i = next(step)
            show(1, during[i], cover=covered[i], dragging=drag_step)
        for k in range(1, slide + 1):
            i = next(step)
            show(
                1,
                during[i],
                cover=covered[i],
                highlight=move,
                animating=(move.from_square, move.to_square, k / (slide + 1)),
            )
        board.push(move)
        last = move
        extra: dict[str, object] = {"highlight": move, "check": True}
        if decoration == "arrow":
            targets = [m for m in board.legal_moves if not board.is_capture(m)]
            if targets:
                extra["arrows"] = (rng.choice(targets),)
        if decoration == "badge":
            extra["badge"] = (move.to_square, rng.choice(["best", "good", "blunder", "brilliant"]))
        for _ in range(3):
            i = next(step)
            show(1, during[i], cover=covered[i], **extra)
        if kind != "none" and when == "during":
            if kind == "offscreen":
                geo = _clamp(Geo(rng.uniform(0, W), rng.uniform(0, H), geo.cell))
            show(3, highlight=move)
        if kind != "none" and when == "after":
            if kind == "cover":
                cover = (
                    rng.uniform(0, 0.5),
                    rng.uniform(0, 0.5),
                    rng.uniform(0.2, 0.5),
                    rng.uniform(0.2, 0.5),
                )
                show(rng.randint(3, 8), cover=cover, highlight=move)
                log.append("cover")
            else:
                event(kind, move)
            show(2, highlight=move)
    # A full re-fit cycle (the tracker re-fits sub-cell drift every 10 frames) before the lock is
    # held to the board.
    show(12, **({"highlight": last} if last else {}))
    assert pipe.reconciler is not None
    same = pipe.reconciler.board.board_fen() == board.board_fen()
    b = pipe.board
    locked = (
        b is not None
        and abs(b.region.x - geo.x) <= 3
        and abs(b.region.y - geo.y) <= 3
        and abs(b.cell_px - round(geo.cell)) <= 1.5
    )
    return Outcome(got, list(sc.sans), same, locked, log)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "seed" in metafunc.fixturenames:
        spec = str(metafunc.config.getoption("--seeds") or "")
        if not spec:
            seeds = [1, 2, 3, 4]
        elif "-" in spec:
            lo, hi = (int(x) for x in spec.split("-", 1))
            seeds = list(range(lo, hi + 1))
        else:
            seeds = list(range(1, int(spec) + 1))
        metafunc.parametrize("seed", seeds)


def test_a_game_survives_the_board_moved_resized_covered_and_scrolled(seed: int) -> None:
    """Every move committed, none invented, the game's position at the end, and the lock on
    the board where it is. Two moves that commute may commit in either order when frames
    between them were held (as in the stress sweep)."""
    sc = scenario(seed)
    out = play(seed)
    assert sorted(out.got) == sorted(out.played) and out.same_board and out.locked_right, (
        f"{sc.where}: committed {out.got}, played {out.played}, same board {out.same_board}, "
        f"locked right {out.locked_right}; geometry {out.log}"
    )
