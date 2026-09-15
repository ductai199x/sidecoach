"""Read the two chess.com clocks drawn next to the board.

What the recordings show (720p copies of a 1440p screen, 43-pixel cells):

- Each clock is a rounded box right-aligned with the board's right edge, one about 0.14 cell
  above the board and one the same distance below it. The box is about 1.56 cells wide and
  0.47 tall. The digits are 0.23 cell tall, right-aligned inside the box with about 0.12 cell
  of padding, and a small clock icon sits at the left of the box. The format is M:SS or
  MM:SS, with tenths under a minute (0:59.9). No clock in the six recordings fell under
  eleven minutes, so the tenths dot is taken from the colon's lower dot, and a tenths
  separator matched as a colon is repaired in `parse_clock`.
- White's box is light with dark text and black's box dark with light text, whichever side
  is to move. The box colours therefore say which clock is white's, not which one runs.
- The running clock's icon animates: its hand sweeps round in bursts, once a second. The
  idle icon never changes a pixel. The digits of the running clock change once a second.
- The digit font is chess.com's own and the same in every recording. Glyphs are cut out of
  the box by column projection, scaled to a fixed line height, and classified by normalised
  correlation against templates learned from the recordings (`clock_templates.npz`, built
  by `scripts/validate_clocks.py --learn`).

The reader is a pure vision component: it reports the top and bottom clocks as drawn, which
box is light, and which clock is running. The pipeline knows which side sits at the bottom
and maps top and bottom to colours itself. Nothing about whose turn it is comes from here:
the reconciler reads the turn off the board. A clock reading is remaining time, the time
control it implies, and a note of which clock is ticking.

Geometry is relative to the board's cell size, and every glyph is scaled to a fixed line
height before classification, so the reader does not depend on the board or screen size.

A bad frame must never take a live session down, so `ClockReader.read` catches everything
and returns None.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from sidecoach.events import Region
from sidecoach.vision.board_detect import BoardCandidate

GLYPH_H = 20
GLYPH_W = 16
TEMPLATE_FILE = Path(__file__).with_name("clock_templates.npz")

# How far from the board's edge a clock box may sit, in cells.
SEARCH_CELLS = 1.3
# A box is a run of near-constant colour along the column just inside its right edge.
BOX_HEIGHT_CELLS = (0.15, 0.9)
BOX_WIDTH_CELLS = (0.6, 3.5)
# The box interior is flat colour; a dark box on a dark page differs from it by only ten
# grey levels, so the tolerances stay well under that.
ROW_STEP_TOLERANCE = 4.0
FILL_TOLERANCE = 4.0
MIN_CONTRAST = 60.0
INK_ON = 0.35
# A glyph run wider than this fraction of the line height holds two touching glyphs.
MAX_GLYPH_ASPECT = 0.85

_CLOCK_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d\d)(?:\.(\d))?$")

Which = Literal["top", "bottom"]


@dataclass(frozen=True)
class ClockReading:
    top_ms: int | None  # what the clock above the board shows, in milliseconds
    bottom_ms: int | None  # what the clock below the board shows
    running: Which | None  # which one is ticking, if either
    confidence: float  # 0..1, how sure the digits were read
    # Which box has the light background: white's clock, on every chess.com theme seen.
    light: Which | None = None


@dataclass(frozen=True)
class ClockBox:
    """Where a clock box was found. Frame coordinates, `right` exclusive."""

    top: int
    bottom: int
    left: int
    right: int
    background: float
    text: float

    @property
    def light(self) -> bool:
        return self.background > self.text

    @property
    def region(self) -> Region:
        return Region(self.left, self.top, self.right - self.left, self.bottom - self.top)


@dataclass(frozen=True)
class Segmentation:
    """The glyphs of one box: column spans inside the ink map, the text line's rows, and the
    column spans of whatever sits left of the digits (the clock icon)."""

    glyphs: list[tuple[int, int]]
    line: tuple[int, int]
    icon: list[tuple[int, int]]


def _runs(on: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True in a 1D boolean array as [start, end) spans."""
    padded = np.concatenate(([False], on, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]


def _fill_extent(row: np.ndarray, start: int, background: float) -> int:
    """Leftmost column, scanning left from `start`, still within tolerance of the colour."""
    inside = np.abs(row[: start + 1] - background) <= FILL_TOLERANCE
    off = np.flatnonzero(~inside)
    return int(off[-1]) + 1 if len(off) else 0


def find_box(
    gray: np.ndarray, x_right: float, rows: tuple[int, int], cell: float
) -> ClockBox | None:
    """The clock box whose right edge is at `x_right`, between the given rows, if any.

    The column band just inside the right edge is the box's padding: along it, the box is a
    run of near-constant colour of plausible height, bounded by the page above and below.
    The run must extend left far enough to be a box, and must hold high-contrast ink of a
    plausible amount, the digits. Among candidates the one with the most ink wins.
    """
    height, width = gray.shape[:2]
    r0, r1 = max(0, rows[0]), min(height, rows[1])
    c0 = int(round(x_right - 0.10 * cell))
    c1 = int(round(x_right - 0.03 * cell))
    if c1 - c0 < 1 or c0 < 0 or c1 > width or r1 - r0 < 4:
        return None
    band = gray[r0:r1, c0:c1].astype(np.float32)
    profile = np.median(band, axis=1)  # the median ignores a rounded corner in one column
    steady = np.abs(np.diff(profile)) <= ROW_STEP_TOLERANCE
    # steady[i] says rows i and i + 1 match, so a run of it from s to e - 1 is the row
    # span [s, e + 1) of one flat colour.
    best: tuple[float, ClockBox] | None = None
    for s_, e_ in _runs(steady):
        a, b = s_, e_ + 1
        span = b - a
        if not BOX_HEIGHT_CELLS[0] * cell <= span <= BOX_HEIGHT_CELLS[1] * cell:
            continue
        background = float(np.median(profile[a:b]))
        top, bottom = r0 + a, r0 + b
        # The leftmost row reaching furthest left is the box's edge: the icon and the
        # digits stop the scan early on the rows they cross, and no row can run past the
        # edge, where the page is a different colour.
        left = min(
            _fill_extent(gray[row].astype(np.float32), c1 - 1, background)
            for row in range(top, bottom)
        )
        box_width = c1 - left
        if not BOX_WIDTH_CELLS[0] * cell <= box_width <= BOX_WIDTH_CELLS[1] * cell:
            continue
        patch = gray[top:bottom, left:c1].astype(np.float32)
        text = float(np.percentile(patch, 1 if background > 127 else 99))
        if abs(text - background) < MIN_CONTRAST:
            continue
        ink = np.abs(patch - background) > 0.5 * abs(text - background)
        fraction = float(ink.mean())
        if not 0.01 < fraction < 0.4:
            continue
        score = float(ink.sum())
        if best is None or score > best[0]:
            best = (score, ClockBox(top, bottom, left, int(round(x_right)), background, text))
    return best[1] if best else None


def ink_map(gray: np.ndarray, box: ClockBox, cell: float) -> np.ndarray:
    """Text intensity inside the box, 0 for background to 1 for the text colour."""
    right = int(round(box.right - 0.03 * cell))
    patch = gray[box.top : box.bottom, box.left : right].astype(np.float32)
    return np.clip((patch - box.background) / (box.text - box.background), 0.0, 1.0)


def segment(ink: np.ndarray) -> Segmentation | None:
    """Cut the ink map into glyph column spans.

    The digits are the rightmost group of ink columns; a gap wider than half the box height
    separates them from the icon. Within the group, the text line's rows are where the
    ink is, and glyph spans are runs of columns holding at least half a pixel of ink over
    those rows. Anti-aliasing can close the one-pixel gap between two round digits, so a
    run wider than a digit is split at its weakest interior column.
    """
    box_h = ink.shape[0]
    on = ink.max(axis=0) > INK_ON
    runs = _runs(on)
    if not runs:
        return None
    group = [runs[-1]]
    for a, b in reversed(runs[:-1]):
        if group[0][0] - b > 0.45 * box_h:
            break
        group.insert(0, (a, b))
    icon = runs[: len(runs) - len(group)]
    g0, g1 = group[0][0], group[-1][1]
    line_rows = np.flatnonzero(ink[:, g0:g1].max(axis=1) > INK_ON)
    t0, t1 = int(line_rows[0]), int(line_rows[-1]) + 1
    line_h = t1 - t0
    if line_h < 4:
        return None
    colsum = ink[t0:t1].sum(axis=0)
    strong = colsum > 0.5
    strong[:g0] = False
    strong[g1:] = False
    glyphs: list[tuple[int, int]] = []
    pending = _runs(strong)
    while pending:
        a, b = pending.pop(0)
        if b - a > MAX_GLYPH_ASPECT * line_h:
            lo = a + int(round(0.3 * line_h))
            hi = b - int(round(0.3 * line_h))
            if hi - lo >= 1:
                k = lo + int(np.argmin(colsum[lo:hi]))
                pending.insert(0, (k, b))
                pending.insert(0, (a, k))
                continue
        glyphs.append((a, b))
    return Segmentation(glyphs, (t0, t1), icon)


def glyph_canvas(ink: np.ndarray, line: tuple[int, int], span: tuple[int, int]) -> np.ndarray:
    """One glyph scaled to the fixed line height and centred on the canvas.

    The width scales with the height, so a narrow 1 stays narrow and a colon stays a pair
    of dots; the rows are the whole text line, so a dot stays at the baseline.
    """
    t0, t1 = line
    a, b = span
    scale = GLYPH_H / (t1 - t0)
    w = min(GLYPH_W, max(1, int(round((b - a) * scale))))
    crop = ink[t0:t1, a:b]
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (w, GLYPH_H), interpolation=interpolation)
    canvas = np.zeros((GLYPH_H, GLYPH_W), dtype=np.float32)
    off = (GLYPH_W - w) // 2
    canvas[:, off : off + w] = resized
    return canvas


def _normalise(canvases: np.ndarray) -> np.ndarray:
    flat = canvases.reshape(len(canvases), -1).astype(np.float32)
    flat = flat - flat.mean(axis=1, keepdims=True)
    return flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-6)


@dataclass(frozen=True)
class GlyphTemplates:
    """Mean glyph canvases per label, several per label when the rendering varies (white
    text on a dark box is bolder than dark text on a light one)."""

    labels: tuple[str, ...]
    canvases: np.ndarray  # (N, GLYPH_H, GLYPH_W) float32 in 0..1
    advances: np.ndarray  # (N,) glyph width as a fraction of the line height
    _normalised: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_normalised", _normalise(self.canvases))

    @classmethod
    def load(cls, path: Path | None = None) -> GlyphTemplates:
        with np.load(path or TEMPLATE_FILE) as data:
            return cls(
                tuple(str(label) for label in data["labels"]),
                data["canvases"].astype(np.float32) / 255.0,
                data["advances"].astype(np.float32),
            )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            labels=np.array(self.labels),
            canvases=np.clip(np.rint(self.canvases * 255), 0, 255).astype(np.uint8),
            advances=self.advances.astype(np.float32),
        )

    def canvas_for(self, label: str, light: bool) -> tuple[np.ndarray, float]:
        """A template canvas and its advance for `label`; the first stored variant of the
        wanted polarity when the file holds both, else any variant."""
        indices = [i for i, name in enumerate(self.labels) if name == label]
        if not indices:
            raise KeyError(label)
        half = len(indices) // 2
        pick = indices[0] if light or half == 0 else indices[half]
        return self.canvases[pick], float(self.advances[pick])

    def classify(self, canvas: np.ndarray) -> tuple[str, float, float]:
        """(label, correlation with the best template, margin over the best other label)."""
        flat = _normalise(canvas[None])[0]
        scores = self._normalised @ flat
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        label = self.labels[best]
        margin = 1.0
        for i in order[1:]:
            if self.labels[int(i)] != label:
                margin = float(scores[best] - scores[int(i)])
                break
        return label, float(scores[best]), margin


def parse_clock(text: str) -> int | None:
    """Milliseconds shown by a clock string such as 15:00, 1:02:03 or 0:59.9; None if it
    is not a clock.

    A tenths dot read as a colon is repaired: seconds always have two digits, so a lone
    digit after the last separator is tenths whichever glyph was matched.
    """
    if text.count(":") > 1 and len(text.rsplit(":", 1)[1]) == 1:
        head, tail = text.rsplit(":", 1)
        text = f"{head}.{tail}"
    match = _CLOCK_RE.match(text)
    if not match:
        return None
    hours, minutes, seconds, tenths = match.groups()
    if int(seconds) >= 60 or (hours is not None and int(minutes) >= 60):
        return None
    total = int(minutes) * 60 + int(seconds)
    if hours is not None:
        total += int(hours) * 3600
    return total * 1000 + (int(tenths) * 100 if tenths else 0)


@dataclass(frozen=True)
class BoxReading:
    """One clock box read: the text, its value, and how sure the glyphs were."""

    box: ClockBox
    text: str
    ms: int | None
    confidence: float
    glyphs: list[tuple[str, float, float]]
    icon: Region | None


def _glyph_confidence(score: float, margin: float) -> float:
    return float(np.clip(min((score - 0.5) / 0.3, margin / 0.12), 0.0, 1.0))


def read_box(
    gray: np.ndarray, box: ClockBox, cell: float, templates: GlyphTemplates
) -> BoxReading | None:
    ink = ink_map(gray, box, cell)
    seg = segment(ink)
    if seg is None:
        return None
    glyphs = [templates.classify(glyph_canvas(ink, seg.line, span)) for span in seg.glyphs]
    text = "".join(label for label, _, _ in glyphs)
    confidence = min(_glyph_confidence(score, margin) for _, score, margin in glyphs)
    icon = None
    if seg.icon:
        a, b = seg.icon[0][0], seg.icon[-1][1]
        icon = Region(box.left + a, box.top, b - a, box.bottom - box.top)
    return BoxReading(box, text, parse_clock(text), confidence, glyphs, icon)


@dataclass
class _BoxState:
    ms: int | None = None
    decreased: float = -np.inf
    increased: float = -np.inf
    icon_moved: float = -np.inf
    icon: np.ndarray | None = None


def _icon_changed(previous: np.ndarray, current: np.ndarray) -> bool:
    """Whether the icon patch changed by more than a pixel's jitter, tolerating a one
    pixel shift of the located box."""
    if previous.shape != current.shape or min(previous.shape) < 4:
        return False
    best = np.inf
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            a = previous[1 + dy : -1 + dy or None, 1 + dx : -1 + dx or None]
            b = current[1:-1, 1:-1]
            diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
            best = min(best, float(np.count_nonzero(diff > 24)))
    return best >= 3


class ClockReader:
    """Reads both clocks near a tracked board, frame by frame.

    `running` is decided from motion, never from the box colours, which are the sides. The
    running clock's icon is a face whose hand sweeps round, so its pixels change from frame
    to frame while the idle icon's never do; and the running clock's value falls once a
    second, while a clock that just rose has taken its increment and stopped, which makes
    the other one the running clock. Evidence expires after `stale_after_s`, so a game that
    has not begun, or that has ended, reports no running clock.

    Whose turn it is comes from the board, not from here. This is a reading for the record
    and for cross-checking, and nothing in the game state depends on it.

    A reader without templates (the file is missing or unreadable) reads nothing rather
    than raising, so a broken install costs the clocks and not the session.
    """

    def __init__(
        self,
        templates: GlyphTemplates | None = None,
        *,
        min_confidence: float = 0.35,
        stale_after_s: float = 3.0,
        handover_holds_s: float = 1.2,
    ) -> None:
        if templates is None:
            try:
                templates = GlyphTemplates.load()
            except (OSError, ValueError, KeyError):
                templates = None
        self.templates = templates
        self.min_confidence = min_confidence
        self.stale_after_s = stale_after_s
        self.handover_holds_s = handover_holds_s
        self.last: dict[Which, BoxReading | None] = {"top": None, "bottom": None}
        self._state: dict[Which, _BoxState] = {"top": _BoxState(), "bottom": _BoxState()}
        self._calls = 0

    def reset(self) -> None:
        self._state = {"top": _BoxState(), "bottom": _BoxState()}
        self.last = {"top": None, "bottom": None}

    def read(
        self, frame: np.ndarray, board: BoardCandidate, ts: float | None = None
    ) -> ClockReading | None:
        """Both clocks for this frame, or None when no clock is drawn near the board (an
        untimed game, a lobby, the board scrolled so the clocks are off screen). Never
        raises on a bad frame."""
        try:
            return self._read(frame, board, ts)
        except Exception:  # noqa: BLE001 - a bad frame must never take the session down
            return None

    def _read(
        self, frame: np.ndarray, board: BoardCandidate, ts: float | None
    ) -> ClockReading | None:
        if self.templates is None or frame.ndim not in (2, 3) or frame.size == 0:
            return None
        self._calls += 1
        now = ts if ts is not None else self._calls * 0.1
        x, y, w, h = board.region
        cell = board.cell_px
        if not 4.0 <= cell <= 4096.0 or w <= 0 or h <= 0:
            return None
        reach = int(round(SEARCH_CELLS * cell))
        readings: dict[Which, BoxReading | None] = {
            "top": self._read_side(frame, x + w, (y - reach, y - 1), cell),
            "bottom": self._read_side(frame, x + w, (y + h + 1, y + h + reach), cell),
        }
        self.last = readings
        if all(r is None for r in readings.values()):
            return None
        for which, reading in readings.items():
            self._observe(which, reading, frame, now)
        running = self._running(now)
        light: Which | None = None
        top, bottom = readings["top"], readings["bottom"]
        if top is not None and bottom is not None and top.box.light != bottom.box.light:
            light = "top" if top.box.light else "bottom"
        elif top is not None and bottom is None:
            light = "top" if top.box.light else None
        elif bottom is not None and top is None:
            light = "bottom" if bottom.box.light else None
        present = [r for r in readings.values() if r is not None]
        confidence = min(r.confidence for r in present)
        return ClockReading(
            top_ms=self._value(top),
            bottom_ms=self._value(bottom),
            running=running,
            confidence=confidence,
            light=light,
        )

    def _value(self, reading: BoxReading | None) -> int | None:
        if reading is None or reading.confidence < self.min_confidence:
            return None
        return reading.ms

    def _read_side(
        self, frame: np.ndarray, x_right: int, rows: tuple[int, int], cell: float
    ) -> BoxReading | None:
        height, width = frame.shape[:2]
        r0, r1 = max(0, rows[0]), min(height, rows[1])
        c0 = max(0, int(x_right - BOX_WIDTH_CELLS[1] * cell) - 2)
        c1 = min(width, int(round(x_right)) + 1)
        if r1 - r0 < 4 or c1 - c0 < 4:
            return None
        assert self.templates is not None
        strip = np.ascontiguousarray(frame[r0:r1, c0:c1])
        if strip.dtype != np.uint8:
            strip = np.clip(strip, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.ndim == 3 else strip
        box = find_box(gray, x_right - c0, (0, r1 - r0), cell)
        if box is None:
            return None
        reading = read_box(gray, box, cell, self.templates)
        if reading is None:
            return None
        shifted = ClockBox(
            box.top + r0, box.bottom + r0, box.left + c0, box.right + c0, box.background, box.text
        )
        icon = reading.icon
        if icon is not None:
            icon = Region(icon.x + c0, icon.y + r0, icon.w, icon.h)
        return BoxReading(
            shifted, reading.text, reading.ms, reading.confidence, reading.glyphs, icon
        )

    def _observe(
        self, which: Which, reading: BoxReading | None, frame: np.ndarray, now: float
    ) -> None:
        state = self._state[which]
        if reading is None:
            state.icon = None
            return
        ms = self._value(reading)
        if ms is not None:
            if state.ms is not None:
                if ms < state.ms:
                    state.decreased = now
                elif ms > state.ms:
                    state.increased = now
            state.ms = ms
        icon = reading.icon
        if icon is None or icon.w < 3 or icon.h < 3:
            state.icon = None
            return
        patch = np.ascontiguousarray(frame[icon.y : icon.y + icon.h, icon.x : icon.x + icon.w])
        if patch.dtype != np.uint8:
            patch = np.clip(patch, 0, 255).astype(np.uint8)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch.copy()
        if state.icon is not None and _icon_changed(state.icon, patch):
            state.icon_moved = now
        state.icon = patch

    def _running(self, now: float) -> Which | None:
        top, bottom = self._state["top"], self._state["bottom"]
        # A clock that rose took its increment: that side has just moved and stopped, and
        # the other one started. For a moment after, the stopped clock's icon is still
        # settling, so the handover outranks the icons rather than competing with them.
        handover = max(top.increased, bottom.increased)
        if now - handover <= self.handover_holds_s:
            return "bottom" if top.increased >= bottom.increased else "top"
        evidence = {
            "top": max(top.decreased, top.icon_moved, bottom.increased),
            "bottom": max(bottom.decreased, bottom.icon_moved, top.increased),
        }
        which = max(evidence, key=lambda k: evidence[k])
        if now - evidence[which] > self.stale_after_s:
            return None
        return which  # type: ignore[return-value]
