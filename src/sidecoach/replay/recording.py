"""Record and replay screen frames.

Layout of a recording directory:

    meta.json     monitor, region, fps, start time, frame size, codec
    frames.mkv    every stored frame, FFV1 lossless, written and read through OpenCV
    index.jsonl   one line per capture tick: frame_id, ts, stored (video index or null)

Only frames that differ from the last stored frame go into the video, so a quiet position
costs nothing. On a busy desktop something changes every tick, so the difference is measured
inside `change_region` when one is given, normally the board plus a margin. A keyframe is
stored every `keyframe_interval_s` regardless, which keeps clocks and move lists sampled at
their natural one-second rate. Replay yields every tick and repeats the last stored frame for
unchanged ticks, which is exactly what the live pipeline saw.

A hard-killed session still replays: a partial final index line is skipped, and a video that
ends before the index does replays as far as it goes. Both log a warning on this module's
logger; without logging configured, Python prints warnings to stderr.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from sidecoach.events import Region
from sidecoach.vision.board_detect import BoardCandidate

FOURCC = "FFV1"
META_FILE = "meta.json"
VIDEO_FILE = "frames.mkv"
INDEX_FILE = "index.jsonl"

log = logging.getLogger(__name__)


@dataclass
class RecordingMeta:
    monitor: int
    region: tuple[int, int, int, int]
    fps: int
    started_at: str
    width: int
    height: int
    codec: str = FOURCC
    version: int = 1
    change_region: tuple[int, int, int, int] | None = None
    # Frames are the capture region resized by this factor (1.0 for a live capture).
    scale: float = 1.0
    keyframe_interval_s: float | None = None


@dataclass(frozen=True)
class Tick:
    frame_id: int
    ts: float
    stored: int | None


def frames_differ(
    a: np.ndarray,
    b: np.ndarray,
    threshold: int = 12,
    region: Region | None = None,
    min_fraction: float = 0.0,
) -> bool:
    """True when more than `min_fraction` of the compared pixels differ by over `threshold`.

    Compared inside `region` when given. A moved piece changes at least two cells, an
    animated avatar or a cursor twitch changes a few dozen pixels; the fraction separates them.
    """
    if region is not None:
        a = a[region.y : region.y + region.h, region.x : region.x + region.w]
        b = b[region.y : region.y + region.h, region.x : region.x + region.w]
    if a.size == 0:
        return False
    diff = cv2.absdiff(a, b)
    if diff.ndim == 3:
        diff = diff.max(axis=2)
    changed = int(np.count_nonzero(diff > threshold))
    return changed > min_fraction * diff.size


def clamp_region(region: Region, width: int, height: int) -> Region:
    x0, y0 = max(0, region.x), max(0, region.y)
    x1, y1 = min(width, region.x + region.w), min(height, region.y + region.h)
    return Region(x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def board_gate(board: BoardCandidate | None, width: int, height: int) -> Region | None:
    """The board plus half a cell of margin, clamped to the frame; None without a board."""
    if board is None:
        return None
    pad = int(round(board.cell_px / 2))
    r = board.region
    gate = clamp_region(Region(r.x - pad, r.y - pad, r.w + 2 * pad, r.h + 2 * pad), width, height)
    if gate.w == 0 or gate.h == 0:
        return None  # the board lies off the frame; gate on the whole frame instead
    return gate


class ChangeGate:
    """Decides which frames count as changed: the one rule for recording and for replay.

    A frame is a change when more than `min_fraction` of the gated region differs from
    the last changed frame by over `threshold`, or when a keyframe is due. The live
    watcher without a recorder and the recorder itself share this object's logic, so a
    session replays with the same `changed` stream it was seen with.

    `due` asks and `commit` moves the baseline; `changed` does both. The recorder keeps them
    apart so that a frame the video refused never becomes the baseline.
    """

    def __init__(
        self,
        *,
        threshold: int = 12,
        min_fraction: float = 0.0005,
        keyframe_interval_s: float | None = 1.0,
        region: Region | None = None,
    ) -> None:
        self.threshold = threshold
        self.min_fraction = min_fraction
        self.keyframe_interval_s = keyframe_interval_s
        self.region = region
        self._last: np.ndarray | None = None
        self._last_ts = -1e9

    def due(self, frame: np.ndarray, ts: float) -> bool:
        """Would `frame` count as a change? The baseline is left where it is."""
        if self._last is None or self._last.shape != frame.shape:
            return True
        region = clamp_region(self.region, frame.shape[1], frame.shape[0]) if self.region else None
        if region is not None and (region.w == 0 or region.h == 0):
            region = None  # an empty gate would store nothing; compare the whole frame
        return frames_differ(self._last, frame, self.threshold, region, self.min_fraction) or (
            self.keyframe_interval_s is not None and ts - self._last_ts >= self.keyframe_interval_s
        )

    def commit(self, frame: np.ndarray, ts: float) -> None:
        """Make `frame` at `ts` the baseline later frames are compared against."""
        self._last = frame.copy()
        self._last_ts = ts

    def changed(self, frame: np.ndarray, ts: float) -> bool:
        due = self.due(frame, ts)
        if due:
            self.commit(frame, ts)
        return due


class Recorder:
    def __init__(
        self,
        out_dir: Path,
        *,
        monitor: int,
        region: Region,
        fps: int,
        change_threshold: int = 12,
        change_region: Region | None = None,
        keyframe_interval_s: float | None = 1.0,
        min_changed_fraction: float = 0.0005,
        scale: float = 1.0,
        size: tuple[int, int] | None = None,
    ) -> None:
        """`size` is the stored frame size when it differs from the region (a shrunk copy).

        `change_threshold`, `min_changed_fraction`, `keyframe_interval_s` and `change_region`
        configure the `ChangeGate` that picks the stored frames; `change_region` can be moved
        while recording through the property of the same name.
        """
        width, height = size if size is not None else (region.w, region.h)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.dir = out_dir
        gate_region = clamp_region(change_region, width, height) if change_region else None
        self._gate = ChangeGate(
            threshold=change_threshold,
            min_fraction=min_changed_fraction,
            keyframe_interval_s=keyframe_interval_s,
            region=gate_region,
        )
        self.meta = RecordingMeta(
            monitor=monitor,
            region=(region.x, region.y, region.w, region.h),
            fps=fps,
            started_at=datetime.now(UTC).isoformat(timespec="seconds"),
            width=width,
            height=height,
            change_region=(
                (gate_region.x, gate_region.y, gate_region.w, gate_region.h)
                if gate_region
                else None
            ),
            keyframe_interval_s=keyframe_interval_s,
            scale=scale,
        )
        (out_dir / META_FILE).write_text(json.dumps(asdict(self.meta), indent=2))
        self._writer = cv2.VideoWriter(
            str(out_dir / VIDEO_FILE),
            cv2.VideoWriter.fourcc(*FOURCC),
            float(fps),
            (width, height),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open {FOURCC} video writer at {out_dir / VIDEO_FILE}")
        self._index = (out_dir / INDEX_FILE).open("w", encoding="utf-8")
        self.ticks = 0
        self.stored = 0

    @property
    def change_region(self) -> Region | None:
        """Where the gate looks for changes; the live watcher moves it as the board moves."""
        return self._gate.region

    @change_region.setter
    def change_region(self, region: Region | None) -> None:
        self._gate.region = region

    def write(
        self,
        frame: np.ndarray,
        ts: float,
        *,
        store: bool | None = None,
        frame_id: int | None = None,
    ) -> Tick:
        """Record one tick. The gate decides whether the frame is stored unless `store` says.

        The tick id is the tick count unless `frame_id` is given; a copy of a recording passes
        the source's ids through.
        """
        if frame.shape[:2] != (self.meta.height, self.meta.width):
            raise ValueError(
                f"frame is {frame.shape[1]}x{frame.shape[0]}, "
                f"recording is {self.meta.width}x{self.meta.height}"
            )
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frames must be 8-bit BGR; the writer silently drops anything else")
        stored: int | None = None
        due = self._gate.due(frame, ts) if store is None else store
        if due:
            if not self._writer.write(frame):
                # A tick that claims a stored frame the video does not hold would shift
                # every later tick onto the wrong frame; better to stop the recording.
                raise RuntimeError(f"video writer refused frame {self.stored} of {self.dir}")
            # The baseline moves only once the video holds the frame, so a caller that goes
            # on after a refused write still compares against the last frame on disk.
            self._gate.commit(frame, ts)
            stored = self.stored
            self.stored += 1
        tick = Tick(self.ticks if frame_id is None else frame_id, ts, stored)
        self.ticks += 1
        self._index.write(json.dumps(asdict(tick)) + "\n")
        self._index.flush()  # a hard kill then loses frames, not the whole index
        return tick

    def close(self) -> None:
        self._writer.release()
        self._index.close()

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.dir.iterdir() if p.is_file())


def _read_index(path: Path) -> list[Tick]:
    """The ticks in an index file. A partial final line, left by a hard kill, is skipped."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ticks: list[Tick] = []
    for n, line in enumerate(lines, start=1):
        try:
            ticks.append(Tick(**json.loads(line)))
        except (ValueError, TypeError) as exc:
            if n == len(lines):
                log.warning(
                    "%s: skipped the incomplete final line %d (hard-killed session?)", path, n
                )
                break
            raise ValueError(f"{path}: corrupt index line {n}: {line[:80]!r}") from exc
    return ticks


class Recording:
    def __init__(self, path: Path) -> None:
        self.dir = path
        raw = json.loads((path / META_FILE).read_text())
        raw["region"] = tuple(raw["region"])
        if raw.get("change_region") is not None:
            raw["change_region"] = tuple(raw["change_region"])
        self.meta = RecordingMeta(**raw)
        self.ticks: list[Tick] = _read_index(path / INDEX_FILE)

    @property
    def region(self) -> Region:
        return Region(*self.meta.region)

    @property
    def stored_count(self) -> int:
        return sum(1 for t in self.ticks if t.stored is not None)

    def frames(self) -> Iterator[tuple[Tick, np.ndarray]]:
        """Every tick with the frame the pipeline saw at that tick.

        When the video ends before the index does (a hard-killed session), the ticks from the
        first missing frame on are dropped with a logged warning, and the iteration ends.
        """
        capture = cv2.VideoCapture(str(self.dir / VIDEO_FILE))
        try:
            last: np.ndarray | None = None
            for n, tick in enumerate(self.ticks):
                if tick.stored is not None:
                    ok, frame = capture.read()
                    if not ok:
                        log.warning(
                            "%s: video ends before stored frame %d; %d of %d ticks dropped",
                            self.dir,
                            tick.stored,
                            len(self.ticks) - n,
                            len(self.ticks),
                        )
                        return
                    last = frame
                if last is None:
                    raise RuntimeError("index references a frame before any was stored")
                yield tick, last
        finally:
            capture.release()


def _copy_ticks(
    src: Recording,
    recorder: Recorder,
    first: int | None,
    last: int | None,
    *,
    keep_ids: bool,
) -> None:
    """Write the ticks of `src` with ids in [first, last] to `recorder`.

    A frame is stored where the source stored one, so the copy repeats frames exactly where
    the source did. The first copied tick is stored whatever the source did: the copy has no
    earlier frame for it to repeat. Frames are resized to the recorder's size when it differs.
    """
    size = (recorder.meta.width, recorder.meta.height)
    fresh = True
    for tick, frame in src.frames():
        if first is not None and tick.frame_id < first:
            continue
        if last is not None and tick.frame_id > last:
            break
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        recorder.write(
            frame,
            tick.ts,
            store=fresh or tick.stored is not None,
            frame_id=tick.frame_id if keep_ids else None,
        )
        fresh = False


def _frame_format(rec: Recording) -> str:
    return f"{rec.meta.width}x{rec.meta.height} at scale {rec.meta.scale:g}"


def concat_recordings(parts: list[tuple[Path, int | None, int | None]], out_dir: Path) -> Recording:
    """Join tick ranges of several recordings into one new recording.

    Each part is (directory, first tick, last tick), with None meaning the whole span. Frames
    are re-encoded losslessly and stored where the source stored them, plus at the first tick
    of every part, so a tick that repeated the previous frame still does. Timestamps are kept,
    so a gap between recordings stays a gap; ticks are renumbered from 0, since parts may
    overlap or come from different recordings. All parts must share the frame size and scale
    of the first (shrink them alike first).
    """
    first = Recording(parts[0][0])
    recorder = Recorder(
        out_dir,
        monitor=first.meta.monitor,
        region=first.region,
        fps=first.meta.fps,
        keyframe_interval_s=first.meta.keyframe_interval_s,
        scale=first.meta.scale,
        size=(first.meta.width, first.meta.height),
    )
    try:
        for path, start, end in parts:
            rec = Recording(path)
            if _frame_format(rec) != _frame_format(first):
                raise ValueError(
                    f"{path} is {_frame_format(rec)} but {parts[0][0]} is "
                    f"{_frame_format(first)}; parts must share frame size and scale"
                )
            _copy_ticks(rec, recorder, start, end, keep_ids=False)
    finally:
        recorder.close()
    return Recording(out_dir)


def shrink_recording(
    src: Path,
    out_dir: Path,
    scale: float,
    first: int | None = None,
    last: int | None = None,
) -> Recording:
    """A copy of `src` with every frame resized by `scale`, still lossless.

    The whole screen is kept: the program's input is the screen, and a recording that held
    less would test an easier problem than the live one. Timestamps, tick ids and the stored
    flags are copied from the source, so a tick that repeated the previous frame still does,
    and a copy trimmed with `first`/`last` keeps the source's tick numbers for `cells` and
    `replay` (its first tick is always stored). The capture region stays in screen
    coordinates and the scale is recorded relative to the screen, so a pixel maps back to
    the screen and shrinking a shrunk copy compounds the scale. Resizing uses area
    averaging, which is what a smaller screen shows.
    """
    rec = Recording(src)
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"scale must be in (0, 1], got {scale}")
    width = max(2, round(rec.meta.width * scale))
    height = max(2, round(rec.meta.height * scale))
    recorder = Recorder(
        out_dir,
        monitor=rec.meta.monitor,
        region=rec.region,
        fps=rec.meta.fps,
        keyframe_interval_s=rec.meta.keyframe_interval_s,
        scale=scale * rec.meta.scale,
        size=(width, height),
    )
    try:
        _copy_ticks(rec, recorder, first, last, keep_ids=True)
    finally:
        recorder.close()
    return Recording(out_dir)
