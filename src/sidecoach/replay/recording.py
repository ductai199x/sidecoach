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
"""

from __future__ import annotations

import json
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

    def changed(self, frame: np.ndarray, ts: float) -> bool:
        if self._last is None or self._last.shape != frame.shape:
            due = True
        else:
            region = (
                clamp_region(self.region, frame.shape[1], frame.shape[0]) if self.region else None
            )
            if region is not None and (region.w == 0 or region.h == 0):
                region = None  # an empty gate would store nothing; compare the whole frame
            due = frames_differ(self._last, frame, self.threshold, region, self.min_fraction) or (
                self.keyframe_interval_s is not None
                and ts - self._last_ts >= self.keyframe_interval_s
            )
        if due:
            self._last = frame.copy()
            self._last_ts = ts
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
        """`size` is the stored frame size when it differs from the region (a shrunk copy)."""
        width, height = size if size is not None else (region.w, region.h)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.dir = out_dir
        self.change_threshold = change_threshold
        self.min_changed_fraction = min_changed_fraction
        self.change_region = (
            clamp_region(change_region, region.w, region.h) if change_region else None
        )
        self.keyframe_interval_s = keyframe_interval_s
        self.meta = RecordingMeta(
            monitor=monitor,
            region=(region.x, region.y, region.w, region.h),
            fps=fps,
            started_at=datetime.now(UTC).isoformat(timespec="seconds"),
            width=width,
            height=height,
            change_region=(
                (
                    self.change_region.x,
                    self.change_region.y,
                    self.change_region.w,
                    self.change_region.h,
                )
                if self.change_region
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
        self._gate = ChangeGate(
            threshold=change_threshold,
            min_fraction=min_changed_fraction,
            keyframe_interval_s=keyframe_interval_s,
            region=self.change_region,
        )
        self._last: np.ndarray | None = None
        self._last_stored_ts: float | None = None
        self.ticks = 0
        self.stored = 0

    def _should_store(self, frame: np.ndarray, ts: float) -> bool:
        self._gate.region = self.change_region
        return self._gate.changed(frame, ts)

    def write(self, frame: np.ndarray, ts: float) -> Tick:
        if frame.shape[:2] != (self.meta.height, self.meta.width):
            raise ValueError(
                f"frame is {frame.shape[1]}x{frame.shape[0]}, "
                f"recording is {self.meta.width}x{self.meta.height}"
            )
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frames must be 8-bit BGR; the writer silently drops anything else")
        stored: int | None = None
        if self._should_store(frame, ts):
            if not self._writer.write(frame):
                # A tick that claims a stored frame the video does not hold would shift
                # every later tick onto the wrong frame; better to stop the recording.
                raise RuntimeError(f"video writer refused frame {self.stored} of {self.dir}")
            stored = self.stored
            self.stored += 1
        tick = Tick(self.ticks, ts, stored)
        self.ticks += 1
        self._index.write(json.dumps(asdict(tick)) + "\n")
        self._index.flush()  # a hard kill then loses frames, not the whole index
        return tick

    def close(self) -> None:
        self._writer.release()
        self._index.close()

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.dir.iterdir() if p.is_file())


class Recording:
    def __init__(self, path: Path) -> None:
        self.dir = path
        raw = json.loads((path / META_FILE).read_text())
        raw["region"] = tuple(raw["region"])
        if raw.get("change_region") is not None:
            raw["change_region"] = tuple(raw["change_region"])
        self.meta = RecordingMeta(**raw)
        self.ticks: list[Tick] = [
            Tick(**json.loads(line))
            for line in (path / INDEX_FILE).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @property
    def region(self) -> Region:
        return Region(*self.meta.region)

    @property
    def stored_count(self) -> int:
        return sum(1 for t in self.ticks if t.stored is not None)

    def frames(self) -> Iterator[tuple[Tick, np.ndarray]]:
        """Every tick with the frame the pipeline saw at that tick."""
        capture = cv2.VideoCapture(str(self.dir / VIDEO_FILE))
        try:
            last: np.ndarray | None = None
            for tick in self.ticks:
                if tick.stored is not None:
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"video ended before stored frame {tick.stored}")
                    last = frame
                if last is None:
                    raise RuntimeError("index references a frame before any was stored")
                yield tick, last
        finally:
            capture.release()

    def stored_frames(self) -> Iterator[tuple[Tick, np.ndarray]]:
        """Only the ticks whose frame changed."""
        for tick, frame in self.frames():
            if tick.stored is not None:
                yield tick, frame


def concat_recordings(parts: list[tuple[Path, int | None, int | None]], out_dir: Path) -> Recording:
    """Join tick ranges of several recordings into one new recording.

    Each part is (directory, first tick, last tick), with None meaning the whole span. Frames
    are re-encoded losslessly; timestamps are kept, so a gap between recordings stays a gap.
    """
    first = Recording(parts[0][0])
    recorder = Recorder(
        out_dir,
        monitor=first.meta.monitor,
        region=first.region,
        fps=first.meta.fps,
        change_region=None,
        keyframe_interval_s=first.meta.keyframe_interval_s,
        min_changed_fraction=0.0,
    )
    try:
        for path, start, end in parts:
            rec = Recording(path)
            if (rec.meta.width, rec.meta.height) != (first.meta.width, first.meta.height):
                raise ValueError(f"{path} has a different frame size than {parts[0][0]}")
            for tick, frame in rec.frames():
                if start is not None and tick.frame_id < start:
                    continue
                if end is not None and tick.frame_id > end:
                    break
                recorder.write(frame, tick.ts)
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
    less would test an easier problem than the live one. Timestamps and the change gating
    of the source are kept, so a tick that repeated the previous frame still does. The
    capture region stays in screen coordinates and the scale is recorded, so a pixel maps
    back to the screen. Resizing uses area averaging, which is what a smaller screen shows.
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
        change_region=None,
        keyframe_interval_s=rec.meta.keyframe_interval_s,
        min_changed_fraction=0.0,
        scale=scale * rec.meta.scale,
        size=(width, height),
    )
    try:
        for tick, frame in rec.frames():
            if first is not None and tick.frame_id < first:
                continue
            if last is not None and tick.frame_id > last:
                break
            if (frame.shape[1], frame.shape[0]) != (width, height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            recorder.write(frame, tick.ts)
    finally:
        recorder.close()
    return Recording(out_dir)
