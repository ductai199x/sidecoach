import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from sidecoach.events import MovePlayed, Region
from sidecoach.replay.recording import (
    Recorder,
    Recording,
    concat_recordings,
    frames_differ,
    shrink_recording,
)

pytestmark = pytest.mark.vision  # the synthetic vision pipeline: run with -m vision


LOGGER = "sidecoach.replay.recording"


def _frame(seed: int, shape: tuple[int, int, int] = (48, 64, 3)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, shape, dtype=np.uint8)


def _record(folder: Path, frames: list[np.ndarray]) -> Recording:
    """A recording of `frames` at 10 fps with no keyframes, so only changes are stored."""
    region = Region(0, 0, frames[0].shape[1], frames[0].shape[0])
    recorder = Recorder(folder, monitor=0, region=region, fps=10, keyframe_interval_s=None)
    for i, frame in enumerate(frames):
        recorder.write(frame, ts=float(i) / 10)
    recorder.close()
    return Recording(folder)


def _half(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (32, 24), interpolation=cv2.INTER_AREA)


def test_frames_differ_threshold() -> None:
    a = _frame(1)
    b = a.copy()
    assert not frames_differ(a, b)
    b[10, 10, 0] = np.uint8((int(b[10, 10, 0]) + 100) % 256)
    assert frames_differ(a, b)


def test_frames_differ_min_fraction_ignores_small_changes() -> None:
    a = np.zeros((100, 100, 3), dtype=np.uint8)
    b = a.copy()
    b[0:5, 0:5] = 255  # 25 of 10,000 pixels
    assert frames_differ(a, b)
    assert not frames_differ(a, b, min_fraction=0.01)
    b[20:60, 20:60] = 255  # 1,600 more pixels
    assert frames_differ(a, b, min_fraction=0.01)


def test_roundtrip_stores_only_changes_and_replays_every_tick(tmp_path: Path) -> None:
    region = Region(0, 0, 64, 48)
    recorder = Recorder(tmp_path / "rec", monitor=0, region=region, fps=10)
    frames = [_frame(1), _frame(1), _frame(2), _frame(2), _frame(2), _frame(3)]
    ticks = [recorder.write(f, ts=float(i) / 10) for i, f in enumerate(frames)]
    recorder.close()

    assert [t.stored for t in ticks] == [0, None, 1, None, None, 2]
    assert recorder.ticks == 6
    assert recorder.stored == 3

    rec = Recording(tmp_path / "rec")
    assert rec.meta.fps == 10
    assert rec.region == region
    assert rec.stored_count == 3
    replayed = list(rec.frames())
    assert len(replayed) == 6
    for (tick, frame), original in zip(replayed, frames, strict=True):
        assert frame.shape == original.shape
        assert np.array_equal(frame, original), f"tick {tick.frame_id} differs"
    assert [t.frame_id for t, _ in rec.stored_frames()] == [0, 2, 5]


def test_change_region_and_keyframe_gating(tmp_path: Path) -> None:
    region = Region(0, 0, 64, 48)
    watch = Region(40, 30, 20, 15)
    recorder = Recorder(
        tmp_path / "rec",
        monitor=0,
        region=region,
        fps=10,
        change_region=watch,
        keyframe_interval_s=1.0,
    )
    base = _frame(7)
    outside = base.copy()
    outside[2:10, 2:10] = 255  # busy desktop outside the watched region
    inside = base.copy()
    inside[35:40, 45:50] = 0  # a change inside the watched region
    stored = [
        recorder.write(base, 0.0).stored,
        recorder.write(outside, 0.1).stored,  # ignored: outside region, too soon for keyframe
        recorder.write(inside, 0.2).stored,  # stored: inside region changed
        recorder.write(inside, 0.3).stored,  # ignored
        recorder.write(inside, 1.25).stored,  # stored: keyframe due
    ]
    recorder.close()
    assert stored == [0, None, 1, None, 2]
    rec = Recording(tmp_path / "rec")
    assert rec.meta.change_region == (40, 30, 20, 15)
    assert rec.meta.keyframe_interval_s == 1.0


def test_wrong_frame_size_rejected(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "rec", monitor=0, region=Region(0, 0, 64, 48), fps=10)
    try:
        import pytest

        with pytest.raises(ValueError):
            recorder.write(_frame(1, (10, 10, 3)), ts=0.0)
    finally:
        recorder.close()


def test_shrink_keeps_the_ticks_and_the_game_at_half_scale(tmp_path: Path) -> None:
    import chess

    from sidecoach.replay.recording import shrink_recording
    from sidecoach.state.pipeline import GamePipeline
    from tests.synth import synthetic_board

    region = Region(0, 0, 1280, 800)
    recorder = Recorder(tmp_path / "full", monitor=0, region=region, fps=10)
    board = chess.Board()
    frames = [synthetic_board(board=board, cell=96, x0=200, y0=20)] * 3
    for san in ("e4", "e5", "Nf3"):
        move = board.push_san(san)
        frames += [synthetic_board(board=board, cell=96, x0=200, y0=20, highlight=move)] * 3
    for i, frame in enumerate(frames):
        recorder.write(frame, ts=float(i) / 10)
    recorder.close()

    full = Recording(tmp_path / "full")
    small = shrink_recording(tmp_path / "full", tmp_path / "half", 0.5)
    assert len(small.ticks) == len(full.ticks)
    assert [t.stored is None for t in small.ticks] == [t.stored is None for t in full.ticks]
    assert (small.meta.width, small.meta.height, small.meta.scale) == (640, 400, 0.5)
    assert (small.region.w, small.region.h) == (1280, 800)  # still the screen region

    pipe = GamePipeline()
    played = []
    for tick, frame in small.frames():
        assert frame.shape[:2] == (400, 640)
        for event in pipe.process(frame, tick.ts, changed=tick.stored is not None):
            if isinstance(event, MovePlayed):
                played.append(event.san)
    assert played == ["e4", "e5", "Nf3"]


def test_change_gate_matches_the_recorder(tmp_path: Path) -> None:
    """The unrecorded live path must see the same `changed` stream a recording would."""
    from sidecoach.replay.recording import ChangeGate

    region = Region(0, 0, 64, 48)
    gate_region = Region(8, 8, 40, 30)
    recorder = Recorder(
        tmp_path / "rec", monitor=0, region=region, fps=10, change_region=gate_region
    )
    gate = ChangeGate(region=gate_region)
    base = _frame(1)
    edited = base.copy()
    edited[20:30, 20:30] = 0  # inside the gate
    outside = base.copy()
    outside[0:3, 0:3] = 0  # outside the gate
    frames = [base, base, outside, edited, edited, base]
    times = [0.0, 0.1, 0.2, 0.3, 0.4, 2.0]  # the last one is a keyframe by time
    recorded = [
        recorder.write(f, ts).stored is not None for f, ts in zip(frames, times, strict=True)
    ]
    recorder.close()
    live = [gate.changed(f, ts) for f, ts in zip(frames, times, strict=True)]
    assert recorded == live == [True, False, False, True, False, True]


def test_zero_area_gate_never_changes() -> None:
    a = _frame(1)
    b = _frame(2)
    assert not frames_differ(a, b, region=Region(0, 0, 0, 10))


def test_off_frame_board_gates_on_the_whole_frame() -> None:
    from sidecoach.replay.recording import ChangeGate, board_gate
    from sidecoach.vision.board_detect import BoardCandidate

    off = BoardCandidate(Region(-700, 100, 512, 512), 64.0, 1.0, 50.0)
    assert board_gate(off, 1280, 800) is None
    gate = ChangeGate(region=Region(2000, 2000, 10, 10), keyframe_interval_s=None)
    a, b = _frame(1), _frame(2)
    assert gate.changed(a, 0.0)  # the first frame
    assert gate.changed(b, 0.1)  # a different frame must still count as a change


def test_concat_joins_shrunk_copies_and_keeps_their_size_and_scale(tmp_path: Path) -> None:
    a, b = _frame(1), _frame(2)
    _record(tmp_path / "full", [a, a, b, b])
    half = shrink_recording(tmp_path / "full", tmp_path / "half", 0.5)
    assert (half.meta.width, half.meta.height, half.meta.scale) == (32, 24, 0.5)

    joined = concat_recordings([(tmp_path / "half", None, None)] * 2, tmp_path / "joined")
    assert (joined.meta.width, joined.meta.height) == (half.meta.width, half.meta.height)
    assert joined.meta.scale == 0.5
    assert [t.frame_id for t in joined.ticks] == list(range(8))
    assert [t.stored for t in joined.ticks] == [0, None, 1, None, 2, None, 3, None]
    frames = [f for _, f in joined.frames()]
    assert len(frames) == 8
    assert np.array_equal(frames[4], _half(a))
    assert np.array_equal(frames[7], _half(b))

    quarter = shrink_recording(tmp_path / "half", tmp_path / "quarter", 0.5)
    assert (quarter.meta.width, quarter.meta.height, quarter.meta.scale) == (16, 12, 0.25)


def test_concat_refuses_parts_of_different_size_or_scale(tmp_path: Path) -> None:
    _record(tmp_path / "full", [_frame(1)])
    shrink_recording(tmp_path / "full", tmp_path / "half", 0.5)
    parts: list[tuple[Path, int | None, int | None]] = [
        (tmp_path / "full", None, None),
        (tmp_path / "half", None, None),
    ]
    with pytest.raises(ValueError, match="parts must share frame size and scale"):
        concat_recordings(parts, tmp_path / "joined")


def test_concat_copies_the_stored_flags_and_renumbers_ticks(tmp_path: Path) -> None:
    a, b, c = _frame(1), _frame(2), _frame(3)
    _record(tmp_path / "rec", [a, a, b, b, c])  # stored 0, None, 1, None, 2
    parts = [(tmp_path / "rec", 1, 3), (tmp_path / "rec", None, None)]
    joined = concat_recordings(parts, tmp_path / "joined")
    assert [t.frame_id for t in joined.ticks] == list(range(8))
    assert [t.ts for t in joined.ticks] == [0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.3, 0.4]
    # The first tick of each part is stored even where the source repeated a frame; the
    # rest repeat exactly where the source did.
    assert [t.stored for t in joined.ticks] == [0, 1, None, 2, None, 3, None, 4]
    frames = [f for _, f in joined.frames()]
    assert np.array_equal(frames[0], a)
    assert np.array_equal(frames[2], b)
    assert np.array_equal(frames[7], c)


def test_shrink_trim_keeps_source_tick_ids_and_stores_its_first_tick(tmp_path: Path) -> None:
    a, b, c = _frame(1), _frame(2), _frame(3)
    _record(tmp_path / "full", [a, a, b, b, c])  # stored 0, None, 1, None, 2
    small = shrink_recording(tmp_path / "full", tmp_path / "trim", 0.5, first=1, last=3)
    assert [(t.frame_id, t.ts, t.stored) for t in small.ticks] == [
        (1, 0.1, 0),
        (2, 0.2, 1),
        (3, 0.3, None),
    ]
    frames = [f for _, f in small.frames()]
    assert np.array_equal(frames[0], _half(a))
    assert np.array_equal(frames[1], _half(b))
    assert np.array_equal(frames[2], _half(b))


def test_partial_final_index_line_is_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _record(tmp_path / "rec", [_frame(1), _frame(1), _frame(2)])
    index = tmp_path / "rec" / "index.jsonl"
    with index.open("a", encoding="utf-8") as f:
        f.write('{"frame_id": 3, "ts": 0.3, "sto')  # the session was killed mid-line
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        rec = Recording(tmp_path / "rec")
    assert [t.frame_id for t in rec.ticks] == [0, 1, 2]
    assert "incomplete final line 4" in caplog.text
    assert len(list(rec.frames())) == 3


def test_corrupt_middle_index_line_is_an_error(tmp_path: Path) -> None:
    _record(tmp_path / "rec", [_frame(1), _frame(1), _frame(2)])
    index = tmp_path / "rec" / "index.jsonl"
    lines = index.read_text(encoding="utf-8").splitlines()
    lines[1] = '{"frame_id": 1, "ts": 0.1, "sto'
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt index line 2"):
        Recording(tmp_path / "rec")


def test_video_shorter_than_index_replays_what_exists(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    a, b = _frame(1), _frame(2)
    _record(tmp_path / "rec", [a, a, b])  # stored 0, None, 1
    index = tmp_path / "rec" / "index.jsonl"
    with index.open("a", encoding="utf-8") as f:  # a hard kill lost the video's last frame
        f.write(json.dumps({"frame_id": 3, "ts": 0.3, "stored": 2}) + "\n")
        f.write(json.dumps({"frame_id": 4, "ts": 0.4, "stored": None}) + "\n")
    rec = Recording(tmp_path / "rec")
    assert len(rec.ticks) == 5
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        replayed = list(rec.frames())
    assert [t.frame_id for t, _ in replayed] == [0, 1, 2]
    assert np.array_equal(replayed[2][1], b)
    assert "video ends before stored frame 2; 2 of 5 ticks dropped" in caplog.text


class _RefusingWriter:
    def write(self, frame: np.ndarray) -> bool:
        return False


def test_refused_write_leaves_the_gate_on_the_last_frame_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder(
        tmp_path / "rec", monitor=0, region=Region(0, 0, 64, 48), fps=10, keyframe_interval_s=None
    )
    a, b = _frame(1), _frame(2)
    assert recorder.write(a, 0.0).stored == 0
    writer = recorder._writer
    monkeypatch.setattr(recorder, "_writer", _RefusingWriter())
    with pytest.raises(RuntimeError, match="refused frame 1"):
        recorder.write(b, 0.1)
    monkeypatch.setattr(recorder, "_writer", writer)
    # `b` still differs from the last frame the video holds, so it is stored rather than
    # skipped as a repeat of a frame that was never written.
    assert recorder.write(b, 0.2).stored == 1
    recorder.close()
    rec = Recording(tmp_path / "rec")
    assert [(t.frame_id, t.stored) for t in rec.ticks] == [(0, 0), (1, 1)]
    assert np.array_equal(list(rec.frames())[1][1], b)
