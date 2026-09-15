from pathlib import Path

import numpy as np

from sidecoach.events import MovePlayed, Region
from sidecoach.replay.recording import Recorder, Recording, frames_differ


def _frame(seed: int, shape: tuple[int, int, int] = (48, 64, 3)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, shape, dtype=np.uint8)


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
    recorded = [recorder.write(f, ts).stored is not None for f, ts in zip(frames, times)]
    recorder.close()
    live = [gate.changed(f, ts) for f, ts in zip(frames, times)]
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
