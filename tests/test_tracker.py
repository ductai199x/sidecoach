import numpy as np

from sidecoach.vision.tracker import BoardTracker
from tests.synth import synthetic_board


def test_tracker_locks_and_validates_cheaply() -> None:
    tracker = BoardTracker()
    frame = synthetic_board(x0=300, y0=120, cell=64)
    first = tracker.update(frame)
    assert first is not None and abs(first.region.x - 300) <= 2
    for _ in range(5):
        board = tracker.update(frame)
        assert board is not None
    assert tracker.detections == 1


def test_tracker_relocks_after_board_moves_half_a_cell() -> None:
    tracker = BoardTracker(max_misses=2)
    tracker.update(synthetic_board(x0=300, y0=120, cell=64))
    moved = synthetic_board(x0=330, y0=120, cell=64)
    board = None
    for _ in range(4):
        board = tracker.update(moved)
    assert board is not None
    assert abs(board.region.x - 330) <= 2
    assert tracker.detections == 2


def test_tracker_periodic_refine_catches_one_cell_slide() -> None:
    tracker = BoardTracker(refine_every=3)
    tracker.update(synthetic_board(x0=300, y0=120, cell=64))
    slid = synthetic_board(x0=364, y0=120, cell=64)
    board = None
    for _ in range(3):
        board = tracker.update(slid)
    assert board is not None
    assert abs(board.region.x - 364) <= 2
    assert tracker.detections == 1


def test_tracker_holds_through_brief_occlusion_then_redetects() -> None:
    tracker = BoardTracker(max_misses=3)
    frame = synthetic_board(x0=300, y0=120, cell=64)
    tracker.update(frame)
    occluded = frame.copy()
    occluded[100:700, 250:850] = 128
    assert tracker.update(occluded) is not None  # miss 1, still holding
    assert tracker.update(occluded) is not None  # miss 2
    assert tracker.update(occluded) is None  # miss 3 drops the lock, detection finds nothing
    board = None
    for _ in range(tracker.redetect_backoff + 1):  # backoff, then a search that succeeds
        board = tracker.update(frame)
        if board is not None:
            break
    assert board is not None and abs(board.region.x - 300) <= 2
    assert tracker.detections == 3


def test_tracker_relocks_after_resize() -> None:
    tracker = BoardTracker(max_misses=2)
    tracker.update(synthetic_board(x0=300, y0=120, cell=64))
    resized = synthetic_board(x0=280, y0=100, cell=76)
    board = None
    for _ in range(4):
        board = tracker.update(resized)
    assert board is not None
    assert abs(board.cell_px - 76) <= 0.5
    assert abs(board.region.x - 280) <= 2


def test_tracker_no_board() -> None:
    rng = np.random.default_rng(2)
    noise = rng.integers(0, 256, (400, 600, 3), dtype=np.uint8)
    tracker = BoardTracker()
    assert tracker.update(noise) is None
    assert not tracker.locked
