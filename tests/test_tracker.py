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


def _dialog_and_page_edge(frame: np.ndarray) -> np.ndarray:
    """A dialog over columns a to c of rows 1 to 6, spilling left, and a page edge a cell right."""
    import cv2

    out = frame.copy()
    cv2.rectangle(out, (300 - 40, 120 + 128), (300 + 192, 120 + 512), (40, 40, 40), -1)
    out[:, 876:] = (30, 30, 30)
    return out


def test_tracker_does_not_slide_under_a_dialog_beside_a_page_edge() -> None:
    tracker = BoardTracker(refine_every=3)
    clean = synthetic_board(x0=300, y0=120, cell=64)
    assert tracker.update(clean) is not None
    covered = _dialog_and_page_edge(clean)
    for _ in range(8):
        board = tracker.update(covered)
        assert board is not None
        assert abs(board.region.x - 300) <= 2, board.region
    assert tracker.detections == 1


def test_refit_cannot_turn_a_passing_frame_into_a_miss(monkeypatch) -> None:
    from dataclasses import replace

    import sidecoach.vision.tracker as tracker_module

    tracker = BoardTracker(refine_every=1)
    frame = synthetic_board(x0=300, y0=120, cell=64)
    locked = tracker.update(frame)
    assert locked is not None
    monkeypatch.setattr(
        tracker_module, "refine", lambda gray, cand, **kw: replace(cand, fit=0.845, contrast=50.0)
    )
    monkeypatch.setattr(
        tracker_module, "fit_score", lambda gray, region, cell, colours=None: (0.86, 50.0)
    )
    board = tracker.update(frame)
    assert board is not None and tracker.misses == 0
    assert board.region == locked.region and board.fit == 0.86


def test_tracker_thresholds_reach_detection() -> None:
    frame = synthetic_board(x0=300, y0=120, cell=64, pieces=False)
    from tests.synth import DARK

    spoiled = frame.copy()
    for r, c in ((0, 0), (0, 2), (2, 0), (4, 4), (6, 2), (2, 6)):  # six light cells painted dark
        spoiled[120 + r * 64 : 184 + r * 64, 300 + c * 64 : 364 + c * 64] = DARK
    tracker = BoardTracker(min_fit=0.95)
    results = [tracker.update(spoiled) for _ in range(12)]
    assert all(b is None for b in results) or tracker.detections == 1


def test_held_frames_report_the_measured_fit() -> None:
    tracker = BoardTracker(max_misses=3)
    frame = synthetic_board(x0=300, y0=120, cell=64)
    locked = tracker.update(frame)
    assert locked is not None and locked.fit >= 0.85
    occluded = frame.copy()
    occluded[100:700, 250:850] = 128
    held = tracker.update(occluded)
    assert held is not None and held.fit < 0.85 and not tracker.verified
    assert held.region == locked.region


def test_periodic_refit_is_stable_on_a_busy_frame() -> None:
    tracker = BoardTracker(refine_every=3)
    frame = synthetic_board()
    first = tracker.update(frame)
    assert first is not None
    for _ in range(30):
        board = tracker.update(frame)
        assert board is not None
        assert (
            abs(board.region.x - first.region.x) <= 1 and abs(board.region.y - first.region.y) <= 1
        )
        assert abs(board.cell_px - first.cell_px) <= 0.1
    assert tracker.detections == 1
