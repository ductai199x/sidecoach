from typing import Any

import chess
import cv2
import numpy as np
import pytest

from sidecoach.events import Region
from sidecoach.vision.board_detect import fit_score, to_gray
from sidecoach.vision.tracker import BoardTracker
from tests.synth import (
    DARK,
    LIGHT,
    bulky_pieces,
    busy_background,
    render_board,
    synthetic_board,
)

pytestmark = pytest.mark.vision  # the synthetic vision pipeline: run with -m vision


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


def test_refit_while_held_relocks_a_one_cell_slide_without_detection() -> None:
    """A whole-cell slide fails validation at once (the locked palette reads every cell
    wrong), so the lock is held, not verified. With the re-fit due before the misses run
    out, its wide window finds the slid board and the lock moves without a detection."""
    tracker = BoardTracker(refine_every=3, max_misses=3)
    locked = tracker.update(synthetic_board(x0=300, y0=120, cell=64))
    assert locked is not None
    slid = synthetic_board(x0=364, y0=120, cell=64)
    for _ in range(2):
        held = tracker.update(slid)
        assert held is not None and not tracker.verified and held.region == locked.region
    board = tracker.update(slid)
    assert board is not None and tracker.verified
    assert abs(board.region.x - 364) <= 2
    assert tracker.detections == 1


def test_sub_cell_nudge_is_recovered_by_the_refit() -> None:
    """A 13-pixel nudge at cell 43 is under a third of a cell: validation still passes at
    the old geometry, so only the periodic re-fit can move the lock."""
    tracker = BoardTracker()
    locked = tracker.update(synthetic_board(x0=300, y0=120, cell=43))
    assert locked is not None and abs(locked.region.x - 300) <= 1
    nudged = synthetic_board(x0=313, y0=120, cell=43)
    board = None
    for _ in range(12):
        board = tracker.update(nudged)
        assert board is not None and tracker.verified
    assert board is not None and abs(board.region.x - 313) <= 2, board
    assert abs(board.region.y - 120) <= 2 and abs(board.cell_px - 43) <= 0.5
    assert tracker.detections == 1


@pytest.mark.parametrize(
    ("cell", "before", "after"),
    [
        (136, (0, 0), (0, 5)),  # against the top-left corner, nudged down
        (88, (985, 0), (993, 0)),  # against the top, nudged right
        (64, (0, 300), (0, 309)),  # against the left, nudged down
        (97, (1920 - 776, 318), (1920 - 776, 291)),  # against the right, nudged up
        (64, (500, 1200 - 512), (509, 1200 - 512)),  # against the bottom, nudged right
        (80, (4, 16), (0, 0)),  # moved into the top-left corner: re-fitted a cell off it
    ],
)
def test_a_board_flush_with_an_edge_is_followed_when_nudged(
    cell: int, before: tuple[int, int], after: tuple[int, int]
) -> None:
    """The frame's edge shows no boundary (the gradient is taken with the border reflected),
    so the re-fit's comb a cell over, its ninth tooth on whatever lay beyond the board,
    outscored the right one, failed validation, and the lock stayed where the board had been
    (the geometry sweep, 2026-09-27). Against the left or top edge the comb could not start
    on the frame's edge, half a pixel before the first pixel, where the board's boundary
    lies, and a comb a cell over, every tooth on a boundary, outscored it."""
    tracker = BoardTracker()
    tracker.update(synthetic_board(width=1920, height=1200, x0=before[0], y0=before[1], cell=cell))
    moved = synthetic_board(width=1920, height=1200, x0=after[0], y0=after[1], cell=cell)
    board = None
    for _ in range(tracker.refine_every + 2):
        board = tracker.update(moved)
    assert board is not None and tracker.verified
    assert abs(board.region.x - after[0]) <= 2 and abs(board.region.y - after[1]) <= 2, board
    assert abs(board.cell_px - cell) <= 0.5


def test_refit_wins_on_boundary_strength_not_on_the_fit_score() -> None:
    """The fit is quantised to 1/64 and blind to sub-cell shifts, so the drifted geometry
    can score 1.0 while the corrected one loses a cell to a badge; the re-fit must still be
    adopted, on the strength of its boundary comb."""
    cell = 43
    tracker = BoardTracker()
    # An empty board: with pieces, the shifted rings of the old geometry cut through them
    # and the old fit drops on its own, which is the easy case.
    locked = tracker.update(synthetic_board(x0=300, y0=120, cell=cell, pieces=False))
    assert locked is not None and abs(locked.region.x - 300) <= 1
    nudged = synthetic_board(x0=313, y0=120, cell=cell, pieces=False)
    # A dark badge over most of the ring of the light cell (2, 2) of the board as it now
    # is, placed where the old geometry's ring cannot see it: the cell's left band lies in
    # the old cell's interior, and a strip down its right side lies past the old cell's
    # right boundary. The new geometry loses the cell; the old one does not.
    band = int(round(cell * 0.12))
    x0, y0 = 313 + 2 * cell, 120 + 2 * cell
    x1, y1 = x0 + cell, y0 + cell
    nudged[y0:y1, x0 : x0 + band] = DARK
    nudged[y0:y1, 300 + 3 * cell : x1] = DARK
    # And a solid blob over the light cell (5, 5) at both geometries, so neither fit is 1.0.
    nudged[120 + 5 * cell : 120 + 6 * cell, 300 + 5 * cell : 313 + 6 * cell] = DARK
    gray = to_gray(nudged)
    old_fit = fit_score(gray, locked.region, locked.cell_px, tracker.colours)[0]
    new_fit = fit_score(gray, Region(313, 120, 8 * cell, 8 * cell), cell, tracker.colours)[0]
    assert tracker.min_fit <= new_fit < old_fit < 1.0, (old_fit, new_fit)
    board = None
    for _ in range(12):
        board = tracker.update(nudged)
        assert board is not None and tracker.verified
    assert board is not None and abs(board.region.x - 313) <= 2, board
    assert tracker.detections == 1


def test_locked_fit_keeps_polarity_on_an_inverted_theme() -> None:
    """On a theme whose light squares are the darker ones the locked palette must still
    read the board as a fit, otherwise the re-fit could never be adopted."""
    theme: dict[str, Any] = {"light": (60, 60, 200), "dark": (200, 200, 60)}
    tracker = BoardTracker()
    locked = tracker.update(synthetic_board(**theme))
    assert locked is not None and tracker.colours is not None
    gray = to_gray(synthetic_board(**theme))
    assert fit_score(gray, locked.region, locked.cell_px, tracker.colours)[0] >= 0.85
    nudged = synthetic_board(x0=316, y0=120, **theme)
    board = None
    for _ in range(tracker.refine_every):
        board = tracker.update(nudged)
        assert board is not None
    assert board is not None and abs(board.region.x - 316) <= 2, board
    assert tracker.detections == 1


def _game_and_empty_board(game_x: int) -> np.ndarray:
    frame = np.full((900, 1400, 3), 90, dtype=np.uint8)
    render_board(frame, game_x, 100, 48, chess.Board())
    bulky_pieces(frame, game_x, 100, 48, chess.Board())
    render_board(frame, 800, 100, 48, chess.Board(None))
    return frame


def test_tracker_relocks_on_the_game_board_not_the_empty_one() -> None:
    """The window moves half a cell with an empty board of the same size on screen: the
    search must see both boards and put the game board first."""
    tracker = BoardTracker(max_misses=2)
    locked = tracker.update(_game_and_empty_board(60))
    assert locked is not None and abs(locked.region.x - 60) <= 2
    moved = _game_and_empty_board(84)
    board = None
    for _ in range(4):
        board = tracker.update(moved)
    assert board is not None and abs(board.region.x - 84) <= 2, board
    assert tracker.detections == 2


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


def test_a_board_back_at_another_size_after_a_moment_is_found_soon() -> None:
    """The board gone for a moment (the full search at the loss finds nothing), then back at
    a size the near search does not cover: the next full search comes `redetect_first`
    frames later, not twenty (the geometry sweep, 2026-09-27: five plies played before a
    board shrunk to nothing and back was seen again)."""
    tracker = BoardTracker()
    tracker.update(synthetic_board(width=1400, height=1000, x0=300, y0=120, cell=64))
    gone = busy_background(1400, 1000, 0)
    for _ in range(tracker.max_misses + 1):
        tracker.update(gone)
    assert not tracker.locked
    back = synthetic_board(width=1400, height=1000, x0=200, y0=60, cell=100)
    frames = 0
    board = None
    while frames < 30:
        frames += 1
        board = tracker.update(back)
        if board is not None and tracker.verified:
            break
    assert board is not None and abs(board.region.x - 200) <= 2 and abs(board.cell_px - 100) <= 1
    assert frames <= tracker.redetect_first, frames


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


@pytest.mark.parametrize("rows", [4, 5], ids=["32 cells", "40 cells"])
@pytest.mark.parametrize("near", [LIGHT, DARK], ids=["near light", "near dark"])
def test_fresh_palette_does_not_relock_from_a_half_covered_board(
    rows: int, near: tuple[int, int, int]
) -> None:
    """A dialog in a colour close to one square colour over half the board or more: the
    locked palette fails the frame, and the fresh palette read from the frame must not
    replace it, since it would be the dialog's."""
    tracker = BoardTracker(max_misses=4)
    clean = synthetic_board(x0=300, y0=120, cell=64)
    assert tracker.update(clean) is not None
    colours = tracker.colours
    assert colours is not None
    covered = clean.copy()
    dialog = tuple(min(255, channel + 4) for channel in near)
    covered[120 : 120 + rows * 64, 300 : 300 + 512] = dialog
    for _ in range(tracker.max_misses - 1):
        held = tracker.update(covered)
        assert held is not None and not tracker.verified
        assert tracker.colours == colours
    board = tracker.update(clean)
    assert board is not None and tracker.verified and tracker.colours == colours
    assert tracker.detections == 1


def test_refit_cannot_turn_a_passing_frame_into_a_miss(monkeypatch) -> None:
    from dataclasses import replace

    import sidecoach.vision.tracker as tracker_module
    from sidecoach.vision.board_detect import Refinement

    tracker = BoardTracker(refine_every=1)
    frame = synthetic_board(x0=300, y0=120, cell=64)
    locked = tracker.update(frame)
    assert locked is not None
    monkeypatch.setattr(
        tracker_module,
        "refine",
        lambda gray, cand, **kw: Refinement(replace(cand, fit=0.845, contrast=50.0), 9.0, 1.0),
    )
    monkeypatch.setattr(
        tracker_module, "fit_score", lambda gray, region, cell, colours=None: (0.86, 50.0)
    )
    board = tracker.update(frame)
    assert board is not None and tracker.misses == 0
    assert board.region == locked.region and board.fit == 0.86


def test_tracker_thresholds_reach_detection() -> None:
    """Six light cells painted dark leave a fit of 58/64: the default threshold accepts the
    board, a tracker asking for 0.95 never locks, and, without a lock, it searches and backs
    off rather than searching every frame: again `redetect_first` frames later, then twice
    that, up to `redetect_backoff`."""
    frame = synthetic_board(x0=300, y0=120, cell=64, pieces=False)
    spoiled = frame.copy()
    for r, c in ((0, 0), (0, 2), (2, 0), (4, 4), (6, 2), (2, 6)):
        spoiled[120 + r * 64 : 184 + r * 64, 300 + c * 64 : 364 + c * 64] = DARK
    assert BoardTracker().update(spoiled) is not None
    tracker = BoardTracker(min_fit=0.95)
    results = [tracker.update(spoiled) for _ in range(12)]
    assert all(board is None for board in results)
    assert tracker.detections == 2  # on the first frame and the sixth; the next on the 16th


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
