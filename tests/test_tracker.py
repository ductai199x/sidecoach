from typing import Any

import chess
import cv2
import numpy as np
import pytest

from sidecoach.events import Region, Side
from sidecoach.vision.board_detect import fit_score, to_gray
from sidecoach.vision.tracker import BoardTracker
from tests.synth import (
    DARK,
    LIGHT,
    PIECE_STYLES,
    bulky_start,
    busy_background,
    render_board,
    synthetic_board,
)

pytestmark = pytest.mark.vision  # the synthetic vision pipeline: run with -m vision


def test_a_lock_holds_in_place_through_validation_and_refits_without_a_search() -> None:
    """Locked once, the board is validated cheaply on every frame and re-fitted on every
    third, and on a busy frame neither moves the lock nor searches again."""
    tracker = BoardTracker(refine_every=3)
    frame = synthetic_board()
    first = tracker.update(frame)
    assert first is not None
    assert abs(first.region.x - 300) <= 2 and abs(first.region.y - 120) <= 2, first
    for _ in range(30):
        board = tracker.update(frame)
        assert board is not None
        assert (
            abs(board.region.x - first.region.x) <= 1 and abs(board.region.y - first.region.y) <= 1
        )
        assert abs(board.cell_px - first.cell_px) <= 0.1
    assert tracker.detections == 1


@pytest.mark.parametrize("x", [330, 364], ids=["half a cell", "a whole cell"])
def test_a_move_is_refitted_on_the_frame_it_fails(x: int) -> None:
    """The frame that fails at the lock is re-fitted at once, and the re-fit reaches the
    board half a cell away, or a whole cell (a whole-cell slide fails validation at once:
    the locked palette reads every cell wrong). The lock moves on that frame, without a
    detection."""
    tracker = BoardTracker()
    locked = tracker.update(synthetic_board(x0=300, y0=120, cell=64))
    assert locked is not None
    board = tracker.update(synthetic_board(x0=x, y0=120, cell=64))
    assert board is not None and tracker.verified
    assert abs(board.region.x - x) <= 2
    assert tracker.detections == 1


def test_a_nudge_is_refitted_before_the_colours_are_read_again() -> None:
    """A wood board five pixels above the frame's bottom, nudged an eighth of a cell: at the
    lock its textured squares read mixed, and read afresh there they validated, so the lock
    kept the mixed colours and stayed 8 px off for good (the near-edge sweep, 2026-09-27).
    The failing frame is re-fitted first, in the lock's own colours."""
    board = chess.Board()
    board.set_board_fen("1r1qk1nr/pbpp4/1pn4p/4Ppp1/P7/bP3PP1/1BPNP2P/2Q1KBNR")
    kw: dict[str, Any] = dict(
        width=1920, height=1200, board=board, theme="wood", cell=64, style="cburnett-wood"
    )
    tracker = BoardTracker()
    locked = tracker.update(synthetic_board(x0=155, y0=683, **kw))
    assert locked is not None and abs(locked.region.x - 155) <= 1
    nudged = synthetic_board(x0=147, y0=677, **kw)
    lock = None
    for _ in range(3):
        lock = tracker.update(nudged)
    assert lock is not None and tracker.verified
    assert abs(lock.region.x - 147) <= 1 and abs(lock.region.y - 677) <= 1, lock


@pytest.mark.parametrize(
    ("cell", "before", "after"),
    [
        (136, (0, 0), (0, 5)),  # against the top-left corner, nudged down
        (88, (985, 0), (993, 0)),  # against the top, nudged right
        (64, (0, 300), (0, 309)),  # against the left, nudged down
        (97, (1920 - 776, 318), (1920 - 776, 291)),  # against the right, nudged up
        (64, (500, 1200 - 512), (509, 1200 - 512)),  # against the bottom, nudged right
        (80, (4, 16), (0, 0)),  # moved into the top-left corner: re-fitted a cell off it
        (80, (734, 76), (726, 81)),  # a cell below the top: re-fitted a cell up, onto the edge
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
    lies, and a comb a cell over, every tooth on a boundary, outscored it. A cell or less
    from the edge, the comb a cell nearer it, its tooth on the edge unseen, scored as much
    as the right one; now the cells decide between them."""
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
    """A 13-pixel nudge at cell 43 is under a third of a cell: validation still passes at
    the old geometry, so only the periodic re-fit can move the lock. The fit is quantised to
    1/64 and blind to sub-cell shifts, so the drifted geometry can score 1.0 while the
    corrected one loses a cell to a badge; the re-fit must still be adopted, on the strength
    of its boundary comb."""
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
    assert abs(board.region.y - 120) <= 2 and abs(board.cell_px - cell) <= 0.5, board
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
    bulky_start(frame, game_x, 100, 48)
    render_board(frame, 800, 100, 48, chess.Board(None))
    return frame


def test_tracker_relocks_on_the_game_board_not_the_empty_one() -> None:
    """The game board moves further than a re-fit reaches, with an empty board of the same
    size on screen: the search must see both boards and put the game board first."""
    tracker = BoardTracker(max_misses=2)
    locked = tracker.update(_game_and_empty_board(60))
    assert locked is not None and abs(locked.region.x - 60) <= 2
    moved = _game_and_empty_board(380)
    board = None
    for _ in range(4):
        board = tracker.update(moved)
    assert board is not None and abs(board.region.x - 380) <= 2, board
    assert tracker.detections == 2


def test_tracker_holds_through_brief_occlusion_then_redetects() -> None:
    tracker = BoardTracker(max_misses=3)
    frame = synthetic_board(x0=300, y0=120, cell=64)
    locked = tracker.update(frame)
    assert locked is not None and locked.fit >= 0.85
    occluded = frame.copy()
    occluded[100:700, 250:850] = 128
    held = tracker.update(occluded)  # miss 1, still holding
    # A held frame reports the fit it measured, at the lock's geometry, unverified.
    assert held is not None and held.fit < 0.85 and not tracker.verified
    assert held.region == locked.region
    assert tracker.update(occluded) is not None  # miss 2
    assert tracker.update(occluded) is None  # miss 3 drops the lock, detection finds nothing
    board = None
    for _ in range(tracker._full.most + 1):  # backoff, then a search that succeeds
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
    assert frames <= tracker._full.first, frames


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
def test_fresh_palette_does_not_relock_from_a_half_covered_board(rows: int) -> None:
    """A dialog in a colour close to one square colour, near the light and then near the
    dark, over half the board or more: the locked palette fails the frame, and the fresh
    palette read from the frame must not replace it, since it would be the dialog's."""
    clean = synthetic_board(x0=300, y0=120, cell=64)
    for near, name in ((LIGHT, "near light"), (DARK, "near dark")):
        tracker = BoardTracker(max_misses=4)
        assert tracker.update(clean) is not None, name
        colours = tracker.colours
        assert colours is not None, name
        covered = clean.copy()
        dialog = tuple(min(255, channel + 4) for channel in near)
        covered[120 : 120 + rows * 64, 300 : 300 + 512] = dialog
        for _ in range(tracker.max_misses - 1):
            held = tracker.update(covered)
            assert held is not None and not tracker.verified, name
            assert tracker.colours == colours, name
        board = tracker.update(clean)
        assert board is not None and tracker.verified and tracker.colours == colours, name
        assert tracker.detections == 1, name


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


def test_a_board_resized_to_near_the_frame_s_bottom_is_locked_where_it_is() -> None:
    """A board resized from 105 to 136 px cells, two pixels above the frame's bottom: the
    near search (84-132 px) found it at 126 and fitted it to 136 two thirds of a cell high,
    and that validated, since the rows of outer cells that would have shown it ran off the
    frame. Every cell then read across two squares (the geometry sweep, 2026-09-27, seed
    260)."""
    board = chess.Board()
    for san in (
        "f3", "Nc6", "g4", "e5", "b4", "b6", "Bh3", "Na5",
        "Kf2", "Bxb4", "c4", "Nxc4", "Bf1", "Nb2", "a3", "d5",
    ):  # fmt: skip
        board.push_san(san)
    kw: dict[str, Any] = {
        "width": 1920, "height": 1200, "theme": "wood", "style": "pixel",
        "bottom": Side.BLACK, "board": board,
    }  # fmt: skip
    tracker = BoardTracker()
    tracker.update(synthetic_board(x0=833, y0=360, cell=105, **kw))
    resized = synthetic_board(x0=583, y0=110, cell=136, **kw)
    locked = None
    for _ in range(tracker.max_misses + 3):
        locked = tracker.update(resized)
    assert locked is not None and tracker.verified
    assert abs(locked.region.x - 583) <= 2 and abs(locked.region.y - 110) <= 2, locked


_GAME = chess.Board("r1bq1rk1/2ppbppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQR1K1 w - - 4 8")
_OTHER = chess.Board("rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 1 5")


def _game_and_thumbnail(game_cell: int, thumbnail: bool, same_theme: bool) -> np.ndarray:
    frame = busy_background(1600, 1000, 4)
    render_board(frame, 40, 40, game_cell, _GAME, style=PIECE_STYLES["cburnett"])
    if thumbnail:
        theme: dict[str, Any] = (
            {} if same_theme else {"light": (200, 220, 235), "dark": (110, 150, 180)}
        )
        render_board(frame, 1200, 60, 40, _OTHER, style=PIECE_STYLES["cburnett"], **theme)
    return frame


@pytest.mark.parametrize("same_theme", [False, True], ids=["other theme", "same theme"])
def test_a_board_resized_past_the_near_range_is_found_not_a_board_of_the_old_size(
    same_theme: bool,
) -> None:
    """The game board enlarged from 40 px cells to 100 with another game's thumbnail at 40
    px beside it: the near-size search found the thumbnail first and the lock stayed on it
    for good (round 2 finding 1). A board found near the old size is taken at once only when
    it shows the lost board's pieces; otherwise the full search decides."""
    tracker = BoardTracker()
    for _ in range(3):
        tracker.update(_game_and_thumbnail(40, False, same_theme))
    board = None
    for _ in range(60):
        board = tracker.update(_game_and_thumbnail(100, True, same_theme))
    assert board is not None and abs(board.region.x - 40) <= 2, board
    assert abs(board.cell_px - 100) <= 1.5, board


def test_a_still_tick_on_a_verified_lock_reads_nothing(monkeypatch) -> None:
    """A tick whose board pixels are the last changed frame's (`changed` False) turned the
    whole frame grey and validated the lock again, nearly all of a still tick's cost: 10 ms
    at 1440p (round 2 finding 7). A verified lock stands unread; a lock in doubt is still
    checked."""
    from sidecoach.vision.board_detect import GrayBuffer

    tracker = BoardTracker()
    frame = synthetic_board()
    locked = tracker.update(frame)
    assert locked is not None and tracker.verified
    calls = []
    real = GrayBuffer.convert
    monkeypatch.setattr(GrayBuffer, "convert", lambda s, f: calls.append(1) or real(s, f))
    for _ in range(5):
        assert tracker.update(frame, changed=False) is locked
    assert calls == []
    tracker.misses = 1  # in doubt: read even on a still tick
    tracker.update(frame, changed=False)
    assert calls == [1]


def test_a_still_board_is_not_refitted_again_on_the_same_pixels(monkeypatch) -> None:
    """The periodic re-fit still came every tenth still tick, on pixels a re-fit had already
    read and left the lock on: 30 ms at 4K each for nothing (round 3). Once a re-fit has run on
    the last changed frame's pixels and kept the lock, still ticks re-fit no more; a changed
    frame re-fits on its schedule again, and a re-fit that moved the lock is run again on the
    still pixels until one keeps it."""
    from sidecoach.vision import tracker as tracker_module

    tracker = BoardTracker()
    frame = synthetic_board()
    locked = tracker.update(frame)
    assert locked is not None and tracker.verified
    refits = []
    real = tracker_module.refine
    monkeypatch.setattr(tracker_module, "refine", lambda *a, **k: refits.append(1) or real(*a, **k))
    for _ in range(10):  # changed ticks: one re-fit, on the tenth, keeps the lock
        tracker.update(frame)
    assert refits == [1]
    for _ in range(40):
        assert tracker.update(frame, changed=False) is not None and tracker.verified
    assert refits == [1]  # not again on the same pixels
    for _ in range(10):
        tracker.update(frame)
    assert refits == [1, 1]  # changed pixels: the schedule again
    # A board nudged two cells' worth of pixels at rest: the re-fit moves the lock, and the
    # still ticks after it re-fit until one keeps it.
    nudged = synthetic_board(x0=306)
    tracker.update(nudged)
    before = len(refits)
    for _ in range(25):
        tracker.update(nudged, changed=False)
    assert tracker.board is not None and abs(tracker.board.region.x - 306) <= 2
    assert len(refits) - before >= 1


def test_a_board_moved_at_its_size_is_relocked_by_the_near_search() -> None:
    """The cheap path stays: the game board moved, same size, same pieces, is re-locked by
    the near-size search without the full one."""
    tracker = BoardTracker(max_misses=2)
    frame = busy_background(1600, 1000, 4)
    render_board(frame, 40, 40, 60, _GAME, style=PIECE_STYLES["cburnett"])
    tracker.update(frame)
    moved = busy_background(1600, 1000, 4)
    render_board(moved, 700, 300, 60, _GAME, style=PIECE_STYLES["cburnett"])
    board = None
    for _ in range(4):
        board = tracker.update(moved)
    assert board is not None and abs(board.region.x - 700) <= 2, board
    assert tracker.full_searches == 1  # the first lock only


# Six plies played while the board was lost: d4 Kh8 g4 d5 exd5 Be6.
_PLAYED_WHILE_LOST = ("d2d4", "g8h8", "g2g4", "d7d5", "e4d5", "c8e6")


@pytest.mark.parametrize("plies", [0, 6], ids=["nothing played", "six plies played"])
@pytest.mark.parametrize("same_theme", [True, False], ids=["same theme", "other theme"])
@pytest.mark.parametrize(
    "new_cell", [30, 37, 80], ids=["shrunk 0.6", "shrunk 0.75", "enlarged 1.6"]
)
def test_a_resized_board_is_found_beside_another_board_of_its_old_size(
    new_cell: int, same_theme: bool, plies: int
) -> None:
    """A compact lost-board sweep: the game board at 50 px cells resized past the near
    range, another game's board on the screen at the old size, the game played on while the
    board was lost. With six plies played, a board found counted as the lost one only within
    eight cells of its occupancy; the shrunk game board was not, the detector put the bigger
    board first, and the other game was locked for good (round 3, finding 1's residual). Of
    the boards found, the nearest in occupancy is taken."""
    game = _GAME.copy()
    for uci in _PLAYED_WHILE_LOST[:plies]:
        game.push_uci(uci)
    theme: dict[str, Any] = (
        {} if same_theme else {"light": (200, 220, 235), "dark": (110, 150, 180)}
    )
    before = busy_background(1600, 1000, 4)
    render_board(before, 40, 40, 50, _GAME, style=PIECE_STYLES["cburnett"])
    after = busy_background(1600, 1000, 4)
    render_board(after, 40, 40, new_cell, game, style=PIECE_STYLES["cburnett"])
    render_board(after, 1100, 60, 50, _OTHER, style=PIECE_STYLES["cburnett"], **theme)
    tracker = BoardTracker()
    for _ in range(3):
        tracker.update(before)
    board = None
    for _ in range(30):
        board = tracker.update(after)
    assert board is not None and tracker.verified, board
    assert abs(board.region.x - 40) <= 2 and abs(board.cell_px - new_cell) <= 1.5, board


def test_a_lock_keeps_its_board_s_pixels_not_the_frame() -> None:
    """The last frame that validated the lock was kept whole, for the lost board's pieces: a
    grey frame of 15 MB at 1440p, 33 at 4K, for the life of the lock (round 3). Only the
    board's region is kept, and it reads the same pieces."""
    from sidecoach.vision.board_detect import board_occupancy

    frame = busy_background(1600, 1000, 4)
    render_board(frame, 40, 40, 50, _GAME, style=PIECE_STYLES["cburnett"])
    tracker = BoardTracker()
    board = tracker.update(frame)
    assert board is not None and tracker._seen is not None
    pixels, placed = tracker._seen
    assert pixels.shape == (board.region.h, board.region.w)
    whole = board_occupancy(to_gray(frame), board.region, board.cell_px)
    assert pixels.dtype == np.uint8  # the grey levels as bytes: a quarter of the float image
    kept = board_occupancy(pixels.astype(np.float32), placed.region, placed.cell_px)
    assert whole is not None and kept is not None and whole.any()
    assert np.array_equal(whole, kept)
