"""Keep a lock on a detected board across frames.

Full detection costs over a second on a 1440p frame, so it runs only when there is no lock.
While locked, every frame is validated with the cheap checker fit at the known region, which
takes milliseconds. The fit reads the cells against the square colours the lock was taken
with, in their polarity, and rejects a pattern that carries on outside the region, so a board
that slid by a whole cell, or a window that moved, fails validation on the first frame; after
`max_misses` consecutive failures the lock is dropped and detection runs again. What
validation cannot see is sub-cell drift and a small resize: the ring medians read the same
colours up to about a third of a cell of shift. Every `refine_every` frames the boundaries are
therefore re-fitted with a wide search window, and the re-fit is adopted when it validates on
its own and its boundary comb is clearly stronger than the one at the lock's geometry. A frame
that fails validation is re-fitted at once, before anything else is tried.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, TypeVar

import numpy as np

from sidecoach.events import Region
from sidecoach.vision.board_detect import (
    MIN_CELL_PX,
    BoardCandidate,
    Colours,
    GrayBuffer,
    board_occupancy,
    detect_boards_gray,
    fit_score,
    readable_cells,
    refine,
    square_colours,
)

_T = TypeVar("_T")


def _edge_shift(a: Region, b: Region) -> int:
    """The most any edge of region `a` moves to become region `b`, in pixels."""
    return max(
        abs(a.x - b.x),
        abs(a.y - b.y),
        abs((a.x + a.w) - (b.x + b.w)),
        abs((a.y + a.h) - (b.y + b.h)),
    )


class _Wait:
    """How long a search that found nothing waits before it runs again: `first` frames, twice
    that after the next miss, up to `most`; a board found starts it over."""

    def __init__(self, first: int, most: int) -> None:
        self.first, self.most = first, most
        self.left = 0  # frames still to wait
        self.next = first  # the wait after the next miss

    def ready(self) -> bool:
        """Counts this frame; whether the wait is over."""
        self.left -= 1
        return self.left <= 0

    def missed(self) -> None:
        self.left, self.next = self.next, min(2 * self.next, self.most)

    def reset(self) -> None:
        self.left, self.next = 0, self.first


class BoardTracker:
    def __init__(
        self,
        *,
        min_fit: float = 0.85,
        min_contrast: float = 6.0,
        max_misses: int = 3,
        refine_every: int = 10,
        refine_offset_cells: float = 1.6,
        refit_gain: float = 1.1,
        refit_deadband_px: int = 2,
        redetect_first: int = 5,
        redetect_backoff: int = 20,
        near_backoff: int = 3,
        probe_lost_cells: int = 8,
        probe_backoff: int = 10,
        probe_backoff_max: int = 80,
        reject: Callable[[np.ndarray, Region, float], bool] | None = None,
    ) -> None:
        self.min_fit = min_fit
        self.min_contrast = min_contrast
        self.max_misses = max_misses
        self.refine_every = refine_every
        self.refine_offset_cells = refine_offset_cells
        self.refit_gain = refit_gain
        # A re-fit that moves no edge by more than this is the same lock: a board flush with
        # the frame's edge re-fitted a pixel or two either way every few seconds, and every
        # cell moved under it was read again -- a second on the phone (Tai's puzzles on the
        # Pixel, 2026-09-23). Sub-cell drift that small costs the reading nothing.
        self.refit_deadband_px = refit_deadband_px
        # The full search that finds nothing waits before it runs again, since it costs a
        # second at 1440p: `redetect_first` frames the first time, twice that the next, up
        # to `redetect_backoff`. A board lost for a moment is usually back within that
        # moment at a size the near search does not cover -- a window dragged small and
        # large again, a board that came back from off the screen resized -- and a flat
        # twenty frames kept it unread for two seconds while the moves went on (the
        # geometry sweep, 2026-09-27: five plies played before the board was seen again).
        self._full = _Wait(redetect_first, redetect_backoff)
        # The search near the last cell size is cheap: every `near_backoff` frames.
        self._near = _Wait(near_backoff, near_backoff)
        # A lock that still validates after the board moved away: the fit counts only the
        # cells that read as square colours, so a board scrolled two and a half cells kept
        # its old lock at fit 0.9 on the cells it left behind, and nothing was read again
        # (the parity audit, 2026-09-26). When a row's worth of the cells the lock could
        # read are gone, the cheap near-size detection looks for the board it lost; while
        # it finds nothing better (a dialog over the board) it waits longer each time.
        self.probe_lost_cells = probe_lost_cells
        self._probe_wait = _Wait(probe_backoff, probe_backoff_max)
        # A veto on detected boards: called with the colour frame, a candidate's region and
        # cell size; True means "not a board we want" (the panel's own board, see
        # vision.ownboard). Vetoed candidates are dropped before one is adopted.
        self.reject = reject
        self.rejected = 0
        self.board: BoardCandidate | None = None
        self.colours: Colours | None = None
        self.last_cell_px: float | None = None
        self.misses = 0
        self.detections = 0
        self._frames_since_refine = 0
        self.probes = 0
        self.full_searches = 0  # detections that ran the full multi-scale search
        self._readable = 0  # the most cells this lock has read as square colours
        # The board's own pixels on the last frame that validated the lock (a copy of its
        # region, not the frame: a whole grey frame pinned for the life of a lock was 15 MB
        # at 1440p, 33 at 4K -- round 3), and the occupied cells read there once the lock is
        # lost: of the boards found then, the one whose pieces are nearest those is taken.
        self._seen: tuple[np.ndarray, BoardCandidate] | None = None
        self._lost: np.ndarray | None = None
        # The lock that was lost, and its colours: a changed frame with no board looks there
        # first (`_back_in_place`).
        self._last: tuple[BoardCandidate, Colours | None] | None = None
        # The grey image of the last frame read, written over by the next (`GrayBuffer`).
        self._gray = GrayBuffer()
        # Whether a re-fit ran on the pixels of the last changed frame and left the lock
        # where it is: a still tick then needs none.
        self._refit_settled = False
        # What was computed on the grey image read last -- a fit at a geometry, the square
        # colours, a re-fit, a search, each a pure function of the image and its arguments --
        # by the arguments; forgotten when the next frame is read (`_once`).
        self._memo: dict[tuple[Any, ...], Any] = {}

    def _once(self, key: tuple[Any, ...], compute: Callable[[], _T]) -> _T:
        """`compute()` on the grey image read last, once: a still tick asks again what was
        asked on these pixels and gets the same answer for nothing. The tracker's own
        bookkeeping -- the ticks since a re-fit, the misses, the back-off -- still runs every
        tick, so its decisions are the same. `BoardTracker.kt`'s `once`."""
        if key in self._memo:
            return self._memo[key]
        value = self._memo[key] = compute()
        return value

    def _fit(
        self, gray: np.ndarray, region: Region, cell: float, colours: Colours | None = None
    ) -> tuple[float, float]:
        return self._once(
            ("fit", region, cell, colours), lambda: fit_score(gray, region, cell, colours)
        )

    def _colours(self, gray: np.ndarray, region: Region, cell: float) -> Colours | None:
        return self._once(("colours", region, cell), lambda: square_colours(gray, region, cell))

    def _readable_cells(
        self, gray: np.ndarray, region: Region, cell: float, colours: Colours
    ) -> int:
        return self._once(
            ("readable", region, cell, colours),
            lambda: readable_cells(gray, region, cell, colours),
        )

    @property
    def gray(self) -> np.ndarray | None:
        """The grey image of the last frame the tracker read -- on a still tick, the pixels
        that are still there -- until the next frame is read; None before the first."""
        return self._gray.image

    @property
    def locked(self) -> bool:
        return self.board is not None

    @property
    def verified(self) -> bool:
        """Whether the last frame validated the lock; False while the lock is only held."""
        return self.board is not None and self.misses == 0

    def refit_next(self) -> None:
        """The lock re-fitted on the next tick, changed or not: a caller found the board off
        the lock (`pipeline._lock_off_board`) and waits on it."""
        self._frames_since_refine = self.refine_every
        self._refit_settled = False

    def _passes(self, fit: float, contrast: float) -> bool:
        return fit >= self.min_fit and contrast >= self.min_contrast

    def update(self, frame_bgr: np.ndarray, changed: bool = True) -> BoardCandidate | None:
        """The lock on this frame. `changed` False says the board's pixels (the caller's
        change gate around the lock) are those of the last changed frame: a verified lock
        then stands as it is, unread -- the whole frame turned grey and the lock validated
        again was nearly all of a still tick's cost, 10 ms at 1440p, 20 at 4K (round 2,
        finding 7) -- but for the periodic re-fit, which still comes every `refine_every`
        ticks: a board that slid a few pixels as it came to rest validates where the lock
        is, and only the re-fit finds it (skipped on still ticks, the lock stayed off for
        good: the gated geometry sweep, seeds 15, 17, 35 and more). A re-fit that already ran
        on these pixels and left the lock where it was would do so again: it is not run again
        until they change (round 3: a still 4K board re-fitted every tenth tick for nothing,
        30 ms each).

        Anything else on a still tick -- no board (the searches' back-off), a lock in doubt, a
        re-fit due -- runs on the grey image of the last changed frame, not this frame turned
        grey again, and what was computed on that image is not computed again (`_once`): a
        still screen with no board searched it every few ticks for what it had not found
        before, 8 ms a tick on average at 1440p, 22 at 4K. The phone's way, made the one for
        both (round 3 audit, E10: each tick takes the cheaper of the two, measured on the same
        frames)."""
        if not changed and self.verified:
            assert self.board is not None
            due = self._frames_since_refine + 1 >= self.refine_every
            if not due or self._refit_settled:
                self._frames_since_refine = 0 if due else self._frames_since_refine + 1
                return self.board
        if changed:
            self._refit_settled = False
        gray = self._gray.image
        if changed or gray is None:
            gray = self._gray.convert(frame_bgr)
            self._memo.clear()  # a new image: nothing computed on it yet
        if self.board is None:
            if changed and (back := self._back_in_place(gray)) is not None:
                return back
            # Without a board on screen, searching every frame would stall capture. The
            # search near the last cell size is cheap and runs every few frames, so a
            # window that comes back, or was covered for a moment, re-locks within a few
            # frames; the full multi-scale search, which costs over a second at 1440p,
            # backs off for longer.
            full = self._full.ready()
            near = self._near.ready() and self.last_cell_px is not None
            return self._search(gray, frame_bgr, full) if full or near else None
        candidate, fit, contrast = self._check(gray)
        if not self._passes(fit, contrast):
            self.misses += 1
            if self.misses < self.max_misses:
                # Held, not verified: the geometry is the last good one, the fit is this
                # frame's, so a caller can see the lock is in doubt.
                self.board = replace(self.board, fit=fit, contrast=contrast)
                return self.board
            self._last = (self.board, self.colours)
            self.board = None
            self.colours = None
            if self._seen is not None:
                seen_levels, seen = self._seen
                seen_gray = seen_levels.astype(np.float32)  # the grey image's own values
                self._lost = board_occupancy(seen_gray, seen.region, seen.cell_px)
                self._seen = None
            return self._search(gray, frame_bgr, True)
        readable = (
            self._readable_cells(gray, candidate.region, candidate.cell_px, self.colours)
            if self.colours is not None
            else 0
        )
        if self._lost_cells(readable):
            moved = self._probe(gray, frame_bgr, candidate, readable)
            if moved is not None:
                return moved
        self.misses = 0
        self._readable = max(self._readable, readable)
        self.board = replace(candidate, fit=fit, contrast=contrast)
        self._keep_seen(gray, self.board)
        self.last_cell_px = self.board.cell_px
        return self.board

    def _back_in_place(self, gray: np.ndarray) -> BoardCandidate | None:
        """The lost lock again, when this frame validates where it was: a cover lifted from a
        board that never moved. One fit, so it is tried on every changed frame; the searches'
        back-off kept a board back from under the dim between two puzzles unread for two
        ticks, and the next puzzle's first board, on screen for one, was never seen (the
        puzzle sweep, 2026-10-03)."""
        if self._last is None:
            return None
        board, colours = self._last
        fit, contrast = self._fit(gray, board.region, board.cell_px, colours)
        if not self._passes(fit, contrast):
            return None
        self._lock(gray, replace(board, fit=fit, contrast=contrast), colours)
        self._full.reset()
        self._near.reset()
        return self.board

    def _keep_seen(self, gray: np.ndarray, board: BoardCandidate) -> None:
        """The board's region of this frame, copied as whole grey levels (a quarter of the
        float image's bytes, the same values), with the board placed in it."""
        x0, y0 = max(0, board.region.x), max(0, board.region.y)
        x1 = min(gray.shape[1], board.region.x + board.region.w)
        y1 = min(gray.shape[0], board.region.y + board.region.h)
        region = board.region._replace(x=board.region.x - x0, y=board.region.y - y0)
        levels = self._gray.bytes if self._gray.image is gray else None
        source = gray if levels is None else levels
        self._seen = (source[y0:y1, x0:x1].astype(np.uint8), replace(board, region=region))

    def _check(self, gray: np.ndarray) -> tuple[BoardCandidate, float, float]:
        """The lock on this frame: its geometry, fit and contrast. Re-fitted when due, and
        at once on the first frame it fails; still failing, its palette is read afresh."""
        assert self.board is not None
        self._frames_since_refine += 1
        candidate = self.board
        fit, contrast = self._fit(gray, candidate.region, candidate.cell_px, self.colours)
        # A frame that fails is re-fitted at once, the first of a run of failures, with the
        # lock's own colours: a board that moved a fraction of a cell is likelier than a new
        # theme, and read where the lock is, its squares' colours come out mixed. Read afresh
        # there first, the mixed colours validated, the lock kept them, and the periodic
        # re-fit read the board itself in them as nothing: the lock stayed an eighth of a
        # cell off for good (the near-edge sweep, 2026-09-27).
        failed_first = not self._passes(fit, contrast) and self.misses == 0
        if self._frames_since_refine >= self.refine_every or failed_first:
            self._frames_since_refine = 0
            at_lock, colours = candidate, self.colours
            refit = self._once(
                ("refine", at_lock, self.refine_offset_cells, colours),
                lambda: refine(
                    gray, at_lock, offset_cells=self.refine_offset_cells, colours=colours
                ),
            )
            # A re-fit is adopted only when it validates on its own, and then only when this
            # frame failed at the lock's geometry or the re-fit moved it and its boundary comb
            # is clearly stronger than the comb there. The fit score cannot arbitrate: it is
            # blind to shifts of up to a third of a cell, so a drifted geometry can still
            # score 1.0 while the corrected one loses a cell to a badge. A re-fit that fails
            # never costs the lock; the region and the fit it measured itself stand.
            refined = refit.board
            moved = _edge_shift(candidate.region, refined.region) > self.refit_deadband_px
            adopted = self._passes(refined.fit, refined.contrast) and (
                not self._passes(fit, contrast)
                or (moved and refit.strength > self.refit_gain * refit.reference)
            )
            # On these pixels, at this lock, a re-fit changes nothing: until they change.
            self._refit_settled = not adopted and self._passes(fit, contrast)
            if adopted:
                if moved:
                    # The board is moving: it is re-fitted the next frame too, until a
                    # re-fit finds it where the lock is. Fitted once mid-slide and left, the
                    # lock trailed the board by a quarter of a cell once it stopped, under
                    # what validation sees, until the next re-fit ten frames on, and no
                    # move was read (the geometry sweep, 2026-09-27, seed 83).
                    self._frames_since_refine = self.refine_every
                candidate = refined
                fit, contrast = refined.fit, refined.contrast
        if not self._passes(fit, contrast) and self.colours is not None:
            # The theme may have changed between games with the board in place: a fresh
            # palette that validates, outer ring included, re-locks the colours. A dialog
            # over most of the board fails this too, since its palette is the dialog, so it
            # cannot rewrite the lock. A new theme keeps the board where it was, its light
            # squares the lighter ones: a palette whose light and dark swap places is the
            # board read a cell off (moved 0.8 of a cell up, the lock re-read its colours
            # inverted and stayed where the board was: the parity audit's sweep, 2026-09-26).
            fresh_fit, fresh_contrast = self._fit(gray, candidate.region, candidate.cell_px)
            fresh = self._colours(gray, candidate.region, candidate.cell_px)
            if (
                self._passes(fresh_fit, fresh_contrast)
                and fresh is not None
                and (fresh[0] > fresh[1]) == (self.colours[0] > self.colours[1])
            ):
                self.colours = fresh
                fit, contrast = fresh_fit, fresh_contrast
                self._readable = 0  # a new palette: what it reads is counted afresh
        return candidate, fit, contrast

    def _search(self, gray: np.ndarray, frame_bgr: np.ndarray, full: bool) -> BoardCandidate | None:
        """Detection with the lock gone; a board found is locked, and a miss backs off."""
        searched = self.full_searches
        boards = self._detect(gray, frame_bgr, full=full, lost=self._lost)
        board = boards[0] if boards else None
        self._lock(gray, board, self._colours(gray, board.region, board.cell_px) if board else None)
        if self.board:
            self._full.reset()
            self._near.reset()
        else:
            self._near.missed()
            if self.full_searches != searched:
                self._full.missed()
        return self.board

    def _lock(
        self, gray: np.ndarray, board: BoardCandidate | None, colours: Colours | None
    ) -> None:
        """A lock taken (or none): what it reads is its measure, and the probe starts fresh."""
        self.misses = 0
        self._frames_since_refine = 0
        self._refit_settled = False
        self.board = board
        self.colours = colours
        if board is not None:
            self.last_cell_px = board.cell_px
            self._keep_seen(gray, board)
            self._lost = None
            self._last = None
        self._readable = (
            self._readable_cells(gray, board.region, board.cell_px, colours)
            if board is not None and colours is not None
            else 0
        )
        self._probe_wait.reset()

    def _kept(self, frame_bgr: np.ndarray, boards: list[BoardCandidate]) -> list[BoardCandidate]:
        """The detected boards the veto (`reject`) lets through, the others counted."""
        if self.reject is None or not boards:
            return boards
        kept = [b for b in boards if not self.reject(frame_bgr, b.region, b.cell_px)]
        self.rejected += len(boards) - len(kept)
        return kept

    def _lost_cells(self, readable: int) -> bool:
        """Whether a row's worth of the cells this lock could read are gone; when they are
        back, a later loss is probed at once."""
        lost = readable <= self._readable - self.probe_lost_cells
        if not lost:
            self._probe_wait.reset()
        return lost

    def _probe(
        self, gray: np.ndarray, frame_bgr: np.ndarray, lock: BoardCandidate, readable: int
    ) -> BoardCandidate | None:
        """The board this lock lost, if the near-size detection finds it elsewhere: a board
        away from the lock that validates in its own colours and reads what the lock read
        before it lost its cells -- the whole board, not only more than now: under a
        window over part of the board a region a cell off read more and was taken. Adopted
        as a new lock. Nothing found (a dialog over the board, an arrow): the lock stands,
        and the next probe waits twice as long. Counted in `probes`, not `detections`."""
        if not self._probe_wait.ready():
            return None
        self.probes += 1
        for board in self._detect(gray, frame_bgr, full=False, count=False):
            if _edge_shift(board.region, lock.region) <= self.refit_deadband_px:
                continue
            colours = self._colours(gray, board.region, board.cell_px)
            if colours is None:
                continue
            fit, contrast = self._fit(gray, board.region, board.cell_px, colours)
            cells = self._readable_cells(gray, board.region, board.cell_px, colours)
            if self._passes(fit, contrast) and cells > self._readable - self.probe_lost_cells:
                self._lock(gray, replace(board, fit=fit, contrast=contrast), colours)
                self.detections += 1
                return self.board
        self._probe_wait.missed()
        return None

    def _detect(
        self,
        gray: np.ndarray,
        frame_bgr: np.ndarray,
        full: bool = True,
        count: bool = True,
        lost: np.ndarray | None = None,
    ) -> list[BoardCandidate]:
        """Detection, trying the last known cell size first; the veto applied to each search.

        A window that was moved keeps its cell size, so a narrow scale search around it
        re-locks in a fraction of the full multi-scale search. Only on failure, and only
        when `full` allows it, does the full range run.

        With `lost` (the occupied cells of the board a lock lost), the boards found are
        ranked by how many cells their occupancy differs from it, nearest first: the game
        board resized past the near range, with another board at the old size beside it --
        a thumbnail of another game, the panel's own board -- had the near search stop at
        that board, and the lock stayed on it for good (round 2, finding 1); a cut-off of a
        few cells still took the thumbnail once a few moves were played while the board was
        lost (round 3). The near search alone sees only the boards near the old size, so it
        settles the matter by itself only for a board that shows exactly the lost pieces --
        the board moved, nothing played; any other board found calls the full search, where
        every board on the screen is ranked.
        """
        if count:
            self.detections += 1
        if self.last_cell_px is not None:
            min_cell = max(MIN_CELL_PX, int(self.last_cell_px * 0.8))
            max_cell = int(self.last_cell_px * 1.25) + 1
            near = self._kept(
                frame_bgr,
                self._once(
                    ("detect", min_cell, max_cell),
                    lambda: detect_boards_gray(
                        gray,
                        min_fit=self.min_fit,
                        min_contrast=self.min_contrast,
                        min_cell=min_cell,
                        max_cell=max_cell,
                    ),
                ),
            )
            if lost is not None and near:
                exact = [board for board, apart in _nearest_first(gray, near, lost) if apart == 0]
                # Boards found but none of them the lost one to the cell: only the full
                # search can say which board on the screen is nearest, so it runs now,
                # whatever its wait (a board uncovered after two plies under a window waited
                # two seconds for it, with the board in plain view).
                full = full or not exact
                near = exact
            if near:
                return near
        if not full:
            return []
        self.full_searches += 1
        boards = self._kept(
            frame_bgr,
            self._once(
                ("detect",),
                lambda: detect_boards_gray(
                    gray, min_fit=self.min_fit, min_contrast=self.min_contrast
                ),
            ),
        )
        if lost is None:
            return boards
        return [board for board, _apart in _nearest_first(gray, boards, lost)]


def _nearest_first(
    gray: np.ndarray, boards: list[BoardCandidate], lost: np.ndarray
) -> list[tuple[BoardCandidate, int]]:
    """`boards` with how many cells their occupancy differs from the lost board's by (all
    64 for a board the frame does not show whole), nearest first, the detector's order
    kept among equals."""
    apart = []
    for board in boards:
        grid = board_occupancy(gray, board.region, board.cell_px)
        apart.append((board, 64 if grid is None else int((grid != lost).sum())))
    return sorted(apart, key=lambda pair: pair[1])
