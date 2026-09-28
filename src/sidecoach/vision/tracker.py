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

import numpy as np

from sidecoach.events import Region
from sidecoach.vision.board_detect import (
    MIN_CELL_PX,
    BoardCandidate,
    Colours,
    detect_boards_gray,
    fit_score,
    readable_cells,
    refine,
    square_colours,
    to_gray,
)


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
        self._readable = 0  # the most cells this lock has read as square colours

    @property
    def locked(self) -> bool:
        return self.board is not None

    @property
    def verified(self) -> bool:
        """Whether the last frame validated the lock; False while the lock is only held."""
        return self.board is not None and self.misses == 0

    def _passes(self, fit: float, contrast: float) -> bool:
        return fit >= self.min_fit and contrast >= self.min_contrast

    def update(self, frame_bgr: np.ndarray) -> BoardCandidate | None:
        gray = to_gray(frame_bgr)
        if self.board is None:
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
            self.board = None
            self.colours = None
            return self._search(gray, frame_bgr, True)
        readable = (
            readable_cells(gray, candidate.region, candidate.cell_px, self.colours)
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
        self.last_cell_px = self.board.cell_px
        return self.board

    def _check(self, gray: np.ndarray) -> tuple[BoardCandidate, float, float]:
        """The lock on this frame: its geometry, fit and contrast. Re-fitted when due, and
        at once on the first frame it fails; still failing, its palette is read afresh."""
        assert self.board is not None
        self._frames_since_refine += 1
        candidate = self.board
        fit, contrast = fit_score(gray, candidate.region, candidate.cell_px, self.colours)
        # A frame that fails is re-fitted at once, the first of a run of failures, with the
        # lock's own colours: a board that moved a fraction of a cell is likelier than a new
        # theme, and read where the lock is, its squares' colours come out mixed. Read afresh
        # there first, the mixed colours validated, the lock kept them, and the periodic
        # re-fit read the board itself in them as nothing: the lock stayed an eighth of a
        # cell off for good (the near-edge sweep, 2026-09-27).
        failed_first = not self._passes(fit, contrast) and self.misses == 0
        if self._frames_since_refine >= self.refine_every or failed_first:
            self._frames_since_refine = 0
            refit = refine(
                gray, candidate, offset_cells=self.refine_offset_cells, colours=self.colours
            )
            # A re-fit is adopted only when it validates on its own, and then only when this
            # frame failed at the lock's geometry or the re-fit moved it and its boundary comb
            # is clearly stronger than the comb there. The fit score cannot arbitrate: it is
            # blind to shifts of up to a third of a cell, so a drifted geometry can still
            # score 1.0 while the corrected one loses a cell to a badge. A re-fit that fails
            # never costs the lock; the region and the fit it measured itself stand.
            refined = refit.board
            moved = _edge_shift(candidate.region, refined.region) > self.refit_deadband_px
            if self._passes(refined.fit, refined.contrast) and (
                not self._passes(fit, contrast)
                or (moved and refit.strength > self.refit_gain * refit.reference)
            ):
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
            fresh_fit, fresh_contrast = fit_score(gray, candidate.region, candidate.cell_px)
            fresh = square_colours(gray, candidate.region, candidate.cell_px)
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
        boards = self._kept(frame_bgr, self._detect(gray, full=full))
        board = boards[0] if boards else None
        self._lock(
            gray, board, square_colours(gray, board.region, board.cell_px) if board else None
        )
        if self.board:
            self._full.reset()
            self._near.reset()
        else:
            self._near.missed()
            if full:
                self._full.missed()
        return self.board

    def _lock(
        self, gray: np.ndarray, board: BoardCandidate | None, colours: Colours | None
    ) -> None:
        """A lock taken (or none): what it reads is its measure, and the probe starts fresh."""
        self.misses = 0
        self._frames_since_refine = 0
        self.board = board
        self.colours = colours
        if board is not None:
            self.last_cell_px = board.cell_px
        self._readable = (
            readable_cells(gray, board.region, board.cell_px, colours)
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
        for board in self._kept(frame_bgr, self._detect(gray, full=False, count=False)):
            if _edge_shift(board.region, lock.region) <= self.refit_deadband_px:
                continue
            colours = square_colours(gray, board.region, board.cell_px)
            if colours is None:
                continue
            fit, contrast = fit_score(gray, board.region, board.cell_px, colours)
            cells = readable_cells(gray, board.region, board.cell_px, colours)
            if self._passes(fit, contrast) and cells > self._readable - self.probe_lost_cells:
                self._lock(gray, replace(board, fit=fit, contrast=contrast), colours)
                self.detections += 1
                return self.board
        self._probe_wait.missed()
        return None

    def _detect(
        self, gray: np.ndarray, full: bool = True, count: bool = True
    ) -> list[BoardCandidate]:
        """Detection, trying the last known cell size first.

        A window that was moved keeps its cell size, so a narrow scale search around it
        re-locks in a fraction of the full multi-scale search. Only on failure, and only
        when `full` allows it, does the full range run.
        """
        if count:
            self.detections += 1
        if self.last_cell_px is not None:
            near = detect_boards_gray(
                gray,
                min_fit=self.min_fit,
                min_contrast=self.min_contrast,
                min_cell=max(MIN_CELL_PX, int(self.last_cell_px * 0.8)),
                max_cell=int(self.last_cell_px * 1.25) + 1,
            )
            if near:
                return near
        if not full:
            return []
        return detect_boards_gray(gray, min_fit=self.min_fit, min_contrast=self.min_contrast)
