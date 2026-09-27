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
its own and its boundary comb is clearly stronger than the one at the lock's geometry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from sidecoach.events import Region
from sidecoach.vision.board_detect import (
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
        self.redetect_first = redetect_first
        self.redetect_backoff = redetect_backoff
        self._full_wait = redetect_first
        self.near_backoff = near_backoff
        # A lock that still validates after the board moved away: the fit counts only the
        # cells that read as square colours, so a board scrolled two and a half cells kept
        # its old lock at fit 0.9 on the cells it left behind, and nothing was read again
        # (the parity audit, 2026-09-26). When a row's worth of the cells the lock could
        # read are gone, the cheap near-size detection looks for the board it lost; while
        # it finds nothing better (a dialog over the board) it waits longer each time.
        self.probe_lost_cells = probe_lost_cells
        self.probe_backoff = probe_backoff
        self.probe_backoff_max = probe_backoff_max
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
        self._cooldown = 0
        self._near_cooldown = 0
        self.probes = 0
        self._readable = 0  # the most cells this lock has read as square colours
        self._probe_cooldown = 0
        self._probe_wait = probe_backoff

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
        if self.board is not None:
            self._frames_since_refine += 1
            candidate = self.board
            fit, contrast = fit_score(gray, candidate.region, candidate.cell_px, self.colours)
            if not self._passes(fit, contrast) and self.colours is not None:
                # The theme may have changed between games with the board in place: a
                # fresh palette that validates, outer ring included, re-locks the colours.
                # A dialog over most of the board fails this too, since its palette is the
                # dialog, so it cannot rewrite the lock.
                # A new theme keeps the board where it was, its light squares the lighter
                # ones: a palette whose light and dark swap places is the board read a
                # cell off (moved 0.8 of a cell up, the lock re-read its colours inverted
                # and stayed where the board was: the parity audit's sweep, 2026-09-26).
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
            if self._frames_since_refine >= self.refine_every:
                self._frames_since_refine = 0
                refit = refine(
                    gray, candidate, offset_cells=self.refine_offset_cells, colours=self.colours
                )
                # A re-fit is adopted only when it validates on its own, and then only when
                # this frame failed at the lock's geometry or the re-fit's boundary comb is
                # clearly stronger than the comb at that geometry. The fit score cannot
                # arbitrate: it is blind to shifts of up to a third of a cell, so a drifted
                # geometry can still score 1.0 while the corrected one loses a cell to a
                # badge. A re-fit that fails never costs the lock; the region and the fit it
                # measured itself stand.
                refined = refit.board
                if self._passes(refined.fit, refined.contrast) and (
                    not self._passes(fit, contrast)
                    or (
                        refit.strength > self.refit_gain * refit.reference
                        and _edge_shift(candidate.region, refined.region) > self.refit_deadband_px
                    )
                ):
                    candidate = refined
                    fit, contrast = refined.fit, refined.contrast
            readable = (
                readable_cells(gray, candidate.region, candidate.cell_px, self.colours)
                if self.colours is not None and self._passes(fit, contrast)
                else 0
            )
            if self._passes(fit, contrast) and self._lost_cells(readable):
                moved = self._probe(gray, frame_bgr, candidate, readable)
                if moved is not None:
                    return moved
            if self._passes(fit, contrast):
                self.misses = 0
                self._readable = max(self._readable, readable)
                self.board = replace(candidate, fit=fit, contrast=contrast)
                self.last_cell_px = self.board.cell_px
                return self.board
            self.misses += 1
            if self.misses < self.max_misses:
                # Held, not verified: the geometry is the last good one, the fit is this
                # frame's, so a caller can see the lock is in doubt.
                self.board = replace(self.board, fit=fit, contrast=contrast)
                return self.board
            self.board = None
            self.colours = None
        else:
            # Without a board on screen, searching every frame would stall capture. The
            # search near the last cell size is cheap and runs every few frames, so a
            # window that comes back, or was covered for a moment, re-locks within a few
            # frames; the full multi-scale search, which costs over a second at 1440p,
            # backs off for longer.
            self._cooldown -= 1
            self._near_cooldown -= 1
            if self._cooldown > 0 and (self.last_cell_px is None or self._near_cooldown > 0):
                return None

        boards = self._detect(gray, full=self._cooldown <= 0)
        if self.reject is not None and boards:
            kept = [b for b in boards if not self.reject(frame_bgr, b.region, b.cell_px)]
            self.rejected += len(boards) - len(kept)
            boards = kept
        self.misses = 0
        self._frames_since_refine = 0
        self.board = boards[0] if boards else None
        self.colours = (
            square_colours(gray, self.board.region, self.board.cell_px) if self.board else None
        )
        self._adopted(gray)
        if self.board:
            self._cooldown = self._near_cooldown = 0
            self._full_wait = self.redetect_first
        else:
            self._near_cooldown = self.near_backoff
            if self._cooldown <= 0:
                self._cooldown = self._full_wait
                self._full_wait = min(2 * self._full_wait, self.redetect_backoff)
        return self.board

    def _adopted(self, gray: np.ndarray) -> None:
        """A lock was just taken: what it reads is its measure, and the probe starts fresh."""
        self._readable = (
            readable_cells(gray, self.board.region, self.board.cell_px, self.colours)
            if self.board is not None and self.colours is not None
            else 0
        )
        self._probe_cooldown = 0
        self._probe_wait = self.probe_backoff

    def _lost_cells(self, readable: int) -> bool:
        """Whether a row's worth of the cells this lock could read are gone; when they are
        back, a later loss is probed at once."""
        lost = readable <= self._readable - self.probe_lost_cells
        if not lost:
            self._probe_cooldown = 0
            self._probe_wait = self.probe_backoff
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
        if self._probe_cooldown > 0:
            self._probe_cooldown -= 1
            return None
        self.probes += 1
        boards = self._detect(gray, full=False, count=False)
        if self.reject is not None and boards:
            boards = [b for b in boards if not self.reject(frame_bgr, b.region, b.cell_px)]
        for board in boards:
            if _edge_shift(board.region, lock.region) <= self.refit_deadband_px:
                continue
            colours = square_colours(gray, board.region, board.cell_px)
            if colours is None:
                continue
            fit, contrast = fit_score(gray, board.region, board.cell_px, colours)
            cells = readable_cells(gray, board.region, board.cell_px, colours)
            if self._passes(fit, contrast) and cells > self._readable - self.probe_lost_cells:
                self.misses = 0
                self._frames_since_refine = 0
                self.board = replace(board, fit=fit, contrast=contrast)
                self.colours = colours
                self.last_cell_px = self.board.cell_px
                self.detections += 1
                self._adopted(gray)
                return self.board
        self._probe_cooldown = self._probe_wait
        self._probe_wait = min(self._probe_wait * 2, self.probe_backoff_max)
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
                min_cell=max(8, int(self.last_cell_px * 0.8)),
                max_cell=int(self.last_cell_px * 1.25) + 1,
            )
            if near:
                self.last_cell_px = near[0].cell_px
                return near
        if not full:
            return []
        boards = detect_boards_gray(gray, min_fit=self.min_fit, min_contrast=self.min_contrast)
        if boards:
            self.last_cell_px = boards[0].cell_px
        return boards
