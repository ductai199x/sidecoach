"""Keep a lock on a detected board across frames.

Full detection costs over a second on a 1440p frame, so it runs only when there is no lock.
While locked, every frame is validated with the cheap checker fit at the known region, which
takes milliseconds. Every `refine_every` frames the boundaries are re-fitted with a wide
search window, so a board that scrolled or was resized slightly is re-locked without a full
search. A board that slid by exactly one cell still passes the checker fit, which is why the
periodic re-fit exists. After `max_misses` consecutive failed validations the lock is dropped
and full detection runs again.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from sidecoach.vision.board_detect import (
    BoardCandidate,
    Colours,
    detect_boards_gray,
    fit_score,
    refine,
    square_colours,
    to_gray,
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
        redetect_backoff: int = 20,
        near_backoff: int = 3,
    ) -> None:
        self.min_fit = min_fit
        self.min_contrast = min_contrast
        self.max_misses = max_misses
        self.refine_every = refine_every
        self.refine_offset_cells = refine_offset_cells
        self.redetect_backoff = redetect_backoff
        self.near_backoff = near_backoff
        self.board: BoardCandidate | None = None
        self.colours: Colours | None = None
        self.last_cell_px: float | None = None
        self.misses = 0
        self.detections = 0
        self._frames_since_refine = 0
        self._cooldown = 0
        self._near_cooldown = 0

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
                fresh_fit, fresh_contrast = fit_score(gray, candidate.region, candidate.cell_px)
                if self._passes(fresh_fit, fresh_contrast):
                    self.colours = square_colours(gray, candidate.region, candidate.cell_px)
                    fit, contrast = fresh_fit, fresh_contrast
            if self._frames_since_refine >= self.refine_every:
                self._frames_since_refine = 0
                refined = refine(
                    gray, candidate, offset_cells=self.refine_offset_cells, colours=self.colours
                )
                # A re-fit is adopted only when it validates on its own; otherwise the lock
                # keeps its region and the fit it measured itself.
                if self._passes(refined.fit, refined.contrast) and (
                    refined.fit >= fit or not self._passes(fit, contrast)
                ):
                    candidate = refined
                    fit, contrast = refined.fit, refined.contrast
            if self._passes(fit, contrast):
                self.misses = 0
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
        self.misses = 0
        self._frames_since_refine = 0
        self.board = boards[0] if boards else None
        self.colours = (
            square_colours(gray, self.board.region, self.board.cell_px) if self.board else None
        )
        if self.board:
            self._cooldown = self._near_cooldown = 0
        else:
            self._near_cooldown = self.near_backoff
            if self._cooldown <= 0:
                self._cooldown = self.redetect_backoff
        return self.board

    def _detect(self, gray: np.ndarray, full: bool = True) -> list[BoardCandidate]:
        """Detection, trying the last known cell size first.

        A window that was moved keeps its cell size, so a narrow scale search around it
        re-locks in a fraction of the full multi-scale search. Only on failure, and only
        when `full` allows it, does the full range run.
        """
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
