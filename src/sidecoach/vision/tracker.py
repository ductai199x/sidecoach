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
    detect_boards_gray,
    fit_score,
    refine,
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
    ) -> None:
        self.min_fit = min_fit
        self.min_contrast = min_contrast
        self.max_misses = max_misses
        self.refine_every = refine_every
        self.refine_offset_cells = refine_offset_cells
        self.redetect_backoff = redetect_backoff
        self.board: BoardCandidate | None = None
        self.last_cell_px: float | None = None
        self.misses = 0
        self.detections = 0
        self._frames_since_refine = 0
        self._cooldown = 0

    @property
    def locked(self) -> bool:
        return self.board is not None

    def update(self, frame_bgr: np.ndarray) -> BoardCandidate | None:
        gray = to_gray(frame_bgr)
        if self.board is not None:
            self._frames_since_refine += 1
            candidate = self.board
            if self._frames_since_refine >= self.refine_every:
                refined = refine(gray, candidate, offset_cells=self.refine_offset_cells)
                self._frames_since_refine = 0
                fit, contrast = fit_score(gray, candidate.region, candidate.cell_px)
                # A dialog's straight edges can attract the re-fit; keep the refined
                # boundaries only when they fit at least as well as the current lock.
                if refined.fit >= fit - 0.02:
                    candidate = refined
                    fit, contrast = refined.fit, refined.contrast
            else:
                fit, contrast = fit_score(gray, candidate.region, candidate.cell_px)
            if fit >= self.min_fit and contrast >= self.min_contrast:
                self.misses = 0
                self.board = replace(candidate, fit=fit, contrast=contrast)
                self.last_cell_px = self.board.cell_px
                return self.board
            self.misses += 1
            if self.misses < self.max_misses:
                return self.board
            self.board = None
        elif self._cooldown > 0:
            # A full search costs over a second; without a board on screen, retrying every
            # frame would stall capture, so failed searches back off for a few frames.
            self._cooldown -= 1
            return None

        boards = self._detect(gray)
        self.misses = 0
        self._frames_since_refine = 0
        self.board = boards[0] if boards else None
        self._cooldown = 0 if self.board else self.redetect_backoff
        return self.board

    def _detect(self, gray: np.ndarray) -> list[BoardCandidate]:
        """Full detection, trying the last known cell size first.

        A window that was moved keeps its cell size, so a narrow scale search around it
        re-locks in a fraction of the full multi-scale search. Only on failure, or when there
        was no previous lock, does the full range run.
        """
        self.detections += 1
        if self.last_cell_px is not None:
            near = detect_boards_gray(
                gray,
                min_cell=max(8, int(self.last_cell_px * 0.8)),
                max_cell=int(self.last_cell_px * 1.25) + 1,
            )
            if near:
                self.last_cell_px = near[0].cell_px
                return near
        boards = detect_boards_gray(gray)
        if boards:
            self.last_cell_px = boards[0].cell_px
        return boards
