package com.sidecoach.vision

/**
 * Keep a lock on a detected board across frames, ported from
 * `sidecoach/vision/tracker.py`.
 *
 * Full detection costs a fifth of a second on a phone frame, so it runs only when there is
 * no lock. While locked, every frame is validated with the cheap checker fit at the known
 * region, which takes milliseconds. The fit reads the cells against the square colours the
 * lock was taken with, in their polarity, and rejects a pattern that carries on outside the
 * region, so a board that slid by a whole cell fails on the first frame; after [maxMisses]
 * consecutive failures the lock is dropped and detection runs again. What validation cannot
 * see is sub-cell drift and a small resize, so every [refineEvery] frames the boundaries are
 * re-fitted with a wide window, and the re-fit is adopted when it validates on its own and
 * its boundary comb is clearly stronger than the one at the lock's geometry.
 */
class BoardTracker(
    private val minFit: Double = 0.85,
    private val minContrast: Double = 6.0,
    private val maxMisses: Int = 3,
    private val refineEvery: Int = 10,
    private val refineOffsetCells: Double = 1.6,
    private val refitGain: Double = 1.1,
    /**
     * A re-fit that moves no edge by more than this is the same lock: a board flush with the
     * frame's edge re-fitted a pixel or two either way every few seconds, and every cell moved
     * under it was read again -- a second on the phone (Tai's puzzles on the Pixel,
     * 2026-09-23). Sub-cell drift that small costs the reading nothing.
     */
    private val refitDeadbandPx: Int = 2,
    private val redetectBackoff: Int = 20,
    private val nearBackoff: Int = 3,
) {

    var board: BoardCandidate? = null
        private set

    /** The square colours the lock was taken with; the theme belongs to these. */
    var colours: Colours? = null
        private set

    var detections = 0
        private set

    private var lastCellPx: Double? = null
    private var misses = 0
    private var framesSinceRefine = 0
    private var cooldown = 0
    private var nearCooldown = 0

    val locked: Boolean get() = board != null

    /** Whether the last frame validated the lock; false while the lock is only held. */
    val verified: Boolean get() = board != null && misses == 0

    private fun passes(fit: Double, contrast: Double) = fit >= minFit && contrast >= minContrast

    // A still screen hands the tracker the same grey image tick after tick. Everything it
    // computes from the image -- the fit at a geometry, the square colours, a re-fit, a search
    // -- is a pure function of the image and its arguments, so what was computed on this image
    // with these arguments is not computed again: the phone's frames on a board that is not
    // moving then cost next to nothing. The tracker's own bookkeeping -- the ticks since a
    // re-fit, the misses, the back-off -- still runs every tick, as the desktop's does, so its
    // decisions are the same.
    private var memoGray: Plane? = null
    private val memo = HashMap<Any, Any?>()

    private fun <T> once(gray: Plane, key: Any, compute: () -> T): T {
        if (gray !== memoGray) {
            memo.clear()
            memoGray = gray
        }
        if (memo.containsKey(key)) {
            @Suppress("UNCHECKED_CAST")
            return memo[key] as T
        }
        return compute().also { memo[key] = it }
    }

    private fun fitScore(gray: Plane, region: Region, cell: Double, colours: Colours?) =
        once(gray, listOf("fit", region, cell, colours)) { BoardRefine.fitScore(gray, region, cell, colours) }

    fun update(gray: Plane): BoardCandidate? {
        val held = board
        if (held != null) {
            framesSinceRefine++
            var candidate = held
            var (fit, contrast) = fitScore(gray, candidate.region, candidate.cellPx, colours)
            if (!passes(fit, contrast) && colours != null) {
                // The theme may have changed between games with the board in place: a
                // fresh palette that validates, outer ring included, re-locks the colours.
                // A dialog over most of the board fails this too, since its palette is the
                // dialog, so it cannot rewrite the lock.
                val (freshFit, freshContrast) = fitScore(gray, candidate.region, candidate.cellPx, null)
                if (passes(freshFit, freshContrast)) {
                    val region = candidate.region
                    val cell = candidate.cellPx
                    colours = once(gray, listOf("colours", region, cell)) {
                        BoardRefine.squareColours(gray, region, cell)
                    }
                    fit = freshFit
                    contrast = freshContrast
                }
            }
            if (framesSinceRefine >= refineEvery) {
                framesSinceRefine = 0
                val lockColours = colours
                val atLock = candidate
                val refit = once(gray, listOf("refine", atLock, refineOffsetCells, lockColours)) {
                    BoardRefine.refine(gray, atLock, offsetCells = refineOffsetCells, colours = lockColours)
                }
                // A re-fit is adopted only when it validates on its own, and then only when
                // this frame failed at the lock's geometry or the re-fit's comb is clearly
                // stronger. The fit score cannot arbitrate: it is blind to shifts of up to
                // a third of a cell.
                val refined = refit.board
                if (passes(refined.fit, refined.contrast) &&
                    (
                        !passes(fit, contrast) ||
                            (
                                refit.strength > refitGain * refit.reference &&
                                    edgeShift(atLock.region, refined.region) > refitDeadbandPx
                                )
                        )
                ) {
                    candidate = refined
                    fit = refined.fit
                    contrast = refined.contrast
                }
            }
            if (passes(fit, contrast)) {
                misses = 0
                board = candidate.copy(fit = fit, contrast = contrast)
                lastCellPx = board?.cellPx
                return board
            }
            misses++
            if (misses < maxMisses) {
                // Held, not verified: the geometry is the last good one and the fit is this
                // frame's, so a caller can see the lock is in doubt.
                board = held.copy(fit = fit, contrast = contrast)
                return board
            }
            board = null
            colours = null
        } else {
            // Without a board on screen, searching every frame would stall the capture. The
            // search near the last cell size is cheap and runs every few frames; the full
            // multi-scale search backs off for longer.
            cooldown--
            nearCooldown--
            if (cooldown > 0 && (lastCellPx == null || nearCooldown > 0)) return null
        }

        val boards = detect(gray, full = cooldown <= 0)
        misses = 0
        framesSinceRefine = 0
        board = boards.firstOrNull()
        val found = board
        colours = if (found != null) {
            once(gray, listOf("colours", found.region, found.cellPx)) {
                BoardRefine.squareColours(gray, found.region, found.cellPx)
            }
        } else {
            null
        }
        if (found != null) {
            cooldown = 0
            nearCooldown = 0
        } else {
            nearCooldown = nearBackoff
            if (cooldown <= 0) cooldown = redetectBackoff
        }
        return board
    }

    /**
     * Detection, trying the last known cell size first: a window that was moved keeps its
     * cell size, so a narrow scale search around it re-locks in a fraction of the full one.
     */
    private fun detect(gray: Plane, full: Boolean): List<BoardCandidate> {
        detections++
        val last = lastCellPx
        if (last != null) {
            val minCell = maxOf(8, (last * 0.8).toInt())
            val maxCell = (last * 1.25).toInt() + 1
            val near = once(gray, listOf("detect", minCell, maxCell)) {
                BoardDetect.detect(gray, minFit = minFit, minContrast = minContrast, minCell = minCell, maxCell = maxCell)
            }
            if (near.isNotEmpty()) {
                lastCellPx = near[0].cellPx
                return near
            }
        }
        if (!full) return emptyList()
        val boards = once(gray, listOf("detect")) { BoardDetect.detect(gray, minFit = minFit, minContrast = minContrast) }
        if (boards.isNotEmpty()) lastCellPx = boards[0].cellPx
        return boards
    }
}

/** The most any edge of region [a] moves to become region [b], in pixels. */
private fun edgeShift(a: Region, b: Region): Int = maxOf(
    kotlin.math.abs(a.x - b.x),
    kotlin.math.abs(a.y - b.y),
    kotlin.math.abs((a.x + a.w) - (b.x + b.w)),
    kotlin.math.abs((a.y + a.h) - (b.y + b.h)),
)
