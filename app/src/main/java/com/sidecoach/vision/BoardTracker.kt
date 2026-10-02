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
 * its boundary comb is clearly stronger than the one at the lock's geometry. A frame that
 * fails validation is re-fitted at once, before anything else is tried.
 */
class BoardTracker(
    val minFit: Double = 0.85,
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
    val refitDeadbandPx: Int = 2,
    /**
     * The full search that finds nothing waits before it runs again, since it costs a second
     * at 1440p: [redetectFirst] frames the first time, twice that the next, up to
     * [redetectBackoff]. A board lost for a moment is usually back within that moment at a
     * size the near search does not cover -- a window dragged small and large again, a board
     * that came back from off the screen resized -- and a flat twenty frames kept it unread
     * for two seconds while the moves went on (the desktop's geometry sweep, 2026-09-27: five
     * plies played before the board was seen again).
     */
    redetectFirst: Int = 5,
    redetectBackoff: Int = 20,
    /** The search near the last cell size is cheap: every [nearBackoff] frames. */
    nearBackoff: Int = 3,
    /**
     * A lock that still validates after the board moved away: the fit counts only the cells
     * that read as square colours, so a board scrolled two and a half cells kept its old lock
     * at fit 0.9 on the cells it left behind, and nothing was read again (the parity audit,
     * 2026-09-26). When a row's worth of the cells the lock could read are gone, the cheap
     * near-size detection looks for the board it lost; while it finds nothing better (a
     * dialog over the board) it waits longer each time.
     */
    private val probeLostCells: Int = 8,
    probeBackoff: Int = 10,
    probeBackoffMax: Int = 80,
) {

    /**
     * How long a search that found nothing waits before it runs again: [first] frames, twice
     * that after the next miss, up to [most]; a board found starts it over (`tracker._Wait`).
     */
    private class Wait(val first: Int, val most: Int) {
        private var left = 0 // frames still to wait
        private var next = first // the wait after the next miss

        /** Counts this frame; whether the wait is over. */
        fun ready(): Boolean {
            left--
            return left <= 0
        }

        fun missed() {
            left = next
            next = minOf(2 * next, most)
        }

        fun reset() {
            left = 0
            next = first
        }
    }

    private val fullWait = Wait(redetectFirst, redetectBackoff)
    private val nearWait = Wait(nearBackoff, nearBackoff)
    private val probeWait = Wait(probeBackoff, probeBackoffMax)

    var board: BoardCandidate? = null
        private set

    /** The square colours the lock was taken with; the theme belongs to these. */
    var colours: Colours? = null
        private set

    var detections = 0
        private set

    /** The near-size searches for a board the lock lost ([probe]), counted apart from [detections]. */
    var probes = 0
        private set

    /** Detections that ran the full multi-scale search. */
    var fullSearches = 0
        private set

    private var lastCellPx: Double? = null
    private var misses = 0
    private var framesSinceRefine = 0

    /**
     * The board's own pixels on the last frame that validated the lock (a copy of its region,
     * not the frame: a whole grey frame pinned for the life of a lock was 15 MB at 1440p on
     * the desktop -- round 3), and the occupied cells read there once the lock is lost: of the
     * boards found then, the one whose pieces are nearest those is taken (`tracker._seen`,
     * `_lost`).
     */
    internal var seen: Pair<Plane, BoardCandidate>? = null // internal: read by the tests
        private set
    private var lost: BooleanArray? = null

    /** The most cells this lock has read as square colours. */
    private var readable = 0

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

    private fun squareColours(gray: Plane, region: Region, cell: Double): Colours? =
        once(gray, listOf("colours", region, cell)) { BoardRefine.squareColours(gray, region, cell) }

    private fun readableCells(gray: Plane, region: Region, cell: Double, colours: Colours): Int =
        once(gray, listOf("readable", region, cell, colours)) { BoardRefine.readableCells(gray, region, cell, colours) }

    fun update(gray: Plane): BoardCandidate? {
        val held = board
        if (held == null) {
            // Without a board on screen, searching every frame would stall the capture. The
            // search near the last cell size is cheap and runs every few frames, so a window
            // that comes back, or was covered for a moment, re-locks within a few frames; the
            // full multi-scale search backs off for longer.
            val full = fullWait.ready()
            val near = nearWait.ready() && lastCellPx != null
            return if (full || near) search(gray, full) else null
        }
        val (candidate, fit, contrast) = check(gray)
        if (!passes(fit, contrast)) {
            misses++
            if (misses < maxMisses) {
                // Held, not verified: the geometry is the last good one and the fit is this
                // frame's, so a caller can see the lock is in doubt.
                board = held.copy(fit = fit, contrast = contrast)
                return board
            }
            board = null
            colours = null
            seen?.let { (seenGray, seenBoard) ->
                lost = BoardRefine.boardOccupancy(seenGray, seenBoard.region, seenBoard.cellPx)
            }
            seen = null
            return search(gray, true)
        }
        val now = colours
        val cells = if (now != null) readableCells(gray, candidate.region, candidate.cellPx, now) else 0
        if (lostCells(cells)) {
            val moved = probe(gray, candidate)
            if (moved != null) return moved
        }
        misses = 0
        readable = maxOf(readable, cells)
        val validated = candidate.copy(fit = fit, contrast = contrast)
        board = validated
        keepSeen(gray, validated)
        lastCellPx = validated.cellPx
        return board
    }

    /** The board's region of this frame, copied, with the board placed in it (`_keep_seen`). */
    private fun keepSeen(gray: Plane, at: BoardCandidate) {
        val x0 = maxOf(0, at.region.x)
        val y0 = maxOf(0, at.region.y)
        val w = maxOf(0, minOf(gray.width, at.region.x + at.region.w) - x0)
        val h = maxOf(0, minOf(gray.height, at.region.y + at.region.h) - y0)
        val pixels = FloatArray(w * h)
        for (y in 0 until h) System.arraycopy(gray.data, (y0 + y) * gray.width + x0, pixels, y * w, w)
        seen = Plane(w, h, pixels) to at.copy(region = at.region.copy(x = at.region.x - x0, y = at.region.y - y0))
    }

    /**
     * The lock on this frame: its geometry, fit and contrast. Re-fitted when due, and at once
     * on the first frame it fails; still failing, its palette is read afresh (`tracker._check`).
     */
    private fun check(gray: Plane): Triple<BoardCandidate, Double, Double> {
        framesSinceRefine++
        var candidate = checkNotNull(board)
        var (fit, contrast) = fitScore(gray, candidate.region, candidate.cellPx, colours)
        // A frame that fails is re-fitted at once, the first of a run of failures, with the
        // lock's own colours: a board that moved a fraction of a cell is likelier than a new
        // theme, and read where the lock is, its squares' colours come out mixed. Read afresh
        // there first, the mixed colours validated, the lock kept them, and the periodic re-fit
        // read the board itself in them as nothing: the lock stayed an eighth of a cell off for
        // good (the desktop's near-edge sweep, 2026-09-27).
        val failedFirst = !passes(fit, contrast) && misses == 0
        if (framesSinceRefine >= refineEvery || failedFirst) {
            framesSinceRefine = 0
            val refitColours = colours
            val atLock = candidate
            val refit = once(gray, listOf("refine", atLock, refineOffsetCells, refitColours)) {
                BoardRefine.refine(gray, atLock, offsetCells = refineOffsetCells, colours = refitColours)
            }
            // A re-fit is adopted only when it validates on its own, and then only when this
            // frame failed at the lock's geometry or the re-fit moved it and its boundary comb is
            // clearly stronger than the comb there. The fit score cannot arbitrate: it is blind
            // to shifts of up to a third of a cell, so a drifted geometry can still score 1.0
            // while the corrected one loses a cell to a badge. A re-fit that fails never costs
            // the lock; the region and the fit it measured itself stand.
            val refined = refit.board
            val moved = edgeShift(atLock.region, refined.region) > refitDeadbandPx
            if (passes(refined.fit, refined.contrast) &&
                (!passes(fit, contrast) || (moved && refit.strength > refitGain * refit.reference))
            ) {
                if (moved) {
                    // The board is moving: it is re-fitted the next frame too, until a re-fit
                    // finds it where the lock is. Fitted once mid-slide and left, the lock
                    // trailed the board by a quarter of a cell once it stopped, under what
                    // validation sees, until the next re-fit ten frames on, and no move was read
                    // (the desktop's geometry sweep, 2026-09-27, seed 83).
                    framesSinceRefine = refineEvery
                }
                candidate = refined
                fit = refined.fit
                contrast = refined.contrast
            }
        }
        val lockColours = colours
        if (!passes(fit, contrast) && lockColours != null) {
            // The theme may have changed between games with the board in place: a fresh
            // palette that validates, outer ring included, re-locks the colours. A dialog over
            // most of the board fails this too, since its palette is the dialog, so it cannot
            // rewrite the lock. A new theme keeps the board where it was, its light squares the
            // lighter ones: a palette whose light and dark swap places is the board read a cell
            // off (moved 0.8 of a cell up, the lock re-read its colours inverted and stayed
            // where the board was: the parity audit's sweep, 2026-09-26).
            val (freshFit, freshContrast) = fitScore(gray, candidate.region, candidate.cellPx, null)
            val fresh = squareColours(gray, candidate.region, candidate.cellPx)
            if (passes(freshFit, freshContrast) &&
                fresh != null &&
                (fresh.light > fresh.dark) == (lockColours.light > lockColours.dark)
            ) {
                colours = fresh
                fit = freshFit
                contrast = freshContrast
                readable = 0 // a new palette: what it reads is counted afresh
            }
        }
        return Triple(candidate, fit, contrast)
    }

    /** Detection with the lock gone; a board found is locked, and a miss backs off (`_search`). */
    private fun search(gray: Plane, full: Boolean): BoardCandidate? {
        val searched = fullSearches
        val found = detect(gray, full = full, lostGrid = lost).firstOrNull()
        lock(gray, found, if (found != null) squareColours(gray, found.region, found.cellPx) else null)
        if (board != null) {
            fullWait.reset()
            nearWait.reset()
        } else {
            nearWait.missed()
            if (fullSearches != searched) fullWait.missed()
        }
        return board
    }

    /**
     * A lock taken (or none): the last cell size is its, what it reads is its measure, and the
     * probe starts fresh (`tracker._lock`).
     */
    private fun lock(gray: Plane, found: BoardCandidate?, lockColours: Colours?) {
        misses = 0
        framesSinceRefine = 0
        board = found
        colours = lockColours
        if (found != null) {
            lastCellPx = found.cellPx
            keepSeen(gray, found)
            lost = null
        }
        readable = if (found != null && lockColours != null) {
            readableCells(gray, found.region, found.cellPx, lockColours)
        } else {
            0
        }
        probeWait.reset()
    }

    /**
     * Whether a row's worth of the cells this lock could read are gone; when they are back, a
     * later loss is probed at once.
     */
    private fun lostCells(cells: Int): Boolean {
        val lost = cells <= readable - probeLostCells
        if (!lost) probeWait.reset()
        return lost
    }

    /**
     * The board this lock lost, if the near-size detection finds it elsewhere: a board away
     * from the lock that validates in its own colours and reads what the lock read before it
     * lost its cells -- the whole board, not only more than now: under a window over part of
     * the board a region a cell off read more and was taken. Adopted as a new lock. Nothing
     * found (a dialog over the board, an arrow): the lock stands, and the next probe waits
     * twice as long. Counted in [probes], not [detections].
     */
    private fun probe(gray: Plane, lock: BoardCandidate): BoardCandidate? {
        if (!probeWait.ready()) return null
        probes++
        val boards = detect(gray, full = false, count = false)
        for (found in boards) {
            if (edgeShift(found.region, lock.region) <= refitDeadbandPx) continue
            val own = squareColours(gray, found.region, found.cellPx) ?: continue
            val (fit, contrast) = fitScore(gray, found.region, found.cellPx, own)
            val cells = readableCells(gray, found.region, found.cellPx, own)
            if (passes(fit, contrast) && cells > readable - probeLostCells) {
                lock(gray, found.copy(fit = fit, contrast = contrast), own)
                detections++
                return board
            }
        }
        probeWait.missed()
        return null
    }

    /**
     * Detection, trying the last known cell size first: a window that was moved keeps its
     * cell size, so a narrow scale search around it re-locks in a fraction of the full one.
     * A probe's search ([count] false) is not counted among the detections.
     *
     * With [lostGrid] (the occupied cells of the board a lock lost), the boards found are
     * ranked by how many cells their occupancy differs from it, nearest first: the game board
     * resized past the near range, with another board at the old size beside it, had the near
     * search stop at that board, and the lock stayed on it for good (round 2, finding 1); a
     * cut-off of a few cells still took the other board once a few moves were played while
     * the board was lost (round 3). The near search alone sees only the boards near the old
     * size, so it settles the matter by itself only for a board that shows exactly the lost
     * pieces -- the board moved, nothing played; any other board found calls the full search,
     * where every board on the screen is ranked (`tracker._detect`).
     */
    private fun detect(
        gray: Plane,
        full: Boolean,
        count: Boolean = true,
        lostGrid: BooleanArray? = null,
    ): List<BoardCandidate> {
        if (count) detections++
        var fullNow = full
        val last = lastCellPx
        if (last != null) {
            val minCell = maxOf(BoardDetect.MIN_CELL_PX, (last * 0.8).toInt())
            val maxCell = (last * 1.25).toInt() + 1
            var near = once(gray, listOf("detect", minCell, maxCell)) {
                BoardDetect.detect(gray, minFit = minFit, minContrast = minContrast, minCell = minCell, maxCell = maxCell)
            }
            if (lostGrid != null && near.isNotEmpty()) {
                val exact = nearestFirst(gray, near, lostGrid).filter { it.second == 0 }.map { it.first }
                // Boards found but none of them the lost one to the cell: only the full search
                // can say which board on the screen is nearest, so it runs now, whatever its
                // wait (a board uncovered after two plies under a window waited two seconds for
                // it, with the board in plain view).
                fullNow = full || exact.isEmpty()
                near = exact
            }
            if (near.isNotEmpty()) return near
        }
        if (!fullNow) return emptyList()
        fullSearches++
        val boards = once(gray, listOf("detect")) { BoardDetect.detect(gray, minFit = minFit, minContrast = minContrast) }
        if (lostGrid == null) return boards
        return nearestFirst(gray, boards, lostGrid).map { it.first }
    }

    /**
     * [boards] with how many cells their occupancy differs from the lost board's by (all 64 for
     * a board the frame does not show whole), nearest first, the detector's order kept among
     * equals (`_nearest_first`).
     */
    private fun nearestFirst(
        gray: Plane,
        boards: List<BoardCandidate>,
        lostGrid: BooleanArray,
    ): List<Pair<BoardCandidate, Int>> = boards.map { found ->
        val grid = BoardRefine.boardOccupancy(gray, found.region, found.cellPx)
        found to (if (grid == null) 64 else grid.indices.count { grid[it] != lostGrid[it] })
    }.sortedBy { it.second } // stable: the detector's order among equals
}

/** The most any edge of region [a] moves to become region [b], in pixels. */
private fun edgeShift(a: Region, b: Region): Int = maxOf(
    kotlin.math.abs(a.x - b.x),
    kotlin.math.abs(a.y - b.y),
    kotlin.math.abs((a.x + a.w) - (b.x + b.w)),
    kotlin.math.abs((a.y + a.h) - (b.y + b.h)),
)
