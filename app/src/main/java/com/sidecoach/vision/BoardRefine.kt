package com.sidecoach.vision

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sign

/** The two square colours a board is read with, and how far a cell may stray from either. */
data class Colours(val light: Double, val dark: Double, val tolerance: Double)

/**
 * A candidate after the boundary fit, with the comb strength at its new and old geometry.
 *
 * [strength] is the summed boundary gradient at the teeth of the comb (the boundaries between
 * the cells) on both axes; [reference] is the same sum at the geometry the fit started from,
 * over the same profiles, so the two compare: a re-fit that found nothing better than where it
 * started scores about the same, one that corrected a drift or a resize scores clearly more.
 */
data class Refinement(val board: BoardCandidate, val strength: Double, val reference: Double)

/**
 * Passes two and three of the detector, ported from `sidecoach/vision/board_detect.py`:
 * the sub-pixel boundary fit, and the validation that decides whether a candidate is a
 * board at all.
 *
 * The comb on the seven boundaries between the cells places the board to within a whole
 * cell, and of the placements a cell apart the one whose cells read most like a board is
 * kept. Each cell's colour is read as the median of a thin border ring, which pieces and
 * highlights rarely disturb; a cell the frame cuts is read on the part it shows when that is
 * half the cell or more. A candidate whose pattern carries on in the ring of cells just
 * outside it is rejected, and so is a board the frame does not show whole.
 *
 * The comments that matter are the ones that record a failure, and they are kept from the
 * Python.
 */
object BoardRefine {

    /** The comb's teeth: the seven boundaries between a board's cells, in cells from its edge. */
    private val TEETH = DoubleArray(7) { (it + 1).toDouble() }

    fun cellBounds(region: Region, cell: Double, r: Int, c: Int): IntArray {
        val x0 = pyRound(region.x + c * cell)
        val y0 = pyRound(region.y + r * cell)
        val x1 = pyRound(region.x + (c + 1) * cell)
        val y1 = pyRound(region.y + (r + 1) * cell)
        return intArrayOf(x0, y0, x1, y1)
    }

    fun ringBand(cell: Double): Int = max(1, pyRound(cell * 0.12))

    /**
     * The border-ring medians of a board's cells and of [pad] rings of cells around it, an
     * (8 + 2 pad)-square grid, row by row (`board_detect._cells`). A cell the frame cuts keeps
     * the part of its ring the frame shows, when that is half the cell or more both ways, and
     * is NaN when it is less.
     */
    fun cells(gray: Plane, region: Region, cell: Double, pad: Int = 1): DoubleArray {
        val band = ringBand(cell)
        val side = 8 + 2 * pad
        val out = DoubleArray(side * side) { Double.NaN }
        var values = FloatArray(0)
        for (r in 0 until side) {
            for (c in 0 until side) {
                val (x0, y0, x1, y1) = cellBounds(region, cell, r - pad, c - pad)
                val cx0 = max(0, x0)
                val cy0 = max(0, y0)
                val cx1 = min(gray.width, x1)
                val cy1 = min(gray.height, y1)
                if (2 * (cx1 - cx0) < x1 - x0 || 2 * (cy1 - cy0) < y1 - y0) continue
                val size = max(0, cx1 - cx0) * max(0, cy1 - cy0)
                if (values.size < size) values = FloatArray(size)
                var n = 0
                for (y in cy0 until cy1) {
                    val row = y * gray.width
                    val onTopOrBottom = y < y0 + band || y >= y1 - band
                    for (x in cx0 until cx1) {
                        if (onTopOrBottom || x < x0 + band || x >= x1 - band) values[n++] = gray.data[row + x]
                    }
                }
                if (n > 0) out[r * side + c] = median(values, n)
            }
        }
        return out
    }

    /**
     * A board is read only when the frame shows it, to within this many pixels a side: a
     * board flush with the frame's edge is fitted a pixel past it now and then. Of a board
     * further off, the cells the frame cuts would reach the pieces' reader squeezed, so it is
     * not locked; it is found when it is back (a partly shown board was locked a cell inward
     * instead, on the window of part board and part background: the desktop's lock_bench,
     * 2026-09-27, 7 boards of 17).
     */
    const val EDGE_SLACK_PX = 2

    fun onFrame(gray: Plane, region: Region): Boolean =
        region.x >= -EDGE_SLACK_PX &&
            region.y >= -EDGE_SLACK_PX &&
            region.x + region.w <= gray.width + EDGE_SLACK_PX &&
            region.y + region.h <= gray.height + EDGE_SLACK_PX

    /**
     * The square colours read from a grid of 64 cell medians, with their tolerance; null when
     * the frame shows fewer than [BoardDetect.MIN_READABLE_CELLS] of the cells. The tolerance
     * follows how much the cells of each colour vary, so a textured wood theme keeps its
     * cells while a dialog or a highlight over a cell is left out of the fit.
     */
    fun palette(medians: DoubleArray): Colours? {
        var shown = 0
        for (v in medians) if (!v.isNaN()) shown++
        if (shown < BoardDetect.MIN_READABLE_CELLS) return null
        val lights = ArrayList<Double>(32)
        val darks = ArrayList<Double>(32)
        for (i in 0 until 64) {
            if (medians[i].isNaN()) continue
            if (BoardDetect.SIGNS[i] > 0) lights.add(medians[i]) else darks.add(medians[i])
        }
        val light = median(lights)
        val dark = median(darks)
        val spread = max(
            median(lights.map { abs(it - light) }),
            median(darks.map { abs(it - dark) }),
        )
        return Colours(light, dark, min(max(6.0, 3.0 * spread), 0.35 * abs(light - dark)))
    }

    /** Cells that read as either square colour; a cell the frame does not show reads as neither. */
    fun readableAgainst(medians: DoubleArray, colours: Colours): BooleanArray =
        BooleanArray(medians.size) {
            abs(medians[it] - colours.light) <= colours.tolerance ||
                abs(medians[it] - colours.dark) <= colours.tolerance
        }

    /**
     * The sign of (median - mid-grey) the pattern expects in each cell. Which colour is
     * brighter is the theme's business, so the pattern is inverted when the light squares
     * are the darker ones.
     */
    fun expectedSigns(colours: Colours): FloatArray =
        if (colours.light > colours.dark) BoardDetect.SIGNS
        else FloatArray(64) { -BoardDetect.SIGNS[it] }

    /**
     * Whether the checker pattern carries on in the ring of cells around a board ([cells],
     * the board and its ring as [cells] reads them, 10 by 10) on any of its sides.
     *
     * A candidate shifted by whole cells, or a window of a larger checkerboard, reads a
     * perfect board inside its 64 cells; only the cells just outside tell. A side continues
     * the board when most of its eight outer cells read as square colours and alternate. A
     * cell the frame does not show does not read, so a board flush with the frame's edge
     * passes; one the frame cuts is read on what it shows, when that is half the cell or more:
     * a candidate two thirds of a cell above a board that ended two pixels above the frame's
     * bottom had its row of outer cells above it off the frame and the row below it a third
     * off, so neither side was read, the board's last row went unseen below it, and the lock
     * validated with every cell reading across two squares (the desktop's geometry sweep,
     * 2026-09-27, seed 260).
     */
    private fun continues(cells: DoubleArray, colours: Colours): Boolean {
        val readable = readableAgainst(cells, colours)
        val mid = 0.5 * (colours.light + colours.dark)
        val sides = listOf(
            IntArray(8) { 0 * 10 + (it + 1) },
            IntArray(8) { 9 * 10 + (it + 1) },
            IntArray(8) { (it + 1) * 10 + 0 },
            IntArray(8) { (it + 1) * 10 + 9 },
        )
        for (side in sides) {
            var read = 0
            for (i in side) if (readable[i]) read++
            if (read < 6) continue
            var alternating = 0
            for (k in 0 until 7) {
                val a = side[k]
                val b = side[k + 1]
                if (readable[a] && readable[b] && (cells[a] > mid) != (cells[b] > mid)) alternating++
            }
            if (alternating >= 5) return true
        }
        return false
    }

    /** How a board reads ([judge]); [colours] is the palette it was read with. */
    private class Reading(
        val fit: Double,
        val contrast: Double,
        val readable: Int,
        val evidence: Int,
        val colours: Colours?,
    )

    /** The 64 cells of a board out of its 10-by-10 grid with the ring. */
    private fun inner(cells: DoubleArray): DoubleArray = DoubleArray(64) { cells[(it / 8 + 1) * 10 + it % 8 + 1] }

    /**
     * How a board reads against [colours], or in its own ([palette]) without them; [cells] is
     * the board and its ring, 10 by 10, as [cells] reads them (`board_detect._judge`).
     *
     * The fit is the fraction of the readable cells that match the checker pattern: zero when
     * fewer than [BoardDetect.MIN_READABLE_CELLS] read, or when the pattern carries on outside
     * the board. The evidence weighs placements a cell apart against each other: the matching
     * cells less the cells the frame shows that do not match; a cell the frame does not show
     * counts neither way. Read in its own colours, a board must also match on most of the
     * cells the frame shows: a window of part board and part background reads a palette of
     * blends from itself, and matched on the few cells that happened to fall within it (26 of
     * 64, the desktop's lock_bench case 106, 2026-09-27). A lock's colours are exempt, so a
     * dialog over most of the board does not cost it.
     */
    private fun judge(cells: DoubleArray, colours: Colours?): Reading {
        val board = inner(cells)
        val read = colours ?: palette(board)
        var shown = 0
        for (v in board) if (!v.isNaN()) shown++
        if (read == null) return Reading(0.0, 0.0, 0, -shown, null)
        // The threshold sits between the two square colours rather than at the board mean,
        // which a dark dialog over a light board would drag below the dark squares.
        val mid = 0.5 * (read.light + read.dark)
        val readable = readableAgainst(board, read)
        val expected = expectedSigns(read)
        var count = 0
        var matched = 0
        for (i in 0 until 64) {
            if (!readable[i]) continue
            count++
            if (sign(board[i] - mid).toFloat() == expected[i]) matched++
        }
        val evidence = 2 * matched - shown
        if (count < BoardDetect.MIN_READABLE_CELLS) return Reading(0.0, 0.0, count, evidence, read)
        val deviations = DoubleArray(count)
        var n = 0
        for (i in 0 until 64) if (readable[i]) deviations[n++] = abs(board[i] - mid)
        val contrast = Numpy.sum(deviations) / count
        var fit = matched.toDouble() / count
        if ((colours == null && evidence <= 0) || (fit >= 0.5 && continues(cells, read))) fit = 0.0
        return Reading(fit, contrast, count, evidence, read)
    }

    /**
     * How many of the 64 cells at [region] read as one of [colours]' square colours.
     *
     * The fit counts only these, so a board that moved away by two cells and a fraction still
     * fits where it was, on the cells it left behind; what gives it away is how few are left
     * to read (the tracker's probe).
     */
    fun readableCells(gray: Plane, region: Region, cell: Double, colours: Colours): Int {
        if (!onFrame(gray, region)) return 0
        return readableAgainst(cells(gray, region, cell, pad = 0), colours).count { it }
    }

    /**
     * (fraction of readable cells matching the checker pattern, contrast).
     *
     * With [colours] (the square colours a lock was taken with) cells are read against them,
     * so a dialog over most of the board cannot rewrite the palette, and the pattern must
     * keep the lock's polarity, so a board that slid by a whole cell reads every cell wrong.
     * Without them the palette is read from the candidate itself, in whichever polarity the
     * theme has, and the cells just outside the candidate decide whether it is shifted. A
     * board the frame does not show whole scores zero.
     */
    fun fitScore(gray: Plane, region: Region, cell: Double, colours: Colours? = null): Pair<Double, Double> {
        if (!onFrame(gray, region)) return 0.0 to 0.0
        val reading = judge(cells(gray, region, cell), colours)
        return reading.fit to reading.contrast
    }

    /**
     * The square colours of a board as read now, for a lock to keep; null for a board the
     * frame does not show whole.
     */
    fun squareColours(gray: Plane, region: Region, cell: Double): Colours? {
        if (!onFrame(gray, region)) return null
        return palette(cells(gray, region, cell, pad = 0))
    }

    /**
     * Readable cells whose interior mean differs from their ring median by over the
     * tolerance: pieces and marks. Well above zero on a game board, near zero on an empty
     * one, and it only breaks ties. [medians] are the board's 64 cells.
     */
    fun occupiedCells(
        gray: Plane,
        region: Region,
        cell: Double,
        medians: DoubleArray,
        colours: Colours,
    ): Int {
        val band = ringBand(cell)
        val readable = readableAgainst(medians, colours)
        var count = 0
        for (r in 0 until 8) {
            for (c in 0 until 8) {
                if (!readable[r * 8 + c]) continue
                val (x0, y0, x1, y1) = cellBounds(region, cell, r, c)
                // `gray[y0 + band : y1 - band, x0 + band : x1 - band]`, as numpy slices it.
                val rows = Numpy.slice(y0 + band, y1 - band, gray.height)
                val cols = Numpy.slice(x0 + band, x1 - band, gray.width)
                if (rows.isEmpty() || cols.isEmpty()) continue
                var sum = 0.0
                for (y in rows) {
                    val row = y * gray.width
                    for (x in cols) sum += gray.data[row + x]
                }
                // The mean of a float32 image in float32, as numpy takes it: the sum of whole
                // grey levels is exact there, the division is float32's.
                val n = (rows.last - rows.first + 1) * (cols.last - cols.first + 1)
                val mean = (sum.toFloat() / n.toFloat()).toDouble()
                if (abs(mean - medians[r * 8 + c]) > colours.tolerance) count++
            }
        }
        return count
    }

    /**
     * Which pixels of [crop] (whose top-left is ([xLo], [yLo]) in the frame) take part in the
     * boundary fit around [board], as a weight of 0 or 1 (`board_detect._boundary_weight`).
     *
     * Only the strips around the cell boundaries take part, and in them only the pixels of
     * the square's own colour. Pieces sit in the middle of their squares and their outlines
     * out-gradient faint boundaries on a low-contrast theme; a dialog over part of the board
     * has straight edges and text that would pull the grid towards them. The mask repeats
     * with the cell pitch, so a board that slid by a whole cell still has its boundaries in
     * unmasked strips, and everything outside the board, such as its outer edge, stays. A
     * real set's pieces reach into the strips (their bases sit at 0.87 of the cell), and a
     * rank of them draws one line as strong as a boundary a tenth of a cell above it: at
     * 144-pixel cells the comb settled between the two, a pitch two percent long and the top
     * edge a quarter of a cell off. So within the board a pixel that is not its cell's own
     * colour, to the palette's tolerance, takes no part: a piece's fill and outline are gone,
     * so are the labels and a dialog. The blends at a boundary, between the two square
     * colours, lie on the grid lines (within [band] of them) and are kept there: without them
     * the comb drifted a third of a percent. A pixel of a piece's edge is not: eight bases in
     * a rank, each one pixel of square kept beside black, summed to three boundaries' worth
     * and pulled the comb a tenth of a cell.
     *
     * A cell that does not read as a square colour, under a dialog or a highlight, keeps only
     * a thin strip along its boundaries. When too few cells read to tell the squares' own
     * colour at all, the geometry is too far off to lay the mask on (a lock a quarter of a
     * cell off on a textured theme reads most cells across two squares): its strips hid the
     * boundaries themselves and kept the pieces' edges, and the comb settled nine percent long
     * (the desktop's lock_bench, 2026-09-27). Then nothing is left out. The boundaries run the
     * width of the board, the pieces' edges do not, and that pass finds the boundaries near
     * enough for the next to lay the mask.
     */
    private fun boundaryWeight(
        crop: Plane,
        xLo: Int,
        yLo: Int,
        board: BoardCandidate,
        medians: DoubleArray,
        colours: Colours?,
        band: Int,
    ): FloatArray {
        val w = crop.width
        val h = crop.height
        val region = board.region
        val cell = board.cellPx
        val weight = FloatArray(w * h) { 1f }
        if (colours == null) return weight
        val readable = readableAgainst(medians, colours)
        if (readable.count { it } < BoardDetect.MIN_READABLE_CELLS) return weight
        val own = BooleanArray(w * h)
        for (r in 0 until 8) {
            for (c in 0 until 8) {
                val (bx0, by0, bx1, by1) = cellBounds(region, cell, r, c)
                val x0 = max(0, bx0 - xLo)
                val y0 = max(0, by0 - yLo)
                val x1 = min(w, bx1 - xLo)
                val y1 = min(h, by1 - yLo)
                if (x1 <= x0 || y1 <= y0) continue
                val target = medians[r * 8 + c]
                for (y in y0 until y1) {
                    val row = y * w
                    for (x in x0 until x1) own[row + x] = abs(crop.data[row + x] - target) <= colours.tolerance
                }
            }
        }
        // numpy compares a float32 image with a Python float in float32.
        val lo = (min(colours.light, colours.dark) - colours.tolerance).toFloat()
        val hi = (max(colours.light, colours.dark) + colours.tolerance).toFloat()
        val lineCols = BooleanArray(w)
        val lineRows = BooleanArray(h)
        for (k in 1 until 8) {
            val gx = pyRound(region.x - xLo + k * cell)
            val gy = pyRound(region.y - yLo + k * cell)
            for (x in Numpy.slice(max(0, gx - band), gx + band + 1, w)) lineCols[x] = true
            for (y in Numpy.slice(max(0, gy - band), gy + band + 1, h)) lineRows[y] = true
        }
        val x0 = max(0, region.x - xLo)
        val y0 = max(0, region.y - yLo)
        val x1 = min(w, region.x - xLo + pyRound(8 * cell))
        val y1 = min(h, region.y - yLo + pyRound(8 * cell))
        for (y in Numpy.slice(y0, y1, h)) {
            val row = y * w
            for (x in Numpy.slice(x0, x1, w)) {
                val i = row + x
                val value = crop.data[i]
                val kept = own[i] || (value >= lo && value <= hi && (lineCols[x] || lineRows[y]))
                weight[i] = if (kept) 1f else 0f
            }
        }
        for (r in 0 until 8) {
            for (c in 0 until 8) {
                val inset = pyRound(cell * (if (readable[r * 8 + c]) 0.2 else 0.08))
                val (bx0, by0, bx1, by1) = cellBounds(region, cell, r, c)
                val cx0 = bx0 - xLo + inset
                val cy0 = by0 - yLo + inset
                val cx1 = bx1 - xLo - inset
                val cy1 = by1 - yLo - inset
                if (cx1 <= cx0 || cy1 <= cy0) continue
                for (y in Numpy.slice(max(0, cy0), cy1, h)) {
                    val row = y * w
                    for (x in Numpy.slice(max(0, cx0), cx1, w)) weight[row + x] = 0f
                }
            }
        }
        return weight
    }

    /**
     * The summed step across every pixel edge; axis 1 gives a profile along x.
     *
     * Entry k is the edge between pixels k - 1 and k, where a board whose cells start at pixel
     * k has its boundary, so a tooth lands on the boundary itself and not half a pixel to one
     * side (entry 0 reads nothing). A step counts only where both its pixels take part
     * ([weight]): the mask leaves out the cells' middles, where the pieces are, and within the
     * board whatever is not the square's own colour.
     */
    fun gradientProfile(plane: Plane, axis: Int, weight: FloatArray): DoubleArray {
        val w = plane.width
        val h = plane.height
        val data = plane.data
        if (axis == 1) {
            val out = DoubleArray(w)
            for (y in 0 until h) {
                val row = y * w
                for (x in 1 until w) {
                    val i = row + x
                    out[x] += (abs(data[i] - data[i - 1]) * (weight[i] * weight[i - 1])).toDouble()
                }
            }
            return out
        }
        val out = DoubleArray(h)
        for (y in 1 until h) {
            val row = y * w
            var total = 0.0
            for (x in 0 until w) {
                val i = row + x
                total += (abs(data[i] - data[i - w]) * (weight[i] * weight[i - w])).toDouble()
            }
            out[y] = total
        }
        return out
    }

    /**
     * Summed profile at the teeth of a board whose first cell starts at [offset]; a tooth past
     * either end of the profile reads nothing.
     */
    fun combStrength(profile: DoubleArray, offset: Double, pitch: Double): Double {
        var total = 0.0
        for (t in TEETH) total += Numpy.interp(profile, offset + pitch * t)
        return total
    }

    /** The profile under a triangle [half] pixels to either side, its peaks where they were. */
    private fun blurred(profile: DoubleArray, half: Int): DoubleArray {
        val window = Numpy.bartlett(2 * half + 3)
        val kernel = window.copyOfRange(1, window.size - 1)
        val total = Numpy.sum(kernel)
        for (i in kernel.indices) kernel[i] = kernel[i] / total
        return Numpy.convolveSame(profile, kernel)
    }

    /**
     * The (offset, pitch) of the best comb over a gradient profile: the offset within
     * [offsetCells] cells of [start], the pitch within [BoardDetect.PITCH_SEARCH] of [pitch].
     *
     * The teeth are the boundaries between the cells, which show whatever lies around the
     * board. Its outer edges are not teeth: against the frame's edge or past it, or on a
     * background of one of the square colours, they show nothing, and a comb that counted
     * them favoured the placement a cell over (a board flush with the frame's edge, or a cell
     * from it, re-fitted a cell off: the desktop's geometry sweep, 2026-09-27). So the comb
     * places the board to within a whole cell: of the placements a cell apart it returns the
     * one nearest [start], and [refine] picks the cell. (Its passes, each started where the
     * last ended, walked two cells off a board past the frame's edge when each took the
     * strongest.)
     *
     * Coarse to fine: whole pixels and one-percent pitch steps over the whole window first,
     * then quarter pixels and 0.2 percent steps around the best, which is twenty times
     * cheaper than a fine grid everywhere and lands on the same optimum. The fine pass
     * searches offsets as far as a coarse pitch step moves the far teeth: a pitch a step off
     * wants its offset a few pixels off to hold the middle of the board, and a window of one
     * pixel kept the comb there, three pixels early and a quarter of a percent long at
     * 144-pixel cells. A boundary's peak is a pixel wide, and each pass reads the profile
     * blurred by as far as its grid lets a tooth fall from a boundary: the coarse pass's last
     * tooth half a pitch step times seven, which at 105-pixel cells is four pixels, and a comb
     * at the board's own pitch read its far boundaries as nothing, lost to one on the pieces'
     * edges, and the re-fit settled seven percent short (the desktop's lock_bench,
     * 2026-09-27); the fine pass's, a pixel.
     */
    fun combFit(
        profile: DoubleArray,
        start: Double,
        pitch: Double,
        offsetCells: Double,
    ): Pair<Double, Double> {
        val step = max(0.25, pitch * 0.01)

        /** The best comb over [seen]: the first of the strongest, and nothing below zero. */
        fun bestOver(seen: DoubleArray, pitches: DoubleArray, offsets: DoubleArray): Pair<Double, Double> {
            var bestOffset = start
            var bestPitch = pitch
            var bestStrength = 0.0
            for (p in pitches) {
                var top = -1
                var topStrength = 0.0
                for (i in offsets.indices) {
                    val o = offsets[i]
                    var strength = 0.0
                    for (t in TEETH) strength += Numpy.interp(seen, o + p * t)
                    if (top < 0 || strength > topStrength) {
                        top = i
                        topStrength = strength
                    }
                }
                if (top >= 0 && topStrength > bestStrength) {
                    bestOffset = offsets[top]
                    bestPitch = p
                    bestStrength = topStrength
                }
            }
            return bestOffset to bestPitch
        }

        val reach = pitch * offsetCells
        val (o, p) = bestOver(
            blurred(profile, ceil(3.5 * step).toInt()),
            Numpy.arange(pitch * (1 - BoardDetect.PITCH_SEARCH), pitch * (1 + BoardDetect.PITCH_SEARCH), step),
            Numpy.arange(start - reach, start + reach, 1.0),
        )
        val span = max(1.0, 4.0 * step)
        val (fo, fp) = bestOver(
            blurred(profile, 1),
            Numpy.arange(p * 0.99, p * 1.01, max(0.05, p * 0.002)),
            Numpy.arange(o - span, o + span + 0.01, 0.25),
        )
        return fo - fp * Math.rint((fo - start) / fp) to fp
    }

    /**
     * The placements a cell apart that [refine] weighs, (dx, dy), the comb's own first so a
     * tie keeps it: `sorted(product((-1, 0, 1), repeat=2), key=|dx| + |dy|)`.
     */
    private val PHASES: List<Pair<Int, Int>> = buildList {
        for (dx in -1..1) for (dy in -1..1) add(dx to dy)
    }.sortedBy { abs(it.first) + abs(it.second) }

    /**
     * Sub-pixel boundaries from gradient peaks, the whole-cell phase from the cells, then
     * validation. [offsetCells] is how far, in cells, the boundaries may move from the
     * candidate. The detector needs less than a cell, since a coarse candidate is within a
     * few pixels; the tracker's periodic re-fit uses a wider window and exists for what
     * validation cannot see, sub-cell drift and a small resize. With [colours] the cells are
     * read against a lock's palette.
     *
     * The comb places the board to within a whole cell ([combFit]). Of the placements a cell
     * apart on each axis, the one kept validates and has the most evidence ([judge]): a board
     * partly off the frame and the window a cell inward, part board and part background, match
     * as many cells, and the window's row of background is what tells them apart. The mask the
     * boundaries are fitted through is laid out on a geometry, and a candidate a cell pitch or
     * two percent off lays it a fraction of a cell off at the far side, where it then masks
     * the boundaries themselves and keeps its own error: the comb settled two percent long
     * there once. So the fit runs again, up to twice, while a pass moves it, each time with
     * the mask laid out on the last pass's geometry.
     */
    fun refine(
        gray: Plane,
        candidate: BoardCandidate,
        offsetCells: Double = 0.6,
        colours: Colours? = null,
    ): Refinement {
        var cell = candidate.cellPx
        val margin = cell * (offsetCells + 0.1)
        val xLo = max(0.0, floor(candidate.region.x - margin)).toInt()
        val yLo = max(0.0, floor(candidate.region.y - margin)).toInt()
        val xHi = min(gray.width.toDouble(), ceil(candidate.region.x + 8 * cell + margin)).toInt()
        val yHi = min(gray.height.toDouble(), ceil(candidate.region.y + 8 * cell + margin)).toInt()
        if (xHi <= xLo || yHi <= yLo) return Refinement(candidate, 0.0, 0.0)
        val crop = gray.crop(xLo, yLo, xHi, yHi)

        var fitted = candidate
        var originX = candidate.region.x.toDouble()
        var originY = candidate.region.y.toDouble()
        var profileX = DoubleArray(0)
        var profileY = DoubleArray(0)
        for (attempt in 0 until 3) {
            cell = fitted.cellPx
            val medians = cells(gray, fitted.region, cell, pad = 0)
            // The first pass does not know where the lines are to within the search window
            // (the tracker's re-fit hands in a lock a third of a cell off), so its bands span
            // the window; the passes after it know to a few pixels.
            val band = if (attempt > 0) 3 else pyRound(cell * offsetCells) + 3
            val read = colours ?: palette(medians)
            val weight = boundaryWeight(crop, xLo, yLo, fitted, medians, read, band)
            profileX = gradientProfile(crop, 1, weight)
            profileY = gradientProfile(crop, 0, weight)
            val (ox, px) = combFit(profileX, (fitted.region.x - xLo).toDouble(), cell, offsetCells)
            val (oy, py) = combFit(profileY, (fitted.region.y - yLo).toDouble(), cell, offsetCells)
            val pitch = (px + py) / 2.0
            originX = xLo + ox
            originY = yLo + oy
            val side = pyRound(8 * pitch)
            val region = Region(pyRound(originX), pyRound(originY), side, side)
            val moved = abs(pitch - cell) > 0.005 * cell ||
                abs(region.x - fitted.region.x) > 2 ||
                abs(region.y - fitted.region.y) > 2
            fitted = fitted.copy(region = region, cellPx = pitch)
            if (!moved) break
        }
        val pitch = fitted.cellPx
        val side = fitted.region.w
        val grid = cells(gray, fitted.region, pitch, pad = 2)

        // Python's max over the phases: the first of the best (validates, evidence, fit).
        var best = PHASES[0]
        var bestKey: Triple<Boolean, Int, Double>? = null
        for (phase in PHASES) {
            val (dx, dy) = phase
            val shifted = DoubleArray(100) { grid[(it / 10 + 1 + dy) * 12 + it % 10 + 1 + dx] }
            val reading = judge(shifted, colours)
            val key = Triple(reading.fit > 0.0, reading.evidence, reading.fit)
            if (bestKey == null || compareKeys(key, bestKey) > 0) {
                best = phase
                bestKey = key
            }
        }
        val (dx, dy) = best
        val region = Region(pyRound(originX + dx * pitch), pyRound(originY + dy * pitch), side, side)
        var fit = 0.0
        var contrast = 0.0
        var occupied = 0
        var readable = 0
        if (onFrame(gray, region)) {
            val around = cells(gray, region, pitch)
            val reading = judge(around, colours)
            fit = reading.fit
            contrast = reading.contrast
            readable = reading.readable
            val read = reading.colours
            if (read != null) occupied = occupiedCells(gray, region, pitch, inner(around), read)
        }
        // The comb's strength at the board kept and at the geometry the fit started from, over
        // the same profiles, for the tracker to weigh a re-fit against its lock.
        val strength = combStrength(profileX, (region.x - xLo).toDouble(), pitch) +
            combStrength(profileY, (region.y - yLo).toDouble(), pitch)
        val reference = combStrength(profileX, (candidate.region.x - xLo).toDouble(), candidate.cellPx) +
            combStrength(profileY, (candidate.region.y - yLo).toDouble(), candidate.cellPx)
        val board = candidate.copy(
            region = region,
            cellPx = pitch,
            fit = fit,
            contrast = contrast,
            occupied = occupied,
            readable = readable,
        )
        return Refinement(board, strength, reference)
    }

    private fun compareKeys(a: Triple<Boolean, Int, Double>, b: Triple<Boolean, Int, Double>): Int {
        if (a.first != b.first) return if (a.first) 1 else -1
        if (a.second != b.second) return a.second.compareTo(b.second)
        return a.third.compareTo(b.third)
    }

    private fun median(values: FloatArray, n: Int): Double {
        val copy = values.copyOf(n)
        copy.sort()
        return if (n % 2 == 1) copy[n / 2].toDouble()
        else (copy[n / 2 - 1].toDouble() + copy[n / 2].toDouble()) / 2.0
    }

    /** numpy's median; NaN of nothing. */
    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return Double.NaN
        val sorted = values.sorted()
        val n = sorted.size
        return if (n % 2 == 1) sorted[n / 2] else (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/** The plane's sub-rectangle, copied. */
fun Plane.crop(x0: Int, y0: Int, x1: Int, y1: Int): Plane {
    val w = x1 - x0
    val h = y1 - y0
    val out = FloatArray(w * h)
    for (y in 0 until h) {
        System.arraycopy(data, (y0 + y) * width + x0, out, y * w, w)
    }
    return Plane(w, h, out)
}

private operator fun IntArray.component1() = this[0]
private operator fun IntArray.component2() = this[1]
private operator fun IntArray.component3() = this[2]
private operator fun IntArray.component4() = this[3]
