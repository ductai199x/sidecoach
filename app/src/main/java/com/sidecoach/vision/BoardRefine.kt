package com.sidecoach.vision

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sign

/** The two square colours a board is read with, and how far a cell may stray from either. */
data class Colours(val light: Double, val dark: Double, val tolerance: Double)

/** A candidate after the boundary fit, with the comb strength at its new and old geometry. */
data class Refinement(val board: BoardCandidate, val strength: Double, val reference: Double)

/**
 * Passes two and three of the detector, ported from `sidecoach/vision/board_detect.py`:
 * the sub-pixel boundary fit, and the validation that decides whether a candidate is a
 * board at all.
 *
 * The comments that matter are the ones that record a failure, and they are kept from the
 * Python: the mask that keeps pieces out of the boundary fit, the blends that must stay,
 * the re-fit when the mask was laid on a geometry that was itself off.
 */
object BoardRefine {

    private val TEETH = DoubleArray(9) { it.toDouble() }

    fun cellBounds(region: Region, cell: Double, r: Int, c: Int): IntArray {
        val x0 = pyRound(region.x + c * cell)
        val y0 = pyRound(region.y + r * cell)
        val x1 = pyRound(region.x + (c + 1) * cell)
        val y1 = pyRound(region.y + (r + 1) * cell)
        return intArrayOf(x0, y0, x1, y1)
    }

    fun ringBand(cell: Double): Int = max(1, pyRound(cell * 0.12))

    /** Median of a cell's border ring, or null when the cell is not wholly in the frame. */
    fun ringMedian(gray: Plane, box: IntArray, band: Int): Double? {
        val (x0, y0, x1, y1) = box
        if (x0 < 0 || y0 < 0 || x1 > gray.width || y1 > gray.height) return null
        if (x1 - x0 <= 2 * band || y1 - y0 <= 2 * band) return null
        val values = FloatArray((x1 - x0) * (y1 - y0))
        var n = 0
        for (y in y0 until y1) {
            val onTopOrBottom = y < y0 + band || y >= y1 - band
            val row = y * gray.width
            if (onTopOrBottom) {
                for (x in x0 until x1) values[n++] = gray.data[row + x]
            } else {
                for (x in x0 until x0 + band) values[n++] = gray.data[row + x]
                for (x in x1 - band until x1) values[n++] = gray.data[row + x]
            }
        }
        return median(values, n)
    }

    /** 8x8 border-ring medians, or null when the board runs off the frame. */
    fun cellMedians(gray: Plane, region: Region, cell: Double): DoubleArray? {
        val band = ringBand(cell)
        val medians = DoubleArray(64)
        for (r in 0 until 8) {
            for (c in 0 until 8) {
                val value = ringMedian(gray, cellBounds(region, cell, r, c), band) ?: return null
                medians[r * 8 + c] = value
            }
        }
        return medians
    }

    /**
     * The square colours read from a grid of cell medians, with their tolerance. The
     * tolerance follows how much the cells of each colour vary, so a textured wood theme
     * keeps its cells while a dialog or a highlight over a cell is left out of the fit.
     */
    fun palette(medians: DoubleArray): Colours {
        val lights = ArrayList<Double>(32)
        val darks = ArrayList<Double>(32)
        for (i in 0 until 64) {
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

    fun readableAgainst(medians: DoubleArray, colours: Colours): BooleanArray =
        BooleanArray(64) {
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
     * Whether the checker pattern carries on outside the candidate on any of its sides: a
     * candidate shifted by whole cells, or a window of a larger checkerboard, reads a
     * perfect board inside its 64 cells and only the cells just outside tell.
     */
    fun outerContinuesBoard(gray: Plane, region: Region, cell: Double, colours: Colours): Boolean {
        val band = ringBand(cell)
        val mid = 0.5 * (colours.light + colours.dark)
        val sides = listOf(
            (0 until 8).map { -1 to it },
            (0 until 8).map { 8 to it },
            (0 until 8).map { it to -1 },
            (0 until 8).map { it to 8 },
        )
        for (cells in sides) {
            val signs = cells.map { (r, c) ->
                val value = ringMedian(gray, cellBounds(region, cell, r, c), band)
                if (value == null ||
                    (abs(value - colours.light) > colours.tolerance &&
                        abs(value - colours.dark) > colours.tolerance)
                ) {
                    null
                } else {
                    if (value > mid) 1.0 else -1.0
                }
            }
            if (signs.count { it != null } < 6) continue
            var alternating = 0
            for (i in 0 until signs.size - 1) {
                val a = signs[i]
                val b = signs[i + 1]
                if (a != null && b != null && a != b) alternating++
            }
            if (alternating >= 5) return true
        }
        return false
    }

    /**
     * How many of the 64 cells at [region] read as one of [colours]' square colours.
     *
     * The fit counts only these, so a board that moved away by two cells and a fraction still
     * fits where it was, on the cells it left behind; what gives it away is how few are left
     * to read (the tracker's probe).
     */
    fun readableCells(gray: Plane, region: Region, cell: Double, colours: Colours): Int {
        val medians = cellMedians(gray, region, cell) ?: return 0
        return readableAgainst(medians, colours).count { it }
    }

    /** (fit, contrast) of a grid of cell medians read against [colours]. */
    fun scoreCells(
        gray: Plane,
        region: Region,
        cell: Double,
        medians: DoubleArray,
        colours: Colours,
    ): Pair<Double, Double> {
        val readable = readableAgainst(medians, colours)
        // The threshold sits between the two square colours rather than at the board mean,
        // which a dark dialog over a light board would drag below the dark squares.
        val mid = 0.5 * (colours.light + colours.dark)
        val readableCount = readable.count { it }
        if (readableCount < BoardDetect.MIN_READABLE_CELLS) {
            var sum = 0.0
            for (i in 0 until 64) sum += abs(medians[i] - mid)
            return 0.0 to sum / 64.0
        }
        var contrastSum = 0.0
        for (i in 0 until 64) if (readable[i]) contrastSum += abs(medians[i] - mid)
        val contrast = contrastSum / readableCount
        val expected = expectedSigns(colours)
        var matched = 0
        for (i in 0 until 64) {
            if (readable[i] && sign(medians[i] - mid).toFloat() == expected[i]) matched++
        }
        val match = matched.toDouble() / readableCount
        if (match >= 0.5 && outerContinuesBoard(gray, region, cell, colours)) return 0.0 to contrast
        return match to contrast
    }

    /** (fraction of readable cells matching the checker pattern, contrast). */
    fun fitScore(gray: Plane, region: Region, cell: Double, colours: Colours? = null): Pair<Double, Double> {
        val medians = cellMedians(gray, region, cell) ?: return 0.0 to 0.0
        return scoreCells(gray, region, cell, medians, colours ?: palette(medians))
    }

    /** The square colours of a board as read now, for a lock to keep. */
    fun squareColours(gray: Plane, region: Region, cell: Double): Colours? =
        cellMedians(gray, region, cell)?.let { palette(it) }

    /**
     * Readable cells whose interior mean differs from their ring median by over the
     * tolerance: pieces and marks. Well above zero on a game board, near zero on an empty
     * one, and it only breaks ties.
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
                val ix0 = x0 + band
                val iy0 = y0 + band
                val ix1 = x1 - band
                val iy1 = y1 - band
                if (ix1 <= ix0 || iy1 <= iy0) continue
                if (ix0 < 0 || iy0 < 0 || ix1 > gray.width || iy1 > gray.height) continue
                var sum = 0.0
                var n = 0
                for (y in iy0 until iy1) {
                    val row = y * gray.width
                    for (x in ix0 until ix1) {
                        sum += gray.data[row + x]; n++
                    }
                }
                if (n > 0 && abs(sum / n - medians[r * 8 + c]) > colours.tolerance) count++
            }
        }
        return count
    }

    /** Summed profile at the nine teeth of a comb; teeth past the profile's end read zero. */
    fun combStrength(profile: DoubleArray, offset: Double, pitch: Double): Double {
        var sum = 0.0
        for (t in TEETH) {
            val position = offset + pitch * t
            if (position < 0 || position > profile.size - 1) continue
            sum += interpolate(profile, position)
        }
        return sum
    }

    /**
     * Best (offset, pitch, strength) of a nine-tooth comb over a gradient profile: whole
     * pixels and one-percent pitch steps over the window first, then quarter pixels and
     * 0.2 percent steps around the best.
     */
    fun combFit(
        profile: DoubleArray,
        start: Double,
        pitch: Double,
        search: Double,
        offsetCells: Double = 0.6,
    ): Triple<Double, Double, Double> {
        val n = profile.size

        fun bestOver(pitches: DoubleArray, offsets: DoubleArray): Triple<Double, Double, Double> {
            var best = Triple(start, pitch, Double.NEGATIVE_INFINITY)
            for (p in pitches) {
                for (o in offsets) {
                    if (o + p * 8 > n - 1) continue
                    var strength = 0.0
                    for (t in TEETH) strength += interpolate(profile, o + p * t)
                    if (strength > best.third) best = Triple(o, p, strength)
                }
            }
            return best
        }

        val reach = pitch * offsetCells
        val step = max(0.25, pitch * 0.01)
        val coarse = bestOver(
            range(pitch * (1 - search), pitch * (1 + search), step),
            range(max(0.0, start - reach), min(n - 1.0, start + reach), 1.0),
        )
        if (coarse.third == Double.NEGATIVE_INFINITY) return Triple(start, pitch, 0.0)
        val (o, p, _) = coarse
        val span = max(1.0, 4.0 * step)
        val fine = bestOver(
            range(p * 0.99, p * 1.01, max(0.05, p * 0.002)),
            range(max(0.0, o - span), min(n - 1.0, o + span) + 0.01, 0.25),
        )
        return if (fine.third >= coarse.third) fine else coarse
    }

    /**
     * Sub-pixel boundaries from gradient peaks, then validation. [offsetCells] is how far,
     * in cells, the boundaries may move from the candidate.
     */
    fun refine(
        gray: Plane,
        candidate: BoardCandidate,
        search: Double = 0.12,
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
        var reference = 0.0
        var strength = 0.0
        for (attempt in 0 until 3) {
            cell = fitted.cellPx
            val weight = FloatArray(crop.width * crop.height) { 1f }
            val medians = cellMedians(gray, fitted.region, cell)
            var readable = BooleanArray(64) { true }
            if (medians != null) {
                val read = colours ?: palette(medians)
                readable = readableAgainst(medians, read)
                if (readable.count { it } < BoardDetect.MIN_READABLE_CELLS) {
                    // Too few cells to place the mask; the candidate stays where it is and
                    // carries the fit measured there, which is a fit of zero.
                    val (fit, contrast) = fitScore(gray, candidate.region, candidate.cellPx, colours)
                    return Refinement(candidate.copy(fit = fit, contrast = contrast), 0.0, 0.0)
                }
                applyMask(crop, weight, gray, fitted, cell, medians, read, xLo, yLo, attempt, offsetCells)
            }
            // The middle of every cell takes no part: a piece's body would out-gradient a
            // faint boundary on a low-contrast theme.
            for (r in 0 until 8) {
                for (c in 0 until 8) {
                    val inset = pyRound(cell * (if (readable[r * 8 + c]) 0.2 else 0.08))
                    val (bx0, by0, bx1, by1) = cellBounds(fitted.region, cell, r, c)
                    val x0 = bx0 - xLo + inset
                    val y0 = by0 - yLo + inset
                    val x1 = bx1 - xLo - inset
                    val y1 = by1 - yLo - inset
                    for (y in max(0, y0) until min(crop.height, y1)) {
                        val row = y * crop.width
                        for (x in max(0, x0) until min(crop.width, x1)) weight[row + x] = 0f
                    }
                }
            }

            val profileX = gradientProfile(crop, axis = 1, weight = weight)
            val profileY = gradientProfile(crop, axis = 0, weight = weight)
            val startX = (fitted.region.x - xLo).toDouble()
            val startY = (fitted.region.y - yLo).toDouble()
            val (ox, px, sx) = combFit(profileX, startX, cell, search, offsetCells)
            val (oy, py, sy) = combFit(profileY, startY, cell, search, offsetCells)
            // The strength the candidate itself has on this pass's profiles, so the caller
            // can weigh the re-fit against it.
            reference = combStrength(profileX, (candidate.region.x - xLo).toDouble(), candidate.cellPx) +
                combStrength(profileY, (candidate.region.y - yLo).toDouble(), candidate.cellPx)
            strength = sx + sy
            val pitch = (px + py) / 2.0
            val region = Region(
                pyRound(xLo + ox),
                pyRound(yLo + oy),
                pyRound(8 * pitch),
                pyRound(8 * pitch),
            )
            val moved = abs(pitch - cell) > 0.005 * cell ||
                abs(region.x - fitted.region.x) > 2 ||
                abs(region.y - fitted.region.y) > 2
            fitted = fitted.copy(region = region, cellPx = pitch)
            if (!moved) break
        }

        val medians = cellMedians(gray, fitted.region, fitted.cellPx)
        var fit = 0.0
        var contrast = 0.0
        var occupied = 0
        var readable = 0
        if (medians != null) {
            val read = colours ?: palette(medians)
            val scored = scoreCells(gray, fitted.region, fitted.cellPx, medians, read)
            fit = scored.first
            contrast = scored.second
            occupied = occupiedCells(gray, fitted.region, fitted.cellPx, medians, read)
            readable = readableAgainst(medians, read).count { it }
        }
        val board = candidate.copy(
            region = fitted.region,
            cellPx = fitted.cellPx,
            fit = fit,
            contrast = contrast,
            occupied = occupied,
            readable = readable,
        )
        return Refinement(board, strength, reference)
    }

    /**
     * Only the strips around the cell boundaries take part in the fit, and in them only the
     * pixels of the square's own colour: a real set's pieces reach into the strips and a
     * rank of them draws a line as strong as a boundary. The blends at a boundary lie on
     * the grid lines and are kept; a pixel grown from a piece's edge is not.
     */
    private fun applyMask(
        crop: Plane,
        weight: FloatArray,
        gray: Plane,
        fitted: BoardCandidate,
        cell: Double,
        medians: DoubleArray,
        read: Colours,
        xLo: Int,
        yLo: Int,
        attempt: Int,
        offsetCells: Double,
    ) {
        val own = Mat.zeros(crop.height, crop.width, CvType.CV_8U)
        val ownBytes = ByteArray(crop.width * crop.height)
        for (r in 0 until 8) {
            for (c in 0 until 8) {
                val (bx0, by0, bx1, by1) = cellBounds(fitted.region, cell, r, c)
                val x0 = max(0, bx0 - xLo)
                val y0 = max(0, by0 - yLo)
                val x1 = min(crop.width, bx1 - xLo)
                val y1 = min(crop.height, by1 - yLo)
                val target = medians[r * 8 + c]
                for (y in y0 until y1) {
                    val row = y * crop.width
                    for (x in x0 until x1) {
                        if (abs(crop.data[row + x] - target) <= read.tolerance) ownBytes[row + x] = 1
                    }
                }
            }
        }
        own.put(0, 0, ownBytes)
        val grown = Mat()
        val kernel = Mat.ones(3, 3, CvType.CV_8U)
        Imgproc.dilate(own, grown, kernel)
        val grownBytes = ByteArray(crop.width * crop.height)
        grown.get(0, 0, grownBytes)
        own.release(); grown.release(); kernel.release()

        val lo = min(read.light, read.dark) - read.tolerance
        val hi = max(read.light, read.dark) + read.tolerance
        // The first pass does not know where the lines are to within the search window, so
        // its bands span the window; the passes after it know to a few pixels.
        val band = if (attempt > 0) 3 else pyRound(cell * offsetCells) + 3
        val onLine = BooleanArray(crop.width * crop.height)
        for (k in 1 until 8) {
            val gx = pyRound(fitted.region.x - xLo + k * cell)
            val gy = pyRound(fitted.region.y - yLo + k * cell)
            for (x in max(0, gx - band)..min(crop.width - 1, gx + band)) {
                for (y in 0 until crop.height) onLine[y * crop.width + x] = true
            }
            for (y in max(0, gy - band)..min(crop.height - 1, gy + band)) {
                val row = y * crop.width
                for (x in 0 until crop.width) onLine[row + x] = true
            }
        }
        // At the board's own edge both sides of the gradient are kept, as everywhere
        // outside: dropping the outer side moved the last tooth a pixel in.
        val bx0 = max(0, fitted.region.x - xLo)
        val by0 = max(0, fitted.region.y - yLo)
        val bx1 = min(crop.width, fitted.region.x - xLo + pyRound(8 * cell))
        val by1 = min(crop.height, fitted.region.y - yLo + pyRound(8 * cell))
        val onEdge = BooleanArray(crop.width * crop.height)
        for (y in max(0, by0 - band)..min(crop.height - 1, by0 + band)) {
            val row = y * crop.width
            for (x in 0 until crop.width) onEdge[row + x] = true
        }
        for (y in max(0, by1 - band)..min(crop.height - 1, by1 + band)) {
            val row = y * crop.width
            for (x in 0 until crop.width) onEdge[row + x] = true
        }
        for (x in max(0, bx0 - band)..min(crop.width - 1, bx0 + band)) {
            for (y in 0 until crop.height) onEdge[y * crop.width + x] = true
        }
        for (x in max(0, bx1 - band)..min(crop.width - 1, bx1 + band)) {
            for (y in 0 until crop.height) onEdge[y * crop.width + x] = true
        }

        for (y in by0 until by1) {
            val row = y * crop.width
            for (x in bx0 until bx1) {
                val i = row + x
                val value = crop.data[i]
                val kept = ownBytes[i].toInt() != 0 ||
                    (value >= lo && value <= hi && onLine[i]) ||
                    (grownBytes[i].toInt() != 0 && onEdge[i])
                if (!kept) weight[i] = 0f
            }
        }
    }

    /** Summed absolute gradient across the other axis; axis 1 gives a profile along x. */
    fun gradientProfile(plane: Plane, axis: Int, weight: FloatArray?): DoubleArray {
        val mat = plane.toMat()
        val grad = Mat()
        if (axis == 1) Imgproc.Sobel(mat, grad, CvType.CV_32F, 1, 0, 3)
        else Imgproc.Sobel(mat, grad, CvType.CV_32F, 0, 1, 3)
        val absolute = Mat()
        Core.absdiff(grad, org.opencv.core.Scalar(0.0), absolute)
        val values = FloatArray(plane.width * plane.height)
        absolute.get(0, 0, values)
        mat.release(); grad.release(); absolute.release()
        val out = DoubleArray(if (axis == 1) plane.width else plane.height)
        for (y in 0 until plane.height) {
            val row = y * plane.width
            for (x in 0 until plane.width) {
                val i = row + x
                val v = if (weight != null) values[i] * weight[i] else values[i]
                if (axis == 1) out[x] += v else out[y] += v
            }
        }
        return out
    }

    private fun interpolate(profile: DoubleArray, position: Double): Double {
        if (position <= 0) return profile[0]
        if (position >= profile.size - 1) return profile[profile.size - 1]
        val i = position.toInt()
        val t = position - i
        return profile[i] * (1 - t) + profile[i + 1] * t
    }

    private fun range(from: Double, until: Double, step: Double): DoubleArray {
        if (step <= 0 || until <= from) return DoubleArray(0)
        val n = ceil((until - from) / step).toInt()
        return DoubleArray(max(0, n)) { from + it * step }
    }

    private fun median(values: FloatArray, n: Int): Double {
        val copy = values.copyOf(n)
        copy.sort()
        return if (n % 2 == 1) copy[n / 2].toDouble()
        else (copy[n / 2 - 1].toDouble() + copy[n / 2].toDouble()) / 2.0
    }

    private fun median(values: List<Double>): Double {
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
