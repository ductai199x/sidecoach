package com.sidecoach.vision

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * What stands on a cell, before any net: the foreground silhouette, ported from the parts
 * of `sidecoach/vision/pieces.py` the bootstrap needs.
 *
 * The square's colour is the median of a ring inside the cell's border; every pixel far
 * enough from it is foreground. The outline is closed first, because a thin stroke breaks
 * at its anti-aliased joins and a broken outline encloses nothing (a wooden knight read as
 * its bare outline on light squares); what the outline encloses is then filled in, so a
 * piece whose fill is the square's own colour is still a piece. Parts confined to a corner
 * are dropped: a coordinate label sits in one, a badge in another, and neither is a piece.
 *
 * Only `fraction` and `luminance` are ported so far -- the occupancy the start-position
 * test reads, and the brightness that says which side sits at the bottom. The rest of the
 * desktop's silhouette (tones, hover, centroid, overlays) belongs with the reconciler.
 */
object Silhouette {

    /** The working size a cell is read at, as the desktop's CELL. */
    const val CELL = 48

    private const val INNER = 0.08
    private const val RING_SKIP = 0.05
    private const val RING_BAND = 0.07
    private const val BG_TOLERANCE = 25.0

    /** A cell is occupied when this much of it is foreground (`PIECE_MIN_FRACTION`). */
    const val PIECE_MIN_FRACTION = 0.16

    /** Of the patch: a component within one corner's box is no part of a piece. */
    private const val CORNER_BOX = 0.3

    /** What one cell shows. */
    data class Reading(val fraction: Double, val luminance: Double)

    /** The silhouette of one cell crop (BGR, any size). */
    fun of(patch: Mat): Reading {
        val (background, _) = ringMedian(patch)
        val border = (patch.rows() * INNER).roundToInt()
        val side = patch.rows() - 2 * border
        if (side < 4) return Reading(0.0, 0.0)
        val cut = patch.submat(Rect(border, border, side, side))
        val inner = Mat()
        Imgproc.resize(cut, inner, Size(CELL.toDouble(), CELL.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)

        // Distance of every pixel from the square's own colour.
        val pixels = ByteArray(CELL * CELL * 3)
        inner.get(0, 0, pixels)
        val foreground = ByteArray(CELL * CELL)
        var sum = 0.0
        var lit = 0
        for (i in 0 until CELL * CELL) {
            val b = (pixels[i * 3].toInt() and 0xFF).toDouble()
            val g = (pixels[i * 3 + 1].toInt() and 0xFF).toDouble()
            val r = (pixels[i * 3 + 2].toInt() and 0xFF).toDouble()
            val db = b - background[0]
            val dg = g - background[1]
            val dr = r - background[2]
            if (sqrt(db * db + dg * dg + dr * dr) > BG_TOLERANCE) {
                foreground[i] = 1
                // The desktop's luminance is the fill's; the mean of the foreground's
                // grey is the same signal for telling white pieces from black ones.
                sum += 0.114 * b + 0.587 * g + 0.299 * r
                lit++
            }
        }
        inner.release()
        cut.release()

        val mask = Mat(CELL, CELL, CvType.CV_8U)
        mask.put(0, 0, foreground)
        val kernel = Mat.ones(3, 3, CvType.CV_8U)
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel)
        fillEnclosed(mask)
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel)
        dropCorners(mask)
        val fraction = Core.countNonZero(mask).toDouble() / (CELL * CELL)
        mask.release()
        kernel.release()
        return Reading(fraction, if (lit > 0) sum / lit else 0.0)
    }

    /** The board's 64 cells, read in screen order. */
    fun board(frame: Mat, geometry: BoardGeometry): List<Reading> {
        val out = ArrayList<Reading>(64)
        for (r in 0 until 8) {
            for (c in 0 until 8) {
                val crop = geometry.crop(frame, r, c, 2 * CELL)
                out.add(of(crop))
                crop.release()
            }
        }
        return out
    }

    /**
     * Rows 0, 1, 6 and 7 occupied and rows 2 to 5 empty, allowing one stray cell: one move
     * away from the start already differs in two, so the allowance is a single cell for a
     * badge, a hover outline or a cursor (`pieces._start_shaped`).
     */
    fun startShaped(readings: List<Reading>): Boolean {
        var wrong = 0
        for (r in 0 until 8) {
            for (c in 0 until 8) {
                val occupied = readings[r * 8 + c].fraction >= PIECE_MIN_FRACTION
                val expected = r <= 1 || r >= 6
                if (occupied != expected) wrong++
            }
        }
        return wrong <= 1
    }

    /** Which colour sits at the bottom of a start position: the brighter pieces are white. */
    fun whiteAtBottom(readings: List<Reading>): Boolean {
        val top = ArrayList<Double>(16)
        val bottom = ArrayList<Double>(16)
        for (c in 0 until 8) {
            top.add(readings[0 * 8 + c].luminance)
            top.add(readings[1 * 8 + c].luminance)
            bottom.add(readings[6 * 8 + c].luminance)
            bottom.add(readings[7 * 8 + c].luminance)
        }
        return median(bottom) > median(top)
    }

    /**
     * The square's colour: the median of a ring inside the cell's border, which pieces sit
     * clear of. Returns (b, g, r) and how far the ring's own pixels stray from it.
     */
    private fun ringMedian(patch: Mat): Pair<DoubleArray, Double> {
        val n = patch.rows()
        val outer = max(1, (n * RING_SKIP).roundToInt())
        val inner = outer + max(1, (n * RING_BAND).roundToInt())
        val pixels = ByteArray(n * patch.cols() * 3)
        patch.get(0, 0, pixels)
        val bs = ArrayList<Double>(4 * n * (inner - outer))
        val gs = ArrayList<Double>(bs.size)
        val rs = ArrayList<Double>(bs.size)
        fun take(y: Int, x: Int) {
            val i = (y * patch.cols() + x) * 3
            bs.add((pixels[i].toInt() and 0xFF).toDouble())
            gs.add((pixels[i + 1].toInt() and 0xFF).toDouble())
            rs.add((pixels[i + 2].toInt() and 0xFF).toDouble())
        }
        for (y in outer until min(inner, n)) for (x in outer until n - outer) take(y, x)
        for (y in max(0, n - inner) until n - outer) for (x in outer until n - outer) take(y, x)
        for (y in inner until n - inner) {
            for (x in outer until min(inner, n)) take(y, x)
            for (x in max(0, n - inner) until n - outer) take(y, x)
        }
        if (bs.isEmpty()) {
            for (y in 0 until n) for (x in 0 until patch.cols()) take(y, x)
        }
        val median = doubleArrayOf(median(bs), median(gs), median(rs))
        var spreadSum = 0.0
        for (i in bs.indices) {
            val db = bs[i] - median[0]
            val dg = gs[i] - median[1]
            val dr = rs[i] - median[2]
            spreadSum += sqrt(db * db + dg * dg + dr * dr)
        }
        return median to (if (bs.isEmpty()) 0.0 else spreadSum / bs.size)
    }

    /** What the foreground encloses becomes foreground (`pieces._fill_enclosed`). */
    private fun fillEnclosed(mask: Mat) {
        val padded = Mat.zeros(mask.rows() + 2, mask.cols() + 2, CvType.CV_8U)
        val inverted = Mat()
        Core.bitwise_not(mask, inverted)
        Core.compare(inverted, Scalar(0.0), inverted, Core.CMP_NE)
        inverted.copyTo(padded.submat(Rect(1, 1, mask.cols(), mask.rows())))
        inverted.release()
        // The border is background by construction, so the flood starts outside the piece.
        Imgproc.rectangle(padded, Point(0.0, 0.0), Point((padded.cols() - 1).toDouble(), (padded.rows() - 1).toDouble()), Scalar(255.0), 1)
        val flood = Mat.zeros(padded.rows() + 2, padded.cols() + 2, CvType.CV_8U)
        Imgproc.floodFill(padded, flood, Point(0.0, 0.0), Scalar(2.0))
        val filled = padded.submat(Rect(1, 1, mask.cols(), mask.rows()))
        Core.compare(filled, Scalar(2.0), filled, Core.CMP_NE)
        filled.copyTo(mask)
        padded.release()
        flood.release()
    }

    /** Components confined to one corner are a label or a badge, not a piece. */
    private fun dropCorners(mask: Mat) {
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        val count = Imgproc.connectedComponentsWithStats(mask, labels, stats, centroids, 8)
        if (count <= 1) {
            labels.release(); stats.release(); centroids.release()
            return
        }
        val n = mask.rows()
        val box = n * CORNER_BOX
        val keep = BooleanArray(count)
        for (k in 1 until count) {
            val left = stats.get(k, Imgproc.CC_STAT_LEFT)[0]
            val top = stats.get(k, Imgproc.CC_STAT_TOP)[0]
            val right = left + stats.get(k, Imgproc.CC_STAT_WIDTH)[0]
            val bottom = top + stats.get(k, Imgproc.CC_STAT_HEIGHT)[0]
            val inX = left >= n - box || right <= box
            val inY = top >= n - box || bottom <= box
            keep[k] = !(inX && inY)
        }
        val ids = IntArray(n * mask.cols())
        labels.get(0, 0, ids)
        val out = ByteArray(ids.size)
        for (i in ids.indices) if (keep[ids[i]]) out[i] = 1
        mask.put(0, 0, out)
        labels.release(); stats.release(); centroids.release()
    }

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val n = sorted.size
        return if (n % 2 == 1) sorted[n / 2] else (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}
