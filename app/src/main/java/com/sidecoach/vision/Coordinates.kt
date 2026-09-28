package com.sidecoach.vision

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * The board's orientation read from the coordinate labels chess.com draws on its edge:
 * `sidecoach/vision/coordinates.py`, ported.
 *
 * A cached piece style lets a puzzle or a mid-game board be classified, but turning that
 * into a position needs the orientation: which corner is a1. A static position is ambiguous
 * under a 180-degree turn, so the pieces alone cannot say. The rank labels can: they run
 * 8-to-1 down the left edge when White sits at the bottom, and 1-to-8 when Black does.
 *
 * They are read with the same learned-exemplar idea the recogniser uses for pieces. When the
 * theme is first learned from the start position -- where the orientation is known -- the
 * rank digit in each left-edge cell is captured and labelled with its value. A board whose
 * owner has turned coordinates off has no labels to read, and the reader says so rather than
 * guessing.
 */
object Coordinates {

    /** Rank-digit exemplars are normalised to this square before matching. */
    const val GLYPH = 24

    private const val CORNER_H = 0.30
    private const val CORNER_W = 0.20

    /**
     * A pixel this far from the square colour is the digit, not the square. The digits are
     * drawn in the other square's colour, and on a dark board the two lie 48 levels apart:
     * at 60 a 44-pixel digit, all anti-aliased blends, was a scatter of specks and nothing
     * was learned but the 8. A wood grain strays 20 at most.
     */
    private const val INK = 30.0

    /** Fewer foreground pixels than this is no label. */
    private const val MIN_INK = 4

    /**
     * A corner window more than this full of ink holds no digit: the square's colour was
     * taken from its neighbour, so the cell is cut off its square -- a lock a tenth of a cell
     * out is enough. A digit fills a quarter to a third of the window, a cut that far out nine
     * tenths or all of it (Tai's Pixel, 2026-09-27: a start seen through a lock that far out
     * cached all eight labels as solid blocks, and every edge digit then read as the same one).
     */
    private const val CORNER_FULL = 0.6

    /**
     * A hole this share of a digit's box or more is a loop: an 8's are 2.4 to 14 % on
     * chess.com's boards (80-86 px) and rendered ones (44-110 px), the pinholes in a fantasy
     * piece's shading that reached the corner 0.7 % (2026-09-28).
     */
    private const val LOOP = 0.02

    /**
     * The square's colour: the median of the bands along the cell's right and bottom edges,
     * a twentieth of the cell in from the edge and as deep again.
     */
    private fun squareColour(cell: ByteArray, height: Int, width: Int): DoubleArray {
        val n = min(height, width)
        val outer = max(1, pyRound(n * 0.05))
        val inner = outer + max(1, pyRound(n * 0.07))
        val channels = Array(3) { ArrayList<Double>() }
        for (y in 0 until height) {
            for (x in max(0, width - inner) until max(0, width - outer)) {
                val at = (y * width + x) * 3
                for (c in 0 until 3) channels[c].add((cell[at + c].toInt() and 0xFF).toDouble())
            }
        }
        for (y in max(0, height - inner) until max(0, height - outer)) {
            for (x in 0 until width) {
                val at = (y * width + x) * 3
                for (c in 0 until 3) channels[c].add((cell[at + c].toInt() and 0xFF).toDouble())
            }
        }
        return DoubleArray(3) { median(channels[it]) }
    }

    /** numpy's median: the middle value, or the mean of the two middle ones. */
    private fun median(values: MutableList<Double>): Double {
        if (values.isEmpty()) return 0.0
        values.sort()
        val n = values.size
        return if (n % 2 == 1) values[n / 2] else (values[n / 2 - 1] + values[n / 2]) / 2.0
    }

    /**
     * The rank digit in a cell's top-left corner as a normalised binary glyph, or null:
     * [digit] scaled to a fixed size, so digits read at slightly different pixel sizes still
     * match.
     */
    fun cornerGlyph(cell: Mat): BooleanArray? {
        val found = digit(cell) ?: return null
        val cropped = Mat.zeros(found.height, found.width, CvType.CV_8UC1)
        cropped.put(0, 0, ByteArray(found.ink.size) { if (found.ink[it].toInt() == 1) 255.toByte() else 0 })
        val resized = Mat()
        Imgproc.resize(
            cropped, resized, Size(GLYPH.toDouble(), GLYPH.toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST,
        )
        val out = ByteArray(GLYPH * GLYPH)
        resized.get(0, 0, out)
        cropped.release(); resized.release()
        return BooleanArray(GLYPH * GLYPH) { (out[it].toInt() and 0xFF) > 127 }
    }

    /** A digit's ink cropped to its box: 1 for ink, 0 for none, row by row. */
    private class Digit(val ink: ByteArray, val height: Int, val width: Int)

    /**
     * The rank digit in a cell's top-left corner, its ink cropped to its box, or null.
     *
     * The digit is drawn to contrast the square, so pixels far from the square's own colour
     * are the digit.
     */
    private fun digit(cell: Mat): Digit? {
        if (cell.channels() != 3 || cell.empty()) return null
        val height = cell.rows()
        val width = cell.cols()
        val bytes = ByteArray(height * width * 3)
        cell.get(0, 0, bytes)
        // The square's colour from the bands just inside the cell's right and bottom edges,
        // which neither the digit (top left) nor a piece (the middle) reaches.
        val background = squareColour(bytes, height, width)

        val cornerH = max(1, (height * CORNER_H).toInt())
        val cornerW = max(1, (width * CORNER_W).toInt())
        var mask = Mat.zeros(cornerH, cornerW, CvType.CV_8UC1)
        val maskBytes = ByteArray(cornerH * cornerW)
        var ink = 0
        for (y in 0 until cornerH) {
            for (x in 0 until cornerW) {
                val at = (y * width + x) * 3
                var sum = 0.0
                for (c in 0 until 3) {
                    val d = (bytes[at + c].toInt() and 0xFF).toDouble() - background[c]
                    sum += d * d
                }
                if (sqrt(sum) > INK) {
                    maskBytes[y * cornerW + x] = 1
                    ink += 1
                }
            }
        }
        if (ink < MIN_INK || ink.toDouble() / (cornerH * cornerW) > CORNER_FULL) {
            mask.release(); return null
        }
        mask.put(0, 0, maskBytes)

        // The digit is one component of ink, inset from the cell's edge. A lock a pixel off
        // puts a strip of the neighbouring cell along the window's top or left edge, and a
        // pawn on a small cell reaches into the window from the lower right: neither is the
        // digit. Components touching the top or left edge go first, then the component
        // nearest the corner is the digit.
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        val count = Imgproc.connectedComponentsWithStats(mask, labels, stats, centroids, 8)
        if (count > 2) {
            val inset = (1 until count).filter {
                stats.get(it, Imgproc.CC_STAT_LEFT)[0] > 0 && stats.get(it, Imgproc.CC_STAT_TOP)[0] > 0
            }.ifEmpty { (1 until count).toList() }
            val nearest = inset.minByOrNull {
                stats.get(it, Imgproc.CC_STAT_LEFT)[0] + stats.get(it, Imgproc.CC_STAT_TOP)[0]
            }!!
            val labelBuffer = IntArray(cornerH * cornerW)
            labels.get(0, 0, labelBuffer)
            ink = 0
            for (i in maskBytes.indices) {
                maskBytes[i] = if (labelBuffer[i] == nearest) 1 else 0
                if (maskBytes[i].toInt() == 1) ink += 1
            }
            if (ink < MIN_INK) {
                mask.release(); labels.release(); stats.release(); centroids.release()
                return null
            }
        }
        labels.release(); stats.release(); centroids.release()

        var top = cornerH; var bottom = -1; var left = cornerW; var right = -1
        for (y in 0 until cornerH) {
            for (x in 0 until cornerW) {
                if (maskBytes[y * cornerW + x].toInt() != 1) continue
                if (y < top) top = y
                if (y > bottom) bottom = y
                if (x < left) left = x
                if (x > right) right = x
            }
        }
        mask.release()
        val cropH = bottom - top + 1
        val cropW = right - left + 1
        val cropped = ByteArray(cropH * cropW)
        for (y in 0 until cropH) {
            for (x in 0 until cropW) {
                cropped[y * cropW + x] = maskBytes[(top + y) * cornerW + (left + x)]
            }
        }
        return Digit(cropped, cropH, cropW)
    }

    /**
     * The closed loops in a digit's ink, each as the height of its middle in the digit (0 at
     * the top, 1 at the foot): an 8 has two, one above the other, a 1 none, in any typeface.
     * A hole under [LOOP] of the digit's box is a pinhole in a piece's shading, not a loop.
     */
    private fun loops(digit: Digit): List<Double> {
        val h = digit.height + 2
        val w = digit.width + 2
        val padded = ByteArray(h * w)
        for (y in 0 until digit.height) {
            for (x in 0 until digit.width) padded[(y + 1) * w + (x + 1)] = digit.ink[y * digit.width + x]
        }
        val ink = Mat(h, w, CvType.CV_8UC1)
        ink.put(0, 0, padded)
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(ink, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_NONE)
        val heights = ArrayList<Double>()
        val box = digit.height * digit.width
        for (i in contours.indices) {
            if (hierarchy.get(0, i)[3] < 0) continue
            val inside = Mat.zeros(h, w, CvType.CV_8UC1)
            Imgproc.drawContours(inside, listOf(contours[i]), -1, Scalar(1.0), -1)
            val filled = ByteArray(h * w)
            inside.get(0, 0, filled)
            inside.release()
            var hole = 0
            for (j in filled.indices) if (filled[j].toInt() == 1 && padded[j].toInt() == 0) hole += 1
            if (hole >= LOOP * box) {
                val ys = contours[i].toArray()
                heights.add((ys.sumOf { it.y } / ys.size - 1) / digit.height)
            }
        }
        for (contour in contours) contour.release()
        ink.release(); hierarchy.release()
        return heights
    }

    private fun eight(digit: Digit): Boolean {
        val found = loops(digit)
        return found.size == 2 && found.min() < 0.5 && 0.5 < found.max()
    }

    /**
     * Which side sits at the bottom, read from where the 8 is, with nothing learned: the
     * corner digit with two loops at the top of the left edge and one with none at its foot
     * is White at the bottom, the other way round Black; anything else (coordinates off, a
     * piece over a digit) is null. Learned labels read more boards (a digit half covered
     * still matches); this reads a board whose labels are not learned, or were learned wrong
     * -- Tai's Pixel, 2026-09-28, the night after a start seen through an off lock cached
     * them as solid blocks: every puzzle asked for Flip, and the 8 was there to read all
     * along. A cell is never shrunk to read it: scaled down to the working size, the loops of
     * an 8 closed up.
     */
    fun eights(frame: Mat, geometry: BoardGeometry): Pieces.Side? {
        val size = max(Squares.CELL, pyRound(geometry.cellPx))
        val topCell = geometry.crop(frame, 0, 0, size)
        val footCell = geometry.crop(frame, 7, 0, size)
        val top = digit(topCell)
        val foot = digit(footCell)
        topCell.release(); footCell.release()
        if (top == null || foot == null) return null
        if (eight(top) && loops(foot).isEmpty()) return Pieces.Side.WHITE
        if (eight(foot) && loops(top).isEmpty()) return Pieces.Side.BLACK
        return null
    }

    /**
     * Capture the rank-digit exemplar in each left-edge cell, labelled by its known value.
     * With `bottom` known, the left-edge cell in row r shows rank 8-r when White is at the
     * bottom, r+1 when Black is. Labels that do not read `bottom` back from the frame they
     * were learned from are none: kept, they would say nothing or the wrong side on every
     * board after it.
     */
    fun learnRankLabels(
        frame: Mat,
        geometry: BoardGeometry,
        bottom: Pieces.Side,
    ): Map<Int, BooleanArray> {
        val labels = LinkedHashMap<Int, BooleanArray>()
        for (row in 0 until 8) {
            val rank = if (bottom.color) 8 - row else row + 1
            val cell = geometry.crop(frame, row, 0)
            val glyph = cornerGlyph(cell)
            cell.release()
            if (glyph != null) labels[rank] = glyph
        }
        return if (readLabels(frame, geometry, labels) == bottom) labels else emptyMap()
    }

    private fun iou(a: BooleanArray, b: BooleanArray): Double {
        var union = 0
        var both = 0
        for (i in a.indices) {
            if (a[i] || b[i]) union += 1
            if (a[i] && b[i]) both += 1
        }
        return if (union > 0) both.toDouble() / union else 0.0
    }

    /**
     * Which side sits at the bottom, read from the edge rank digits -- the learned labels
     * ([readLabels]), else where the 8 is ([eights]) -- or null if unreadable.
     */
    fun readBottomSide(
        frame: Mat,
        geometry: BoardGeometry,
        labels: Map<Int, BooleanArray>,
    ): Pieces.Side? = readLabels(frame, geometry, labels) ?: eights(frame, geometry)

    /**
     * Which side sits at the bottom, read with the learned labels, or null if unreadable.
     *
     * Each left-edge digit is matched against the learned exemplars; the sequence down the
     * column is 8..1 for White at the bottom and 1..8 for Black. A clear majority (at least
     * six of the eight in one direction) decides it; anything less returns null so the caller
     * can ask the user instead of guessing. A cell whose digit cannot be read is left out
     * rather than ending the read: on a mid-game board the edge cells hold pieces.
     */
    fun readLabels(
        frame: Mat,
        geometry: BoardGeometry,
        labels: Map<Int, BooleanArray>,
    ): Pieces.Side? {
        if (labels.size < 6) return null
        val reads = HashMap<Int, Int>()
        for (row in 0 until 8) {
            val cell = geometry.crop(frame, row, 0)
            val glyph = cornerGlyph(cell)
            cell.release()
            if (glyph == null) continue
            var bestDigit = -1
            var bestScore = Double.NEGATIVE_INFINITY
            for ((value, exemplar) in labels) {
                val score = iou(glyph, exemplar)
                if (score > bestScore) { bestScore = score; bestDigit = value }
            }
            if (bestScore >= 0.3) reads[row] = bestDigit
        }
        val descending = reads.count { (row, digit) -> digit == 8 - row }
        val ascending = reads.count { (row, digit) -> digit == row + 1 }
        if (descending >= 6 && descending > ascending) return Pieces.Side.WHITE
        if (ascending >= 6 && ascending > descending) return Pieces.Side.BLACK
        return null
    }
}
