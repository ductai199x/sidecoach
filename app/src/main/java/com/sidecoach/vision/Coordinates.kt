package com.sidecoach.vision

import org.opencv.core.CvType
import org.opencv.core.Mat
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
     * The rank digit in a cell's top-left corner as a normalised binary glyph, or null.
     *
     * The digit is drawn to contrast the square, so pixels far from the square's own colour
     * are the glyph. It is cropped to its bounding box and scaled to a fixed size so digits
     * read at slightly different pixel sizes still match.
     */
    fun cornerGlyph(cell: Mat): BooleanArray? {
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
        if (ink < MIN_INK) { mask.release(); return null }
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
        val cropH = bottom - top + 1
        val cropW = right - left + 1
        val cropped = Mat.zeros(cropH, cropW, CvType.CV_8UC1)
        val croppedBytes = ByteArray(cropH * cropW)
        for (y in 0 until cropH) {
            for (x in 0 until cropW) {
                croppedBytes[y * cropW + x] =
                    if (maskBytes[(top + y) * cornerW + (left + x)].toInt() == 1) 255.toByte() else 0
            }
        }
        cropped.put(0, 0, croppedBytes)
        val resized = Mat()
        Imgproc.resize(
            cropped, resized, Size(GLYPH.toDouble(), GLYPH.toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST,
        )
        val out = ByteArray(GLYPH * GLYPH)
        resized.get(0, 0, out)
        mask.release(); cropped.release(); resized.release()
        return BooleanArray(GLYPH * GLYPH) { (out[it].toInt() and 0xFF) > 127 }
    }

    /**
     * Capture the rank-digit exemplar in each left-edge cell, labelled by its known value.
     * With `bottom` known, the left-edge cell in row r shows rank 8-r when White is at the
     * bottom, r+1 when Black is.
     */
    fun learnRankLabels(
        frame: Mat,
        geometry: BoardGeometry,
        bottom: Pieces.Side,
    ): Map<Int, BooleanArray> {
        val labels = LinkedHashMap<Int, BooleanArray>()
        for (row in 0 until 8) {
            val rank = if (bottom == Pieces.Side.WHITE) 8 - row else row + 1
            val cell = geometry.crop(frame, row, 0)
            val glyph = cornerGlyph(cell)
            cell.release()
            if (glyph != null) labels[rank] = glyph
        }
        return labels
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
     * Which side sits at the bottom, read from the edge rank digits, or null if unreadable.
     *
     * Each left-edge digit is matched against the learned exemplars; the sequence down the
     * column is 8..1 for White at the bottom and 1..8 for Black. A clear majority (at least
     * six of the eight in one direction) decides it; anything less returns null so the caller
     * can ask the user instead of guessing. A cell whose digit cannot be read is left out
     * rather than ending the read: on a mid-game board the edge cells hold pieces.
     */
    fun readBottomSide(
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
