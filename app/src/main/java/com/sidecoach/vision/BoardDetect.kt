package com.sidecoach.vision

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Classical board detection, ported from the desktop's `sidecoach/vision/board_detect.py`,
 * which is the source of truth for the algorithm and the numbers in it.
 *
 * A 2D chess board is an axis-aligned square of 8 by 8 equal cells in two colours. Three
 * passes:
 *
 * 1. Coarse search. For every cell size in a geometric range the frame is shrunk so a cell
 *    is a few pixels, and every position is scored on how well the border-ring means of its
 *    64 cells alternate: |sum(sign * mean)| / sum|mean - board mean|, which is 1 for a
 *    perfect empty board and near 0 for anything else, and is scale-selective because a
 *    doubled or halved cell cancels the pattern. The rings rather than whole cells, so
 *    pieces do not drag the means. The amplitude |sum(sign * mean)| / 64 is the contrast
 *    gate: background cannot inflate it the way it inflates the deviation.
 * 2. Refinement ([BoardRefine]): the nine boundary peaks of the gradient profiles are fitted
 *    with a nine-tooth comb for sub-pixel offset and pitch.
 * 3. Validation: each cell's colour is the median of a thin border ring; the fit is the
 *    share of readable cells matching the checker pattern, and a candidate whose pattern
 *    carries on just outside it is rejected.
 */
object BoardDetect {

    const val MIN_READABLE_CELLS = 24

    /** Boards whose width is within this fraction of the largest are one size to the ranking. */
    const val SAME_SIZE = 0.95

    /** The checker pattern's sign for each cell, light (+1) at the top-left. */
    val SIGNS = FloatArray(64) { if (((it / 8) + (it % 8)) % 2 == 0) 1f else -1f }

    /** A frame as the detector reads it: one grey plane of floats. */
    fun toGray(bgr: Mat): Plane {
        val gray = Mat()
        Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY)
        val floats = Mat()
        gray.convertTo(floats, CvType.CV_32F)
        gray.release()
        val plane = Plane.of(floats)
        floats.release()
        return plane
    }

    /**
     * Boards in a grey frame, largest first, each validated against the checker pattern.
     *
     * The coarse amplitude is shaved by the work-scale misalignment of up to half a pixel,
     * so its gate sits below the one the refined ring contrast must pass.
     */
    fun detect(
        gray: Plane,
        topK: Int = 3,
        minFit: Double = 0.85,
        minContrast: Double = 6.0,
        minCell: Int = 24,
        maxCell: Int? = null,
    ): List<BoardCandidate> {
        val coarse = coarseSearch(gray, minCell = minCell, maxCell = maxCell, minContrast = 0.6 * minContrast)
        val refined = coarse.take(topK * 4).map { BoardRefine.refine(gray, it).board }
        val good = nms(refined.filter { it.fit >= minFit && it.contrast >= minContrast }) { it.fit }
        // Size first, then fit, so a window that is part board and part background, which
        // can pass on a palette read from itself, never outranks a clean board; the
        // occupied count only decides between boards of one size and fit.
        val ranked = good.sortedWith(
            compareByDescending<BoardCandidate> { it.region.w }
                .thenByDescending { it.fit }
                .thenByDescending { it.occupied },
        ).toMutableList()
        if (ranked.isNotEmpty()) {
            // Among boards of about the size of the largest, the one that reads more of its
            // cells as squares comes first: a window 4 px wider, part board and part
            // background (24 cells readable against the board's 64), outranked a board moved
            // two cells away and the game was never read again (the parity audit's sweep,
            // 2026-09-26). The first of the most readable, as Python's `max` keeps it.
            val widest = ranked[0].region.w
            val peers = ranked.filter { it.region.w >= SAME_SIZE * widest }
            val best = peers.maxWith(
                compareBy<BoardCandidate> { it.readable }.thenBy { it.fit }.thenBy { it.occupied },
            )
            ranked.remove(best)
            ranked.add(0, best)
        }
        return ranked.take(topK)
    }

    /**
     * Candidates from the multi-scale checker response, before refinement. Each scale
     * contributes at most [perScale] peaks, no two within half a board of each other, so a
     * plateau of perfect scores around one board cannot hide a second board.
     */
    fun coarseSearch(
        gray: Plane,
        minCell: Int = 24,
        maxCell: Int? = null,
        workCell: Int = 8,
        scaleStep: Double = 1.06,
        minScore: Double = 0.45,
        minContrast: Double = 6.0,
        perScale: Int = 10,
    ): List<BoardCandidate> {
        val limit = maxCell ?: (min(gray.height, gray.width) / 8)
        val found = ArrayList<BoardCandidate>()
        var size = minCell.toDouble()
        while (size <= limit) {
            val factor = min(1.0, workCell / size)
            val small: Plane
            val cell: Int
            if (factor < 1.0) {
                small = gray.resized(factor)
                cell = workCell
            } else {
                small = gray
                cell = pyRound(size)
            }
            val response = checkerResponse(small, cell, minContrast)
            if (response != null) {
                val fullCell = cell / factor
                val side = pyRound(8 * fullCell)
                for ((y, x) in localMaxima(response, cell, radius = 4 * cell, limit = perScale, minScore = minScore, minContrast = minContrast)) {
                    val region = Region(
                        pyRound(x / factor),
                        pyRound(y / factor),
                        side,
                        side,
                    )
                    found.add(
                        BoardCandidate(
                            region = region,
                            cellPx = fullCell,
                            score = response.score[y * response.width + x],
                            contrast = response.amplitude[y * response.width + x],
                        ),
                    )
                }
            }
            size *= scaleStep
        }
        return nms(found) { it.score }
    }

    /** (score, amplitude) for a board whose top-left cell starts at each position. */
    fun checkerResponse(gray: Plane, cell: Int, minContrast: Double?): Response? {
        val means = ringMeans(gray, cell, band = 1)
        val h = means.height - 7 * cell
        val w = means.width - 7 * cell
        if (h <= 0 || w <= 0) return null

        // Along the columns: the signed and plain sums of the eight cells of each row of
        // the board, for every row position of the map. Separable: sixteen passes, not 64.
        val rowsH = means.height
        val signedCols = DoubleArray(rowsH * w)
        val plainCols = DoubleArray(rowsH * w)
        for (c in 0 until 8) {
            val sign = if (c % 2 == 0) 1.0 else -1.0
            val xOff = c * cell
            for (y in 0 until rowsH) {
                val src = y * means.width + xOff
                val dst = y * w
                for (x in 0 until w) {
                    val v = means.data[src + x]
                    plainCols[dst + x] += v
                    signedCols[dst + x] += sign * v
                }
            }
        }
        val acc = DoubleArray(h * w)
        val total = DoubleArray(h * w)
        for (r in 0 until 8) {
            val sign = if (r % 2 == 0) 1.0 else -1.0
            val yOff = r * cell
            for (y in 0 until h) {
                val src = (y + yOff) * w
                val dst = y * w
                for (x in 0 until w) {
                    acc[dst + x] += sign * signedCols[src + x]
                    total[dst + x] += plainCols[src + x]
                }
            }
        }
        // In doubles, as numpy has them: a plateau of scores round a board that float32 would
        // round to one value has its own highest point in float64, and the phone took the
        // first of the tie instead -- a board a cell off found a pixel off (the parity audit,
        // 2026-09-26).
        val amplitude = DoubleArray(h * w)
        val boardMean = DoubleArray(h * w)
        for (i in 0 until h * w) {
            amplitude[i] = abs(acc[i]) / 64.0
            boardMean[i] = total[i] / 64.0
        }
        val score = DoubleArray(h * w)
        // The deviation is not separable; it is summed only where the amplitude passes the
        // gate -- a few thousand positions of a frame's million -- and the score is zero
        // elsewhere.
        for (y in 0 until h) {
            for (x in 0 until w) {
                val i = y * w + x
                if (minContrast != null && amplitude[i] < minContrast) continue
                var deviation = 0.0
                val mean = boardMean[i]
                for (r in 0 until 8) {
                    val row = (y + r * cell) * means.width + x
                    for (c in 0 until 8) {
                        deviation += abs(means.data[row + c * cell] - mean)
                    }
                }
                score[i] = abs(acc[i]) / (deviation + 1e-6)
            }
        }
        return Response(w, h, score, amplitude)
    }

    /**
     * (y, x) of the gated score's peaks, best first, at most [limit], none within [radius].
     * A cell-sized dilation thins every plateau to the positions that equal their
     * neighbourhood's maximum, so a plateau yields one position and the windows into a
     * larger checkerboard one per neighbourhood.
     */
    fun localMaxima(
        response: Response,
        cell: Int,
        radius: Int,
        limit: Int,
        minScore: Double,
        minContrast: Double,
    ): List<Pair<Int, Int>> {
        val w = response.width
        val h = response.height
        val mask = BooleanArray(w * h) { response.score[it] >= minScore && response.amplitude[it] >= minContrast }
        val gated = DoubleArray(w * h) { if (mask[it]) response.score[it] else 0.0 }
        val mat = Mat(h, w, CvType.CV_64F)
        mat.put(0, 0, *gated)
        val dilated = Mat()
        val kernel = Mat.ones(cell, cell, CvType.CV_8U)
        Imgproc.dilate(mat, dilated, kernel)
        val peakOf = DoubleArray(w * h)
        dilated.get(0, 0, peakOf)
        mat.release()
        dilated.release()
        kernel.release()

        val ys = ArrayList<Int>()
        val xs = ArrayList<Int>()
        val values = ArrayList<Double>()
        for (y in 0 until h) {
            for (x in 0 until w) {
                val i = y * w + x
                if (mask[i] && gated[i] >= peakOf[i]) {
                    ys.add(y); xs.add(x); values.add(gated[i])
                }
            }
        }
        val taken = BooleanArray(ys.size)
        val peaks = ArrayList<Pair<Int, Int>>()
        while (peaks.size < limit) {
            var best = -1
            var bestValue = Double.NEGATIVE_INFINITY
            for (i in ys.indices) {
                if (!taken[i] && values[i] > bestValue) {
                    bestValue = values[i]; best = i
                }
            }
            if (best < 0) break
            val y = ys[best]
            val x = xs[best]
            peaks.add(y to x)
            for (i in ys.indices) {
                if (!taken[i] && abs(ys[i] - y) <= radius && abs(xs[i] - x) <= radius) taken[i] = true
            }
        }
        return peaks
    }

    /** Mean of the border ring of every cell-sized window, at stride 1, in doubles as cv2's integral. */
    fun ringMeans(gray: Plane, cell: Int, band: Int = 1): Means {
        if (cell <= 2 * band + 1) return cellMeans(gray, cell)
        val ii = integral(gray)
        val outer = windowSums(ii, gray.width, gray.height, cell)
        val innerSize = cell - 2 * band
        val inner = windowSums(ii, gray.width, gray.height, innerSize)
        val w = gray.width - cell + 1
        val h = gray.height - cell + 1
        val innerW = gray.width - innerSize + 1
        val denominator = (cell * cell - innerSize * innerSize).toDouble()
        val out = DoubleArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val o = outer[y * w + x]
                val i = inner[(y + band) * innerW + (x + band)]
                out[y * w + x] = (o - i) / denominator
            }
        }
        return Means(w, h, out)
    }

    /** Mean of every cell-sized window at stride 1. */
    fun cellMeans(gray: Plane, cell: Int): Means {
        val sums = windowSums(integral(gray), gray.width, gray.height, cell)
        val w = gray.width - cell + 1
        val h = gray.height - cell + 1
        val out = DoubleArray(w * h)
        val denominator = (cell * cell).toDouble()
        for (i in 0 until w * h) out[i] = sums[i] / denominator
        return Means(w, h, out)
    }

    /** A map of window means, in doubles. */
    class Means(val width: Int, val height: Int, val data: DoubleArray)

    /** The summed-area table, (width + 1) by (height + 1). */
    fun integral(plane: Plane): DoubleArray {
        val w = plane.width
        val h = plane.height
        val ii = DoubleArray((w + 1) * (h + 1))
        for (y in 0 until h) {
            var rowSum = 0.0
            val src = y * w
            val dst = (y + 1) * (w + 1)
            val above = y * (w + 1)
            for (x in 0 until w) {
                rowSum += plane.data[src + x]
                ii[dst + x + 1] = ii[above + x + 1] + rowSum
            }
        }
        return ii
    }

    /** Sum of every size-by-size window at stride 1, from an integral image. */
    fun windowSums(ii: DoubleArray, width: Int, height: Int, size: Int): DoubleArray {
        val w = width - size + 1
        val h = height - size + 1
        val out = DoubleArray(max(0, w) * max(0, h))
        if (w <= 0 || h <= 0) return out
        val stride = width + 1
        for (y in 0 until h) {
            val top = y * stride
            val bottom = (y + size) * stride
            val dst = y * w
            for (x in 0 until w) {
                out[dst + x] = ii[bottom + x + size] - ii[top + x + size] - ii[bottom + x] + ii[top + x]
            }
        }
        return out
    }

    fun <T> nms(candidates: List<T>, iouThreshold: Double = 0.3, key: (T) -> Double): List<T>
        where T : Any {
        @Suppress("UNCHECKED_CAST")
        val regionOf = { c: T -> (c as BoardCandidate).region }
        val kept = ArrayList<T>()
        for (candidate in candidates.sortedByDescending(key)) {
            if (kept.all { iou(regionOf(candidate), regionOf(it)) < iouThreshold }) kept.add(candidate)
        }
        return kept
    }

    fun iou(a: Region, b: Region): Double {
        val x1 = max(a.x, b.x)
        val y1 = max(a.y, b.y)
        val x2 = min(a.x + a.w, b.x + b.w)
        val y2 = min(a.y + a.h, b.y + b.h)
        val inter = max(0, x2 - x1).toLong() * max(0, y2 - y1).toLong()
        val union = a.w.toLong() * a.h + b.w.toLong() * b.h - inter
        return if (union > 0) inter.toDouble() / union else 0.0
    }

    /** The score and amplitude maps of one scale. */
    class Response(val width: Int, val height: Int, val score: DoubleArray, val amplitude: DoubleArray)
}

/** A pixel rectangle, as `sidecoach.events.Region` is on the desktop. */
data class Region(val x: Int, val y: Int, val w: Int, val h: Int)

/** A board found in a frame. `score` is the coarse checker score, before refinement. */
data class BoardCandidate(
    val region: Region,
    val cellPx: Double,
    val score: Double,
    val contrast: Double,
    val fit: Double = 0.0,
    /** Cells whose interior differs from their ring: pieces and marks. */
    val occupied: Int = 0,
    /** Cells whose ring reads as one of the two square colours. */
    val readable: Int = 0,
)

/** One grey plane of floats: the frame as the detector works on it. */
class Plane(val width: Int, val height: Int, val data: FloatArray) {

    operator fun get(y: Int, x: Int): Float = data[y * width + x]

    /** The plane scaled by [factor], with OpenCV's INTER_AREA, as the desktop shrinks it. */
    fun resized(factor: Double): Plane {
        val mat = toMat()
        val out = Mat()
        Imgproc.resize(mat, out, Size(0.0, 0.0), factor, factor, Imgproc.INTER_AREA)
        val plane = of(out)
        mat.release()
        out.release()
        return plane
    }

    fun toMat(): Mat {
        val mat = Mat(height, width, CvType.CV_32F)
        mat.put(0, 0, data)
        return mat
    }

    companion object {
        fun of(mat: Mat): Plane {
            val floats = FloatArray(mat.rows() * mat.cols())
            mat.get(0, 0, floats)
            return Plane(mat.cols(), mat.rows(), floats)
        }
    }
}
