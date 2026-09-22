package com.sidecoach.vision

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * One cell's silhouette, ported from `sidecoach/vision/pieces.py`: the mask, its tone, the
 * foreground fraction, the square's colour and spread, the centroid, the hover outline,
 * the corner pixels a tint is told by, and the share of the cell a drawing covers.
 *
 * The fields are the Python's, name for name. Masks are flat [BooleanArray]s of
 * [Squares.CELL] squared; the inner patch is BGR floats, the layout OpenCV hands back.
 */
class Sil(
    val mask: BooleanArray,
    /**
     * The mask restricted to the piece's own tone -- the dark fill of a black piece, the
     * bright fill of a white one -- with its enclosed detail filled in. A piece's outline
     * is the other tone, and whether it shows depends on the square under it: a light
     * outline vanishes on a light square and stands out on a dark one, so the full mask of
     * one piece is two shapes, and at small cells a black queen on a dark square matched
     * the pawn learned on a light one.
     */
    val tone: BooleanArray,
    val fraction: Double,
    val luminance: Double,
    val brightness: Double,
    val background: DoubleArray,
    val centroid: Pair<Double, Double>,
    val hover: Boolean,
    /** The inner patch the mask was taken from: BGR floats, row after row. */
    val inner: FloatArray,
    /** The four corner blocks, deeper than the ring: a tint reaches them, an outline does not. */
    val cornerPixels: FloatArray,
    /** The share of inner pixels that are a drawing over the square. */
    val overlay: Double = 0.0,
    val coreColour: DoubleArray? = null,
    val edgeColour: DoubleArray? = null,
    /** The two tones of the ink, each as (colour, weight), the fill first. */
    val tones: List<Pair<DoubleArray, Double>> = emptyList(),
    /** How far the square's own pixels stray from its colour (a wood grain's texture). */
    val spread: Double = 0.0,
)

/** The set's colours: a (fill, outline) pair for each side, learned from the start position. */
typealias PieceColours = List<Pair<DoubleArray, DoubleArray?>>

object SilhouetteReader {

    private val CELL = Squares.CELL

    /** Pixels that take part in silhouette comparison: everything but the badge corner. */
    val BADGE_BLIND: BooleanArray = badgeBlind(CELL)

    /** Pixels the overlay measure looks at: everything outside the badge's corner disc. */
    private val OVERLAY_BLIND: BooleanArray = overlayBlind(CELL)

    private fun badgeBlind(size: Int): BooleanArray {
        val keep = BooleanArray(size * size) { true }
        val rows = (size * 0.38).toInt()
        val cols = (size * 0.62).toInt()
        for (y in 0 until rows) for (x in cols until size) keep[y * size + x] = false
        return keep
    }

    private fun overlayBlind(size: Int): BooleanArray {
        val out = BooleanArray(size * size)
        for (y in 0 until size) {
            for (x in 0 until size) {
                val dx = (x - (size - 1)).toDouble()
                out[y * size + x] = hypot(dx, y.toDouble()) / size > Pieces.OVERLAY_BADGE_RADIUS
            }
        }
        return out
    }

    /**
     * Silhouette mask, foreground fraction, colour statistics, centroid, hover, background.
     * [colours] are the set's fills and outlines, for the tones; [palette] every colour the
     * set draws with, for telling a drawing from a piece's detail.
     */
    fun of(
        patch: Mat,
        size: Int = CELL,
        colours: PieceColours = emptyList(),
        palette: List<DoubleArray> = emptyList(),
    ): Sil {
        val n = patch.rows()
        // The crop is 8-bit BGR, as the capture hands it over; the maths wants floats.
        val patchBytes = ByteArray(n * patch.cols() * 3)
        patch.get(0, 0, patchBytes)
        val patchPixels = FloatArray(patchBytes.size) { (patchBytes[it].toInt() and 0xFF).toFloat() }

        val (background, spread) = ringMedianAndSpread(patchPixels, n, patch.cols())
        val border = (n * Pieces.INNER).roundToInt()
        val side = n - 2 * border
        if (side < 4) {
            return Sil(
                BooleanArray(size * size), BooleanArray(size * size), 0.0, 0.0, 0.0,
                background, 0.5 to 0.5, false, FloatArray(size * size * 3),
                FloatArray(0), 0.0, null, null, emptyList(), spread,
            )
        }
        val cut = patch.submat(Rect(border, border, side, side))
        val innerMat = Mat()
        Imgproc.resize(cut, innerMat, Size(size.toDouble(), size.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        cut.release()
        val innerBytes = ByteArray(size * size * 3)
        innerMat.get(0, 0, innerBytes)
        innerMat.release()
        val inner = FloatArray(innerBytes.size) { (innerBytes[it].toInt() and 0xFF).toFloat() }

        // Distance of every inner pixel from the square's colour, and its chroma.
        val distance = FloatArray(size * size)
        val chroma = FloatArray(size * size)
        val gray = FloatArray(size * size)
        for (i in 0 until size * size) {
            val b = inner[i * 3]
            val g = inner[i * 3 + 1]
            val r = inner[i * 3 + 2]
            val db = b - background[0]
            val dg = g - background[1]
            val dr = r - background[2]
            distance[i] = sqrt(db * db + dg * dg + dr * dr).toFloat()
            chroma[i] = max(b, max(g, r)) - min(b, min(g, r))
            gray[i] = (0.114f * b + 0.587f * g + 0.299f * r)
        }

        val overlay = overlayFraction(inner, size, background, distance, chroma, palette)
        val fills = colours.map { it.first }
        val pocket = min(Pieces.BG_TOLERANCE, max(Pieces.POCKET_TOLERANCE, 3.0 * spread))
        var built = maskAndFill(inner, size, gray, distance, Pieces.BG_TOLERANCE, pocket, fills)
        if (built.stats != null && fills.isNotEmpty()) {
            // The piece's own fill against this square: with little contrast, read again
            // with a tolerance to match (see BG_TOLERANCE_MIN).
            val core = built.stats.colour
            val nearest = fills.minByOrNull { distanceOf(core, it) }!!
            val contrast = distanceOf(nearest, background)
            val tolerance = max(Pieces.BG_TOLERANCE_MIN, Pieces.BG_CONTRAST_SHARE * contrast)
            if (tolerance < Pieces.BG_TOLERANCE) {
                built = maskAndFill(inner, size, gray, distance, tolerance, pocket, fills)
            }
        }

        var luminance = 0.0
        var brightness = 0.0
        var centroid = 0.5 to 0.5
        var tone = built.mask
        var coreColour: DoubleArray? = null
        var edgeColour: DoubleArray? = null
        var tones: List<Pair<DoubleArray, Double>> = emptyList()
        val stats = built.stats
        if (stats != null) {
            luminance = stats.luminance
            tones = stats.tones
            coreColour = stats.colour
            tone = toneMask(built.mask, size, inner, stats.colour, background)
            brightness = luminance / 255.0
            centroid = centroidOf(built.mask, size)
            var outlinePixels = 0
            for (i in built.mask.indices) if (built.mask[i] && !tone[i]) outlinePixels++
            if (outlinePixels >= Pieces.EDGE_MIN_PIXELS) {
                edgeColour = medianColour(inner) { i -> built.mask[i] && !tone[i] }
            }
        }
        // Chess.com's hover outline is about 5% of the cell wide and sits at its edge, so
        // the ring the background is read from stays clean and serves as the reference.
        val hover = hoverOutline(patchPixels, n, patch.cols(), background)
        return Sil(
            built.mask, tone, built.fraction, luminance, brightness, background, centroid,
            hover, inner, cornerPixels(patchPixels, n, patch.cols()), overlay,
            coreColour, edgeColour, tones, spread,
        )
    }

    /**
     * The square's colour (the ring's median) and how far its own pixels stray from it
     * (the median distance): nothing on a flat square, a dozen levels on a wood grain.
     */
    fun ringMedianAndSpread(
        pixels: FloatArray,
        n: Int,
        width: Int,
        skip: Double = Pieces.RING_SKIP,
        band: Double = Pieces.RING_BAND,
    ): Pair<DoubleArray, Double> {
        val outer = max(1, (n * skip).roundToInt())
        val inner = outer + max(1, (n * band).roundToInt())
        val bs = ArrayList<Float>(4 * n * (inner - outer))
        val gs = ArrayList<Float>(bs.size)
        val rs = ArrayList<Float>(bs.size)
        fun take(y: Int, x: Int) {
            val i = (y * width + x) * 3
            bs.add(pixels[i]); gs.add(pixels[i + 1]); rs.add(pixels[i + 2])
        }
        if (n - outer > outer && inner < n - inner) {
            for (y in outer until inner) for (x in outer until n - outer) take(y, x)
            for (y in n - inner until n - outer) for (x in outer until n - outer) take(y, x)
            for (y in inner until n - inner) {
                for (x in outer until inner) take(y, x)
                for (x in n - inner until n - outer) take(y, x)
            }
        }
        if (bs.isEmpty()) {  // a patch too small to have a ring at all
            for (y in 0 until n) for (x in 0 until width) take(y, x)
        }
        val median = doubleArrayOf(medianOf(bs), medianOf(gs), medianOf(rs))
        val distances = ArrayList<Float>(bs.size)
        for (i in bs.indices) {
            val db = bs[i] - median[0]
            val dg = gs[i] - median[1]
            val dr = rs[i] - median[2]
            distances.add(sqrt(db * db + dg * dg + dr * dr).toFloat())
        }
        return median to medianOf(distances)
    }

    /**
     * The pixels of the four corner blocks between the two insets. Deeper than the ring,
     * so a thin hover outline never reaches them, yet where a piece rarely is.
     */
    fun cornerPixels(
        pixels: FloatArray,
        n: Int,
        width: Int,
        inset: Pair<Double, Double> = Pieces.CORNER_INSET,
    ): FloatArray {
        val outer = (n * inset.first).roundToInt()
        val inner = max(outer + 1, (n * inset.second).roundToInt())
        val out = ArrayList<Float>(4 * (inner - outer) * (inner - outer) * 3)
        fun block(y0: Int, y1: Int, x0: Int, x1: Int) {
            for (y in max(0, y0) until min(n, y1)) {
                for (x in max(0, x0) until min(width, x1)) {
                    val i = (y * width + x) * 3
                    out.add(pixels[i]); out.add(pixels[i + 1]); out.add(pixels[i + 2])
                }
            }
        }
        block(outer, inner, outer, inner)
        block(outer, inner, n - inner, n - outer)
        block(n - inner, n - outer, outer, inner)
        block(n - inner, n - outer, n - inner, n - outer)
        if (out.isEmpty()) return pixels.copyOf()  // a patch too small to have corners
        return FloatArray(out.size) { out[it] }
    }

    private class Built(val mask: BooleanArray, val fraction: Double, val stats: Fill?)

    private class Fill(
        val colour: DoubleArray,
        val luminance: Double,
        val tones: List<Pair<DoubleArray, Double>>,
    )

    /**
     * The silhouette at [tolerance] and, unless it is empty, its fill's colour, luminance
     * and tones. A piece whose fill is the square's own colour is only its outline, so what
     * the outline encloses is filled in; the outline is closed first, since a thin stroke
     * breaks at its anti-aliased joins, and a broken outline enclosed nothing.
     */
    private fun maskAndFill(
        inner: FloatArray,
        size: Int,
        gray: FloatArray,
        distance: FloatArray,
        tolerance: Double,
        pocket: Double,
        fills: List<DoubleArray>,
    ): Built {
        val foreground = ByteArray(size * size)
        for (i in foreground.indices) if (distance[i] > tolerance) foreground[i] = 1
        val mat = Mat(size, size, CvType.CV_8U)
        mat.put(0, 0, foreground)
        val kernel = Mat.ones(3, 3, CvType.CV_8U)
        Imgproc.morphologyEx(mat, mat, Imgproc.MORPH_CLOSE, kernel)
        fillEnclosed(mat)
        Imgproc.morphologyEx(mat, mat, Imgproc.MORPH_OPEN, kernel)
        centred(mat)
        val bytes = ByteArray(size * size)
        mat.get(0, 0, bytes)
        mat.release()
        kernel.release()
        val mask = BooleanArray(size * size) { bytes[it].toInt() != 0 }
        var on = 0
        for (m in mask) if (m) on++
        val fraction = on.toDouble() / (size * size)
        if (on == 0) return Built(mask, fraction, null)

        // The fill's colour. Not the badge corner: a classification badge's disc lies
        // inside the mask, and its bright ink swung a black queen to white. The ink: the
        // silhouette less the pockets where the square itself shows through.
        var anyBadge = false
        for (i in mask.indices) if (mask[i] && BADGE_BLIND[i]) { anyBadge = true; break }
        val keep = if (anyBadge) BADGE_BLIND else BooleanArray(size * size) { true }
        var ink = BooleanArray(size * size) { mask[it] && distance[it] > pocket }
        var anyInk = false
        for (v in ink) if (v) { anyInk = true; break }
        if (!anyInk) ink = mask
        return Built(mask, fraction, fillColour(inner, size, gray, ink, keep, fills))
    }

    /** What the foreground encloses becomes foreground (`pieces._fill_enclosed`). */
    private fun fillEnclosed(mask: Mat) {
        val n = mask.rows()
        val bytes = ByteArray(n * mask.cols())
        mask.get(0, 0, bytes)
        val padded = Mat.zeros(n + 2, mask.cols() + 2, CvType.CV_8U)
        val paddedBytes = ByteArray((n + 2) * (mask.cols() + 2))
        for (y in 0 until n) {
            for (x in 0 until mask.cols()) {
                paddedBytes[(y + 1) * (mask.cols() + 2) + x + 1] = if (bytes[y * mask.cols() + x].toInt() == 0) 1 else 0
            }
        }
        for (x in 0 until mask.cols() + 2) {
            paddedBytes[x] = 1
            paddedBytes[(n + 1) * (mask.cols() + 2) + x] = 1
        }
        for (y in 0 until n + 2) {
            paddedBytes[y * (mask.cols() + 2)] = 1
            paddedBytes[y * (mask.cols() + 2) + mask.cols() + 1] = 1
        }
        padded.put(0, 0, paddedBytes)
        val flood = Mat.zeros(n + 4, mask.cols() + 4, CvType.CV_8U)
        Imgproc.floodFill(padded, flood, Point(0.0, 0.0), Scalar(2.0))
        padded.get(0, 0, paddedBytes)
        for (y in 0 until n) {
            for (x in 0 until mask.cols()) {
                bytes[y * mask.cols() + x] =
                    if (paddedBytes[(y + 1) * (mask.cols() + 2) + x + 1].toInt() != 2) 1 else 0
            }
        }
        mask.put(0, 0, bytes)
        padded.release()
        flood.release()
    }

    /**
     * The connected parts of the mask that are not confined to a corner: a coordinate label
     * sits in the top-left or bottom-right corner, a badge in the top-right, and each lies
     * wholly within its corner's box; a piece, whatever it is made of, does not.
     */
    private fun centred(mask: Mat) {
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        val count = Imgproc.connectedComponentsWithStats(mask, labels, stats, centroids, 8)
        if (count <= 1) {
            labels.release(); stats.release(); centroids.release(); return
        }
        val n = mask.rows()
        val box = n * Pieces.CORNER_BOX
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

    /**
     * The mask restricted to the piece's own tone: the pixels nearer the colour of the
     * piece's core than the square's. The outline is the other tone, and it is nearer the
     * square wherever the square can show it.
     */
    private fun toneMask(
        mask: BooleanArray,
        size: Int,
        inner: FloatArray,
        coreColour: DoubleArray,
        background: DoubleArray,
    ): BooleanArray {
        val core = BooleanArray(size * size)
        var on = 0
        var maskOn = 0
        for (i in mask.indices) {
            if (!mask[i]) continue
            maskOn++
            val b = inner[i * 3].toDouble()
            val g = inner[i * 3 + 1].toDouble()
            val r = inner[i * 3 + 2].toDouble()
            val toCore = distanceOf(doubleArrayOf(b, g, r), coreColour)
            val toSquare = distanceOf(doubleArrayOf(b, g, r), background)
            if (toCore < toSquare) { core[i] = true; on++ }
        }
        if (on < max(Pieces.TONE_MIN_SHARE * maskOn, Pieces.TONE_MIN_AREA * size * size)) return mask
        val mat = Mat(size, size, CvType.CV_8U)
        val bytes = ByteArray(size * size)
        for (i in core.indices) bytes[i] = if (core[i]) 1 else 0
        mat.put(0, 0, bytes)
        fillEnclosed(mat)
        mat.get(0, 0, bytes)
        mat.release()
        return BooleanArray(size * size) { bytes[it].toInt() != 0 && mask[it] }
    }

    /**
     * The fill's colour and luminance, and the ink's tones as (colour, weight), the fill
     * first. A tone's weight is the depth of its deepest pixel: the widest region of one
     * tone is the fill.
     */
    private fun fillColour(
        inner: FloatArray,
        size: Int,
        gray: FloatArray,
        ink: BooleanArray,
        keep: BooleanArray,
        fills: List<DoubleArray>,
    ): Fill? {
        val classes: List<BooleanArray>
        if (fills.size >= 2) {
            classes = List(fills.size) { BooleanArray(size * size) }
            for (i in ink.indices) {
                if (!ink[i]) continue
                val pixel = doubleArrayOf(inner[i * 3].toDouble(), inner[i * 3 + 1].toDouble(), inner[i * 3 + 2].toDouble())
                var best = 0
                var bestDistance = Double.MAX_VALUE
                for ((k, fill) in fills.withIndex()) {
                    val d = distanceOf(pixel, fill)
                    if (d < bestDistance) { bestDistance = d; best = k }
                }
                classes[best][i] = true
            }
        } else {
            val values = ArrayList<Float>()
            for (i in ink.indices) if (ink[i]) values.add(gray[i])
            if (values.isEmpty()) return null
            val split = otsu(values)
            classes = listOf(
                BooleanArray(size * size) { ink[it] && gray[it] > split },
                BooleanArray(size * size) { ink[it] && gray[it] <= split },
            )
        }
        val tones = ArrayList<Triple<DoubleArray, Double, Double>>()
        for (tone in classes) {
            var any = false
            for (v in tone) if (v) { any = true; break }
            if (!any) continue
            // Measured from the patch's edge as well, or a piece reaching the edge has a
            // tone that runs off it and no edge to be shallow against.
            val depth = distanceTransform(tone, size)
            var deepest = 0.0
            val weights = FloatArray(size * size)
            for (i in tone.indices) {
                if (!tone[i]) continue
                val d = min(depth[i].toDouble(), size.toDouble()) * (if (keep[i]) 1.0 else 0.0)
                weights[i] = (d * d).toFloat()
                if (d > deepest) deepest = d
            }
            var sum = 0.0
            for (w in weights) sum += w
            if (sum <= 0) for (i in tone.indices) if (tone[i]) weights[i] = 1f
            val colour = DoubleArray(3) { ch -> weightedMedianOf(tone, weights) { i -> inner[i * 3 + ch] } }
            val luminance = weightedMedianOf(tone, weights) { i -> gray[i] }
            tones.add(Triple(colour, deepest, luminance))
        }
        if (tones.isEmpty()) return null
        tones.sortByDescending { it.second }
        val (colour, _, luminance) = tones[0]
        return Fill(colour, luminance, tones.map { it.first to it.second })
    }

    /**
     * Whether an outline differing from the cell's background runs along all four sides:
     * chess.com draws a thin bright one around the square under the cursor during a drag.
     * A badge, a dialog edge or a blended boundary marks one or two sides, never all four.
     */
    private fun hoverOutline(pixels: FloatArray, n: Int, width: Int, background: DoubleArray): Boolean {
        val band = max(1, (n * Pieces.HOVER_BAND).roundToInt())
        val edge = max(1, (n * Pieces.HOVER_EDGE).roundToInt())
        if (2 * band >= n) return false  // a patch too small to have an outer band
        val distance = FloatArray(n * width)
        for (y in 0 until n) {
            for (x in 0 until width) {
                val i = (y * width + x) * 3
                val db = pixels[i] - background[0]
                val dg = pixels[i + 1] - background[1]
                val dr = pixels[i + 2] - background[2]
                distance[y * width + x] = sqrt(db * db + dg * dg + dr * dr).toFloat()
            }
        }
        // The four sides, each read as (row, column) of a rotated view.
        val strong = DoubleArray(4)
        val weak = DoubleArray(4)
        val medians = DoubleArray(4)
        val steps = DoubleArray(4)
        for (sideIndex in 0 until 4) {
            fun at(k: Int, along: Int): Int = when (sideIndex) {
                0 -> k * width + along                    // top
                1 -> (n - 1 - k) * width + along          // bottom
                2 -> along * width + k                    // left
                else -> along * width + (width - 1 - k)   // right
            }
            var bestStrong = 0.0
            var bestWeak = 0.0
            for (k in 0 until band) {
                var hits = 0
                var halfHits = 0
                var count = 0
                for (along in band until n - band) {
                    val d = distance[at(k, along)]
                    if (d > Pieces.HOVER_TOLERANCE) hits++
                    if (d > Pieces.HOVER_TOLERANCE / 2) halfHits++
                    count++
                }
                if (count > 0) {
                    bestStrong = max(bestStrong, hits.toDouble() / count)
                    bestWeak = max(bestWeak, halfHits.toDouble() / count)
                }
            }
            strong[sideIndex] = bestStrong
            weak[sideIndex] = bestWeak
            val edgeValues = ArrayList<Float>(edge * n)
            for (k in 0 until edge) for (along in 0 until n) edgeValues.add(distance[at(k, along)])
            medians[sideIndex] = medianOf(edgeValues)
            // An outline is a step between one row and the next, and it runs through both
            // ends of the side; a tint that fades toward the edge is a ramp of a few levels
            // a row, and a piece's base or a label reaches one end at most.
            val reach = max(2, (n * Pieces.HOVER_CORNER).roundToInt())
            var worst = Double.MAX_VALUE
            for ((lo, hi) in listOf(band to band + reach, n - band - reach to n - band)) {
                val lines = ArrayList<DoubleArray>()
                for (k in 0 until min(n, 2 * band + 1)) {
                    val bs = ArrayList<Float>(); val gs = ArrayList<Float>(); val rs = ArrayList<Float>()
                    for (along in max(0, lo) until min(n, hi)) {
                        val p = at(k, along) * 3
                        bs.add(pixels[p]); gs.add(pixels[p + 1]); rs.add(pixels[p + 2])
                    }
                    if (bs.isEmpty()) continue
                    lines.add(doubleArrayOf(medianOf(bs), medianOf(gs), medianOf(rs)))
                }
                var biggest = 0.0
                for (k in 0 until lines.size - 1) {
                    biggest = max(biggest, distanceOf(lines[k + 1], lines[k]))
                }
                worst = min(worst, biggest)
            }
            steps[sideIndex] = if (worst == Double.MAX_VALUE) 0.0 else worst
        }
        strong.sort(); weak.sort(); medians.sort(); steps.sort()
        if (steps[1] < Pieces.HOVER_STEP) return false  // no step on three sides: a ramp
        if (medians[0] > Pieces.HOVER_TOLERANCE / 2 && medians[2] > Pieces.HOVER_TOLERANCE) return true
        return weak[0] >= Pieces.HOVER_LINE_FILL && strong[1] >= Pieces.HOVER_LINE_FILL
    }

    /**
     * The share of inner pixels that are a drawing: chromatic, far from the square's
     * colour, and off the lines from that colour to black, to white and to each colour of
     * the set's palette, and off the lines between any two of those.
     */
    private fun overlayFraction(
        inner: FloatArray,
        size: Int,
        background: DoubleArray,
        distance: FloatArray,
        chroma: FloatArray,
        palette: List<DoubleArray>,
    ): Double {
        val blind = if (size == CELL) OVERLAY_BLIND else overlayBlind(size)
        val candidates = ArrayList<Int>()
        for (i in 0 until size * size) {
            if (distance[i] > Pieces.OVERLAY_DISTANCE && chroma[i] > Pieces.OVERLAY_CHROMA && blind[i]) {
                candidates.add(i)
            }
        }
        if (candidates.isEmpty()) return 0.0
        val colours = ArrayList<DoubleArray>()
        colours.add(doubleArrayOf(0.0, 0.0, 0.0))
        colours.add(doubleArrayOf(255.0, 255.0, 255.0))
        colours.addAll(palette)
        val segments = ArrayList<Pair<DoubleArray, DoubleArray>>()
        for (colour in colours) segments.add(background to colour)
        for (i in colours.indices) for (j in i + 1 until colours.size) segments.add(colours[i] to colours[j])
        var drawn = 0
        for (index in candidates) {
            val pixel = doubleArrayOf(
                inner[index * 3].toDouble(), inner[index * 3 + 1].toDouble(), inner[index * 3 + 2].toDouble(),
            )
            var nearest = Double.MAX_VALUE
            for ((start, end) in segments) {
                nearest = min(nearest, offLine(pixel, start, end))
                if (nearest <= Pieces.OVERLAY_OFF_LINE) break
            }
            if (nearest > Pieces.OVERLAY_OFF_LINE) drawn++
        }
        return drawn.toDouble() / (size * size)
    }

    /** How far a pixel lies from the segment of colours between [start] and [end]. */
    private fun offLine(pixel: DoubleArray, start: DoubleArray, end: DoubleArray): Double {
        var length = 0.0
        val direction = DoubleArray(3) { end[it] - start[it] }
        for (d in direction) length += d * d
        if (length == 0.0) length = 1.0
        var dot = 0.0
        for (i in 0 until 3) dot += (pixel[i] - start[i]) * direction[i]
        val along = (dot / length).coerceIn(0.0, 1.0)
        var sum = 0.0
        for (i in 0 until 3) {
            val d = pixel[i] - (start[i] + along * direction[i])
            sum += d * d
        }
        return sqrt(sum)
    }

    /** Centre of mass of a mask in units of its own size; the middle when it is empty. */
    fun centroidOf(mask: BooleanArray, size: Int): Pair<Double, Double> {
        var sx = 0.0
        var sy = 0.0
        var n = 0
        for (i in mask.indices) {
            if (!mask[i]) continue
            sx += (i % size).toDouble()
            sy += (i / size).toDouble()
            n++
        }
        if (n == 0) return 0.5 to 0.5
        return (sx / n / size) to (sy / n / size)
    }

    /** Where a piece sits for alignment: the centroid of the mask outside the badge corner. */
    fun alignCentroid(mask: BooleanArray, size: Int = CELL): Pair<Double, Double> {
        val blind = if (size == CELL) BADGE_BLIND else badgeBlind(size)
        return centroidOf(BooleanArray(mask.size) { mask[it] && blind[it] }, size)
    }

    /** Intersection over union of two masks, outside the badge corner. */
    fun iou(a: BooleanArray, b: BooleanArray, keep: BooleanArray = BADGE_BLIND): Double {
        var both = 0
        var either = 0
        for (i in a.indices) {
            val x = a[i] && keep[i]
            val y = b[i] && keep[i]
            if (x && y) both++
            if (x || y) either++
        }
        return if (either > 0) both.toDouble() / either else 0.0
    }

    // ---- small helpers -------------------------------------------------------------

    private fun distanceOf(a: DoubleArray, b: DoubleArray): Double {
        var sum = 0.0
        for (i in 0 until 3) {
            val d = a[i] - b[i]
            sum += d * d
        }
        return sqrt(sum)
    }

    private fun medianOf(values: List<Float>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val n = sorted.size
        return if (n % 2 == 1) sorted[n / 2].toDouble()
        else (sorted[n / 2 - 1].toDouble() + sorted[n / 2].toDouble()) / 2.0
    }

    private fun medianColour(inner: FloatArray, where: (Int) -> Boolean): DoubleArray {
        val bs = ArrayList<Float>(); val gs = ArrayList<Float>(); val rs = ArrayList<Float>()
        for (i in 0 until inner.size / 3) {
            if (!where(i)) continue
            bs.add(inner[i * 3]); gs.add(inner[i * 3 + 1]); rs.add(inner[i * 3 + 2])
        }
        return doubleArrayOf(medianOf(bs), medianOf(gs), medianOf(rs))
    }

    private fun weightedMedianOf(mask: BooleanArray, weights: FloatArray, value: (Int) -> Float): Double {
        val picked = ArrayList<Pair<Float, Float>>()
        var total = 0.0
        for (i in mask.indices) {
            if (!mask[i]) continue
            picked.add(value(i) to weights[i])
            total += weights[i]
        }
        if (picked.isEmpty()) return 0.0
        picked.sortBy { it.first }
        var running = 0.0
        for ((v, w) in picked) {
            running += w
            if (running >= total / 2) return v.toDouble()
        }
        return picked.last().first.toDouble()
    }

    /** Otsu's threshold over the ink's grey values. */
    private fun otsu(values: List<Float>): Float {
        val histogram = IntArray(256)
        for (v in values) histogram[v.toInt().coerceIn(0, 255)]++
        val total = values.size
        var sum = 0.0
        for (i in 0 until 256) sum += i * histogram[i]
        var sumB = 0.0
        var weightB = 0
        var best = 0.0
        var threshold = 0
        for (t in 0 until 256) {
            weightB += histogram[t]
            if (weightB == 0) continue
            val weightF = total - weightB
            if (weightF == 0) break
            sumB += t * histogram[t]
            val meanB = sumB / weightB
            val meanF = (sum - sumB) / weightF
            val between = weightB.toDouble() * weightF * (meanB - meanF) * (meanB - meanF)
            if (between > best) { best = between; threshold = t }
        }
        return threshold.toFloat()
    }

    /** The L2 distance transform of a mask, measured from the patch's edge as well. */
    private fun distanceTransform(mask: BooleanArray, size: Int): FloatArray {
        val padded = Mat.zeros(size + 2, size + 2, CvType.CV_8U)
        val bytes = ByteArray((size + 2) * (size + 2))
        for (y in 0 until size) {
            for (x in 0 until size) {
                if (mask[y * size + x]) bytes[(y + 1) * (size + 2) + x + 1] = 1
            }
        }
        padded.put(0, 0, bytes)
        val distance = Mat()
        Imgproc.distanceTransform(padded, distance, Imgproc.DIST_L2, 3)
        val all = FloatArray((size + 2) * (size + 2))
        distance.get(0, 0, all)
        padded.release(); distance.release()
        val out = FloatArray(size * size)
        for (y in 0 until size) {
            for (x in 0 until size) out[y * size + x] = all[(y + 1) * (size + 2) + x + 1]
        }
        return out
    }
}
