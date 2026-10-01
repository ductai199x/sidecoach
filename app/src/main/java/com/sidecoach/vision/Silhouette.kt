package com.sidecoach.vision

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * One cell's silhouette, ported from `sidecoach/vision/pieces.py`: the mask, its tone, the
 * foreground fraction, the square's colour and spread, the hover outline,
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
    val hover: Boolean,
    /** The inner patch the mask was taken from: BGR floats, row after row. */
    val inner: FloatArray,
    /** The four corner blocks, deeper than the ring: a tint reaches them, an outline does not. */
    val cornerPixels: FloatArray,
    /** The share of inner pixels that are a drawing over the square. */
    val overlay: Double = 0.0,
    /** The two tones of the ink, each as (colour, weight), the fill first. */
    val tones: List<Pair<DoubleArray, Double>> = emptyList(),
    /** How far the square's own pixels stray from its colour (a wood grain's texture). */
    val spread: Double = 0.0,
)

/** The set's colours: a (fill, outline) pair for each side, learned from the start position. */
typealias PieceColours = List<Pair<DoubleArray, DoubleArray?>>

object SilhouetteReader {

    private val CELL = Squares.CELL

    /** Everything but the badge corner: where the fill's colour is read. */
    private val BADGE_BLIND: BooleanArray = badgeBlind(CELL)

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
     * Silhouettes by the square's pixels and what they were read with (`pieces.silhouette`,
     * desktop a3479bd): every frame shows the same few square images again (an empty light
     * square, a pawn on a dark one), each was read again every frame, at every square, and by
     * each reader (the frame's, the occupancy's, the start position's). A [Sil] is never
     * changed once made, so one serves every square that shows those pixels. The desktop keeps
     * 1024; a Kotlin [Sil] holds its inner patch as floats, about 33 kB, four times the
     * desktop's bytes, so the phone keeps fewer.
     */
    const val SILHOUETTES_KEPT = 256

    private val silhouettes = object : LinkedHashMap<Digest, Sil>(SILHOUETTES_KEPT, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<Digest, Sil>?): Boolean =
            size > SILHOUETTES_KEPT
    }

    /**
     * Silhouette mask, foreground fraction, colour statistics, hover, background.
     * [colours] are the set's fills and outlines, for the tones; [palette] every colour the
     * set draws with, for telling a drawing from a piece's detail. Read once for the same
     * pixels and the same colours (see [SILHOUETTES_KEPT]). [bytes] are the patch's own when
     * the caller has them already.
     */
    fun of(
        patch: Mat,
        size: Int = CELL,
        colours: PieceColours = emptyList(),
        palette: List<DoubleArray> = emptyList(),
        bytes: ByteArray? = null,
    ): Sil {
        // The crop is 8-bit BGR, as the capture hands it over.
        val patchBytes = bytes ?: ByteArray(patch.rows() * patch.cols() * 3).also { patch.get(0, 0, it) }
        val key = Digest.of(patchBytes) { digest ->
            digest.longs(patch.rows().toLong(), patch.cols().toLong(), size.toLong())
            for ((fill, outline) in colours) {
                digest.doubles(fill)
                if (outline == null) digest.longs(0L) else digest.longs(1L).doubles(outline)
            }
            digest.longs(-1L)
            for (colour in palette) digest.doubles(colour)
        }
        synchronized(silhouettes) { silhouettes[key] }?.let { return it }
        val sil = read(patch, patchBytes, size, colours, palette)
        synchronized(silhouettes) { silhouettes[key] = sil }
        return sil
    }

    private fun read(
        patch: Mat,
        patchBytes: ByteArray,
        size: Int,
        colours: PieceColours,
        palette: List<DoubleArray>,
    ): Sil {
        val n = patch.rows()
        // The maths wants floats.
        val patchPixels = FloatArray(patchBytes.size) { (patchBytes[it].toInt() and 0xFF).toFloat() }

        val (background, spread) = ringMedianAndSpread(patchPixels, n, patch.cols())
        val border = pyRound(n * Pieces.INNER)
        val side = n - 2 * border
        if (side < 4) {
            return Sil(
                BooleanArray(size * size), BooleanArray(size * size), 0.0, 0.0, 0.0,
                background, false, FloatArray(size * size * 3),
                FloatArray(0), 0.0, emptyList(), spread,
            )
        }
        val cut = patch.submat(Rect(border, border, side, side))
        val innerMat = Mat()
        Imgproc.resize(cut, innerMat, Size(size.toDouble(), size.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        cut.release()
        val innerBytes = ByteArray(size * size * 3)
        innerMat.get(0, 0, innerBytes)
        // The grey OpenCV makes of the 8-bit patch, `cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)`:
        // whole levels, rounded its own way, which the Otsu split and the luminance read.
        val grayMat = Mat()
        Imgproc.cvtColor(innerMat, grayMat, Imgproc.COLOR_BGR2GRAY)
        val grayBytes = ByteArray(size * size)
        grayMat.get(0, 0, grayBytes)
        grayMat.release()
        innerMat.release()
        val inner = FloatArray(innerBytes.size) { (innerBytes[it].toInt() and 0xFF).toFloat() }

        // Distance of every inner pixel from the square's colour, and its chroma.
        val distance = FloatArray(size * size)
        val chroma = FloatArray(size * size)
        val gray = FloatArray(size * size) { (grayBytes[it].toInt() and 0xFF).toFloat() }
        for (i in 0 until size * size) {
            val b = inner[i * 3]
            val g = inner[i * 3 + 1]
            val r = inner[i * 3 + 2]
            val db = b - background[0]
            val dg = g - background[1]
            val dr = r - background[2]
            distance[i] = sqrt(db * db + dg * dg + dr * dr).toFloat()
            chroma[i] = max(b, max(g, r)) - min(b, min(g, r))
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
        var tone = built.mask
        var tones: List<Pair<DoubleArray, Double>> = emptyList()
        val stats = built.stats
        if (stats != null) {
            luminance = stats.luminance
            tones = stats.tones
            tone = toneMask(built.mask, size, inner, stats.colour, background)
            brightness = luminance / 255.0
        }
        // Chess.com's hover outline is about 5% of the cell wide and sits at its edge, so
        // the ring the background is read from stays clean and serves as the reference.
        val hover = hoverOutline(patchPixels, n, patch.cols(), background)
        return Sil(
            built.mask, tone, built.fraction, luminance, brightness, background,
            hover, inner, cornerPixels(patchPixels, n, patch.cols()), overlay, tones, spread,
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
        val outer = max(1, pyRound(n * skip))
        val inner = outer + max(1, pyRound(n * band))
        val ringed = n - outer > outer && inner < n - inner
        val capacity = if (ringed) {
            2 * (inner - outer) * (n - 2 * outer) + 2 * (n - 2 * inner) * (inner - outer)
        } else {
            n * width
        }
        val bs = FloatArray(capacity)
        val gs = FloatArray(capacity)
        val rs = FloatArray(capacity)
        var count = 0
        fun take(y: Int, x: Int) {
            val i = (y * width + x) * 3
            bs[count] = pixels[i]; gs[count] = pixels[i + 1]; rs[count] = pixels[i + 2]
            count++
        }
        if (ringed) {
            for (y in outer until inner) for (x in outer until n - outer) take(y, x)
            for (y in n - inner until n - outer) for (x in outer until n - outer) take(y, x)
            for (y in inner until n - inner) {
                for (x in outer until inner) take(y, x)
                for (x in n - inner until n - outer) take(y, x)
            }
        }
        if (count == 0) {  // a patch too small to have a ring at all
            for (y in 0 until n) for (x in 0 until width) take(y, x)
        }
        val median = doubleArrayOf(medianOf(bs, count), medianOf(gs, count), medianOf(rs, count))
        // In doubles, as numpy measures a float32 ring against a float64 median.
        val distances = DoubleArray(count)
        for (i in 0 until count) {
            val db = bs[i] - median[0]
            val dg = gs[i] - median[1]
            val dr = rs[i] - median[2]
            distances[i] = sqrt(db * db + dg * dg + dr * dr)
        }
        distances.sort()
        val spread = if (count % 2 == 1) distances[count / 2]
        else (distances[count / 2 - 1] + distances[count / 2]) / 2.0
        return median to spread
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
        val outer = pyRound(n * inset.first)
        val inner = max(outer + 1, pyRound(n * inset.second))
        val out = FloatArray(4 * (inner - outer) * (inner - outer) * 3)
        var count = 0
        fun block(y0: Int, y1: Int, x0: Int, x1: Int) {
            for (y in max(0, y0) until min(n, y1)) {
                for (x in max(0, x0) until min(width, x1)) {
                    val i = (y * width + x) * 3
                    out[count++] = pixels[i]; out[count++] = pixels[i + 1]; out[count++] = pixels[i + 2]
                }
            }
        }
        block(outer, inner, outer, inner)
        block(outer, inner, n - inner, n - outer)
        block(n - inner, n - outer, outer, inner)
        block(n - inner, n - outer, n - inner, n - outer)
        if (count == 0) return pixels.copyOf()  // a patch too small to have corners
        return if (count == out.size) out else out.copyOf(count)
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
        // numpy compares a float32 map with a Python float in float32 (NEP 50): so here.
        val cutoff = tolerance.toFloat()
        val foreground = ByteArray(size * size)
        var over = false
        for (i in foreground.indices) if (distance[i] > cutoff) { foreground[i] = 1; over = true }
        // Nothing off the square's colour: the mask is empty, as the closing, the filling and
        // the opening below would leave it -- most cells of a board, every frame (a3479bd).
        if (!over) return Built(BooleanArray(size * size), 0.0, null)
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
        val pocketCut = pocket.toFloat()
        var ink = BooleanArray(size * size) { mask[it] && distance[it] > pocketCut }
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
            val toCore = distanceAt(inner, i, coreColour)
            val toSquare = distanceAt(inner, i, background)
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
                var best = 0
                var bestDistance = Double.MAX_VALUE
                for (k in fills.indices) {
                    val d = distanceAt(inner, i, fills[k])
                    if (d < bestDistance) { bestDistance = d; best = k }
                }
                classes[best][i] = true
            }
        } else {
            var count = 0
            for (i in ink.indices) if (ink[i]) count++
            if (count == 0) return null
            val values = ByteArray(count)
            var at = 0
            for (i in ink.indices) if (ink[i]) values[at++] = gray[i].toInt().toByte()
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
            val colour = DoubleArray(3) { ch -> weightedMedianOf(tone, weights, inner, 3, ch) }
            val luminance = weightedMedianOf(tone, weights, gray, 1, 0)
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
        val band = max(1, pyRound(n * Pieces.HOVER_BAND))
        val edge = max(1, pyRound(n * Pieces.HOVER_EDGE))
        if (2 * band >= n) return false  // a patch too small to have an outer band
        // The four sides, each read as (row, column) of a rotated view.
        fun at(sideIndex: Int, k: Int, along: Int): Int = when (sideIndex) {
            0 -> k * width + along                    // top
            1 -> (n - 1 - k) * width + along          // bottom
            2 -> along * width + k                    // left
            else -> along * width + (width - 1 - k)   // right
        }
        // An outline is a step between one row and the next, and it runs through both ends of
        // the side; a tint that fades toward the edge is a ramp of a few levels a row, and a
        // piece's base or a label reaches one end at most. So the step is read at each end of
        // the side, and the smaller counts. Read first: most cells have no step and are done
        // here (every cell of every frame comes through; desktop a3479bd).
        val steps = DoubleArray(4)
        val reach = max(2, pyRound(n * Pieces.HOVER_CORNER))
        for (sideIndex in 0 until 4) {
            var worst = Double.MAX_VALUE
            for ((lo, hi) in listOf(band to band + reach, n - band - reach to n - band)) {
                val lines = ArrayList<DoubleArray>()
                val from = max(0, lo)
                val until = min(n, hi)
                val bs = FloatArray(max(0, until - from))
                val gs = FloatArray(bs.size)
                val rs = FloatArray(bs.size)
                for (k in 0 until min(n, 2 * band + 1)) {
                    var count = 0
                    for (along in from until until) {
                        val p = at(sideIndex, k, along) * 3
                        bs[count] = pixels[p]; gs[count] = pixels[p + 1]; rs[count] = pixels[p + 2]
                        count++
                    }
                    if (count == 0) continue
                    lines.add(doubleArrayOf(medianOf(bs, count), medianOf(gs, count), medianOf(rs, count)))
                }
                var biggest = 0.0
                for (k in 0 until lines.size - 1) {
                    biggest = max(biggest, distanceOf(lines[k + 1], lines[k]))
                }
                worst = min(worst, biggest)
            }
            steps[sideIndex] = if (worst == Double.MAX_VALUE) 0.0 else worst
        }
        steps.sort()
        if (steps[1] < Pieces.HOVER_STEP) return false  // no step on three sides: a ramp
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
        val strong = DoubleArray(4)
        val weak = DoubleArray(4)
        val medians = DoubleArray(4)
        for (sideIndex in 0 until 4) {
            fun at(k: Int, along: Int): Int = at(sideIndex, k, along)
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
            val edgeValues = FloatArray(edge * n)
            var edgeCount = 0
            for (k in 0 until edge) for (along in 0 until n) edgeValues[edgeCount++] = distance[at(k, along)]
            medians[sideIndex] = medianOf(edgeValues, edgeCount)
        }
        strong.sort(); weak.sort(); medians.sort()
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
        var any = false
        for (i in 0 until size * size) {
            if (distance[i] > Pieces.OVERLAY_DISTANCE && chroma[i] > Pieces.OVERLAY_CHROMA && blind[i]) {
                any = true
                break
            }
        }
        if (!any) return 0.0
        val colours = ArrayList<DoubleArray>()
        colours.add(doubleArrayOf(0.0, 0.0, 0.0))
        colours.add(doubleArrayOf(255.0, 255.0, 255.0))
        colours.addAll(palette)
        // The segments laid out once, each as its start, its direction and its squared length
        // (1 when it has none), so the pixels are measured against them without a single
        // allocation: with the set's sixteen colours there are 171 of them.
        val count = colours.size + colours.size * (colours.size - 1) / 2
        val starts = DoubleArray(count * 3)
        val directions = DoubleArray(count * 3)
        val lengths = DoubleArray(count)
        var s = 0
        fun segment(start: DoubleArray, end: DoubleArray) {
            var length = 0.0
            for (c in 0 until 3) {
                starts[s * 3 + c] = start[c]
                val d = end[c] - start[c]
                directions[s * 3 + c] = d
                length += d * d
            }
            lengths[s] = if (length == 0.0) 1.0 else length
            s++
        }
        for (colour in colours) segment(background, colour)
        for (i in colours.indices) for (j in i + 1 until colours.size) segment(colours[i], colours[j])
        var drawn = 0
        for (index in 0 until size * size) {
            if (!(distance[index] > Pieces.OVERLAY_DISTANCE && chroma[index] > Pieces.OVERLAY_CHROMA && blind[index])) {
                continue
            }
            val b = inner[index * 3].toDouble()
            val g = inner[index * 3 + 1].toDouble()
            val r = inner[index * 3 + 2].toDouble()
            var nearest = Double.MAX_VALUE
            for (k in 0 until count) {
                nearest = min(nearest, offLine(b, g, r, starts, directions, lengths, k))
                if (nearest <= Pieces.OVERLAY_OFF_LINE) break
            }
            if (nearest > Pieces.OVERLAY_OFF_LINE) drawn++
        }
        return drawn.toDouble() / (size * size)
    }

    /**
     * How far the pixel (b, g, r) lies from segment [k] of colours: its start, direction and
     * squared length laid out in the arrays (see [overlayFraction]).
     */
    private fun offLine(
        b: Double,
        g: Double,
        r: Double,
        starts: DoubleArray,
        directions: DoubleArray,
        lengths: DoubleArray,
        k: Int,
    ): Double {
        val at = k * 3
        val d0 = directions[at]
        val d1 = directions[at + 1]
        val d2 = directions[at + 2]
        val s0 = starts[at]
        val s1 = starts[at + 1]
        val s2 = starts[at + 2]
        var dot = 0.0
        dot += (b - s0) * d0
        dot += (g - s1) * d1
        dot += (r - s2) * d2
        val along = (dot / lengths[k]).coerceIn(0.0, 1.0)
        val e0 = b - (s0 + along * d0)
        val e1 = g - (s1 + along * d1)
        val e2 = r - (s2 + along * d2)
        var sum = 0.0
        sum += e0 * e0
        sum += e1 * e1
        sum += e2 * e2
        return sqrt(sum)
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

    /** The distance of inner pixel [i] (BGR floats, row after row) from [colour]. */
    private fun distanceAt(inner: FloatArray, i: Int, colour: DoubleArray): Double {
        val db = inner[i * 3].toDouble() - colour[0]
        val dg = inner[i * 3 + 1].toDouble() - colour[1]
        val dr = inner[i * 3 + 2].toDouble() - colour[2]
        var sum = 0.0
        sum += db * db
        sum += dg * dg
        sum += dr * dr
        return sqrt(sum)
    }

    /** numpy's median of the first [count] values: the middle one, or the two middle ones' mean. */
    private fun medianOf(values: FloatArray, count: Int = values.size): Double {
        if (count == 0) return 0.0
        val sorted = values.copyOf(count)
        sorted.sort()
        return if (count % 2 == 1) sorted[count / 2].toDouble()
        else (sorted[count / 2 - 1].toDouble() + sorted[count / 2].toDouble()) / 2.0
    }

    /**
     * `pieces._weighted_median` of `values[i * stride + offset]` over the pixels [mask] marks,
     * with float32 weights: numpy's cumulative sum of them is a float32 one, run in order of
     * value, and the median is where it first reaches half its end. The values are pixel
     * levels, never negative, so their bits sort as they do; packed above the pixel's index,
     * one sort of longs orders them by value and then by place, as a stable sort would.
     */
    private fun weightedMedianOf(
        mask: BooleanArray,
        weights: FloatArray,
        values: FloatArray,
        stride: Int,
        offset: Int,
    ): Double {
        var count = 0
        for (on in mask) if (on) count++
        if (count == 0) return 0.0
        val keys = LongArray(count)
        var k = 0
        for (i in mask.indices) {
            if (!mask[i]) continue
            val bits = java.lang.Float.floatToIntBits(values[i * stride + offset]).toLong()
            keys[k++] = (bits shl 32) or i.toLong()
        }
        keys.sort()
        val cumulative = FloatArray(count)
        var running = 0f
        for (j in 0 until count) {
            running += weights[(keys[j] and 0xffffffffL).toInt()]
            cumulative[j] = running
        }
        val half = cumulative[count - 1] / 2
        var pick = count - 1
        for (j in 0 until count) {
            if (cumulative[j] >= half) {
                pick = j
                break
            }
        }
        return java.lang.Float.intBitsToFloat((keys[pick] ushr 32).toInt()).toDouble()
    }

    /**
     * Otsu's threshold over the ink's grey levels, by OpenCV's own
     * (`cv2.threshold(values, 0, 255, THRESH_BINARY + THRESH_OTSU)`).
     */
    private fun otsu(values: ByteArray): Float {
        val column = Mat(values.size, 1, CvType.CV_8U)
        column.put(0, 0, values)
        val out = Mat()
        val split = Imgproc.threshold(column, out, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)
        column.release()
        out.release()
        return split.toFloat()
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
