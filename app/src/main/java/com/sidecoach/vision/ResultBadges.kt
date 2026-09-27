package com.sidecoach.vision

import org.opencv.core.Mat
import kotlin.math.ceil
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * The site's result badges: a disc on each king when the game ends.
 * `sidecoach/vision/result_badges.py`, ported.
 *
 * Chess.com marks the end of a game on the board itself, before and under its dialog: a
 * green disc with a white crown on the winner's king, a red disc on the loser's (the fallen
 * king, a flag or a clock, in the loser's colour). The disc is about a third of a square
 * across and sits just inside the square's top-right corner as drawn on screen. Read at the
 * two kings' squares, the pair says
 * who won without reading a word of the dialog; what the pipeline already knows -- a mate on
 * the board, a clock at zero -- says why.
 *
 * A badge is a disc, not a colour. The winner's green is a cousin of the default theme's dark
 * squares, a check is a red glow, a highlight is a whole square: so the reader looks for a
 * disc near the corner -- a rim of one colour with a ring of other colours around it, and not
 * the colour of the square it sits on -- and only then asks which colour.
 */
enum class Badge { WIN, LOSS }

/**
 * One frame's pixels, read out of the Mat once. The desktop indexes the numpy array
 * directly; a Mat read per sample would cost more than the search.
 */
class FramePixels(val bytes: ByteArray, val width: Int, val height: Int) {

    fun blue(y: Int, x: Int): Int = bytes[(y * width + x) * 3].toInt() and 0xFF
    fun green(y: Int, x: Int): Int = bytes[(y * width + x) * 3 + 1].toInt() and 0xFF
    fun red(y: Int, x: Int): Int = bytes[(y * width + x) * 3 + 2].toInt() and 0xFF

    companion object {
        fun of(frame: Mat): FramePixels {
            val bytes = ByteArray(frame.rows() * frame.cols() * frame.channels())
            frame.get(0, 0, bytes)
            return FramePixels(bytes, frame.cols(), frame.rows())
        }
    }
}

object ResultBadges {

    /**
     * The disc's radius in squares, and where its centre may lie relative to the square's
     * top-right corner: normally 0.04 in and 0.08 down, up to 0.21 in or down when pushed off
     * the board's edge.
     */
    const val RADIUS = 0.17
    val SEARCH_X = -0.27 to 0.03
    val SEARCH_Y = -0.03 to 0.27

    /** The search is coarse then fine; rim and ring are sampled at most this many pixels. */
    const val SEARCH_STEP = 0.03
    const val SAMPLES = 240

    /** Fewer badge-coloured pixels than this share of a square's area, and there is no disc. */
    const val PRESENCE = 0.02

    /** The rim sampled inside the disc (outside the icon at its centre) and the ring outside. */
    val RIM = 0.08 to 0.135
    val RING = 0.21 to 0.28

    /**
     * A pixel this close to the rim's median colour, per channel, is the disc's colour. The
     * icon takes up to half the rim; the ring outside a disc carries next to none of its
     * colour, and that is the test a uniform square, a glow or an arrow's shaft fails.
     */
    const val TOLERANCE = 22
    const val RIM_SHARE = 0.45
    const val RING_SHARE = 0.15

    /** A disc closer than this to the colour of the square it sits on is the square. */
    const val DISTINCT = 25

    /** 1 win, 2 loss, 0 none, for one BGR colour. */
    fun classOf(b: Int, g: Int, r: Int): Int {
        if (g >= 120 && g - r >= 25 && g - b >= 60) return 1
        if (r >= 150 && r - g >= 70 && r - b >= 70) return 2
        // The draw badge is unseen so far. A grey disc was tried for it and read the rounded
        // end of a white label ([resultOf]).
        return 0
    }

    private fun classOf(colour: DoubleArray): Int =
        classOf(colour[0].toInt(), colour[1].toInt(), colour[2].toInt())

    fun classify(colour: DoubleArray): Badge? = when (classOf(colour)) {
        1 -> Badge.WIN
        2 -> Badge.LOSS
        else -> null
    }

    /**
     * (dy, dx) offsets of the pixels between `inner` and `outer` squares of a centre, thinned
     * evenly to at most [SAMPLES] of them.
     */
    private fun offsets(cell: Double, inner: Double, outer: Double): Array<IntArray> {
        val reach = ceil(outer * cell).toInt()
        val kept = ArrayList<IntArray>()
        for (y in -reach..reach) {
            for (x in -reach..reach) {
                val d = hypot(x.toDouble(), y.toDouble())
                if (d >= inner * cell && d < outer * cell) kept.add(intArrayOf(y, x))
            }
        }
        if (kept.size <= SAMPLES) return kept.toTypedArray()
        val picked = linspaceIndices(kept.size, SAMPLES)
        return Array(SAMPLES) { kept[picked[it]] }
    }

    /**
     * Whether the window a badge could occupy holds enough badge-coloured pixels to be worth
     * a search. Cheap, and it is the whole cost on most frames of a game.
     */
    private fun present(frame: FramePixels, cornerX: Double, cornerY: Double, cell: Double): Boolean {
        val reach = RADIUS + RING.first
        val x0 = (cornerX + (SEARCH_X.first - reach) * cell).toInt()
        val x1 = ceil(cornerX + (SEARCH_X.second + reach) * cell).toInt()
        val y0 = (cornerY + (SEARCH_Y.first - reach) * cell).toInt()
        val y1 = ceil(cornerY + (SEARCH_Y.second + reach) * cell).toInt()
        val xLo = max(0, x0)
        val yLo = max(0, y0)
        val xHi = min(frame.width, max(0, x1))
        val yHi = min(frame.height, max(0, y1))
        if (xHi <= xLo || yHi <= yLo) return false
        var found = 0
        for (y in yLo until yHi) {
            for (x in xLo until xHi) {
                if (classOf(frame.blue(y, x), frame.green(y, x), frame.red(y, x)) != 0) found += 1
            }
        }
        return found >= PRESENCE * cell * cell
    }

    private class Scored(val colour: DoubleArray, val rimShare: Double, val ringShare: Double)

    /**
     * For a candidate centre: the rim's median colour, the share of the rim within tolerance
     * of it, and the share of the ring.
     */
    private fun score(
        frame: FramePixels,
        cy: Double,
        cx: Double,
        rim: Array<IntArray>,
        ring: Array<IntArray>,
    ): Scored {
        val rimPx = Array(rim.size) { IntArray(3) }
        val rimIn = BooleanArray(rim.size)
        gather(frame, cy, cx, rim, rimPx, rimIn)
        val ringPx = Array(ring.size) { IntArray(3) }
        val ringIn = BooleanArray(ring.size)
        gather(frame, cy, cx, ring, ringPx, ringIn)

        val colour = DoubleArray(3) { c -> median(IntArray(rimPx.size) { rimPx[it][c] }) }
        var rimSame = 0
        var rimInside = 0
        for (i in rimPx.indices) {
            if (rimIn[i]) rimInside += 1
            var worst = 0.0
            for (c in 0 until 3) {
                val d = kotlin.math.abs(rimPx[i][c] - colour[c])
                if (d > worst) worst = d
            }
            if (worst <= TOLERANCE && rimIn[i]) rimSame += 1
        }
        var ringSame = 0
        var ringInside = 0
        for (i in ringPx.indices) {
            if (ringIn[i]) ringInside += 1
            var worst = 0.0
            for (c in 0 until 3) {
                val d = kotlin.math.abs(ringPx[i][c] - colour[c])
                if (d > worst) worst = d
            }
            if (worst <= TOLERANCE && ringIn[i]) ringSame += 1
        }
        var rimShare = rimSame.toDouble() / max(1, rimInside)
        val ringShare = ringSame.toDouble() / max(1, ringInside)
        // A rim mostly off the frame is no rim, and a rim of no badge's colour is no badge.
        val usable = rimInside.toDouble() / rimPx.size >= 0.8 && classOf(colour) != 0
        if (!usable) rimShare = 0.0
        return Scored(colour, rimShare, ringShare)
    }

    private fun gather(
        frame: FramePixels,
        cy: Double,
        cx: Double,
        offsets: Array<IntArray>,
        out: Array<IntArray>,
        inside: BooleanArray,
    ) {
        for (i in offsets.indices) {
            val y = Math.rint(cy + offsets[i][0]).toInt()
            val x = Math.rint(cx + offsets[i][1]).toInt()
            inside[i] = y >= 0 && y < frame.height && x >= 0 && x < frame.width
            val yy = y.coerceIn(0, frame.height - 1)
            val xx = x.coerceIn(0, frame.width - 1)
            out[i][0] = frame.blue(yy, xx)
            out[i][1] = frame.green(yy, xx)
            out[i][2] = frame.red(yy, xx)
        }
    }

    private fun median(values: IntArray): Double {
        if (values.isEmpty()) return 0.0
        values.sort()
        val n = values.size
        return if (n % 2 == 1) values[n / 2].toDouble() else (values[n / 2 - 1] + values[n / 2]) / 2.0
    }

    /**
     * The colour of the square whose top-right corner is given, from small patches at its
     * other three corners: what a badge at the fourth must differ from. A highlight tints all
     * four alike, a piece seldom reaches a corner, a check's glow has faded there.
     */
    fun squareColour(frame: FramePixels, cornerX: Double, cornerY: Double, cell: Double): DoubleArray {
        val inset = 0.05 * cell
        val size = max(2, pyRound(0.08 * cell))
        val xLeft = cornerX - cell + inset
        val xRight = cornerX - inset - size
        val yTop = cornerY + inset
        val yBottom = cornerY + cell - inset - size
        val channels = Array(3) { ArrayList<Int>() }
        var any = false
        for ((x, y) in listOf(xLeft to yTop, xLeft to yBottom, xRight to yBottom)) {
            val xi = pyRound(x)
            val yi = pyRound(y)
            val x0 = max(0, xi)
            val y0 = max(0, yi)
            val x1 = min(frame.width, xi + size)
            val y1 = min(frame.height, yi + size)
            if (x1 <= x0 || y1 <= y0) continue
            any = true
            for (py in y0 until y1) {
                for (px in x0 until x1) {
                    channels[0].add(frame.blue(py, px))
                    channels[1].add(frame.green(py, px))
                    channels[2].add(frame.red(py, px))
                }
            }
        }
        if (!any) return doubleArrayOf(0.0, 0.0, 0.0)
        return DoubleArray(3) { median(channels[it].toIntArray()) }
    }

    /**
     * A disc near a square's top-right corner: its rim colour and the shares of the rim and
     * the ring that carry it, or null when nothing disc-shaped is there. The candidate centres
     * are scored coarsely first and then finely around the best.
     */
    fun findDisc(
        frame: FramePixels,
        cornerX: Double,
        cornerY: Double,
        cell: Double,
        background: DoubleArray? = null,
    ): Triple<DoubleArray, Double, Double>? {
        if (cell < 16 || !present(frame, cornerX, cornerY, cell)) return null
        val step = max(1.0, SEARCH_STEP * cell)
        val rim = offsets(cell, RIM.first, RIM.second)
        val ring = offsets(cell, RING.first, RING.second)
        val cxs = arange(cornerX + SEARCH_X.first * cell, cornerX + SEARCH_X.second * cell + 1e-6, 2 * step)
        val cys = arange(cornerY + SEARCH_Y.first * cell, cornerY + SEARCH_Y.second * cell + 1e-6, 2 * step)
        val coarse = ArrayList<DoubleArray>(cys.size * cxs.size)
        for (cy in cys) for (cx in cxs) coarse.add(doubleArrayOf(cy, cx))
        var pick = -1
        var bestValue = Double.NEGATIVE_INFINITY
        var picked: Scored? = null
        for (i in coarse.indices) {
            val scored = score(frame, coarse[i][0], coarse[i][1], rim, ring)
            val value = scored.rimShare - scored.ringShare
            if (value > bestValue) { bestValue = value; pick = i; picked = scored }
        }
        if (picked == null || picked.rimShare == 0.0) return null
        var best: Scored? = null
        var bestFine = Double.NEGATIVE_INFINITY
        for (dy in doubleArrayOf(-step, 0.0, step)) {
            for (dx in doubleArrayOf(-step, 0.0, step)) {
                val scored = score(frame, coarse[pick][0] + dy, coarse[pick][1] + dx, rim, ring)
                val value = scored.rimShare - scored.ringShare
                if (value > bestFine) { bestFine = value; best = scored }
            }
        }
        val disc = best ?: return null
        if (disc.rimShare < RIM_SHARE || disc.ringShare > RING_SHARE) return null
        if (background != null) {
            var worst = 0.0
            for (c in 0 until 3) {
                val d = kotlin.math.abs(disc.colour[c] - background[c])
                if (d > worst) worst = d
            }
            if (worst < DISTINCT) return null
        }
        return Triple(disc.colour, disc.rimShare, disc.ringShare)
    }

    /** `np.arange(start, stop, step)`: values below `stop`, the way numpy counts them. */
    private fun arange(start: Double, stop: Double, step: Double): DoubleArray {
        val n = max(0, ceil((stop - start) / step).toInt())
        return DoubleArray(n) { start + it * step }
    }

    /** The badge on the square whose top-right corner (on screen) is at the given point. */
    fun badgeAt(frame: FramePixels, cornerX: Double, cornerY: Double, cell: Double): Badge? {
        val background = squareColour(frame, cornerX, cornerY, cell)
        val found = findDisc(frame, cornerX, cornerY, cell, background) ?: return null
        return classify(found.first)
    }

    /** The badge on a square of a locked board, if the site has stamped one there. */
    fun badgeOn(
        frame: FramePixels,
        geometry: BoardGeometry,
        square: Int,
        bottom: Pieces.Side,
    ): Badge? {
        val (row, col) = Squares.rowCol(square, bottom.color)
        val cornerX = geometry.region.x + (col + 1) * geometry.cellPx
        val cornerY = geometry.region.y + row * geometry.cellPx
        return badgeAt(frame, cornerX, cornerY, geometry.cellPx)
    }

    /**
     * The result a pair of badges on the two kings says, as a PGN result string, or null
     * (`result_badges.result_of`). Only the pair counts: a win on one king with a loss on the
     * other. One badge alone, or two that disagree, is nothing -- the rim of a glow, an arrow's
     * head, a mark the site draws for some other reason. A draw's badge has never been seen: a
     * draw ends through the dialog's cover or the End game menu, which carries the reason.
     */
    fun resultOf(white: Badge?, black: Badge?): String? {
        if (white == Badge.WIN && black == Badge.LOSS) return "1-0"
        if (white == Badge.LOSS && black == Badge.WIN) return "0-1"
        return null
    }
}
