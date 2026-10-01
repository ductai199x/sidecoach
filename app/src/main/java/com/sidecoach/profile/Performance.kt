package com.sidecoach.profile

import com.sidecoach.analysis.Thresholds
import kotlin.math.exp
import kotlin.math.ln

/**
 * The rating a game was played at, from its moves alone (`profile/performance.py`; the players'
 * ratings are never read: chess.com's Game Rating is criticised for echoing them).
 *
 * Each move falls in a bucket by what it cost (best, a sliver, a little, an inaccuracy, a
 * mistake, a blunder: the coach's own lines), and how likely each bucket is was fitted on real
 * players' games at known ratings (Lichess rapid, 800 to 2300 and up) as a softmax whose logits
 * are quadratic in the rating, one per context -- the opening or after it, and whether the mover
 * was lost, level or won. A side's moves then say how likely each rating on a grid is to have
 * played them, and the estimate is the mean over the grid, weighted by that and a broad prior.
 *
 * The model's scale is Lichess rapid; [chesscom] converts to chess.com's.
 */
object Performance {
    private val DEFAULTS = Thresholds()

    // The loss lines of the buckets, in expected score: below each line is the bucket before it.
    private val EDGES = doubleArrayOf(1e-9, 0.02, DEFAULTS.inaccuracy, DEFAULTS.mistake, DEFAULTS.blunder)
    private val BUCKETS = EDGES.size + 1
    private const val OPENING_PLIES = 20
    private const val LOST = 0.15
    private const val WON = 0.85
    private const val CONTEXTS = 6
    private const val CENTRE = 1500.0
    private const val SPREAD = 500.0
    private val GRID = DoubleArray(521) { 400.0 + 5.0 * it } // np.arange(400.0, 3001.0, 5.0)
    private const val PRIOR_MEAN = 1500.0
    private const val PRIOR_SPREAD = 400.0

    /** Fewer of a side's moves than this say too little to name a rating. */
    const val MIN_MOVES = 8

    /** `performance.json`, the fitted weights: per context, per bucket, the logit's (1, x, x^2). */
    private val WEIGHTS = arrayOf(
        arrayOf(
            doubleArrayOf(0.62478, -0.06286, 0.23979),
            doubleArrayOf(0.38189, -0.13677, 0.3021),
            doubleArrayOf(0.10139, -0.0669, 0.06778),
            doubleArrayOf(-0.19806, -0.02644, -0.10983),
            doubleArrayOf(-0.44801, 0.13579, -0.23877),
            doubleArrayOf(-0.46199, 0.15719, -0.26106),
        ),
        arrayOf(
            doubleArrayOf(1.12945, 0.33595, 0.17336),
            doubleArrayOf(0.45392, 0.37001, 0.10555),
            doubleArrayOf(0.11268, 0.15298, 0.04249),
            doubleArrayOf(0.20694, -0.03467, -0.01443),
            doubleArrayOf(-0.44815, -0.28937, -0.04407),
            doubleArrayOf(-1.45484, -0.53491, -0.2629),
        ),
        arrayOf(
            doubleArrayOf(0.82871, 0.09982, 0.32014),
            doubleArrayOf(-0.01711, -0.12927, 0.04471),
            doubleArrayOf(-0.26021, 0.04679, -0.17632),
            doubleArrayOf(-0.24722, 0.0459, -0.02886),
            doubleArrayOf(-0.18674, -0.01739, -0.14249),
            doubleArrayOf(-0.11743, -0.04585, -0.01718),
        ),
        arrayOf(
            doubleArrayOf(1.80376, -0.11057, 0.37205),
            doubleArrayOf(1.2623, -0.01734, 0.21487),
            doubleArrayOf(0.32082, -0.02912, 0.2792),
            doubleArrayOf(-0.29702, 0.04235, 0.18609),
            doubleArrayOf(-1.48257, 0.07699, -0.41915),
            doubleArrayOf(-1.60729, 0.03771, -0.63306),
        ),
        arrayOf(
            doubleArrayOf(1.06141, 0.15218, 0.09697),
            doubleArrayOf(-0.27258, 0.20642, 0.03653),
            doubleArrayOf(-0.44219, 0.24002, -0.04942),
            doubleArrayOf(0.10864, 0.02351, -5e-05),
            doubleArrayOf(0.15324, -0.13596, -0.03479),
            doubleArrayOf(-0.60865, -0.4863, -0.04936),
        ),
        arrayOf(
            doubleArrayOf(1.65027, 0.09757, 0.11288),
            doubleArrayOf(0.56056, 0.01404, -0.01127),
            doubleArrayOf(-0.34898, 0.02087, 0.04811),
            doubleArrayOf(-0.56342, 0.06939, -0.0331),
            doubleArrayOf(-0.7772, 0.02295, -0.07486),
            doubleArrayOf(-0.52122, -0.22481, -0.04175),
        ),
    )

    /** `bisect_right(EDGES, loss)`. */
    fun bucket(loss: Double): Int = EDGES.count { it <= loss }

    /**
     * The move's context: the opening or after it ([ply], the half-moves before the move), times
     * lost / level / won for the mover ([score], their expected score before it).
     */
    fun context(ply: Int, score: Double): Int {
        val zone = if (score < LOST) 0 else if (score > WON) 2 else 1
        return (if (ply < OPENING_PLIES) 0 else 3) + zone
    }

    private val TABLE: Array<Array<DoubleArray>> by lazy { table(WEIGHTS) }

    /** Each context's and bucket's log chance at every rating of the grid: (C, B, G). `_table`. */
    private fun table(weights: Array<Array<DoubleArray>>): Array<Array<DoubleArray>> =
        Array(CONTEXTS) { c ->
            val logits = Array(BUCKETS) { b ->
                val w = weights[c][b]
                DoubleArray(GRID.size) { g ->
                    val x = (GRID[g] - CENTRE) / SPREAD
                    w[0] * 1.0 + w[1] * x + w[2] * (x * x)
                }
            }
            for (g in GRID.indices) {
                val top = (0 until BUCKETS).maxOf { logits[it][g] }
                val norm = top + ln((0 until BUCKETS).sumOf { exp(logits[it][g] - top) })
                for (b in 0 until BUCKETS) logits[b][g] -= norm
            }
            logits
        }

    /** A side's moves' rating (Lichess rapid), the middle 80% of the weight, and how many moves. */
    data class Estimate(val rating: Int, val low: Int, val high: Int, val moves: Int)

    /** One move as the estimate reads it: its ply, its loss, and the mover's expected score before it. */
    data class Played(val ply: Int, val loss: Double, val score: Double)

    /**
     * The rating one side's moves were played at, with the broad prior. [model]: fitted weights
     * (per context, per bucket, the logit's (1, x, x^2)), when not the shipped model's -- a Stats
     * category's (`categories.json`). `estimate`.
     */
    fun estimate(moves: List<Played>, model: Array<Array<DoubleArray>>? = null): Estimate {
        val logChance = if (model != null) table(model) else TABLE
        val counts = Array(CONTEXTS) { DoubleArray(BUCKETS) }
        for (move in moves) counts[context(move.ply, move.score)][bucket(move.loss)] += 1.0
        val logWeight = DoubleArray(GRID.size) { g ->
            var total = 0.0
            for (c in 0 until CONTEXTS) for (b in 0 until BUCKETS) total += counts[c][b] * logChance[c][b][g]
            val z = (GRID[g] - PRIOR_MEAN) / PRIOR_SPREAD
            total - 0.5 * (z * z)
        }
        val top = logWeight.max()
        val mass = DoubleArray(GRID.size) { exp(logWeight[it] - top) }
        val sum = mass.sum()
        for (g in mass.indices) mass[g] /= sum
        val cumulative = DoubleArray(GRID.size)
        var running = 0.0
        for (g in mass.indices) {
            running += mass[g]
            cumulative[g] = running
        }
        // np.searchsorted(cumulative, q): the first index whose total reaches q.
        fun at(q: Double): Int = GRID[cumulative.indexOfFirst { it >= q }.let { if (it < 0) GRID.size - 1 else it }].toInt()
        return Estimate(
            rating = Math.rint(GRID.indices.sumOf { GRID[it] * mass[it] }).toInt(),
            low = at(0.1),
            high = at(0.9),
            moves = moves.size,
        )
    }

    // Lichess rapid against chess.com rapid for the same players (ChessGoals' survey, July 2026):
    // (chess.com, Lichess), a straight line between the points and the end segments carried on.
    private val CHESSCOM_LICHESS = listOf(
        815 to 1290,
        1000 to 1430,
        1255 to 1615,
        1500 to 1795,
        1995 to 2165,
        2430 to 2615,
    )

    /** A Lichess rapid rating on chess.com's rapid scale, no lower than its floor of 100. */
    fun chesscom(lichess: Double): Int {
        val points = CHESSCOM_LICHESS
        var i = points.indexOfFirst { it.second >= lichess }.let { if (it < 0) points.size else it } // bisect_left
        i = minOf(maxOf(i, 1), points.size - 1)
        val (c0, l0) = points[i - 1]
        val (c1, l1) = points[i]
        return maxOf(100, Math.rint(c0 + (lichess - l0) * (c1 - c0) / (l1 - l0)).toInt())
    }

    /** A chess.com rapid rating on Lichess's rapid scale: [CHESSCOM_LICHESS] read the other way. `lichess`. */
    fun lichess(chesscomRating: Double): Double {
        val points = CHESSCOM_LICHESS
        var i = points.indexOfFirst { it.first >= chesscomRating }.let { if (it < 0) points.size else it } // bisect_left
        i = minOf(maxOf(i, 1), points.size - 1)
        val (c0, l0) = points[i - 1]
        val (c1, l1) = points[i]
        return l0 + (chesscomRating - c0) * (l1 - l0) / (c1 - c0)
    }

    /**
     * The rating on the scale of the site the game was played on: Lichess's for a Lichess game,
     * chess.com's otherwise (a game Sidecoach watched is on chess.com).
     */
    fun forSite(lichess: Double, site: String?): Int =
        if (!site.isNullOrEmpty() && "lichess" in site.lowercase()) Math.rint(lichess).toInt() else chesscom(lichess)
}
