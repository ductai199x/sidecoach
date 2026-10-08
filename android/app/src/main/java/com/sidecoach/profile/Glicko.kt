package com.sidecoach.profile

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * The puzzle rating as Lichess computes it: Glicko-2, line for line (Tai, 2026-10-08: "Ofc we
 * want to stick to how lichess does it and we want to match exactly"). `sidecoach/profile/glicko.py`.
 *
 * A port of Lichess's own code, read at lila f5b261e3 and the scalachess it pins (18.0.0,
 * 2ab0a403):
 *
 * - the calculator: scalachess `rating/src/main/scala/glicko/impl/RatingCalculator.scala`
 *   (Glickman's steps 1-8, tau 0.75, the Illinois iteration to 1e-6) and `glicko/model.scala`
 *   (the Glicko triple, `average`, provisional at a deviation of 110, clueless at 230);
 * - Lichess's bounds: lila `modules/rating/src/main/Glicko.scala` (a new player 1500 / 500 /
 *   0.09; the rating at least 400; the deviation 45 to 500; the volatility at most 0.1; at most
 *   700 points a game; the deviation grows 0.21436 rating periods a day while the player is
 *   away) and `Perf.scala` (`addOrReset`, the deviation as of now on every read);
 * - a puzzle attempt: lila `modules/puzzle/src/main/PuzzleFinisher.scala`, a game the player
 *   plays as White against the puzzle as Black, won when solved, whose result is then weighed by
 *   the puzzle's theme (`ponder.player`) -- a themed puzzle moves the rating less than a mixed
 *   one, an obvious theme (a mate in one) least.
 *
 * Ratings are doubles and shown truncated (`intRating`, Scala's `toInt`); the change shown after
 * a puzzle is the difference of the two shown ratings (lila `JsonView.scala`). The theme weights
 * are Floats, as on Lichess's JVM.
 *
 * Pure functions over values; nothing here reads or writes the profile.
 */
object Glicko {

    // ---- scalachess glicko/model.scala --------------------------------------------------------
    const val TAU = 0.75
    const val PROVISIONAL_DEVIATION = 110
    const val CLUELESS_DEVIATION = 230

    // ---- scalachess glicko/impl/RatingCalculator.scala ----------------------------------------
    const val MULTIPLIER = 173.7178
    const val DEFAULT_RATING = 1500.0
    const val CONVERGENCE_TOLERANCE = 0.000001
    const val ITERATION_MAX = 1000
    const val DAYS_PER_MILLI = 1.0 / (1000 * 60 * 60 * 24)

    // ---- lila modules/rating/src/main/Glicko.scala --------------------------------------------
    const val MIN_RATING = 400
    const val MAX_RATING = 4000
    const val MIN_DEVIATION = 45
    const val MAX_DEVIATION = 500.0
    const val MAX_VOLATILITY = 0.1
    const val DEFAULT_VOLATILITY = 0.09
    const val MAX_RATING_DELTA = 700

    /** "chosen so a typical player's RD goes from 60 -> 110 in 1 year" */
    const val PERIODS_PER_DAY = 0.21436

    // ---- lila modules/puzzle/src/main/PuzzleFinisher.scala (ponder) ---------------------------
    /** Themes that do not hint at the solution. */
    val NON_HINTING_THEMES: Set<String> = setOf(
        "opening",
        "middlegame",
        "endgame",
        "rookEndgame",
        "bishopEndgame",
        "pawnEndgame",
        "knightEndgame",
        "queenEndgame",
        "queenRookEndgame",
        "master",
        "masterVsMaster",
        "superGM",
    )

    /** PuzzleTheme.allMates: the visible themes whose key ends in "Mate" (PuzzleTheme.scala). */
    val MATE_THEMES: Set<String> = setOf(
        "anastasiaMate",
        "arabianMate",
        "backRankMate",
        "balestraMate",
        "blindSwineMate",
        "bodenMate",
        "cornerMate",
        "doubleBishopMate",
        "dovetailMate",
        "epauletteMate",
        "hookMate",
        "killBoxMate",
        "pillsburysMate",
        "morphysMate",
        "operaMate",
        "swallowstailMate",
        "triangleMate",
        "vukovicMate",
        "smotheredMate",
    )

    /** Themes that make the solution very obvious. */
    val OBVIOUS_THEMES: Set<String> =
        setOf("enPassant", "attackingF2F7", "doubleCheck", "mateIn1", "castling") + MATE_THEMES

    const val MIX = "mix"

    data class Rating(val rating: Double, val deviation: Double, val volatility: Double) {
        /** The rating as Lichess shows it: truncated (Scala's Double.toInt). */
        val intRating: Int get() = rating.toInt()
        val provisional: Boolean get() = deviation >= PROVISIONAL_DEVIATION
        val clueless: Boolean get() = deviation >= CLUELESS_DEVIATION

        /** "1500?" while provisional, as Lichess writes a rating. */
        fun display(): String = "$intRating${if (provisional) "?" else ""}"

        /** [weight] of the way from this one to [other]. */
        fun average(other: Rating, weight: Float = 0.5f): Rating =
            if (weight >= 1) other
            else if (weight <= 0) this
            else Rating(
                rating = rating * (1 - weight) + other.rating * weight,
                deviation = deviation * (1 - weight) + other.deviation * weight,
                volatility = volatility * (1 - weight) + other.volatility * weight,
            )

        /** lila GlickoExt.cap: the rating at least 400, the deviation 45 to 500, the volatility
         * at most 0.1. */
        fun cap(): Rating = Rating(
            rating = rating.coerceAtLeast(MIN_RATING.toDouble()),
            deviation = deviation.coerceAtLeast(MIN_DEVIATION.toDouble()).coerceAtMost(MAX_DEVIATION),
            volatility = volatility.coerceAtMost(MAX_VOLATILITY),
        )

        fun sanityCheck(): Boolean =
            rating > 0 && rating < MAX_RATING &&
                deviation > 0 && deviation < 1000 &&
                volatility > 0 && volatility < MAX_VOLATILITY * 2
    }

    val DEFAULT = Rating(DEFAULT_RATING, MAX_DEVIATION, DEFAULT_VOLATILITY)

    /**
     * A player's puzzle rating as Lichess keeps it (lila.core.perf.Perf): the Glicko triple, how
     * many rated puzzles, and when the last one was (epoch milliseconds, null before any).
     */
    data class Perf(val glicko: Rating = DEFAULT, val nb: Int = 0, val latest: Long? = null)

    // ---- the calculator (RatingCalculator.scala) ----------------------------------------------

    private fun toGlicko2(rating: Double): Double = (rating - DEFAULT_RATING) / MULTIPLIER

    private fun fromGlicko2(rating: Double): Double = (rating * MULTIPLIER) + DEFAULT_RATING

    private fun g(deviation: Double): Double =
        1.0 / sqrt(1.0 + (3.0 * (deviation * deviation) / (Math.PI * Math.PI)))

    private fun e(player: Double, opponent: Double, opponentDeviation: Double): Double =
        1.0 / (1.0 + exp(-1.0 * g(opponentDeviation) * (player - opponent)))

    private fun f(x: Double, delta: Double, phi: Double, v: Double, a: Double, tau: Double): Double {
        val ex = exp(x)
        val denom = phi * phi + v + ex
        return (ex * (delta * delta - phi * phi - v - ex) / (2.0 * (denom * denom))) -
            ((x - a) / (tau * tau))
    }

    /** Step 6, also the deviation of a player away for [elapsedPeriods]. */
    private fun newRd(phi: Double, sigma: Double, elapsedPeriods: Double): Double =
        sqrt(phi * phi + elapsedPeriods * (sigma * sigma))

    /**
     * The player after one rating period of [results] (each an opponent as they stood before it
     * and the score, 1 a win, 0.5 a draw, 0 a loss): calculateNewRating.
     */
    fun rate(
        player: Rating,
        results: List<Pair<Rating, Double>>,
        elapsedPeriods: Double = 1.0,
        tau: Double = TAU,
    ): Rating {
        val phi = player.deviation / MULTIPLIER
        val sigma = player.volatility
        val mu = toGlicko2(player.rating)
        val a = ln(sigma * sigma)
        var vSum = 0.0
        var outcome = 0.0
        for ((opponent, score) in results) {
            val oPhi = opponent.deviation / MULTIPLIER
            val expected = e(mu, toGlicko2(opponent.rating), oPhi)
            val gv = g(oPhi)
            vSum += (gv * gv) * expected * (1.0 - expected)
            outcome += gv * (score - expected)
        }
        val v = 1 / vSum
        val delta = v * outcome

        // step 5.2
        var bigA = a
        var bigB: Double
        if (delta * delta > phi * phi + v) {
            bigB = ln(delta * delta - phi * phi - v)
        } else {
            var k = 1.0
            bigB = a - (k * abs(tau))
            while (f(bigB, delta, phi, v, a, tau) < 0) {
                k = k + 1
                bigB = a - (k * abs(tau))
            }
        }
        // step 5.3
        var fA = f(bigA, delta, phi, v, a, tau)
        var fB = f(bigB, delta, phi, v, a, tau)
        // step 5.4
        var iterations = 0
        while (abs(bigB - bigA) > CONVERGENCE_TOLERANCE && iterations < ITERATION_MAX) {
            iterations += 1
            val bigC = bigA + (((bigA - bigB) * fA) / (fB - fA))
            val fC = f(bigC, delta, phi, v, a, tau)
            if (fC * fB <= 0) {
                bigA = bigB
                fA = fB
            } else {
                fA = fA / 2.0
            }
            bigB = bigC
            fB = fC
        }
        if (iterations == ITERATION_MAX) throw RuntimeException("Convergence fail")
        val newSigma = exp(bigA / 2.0)
        // steps 6-7
        val phiStar = newRd(phi, newSigma, elapsedPeriods)
        val newPhi = 1.0 / sqrt((1.0 / (phiStar * phiStar)) + (1.0 / v))
        return Rating(
            rating = fromGlicko2(mu + (newPhi * newPhi) * outcome),
            deviation = newPhi * MULTIPLIER,
            volatility = newSigma,
        )
    }

    /**
     * Both players after one game (GlickoCalculator.computeGame, no colour advantage): each rated
     * against the other as they stood before it.
     */
    fun computeGame(
        white: Rating,
        black: Rating,
        whiteScore: Double,
        skipDeviationIncrease: Boolean = false,
    ): Pair<Rating, Rating> {
        val elapsed = if (skipDeviationIncrease) 0.0 else 1.0
        return Pair(
            rate(white, listOf(Pair(black, whiteScore)), elapsedPeriods = elapsed),
            rate(black, listOf(Pair(white, 1.0 - whiteScore)), elapsedPeriods = elapsed),
        )
    }

    // ---- the deviation over time (Glicko.liveDeviation, previewDeviation) ---------------------

    /**
     * The deviation as of [nowMs]: grown by the rating periods since the last rated puzzle (shrunk
     * back when [reverse], as Lichess stores it), within 45 to 500.
     */
    fun liveDeviation(perf: Perf, nowMs: Long, reverse: Boolean = false): Double {
        val capped = perf.glicko.cap()
        var elapsed = 0.0
        if (perf.latest != null) elapsed = (nowMs - perf.latest) * (PERIODS_PER_DAY * DAYS_PER_MILLI)
        if (reverse) elapsed = -elapsed
        val rd = newRd(capped.deviation / MULTIPLIER, capped.volatility, elapsed) * MULTIPLIER
        return rd.coerceAtLeast(MIN_DEVIATION.toDouble()).coerceAtMost(MAX_DEVIATION)
    }

    /** The perf as Lichess reads it from its store: the deviation as of now. */
    fun read(perf: Perf, nowMs: Long): Perf =
        perf.copy(glicko = perf.glicko.copy(deviation = liveDeviation(perf, nowMs)))

    /** The perf as Lichess writes it to its store: the growth since the last puzzle taken out. */
    fun stored(perf: Perf, nowMs: Long): Perf =
        perf.copy(glicko = perf.glicko.copy(deviation = liveDeviation(perf, nowMs, reverse = true)))

    // ---- a puzzle attempt (PuzzleFinisher.scala) ----------------------------------------------

    /**
     * ponder.weightOf: how much of the computed change a puzzle of this angle keeps. [theme] is
     * the angle's theme, null for an angle without one (an opening).
     */
    fun themeWeight(theme: String?, win: Boolean): Float =
        if (theme == null || theme == MIX) 1f
        else if (theme in OBVIOUS_THEMES) (if (win) 0.1f else 0.4f)
        else if (theme !in NON_HINTING_THEMES) (if (win) 0.2f else 0.7f)
        else if (win) 0.7f
        else 0.8f

    /** ponder.player's weight: the theme's, less for a provisional puzzle, at least 0.1. */
    fun playerWeight(theme: String?, win: Boolean, puzzle: Rating): Float {
        val provisional = if (puzzle.provisional) (if (win) -0.2f else -0.7f) else 0f
        return (themeWeight(theme, win) + provisional).coerceAtLeast(0.1f)
    }

    /**
     * Perf.addOrReset: at most 700 points either way; a result that fails the sanity check starts
     * over from the default; then capped, counted, dated.
     */
    fun addOrReset(perf: Perf, player: Rating, nowMs: Long): Perf {
        val new = player.copy(
            rating = player.rating
                .coerceAtMost(perf.glicko.rating + MAX_RATING_DELTA)
                .coerceAtLeast(perf.glicko.rating - MAX_RATING_DELTA),
        )
        val glicko = if (new.sanityCheck()) new else DEFAULT
        return Perf(glicko = glicko.cap(), nb = perf.nb + 1, latest = nowMs)
    }

    /**
     * The player's puzzle perf after a rated first attempt at [puzzle] (its own Glicko, as
     * Lichess holds it), [perf] as read now (see [read]): PuzzleFinisher.apply's rated branch.
     */
    fun finish(perf: Perf, puzzle: Rating, win: Boolean, nowMs: Long, theme: String? = null): Perf {
        val (white, _) = computeGame(perf.glicko.cap(), puzzle.cap(), if (win) 1.0 else 0.0)
        val after = addOrReset(perf, white, nowMs)
        val weight = playerWeight(theme, win, puzzle)
        return after.copy(glicko = perf.glicko.average(after.glicko, weight))
    }

    /** The change Lichess shows after a puzzle: the shown ratings' difference. */
    fun ratingDiff(before: Perf, after: Perf): Int = after.glicko.intRating - before.glicko.intRating

    /** The change as Lichess writes it beside the rating: "+12", "−9" (a minus sign), "" for
     * none (ui/puzzle/src/view/side.ts). */
    fun diffText(diff: Int): String =
        if (diff > 0) "+$diff" else if (diff < 0) "−${-diff}" else ""
}
