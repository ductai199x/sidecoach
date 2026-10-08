package com.sidecoach.profile

import com.sidecoach.coach.fixture
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.max

/**
 * The puzzle rating against Lichess's own numbers (Glicko.kt; the desktop's tests/test_glicko.py).
 * The calculator cases are scalachess's reference tests with their tolerances (scalachess 18.0.0,
 * GlickoCalculatorTest.scala and impl/RatingCalculatorTest.scala) and Glickman's worked example;
 * the rest replays what the desktop answered (profile/glicko.json, make_glicko_fixtures.py).
 */
class GlickoTest {
    private val white = 1.0
    private val black = 0.0
    private val draw = 0.5

    private class Case(
        val w: DoubleArray, val b: DoubleArray, val score: Double,
        val wantW: DoubleArray, val wantB: DoubleArray, val tol: DoubleArray,
    )

    private fun d(vararg x: Double) = x

    private val cases = listOf(
        // GlickoCalculatorTest
        Case(d(1500.0, 350.0, 0.06), d(1500.0, 350.0, 0.06), white, d(1662.21, 290.2305, 0.0599993), d(1337.79, 290.2305, 0.0599993), d(0.005, 0.00005, 0.0000001)),
        Case(d(1500.0, 350.0, 0.06), d(1500.0, 350.0, 0.06), black, d(1337.79, 290.2305, 0.0599993), d(1662.21, 290.2305, 0.0599993), d(0.005, 0.00005, 0.0000001)),
        Case(d(1500.0, 350.0, 0.06), d(1500.0, 350.0, 0.06), draw, d(1500.0, 290.2305, 0.0599977), d(1500.0, 290.2305, 0.0599977), d(0.005, 0.00005, 0.0000001)),
        Case(d(1400.0, 79.0, 0.06), d(1550.0, 110.0, 0.065), white, d(1422.63, 77.4956, 0.06), d(1506.32, 105.8706, 0.065), d(0.005, 0.00005, 0.00001)),
        Case(d(1400.0, 79.0, 0.06), d(1550.0, 110.0, 0.065), black, d(1389.99, 77.4956, 0.06), d(1568.90, 105.8706, 0.065), d(0.005, 0.00005, 0.00001)),
        Case(d(1400.0, 79.0, 0.06), d(1550.0, 110.0, 0.065), draw, d(1406.31, 77.4956, 0.06), d(1537.61, 105.8706, 0.065), d(0.005, 0.00005, 0.00001)),
        Case(d(1200.0, 60.0, 0.053), d(1850.0, 200.0, 0.062), white, d(1216.73, 59.9006, 0.053013), d(1635.99, 196.9873, 0.062028), d(0.005, 0.00005, 0.000001)),
        Case(d(1200.0, 60.0, 0.053), d(1850.0, 200.0, 0.062), black, d(1199.29, 59.9006, 0.052999), d(1855.42, 196.9873, 0.061999), d(0.005, 0.00005, 0.000001)),
        Case(d(1200.0, 60.0, 0.053), d(1850.0, 200.0, 0.062), draw, d(1208.01, 59.9006, 0.053002), d(1745.71, 196.9873, 0.062006), d(0.005, 0.00005, 0.000001)),
        // RatingCalculatorTest, Lichess's default player among them
        Case(d(1500.0, 500.0, 0.09), d(1500.0, 500.0, 0.09), white, d(1741.0, 396.0, 0.0899983), d(1258.0, 396.0, 0.0899983), d(1.0, 1.0, 0.0000001)),
        Case(d(1500.0, 500.0, 0.09), d(1500.0, 500.0, 0.09), black, d(1258.0, 396.0, 0.0899983), d(1741.0, 396.0, 0.0899983), d(1.0, 1.0, 0.0000001)),
        Case(d(1500.0, 500.0, 0.09), d(1500.0, 500.0, 0.09), draw, d(1500.0, 396.0, 0.0899954), d(1500.0, 396.0, 0.0899954), d(1.0, 1.0, 0.0000001)),
        Case(d(1500.0, 80.0, 0.06), d(1500.0, 80.0, 0.06), white, d(1517.0, 78.0, 0.06), d(1482.0, 78.0, 0.06), d(1.0, 1.0, 0.00001)),
        Case(d(1500.0, 80.0, 0.06), d(1500.0, 80.0, 0.06), black, d(1482.0, 78.0, 0.06), d(1517.0, 78.0, 0.06), d(1.0, 1.0, 0.00001)),
        Case(d(1500.0, 80.0, 0.06), d(1500.0, 80.0, 0.06), draw, d(1500.0, 78.0, 0.06), d(1500.0, 78.0, 0.06), d(1.0, 1.0, 0.00001)),
        Case(d(1400.0, 79.0, 0.06), d(1550.0, 110.0, 0.065), white, d(1422.0, 77.0, 0.06), d(1506.0, 105.0, 0.065), d(1.0, 1.0, 0.00001)),
        Case(d(1400.0, 79.0, 0.06), d(1550.0, 110.0, 0.065), black, d(1389.0, 78.0, 0.06), d(1568.0, 105.0, 0.065), d(1.0, 1.0, 0.00001)),
        Case(d(1400.0, 79.0, 0.06), d(1550.0, 110.0, 0.065), draw, d(1406.0, 78.0, 0.06), d(1537.0, 105.87, 0.065), d(1.0, 1.0, 0.00001)),
        Case(d(1200.0, 60.0, 0.053), d(1850.0, 200.0, 0.062), white, d(1216.7, 59.9, 0.053013), d(1636.0, 196.9, 0.062028), d(0.1, 0.1, 0.000001)),
        Case(d(1200.0, 60.0, 0.053), d(1850.0, 200.0, 0.062), black, d(1199.3, 59.9, 0.052999), d(1855.4, 196.9, 0.061999), d(0.1, 0.1, 0.000001)),
        Case(d(1200.0, 60.0, 0.053), d(1850.0, 200.0, 0.062), draw, d(1208.0, 59.90056, 0.053002), d(1745.7, 196.98729, 0.062006), d(0.1, 0.1, 0.000001)),
    )

    private fun rating(x: DoubleArray) = Glicko.Rating(x[0], x[1], x[2])

    private fun close(got: Glicko.Rating, want: DoubleArray, tol: DoubleArray, what: String) {
        assertTrue("$what rating $got vs ${want.toList()}", abs(got.rating - want[0]) <= tol[0])
        assertTrue("$what deviation $got vs ${want.toList()}", abs(got.deviation - want[1]) <= tol[1])
        assertTrue("$what volatility $got vs ${want.toList()}", abs(got.volatility - want[2]) <= tol[2])
    }

    @Test
    fun aGameRatesBothPlayersAsScalachessDoes() {
        for ((i, c) in cases.withIndex()) {
            val (w, b) = Glicko.computeGame(rating(c.w), rating(c.b), c.score, skipDeviationIncrease = true)
            close(w, c.wantW, c.tol, "case $i white")
            close(b, c.wantB, c.tol, "case $i black")
        }
    }

    @Test
    fun glickmansWorkedExample() {
        val player = Glicko.rate(
            Glicko.Rating(1500.0, 200.0, 0.06),
            listOf(
                Pair(Glicko.Rating(1400.0, 30.0, 0.06), 1.0),
                Pair(Glicko.Rating(1550.0, 100.0, 0.06), 0.0),
                Pair(Glicko.Rating(1700.0, 300.0, 0.06), 0.0),
            ),
            tau = 0.5,
        )
        close(player, d(1464.06, 151.52, 0.05999), d(0.01, 0.01, 0.00001), "glickman")
    }

    @Test
    fun theDeviationGrowsFrom60To110InAYearAway() {
        val perf = Glicko.Perf(Glicko.Rating(1500.0, 60.0, 0.06), nb = 100, latest = 0L)
        assertEquals(110.0, Glicko.liveDeviation(perf, 365L * 24 * 60 * 60 * 1000), 0.5)
    }

    // ---- parity with the desktop ----------------------------------------------------------------

    private fun same(kotlin: Double, python: Double): Boolean =
        abs(kotlin - python) <= 1e-9 * max(1.0, abs(python))

    private fun g(a: JSONArray) = Glicko.Rating(a.getDouble(0), a.getDouble(1), a.getDouble(2))

    private fun sameRating(k: Glicko.Rating, p: JSONArray): Boolean =
        same(k.rating, p.getDouble(0)) && same(k.deviation, p.getDouble(1)) && same(k.volatility, p.getDouble(2))

    private fun perf(o: JSONObject) = Glicko.Perf(
        g(o.getJSONArray("glicko")), o.getInt("nb"), if (o.isNull("latest")) null else o.getLong("latest"),
    )

    private fun theme(o: JSONObject): String? = if (o.isNull("theme")) null else o.getString("theme")

    private val data: JSONObject by lazy { fixture("profile/glicko.json") }

    @Test
    fun gamesAnswerAsTheDesktop() {
        val games = data.getJSONArray("games")
        val failures = mutableListOf<String>()
        for (i in 0 until games.length()) {
            val c = games.getJSONObject(i)
            val (w, b) = Glicko.computeGame(
                g(c.getJSONArray("white")), g(c.getJSONArray("black")), c.getDouble("score"), c.getBoolean("skip"),
            )
            val out = c.getJSONArray("out")
            if (!sameRating(w, out.getJSONArray(0)) || !sameRating(b, out.getJSONArray(1))) {
                failures.add("game $i: kotlin $w $b, python $out")
            }
        }
        assertEquals(emptyList<String>(), failures)
        assertEquals(300, games.length())
    }

    @Test
    fun puzzleAttemptsAnswerAsTheDesktop() {
        val finishes = data.getJSONArray("finishes")
        val failures = mutableListOf<String>()
        for (i in 0 until finishes.length()) {
            val c = finishes.getJSONObject(i)
            val now = c.getLong("now")
            val before = Glicko.read(perf(c.getJSONObject("stored")), now)
            val after = Glicko.finish(before, g(c.getJSONArray("puzzle")), c.getBoolean("win"), now, theme(c))
            val want = c.getJSONObject("after")
            val diff = Glicko.ratingDiff(before, after)
            val ok = sameRating(before.glicko, c.getJSONObject("before").getJSONArray("glicko")) &&
                sameRating(after.glicko, want.getJSONArray("glicko")) &&
                after.nb == want.getInt("nb") && after.latest == want.getLong("latest") &&
                diff == c.getInt("diff") && Glicko.diffText(diff) == c.getString("diff_text") &&
                after.glicko.display() == c.getString("display")
            if (!ok) failures.add("finish $i: kotlin $before -> $after ($diff), python $c")
        }
        assertEquals(emptyList<String>(), failures)
        assertEquals(400, finishes.length())
    }

    @Test
    fun deviationsAndWeightsAnswerAsTheDesktop() {
        val failures = mutableListOf<String>()
        val live = data.getJSONArray("live")
        for (i in 0 until live.length()) {
            val c = live.getJSONObject(i)
            val p = perf(c.getJSONObject("perf"))
            val now = c.getLong("now")
            val forward = Glicko.liveDeviation(p, now)
            val back = Glicko.liveDeviation(p, now, reverse = true)
            val backOk = if (c.isNull("stored")) back.isNaN() else same(back, c.getDouble("stored"))
            if (!same(forward, c.getDouble("live")) || !backOk) failures.add("live $i: $forward $back vs $c")
        }
        val weights = data.getJSONArray("weights")
        for (i in 0 until weights.length()) {
            val c = weights.getJSONObject(i)
            val win = c.getBoolean("win")
            val puzzle = Glicko.Rating(1500.0, c.getDouble("deviation"), 0.09)
            // The desktop's weights are single precision held in doubles: exact as Floats.
            val t = Glicko.themeWeight(theme(c), win)
            val p = Glicko.playerWeight(theme(c), win, puzzle)
            if (t.toDouble() != c.getDouble("theme_weight") || p.toDouble() != c.getDouble("player_weight")) {
                failures.add("weight $i: $t $p vs $c")
            }
        }
        assertEquals(emptyList<String>(), failures)
    }
}
