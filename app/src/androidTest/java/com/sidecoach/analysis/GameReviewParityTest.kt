package com.sidecoach.analysis

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.Lane
import com.sidecoach.chess.Board
import com.sidecoach.chess.PgnReader
import com.sidecoach.events.Line
import com.sidecoach.events.ReviewJson
import com.sidecoach.events.Side
import com.sidecoach.profile.GameSummaries
import com.sidecoach.ui.ReviewViews
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Game Review as the desktop's (analysis/GameReview.kt, profile/GameSummaries.kt,
 * ui/ReviewViews.kt): `tools/make_review_fixtures.py` reviews seven games with real engines
 * and records every position's measurement. Here the same measurements are adopted and the moves
 * walked (no engine: a search would fail the test), and the review, each move's clock, the
 * summary, the moves as the page draws them and the tactics the mistakes passed up are held to
 * the desktop's. A field the clock sets (`ts`) and the review's own time (`seconds`) are left out.
 */
@Lane(Lane.Kind.FREE)
@RunWith(AndroidJUnit4::class)
class GameReviewParityTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()

    private val fixture: JSONObject by lazy {
        JSONObject(instrumentation.context.assets.open("profile/reviews.json").bufferedReader().use { it.readText() })
    }

    /** An engine never asked: every position is measured already. */
    private class NoEngine : Engine(File("none"), null) {
        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> =
            throw AssertionError("searched ${board.fen()}")

        override fun close() = Unit
    }

    private fun book(): OpeningBook {
        val folder = File(instrumentation.targetContext.cacheDir, "openings-review").apply {
            deleteRecursively()
            mkdirs()
        }
        val assets = instrumentation.context.assets
        for (name in assets.list("openings").orEmpty().filter { it.endsWith(".tsv") }) {
            assets.open("openings/$name").use { input -> File(folder, name).outputStream().use { input.copyTo(it) } }
        }
        return OpeningBook.load(folder).also { assertNotNull("the tables load", it) }!!
    }

    @Test
    fun theWalkTheSummaryAndTheViewsAreTheDesktops() {
        val book = book()
        val games = fixture.getJSONArray("games")
        val failures = ArrayList<String>()
        var moves = 0
        for (i in 0 until games.length()) {
            val g = games.getJSONObject(i)
            val name = g.getString("name")
            val strength = GameReview.STRENGTHS.getValue(g.getString("strength"))
            val game = PgnReader.games(g.getString("pgn")).single()
            val measured = g.getJSONArray("measured").let { m -> (0 until m.length()).map { ReviewJson.measured(m.getJSONObject(it)) } }
            val reviewed = GameReview.reviewGame(
                game,
                listOf(NoEngine()),
                depth = strength.depth,
                multipv = strength.multipv,
                thresholds = Thresholds(),
                tagger = Motifs,
                book = book,
                strength = strength.name,
                gameId = "review",
                known = measured.associateBy { it.fen },
            )
            val want = g.getJSONObject("reviewed")
            moves += reviewed.moves.size
            same("$name review", ReviewJson.reviewed(reviewed), want, failures)
            // Its JSON reads back to the same review.
            same("$name review read back", ReviewJson.reviewed(ReviewJson.reviewed(want.toString())), want, failures)

            val clocks = GameReview.clocks(game).map { JSONArray().put(it.first ?: JSONObject.NULL).put(it.second ?: JSONObject.NULL) }
            same("$name clocks", JSONArray(clocks), g.getJSONArray("clocks"), failures)

            val desktop = ReviewJson.reviewed(want)
            same("$name summary", ReviewJson.summary(GameSummaries.summarize(desktop)), g.getJSONObject("summary"), failures)
            same("$name summary read back", ReviewJson.summary(ReviewJson.summary(g.getJSONObject("summary").toString())), g.getJSONObject("summary"), failures)
            same("$name views", ReviewViews.moveViews(desktop), g.getJSONArray("views"), failures)
            val missed = g.getJSONObject("missed")
            for ((key, sides) in listOf("white" to setOf(Side.WHITE), "black" to setOf(Side.BLACK), "both" to setOf(Side.WHITE, Side.BLACK))) {
                val mine = JSONArray(GameSummaries.missedMotifs(desktop, sides).map { JSONArray().put(it.first).put(it.second) })
                same("$name missed $key", mine, missed.getJSONArray(key), failures)
            }
        }
        assertTrue("$moves moves; ${failures.size} failures:\n" + failures.take(30).joinToString("\n"), failures.isEmpty())
    }

    companion object {
        private val SKIPPED = setOf("ts", "seconds")

        /** The two JSON values the same, numbers by value, `ts` and `seconds` left out. */
        fun same(path: String, mine: Any?, theirs: Any?, failures: MutableList<String>) {
            if (failures.size >= 40) return
            val a = if (mine == JSONObject.NULL) null else mine
            val b = if (theirs == JSONObject.NULL) null else theirs
            when {
                a is JSONObject && b is JSONObject -> {
                    val keys = (a.keys().asSequence() + b.keys().asSequence()).toSortedSet() - SKIPPED
                    for (k in keys) {
                        if (!a.has(k) || !b.has(k)) {
                            failures.add("$path.$k: ${if (a.has(k)) "only mine" else "only python's"}")
                        } else {
                            same("$path.$k", a.get(k), b.get(k), failures)
                        }
                    }
                }
                a is JSONArray && b is JSONArray -> {
                    if (a.length() != b.length()) failures.add("$path: ${a.length()} items, python ${b.length()}\n  mine:   $a\n  python: $b")
                    else for (i in 0 until a.length()) same("$path[$i]", a.get(i), b.get(i), failures)
                }
                a is Number && b is Number -> if (a.toDouble() != b.toDouble()) failures.add("$path: mine $a, python $b")
                else -> if (a != b) failures.add("$path: mine $a, python $b")
            }
        }
    }
}
