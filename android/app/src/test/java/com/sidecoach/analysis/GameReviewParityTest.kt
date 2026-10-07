package com.sidecoach.analysis

import com.sidecoach.TestAssets
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
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/**
 * Game Review as the desktop's (analysis/GameReview.kt, profile/GameSummaries.kt,
 * ui/ReviewViews.kt): `tools/make_review_fixtures.py` reviews seven games with real engines
 * and records every position's measurement. Here the same measurements are adopted and the moves
 * walked (no engine: a search would fail the test), and the review, each move's clock, the
 * summary, the moves as the page draws them and the tactics the mistakes passed up are held to
 * the desktop's. A field the clock sets (`ts`) and the review's own time (`seconds`) are left out.
 */
class GameReviewParityTest {
    /** The opening tables' folder: the device's test used the app's cache, the host a folder of its own. */
    @get:Rule
    val temporary = TemporaryFolder()

    private val fixture: JSONObject by lazy {
        JSONObject(TestAssets.text("profile/reviews.json"))
    }

    /** An engine never asked: every position is measured already. */
    private class NoEngine : Engine(File("none"), null) {
        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> =
            throw AssertionError("searched ${board.fen()}")

        override fun close() = Unit
    }

    private fun book(): OpeningBook {
        val folder = temporary.newFolder("openings-review")
        for (name in TestAssets.list("openings").filter { it.endsWith(".tsv") }) {
            TestAssets.open("openings/$name").use { input -> File(folder, name).outputStream().use { input.copyTo(it) } }
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

    /**
     * `test_a_review_relabelled_with_other_thresholds_is_the_review_made_with_them` (the audit of
     * 2026-10-07): a saved review made again ([GameReview.relabel]) is the review made today --
     * with today's thresholds, the desktop's review itself; with other thresholds and its motifs
     * gone (saved before a motif existed), what the desktop's relabel made of it (`relabelled`):
     * labels, severities, the positions' calls and the motifs read again.
     */
    @Test
    fun aSavedReviewMadeAgainIsTheReviewMadeToday() {
        val book = book()
        val games = fixture.getJSONArray("games")
        val failures = ArrayList<String>()
        for (i in 0 until games.length()) {
            val g = games.getJSONObject(i)
            val name = g.getString("name")
            val want = g.getJSONObject("reviewed")
            val desktop = ReviewJson.reviewed(want)
            same("$name relabelled today", ReviewJson.reviewed(GameReview.relabel(desktop, Thresholds(), book, Motifs)), want, failures)
            val relabelled = g.getJSONObject("relabelled")
            val t = relabelled.getJSONObject("thresholds")
            val other = Thresholds(
                inaccuracy = t.getDouble("inaccuracy"),
                mistake = t.getDouble("mistake"),
                blunder = t.getDouble("blunder"),
                decisiveGap = t.getDouble("decisive_gap"),
                onlyMoveGap = t.getDouble("only_move_gap"),
                excellent = t.getDouble("excellent"),
                decided = t.getDouble("decided"),
            )
            val untagged = JSONObject(want.toString())
            val moves = untagged.getJSONArray("moves")
            for (m in 0 until moves.length()) moves.getJSONObject(m).getJSONObject("assessment").put("tags", JSONArray())
            val again = GameReview.relabel(ReviewJson.reviewed(untagged), other, tagger = Motifs)
            same("$name relabelled", ReviewJson.reviewed(again), relabelled.getJSONObject("reviewed"), failures)
        }
        assertTrue("${failures.size} failures:\n" + failures.take(30).joinToString("\n"), failures.isEmpty())
    }

    /** The two JSON values the same, numbers by value, `ts` and `seconds` left out ([GameReviewParity]). */
    private fun same(path: String, mine: Any?, theirs: Any?, failures: MutableList<String>) =
        GameReviewParity.same(path, mine, theirs, failures)
}
