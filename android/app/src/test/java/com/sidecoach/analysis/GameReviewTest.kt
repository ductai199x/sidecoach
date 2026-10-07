package com.sidecoach.analysis

import com.sidecoach.chess.Board
import com.sidecoach.chess.PgnReader
import com.sidecoach.events.Line
import com.sidecoach.events.ReviewedGame
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * A review's engines failing or stopped ([GameReview.reviewGame], `tests/test_game_review.py`):
 * scripted engines, nothing searched for real.
 */
class GameReviewTest {

    private val game = PgnReader.games("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *").single()

    private fun verdicts(review: ReviewedGame) =
        review.moves.map { Triple(it.assessment.uci, it.assessment.label, it.assessment.expectedLoss) }

    /**
     * `test_an_engine_that_fails_leaves_its_share_to_the_others`: side by side, one engine dies
     * on its first search -- its best line another engine may be waiting on: the others finish the
     * game, nothing waits for good, and the review's verdicts are a single engine's.
     */
    @Test
    fun anEngineThatFailsLeavesItsShareToTheOthers() {
        val dead = object : ScriptEngine() {
            override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> =
                throw RuntimeException("engine gone")
        }
        val reviewed = GameReview.reviewGame(game, listOf(dead, ScriptEngine(), ScriptEngine()), depth = 18, multipv = 2, thresholds = Thresholds())
        val alone = GameReview.reviewGame(game, listOf(ScriptEngine()), depth = 18, multipv = 2, thresholds = Thresholds())
        assertEquals(6, alone.moves.size)
        assertEquals(verdicts(alone), verdicts(reviewed))
    }

    /**
     * `test_a_review_stopped_before_it_measured_anything_searches_nothing`: stopped before the
     * engines took a position: the walk searches nothing more -- a review the room throws away --
     * where it used to measure the start position whole.
     */
    @Test
    fun aReviewStoppedBeforeItMeasuredAnythingSearchesNothing() {
        val engine = ScriptEngine()
        val reviewed = GameReview.reviewGame(game, listOf(engine), depth = 18, multipv = 2, thresholds = Thresholds(), stop = AtomicBoolean(true))
        assertTrue(engine.calls.isEmpty())
        assertTrue(reviewed.moves.isEmpty())
    }

    /**
     * `test_a_review_stopped_mid_search_cuts_the_search`: the room's Pause lands while an engine
     * searches: the search under way is stopped, not finished first with the rest of its position
     * (the audit of 2026-10-05, round 2), and nothing is counted failed.
     */
    @Test
    fun aReviewStoppedMidSearchCutsTheSearch() {
        val start = Board().fen()
        val engine = GatedEngine(gated = listOf(start))
        val stop = AtomicBoolean(false)
        val reviewing = Thread({
            GameReview.reviewGame(game, listOf(engine), depth = 18, multipv = 2, thresholds = Thresholds(), stop = stop)
        }, "the review").apply { isDaemon = true }
        reviewing.start()
        assertTrue(engine.entered.getValue(start).await(HANG_MS, TimeUnit.MILLISECONDS))
        stop.set(true)
        reviewing.join(HANG_MS)
        assertTrue(!reviewing.isAlive)
        assertEquals(listOf(start), engine.halted.toList())
    }
}
