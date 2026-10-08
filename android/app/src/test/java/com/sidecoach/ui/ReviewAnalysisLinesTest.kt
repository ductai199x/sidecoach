package com.sidecoach.ui

import com.sidecoach.analysis.GameReview
import com.sidecoach.analysis.GatedEngine
import com.sidecoach.analysis.HANG_MS
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit

/**
 * The engine's lines of the analysis board (`ReviewAnalysis.Lines`; the desktop's
 * tests/test_review_server.py::test_the_lines_of_the_position_on_the_board_never_wait_for_one_stepped_past).
 */
class ReviewAnalysisLinesTest {
    /**
     * Stepping through the analysis board asks the engine's lines of each position it passes;
     * the position the player stops on is answered as soon as its own search is done. A search
     * for a position the board has left is stopped, never finished first: every one was searched
     * to its end, one after another, and the lines of the one on the board came ten seconds
     * after ten steps (the audit of 2026-10-07).
     */
    @Test
    fun theLinesOfThePositionOnTheBoardNeverWaitForOneSteppedPast() {
        val left = Board().also { it.push(Move.fromUci("e2e4")) }
        val here = left.copy().also { it.push(Move.fromUci("e7e5")) }
        val engine = GatedEngine(gated = listOf(left.fen()))
        val fast = GameReview.STRENGTHS.getValue("fast")
        val lines = ReviewAnalysis.Lines(fast) { _, _ -> engine }
        val answers = ConcurrentHashMap<String, Any>()

        fun ask(fen: String) = Thread({
            answers[fen] = try {
                lines.lines(fen)
            } catch (exc: RuntimeException) { // Superseded; or a search never let go
                exc
            }
        }, "asking $fen").apply { isDaemon = true }

        val before = ask(left.fen())
        val now = ask(here.fen())
        before.start()
        try {
            assertTrue("the first search never began", engine.entered.getValue(left.fen()).await(HANG_MS, TimeUnit.MILLISECONDS))
            now.start() // the player stepped on while the position left was being searched
            now.join(HANG_MS)
            // Answered with the search of the position left never let go: stopped for it.
            assertEquals("the board's lines waited for a position left", listOf(left.fen()), engine.halted.toList())
            before.join(HANG_MS)
        } finally {
            engine.release(left.fen())
            before.join(HANG_MS)
            now.join(HANG_MS)
        }

        val shown = answers.getValue(here.fen()) as JSONObject
        assertEquals(here.fen(), shown.getString("fen"))
        assertEquals(fast.depth, shown.getInt("depth"))
        val drawn = shown.getJSONArray("lines")
        assertTrue("the same lines as ever", drawn.length() == 1 && drawn.getJSONObject(0).getJSONArray("steps").length() > 0)
        assertTrue("${answers[left.fen()]}", answers[left.fen()]?.javaClass?.simpleName == "Superseded")
        // The stop holds for that search alone: the position asked again is answered whole.
        val again = lines.lines(left.fen())
        assertEquals(left.fen(), again.getString("fen"))
        assertEquals(fast.depth, again.getInt("depth"))
        lines.close()
    }
}
