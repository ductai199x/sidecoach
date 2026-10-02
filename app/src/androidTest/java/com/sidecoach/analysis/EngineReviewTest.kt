package com.sidecoach.analysis

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.Lane
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.MoveAssessment
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * The engine review of 2026-10-01, ported with the desktop's tests (`tests/test_analysis.py`,
 * desktop branch desk-engine): the move labels as chess.com defines them, a draw that could be
 * claimed played on, positions cached with the moves that led there, settings saved
 * mid-search, a cut search's lines of one depth, and an engine that stops answering replaced.
 */
@Lane(Lane.Kind.FREE)
@RunWith(AndroidJUnit4::class)
class EngineReviewTest {

    // ---- the labels ---------------------------------------------------------------------

    private fun label(
        loss: Double,
        isBest: Boolean = false,
        inBook: Boolean = false,
        decisive: Boolean = false,
        sacrifice: Boolean = false,
        beforeScore: Double = 0.5,
        secondScore: Double? = null,
        opponentBlundered: Boolean = false,
        chance: Boolean = false,
        mateMissed: Boolean = false,
        mateAllowed: Boolean = false,
        obvious: Boolean = false,
    ): Label = labelMove(
        loss, Thresholds(), isBest = isBest, inBook = inBook, decisive = decisive,
        sacrifice = sacrifice, beforeScore = beforeScore, opponentBlundered = opponentBlundered,
        secondScore = secondScore, chance = chance, mateMissed = mateMissed,
        mateAllowed = mateAllowed, obvious = obvious,
    )

    @Test
    fun labelsFollowChessComsDefinitions() {
        assertEquals(Label.BEST, label(0.0, isBest = true))
        assertEquals(Label.EXCELLENT, label(0.01))
        assertEquals(Label.EXCELLENT, label(0.0))
        assertEquals(Label.GOOD, label(0.03))
        assertEquals(Label.INACCURACY, label(0.06))
        assertEquals(Label.MISTAKE, label(0.12))
        assertEquals(Label.BLUNDER, label(0.30))
        assertEquals(Label.BOOK, label(0.0, inBook = true))
        assertEquals(Label.BOOK, label(0.06, inBook = true))
        assertEquals(Label.MISTAKE, label(0.14, inBook = true))
    }

    @Test
    fun greatIsTheMoveThatAlteredTheCourseOfTheGame() {
        assertEquals(Label.GREAT, label(0.0, isBest = true, decisive = true))
        assertEquals(Label.GREAT, label(0.0, isBest = true, decisive = true, beforeScore = 0.95))
        assertEquals(Label.GREAT, label(0.0, isBest = true, opponentBlundered = true))
        assertEquals(Label.BEST, label(0.0, isBest = true))
        assertEquals(Label.EXCELLENT, label(0.0, decisive = true))
        assertEquals(Label.BEST, label(0.0, isBest = true, decisive = true, obvious = true))
        assertEquals(Label.BEST, label(0.0, isBest = true, opponentBlundered = true, obvious = true))
        assertEquals(Label.BEST, label(0.0, isBest = true, decisive = true, beforeScore = 0.02))
    }

    @Test
    fun brilliantIsASoundSacrificeThatWasNeeded() {
        assertEquals(Label.BRILLIANT, label(0.0, isBest = true, sacrifice = true))
        assertEquals(Label.BRILLIANT, label(0.01, sacrifice = true))
        assertEquals(Label.GOOD, label(0.03, sacrifice = true))
        assertEquals(Label.BLUNDER, label(0.3, sacrifice = true))
        assertEquals(Label.BEST, label(0.0, isBest = true, sacrifice = true, beforeScore = 0.97, secondScore = 0.95))
        assertEquals(Label.BRILLIANT, label(0.0, isBest = true, sacrifice = true, beforeScore = 1.0, secondScore = 0.3))
        assertEquals(Label.BEST, label(0.0, isBest = true, sacrifice = true, beforeScore = 0.97))
        assertEquals(Label.BEST, label(0.0, isBest = true, sacrifice = true, beforeScore = 0.05))
    }

    @Test
    fun aMissIsATacticOrAMateLetGo() {
        assertEquals(Label.MISS, label(0.06, chance = true))
        assertEquals(Label.MISS, label(0.15, chance = true))
        assertEquals(Label.MISS, label(0.90, chance = true))
        assertEquals(Label.EXCELLENT, label(0.01, chance = true))
        assertEquals(Label.MISTAKE, label(0.15))
        assertEquals(Label.MISS, label(0.02, chance = true, mateMissed = true))
        assertEquals(Label.BLUNDER, label(0.02, mateAllowed = true, beforeScore = 0.3))
        assertEquals(Label.GOOD, label(0.02, mateAllowed = true, beforeScore = 0.03))
        assertEquals(Label.BEST, label(0.0, mateAllowed = true, beforeScore = 0.3, isBest = true))
    }

    @Test
    fun whatALineWinsIsCountedAfterTheReplies() {
        val board = Board()
        for (san in listOf("e4", "d5")) board.push(board.parseSan(san)!!)
        assertEquals(0, materialWon(board, listOf("e4d5", "d8d5")))
        assertEquals(3, materialWon(Board("4k3/8/8/3n4/8/8/8/3QK3 w - - 0 1"), listOf("d1d5", "e8f8")))
        val traded = Board("4k3/8/8/3n4/8/2N5/8/4K3 b - - 0 1")
        traded.push(traded.parseSan("Nxc3")!!)
        assertEquals(0, materialWon(traded, listOf("e1d2", "c3d5")))
        assertEquals(99, materialWon(Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"), listOf("a1a8")))
    }

    // ---- a claimable draw ----------------------------------------------------------------

    /** White a queen up, the kings shuffling, black to move first; Ke1 next would repeat thrice. */
    private fun claimable(): Board {
        val board = Board("4k3/8/8/8/8/8/8/Q3K3 b - - 0 1")
        for (san in listOf("Kd8", "Ke2", "Ke8", "Ke1", "Kd8", "Ke2", "Ke8")) board.push(board.parseSan(san)!!)
        return board
    }

    @Test
    fun aDrawThatCouldBeClaimedIsNotAFinishedGame() {
        val board = claimable()
        assertTrue(board.canClaimThreefoldRepetition() && !board.isRepetition(3))
        assertNull(terminalExpectedScore(board))
        board.push(board.parseSan("Ke1")!!)
        assertTrue(board.isRepetition(3))
        assertEquals(0.5, terminalExpectedScore(board)!!, 0.0)
        assertEquals(0.5, terminalExpectedScore(Board("4k3/8/8/8/8/8/8/Q3K3 w - - 100 80"))!!, 0.0)
        val near = Board("4k3/8/8/8/8/8/8/Q3K3 w - - 99 80")
        assertTrue(near.canClaimFiftyMoves())
        assertNull(terminalExpectedScore(near))
    }

    @Test
    fun aClaimableDrawIsSearchedLikeAnyPosition() {
        val sf = engine()
        assumeTrue("the engine or its network is missing", sf != null)
        sf!!.use {
            val lines = it.analyse(claimable(), depth = 10)
            assertTrue("the side to move has moves to play", lines.isNotEmpty())
            assertTrue("a queen up, white plays on: ${lines[0].expectedScore}", lines[0].expectedScore > 0.8)
        }
    }

    // ---- the caches ------------------------------------------------------------------------

    /** The desktop test's `FakeEngine`: scripted lines, a quiet move for anything else, the calls kept. */
    private open class FakeEngine(private val answers: Map<String, List<Line>> = emptyMap()) : Engine(File("fake"), null) {
        val calls = java.util.Collections.synchronizedList(ArrayList<String>())
        val histories = java.util.Collections.synchronizedList(ArrayList<List<String>>())

        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
            calls.add(board.fen())
            histories.add(board.moveStack.map { it.uci() })
            if (gameOver(board)) return emptyList()
            return answers[board.fen()] ?: listOf(Line(moves = listOf("a2a3"), depth = 1, expectedScore = 0.5))
        }

        override fun close() = Unit
    }

    private fun played(vararg sans: String): Board {
        val board = Board()
        for (san in sans) board.push(board.parseSan(san)!!)
        return board
    }

    @Test
    fun onePositionReachedByTwoHistoriesIsMeasuredForEach() {
        val one = played("Nf3", "Nc6", "Nc3", "Nf6")
        val two = played("Nc3", "Nf6", "Nf3", "Nc6")
        assertEquals(one.fen(), two.fen())
        val fake = FakeEngine()
        val annotator = GameAnnotator(fake, depth = 6, withThreats = false)
        annotator.measure(one)
        annotator.measure(two)
        annotator.measure(two.copy())
        assertEquals("each history searched once", 2, fake.calls.count { it == one.fen() })
        one.push(one.parseSan("e4")!!)
        two.push(two.parseSan("e4")!!)
        annotator.measure(one)
        annotator.measure(two)
        assertEquals(1, fake.calls.count { it == one.fen() })
    }

    @Test
    fun aPositionLookedAtAheadCarriesTheMovesThatLeadThere() {
        val board = played("Nf3", "Nf6", "Ng1")
        val fake = FakeEngine(mapOf(board.fen() to listOf(Line(moves = listOf("f6g8"), depth = 6, expectedScore = 0.5))))
        val annotator = GameAnnotator(fake, depth = 6, withThreats = false)
        // Driven a step at a time on this thread, as the worker drives it.
        val analyser = Analyser(annotator, lookAhead = 1)
        analyser.planAhead(board, listOf(Line(moves = listOf("f6g8"), depth = 6, expectedScore = 0.5)))
        while (analyser.ahead.isNotEmpty()) {
            val (next, moves) = analyser.ahead.removeAt(0)
            analyser.lookAhead(next, moves, analyser.ahead)
        }
        assertEquals(2, analyser.looked)
        val target = played("Nf3", "Nf6", "Ng1", "Ng8").fen()
        val ahead = fake.calls.indices.filter { fake.calls[it] == target }.map { fake.histories[it] }
        assertEquals(listOf(listOf("g1f3", "g8f6", "f3g1", "f6g8")), ahead)
    }

    @Test
    fun aChangedSettingForgetsWhatWasMeasured() {
        val fake = FakeEngine()
        val annotator = GameAnnotator(fake, depth = 6, withThreats = false)
        val board = Board()
        annotator.measure(board)
        assertTrue(annotator.measured(board))
        annotator.forgetMeasured()
        assertFalse(annotator.measured(board))
    }

    /** The desktop's test_a_search_under_way_when_the_settings_change_is_not_kept. */
    @Test
    fun aSearchUnderWayWhenTheSettingsChangeIsNotKept() {
        class ChangedMidSearch : FakeEngine() {
            var annotator: GameAnnotator? = null

            override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
                val lines = super.analyse(board, depth, timeS, multipv, pvPlies, carryBest, onDepth)
                if (calls.size == 1) annotator?.forgetMeasured()  // the save, while this search ran
                return lines
            }
        }
        for (verdict in listOf(false, true)) {
            val fake = ChangedMidSearch()
            val annotator = GameAnnotator(fake, depth = 6, withThreats = false)
            fake.annotator = annotator
            val board = Board()
            val measure = { if (verdict) annotator.measureForVerdict(board) else annotator.measure(board) }
            assertTrue("the caller still gets what it asked for", measure().lines.isNotEmpty())
            assertFalse("verdict=$verdict", annotator.measured(board))
            assertTrue("verdict=$verdict", annotator.computed.lines(board).isEmpty())
            measure()
            assertEquals("verdict=$verdict: searched again the new way", 2, fake.calls.size)
            measure()
            assertEquals("verdict=$verdict: and that one is kept", 2, fake.calls.size)
        }
    }

    private fun judged(before: Board, move: String, beforeLine: Line, afterLine: Line): MoveAssessment {
        val after = before.copy()
        after.push(Move.fromUci(move))
        val fake = FakeEngine(mapOf(before.fen() to listOf(beforeLine), after.fen() to listOf(afterLine)))
        val annotator = GameAnnotator(fake, depth = 6, withThreats = false)
        return annotator.assessMove("g", before, Move.fromUci(move), 1)
    }

    @Test
    fun aForcedMateLetGoIsAMissAndOneWalkedIntoABlunder() {
        val board = Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
        val mating = Line(moves = listOf("a1a8"), depth = 6, expectedScore = 1.0, mate = 1)
        val stillWon = Line(moves = listOf("g8f8"), depth = 6, expectedScore = 0.03, scoreCp = -1000)
        assertEquals(Label.MISS, judged(board, "h2h3", mating, stillWon).label)
        val worse = Line(moves = listOf("g8f8"), depth = 6, expectedScore = 0.2, scoreCp = -400)
        val mates = Line(moves = listOf("a1a8"), depth = 6, expectedScore = 1.0, mate = 1)
        val black = Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1")
        assertEquals(Label.BLUNDER, judged(black, "h7h6", worse, mates).label)
    }

    /** The desktop's test_the_best_move_of_a_mating_line_is_never_a_mate_let_go. */
    @Test
    fun theBestMoveOfAMatingLineIsNeverAMateLetGo() {
        val board = Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
        val longMate = Line(moves = listOf("h2h3", "g8f8"), depth = 6, expectedScore = 1.0, mate = 9)
        val seenAsWon = Line(moves = listOf("g8f8"), depth = 6, expectedScore = 0.03, scoreCp = -1000)
        val best = judged(board, "h2h3", longMate, seenAsWon)
        assertEquals(0.0, best.expectedLoss, 0.0)
        assertEquals(Label.BEST, best.label)
    }

    // ---- the engine ------------------------------------------------------------------------

    /**
     * An engine whose process is a shell script speaking UCI: the handshake, the options a
     * Stockfish has, and to `go` the lines given, then `bestmove`. No Stockfish, no clock.
     */
    private class Scripted(binary: File, private val answer: List<String>) :
        Engine(binary, null, threads = 2, hashMb = 64, multipv = 4) {
        override fun launch(): Process {
            val lines = answer.joinToString("; ") { "echo '$it'" }
            val script = "while read l; do case \"\$l\" in " +
                "uci) echo 'id name scripted'; " +
                "echo 'option name MultiPV type spin default 1 min 1 max 500'; " +
                "echo 'option name Threads type spin default 1 min 1 max 1024'; " +
                "echo 'option name Hash type spin default 16 min 1 max 33554432'; echo uciok;; " +
                "isready) echo readyok;; go*) $lines; echo 'bestmove d2d4';; quit) exit 0;; esac; done"
            return ProcessBuilder("sh", "-c", script).redirectErrorStream(true).start()
        }
    }

    private fun scripted(answer: List<String>): Scripted {
        val binary = File(InstrumentationRegistry.getInstrumentation().targetContext.cacheDir, "scripted-engine")
        binary.writeText("")
        return Scripted(binary, answer)
    }

    /** Four lines at depth 13, then the first two of depth 14 before the time cap cut it. */
    private val cut = listOf(
        "info depth 13 multipv 1 score cp 40 pv e2e4",
        "info depth 13 multipv 2 score cp 30 pv d2d4",
        "info depth 13 multipv 3 score cp 20 pv g1f3",
        "info depth 13 multipv 4 score cp 10 pv c2c4",
        "info depth 14 multipv 1 score cp 45 pv d2d4",
        "info depth 14 multipv 2 score cp 35 pv e2e4",
    )

    @Test
    fun aSearchCutShortGivesLinesOfOneDepth() {
        scripted(cut).use { sf ->
            val lines = sf.analyse(Board(), depth = 40, multipv = 4)
            assertEquals(listOf(13, 13, 13, 13), lines.map { it.depth })
            assertEquals(listOf("e2e4", "d2d4", "g1f3", "c2c4"), lines.map { it.moves[0] })
        }
        scripted(cut.take(4)).use { sf ->
            assertEquals(listOf(13, 13, 13, 13), sf.analyse(Board(), depth = 13, multipv = 4).map { it.depth })
        }
    }

    @Test
    fun settingsSavedMidSearchWaitForTheSearchToEnd() {
        scripted(cut.take(4)).use { sf ->
            var owedDuring: Boolean? = null
            sf.analyse(Board(), depth = 13, multipv = 4, onDepth = {
                // The save comes from another thread while this search holds the engine.
                val saver = Thread { sf.reconfigure(threads = 1, hashMb = 32) }
                saver.start()
                saver.join()
                owedDuring = sf.optionsOwed
            })
            assertEquals("kept for after the search", true, owedDuring)
            assertEquals(1, sf.threads)
            assertEquals(32, sf.hashMb)
            sf.analyse(Board(), depth = 13, multipv = 4)
            assertFalse("set before the next search", sf.optionsOwed)
            sf.reconfigure(threads = 2)
            assertFalse("with no search running, set at once", sf.optionsOwed)
        }
    }

    /**
     * An engine whose first process answers the handshake and then never a search, on a clock
     * that runs an hour at every look while that process lives: the watchdog's patience runs
     * out at once on it, and the real engine after it runs on the real clock.
     */
    private class FirstSilent(binary: File, net: File) : Engine(binary, net, threads = 1, hashMb = 16, multipv = 2) {
        var launched = 0
        private var fake = 0L

        override fun now(): Long = if (launched == 1) {
            fake += 3_600_000
            fake
        } else {
            super.now()
        }

        override fun launch(): Process {
            launched += 1
            if (launched > 1) return super.launch()
            val script = "while read l; do case \"\$l\" in " +
                "uci) echo 'id name silent'; echo 'option name MultiPV type spin default 1 min 1 max 500'; echo uciok;; " +
                "isready) echo readyok;; quit) exit 0;; esac; done"
            return ProcessBuilder("sh", "-c", script).redirectErrorStream(true).start()
        }
    }

    @Test
    fun anEngineThatStopsAnsweringIsReplaced() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val binary = Engine.binaryIn(context.applicationInfo.nativeLibraryDir)
        val net = context.filesDir.listFiles { f: File -> f.name.endsWith(".nnue") }?.firstOrNull()
        assumeTrue("the engine or its network is missing", binary.exists() && net != null)
        FirstSilent(binary, net!!).use { sf ->
            sf.maxSeconds = 5.0
            val lines = sf.analyse(Board(), depth = 6)
            assertTrue("searched again on a new engine", lines.isNotEmpty())
            assertEquals("the silent engine was replaced", 2, sf.launched)
        }
    }

    private fun engine(): Engine? {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val binary = Engine.binaryIn(context.applicationInfo.nativeLibraryDir)
        if (!binary.exists()) return null
        val net = context.filesDir.listFiles { f: File -> f.name.endsWith(".nnue") }?.firstOrNull() ?: return null
        return Engine(binary, net, threads = 2, hashMb = 64, multipv = 4)
    }
}
