package com.sidecoach.analysis

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.Lane
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Analysis
import com.sidecoach.events.Label
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.Severity
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * The annotator on the phone: a game's moves, measured and named.
 *
 * The scoring itself is held to the Python by `ScoreTest`; this checks the layer that drives
 * the engine around it -- the same position measured once, the loss taken between two
 * searches of the same depth, the label that comes out, and the worker thread that keeps all
 * of it off the capture loop.
 *
 * It needs the engine's network: `tools/test.ps1` puts one there.
 */
@Lane(Lane.Kind.FREE)
@RunWith(AndroidJUnit4::class)
class AnnotatorTest {

    /** Stockfish, one thread as every search, every search asked of it counted. */
    private class Counted(binary: File, net: File) : Engine(binary, net, threads = 1, hashMb = 64, multipv = 4) {
        @Volatile var searches = 0

        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<com.sidecoach.events.Line>) -> Unit)?): List<com.sidecoach.events.Line> {
            searches += 1
            return super.analyse(board, depth, timeS, multipv, pvPlies, carryBest, onDepth)
        }
    }

    private fun engine(): Counted? {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val binary = Engine.binaryIn(context.applicationInfo.nativeLibraryDir)
        if (!binary.exists()) return null
        val net = context.filesDir.listFiles { f: File -> f.name.endsWith(".nnue") }?.firstOrNull()
            ?: return null
        return Counted(binary, net)
    }

    @Test
    fun aBlunderIsNamedABlunder() {
        val engine = engine()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.use { sf ->
            // The annotator's depth is the desktop's 18 by default; 14 is enough here and
            // keeps the test short.
            val annotator = GameAnnotator(sf, depth = 14, multipv = 4)
            // 1. e4 e5 2. Nf3 Nc6 3. Bc4 and now Black hangs the queen with Qg5.
            val board = Board()
            for (san in listOf("e4", "e5", "Nf3", "Nc6", "Bc4")) board.push(playSan(board, san))
            val blunder = playSan(board, "Qg5")
            val before = board.copy()
            val assessment = annotator.assessMove("game", before, blunder, ply = 6)
            Log.i(
                TAG,
                "Qg5: loss ${"%.3f".format(assessment.expectedLoss)}, ${assessment.severity}, " +
                    "${assessment.label}, accuracy ${assessment.accuracy}, " +
                    "best ${sanLine(before, assessment.best, 3)}",
            )
            assertEquals(Severity.BLUNDER, assessment.severity)
            assertEquals(Label.BLUNDER, assessment.label)
            assertTrue("the loss should be large", assessment.expectedLoss > 0.28)
            assertTrue("the accuracy should be low", (assessment.accuracy ?: 100.0) < 10.0)
        }
    }

    @Test
    fun theEnginesOwnMoveCostsNothing() {
        val engine = engine()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.use { sf ->
            val annotator = GameAnnotator(sf, depth = 14, multipv = 4)
            val board = Board("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
            val best = annotator.measureForVerdict(board).lines.first().moves.first()
            val assessment = annotator.assessMove("game", board, Move.fromUci(best), ply = 7)
            Log.i(TAG, "the engine's own move $best: ${assessment.label}, loss ${assessment.expectedLoss}")
            assertEquals(0.0, assessment.expectedLoss, 0.0)
            assertEquals(Severity.NONE, assessment.severity)
            assertEquals(Label.BEST, assessment.label)
        }
    }

    @Test
    fun aPositionIsMeasuredOnceAndCarriesItsThreats() {
        val engine = engine()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.use { sf ->
            val annotator = GameAnnotator(sf, depth = 12, multipv = 4)
            val board = Board("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4")
            val a = annotator.analysisEvent("game", board, ply = 8)
            val once = sf.searches
            val b = annotator.analysisEvent("game", board, ply = 8)
            Log.i(
                TAG,
                "the Italian for Black: ${a.lines.size} lines, decisive ${a.decisive}, " +
                    "only ${a.onlyMove}, threats ${a.threats}, $once searches then ${sf.searches - once}",
            )
            assertTrue("no lines", a.lines.isNotEmpty())
            assertTrue("the first look searched", once > 0)
            // Counted, not timed (it was: under 50 ms): the second look asks the engine nothing.
            assertEquals("the second look should be free", once, sf.searches)
            assertEquals(a.fen, b.fen)
            assertTrue("a quiet middlegame has a threat to name", a.threats.isNotEmpty())
        }
    }

    @Test
    fun theWorkerAnswersWithoutBlocking() {
        val engine = engine()
        assumeTrue("the engine or its network is missing", engine != null)
        val analyser = Analyser(GameAnnotator(engine!!, depth = 12, multipv = 4))
        val watch = AnalyserWatch(analyser)
        analyser.start()
        try {
            val board = Board()
            var ply = 0
            for (san in listOf("e4", "e5", "Nf3", "Nc6")) {
                val move = playSan(board, san)
                ply += 1
                analyser.submitMove("game", board, move, ply)
                board.push(move)
            }
            // Nothing above waited for the engine: that is the point.
            // Every request answered (signalled by the worker, not polled).
            val events = watch.settled()
            val assessments = events.filterIsInstance<MoveAssessment>().sortedBy { it.ply }
            val analyses = events.filterIsInstance<Analysis>()
            Log.i(
                TAG,
                "four plies: ${assessments.size} verdicts, ${analyses.size} analyses, " +
                    "${analyser.failures} failures — " +
                    assessments.joinToString(", ") { "${it.ply}:${it.label}" },
            )
            assertEquals("a verdict per move", 4, assessments.size)
            assertEquals("no failures: ${analyser.lastError}", 0, analyser.failures)
            for (one in assessments) {
                assertTrue("${one.ply} was not measured", one.depth > 0)
            }
        } finally {
            analyser.close()
        }
    }

    /**
     * `test_each_depth_is_delivered_as_the_search_finishes_it` (desktop 316026b). Tai,
     * 2026-09-28: "if the consumers read after stockfish reaches depth > 12 then they will get
     * whatever lines that is available at the time". The position the player faces is delivered
     * at the early depth, then at every depth the full search finishes on its way, provisional,
     * then whole at the full depth -- the same position, deeper each time.
     */
    @Test
    fun eachDepthIsDeliveredAsTheSearchFinishesIt() {
        val engine = engine()
        assumeTrue("the engine or its network is missing", engine != null)
        val board = Board(SCHOLAR)
        val annotator = GameAnnotator(engine!!, depth = 16, multipv = 3, withThreats = false)
        val analyser = Analyser(annotator)
        val watch = AnalyserWatch(analyser)
        analyser.start()
        val got = try {
            analyser.submitPosition("g", board, 0)
            watch.settled().filterIsInstance<Analysis>()
        } finally {
            analyser.close()
        }
        val depths = got.map { it.lines[0].depth }
        Log.i(TAG, "the depths delivered: $depths, provisional ${got.map { it.provisional }}")
        assertEquals(List(got.size - 1) { true } + false, got.map { it.provisional })
        assertTrue("$depths", depths.first() == EARLY_DEPTH && depths.last() == 16)
        assertTrue("$depths", depths == depths.sorted() && depths.toSet().size == depths.size)
        assertEquals(setOf(board.fen()), got.map { it.fen }.toSet())
    }

    /** The desktop pushes SAN; here the move is found by name among the legal ones. */
    private fun playSan(board: Board, san: String): Move =
        board.legalMoves().first { board.san(it).trimEnd('+', '#') == san.trimEnd('+', '#') }

    companion object {
        private const val TAG = "sidecoach.analysis"

        /** 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6??, where 4. Qxf7# is mate (the desktop's `SCHOLAR`). */
        private const val SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    }
}
