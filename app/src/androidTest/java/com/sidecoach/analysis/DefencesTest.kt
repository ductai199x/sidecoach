package com.sidecoach.analysis

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Line
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * The recaptures a winning line declines ([Defences]) and the solution played out of order
 * ([Traps.tempting]): desktop a1f7f81's three tests in `tests/test_analysis.py`, on Tai's
 * chess.com puzzles of 2026-09-24. The last needs the engine's network: `tools/test.ps1` puts
 * one there.
 */
@RunWith(AndroidJUnit4::class)
class DefencesTest {

    /** The desktop test's `engine` fixture: two threads, 64 MB, four lines. */
    private fun engine(): Engine? {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val binary = Engine.binaryIn(context.applicationInfo.nativeLibraryDir)
        if (!binary.exists()) return null
        val net = context.filesDir.listFiles { f: File -> f.name.endsWith(".nnue") }?.firstOrNull()
            ?: return null
        return Engine(binary, net, threads = 2, hashMb = 64, multipv = 4)
    }

    @Test
    fun theRecapturesALineDeclinesAreFoundWithoutAnEngine() {
        val board = Board(OVERLOADED)
        val found = Defences.declined(board, Line(moves = OVERLOADED_LINE, depth = 18, expectedScore = 0.975))
        assertEquals(listOf(0 to "f8f7", 2 to "f8b8"), found.map { (i, m) -> i to m.uci() })
        val queen = Board("2kr1b1r/ppq2p2/2p1pp2/8/2B5/3P2Q1/PPPB1PP1/R4RK1 b - - 1 1")
        val line = Line(moves = listOf("c7g3", "f1e1", "g3h4"), depth = 18, expectedScore = 0.975)
        assertEquals(listOf(0 to "f2g3"), Defences.declined(queen, line).map { (i, m) -> i to m.uci() })
        val takenBack = Line(moves = listOf("c7g3", "f2g3", "f8c5"), depth = 18, expectedScore = 0.94)
        assertEquals("the line takes back itself", emptyList<Pair<Int, Move>>(), Defences.declined(queen, takenBack))
    }

    @Test
    fun theSolutionPlayedOutOfOrderIsTheFirstTemptation() {
        val board = Board(OVERLOADED)
        val lines = listOf(Line(moves = OVERLOADED_LINE, depth = 18, expectedScore = 0.975))
        assertEquals("d6b8", Traps.tempting(board, lines)[0].uci())
    }

    @Test
    fun aDeclinedRecaptureIsPlayedOutByTheEngine() {
        val engine = engine()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.use { sf ->
            val board = Board(OVERLOADED)
            val lines = listOf(Line(moves = OVERLOADED_LINE, depth = 18, expectedScore = 0.975))
            val defences = Defences.defenceLines(sf, board, lines, depth = 12)
            val rook = defences.filter { it.moves.take(4) == listOf("e6f7", "g8h8", "d6b8", "f8b8") }
            val mate = rook.firstOrNull()?.mate
            assertTrue(defences.toString(), mate != null && mate > 0)
            assertTrue("the pawn queens where the rook stood", rook[0].moves[4].startsWith("f7f8"))
        }
    }

    private companion object {
        // Tai's chess.com puzzle of 2026-09-24: 2.exf7+ Kh8 3.Qxb8, and the rook on f8 cannot take back.
        const val OVERLOADED = "1r3rk1/pp3ppp/2pQP3/1P1pP1q1/3Nn3/6P1/P5KP/2R2R2 w - - 1 2"
        val OVERLOADED_LINE = listOf("e6f7", "g8h8", "d6b8", "g5d2", "g2h1", "d2b4")
    }
}
