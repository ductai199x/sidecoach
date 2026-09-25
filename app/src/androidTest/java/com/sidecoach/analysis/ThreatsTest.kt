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
 * What a quiet best move threatens ([aimLines]): desktop 29ae423's two tests in
 * `tests/test_analysis.py`, on Lichess 06VoP, 32.Qg4 with Qxg7# its point. The second needs the
 * engine's network: `tools/test.ps1` puts one there.
 */
@RunWith(AndroidJUnit4::class)
class ThreatsTest {

    /** The desktop test's `FakeEngine`: scripted lines, a quiet move for anything else. */
    private class FakeEngine(private val answers: Map<String, List<Line>>) : Engine(File("fake"), null) {
        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean): List<Line> {
            if (board.isGameOver(claimDraw = true)) return emptyList()
            return answers[board.fen()] ?: listOf(Line(moves = listOf("a2a3"), depth = 1, expectedScore = 0.5))
        }

        override fun close() = Unit
    }

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
    fun aQuietBestMoveIsAskedWhatItThreatens() {
        val board = Board(QUIET_MATE)
        val passed = board.copy(withStack = false)
        passed.push(Move.fromUci("d1g4"))
        passed.push(Move.NULL)
        val threat = listOf(Line(moves = listOf("g4g7"), depth = 14, expectedScore = 1.0, mate = 1))
        val engine = FakeEngine(mapOf(Board(passed.fen()).fen() to threat))
        assertEquals(threat, aimLines(engine, board, listOf(Line(moves = listOf("d1g4"), depth = 18, expectedScore = 0.92))))
        // A capture is its own point, and a line that only holds is not a win to explain.
        assertEquals(emptyList<Line>(), aimLines(engine, board, listOf(Line(moves = listOf("h5g7"), depth = 18, expectedScore = 0.9))))
        assertEquals(emptyList<Line>(), aimLines(engine, board, listOf(Line(moves = listOf("d1g4"), depth = 18, expectedScore = 0.5))))
    }

    @Test
    fun theEngineFindsWhatAQuietMoveThreatens() {
        val engine = engine()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.use { sf ->
            val board = Board(QUIET_MATE)
            val aims = aimLines(sf, board, listOf(Line(moves = listOf("d1g4"), depth = 18, expectedScore = 0.92)), depth = 10)
            assertTrue(aims.toString(), aims.isNotEmpty() && aims[0].moves[0] == "g4g7" && aims[0].mate == 1)
        }
    }

    private companion object {
        // 32.Qg4 in Lichess 06VoP: a quiet queen move whose point is Qxg7#.
        const val QUIET_MATE = "2rr2k1/5pp1/p1qp3p/4pP1N/1p6/1P4PK/PP5P/R2Q4 w - - 5 32"
    }
}
