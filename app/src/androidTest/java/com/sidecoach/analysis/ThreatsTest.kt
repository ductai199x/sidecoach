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
 * The opponent's threat ([threatLines]) as `tests/test_threats.py` pins its plumbing -- the
 * position handed over, the guards that must not call the engine at all -- and what a quiet
 * best move threatens ([aimLines]): desktop 29ae423's two tests in `tests/test_analysis.py`, on
 * Lichess 06VoP, 32.Qg4 with Qxg7# its point. The last needs the engine's network:
 * `tools/test.ps1` puts one there.
 */
@RunWith(AndroidJUnit4::class)
class ThreatsTest {

    /** The desktop test's `FakeEngine`: scripted lines, a quiet move for anything else. */
    private class FakeEngine(private val answers: Map<String, List<Line>>) : Engine(File("fake"), null) {
        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
            if (board.isGameOver(claimDraw = true)) return emptyList()
            return answers[board.fen()] ?: listOf(Line(moves = listOf("a2a3"), depth = 1, expectedScore = 0.5))
        }

        override fun close() = Unit
    }

    /** `test_threats.py`'s `_RecordingEngine`: canned lines, and every position it was asked about. */
    private class RecordingEngine(private val lines: List<Line>) : Engine(File("fake"), null) {
        val seen = ArrayList<Triple<String, Int?, Int?>>()

        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
            seen.add(Triple(board.fen(), depth, multipv))
            return lines
        }

        override fun close() = Unit
    }

    /**
     * `test_the_engine_sees_the_position_with_the_move_handed_over`: one call, on the position
     * with the opponent to move and no null move in its history, depth and lines passed
     * through, the caller's board untouched, and the threats the first move of each line.
     */
    @Test
    fun theEngineSeesThePositionWithTheMoveHandedOver() {
        val engine = RecordingEngine(CANNED)
        val board = Board()
        assertEquals(CANNED, threatLines(engine, board, depth = 8, multipv = 2))
        assertEquals(1, engine.seen.size)
        val (fen, depth, multipv) = engine.seen[0]
        assertEquals("the opponent is now to move", "b", fen.split(" ")[1])
        assertEquals("depth and multipv are passed through", 8 to 2, depth to multipv)
        assertEquals(com.sidecoach.chess.STARTING_FEN, board.fen())
        assertEquals(listOf("d8d1", "a7a5"), threatMoves(RecordingEngine(CANNED), Board(), depth = 8, multipv = 2))
    }

    /**
     * `test_no_threat_and_no_engine_call_in_check_or_when_the_game_is_over`: you cannot pass
     * out of check, and a finished game carries no threat to ask about. The finished game is
     * the desktop's stalemate, not in check: a checkmate is caught by the in-check guard
     * first, so the game-over guard was never reached (the parity audit, 2026-09-29).
     */
    @Test
    fun noThreatAndNoEngineCallInCheckOrWhenTheGameIsOver() {
        val inCheck = Board("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1")
        assertTrue("in check, the game not over", inCheck.isCheck() && !inCheck.isGameOver())
        val stalemate = Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        assertTrue("stalemate, not in check", stalemate.isStalemate() && !stalemate.isCheck())
        for (board in listOf(inCheck, stalemate)) {
            val engine = RecordingEngine(CANNED)
            assertEquals(board.fen(), emptyList<Line>(), threatLines(engine, board, depth = 8))
            assertEquals("the engine is never asked: ${board.fen()}", 0, engine.seen.size)
        }
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

        val CANNED = listOf(
            Line(moves = listOf("d8d1"), depth = 8, expectedScore = 0.9),
            Line(moves = listOf("a7a5"), depth = 8, expectedScore = 0.4),
        )
    }
}
