package com.sidecoach.analysis

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Stockfish, built for the phone and driven over UCI.
 *
 * Not a parity test: a search is not a pure function of the position, and two builds of the
 * same engine on two architectures need not agree on the twelfth-best move. What must hold
 * is that the engine starts on the device, loads the network the app gives it, answers in
 * the protocol the desktop's wrapper expects, and sees what any engine must see -- the mate,
 * the hanging queen, the side that is winning.
 *
 * It needs the network: `tools/fetch_stockfish.py --net` then `tools/push_net.ps1`. Without
 * one the test says so and skips, rather than failing for the wrong reason.
 */
@RunWith(AndroidJUnit4::class)
class EngineTest {

    private fun engine(): Engine? {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val binary = Engine.binaryIn(context.applicationInfo.nativeLibraryDir)
        if (!binary.exists()) {
            Log.w(TAG, "no engine at $binary")
            return null
        }
        val net = context.filesDir.listFiles { f: File -> f.name.endsWith(".nnue") }?.firstOrNull()
        if (net == null) {
            Log.w(TAG, "no network in ${context.filesDir}; run tools/push_net.ps1")
            return null
        }
        return Engine(binary, net, threads = 2, hashMb = 64, multipv = 3)
    }

    @Test
    fun theEngineStartsAndNamesItself() {
        val engine = engine()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.use {
            it.open()
            Log.i(TAG, "engine: ${it.name}")
            assertTrue("the engine did not name itself", it.name.contains("Stockfish"))
        }
    }

    @Test
    fun itSeesWhatAnyEngineMustSee() {
        val engine = engine()
        assumeTrue("the engine or its network is missing", engine != null)
        val cases = listOf(
            // Mate in one: the back rank.
            Triple("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8", "mate in one"),
            // A queen hanging on d5 with nothing behind it.
            Triple("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1", "d1d5", "the queen"),
            // The Italian: no blunder, just a sane developing move.
            Triple(
                "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                null,
                "the Italian",
            ),
        )
        engine!!.use { sf ->
            for ((fen, best, what) in cases) {
                val board = Board(fen)
                val began = System.currentTimeMillis()
                val lines = sf.analyse(board, depth = 14)
                val took = System.currentTimeMillis() - began
                assertTrue("$what: no lines", lines.isNotEmpty())
                val top = lines[0]
                Log.i(
                    TAG,
                    "$what: ${sanLine(board, top.moves, 6)} — depth ${top.depth}, " +
                        "cp ${top.scoreCp} mate ${top.mate} wdl ${top.wdl}, " +
                        "expected ${"%.3f".format(top.expectedScore)}, $took ms, " +
                        "${lines.size} lines",
                )
                if (best != null) {
                    assertEquals("$what: the wrong move", best, top.moves.first())
                }
                assertTrue("$what: the depth was not reported", top.depth > 0)
            }
        }
    }

    @Test
    fun aPausedEngineHoldsStillAndStillSearchesItsTime() {
        val engine = engine()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.use { sf ->
            sf.open()
            val board = Board("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
            var lines: List<com.sidecoach.events.Line> = emptyList()
            val began = System.currentTimeMillis()
            val search = kotlin.concurrent.thread { lines = sf.analyse(board, timeS = 1.0) }
            Thread.sleep(300)
            sf.pause()
            Thread.sleep(100)
            val held = sf.cpuMs()
            Thread.sleep(1000)
            val stillHeld = sf.cpuMs()
            sf.resume()
            search.join()
            val took = System.currentTimeMillis() - began
            val after = sf.cpuMs()
            Log.i(TAG, "paused search: $took ms, engine cpu $held -> $stillHeld -> $after ms, depth ${lines.firstOrNull()?.depth}")
            assertTrue("the engine ran while paused: $held -> $stillHeld ms", stillHeld - held <= 20)
            // A second of search, less the pause: the pause does not count against it.
            assertTrue("the pause cut the search short: $took ms", took >= 2000)
            assertTrue("no lines", lines.isNotEmpty())
            // The same process ran on, its hash with it: a restarted engine's time starts at 0.
            assertTrue("the engine was restarted: $stillHeld -> $after ms", after > stillHeld)
        }
    }

    @Test
    fun theBestLineIsCarriedOnForTheCoach() {
        val engine = engine()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.use { sf ->
            // `carry_best`: the best line kept to MIN_BEST_PLIES, searched on from its end when
            // Stockfish's own stops short; the other lines cut as always; a verdict's search
            // (no `carry_best`) cut like the rest.
            val board = Board("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
            val carried = sf.analyse(board, depth = 10, pvPlies = 4, carryBest = true)
            val plain = sf.analyse(board, depth = 10, pvPlies = 4)
            assertTrue("no lines", carried.isNotEmpty() && plain.isNotEmpty())
            val best = carried[0].moves
            Log.i(TAG, "carried best line: ${sanLine(board, best)} (${best.size} plies); plain ${plain[0].moves.size}")
            // Every move of the carried line is legal where it stands.
            assertEquals("an illegal move in the carried line", best.size, sanLine(board, best, numbered = false).split(" ").size)
            val end = board.copy(withStack = false)
            for (uci in best) end.push(com.sidecoach.chess.Move.fromUci(uci))
            assertTrue(
                "the best line stopped at ${best.size} plies",
                best.size == Engine.MIN_BEST_PLIES || end.isGameOver(claimDraw = true),
            )
            for (line in carried.drop(1)) assertTrue("another line was carried on", line.moves.size <= 4)
            assertTrue("a verdict's line was carried on", plain[0].moves.size <= 4)
        }
    }

    @Test
    fun aFinishedPositionHasNoLines() {
        val engine = engine()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.use {
            // Checkmate: there is nothing to play.
            assertTrue(it.analyse(Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"), depth = 8).isEmpty())
            // Stalemate, likewise.
            assertTrue(it.analyse(Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"), depth = 8).isEmpty())
        }
    }

    @Test
    fun theMoverAlwaysSeesItsOwnScore() {
        val engine = engine()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.use { sf ->
            // The same position, a rook up for White, asked with either side to move: the
            // expected score is always the mover's own, so the two must be near opposites.
            val white = sf.analyse(Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1"), depth = 12)
            val black = sf.analyse(Board("4k3/8/8/8/8/8/8/R3K3 b - - 0 1"), depth = 12)
            assertTrue("no lines", white.isNotEmpty() && black.isNotEmpty())
            Log.i(
                TAG,
                "a rook up: white to move ${"%.3f".format(white[0].expectedScore)}, " +
                    "black to move ${"%.3f".format(black[0].expectedScore)}",
            )
            assertTrue("white is not winning", white[0].expectedScore > 0.7)
            assertTrue("black is not losing", black[0].expectedScore < 0.3)
        }
    }

    companion object {
        private const val TAG = "sidecoach.engine"
    }
}
