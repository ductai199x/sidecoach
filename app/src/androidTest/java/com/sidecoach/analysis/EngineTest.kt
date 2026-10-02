package com.sidecoach.analysis

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.Lane
import com.sidecoach.chess.Board
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.AfterClass
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
@Lane(Lane.Kind.FREE)
@RunWith(AndroidJUnit4::class)
class EngineTest {

    /** The engine, and every process it started, so a test can end one behind its back. */
    private class Watched(binary: File, net: File) : Engine(binary, net, threads = 2, hashMb = 64, multipv = 3) {
        val started = ArrayList<Process>()

        override fun launch(): Process = super.launch().also { started.add(it) }
    }

    private fun engine(): Watched? = Companion.engine()

    @Test
    fun theEngineStartsAndNamesItself() {
        val engine = shared()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.let {
            it.open()
            Log.i(TAG, "engine: ${it.name}")
            assertTrue("the engine did not name itself", it.name.contains("Stockfish"))
        }
    }

    @Test
    fun itSeesWhatAnyEngineMustSee() {
        val engine = shared()
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
        engine!!.let { sf ->
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
                if (best == "a1a8") {
                    // A mate is a distance and a certain score (`test_a_real_game_is_assessed_end_to_end`).
                    assertEquals("$what: the mate distance", 1, top.mate)
                    assertEquals("$what: a mate is a certain score", 1.0, top.expectedScore, 0.0)
                }
            }
        }
    }

    /** `test_the_engine_is_restarted_when_it_dies`: a dead engine is restarted, not the end of a session. */
    @Test
    fun theEngineIsRestartedWhenItDies() {
        val engine = engine()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.use { sf ->
            sf.open()
            val first = sf.started.single()
            first.destroy()
            first.waitFor()
            val lines = sf.analyse(Board(), depth = 10)
            assertTrue("no lines from the restarted engine", lines.isNotEmpty())
            assertEquals("the engine was not started again", 2, sf.started.size)
        }
    }

    /** `test_no_single_position_may_hold_up_a_game`: a search that would take minutes is cut at the cap. */
    @Test
    fun noSinglePositionMayHoldUpAGame() {
        val engine = shared()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.let { sf ->
            // A knight endgame that takes minutes to reach depth 14 with four lines.
            val slow = Board("8/k7/2p3NP/8/1P1K2P1/2P5/3n4/8 b - - 2 50")
            val cap = sf.maxSeconds
            sf.maxSeconds = 0.5
            try {
                val began = System.currentTimeMillis()
                val lines = sf.analyse(slow, depth = 40, multipv = 4)
                val took = System.currentTimeMillis() - began
                assertTrue("no lines", lines.isNotEmpty())
                assertTrue("the cap did not hold: $took ms", took < 5000)
                assertTrue("the depth asked for was reached: ${lines[0].depth}", lines[0].depth < 40)
            } finally {
                sf.maxSeconds = cap
            }
        }
    }

    /** `test_multipv_is_ordered_best_first_and_bounded_by_the_legal_moves`. */
    @Test
    fun theLinesAreBestFirstAndNoMoreThanTheLegalMoves() {
        val engine = shared()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.let { sf ->
            val lines = sf.analyse(Board(), depth = 10, multipv = 4)
            assertEquals(4, lines.size)
            assertEquals(lines.map { it.expectedScore }.sortedDescending(), lines.map { it.expectedScore })
            assertEquals(1, sf.analyse(Board("7k/8/8/8/8/8/6q1/7K w - - 0 1"), depth = 10, multipv = 4).size)
        }
    }

    /**
     * The engine's own numbers when asked, (W + D/2) / 1000, above the human curve's
     * (`test_lines_come_back_from_the_movers_point_of_view_on_the_curve_asked_for`).
     */
    @Test
    fun theEngineModelCanBeAskedForInstead() {
        val engine = shared()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.let { sf ->
            val board = Board(BLACK_WINS)
            val human = sf.analyse(board, depth = 10)[0]
            assertTrue(sf.scoreModel == "human" && human.expectedScore < 1.0)
            sf.scoreModel = "engine"
            val strict = try {
                sf.analyse(board, depth = 10)[0]
            } finally {
                sf.scoreModel = "human"
            }
            val wdl = strict.wdl!!
            assertEquals((wdl.first + wdl.second / 2.0) / 1000, strict.expectedScore, 1e-9)
            assertTrue("${strict.expectedScore} <= ${human.expectedScore}", strict.expectedScore > human.expectedScore)
        }
    }

    /** `test_san_line_reads_like_a_move_list`: numbered from whoever is to move; a line that does not fit stops. */
    @Test
    fun aLineReadsLikeAMoveList() {
        val board = Board()
        assertEquals("1. e4 e5 2. Nf3", sanLine(board, listOf("e2e4", "e7e5", "g1f3")))
        board.push(com.sidecoach.chess.Move.fromUci("e2e4"))
        assertEquals("1... e5 2. Nf3", sanLine(board, listOf("e7e5", "g1f3")))
        assertEquals("a line that does not fit the board stops", "", sanLine(board, listOf("e2e4")))
        assertEquals("", sanLine(board, listOf("nonsense")))
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

    /**
     * `test_the_engine_stops_a_search_it_is_interrupted_in` (desktop 9c0774d): a search inside
     * `interruptible` ends within the second of `interrupt`, from another thread, with
     * [Interrupted]; the engine answers as before afterwards.
     */
    @Test
    fun theEngineStopsASearchItIsInterruptedIn() {
        val engine = engine()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.use { sf ->
            sf.open()  // as the desktop's fixture holds it: running, its network loaded
            val raised = java.util.Collections.synchronizedList(ArrayList<Throwable>())
            val worker = kotlin.concurrent.thread {
                try {
                    sf.interruptible { sf.analyse(Board(), depth = 60, multipv = 1) }
                } catch (exc: Throwable) {
                    raised.add(exc)
                }
            }
            Thread.sleep(300)
            val t = android.os.SystemClock.elapsedRealtime()
            sf.interrupt()
            worker.join(5000)
            val took = android.os.SystemClock.elapsedRealtime() - t
            Log.i(TAG, "interrupted: the search ended $took ms after the interrupt")
            assertTrue("the search ran on: $took ms", !worker.isAlive && took < 1000)
            assertTrue("$raised", raised.isNotEmpty() && raised[0] is Interrupted)
            // The engine answers as before afterwards.
            assertTrue(sf.analyse(Board(), depth = 6, multipv = 1).isNotEmpty())
        }
    }

    /**
     * The phone's own: the voice holds the engine still ([Engine.pause]) while a move comes and
     * interrupts the look ahead. The interrupt returns at once -- the capture thread never
     * waits on a stopped process -- and the search ends the moment the engine runs on.
     */
    @Test
    fun anInterruptWhilePausedEndsTheSearchOnceResumed() {
        val engine = engine()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.use { sf ->
            sf.open()
            val raised = java.util.Collections.synchronizedList(ArrayList<Throwable>())
            val worker = kotlin.concurrent.thread {
                try {
                    sf.interruptible { sf.analyse(Board(), depth = 60, multipv = 1) }
                } catch (exc: Throwable) {
                    raised.add(exc)
                }
            }
            Thread.sleep(300)
            sf.pause()
            val t = android.os.SystemClock.elapsedRealtime()
            sf.interrupt()
            val waited = android.os.SystemClock.elapsedRealtime() - t
            assertTrue("the interrupt waited on the paused engine: $waited ms", waited < 100)
            Thread.sleep(300)
            assertTrue("the search ended while the engine was held", worker.isAlive)
            val resumed = android.os.SystemClock.elapsedRealtime()
            sf.resume()
            worker.join(5000)
            val took = android.os.SystemClock.elapsedRealtime() - resumed
            Log.i(TAG, "interrupted while paused: the search ended $took ms after the resume")
            assertTrue("the search ran on after the resume: $took ms", !worker.isAlive && took < 1000)
            assertTrue("$raised", raised.isNotEmpty() && raised[0] is Interrupted)
            assertTrue(sf.analyse(Board(), depth = 6, multipv = 1).isNotEmpty())
        }
    }

    @Test
    fun theBestLineIsCarriedOnForTheCoach() {
        val engine = shared()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.let { sf ->
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
        val engine = shared()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.let {
            // Checkmate: there is nothing to play.
            assertTrue(it.analyse(Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"), depth = 8).isEmpty())
            // Stalemate, likewise.
            assertTrue(it.analyse(Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"), depth = 8).isEmpty())
        }
    }

    @Test
    fun theMoverAlwaysSeesItsOwnScore() {
        val engine = shared()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.let { sf ->
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
            // `test_lines_come_back_from_the_movers_point_of_view_on_the_curve_asked_for`: the
            // engine's own numbers, and the centipawns from the mover's side too, at the depth
            // asked for.
            for ((lines, sign) in listOf(white to 1, black to -1)) {
                val top = lines[0]
                assertEquals("the win-draw-loss numbers", 1000, top.wdl!!.toList().sum())
                assertTrue("the centipawns are not the mover's: ${top.scoreCp}", top.scoreCp == null || top.scoreCp!! * sign > 0)
                assertTrue("the depth: ${top.depth}", top.depth >= 12)
            }
        }
    }

    companion object {
        private const val TAG = "sidecoach.engine"

        private fun engine(): Watched? {
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
            return Watched(binary, net)
        }

        /**
         * One engine for the tests that only ask it things, started once for the class: each
         * start loads the 98 MB network, nearly half a second of every such test. The tests that
         * end, pause or interrupt an engine start their own, and a test that changes a setting
         * puts it back.
         */
        private var kept: Watched? = null

        private fun shared(): Watched? = kept ?: engine()?.also { kept = it }

        @JvmStatic
        @AfterClass
        fun closeShared() {
            kept?.close()
            kept = null
        }

        /** Black to move at queen odds: winning, with no mate in sight to hide the numbers. */
        private const val BLACK_WINS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1"
    }
}
