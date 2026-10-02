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
import java.io.FilterInputStream
import java.io.FilterOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

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

    /**
     * The engine, and every process it started, so a test can end one behind its back; every
     * command written to the process and every line read from it noted in [log], in order with
     * the test's own marks ([mark]); and, while [manual] is set, the clock a search's time is
     * kept on ([Engine.now]) is the test's, every reading of it by the engine's time keeper
     * counted ([readings], [untilRead]).
     */
    private class Watched(binary: File, net: File) : Engine(binary, net, threads = 2, hashMb = 64, multipv = 3) {
        val started = ArrayList<Process>()
        val log: MutableList<String> = java.util.Collections.synchronizedList(ArrayList())
        @Volatile var manual: Long? = null
        private val read = Object()
        var readings = 0 // under [read]
            private set

        override fun launch(): Process = Spied(super.launch(), log).also { started.add(it) }

        override fun now(): Long {
            val time = manual ?: return super.now()
            if (Thread.currentThread().name == "engine-clock") {
                synchronized(read) {
                    readings += 1
                    read.notifyAll()
                }
            }
            return time
        }

        fun mark(what: String) = log.add("! $what")

        /** Until the time keeper has read the clock more than [count] times. */
        fun untilRead(count: Int) = synchronized(read) {
            val deadline = System.nanoTime() + HANG_MS * 1_000_000
            while (readings <= count) {
                val left = (deadline - System.nanoTime()) / 1_000_000
                assertTrue("the engine's time keeper never read the clock", left > 0)
                read.wait(left)
            }
        }

        /** The process's state letter from /proc (`T`: stopped by a signal). */
        fun state(): Char {
            val stat = File("/proc/${pidOf(started.last())}/stat").readText()
            return stat[stat.lastIndexOf(')') + 2]
        }

        /**
         * Until the process stands still ([pause] sends SIGSTOP, which the kernel carries out as
         * the process is next scheduled): the process's own state, read until it says so, no
         * clock -- [HANG_MS] bounds a hang.
         */
        fun untilStopped() {
            val deadline = System.nanoTime() + HANG_MS * 1_000_000
            while (state() != 'T') {
                assertTrue("the paused engine never stood still: ${state()}", System.nanoTime() < deadline)
                Thread.yield()
            }
        }

        fun indexOf(entry: String): Int = synchronized(log) { log.indexOfFirst { it == entry || it.startsWith("$entry ") } }

        fun count(entry: String): Int = synchronized(log) { log.count { it == entry || it.startsWith("$entry ") } }
    }

    /**
     * The engine's process, every command written to it (`> stop`) and every line read from it
     * (`< bestmove e2e4`) noted in [log] as it passes. Its `pid` is the real one's: the engine
     * reads a process's `pid` to pause it.
     */
    private class Spied(private val real: Process, private val log: MutableList<String>) : Process() {
        @JvmField val pid: Int = pidOf(real)

        private val input = object : FilterInputStream(real.inputStream) {
            private val line = StringBuilder()

            private fun seen(bytes: ByteArray, off: Int, len: Int) {
                for (i in off until off + len) {
                    val c = bytes[i].toInt().toChar()
                    if (c == '\n') {
                        log.add("< $line")
                        line.setLength(0)
                    } else if (c != '\r') {
                        line.append(c)
                    }
                }
            }

            override fun read(): Int = super.read().also { if (it >= 0) seen(byteArrayOf(it.toByte()), 0, 1) }

            override fun read(b: ByteArray, off: Int, len: Int): Int = super.read(b, off, len).also { if (it > 0) seen(b, off, it) }
        }

        private val output = object : FilterOutputStream(real.outputStream) {
            private val line = StringBuilder()

            override fun write(b: Int) {
                out.write(b)
                val c = b.toChar()
                if (c == '\n') {
                    log.add("> $line")
                    line.setLength(0)
                } else {
                    line.append(c)
                }
            }

            override fun write(b: ByteArray, off: Int, len: Int) {
                for (i in off until off + len) write(b[i].toInt() and 0xff)
            }
        }

        override fun getOutputStream(): OutputStream = output

        override fun getInputStream(): InputStream = input

        override fun getErrorStream(): InputStream = real.errorStream

        override fun waitFor(): Int = real.waitFor()

        override fun exitValue(): Int = real.exitValue()

        override fun destroy() = real.destroy()
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

    /**
     * `test_no_single_position_may_hold_up_a_game`: a search that would take minutes is cut at
     * the cap. Counted, not timed (it was: under five seconds by the stopwatch): on a clock the
     * test moves, no `stop` goes to the engine before the cap has run, and one does when it has,
     * which ends the search short of the depth asked for.
     */
    @Test
    fun noSinglePositionMayHoldUpAGame() {
        val engine = shared()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.let { sf ->
            // A knight endgame that takes minutes to reach depth 14 with four lines.
            val slow = Board("8/k7/2p3NP/8/1P1K2P1/2P5/3n4/8 b - - 2 50")
            val cap = sf.maxSeconds
            sf.maxSeconds = 0.5
            sf.manual = 1_000_000L
            val stops = sf.count("> stop")
            try {
                var lines: List<com.sidecoach.events.Line> = emptyList()
                val searching = CountDownLatch(1)
                val search = kotlin.concurrent.thread {
                    lines = sf.analyse(slow, depth = 40, multipv = 4, onDepth = { searching.countDown() })
                }
                assertTrue("the search never began", searching.await(HANG_MS, TimeUnit.MILLISECONDS))
                // The time keeper has read the clock and let go of it (the pause waits for it):
                // the cap has not run, and nothing was stopped.
                val reads = sf.readings
                sf.pause()
                sf.resume()
                sf.untilRead(reads)
                sf.pause()
                sf.resume()
                assertEquals("stopped before the cap had run", stops, sf.count("> stop"))
                sf.manual = sf.manual!! + 500
                sf.mark("the cap run")
                sf.pause()
                sf.resume() // wakes the time keeper
                search.join(HANG_MS)
                assertTrue("the cap did not hold", !search.isAlive)
                assertEquals("one stop, the cap's", stops + 1, sf.count("> stop"))
                assertTrue("no lines", lines.isNotEmpty())
                assertTrue("the depth asked for was reached: ${lines[0].depth}", lines[0].depth < 40)
            } finally {
                sf.manual = null
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

    /**
     * A search of a second, paused an hour: the process stands still the whole time, and the
     * pause does not count against the search's second. Counted, not timed (it was: a second's
     * pause, the CPU read before and after it, and the search at least two seconds long): the
     * search's time is kept on a clock the test moves, and what the engine is told is read from
     * the pipe in order -- no `stop` until the clock has run the search's whole second outside
     * the pause.
     */
    @Test
    fun aPausedEngineHoldsStillAndStillSearchesItsTime() {
        val engine = engine()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.use { sf ->
            sf.open()
            sf.manual = 1_000_000L
            val board = Board("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
            var lines: List<com.sidecoach.events.Line> = emptyList()
            val searching = CountDownLatch(1)
            val search = kotlin.concurrent.thread { lines = sf.analyse(board, timeS = 1.0, onDepth = { searching.countDown() }) }
            assertTrue("the search never began", searching.await(HANG_MS, TimeUnit.MILLISECONDS))
            sf.pause()
            sf.untilStopped()
            val held = sf.cpuMs()
            sf.manual = sf.manual!! + 3_600_000 // an hour paused
            val stillHeld = sf.cpuMs()
            assertEquals("the engine ran while paused", 'T', sf.state())
            // Resumed: once the time keeper has read the clock again and let go of it (the pause
            // and resume after wait for it), it has decided -- and no `stop` was sent.
            val reads = sf.readings
            sf.resume()
            sf.untilRead(reads)
            sf.pause()
            sf.resume()
            assertEquals("the pause was counted against the search", -1, sf.indexOf("> stop"))
            assertTrue("the search ended early", search.isAlive)
            // The search's second run out: now it is stopped.
            sf.manual = sf.manual!! + 1000
            sf.mark("the second run")
            sf.pause()
            sf.resume() // wakes the time keeper
            search.join(HANG_MS)
            assertTrue("the search never ended", !search.isAlive)
            Log.i(TAG, "paused search: engine cpu $held -> $stillHeld ms, depth ${lines.firstOrNull()?.depth}")
            assertEquals("the engine ran while paused: $held -> $stillHeld ms", held, stillHeld)
            assertEquals("one stop", 1, sf.count("> stop"))
            assertTrue("stopped before its second had run", sf.indexOf("> stop") > sf.indexOf("! the second run"))
            assertTrue("no lines", lines.isNotEmpty())
            // The same process ran on, its hash with it.
            assertEquals("the engine was restarted", 1, sf.started.size)
        }
    }

    /**
     * `test_the_engine_stops_a_search_it_is_interrupted_in` (desktop 9c0774d): a search inside
     * `interruptible` ends on `interrupt`, from another thread, with [Interrupted]; the engine
     * answers as before afterwards. A search to depth 60 from the start does not end by itself:
     * it ends because `stop` went to the engine and its `bestmove` came back (read from the pipe
     * in order; it was timed: within the second).
     */
    @Test
    fun theEngineStopsASearchItIsInterruptedIn() {
        val engine = engine()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.use { sf ->
            sf.open()  // as the desktop's fixture holds it: running, its network loaded
            val (worker, raised) = searchToDepth60(sf)
            sf.mark("interrupt")
            sf.interrupt()
            worker.join(HANG_MS)
            assertTrue("the search ran on", !worker.isAlive)
            assertTrue("$raised", raised.isNotEmpty() && raised[0] is Interrupted)
            val stop = sf.indexOf("> stop")
            val answer = sf.indexOf("< bestmove")
            Log.i(TAG, "interrupted: stop at ${sf.indexOf("! interrupt")} -> $stop, bestmove at $answer, ended at ${sf.indexOf("! ended")}")
            assertTrue("no stop after the interrupt", stop > sf.indexOf("! interrupt"))
            assertTrue("the search ended before the engine answered the stop", answer in (stop + 1) until sf.indexOf("! ended"))
            // The engine answers as before afterwards.
            assertTrue(sf.analyse(Board(), depth = 6, multipv = 1).isNotEmpty())
        }
    }

    /**
     * The phone's own: the voice holds the engine still ([Engine.pause]) while a move comes and
     * interrupts the look ahead. The interrupt returns while the engine still stands still --
     * the capture thread never waits on a stopped process -- and the search ends once the engine
     * runs on and answers the `stop` (read from the pipe in order; it was timed: the interrupt
     * under 100 ms, the search alive 300 ms later, ended within the second of the resume).
     */
    @Test
    fun anInterruptWhilePausedEndsTheSearchOnceResumed() {
        val engine = engine()
        assumeTrue("the engine or its network is missing", engine != null)
        engine!!.use { sf ->
            sf.open()
            val (worker, raised) = searchToDepth60(sf)
            sf.pause()
            sf.untilStopped()
            // From another thread, as the capture thread's: it returns while the engine stands still.
            val interrupter = kotlin.concurrent.thread { sf.interrupt() }
            interrupter.join(HANG_MS)
            assertTrue("the interrupt waited on the paused engine", !interrupter.isAlive)
            assertEquals("the engine ran before the interrupt returned", 'T', sf.state())
            assertTrue("no stop went to the engine", sf.indexOf("> stop") >= 0)
            assertTrue("the search ended while the engine was held", worker.isAlive)
            sf.mark("resume")
            sf.resume()
            worker.join(HANG_MS)
            assertTrue("the search ran on after the resume", !worker.isAlive)
            assertTrue("$raised", raised.isNotEmpty() && raised[0] is Interrupted)
            val answer = sf.indexOf("< bestmove")
            Log.i(TAG, "interrupted while paused: resumed at ${sf.indexOf("! resume")}, bestmove at $answer, ended at ${sf.indexOf("! ended")}")
            // The engine answered the stop only once it ran on, and the search ended on that answer.
            assertTrue("answered while held", answer > sf.indexOf("! resume"))
            assertTrue("the search ended before the engine answered", answer < sf.indexOf("! ended"))
            assertTrue(sf.analyse(Board(), depth = 6, multipv = 1).isNotEmpty())
        }
    }

    /**
     * A search to depth 60 from the start, inside `interruptible`, on its own thread: it never
     * ends by itself. Returned once its first depth is in (it is under way, not slept on); its
     * end marked in the engine's log (`! ended`), and what it raised.
     */
    private fun searchToDepth60(sf: Watched): Pair<Thread, List<Throwable>> {
        val raised = java.util.Collections.synchronizedList(ArrayList<Throwable>())
        val searching = CountDownLatch(1)
        val worker = kotlin.concurrent.thread {
            try {
                sf.interruptible { sf.analyse(Board(), depth = 60, multipv = 1, onDepth = { searching.countDown() }) }
            } catch (exc: Throwable) {
                raised.add(exc)
            } finally {
                sf.mark("ended")
            }
        }
        assertTrue("the search never began", searching.await(HANG_MS, TimeUnit.MILLISECONDS))
        return worker to raised
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

        /** A process's id, read as the engine reads it (`Engine.processId`). */
        private fun pidOf(process: Process): Int {
            var type: Class<*>? = process.javaClass
            while (type != null) {
                runCatching {
                    val field = type.getDeclaredField("pid")
                    field.isAccessible = true
                    return field.getInt(process)
                }
                type = type.superclass
            }
            throw AssertionError("no pid in $process")
        }

        /** Black to move at queen odds: winning, with no mate in sight to hide the numbers. */
        private const val BLACK_WINS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1"
    }
}
