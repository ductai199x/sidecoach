package com.sidecoach.analysis

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.Lane
import com.sidecoach.chess.Board
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * The Stockfish process's life: a closed engine stays closed under a search already under way.
 * The desktop's `tests/test_engine_lifecycle.py`. No sleeps: the test waits for the search's
 * first depth and for the processes' exit, never for a time. (A wedged engine's watchdog is
 * tested with the engine's clock driven by the test.)
 */
@Lane(Lane.Kind.FREE)
@RunWith(AndroidJUnit4::class)
class EngineLifecycleTest {

    /** The engine, and every process it started; [closing], the engine closed as each starts. */
    private class Watched(binary: File, net: File, private val closing: Boolean = false) :
        Engine(binary, net, threads = 1, hashMb = 16, multipv = 3) {
        val started = ArrayList<Process>()

        override fun launch(): Process = super.launch().also {
            started.add(it)
            if (closing) close() // the owner closes it meanwhile
        }
    }

    private fun engine(closing: Boolean = false): Watched? {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val binary = Engine.binaryIn(context.applicationInfo.nativeLibraryDir)
        if (!binary.exists()) return null
        val net = context.filesDir.listFiles { f: File -> f.name.endsWith(".nnue") }?.firstOrNull() ?: return null
        return Watched(binary, net, closing)
    }

    /**
     * `test_an_engine_closed_while_its_process_starts_quits_it`: the room lets go while a
     * crashed engine restarts: the process started as the engine closed is ended, never kept by a
     * closed engine nobody will close again (the audit of 2026-10-05, round 2). The real start,
     * the close made from inside it.
     */
    @Test
    fun anEngineClosedWhileItsProcessStartsEndsIt() {
        val sf = engine(closing = true)
        assumeTrue("the engine or its network is missing", sf != null)
        var raised: Throwable? = null
        try {
            sf!!.analyse(Board(), depth = 2)
        } catch (exc: Throwable) {
            raised = exc
        }
        assertTrue("$raised", raised is EngineUnavailable)
        assertEquals(1, sf!!.started.size)
        assertTrue("a Stockfish nobody closes", sf.started[0].waitFor(30, TimeUnit.SECONDS))
    }

    @Test
    fun aClosedEngineIsNotReopenedByASearchUnderWay() {
        val sf = engine()
        assumeTrue("the engine or its network is missing", sf != null)
        sf!!.open()
        var raised: Throwable? = null
        val searching = CountDownLatch(1) // the first search has reported a depth: it is under way
        val request = Thread {
            try {
                // A request of several searches: the coach's, then the next position's.
                sf.analyse(Board(), depth = 40, timeS = 30.0, carryBest = true, onDepth = { searching.countDown() })
                val board = Board()
                board.push(com.sidecoach.chess.Move.fromUci("e2e4"))
                sf.analyse(board, depth = 12)
            } catch (exc: Throwable) {
                raised = exc
            }
        }
        request.start()
        assertTrue("the search never began", searching.await(60, TimeUnit.SECONDS)) // a safety net
        sf.close() // the session's teardown, under a search
        request.join(60000)
        assertFalse(request.isAlive)
        // Every process the engine started is gone (waited on, not slept for).
        for (p in sf.started) assertTrue("a Stockfish nobody closes", p.waitFor(30, TimeUnit.SECONDS))
        assertTrue("$raised", raised is EngineUnavailable)
        // Opened again by its owner, it works.
        sf.open()
        assertTrue(sf.analyse(Board(), depth = 4, multipv = 1).isNotEmpty())
        sf.close()
    }
}
