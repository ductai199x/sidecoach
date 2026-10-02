package com.sidecoach.analysis

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * The Stockfish process's life: a closed engine stays closed under a search already under way,
 * and a wedged one does not hold its lock forever. The desktop's `tests/test_engine_lifecycle.py`.
 */
@RunWith(AndroidJUnit4::class)
class EngineLifecycleTest {

    /** The engine, and every process it started. */
    private class Watched(binary: File, net: File) : Engine(binary, net, threads = 1, hashMb = 16, multipv = 3) {
        val started = ArrayList<Process>()

        override fun launch(): Process = super.launch().also { started.add(it) }
    }

    private fun engine(): Watched? {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val binary = Engine.binaryIn(context.applicationInfo.nativeLibraryDir)
        if (!binary.exists()) return null
        val net = context.filesDir.listFiles { f: File -> f.name.endsWith(".nnue") }?.firstOrNull() ?: return null
        return Watched(binary, net)
    }

    private fun alive(p: Process): Boolean = runCatching { p.exitValue(); false }.getOrDefault(true)

    @Test
    fun aClosedEngineIsNotReopenedByASearchUnderWay() {
        val sf = engine()
        assumeTrue("the engine or its network is missing", sf != null)
        sf!!.open()
        var raised: Throwable? = null
        val request = Thread {
            try {
                // A request of several searches: the coach's, then the next position's.
                sf.analyse(Board(), depth = 40, timeS = 1.5, carryBest = true)
                val board = Board()
                board.push(com.sidecoach.chess.Move.fromUci("e2e4"))
                sf.analyse(board, depth = 12)
            } catch (exc: Throwable) {
                raised = exc
            }
        }
        request.start()
        Thread.sleep(400)
        sf.close() // the session's teardown
        request.join(30000)
        assertFalse(request.isAlive)
        Thread.sleep(300)
        assertTrue("a Stockfish nobody closes", sf.started.none { alive(it) })
        assertTrue("$raised", raised is EngineUnavailable)
        // Opened again by its owner, it works.
        sf.open()
        assertTrue(sf.analyse(Board(), depth = 4, multipv = 1).isNotEmpty())
        sf.close()
    }

    @Test
    fun aWedgedEngineLetsGoOfItsSearch() {
        val sf = engine()
        assumeTrue("the engine or its network is missing", sf != null)
        sf!!.streamGraceMs = 500
        sf.use {
            sf.open()
            val pid = Engine::class.java.getDeclaredField("enginePid").apply { isAccessible = true }.get(sf) as Int
            var lines: List<com.sidecoach.events.Line>? = null
            val search = Thread { lines = runCatching { sf.analyse(Board(), timeS = 0.5) }.getOrNull() }
            search.start()
            Thread.sleep(100)
            android.os.Process.sendSignal(pid, SIGSTOP) // wedged: alive, saying nothing
            search.join(10000)
            assertFalse("the search held its lock past its time", search.isAlive)
            // The wedged process is replaced and the search answered by the new one.
            assertEquals(2, sf.started.size)
            assertFalse(alive(sf.started[0]))
            assertTrue(lines?.isNotEmpty() == true)
        }
    }

    companion object {
        private const val SIGSTOP = 19
    }
}
