package com.sidecoach.analysis

import com.sidecoach.events.Event
import org.junit.Assert.assertTrue

/**
 * How long a test waits on a thread it started (or a fake on the test) before it calls the step
 * lost: a bound on a hang, never what a test asserts (the desktop's `HANG_S`).
 */
internal const val HANG_MS = 30_000L

/**
 * An analyser's worker, watched without a clock (the desktop's `AnalyserWatch`, tests/
 * test_analysis.py): every word it says, every request it settles, every step it looks ahead,
 * every takeback it is told of and its stop put in by `close` wakes [until], which returns once its condition holds. [HANG_MS]
 * only bounds a hang; it is never the expected path. Made before the worker starts. [also] is
 * told the same, first (a test's own log of what the worker did, in order).
 */
internal class AnalyserWatch(val analyser: Analyser, private val also: Analyser.Watch? = null) {
    private val lock = Object()

    init {
        analyser.watch = object : Analyser.Watch {
            override fun said(event: Event) {
                also?.said(event)
                poke()
            }

            override fun settled() {
                also?.settled()
                poke()
            }

            override fun lookedAhead() {
                also?.lookedAhead()
                poke()
            }

            override fun withdrew(gameId: String, ply: Int) {
                also?.withdrew(gameId, ply)
                poke()
            }

            override fun closing() {
                also?.closing()
                poke()
            }
        }
    }

    fun poke() {
        synchronized(lock) { lock.notifyAll() }
    }

    /** Until [condition] holds, woken by the worker's own steps. */
    fun until(what: String = "the condition", condition: () -> Boolean) {
        val deadline = System.nanoTime() + HANG_MS * 1_000_000
        synchronized(lock) {
            while (!condition()) {
                val left = (deadline - System.nanoTime()) / 1_000_000
                assertTrue("never came: $what", left > 0)
                lock.wait(left)
            }
        }
    }

    /**
     * Every request answered. A word is put before its request settles, so none can come after
     * `pending` is 0; what was said stays queued for whoever drains it (the glue's `drain`).
     */
    fun idle() = until("the analyser settled") { analyser.pending == 0 }

    /** Every request answered ([idle]), and what the worker said, polled. */
    fun settled(): List<Event> {
        idle()
        return analyser.poll()
    }
}
