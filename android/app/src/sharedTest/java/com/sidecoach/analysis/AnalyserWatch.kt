package com.sidecoach.analysis

import com.sidecoach.events.Event
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue

/**
 * How long a test waits on a thread it started (or a fake on the test) before it calls the step
 * lost: a bound on a hang, never what a test asserts (the desktop's `HANG_S`).
 */
internal const val HANG_MS = 30_000L

/**
 * An analyser's workers, watched without a clock (the desktop's `AnalyserWatch`, tests/fakes.py):
 * every word they put and every change of state they wake each other for wakes [until], which
 * returns once its condition holds. [HANG_MS] only bounds a hang; it is never the expected path.
 * Made before the workers start. [also] is told the same, first. The analyser counts what failed
 * and goes on (a failed search, a verdict that could not be made); settled, nothing may have
 * failed unless the test says so ([failuresOk]): a test whose every verdict silently failed would
 * pass otherwise.
 *
 * The condition is read outside the watch's own lock: it takes the analyser's, under which the
 * workers tell the watch -- read inside, the two locks were taken in both orders.
 */
internal class AnalyserWatch(
    val analyser: Analyser,
    private val also: Analyser.Watch? = null,
    private val failuresOk: Boolean = false,
) {
    private val lock = Object()
    private var pokes = 0L

    init {
        analyser.watch = object : Analyser.Watch {
            override fun said(event: Event) {
                also?.said(event)
                poke()
            }

            override fun changed() {
                also?.changed()
                poke()
            }

            override fun withdrew(gameId: String, ply: Int) {
                also?.withdrew(gameId, ply)
                poke()
            }
        }
        // A gated fake's search beginning or stopped wakes the watch too: a test waits on it.
        for (engine in analyser.engines) (engine as? GatedEngine)?.poke = ::poke
    }

    fun poke() {
        synchronized(lock) {
            pokes += 1
            lock.notifyAll()
        }
    }

    /** Until [condition] holds, woken by the workers' own steps. */
    fun until(what: String = "the condition", condition: () -> Boolean) {
        val deadline = System.nanoTime() + HANG_MS * 1_000_000
        while (true) {
            val seen = synchronized(lock) { pokes }
            if (condition()) return
            synchronized(lock) {
                while (pokes == seen) {
                    val left = (deadline - System.nanoTime()) / 1_000_000
                    assertTrue("never came: $what", left > 0)
                    lock.wait(left)
                }
            }
        }
    }

    /**
     * Everything owed to the game answered. A word is put before the state that says it was told,
     * so none can come after `pending` is 0; what was said stays queued for whoever drains it (the
     * glue's `drain`).
     */
    fun idle() {
        until("the analyser settled") { analyser.pending == 0 }
        if (!failuresOk) assertEquals("${analyser.lastError}", 0, analyser.failures)
    }

    /**
     * Settled ([idle]), and every search done that the analyser still wants made -- the
     * candidates' own, which nothing owed waits on.
     */
    fun quiet() {
        idle()
        until("the analyser's searches done") {
            synchronized(analyser.lock) { analyser.claimed.isEmpty() && analyser.pick() == null }
        }
    }

    /** Everything owed answered ([idle]), and what the workers said, polled. */
    fun settled(): List<Event> {
        idle()
        return analyser.poll()
    }
}
