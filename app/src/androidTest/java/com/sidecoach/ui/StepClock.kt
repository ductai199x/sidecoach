package com.sidecoach.ui

import android.app.Instrumentation

/**
 * The caption box's clock for a test ([Captions.Clock]): time stands still until the test moves
 * it, and every step that comes due on the way runs then, on the main thread, in the order of
 * its time (two at the same time in the order they were asked for, as a Handler runs them). A
 * fade or a timeout is then checked at its shipped length to the millisecond -- not yet one
 * millisecond before, done at it -- and never waited out.
 */
internal class StepClock(private val instrumentation: Instrumentation) : Captions.Clock {
    private class Step(val at: Long, val order: Long, val step: Runnable)

    private var now = 0L
    private var asked = 0L
    private val due = ArrayList<Step>()

    @Synchronized
    override fun now(): Long = now

    @Synchronized
    override fun post(step: Runnable, delayMs: Long) {
        due += Step(now + maxOf(0L, delayMs), asked++, step)
    }

    @Synchronized
    override fun cancel(step: Runnable) {
        due.removeAll { it.step === step }
    }

    /** How many steps wait for their time. */
    @Synchronized
    fun waiting(): Int = due.size

    /**
     * Time moved on by [ms]: each step due by then run on the main thread at its own time, the
     * steps those steps ask for too if they come due in the span. The main thread is let finish
     * what it was given first, so a step a posted call is about to ask for is asked for first.
     */
    fun advance(ms: Long) {
        instrumentation.waitForIdleSync()
        val until = synchronized(this) { now + ms }
        while (step(until)) Unit
        synchronized(this) { now = until }
    }

    /**
     * Time moved on step by step, at most [ms], until [holds] does: the time it held from (now,
     * if it holds already), the clock stopped there; null if it never did. For a moment the box
     * decides on its own steps -- the words read through -- that a length is then counted from.
     */
    fun advanceUntil(ms: Long, holds: () -> Boolean): Long? {
        instrumentation.waitForIdleSync()
        val until = synchronized(this) { now + ms }
        while (true) {
            if (holds()) return synchronized(this) { now }
            if (!step(until)) break
        }
        synchronized(this) { now = until }
        return if (holds()) until else null
    }

    /** The next step due by [until] run on the main thread at its time; false if none is due. */
    private fun step(until: Long): Boolean {
        val next = synchronized(this) {
            due.filter { it.at <= until }.minWithOrNull(compareBy<Step>({ it.at }, { it.order }))?.also {
                due.remove(it)
                now = it.at
            }
        } ?: return false
        instrumentation.runOnMainSync { next.step.run() }
        return true
    }
}
