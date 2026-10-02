package com.sidecoach

import com.sidecoach.ui.PanelListen
import java.util.concurrent.Future
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit

/**
 * How long a fake waits on the test (or a test on a thread it started) before it calls the
 * step lost: a bound on a hang, never what a test asserts (the desktop's `tests/fakes.py`
 * HANG_S).
 */
const val HANG_MS = 30_000L

/**
 * A clock the test moves by hand, for code that reads the time or waits on a timer: the
 * desktop's `tests/fakes.py` ManualClock. [seconds] reads it; a timer made on it ([schedule])
 * fires when [advance] or [advanceTo] passes its time -- on the thread that moved the clock, in
 * the order they fall due. Given to one piece of the app in place of its own clock
 * ([PanelListen.sendClock], `PanelVoice.clock`, the Listener's), every other clock stays real.
 */
class ManualClock(start: Double = 1000.0) {
    @Volatile var now: Double = start
        private set

    private val lock = Any()
    private val timers = ArrayList<Timer>()

    /** A timer on this clock: it fires when the clock is moved past [due]. */
    inner class Timer(val due: Double, private val run: () -> Unit) : Future<Unit> {
        @Volatile var cancelled = false
            private set

        @Volatile var fired = false
            private set

        val alive: Boolean get() = !cancelled && !fired

        override fun cancel(mayInterruptIfRunning: Boolean): Boolean {
            if (!alive) return false
            cancelled = true
            return true
        }

        override fun isCancelled() = cancelled

        override fun isDone() = !alive

        override fun get() = Unit

        override fun get(timeout: Long, unit: TimeUnit) = Unit

        internal fun fire() {
            fired = true
            run()
        }
    }

    fun seconds(): Double = now

    fun schedule(delaySeconds: Double, run: () -> Unit): Timer = synchronized(lock) {
        Timer(now + delaySeconds, run).also { timers.add(it) }
    }

    /** The timers started and neither cancelled nor fired. */
    fun pending(): List<Timer> = synchronized(lock) { timers.filter { it.alive } }

    fun advance(seconds: Double) = advanceTo(now + seconds)

    /** Move the clock to [time], firing every timer due by then, each at its own time. */
    fun advanceTo(time: Double) {
        while (true) {
            val timer = synchronized(lock) {
                val due = timers.filter { it.alive && it.due <= time }.minByOrNull { it.due }
                if (due == null) {
                    now = maxOf(now, time)
                    return
                }
                now = maxOf(now, due.due)
                due
            }
            timer.fire()
        }
    }

    /**
     * This clock as the dictation's ([PanelListen.SendClock]): the page's `sends_at` in its
     * milliseconds, the send a timer on it.
     */
    internal fun sendClock(): PanelListen.SendClock = object : PanelListen.SendClock {
        override fun wallMs(): Long = Math.round(now * 1000)

        override fun schedule(delayMs: Long, send: () -> Unit): Future<*> = this@ManualClock.schedule(delayMs / 1000.0, send)
    }
}

/**
 * A microphone's frames, put in by the test, read by the listener: the desktop's
 * `tests/fakes.py` SteppedFrames. When there are none, the listener's wait for the next ends only
 * when the test says ([idle]), never on a clock. So the listener's quiet rounds (its level
 * decaying, a dead input reopened) run exactly when and as often as the test says, and [idle]
 * returning means every frame put in was read and handled. [free] lets the listener go (before
 * it is switched off).
 */
class SteppedFrames : LinkedBlockingQueue<FloatArray>() {
    private val go = Semaphore(0)
    private val waiting = Semaphore(0)
    @Volatile private var parked = false
    @Volatile private var free = false

    override fun poll(timeout: Long, unit: TimeUnit): FloatArray? {
        super.poll()?.let { return it }
        if (free) return null
        waiting.release() // every frame read and handled: the listener waits here
        go.tryAcquire(HANG_MS, TimeUnit.MILLISECONDS)
        return null
    }

    private fun park() {
        check(waiting.tryAcquire(HANG_MS, TimeUnit.MILLISECONDS)) { "the listener never came for a frame" }
    }

    /**
     * Every frame put in read, then [rounds] quiet rounds of the listener, each run to its end:
     * on return the listener waits for the next frame.
     */
    fun idle(rounds: Int = 1) {
        if (!parked) {
            park()
            parked = true
        }
        repeat(rounds) {
            go.release()
            park()
        }
    }

    /** No more stepping: the listener's wait for a frame ends at once (it can stop). */
    fun free() {
        free = true
        go.release()
    }
}
