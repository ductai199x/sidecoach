package com.sidecoach

import android.os.Bundle
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.runner.Description
import org.junit.runner.Request
import org.junit.runner.Runner
import org.junit.runner.notification.Failure
import org.junit.runner.notification.RunListener
import org.junit.runner.notification.RunNotifier
import org.junit.runners.ParentRunner
import java.io.File
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicLong

/**
 * Which lane a test class runs in ([Lanes]). A class that names none runs in [Kind.APP], the
 * lane that is safe for anything.
 */
@Target(AnnotationTarget.CLASS)
@Retention(AnnotationRetention.RUNTIME)
annotation class Lane(val value: Kind) {
    enum class Kind {
        /**
         * The app's shared state: the settings, the panel and its voice, the caption box and
         * the screen, the puzzle and review rooms, the app's own files. One class at a time,
         * as every class ran before the lanes.
         */
        APP,

        /**
         * The cell embedder (CellEmbedder): each one's onnxruntime holds about 350 MB of native
         * memory once it has read a start position, and PipelineTest runs two. Side by side with
         * the voice's models and the app lane's panel they took the emulator's 4 GB down to 400
         * MB free, into swap, and every lane ran four times slower (2026-10-02, measured). One
         * class at a time. Nothing of the app's shared state.
         */
        VISION,

        /**
         * The voice's and the ear's models (Kokoro, Parakeet, Silero): hundreds of MB of native
         * memory each, and onnxruntime threads of their own. One class at a time. Nothing of the
         * app's shared state.
         */
        VOICE,

        /**
         * Everything a class uses is its own and light (its engine, its store in a folder of its
         * own, its fixtures): any number of them side by side.
         */
        FREE,
    }
}

/**
 * The suite's classes in lanes side by side. Each lane runs its classes one at a time, as the
 * whole suite ran before; the lanes run at once: [Lane.Kind.APP], [Lane.Kind.VISION] and
 * [Lane.Kind.VOICE] on a thread each, [Lane.Kind.FREE] on [FREE_THREADS] threads that take its
 * classes longest first (the seconds each took last, from `lanes/seconds.tsv` in the test
 * assets; a class it does not know goes first). The device's four cores were a quarter used:
 * most classes run on one thread, and many wait on a window, a voice or an engine. The lanes
 * are drawn by what a class holds -- the app's shared state, or hundreds of MB of a model --
 * so the classes side by side never share state and fit the emulator's memory together.
 *
 * What the listeners see is the same as before: every test's events -- started, its failures,
 * finished -- are handed on together, one test at a time, so AndroidJUnitRunner's printer and
 * log, which follow one test at a time, print each as it always did. Each test's own times, and
 * its lane, go to `timings.tsv` in the app's external files, for `tools/test.ps1` to read.
 *
 * `-e lanes off` runs the classes one after another, in AndroidJUnitRunner's order, on one
 * thread: for telling a lane's fault from a test's, and for each test's own CPU in `timings.tsv`.
 */
object Lanes {
    private const val TAG = "sidecoach.lanes"

    /**
     * The [Lane.Kind.FREE] lane's threads. With the other three lanes, five on four cores: the app
     * and voice lanes spend much of their time waiting on a window, a voice or an engine.
     */
    private const val FREE_THREADS = 2

    @JvmStatic
    fun request(request: Request, args: Bundle): Request {
        val top = request.runner
        val children = (top as? ParentRunner<*>)?.let { childrenOf(it) } ?: return request
        if (children.isEmpty()) return request
        return Request.runner(LaneRunner(top, children, serial = args.getString("lanes") == "off"))
    }

    /** The runners the request kept (its filters applied): ParentRunner's own list. */
    @Suppress("UNCHECKED_CAST")
    private fun childrenOf(parent: ParentRunner<*>): List<Runner>? {
        val method = ParentRunner::class.java.getDeclaredMethod("getFilteredChildren").apply { isAccessible = true }
        val children = method.invoke(parent) as Collection<Any?>
        if (children.any { it !is Runner }) return null
        return children.map { it as Runner }
    }

    fun laneOf(runner: Runner): Lane.Kind =
        runner.description.testClass?.getAnnotation(Lane::class.java)?.value ?: Lane.Kind.APP

    /** The seconds each class took last time, for dealing the free lane longest first. */
    private fun seconds(): Map<String, Double> = try {
        InstrumentationRegistry.getInstrumentation().context.assets.open("lanes/seconds.tsv")
            .bufferedReader().useLines { lines ->
                lines.mapNotNull { line ->
                    val parts = line.split('\t')
                    if (parts.size == 2) parts[0] to (parts[1].toDoubleOrNull() ?: return@mapNotNull null) else null
                }.toMap()
            }
    } catch (_: java.io.IOException) {
        emptyMap()
    }

    private class LaneRunner(
        private val top: Runner,
        private val children: List<Runner>,
        private val serial: Boolean,
    ) : Runner() {
        override fun getDescription(): Description = top.description

        override fun run(notifier: RunNotifier) {
            val handOn = Any()
            val began = System.nanoTime()
            val times = ConcurrentLinkedQueue<String>()
            val byLane = children.groupBy { laneOf(it) }
            val known = seconds()
            // Longest first; a class not measured yet first of all, its length unknown.
            val free = ConcurrentLinkedQueue(
                byLane[Lane.Kind.FREE].orEmpty().sortedByDescending {
                    known[it.description.className] ?: Double.MAX_VALUE
                },
            )
            fun runOne(child: Runner, lane: String) {
                val relay = Relay(notifier, handOn, lane, began, times)
                val local = RunNotifier().apply { addListener(relay) }
                try {
                    child.run(local)
                } catch (exc: Throwable) {
                    synchronized(handOn) { notifier.fireTestFailure(Failure(child.description, exc)) }
                }
            }
            val threads = ArrayList<Thread>()
            fun lane(name: String, work: () -> Unit) {
                threads += Thread({
                    try {
                        work()
                    } catch (exc: Throwable) {
                        Log.e(TAG, "lane $name stopped: $exc")
                        throw exc
                    }
                }, "lane-$name").apply { start() }
            }
            if (serial) {
                lane("serial") { children.forEach { runOne(it, "serial") } }
            } else {
                for (kind in listOf(Lane.Kind.APP, Lane.Kind.VISION, Lane.Kind.VOICE)) {
                    val mine = byLane[kind].orEmpty()
                    if (mine.isNotEmpty()) lane(kind.name.lowercase()) { mine.forEach { runOne(it, kind.name.lowercase()) } }
                }
                repeat(minOf(FREE_THREADS, free.size)) { i ->
                    lane("free$i") {
                        while (true) runOne(free.poll() ?: break, "free")
                    }
                }
            }
            threads.forEach { it.join() }
            val wall = (System.nanoTime() - began) / 1e9
            Log.i(TAG, "lanes: ${byLane.mapValues { it.value.size }}, ${"%.1f".format(java.util.Locale.ROOT, wall)} s")
            write(times)
        }

        private fun write(times: Collection<String>) {
            try {
                val context = InstrumentationRegistry.getInstrumentation().targetContext
                val dir = context.getExternalFilesDir(null) ?: return
                File(dir, "timings.tsv").writeText(
                    "class\tmethod\tlane\tthread\tstart_s\tend_s\tprocess_cpu_s\n" + times.joinToString(""),
                )
            } catch (exc: Exception) {
                Log.w(TAG, "no timings written: $exc")
            }
        }
    }

    /**
     * One class's events, handed on to the run's listeners a whole test at a time: started,
     * its failures, finished, under [handOn], so a listener that follows one test at a time
     * (AndroidJUnitRunner's printer) never sees two tests interleaved.
     */
    private class Relay(
        private val out: RunNotifier,
        private val handOn: Any,
        private val lane: String,
        private val began: Long,
        private val times: MutableCollection<String>,
    ) : RunListener() {
        private val held = ArrayList<(RunNotifier) -> Unit>()
        private var inTest = false
        private val started = AtomicLong()
        private val cpuAtStart = AtomicLong()

        private fun now(): Double = (System.nanoTime() - began) / 1e9

        private fun handOn(events: List<(RunNotifier) -> Unit>) {
            synchronized(handOn) { events.forEach { it(out) } }
        }

        override fun testSuiteStarted(description: Description) = handOn(listOf { it.fireTestSuiteStarted(description) })

        override fun testSuiteFinished(description: Description) = handOn(listOf { it.fireTestSuiteFinished(description) })

        override fun testStarted(description: Description) {
            held.clear()
            inTest = true
            started.set(System.nanoTime())
            cpuAtStart.set(android.os.Process.getElapsedCpuTime())
            held +={ it.fireTestStarted(description) }
        }

        override fun testFailure(failure: Failure) {
            if (inTest) held += { it.fireTestFailure(failure) } else handOn(listOf { it.fireTestFailure(failure) })
        }

        override fun testAssumptionFailure(failure: Failure) {
            if (inTest) held += { it.fireTestAssumptionFailed(failure) } else handOn(listOf { it.fireTestAssumptionFailed(failure) })
        }

        override fun testIgnored(description: Description) = handOn(listOf { it.fireTestIgnored(description) })

        override fun testFinished(description: Description) {
            held += { it.fireTestFinished(description) }
            inTest = false
            val end = now()
            val start = (started.get() - began) / 1e9
            // The whole process's CPU over the test, every lane's: the test's own only with the
            // lanes off. Its engines' processes are not in it.
            val cpu = (android.os.Process.getElapsedCpuTime() - cpuAtStart.get()) / 1e3
            times += "${description.className}\t${description.methodName}\t$lane\t${Thread.currentThread().name}\t" +
                "%.3f\t%.3f\t%.3f\n".format(java.util.Locale.ROOT, start, end, cpu)
            handOn(ArrayList(held))
            held.clear()
        }
    }
}
