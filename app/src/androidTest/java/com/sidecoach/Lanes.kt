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
         * MB free, into swap, and every lane ran four times slower (2026-10-02, measured). With
         * [VOICE], the models' pool: two such classes at a time at most. Nothing of the app's
         * shared state.
         */
        VISION,

        /**
         * The voice's and the ear's models (Kokoro, Parakeet, Silero): hundreds of MB of native
         * memory each, and onnxruntime threads of their own. With [VISION], the models' pool.
         * Nothing of the app's shared state.
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
 * The suite's classes in lanes side by side: [Lane.Kind.APP]'s one at a time on one thread, in
 * the order AndroidJUnitRunner gives them, as the whole suite ran before; [Lane.Kind.VISION] and
 * [Lane.Kind.VOICE] together on [modelThreads] threads, and [Lane.Kind.FREE] on [freeThreads],
 * each pool taking its classes longest first (the seconds each took last, from
 * `lanes/seconds.tsv` in the test assets, which `tools/lane_seconds.py` writes from a run's
 * `timings.tsv`; a class it does not know goes first). The device's four cores were a quarter used:
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

    /** The device's cores, as the tests' own pools count them ([Parallel.cores]). */
    private val cores = Runtime.getRuntime().availableProcessors()

    /** The test process's heap limit, in MB: the classes side by side all draw on it. */
    internal val heapMb = Runtime.getRuntime().maxMemory() / (1024 * 1024)

    /**
     * The threads of the pool [Lane.Kind.VISION] and [Lane.Kind.VOICE] share: one class that
     * holds a model per 192 MB of heap. A model class's start-position read alone takes most of an
     * ordinary app's 192 MB (an embedder's 38 MB input buffer is on the heap, PipelineTest holds
     * two), and two side by side on the emulator's 4 GB took it into swap and ran a lane's other
     * class out of heap (2026-10-02, measured).
     */
    private val modelThreads = (heapMb / 192).toInt().coerceIn(1, maxOf(1, cores / 2))

    /**
     * The [Lane.Kind.FREE] pool's threads: half the cores, two at least. With the app lane and the
     * models' pool, more threads than cores: the app lane spends much of its time waiting on a
     * window, a voice or an engine.
     */
    private val freeThreads = maxOf(2, cores / 2)

    @JvmStatic
    fun request(request: Request, args: Bundle): Request {
        Log.i(TAG, "the heap's limit: $heapMb MB, $cores cores: $modelThreads model and $freeThreads free threads")
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

    /**
     * The most resident memory the process had while each test ran, sampled every 100 ms on a
     * thread of its own while any test runs: what a class costs the emulator's memory, the
     * thing the lanes are drawn by (the test's own with the lanes off).
     */
    private object Resident {
        private val open = java.util.Collections.newSetFromMap(java.util.concurrent.ConcurrentHashMap<AtomicLong, Boolean>())

        /** The Java heap in use now, in MB (the growth limit is 192 MB on the emulator). */
        fun heapMb(): Long = Runtime.getRuntime().let { (it.totalMemory() - it.freeMemory()) / (1024 * 1024) }

        private val heaps = java.util.concurrent.ConcurrentHashMap<AtomicLong, AtomicLong>()

        init {
            Thread({
                while (true) {
                    if (open.isNotEmpty()) {
                        val now = residentMb()
                        val heap = heapMb()
                        for (most in open) {
                            most.accumulateAndGet(now) { a, b -> maxOf(a, b) }
                            heaps[most]?.accumulateAndGet(heap) { a, b -> maxOf(a, b) }
                        }
                    }
                    Thread.sleep(100)
                }
            }, "lanes-resident").apply { isDaemon = true; start() }
        }

        fun watch(): AtomicLong = AtomicLong(residentMb()).also {
            heaps[it] = AtomicLong(heapMb())
            open += it
        }

        /** The most resident memory and the most Java heap the test saw, in MB. */
        fun done(most: AtomicLong): Pair<Long, Long> {
            open -= most
            val heap = heaps.remove(most)?.accumulateAndGet(heapMb()) { a, b -> maxOf(a, b) } ?: heapMb()
            return most.accumulateAndGet(residentMb()) { a, b -> maxOf(a, b) } to heap
        }
    }

    /** The test process's resident memory now, in MB (`/proc/self/statm`'s resident pages). */
    internal fun residentMb(): Long = try {
        File("/proc/self/statm").readText().trim().split(" ")[1].toLong() *
            android.system.Os.sysconf(android.system.OsConstants._SC_PAGESIZE) / (1024 * 1024)
    } catch (_: Exception) {
        -1
    }

    fun laneOf(runner: Runner): Lane.Kind =
        runner.description.testClass?.getAnnotation(Lane::class.java)?.value ?: Lane.Kind.APP

    /**
     * [classes] longest first, by the seconds each took last time; a class not measured yet goes
     * first of all, its length unknown.
     */
    private fun longestFirst(classes: List<Runner>, known: Map<String, Double>): ConcurrentLinkedQueue<Runner> =
        ConcurrentLinkedQueue(classes.sortedByDescending { known[it.description.className] ?: Double.MAX_VALUE })

    /** The seconds each class took last time, for dealing a pool's classes longest first. */
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
                val app = byLane[Lane.Kind.APP].orEmpty()
                if (app.isNotEmpty()) lane("app") { app.forEach { runOne(it, "app") } }
                fun pool(name: String, threads: Int, kinds: List<Lane.Kind>) {
                    val queue = longestFirst(kinds.flatMap { byLane[it].orEmpty() }, known)
                    repeat(minOf(threads, queue.size)) { i ->
                        lane("$name$i") {
                            while (true) {
                                val next = queue.poll() ?: break
                                runOne(next, laneOf(next).name.lowercase())
                            }
                        }
                    }
                }
                pool("models", modelThreads, listOf(Lane.Kind.VISION, Lane.Kind.VOICE))
                pool("free", freeThreads, listOf(Lane.Kind.FREE))
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
                    "class\tmethod\tlane\tthread\tstart_s\tend_s\tprocess_cpu_s\tpeak_rss_mb\tpeak_heap_mb\n" + times.joinToString(""),
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
        private var peak: AtomicLong? = null

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
            peak = Resident.watch()
            held += { it.fireTestStarted(description) }
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
            val (rss, heap) = peak?.let { Resident.done(it) } ?: (residentMb() to Resident.heapMb())
            times += "${description.className}\t${description.methodName}\t$lane\t${Thread.currentThread().name}\t" +
                "%.3f\t%.3f\t%.3f\t%d\t%d\n".format(java.util.Locale.ROOT, start, end, cpu, rss, heap)
            handOn(ArrayList(held))
            held.clear()
        }
    }
}
