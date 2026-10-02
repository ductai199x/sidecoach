package com.sidecoach

import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/**
 * A parity replay spread over the device's cores. Its cases are independent -- a position, a
 * rendered game -- and the code they hold to the desktop's keeps nothing between calls, so they
 * may run side by side; a case that fails fails the test as before, its error thrown once every
 * worker has stopped. On one core of four the coach's 1,465 positions took most of a minute.
 */
internal object Parallel {
    val cores: Int = Runtime.getRuntime().availableProcessors()

    /** [each] for every one of [items], on [threads] threads. */
    fun <T> forEach(items: List<T>, threads: Int = cores, each: (T) -> Unit) =
        forEach(items, threads, { }, { item, _ -> each(item) })

    /**
     * [each] for every one of [items], on [threads] threads, each thread with its own [resource]
     * (an embedding net, whose buffers one thread at a time may fill), closed after.
     */
    fun <T, R> forEach(items: List<T>, threads: Int = cores, resource: () -> R, each: (T, R) -> Unit) {
        val next = AtomicInteger()
        val failure = AtomicReference<Throwable?>()
        val workers = List(minOf(threads, items.size).coerceAtLeast(1)) {
            Thread {
                val held = try {
                    resource()
                } catch (exc: Throwable) {
                    failure.compareAndSet(null, exc)
                    return@Thread
                }
                try {
                    while (failure.get() == null) {
                        val i = next.getAndIncrement()
                        if (i >= items.size) break
                        each(items[i], held)
                    }
                } catch (exc: Throwable) {
                    failure.compareAndSet(null, exc)
                } finally {
                    (held as? AutoCloseable)?.close()
                }
            }.apply { start() }
        }
        workers.forEach { it.join() }
        failure.get()?.let { throw it }
    }
}

/**
 * Work handed over as it comes -- the cases of a fixture as they stream from the file -- to
 * [threads] workers, at most a few cases ahead of them, so the file is never all in memory.
 * [close] waits for the last and throws the first failure.
 */
internal class Workers(threads: Int = Parallel.cores) : AutoCloseable {
    private val queue = ArrayBlockingQueue<() -> Unit>(threads * 2)
    private val failure = AtomicReference<Throwable?>()
    private val done = {}
    private val threads = List(threads) {
        Thread {
            while (true) {
                val work = queue.take()
                if (work === done) break
                if (failure.get() != null) continue
                try {
                    work()
                } catch (exc: Throwable) {
                    failure.compareAndSet(null, exc)
                }
            }
        }.apply { start() }
    }

    fun submit(work: () -> Unit) {
        // A failure stops the reading: there is no need to read the rest of a failed replay.
        failure.get()?.let { return }
        while (!queue.offer(work, 100, TimeUnit.MILLISECONDS)) {
            if (threads.none { it.isAlive }) error("no worker is left to take the work")
        }
    }

    override fun close() {
        for (thread in threads) queue.put(done)
        threads.forEach { it.join() }
        failure.get()?.let { throw it }
    }
}
