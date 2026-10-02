package com.sidecoach.setup

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.CountDownLatch
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * The setup's downloads and its run, against a fake local server: a body cut short, a whole
 * .part, a double tap on Download, Cancel mid-item, too little space.
 * `tests/test_setup_download.py`, case for case where the phone has the same code.
 */
@RunWith(AndroidJUnit4::class)
class SetupDownloadTest {

    private val full = ByteArray(1024) { (it % 256).toByte() }
    private val app = InstrumentationRegistry.getInstrumentation().targetContext

    private fun dir(): File = File(app.cacheDir, "setup-dl-${System.nanoTime()}").apply { mkdirs() }

    /** A server answering each connection with [handle] (the request's head as text). */
    private fun serve(handle: (Socket, String) -> Unit): Pair<ServerSocket, String> {
        val server = ServerSocket(0, 8, InetAddress.getByName("127.0.0.1"))
        Thread {
            while (true) {
                val client = try { server.accept() } catch (_: Exception) { return@Thread }
                client.use {
                    val head = StringBuilder()
                    val input = it.getInputStream()
                    while (!head.endsWith("\r\n\r\n")) {
                        val ch = input.read()
                        if (ch < 0) break
                        head.append(ch.toChar())
                    }
                    handle(it, head.toString())
                }
            }
        }.apply { isDaemon = true }.start()
        return server to "http://127.0.0.1:${server.localPort}"
    }

    private fun ranged(body: ByteArray): (Socket, String) -> Unit = { socket, head ->
        val start = Regex("(?i)range: bytes=(\\d+)-").find(head)?.groupValues?.get(1)?.toInt()
        val out = socket.getOutputStream()
        when {
            start == null -> {
                out.write("HTTP/1.1 200 OK\r\nContent-Length: ${body.size}\r\nConnection: close\r\n\r\n".toByteArray())
                out.write(body)
            }
            start >= body.size -> out.write(
                ("HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */${body.size}\r\n" +
                    "Content-Length: 0\r\nConnection: close\r\n\r\n").toByteArray(),
            )
            else -> {
                val rest = body.copyOfRange(start, body.size)
                out.write(
                    ("HTTP/1.1 206 Partial Content\r\nContent-Length: ${rest.size}\r\n" +
                        "Content-Range: bytes $start-${body.size - 1}/${body.size}\r\nConnection: close\r\n\r\n").toByteArray(),
                )
                out.write(rest)
            }
        }
        out.flush()
    }

    @Test
    fun aBodyCutShortIsNotTakenAsTheFile() {
        val (server, url) = serve { socket, _ ->
            val out = socket.getOutputStream()
            out.write("HTTP/1.1 200 OK\r\nContent-Length: 1024\r\nConnection: close\r\n\r\n".toByteArray())
            out.write(full, 0, 500)
            out.flush()
            socket.shutdownOutput()
        }
        val target = File(dir(), "kokoro-v1.0.onnx")
        try {
            download("$url/x", target)
            fail("a body cut short was taken")
        } catch (_: java.io.IOException) {
            // IncompleteDownload, or the platform's own early-end error: either way, not the file.
        } finally {
            server.close()
        }
        assertFalse(target.exists())
    }

    @Test
    fun aWholePartIsFinishedNotRefusedForever() {
        val (server, url) = serve(ranged(full))
        try {
            val target = File(dir(), "voices-v1.0.bin")
            File(target.path + ".part").writeBytes(full)
            download("$url/y", target)  // the server answers 416 to the Range
            assertArrayEquals(full, target.readBytes())
            // With the size known, no request at all.
            val other = File(dir(), "other.bin")
            File(other.path + ".part").writeBytes(full)
            download("http://127.0.0.1:9/never", other, expectSize = full.size.toLong())
            assertArrayEquals(full, other.readBytes())
            // A .part longer than the file starts over.
            val long = File(dir(), "long.bin")
            File(long.path + ".part").writeBytes(full + "junk".toByteArray())
            download("$url/z", long)
            assertArrayEquals(full, long.readBytes())
        } finally {
            server.close()
        }
    }

    /**
     * The setup's run thread has ended (joined, not polled against a clock), and its state says
     * so. [workers] has the run's thread, put there by the item's fetch (or the space check) on it.
     */
    private fun runOver(setup: Setup, workers: LinkedBlockingQueue<Thread>) {
        val worker = workers.poll(HANG_MS, TimeUnit.MILLISECONDS)
        assertTrue("the run never began", worker != null)
        worker!!.join(HANG_MS)
        assertFalse("the setup run never ended", worker.isAlive)
        assertFalse(setup.runState().optBoolean("running"))
    }

    /**
     * A second tap while the first setup still looks at the items (detection and the disk checks
     * take a while in the real items()), or while it fetches them, is refused: one run, one fetch.
     * Each step is held by a latch, so the second tap lands exactly there (desktop 0bf0f27).
     */
    @Test
    fun aDoubleTapRunsOneSetup() {
        val fetched = java.util.Collections.synchronizedList(ArrayList<String>())
        val looked = java.util.concurrent.atomic.AtomicInteger()
        val looking = CountDownLatch(1)
        val lookedEnough = CountDownLatch(1)
        val fetching = CountDownLatch(1)
        val fetchedEnough = CountDownLatch(1)
        val workers = LinkedBlockingQueue<Thread>()
        val setup = Setup(app, itemsOf = {
            looked.incrementAndGet()
            looking.countDown()
            assertTrue(lookedEnough.await(HANG_MS, TimeUnit.MILLISECONDS))
            listOf(
                Item("kokoro", "voice", 10, { false }, {
                    workers.put(Thread.currentThread())
                    fetched.add(Thread.currentThread().name)
                    fetching.countDown()
                    assertTrue(fetchedEnough.await(HANG_MS, TimeUnit.MILLISECONDS))
                }),
            )
        })
        val results = java.util.Collections.synchronizedList(ArrayList<Boolean>())
        val a = Thread { results += setup.start() }
        a.start()
        assertTrue("the first tap never looked at the items", looking.await(HANG_MS, TimeUnit.MILLISECONDS))
        results += setup.start() // the second tap, while the first one looks
        assertEquals("refused", listOf(false), results.toList())
        assertEquals("without looking at the items again", 1, looked.get())
        lookedEnough.countDown()
        a.join(HANG_MS)
        assertFalse(a.isAlive)
        assertEquals(listOf(false, true), results.sorted())
        assertTrue("the run never fetched", fetching.await(HANG_MS, TimeUnit.MILLISECONDS))
        assertFalse("a tap while the item is fetched is refused too", setup.start())
        fetchedEnough.countDown()
        runOver(setup, workers)
        assertEquals("one fetch, on the setup's thread", listOf("setup"), fetched.toList())
        // And a later run is allowed once this one is over.
        assertTrue(setup.start())
        runOver(setup, workers)
        assertEquals(2, fetched.size)
    }

    /**
     * Cancel pressed while the first item is mid-way: the item stops at its next chunk (its
     * progress call raises), and the second item never begins. The fetch holds at chunk 10
     * until the Cancel is in, so the chunk it stops at is exact.
     */
    @Test
    fun cancelStopsTheItemUnderWay() {
        val chunks = java.util.Collections.synchronizedList(ArrayList<Int>())
        val begun = java.util.Collections.synchronizedList(ArrayList<String>())
        val midway = CountDownLatch(1)
        val cancelled = CountDownLatch(1)
        val workers = LinkedBlockingQueue<Thread>()
        fun fetcher(name: String): (Progress) -> Unit = { progress ->
            workers.put(Thread.currentThread())
            begun.add(name)
            for (n in 0 until 200) {
                progress(n.toLong(), 200)
                chunks.add(n)
                if (n == 10) {
                    midway.countDown()
                    assertTrue(cancelled.await(HANG_MS, TimeUnit.MILLISECONDS))
                }
            }
        }
        val setup = Setup(app, itemsOf = {
            listOf(Item("a", "A", 200, { false }, fetcher("a")), Item("b", "B", 200, { false }, fetcher("b")))
        })
        assertTrue(setup.start())
        assertTrue("the first item never got under way", midway.await(HANG_MS, TimeUnit.MILLISECONDS))
        assertEquals("a", setup.runState().getString("item"))
        assertTrue(setup.runState().getBoolean("running"))
        setup.cancel()
        cancelled.countDown()
        runOver(setup, workers)
        val state = setup.runState()
        assertFalse(state.getBoolean("running"))
        assertTrue("the first item ran on: ${chunks.size}", chunks.size < 100)
        assertEquals("stopped at the first chunk after the Cancel", (0..10).toList(), chunks.toList())
        assertEquals("the second never began", listOf("a"), begun.toList())
        assertEquals(0, state.getJSONArray("finished").length())
        assertEquals(0, state.getJSONObject("errors").length())
    }

    @Test
    fun tooLittleSpaceIsAnErrorNotAFetch() {
        var fetched = false
        val workers = LinkedBlockingQueue<Thread>()
        val setup = Setup(
            app,
            itemsOf = { listOf(Item("big", "Big", 1_000_000_000_000, { false }, { fetched = true })) },
            freeSpace = {
                workers.put(Thread.currentThread()) // the run's thread looks at the space
                1_000_000_000L
            },
        )
        assertTrue(setup.start())
        runOver(setup, workers)
        assertFalse(fetched)
        assertTrue(setup.runState().getJSONObject("errors").getString("big").contains("free space"))
    }

    private companion object {
        // A safety net for a thread's signal, never the expected path.
        const val HANG_MS = 60_000L
    }
}
