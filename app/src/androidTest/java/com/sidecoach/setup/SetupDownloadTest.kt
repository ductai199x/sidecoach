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

    private fun waitIdle(setup: Setup) {
        val deadline = System.currentTimeMillis() + 10_000
        while (setup.runState().optBoolean("running") && System.currentTimeMillis() < deadline) Thread.sleep(20)
    }

    /**
     * Voice finding 4: Setup fetched the voice and the recogniser and nothing loaded them; they
     * stayed off until a restart. A run that ends tells its followers what it fetched (the panel
     * loads the voice again, [com.sidecoach.ui.PanelWarm.setupFetched]); a failed item is not
     * among them (`test_speech_recognition_is_on_once_setup_has_fetched_it`).
     */
    @Test
    fun aRunThatEndsTellsWhatItFetched() {
        val told = java.util.concurrent.LinkedBlockingQueue<List<String>>()
        val setup = Setup(app, itemsOf = {
            listOf(
                Item("stt", "recogniser", 10, { false }, { }),
                Item("kokoro", "voice", 10, { false }, { throw java.io.IOException("offline") }),
            )
        })
        setup.onFetched.add { ids -> told.put(ids) }
        assertTrue(setup.start())
        assertEquals(listOf("stt"), told.poll(10, java.util.concurrent.TimeUnit.SECONDS))
    }

    @Test
    fun aDoubleTapRunsOneSetup() {
        val fetched = java.util.concurrent.atomic.AtomicInteger()
        val setup = Setup(app, itemsOf = {
            Thread.sleep(300)  // what looking at the items on disk costs
            listOf(Item("kokoro", "voice", 10, { false }, { fetched.incrementAndGet(); Thread.sleep(300) }))
        })
        val results = java.util.Collections.synchronizedList(ArrayList<Boolean>())
        val a = Thread { results += setup.start() }
        val b = Thread { results += setup.start() }
        a.start()
        Thread.sleep(50)
        b.start()
        a.join()
        b.join()
        waitIdle(setup)
        assertEquals(listOf(false, true), results.sorted())
        assertEquals(1, fetched.get())
    }

    @Test
    fun cancelStopsTheItemUnderWay() {
        val chunks = java.util.concurrent.atomic.AtomicInteger()
        val fetch: (Progress) -> Unit = { progress ->
            for (n in 0 until 200) {
                progress(n.toLong(), 200)
                chunks.incrementAndGet()
                Thread.sleep(10)
            }
        }
        val setup = Setup(app, itemsOf = {
            listOf(Item("a", "A", 200, { false }, fetch), Item("b", "B", 200, { false }, fetch))
        })
        assertTrue(setup.start())
        Thread.sleep(200)
        setup.cancel()
        waitIdle(setup)
        val state = setup.runState()
        assertFalse(state.getBoolean("running"))
        assertTrue("the first item ran on: ${chunks.get()}", chunks.get() < 100)
        assertEquals(0, state.getJSONArray("finished").length())
        assertEquals(0, state.getJSONObject("errors").length())
    }

    @Test
    fun tooLittleSpaceIsAnErrorNotAFetch() {
        var fetched = false
        val setup = Setup(
            app,
            itemsOf = { listOf(Item("big", "Big", 1_000_000_000_000, { false }, { fetched = true })) },
            freeSpace = { 1_000_000_000L },
        )
        assertTrue(setup.start())
        waitIdle(setup)
        assertFalse(fetched)
        assertTrue(setup.runState().getJSONObject("errors").getString("big").contains("free space"))
    }
}
