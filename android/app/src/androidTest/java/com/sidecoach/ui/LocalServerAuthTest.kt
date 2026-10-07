package com.sidecoach.ui

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream
import java.net.Socket

/**
 * The local server answers its API only to the WebView that holds the launch's secret
 * (setup findings #1): any other app on the phone reaches the loopback port too. Health
 * stays open, as the desktop's does. Also the caps on what a request may claim.
 */
@RunWith(AndroidJUnit4::class)
class LocalServerAuthTest {

    private val calls = mutableListOf<String>()

    private val api = object : LocalServer.Api {
        override fun handle(
            method: String,
            path: String,
            query: Map<String, String>,
            body: String,
            raw: ByteArray,
        ): LocalServer.Reply {
            calls += "$method $path"
            return LocalServer.Reply("""{"ok":true}""")
        }
    }

    private fun serve(token: String?): LocalServer =
        LocalServer(InstrumentationRegistry.getInstrumentation().targetContext.assets, api = api, token = token)

    /** A raw request (headers as given); the reply's status. */
    private fun send(port: Int, request: String): Int {
        Socket("127.0.0.1", port).use { socket ->
            socket.soTimeout = 10_000
            socket.getOutputStream().apply {
                write(request.toByteArray(Charsets.ISO_8859_1))
                flush()
            }
            val out = ByteArrayOutputStream()
            val input = socket.getInputStream()
            while (true) {
                val ch = input.read()
                if (ch < 0 || ch == '\n'.code) break
                out.write(ch)
            }
            return out.toString().split(" ")[1].toInt()
        }
    }

    private fun get(port: Int, path: String, vararg headers: String, host: String = "127.0.0.1:$port"): Int =
        send(port, "GET $path HTTP/1.1\r\nHost: $host\r\n" + headers.joinToString("") { "$it\r\n" } + "Connection: close\r\n\r\n")

    @Test
    fun theApiWantsTheLaunchSecret() {
        val token = LocalServer.newToken()
        val server = serve(token)
        val port = server.start()
        try {
            // No secret, a wrong one, the right one in the wrong cookie: refused, unseen by the API.
            assertEquals(401, get(port, "/api/poll"))
            assertEquals(401, get(port, "/api/poll", "Cookie: ${LocalServer.COOKIE}=nope"))
            assertEquals(401, get(port, "/api/poll", "Authorization: Bearer nope"))
            assertEquals(401, get(port, "/api/poll", "Cookie: other=$token"))
            assertEquals(401, get(port, "/api/settings", "Cookie: ${LocalServer.COOKIE}=${token}x"))
            assertEquals(emptyList<String>(), calls)
            // Health is open, as the desktop's.
            assertEquals(200, get(port, "/api/health"))
            // The WebView's cookie (among others), or a bearer header.
            assertEquals(200, get(port, "/api/poll", "Cookie: a=b; ${LocalServer.COOKIE}=$token; c=d"))
            assertEquals(200, get(port, "/api/poll", "Authorization: Bearer $token"))
            // The page itself is served to anyone, as on the desktop: it holds no secret.
            assertTrue(get(port, "/") in setOf(200, 404))
        } finally {
            server.stop()
        }
    }

    @Test
    fun anotherOriginIsRefusedEvenWithTheSecret() {
        val token = LocalServer.newToken()
        val server = serve(token)
        val port = server.start()
        val cookie = "Cookie: ${LocalServer.COOKIE}=$token"
        try {
            // DNS rebinding: a name that resolves here but is not this server's.
            assertEquals(401, get(port, "/api/poll", cookie, host = "evil.example:$port"))
            // A page on another origin posting text/plain.
            val post = "POST /api/settings HTTP/1.1\r\nHost: 127.0.0.1:$port\r\nOrigin: http://evil.example\r\n" +
                "$cookie\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}"
            assertEquals(401, send(port, post))
            // Its own page.
            val own = post.replace("http://evil.example", "http://127.0.0.1:$port")
            assertEquals(200, send(port, own))
        } finally {
            server.stop()
        }
    }

    @Test
    fun aRequestOverTheCapsIsRefusedNotAllocated() {
        val server = serve(null)
        val port = server.start()
        try {
            val huge = "POST /api/x HTTP/1.1\r\nHost: 127.0.0.1:$port\r\nContent-Length: 2000000000\r\n\r\n"
            assertEquals(413, send(port, huge))
            val line = "GET /" + "a".repeat(LocalServer.MAX_LINE + 10) + " HTTP/1.1\r\n\r\n"
            assertEquals(413, send(port, line))
            // And the server is still there.
            assertEquals(200, get(port, "/api/health"))
        } finally {
            server.stop()
        }
    }

    @Test
    fun aWavWithANegativeChunkSizeEnds() {
        // RIFF....WAVE then a chunk of size -8: it walked back onto itself forever.
        val wav = java.nio.ByteBuffer.allocate(28).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        wav.put("RIFF".toByteArray()).putInt(20).put("WAVE".toByteArray())
        wav.put("junk".toByteArray()).putInt(-8).putInt(0)
        val app = InstrumentationRegistry.getInstrumentation().targetContext
        val panel = PanelServerTest.serve(app)
        val port = panel.start()
        try {
            val worker = Thread {
                val connection = java.net.URL("http://127.0.0.1:$port/api/hear").openConnection() as java.net.HttpURLConnection
                connection.requestMethod = "POST"
                connection.doOutput = true
                connection.outputStream.use { it.write(wav.array()) }
                connection.responseCode
            }
            worker.start()
            worker.join(10_000)
            assertTrue("a bad WAV must not hang its request", !worker.isAlive)
        } finally {
            panel.stop()
        }
    }
}
