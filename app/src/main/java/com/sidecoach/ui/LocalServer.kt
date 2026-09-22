package com.sidecoach.ui

import android.content.res.AssetManager
import android.util.Log
import java.io.BufferedOutputStream
import java.io.IOException
import java.io.InputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.net.URLDecoder
import java.util.concurrent.Executors

/**
 * The panel's backend, over ordinary HTTP on the loopback address.
 *
 * The page is the desktop's own build (`sidecoach/ui/web`, shipped here as the asset folder
 * `web/`) and it talks to the backend exactly as it does on a computer: same origin, plain
 * `fetch` to `/api/...`. That is the whole point of keeping the transport HTTP rather than a
 * bridge in the window -- one frontend, one contract, and what runs on the phone is the same
 * code that runs on the desktop.
 *
 * `WebViewAssetLoader` would serve the files without a socket, but `shouldInterceptRequest`
 * is not handed a POST body, and half the panel's controls are POSTs with one. So: a real
 * server, bound to 127.0.0.1 on a port the system picks, running only while the panel is
 * open. Nothing outside the phone can reach it.
 */
class LocalServer(
    private val assets: AssetManager,
    private val root: String = "web",
    private val api: Api,
) {

    /** What answers `/api/...`; anything else is a file from the asset folder. */
    fun interface Api {
        /** The reply, or null for "no such route". */
        fun handle(method: String, path: String, query: Map<String, String>, body: String): Reply?
    }

    class Reply(
        val status: Int = 200,
        val contentType: String = "application/json; charset=utf-8",
        val body: ByteArray = ByteArray(0),
    ) {
        constructor(json: String) : this(200, "application/json; charset=utf-8", json.toByteArray())
    }

    private var server: ServerSocket? = null
    private val pool = Executors.newCachedThreadPool()

    @Volatile
    private var running = false

    var port: Int = 0
        private set

    /** Bind and serve; the port the system gave us. */
    fun start(): Int {
        if (running) return port
        val socket = ServerSocket(0, 16, InetAddress.getByName("127.0.0.1"))
        server = socket
        port = socket.localPort
        running = true
        Thread({ accept(socket) }, "sidecoach-panel").apply { isDaemon = true }.start()
        Log.i(TAG, "the panel is served on http://127.0.0.1:$port/")
        return port
    }

    fun stop() {
        running = false
        runCatching { server?.close() }
        server = null
        pool.shutdownNow()
    }

    private fun accept(socket: ServerSocket) {
        while (running) {
            val client = try {
                socket.accept()
            } catch (_: IOException) {
                return
            }
            pool.execute { serve(client) }
        }
    }

    private fun serve(client: Socket) {
        client.use {
            it.tcpNoDelay = true
            val input = it.getInputStream().buffered()
            val output = BufferedOutputStream(it.getOutputStream())
            // Keep the connection: the page polls four times a second and the panel's
            // assets come in a burst when it loads.
            while (running) {
                val request = readRequest(input) ?: return
                val reply = answer(request)
                write(output, reply)
                output.flush()
            }
        }
    }

    private class Request(
        val method: String,
        val path: String,
        val query: Map<String, String>,
        val body: String,
    )

    private fun readRequest(input: InputStream): Request? {
        val line = readLine(input) ?: return null
        if (line.isEmpty()) return null
        val parts = line.split(" ")
        if (parts.size < 2) return null
        val method = parts[0]
        val target = parts[1]
        var length = 0
        while (true) {
            val header = readLine(input) ?: return null
            if (header.isEmpty()) break
            val colon = header.indexOf(':')
            if (colon > 0 && header.substring(0, colon).equals("Content-Length", true)) {
                length = header.substring(colon + 1).trim().toIntOrNull() ?: 0
            }
        }
        var body = ""
        if (length > 0) {
            val bytes = ByteArray(length)
            var read = 0
            while (read < length) {
                val n = input.read(bytes, read, length - read)
                if (n < 0) break
                read += n
            }
            body = String(bytes, 0, read, Charsets.UTF_8)
        }
        val mark = target.indexOf('?')
        val path = if (mark < 0) target else target.substring(0, mark)
        val query = if (mark < 0) emptyMap() else parseQuery(target.substring(mark + 1))
        return Request(method, decode(path), query, body)
    }

    /** A request line or header, up to CRLF; null at the end of the stream. */
    private fun readLine(input: InputStream): String? {
        val out = StringBuilder()
        while (true) {
            val ch = input.read()
            if (ch < 0) return if (out.isEmpty()) null else out.toString()
            if (ch == '\n'.code) return out.toString().removeSuffix("\r")
            out.append(ch.toChar())
        }
    }

    private fun parseQuery(text: String): Map<String, String> {
        val out = HashMap<String, String>()
        for (pair in text.split("&")) {
            if (pair.isEmpty()) continue
            val equals = pair.indexOf('=')
            if (equals < 0) out[decode(pair)] = "" else {
                out[decode(pair.substring(0, equals))] = decode(pair.substring(equals + 1))
            }
        }
        return out
    }

    private fun decode(text: String): String =
        runCatching { URLDecoder.decode(text, "UTF-8") }.getOrDefault(text)

    private fun answer(request: Request): Reply {
        if (request.path.startsWith("/api/")) {
            val reply = try {
                api.handle(request.method, request.path, request.query, request.body)
            } catch (exc: Exception) {
                Log.w(TAG, "${request.path} failed: $exc")
                return Reply(500, "application/json; charset=utf-8", """{"ok":false}""".toByteArray())
            }
            return reply ?: Reply(404, "application/json; charset=utf-8", """{"ok":false}""".toByteArray())
        }
        return file(request.path)
    }

    /**
     * A file of the shipped page. The frontend is one page with client-side routing, so a
     * path that names no file is answered with index.html, as the desktop's server does.
     */
    private fun file(path: String): Reply {
        val name = path.trimStart('/').ifEmpty { "index.html" }
        val bytes = read(name) ?: read("index.html")
        ?: return Reply(404, "text/plain; charset=utf-8", "no page".toByteArray())
        return Reply(200, typeOf(if (read(name) != null) name else "index.html"), bytes)
    }

    private fun read(name: String): ByteArray? = try {
        assets.open("$root/$name").use { it.readBytes() }
    } catch (_: IOException) {
        null
    }

    private fun typeOf(name: String): String = when (name.substringAfterLast('.', "")) {
        "html" -> "text/html; charset=utf-8"
        "js" -> "text/javascript; charset=utf-8"
        "css" -> "text/css; charset=utf-8"
        "json" -> "application/json; charset=utf-8"
        "svg" -> "image/svg+xml"
        "png" -> "image/png"
        "ico" -> "image/x-icon"
        "woff2" -> "font/woff2"
        "woff" -> "font/woff"
        "txt" -> "text/plain; charset=utf-8"
        "map" -> "application/json; charset=utf-8"
        else -> "application/octet-stream"
    }

    private fun write(out: BufferedOutputStream, reply: Reply) {
        val head = StringBuilder()
        head.append("HTTP/1.1 ").append(reply.status).append(' ')
            .append(if (reply.status == 200) "OK" else "Error").append("\r\n")
        head.append("Content-Type: ").append(reply.contentType).append("\r\n")
        head.append("Content-Length: ").append(reply.body.size).append("\r\n")
        // The page is polled constantly and the assets are versioned by name.
        head.append("Cache-Control: no-store\r\n")
        head.append("Connection: keep-alive\r\n\r\n")
        out.write(head.toString().toByteArray(Charsets.ISO_8859_1))
        out.write(reply.body)
    }

    companion object {
        private const val TAG = "sidecoach.panel"
    }
}
