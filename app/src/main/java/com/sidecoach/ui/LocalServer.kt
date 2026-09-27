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
    interface Api {
        /**
         * The reply, or null for "no such route". [body] is the request's body as text; [raw]
         * the same bytes as they came, for the one route that takes audio (`/api/hear`).
         */
        fun handle(
            method: String,
            path: String,
            query: Map<String, String>,
            body: String,
            raw: ByteArray = ByteArray(0),
        ): Reply?
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

    /**
     * Bind and serve; the port it is served on. [preferred] first, when given and free: the
     * page keeps its preferences in localStorage, which belongs to an origin -- a port -- so
     * a new port each launch forgot every one of them. The system's choice otherwise.
     */
    fun start(preferred: Int = 0): Int {
        if (running) return port
        val host = InetAddress.getByName("127.0.0.1")
        val socket = (if (preferred > 0) runCatching { ServerSocket(preferred, 16, host) }.getOrNull() else null)
            ?: ServerSocket(0, 16, host)
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
            pool.execute {
                try {
                    serve(client)
                } catch (_: IOException) {
                    // The page navigated away and dropped the socket: ordinary.
                } catch (exc: Exception) {
                    Log.w(TAG, "a connection ended badly: $exc")
                } finally {
                    runCatching { client.close() }
                }
            }
        }
    }

    private fun serve(client: Socket) {
        client.tcpNoDelay = true
        val input = client.getInputStream().buffered()
        val output = BufferedOutputStream(client.getOutputStream())
        // Keep the connection: the page polls four times a second and the panel's assets
        // come in a burst when it loads.
        while (running) {
            val request = readRequest(input) ?: return
            val reply = answer(request)
            write(output, reply)
            output.flush()
        }
    }

    private class Request(
        val method: String,
        val path: String,
        val query: Map<String, String>,
        val raw: ByteArray,
    ) {
        val body: String get() = String(raw, Charsets.UTF_8)
    }

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
        var body = ByteArray(0)
        if (length > 0) {
            val bytes = ByteArray(length)
            var read = 0
            while (read < length) {
                val n = input.read(bytes, read, length - read)
                if (n < 0) break
                read += n
            }
            body = if (read == length) bytes else bytes.copyOf(read)
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
                api.handle(request.method, request.path, request.query, request.body, request.raw)
            } catch (exc: Exception) {
                // What Starlette answers an exception a route did not catch.
                Log.w(TAG, "${request.path} failed: $exc")
                return Reply(500, "text/plain; charset=utf-8", "Internal Server Error".toByteArray())
            }
            return reply ?: unserved(request.method)
        }
        // The static files take a GET (or a HEAD) and nothing else, as `StaticFiles` does.
        if (request.method != "GET" && request.method != "HEAD") return unserved(request.method)
        return file(request.path)
    }

    /**
     * A route the desktop's server does not serve: its static files answer, as the mount at
     * "/" does there -- 404 to a GET, 405 to any other method.
     */
    private fun unserved(method: String): Reply =
        if (method == "GET" || method == "HEAD") {
            Reply(404, "application/json; charset=utf-8", """{"detail":"Not Found"}""".toByteArray())
        } else {
            Reply(405, "application/json; charset=utf-8", """{"detail":"Method Not Allowed"}""".toByteArray())
        }

    /**
     * A file of the shipped page. The frontend is one page with client-side routing, so a
     * path that names no file is answered with index.html, as the desktop's server does.
     */
    private fun file(path: String): Reply {
        val asked = path.trimStart('/').ifEmpty { "index.html" }
        val found = read(asked)
        val name = if (found != null) asked else "index.html"
        val bytes = found ?: read(name)
            ?: return Reply(404, "text/plain; charset=utf-8", "no page".toByteArray())
        if (name.endsWith(".html") && htmlStyle.isNotEmpty()) {
            // The system bars' heights, on <html> before the first paint (see `htmlStyle`).
            val page = String(bytes, Charsets.UTF_8)
                .replaceFirst("<html lang=\"en\">", "<html lang=\"en\" style=\"$htmlStyle\">")
            return Reply(200, typeOf(name), page.toByteArray(Charsets.UTF_8))
        }
        return Reply(200, typeOf(name), bytes)
    }

    /**
     * Written into every page's <html> as its style: the system bars' heights as CSS
     * variables (`--inset-top: 24px; --inset-bottom: 48px`), so a page drawn under them
     * keeps its content clear from its first paint. Set by the activity from its insets.
     */
    @Volatile var htmlStyle: String = ""

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
