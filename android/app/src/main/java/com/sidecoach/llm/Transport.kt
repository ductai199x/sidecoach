package com.sidecoach.llm

import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL

/**
 * What the provider needs of the network, behind an interface so a test can answer without
 * one -- the place `httpx`'s injected `transport` holds on the desktop.
 *
 * The one implementation is [UrlTransport], on `HttpURLConnection`: Android's own client,
 * so the app carries no HTTP library of its own.
 */

/**
 * A reply: its status, its body, the reason phrase the server sent, when the transport has it,
 * and the headers a caller reads (by lower-case name: the rate limit's `retry-after`).
 */
data class HttpReply(
    val status: Int,
    val text: String,
    val reason: String? = null,
    val headers: Map<String, String> = emptyMap(),
)

/** The headers of a reply the callers read, by lower-case name. */
private val KEPT_HEADERS = listOf("retry-after", "retry-after-ms")

interface Transport {
    fun get(url: String, headers: Map<String, String>, timeoutMs: Int): HttpReply

    fun post(url: String, headers: Map<String, String>, body: String, timeoutMs: Int): HttpReply

    /**
     * The same POST, read line by line as it arrives; [onLine] returning false closes the
     * connection. The reply's text is filled only when the status is not 200, which is
     * what the caller then raises with.
     */
    fun postLines(
        url: String,
        headers: Map<String, String>,
        body: String,
        timeoutMs: Int,
        onLine: (String) -> Boolean,
    ): HttpReply

    /**
     * [post], with [sent] called once the request has gone out whole (the desktop's httpx
     * trace, `send_request_body.complete`): a timeout after it is billed, one before it is not.
     * A transport that cannot tell calls nothing, and the reply arriving says it.
     */
    fun post(
        url: String,
        headers: Map<String, String>,
        body: String,
        timeoutMs: Int,
        sent: () -> Unit,
    ): HttpReply = post(url, headers, body, timeoutMs)

    /** [postLines], with [sent] as [post] has it. */
    fun postLines(
        url: String,
        headers: Map<String, String>,
        body: String,
        timeoutMs: Int,
        sent: () -> Unit,
        onLine: (String) -> Boolean,
    ): HttpReply = postLines(url, headers, body, timeoutMs, onLine)
}

class UrlTransport : Transport {

    override fun get(url: String, headers: Map<String, String>, timeoutMs: Int): HttpReply =
        open(url, "GET", headers, null, timeoutMs).use { connection ->
            HttpReply(connection.responseCode, readAll(connection), connection.responseMessage)
        }

    override fun post(
        url: String,
        headers: Map<String, String>,
        body: String,
        timeoutMs: Int,
    ): HttpReply = post(url, headers, body, timeoutMs) {}

    override fun post(
        url: String,
        headers: Map<String, String>,
        body: String,
        timeoutMs: Int,
        sent: () -> Unit,
    ): HttpReply = open(url, "POST", headers, body, timeoutMs, sent).use { connection ->
        HttpReply(connection.responseCode, readAll(connection), headers = kept(connection))
    }

    override fun postLines(
        url: String,
        headers: Map<String, String>,
        body: String,
        timeoutMs: Int,
        onLine: (String) -> Boolean,
    ): HttpReply = postLines(url, headers, body, timeoutMs, {}, onLine)

    override fun postLines(
        url: String,
        headers: Map<String, String>,
        body: String,
        timeoutMs: Int,
        sent: () -> Unit,
        onLine: (String) -> Boolean,
    ): HttpReply = open(url, "POST", headers, body, timeoutMs, sent).use { connection ->
        val status = connection.responseCode
        if (status != 200) return@use HttpReply(status, readAll(connection), headers = kept(connection))
        val stream = connection.inputStream
        BufferedReader(InputStreamReader(stream, Charsets.UTF_8)).use { reader ->
            while (true) {
                val line = reader.readLine() ?: break
                if (!onLine(line)) break
            }
        }
        HttpReply(status, "")
    }

    private fun open(
        url: String,
        method: String,
        headers: Map<String, String>,
        body: String?,
        timeoutMs: Int,
        sent: () -> Unit = {},
    ): HttpURLConnection {
        val connection = URL(url).openConnection() as HttpURLConnection
        connection.requestMethod = method
        connection.connectTimeout = timeoutMs
        connection.readTimeout = timeoutMs
        for ((key, value) in headers) connection.setRequestProperty(key, value)
        if (body != null) {
            connection.doOutput = true
            connection.setRequestProperty("Content-Type", "application/json")
            connection.setFixedLengthStreamingMode(body.toByteArray(Charsets.UTF_8).size)
            connection.outputStream.use { it.write(body.toByteArray(Charsets.UTF_8)) }
            sent()
        }
        return connection
    }

    private fun kept(connection: HttpURLConnection): Map<String, String> =
        KEPT_HEADERS.mapNotNull { name -> connection.getHeaderField(name)?.let { name to it } }.toMap()

    private fun readAll(connection: HttpURLConnection): String {
        val stream = if (connection.responseCode in 200..299) {
            connection.inputStream
        } else {
            connection.errorStream ?: return ""
        }
        return stream.bufferedReader(Charsets.UTF_8).use { it.readText() }
    }

    private inline fun <T> HttpURLConnection.use(body: (HttpURLConnection) -> T): T = try {
        body(this)
    } finally {
        disconnect()
    }
}
