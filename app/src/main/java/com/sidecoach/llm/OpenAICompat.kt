package com.sidecoach.llm

import com.sidecoach.coach.pyJsonOr
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

/**
 * One provider for every OpenAI-compatible chat endpoint: LM Studio on a computer on the
 * network, DashScope, OpenAI itself, or anything else that serves
 * `/v1/chat/completions`. `llm/openai_compat.py`.
 *
 * What differs between servers is how reasoning is controlled, and that is the one thing a
 * provider is configured with ([thinkingControl]). Measured 2026-09-14 on the first two:
 *
 * - "prefill": LM Studio serving the Qwen 3.8 GGUFs ignores `/no_think` and every reasoning
 *   parameter, but honours a trailing assistant message holding an empty think block: the
 *   server continues that message and the model skips reasoning.
 * - "param": DashScope honours `enable_thinking` in the request body. With qwen3.8-max a
 *   turn took 9.7 s with thinking and 1.8 s without; qwen3.8-flash 6.1 s and 1.1 s.
 * - "effort": OpenAI's reasoning models take `reasoning_effort` (none, low, medium, high,
 *   ...) and reject `max_tokens` and `temperature`; the answer budget goes in
 *   `max_completion_tokens`, with headroom for the reasoning tokens that count against it.
 *   The effort is a property of the provider, not of the call: a coach at medium and a
 *   judge at none are two providers of the same model ([withEffort]).
 * - "none": nothing is sent and the model keeps its default.
 *
 * Reasoning text, when the server returns it, comes back in `message.reasoning_content`;
 * the cached share of the prompt in `usage.prompt_tokens_details.cached_tokens`.
 */

enum class ThinkingControl(val slug: String) {
    PREFILL("prefill"),
    PARAM("param"),
    EFFORT("effort"),
    NONE("none"),
    ;

    companion object {
        fun of(slug: String): ThinkingControl? = entries.firstOrNull { it.slug == slug }
    }
}

val THINKING_CONTROLS: List<String> = ThinkingControl.entries.map { it.slug }

const val EMPTY_THINK_PREFILL = "<think>\n\n</think>\n\n"

/**
 * Reasoning tokens count against max_completion_tokens: a caller's budget is for the
 * answer, so the reasoning gets headroom of its own or a short answer comes back empty.
 */
const val REASONING_HEADROOM = 4000

/**
 * What httpx's `raise_for_status` says of a reply that is not a success, as the desktop's
 * healthcheck reports it: the kind of error, the status with its reason, the address. The
 * reason is the server's own when the transport has it, else the standard one.
 */
fun statusError(status: Int, reason: String?, url: String): String {
    val kind = when (status / 100) {
        1 -> "Informational response"
        3 -> "Redirect response"
        4 -> "Client error"
        5 -> "Server error"
        else -> "Invalid status code"
    }
    val phrase = reason ?: REASON_PHRASES[status] ?: ""
    return "$kind '$status $phrase' for url '$url'\n" +
        "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/$status"
}

/** The standard reason phrases (httpx's `codes`), for a reply that came without its own. */
private val REASON_PHRASES: Map<Int, String> = mapOf(
    100 to "Continue", 101 to "Switching Protocols", 102 to "Processing", 103 to "Early Hints",
    300 to "Multiple Choices", 301 to "Moved Permanently", 302 to "Found", 303 to "See Other",
    304 to "Not Modified", 305 to "Use Proxy", 307 to "Temporary Redirect", 308 to "Permanent Redirect", 400 to "Bad Request",
    401 to "Unauthorized", 402 to "Payment Required", 403 to "Forbidden", 404 to "Not Found",
    405 to "Method Not Allowed", 406 to "Not Acceptable", 407 to "Proxy Authentication Required",
    408 to "Request Timeout", 409 to "Conflict", 410 to "Gone", 411 to "Length Required",
    412 to "Precondition Failed", 413 to "Request Entity Too Large", 414 to "Request-URI Too Long",
    415 to "Unsupported Media Type", 416 to "Requested Range Not Satisfiable",
    417 to "Expectation Failed", 418 to "I'm a teapot", 421 to "Misdirected Request",
    422 to "Unprocessable Entity", 423 to "Locked", 424 to "Failed Dependency", 425 to "Too Early",
    426 to "Upgrade Required", 428 to "Precondition Required", 429 to "Too Many Requests",
    431 to "Request Header Fields Too Large", 451 to "Unavailable For Legal Reasons",
    500 to "Internal Server Error", 501 to "Not Implemented", 502 to "Bad Gateway",
    503 to "Service Unavailable", 504 to "Gateway Timeout", 505 to "HTTP Version Not Supported",
    506 to "Variant Also Negotiates", 507 to "Insufficient Storage", 508 to "Loop Detected",
    510 to "Not Extended", 511 to "Network Authentication Required",
)

/** The wire message list: system, the conversation, and the no-think prefill if used. */
fun buildMessages(
    system: String,
    messages: List<ChatMessage>,
    thinking: Boolean,
    control: ThinkingControl,
): List<ChatMessage> {
    val wire = ArrayList<ChatMessage>(messages.size + 2)
    wire.add(ChatMessage("system", system))
    wire.addAll(messages)
    if (!thinking && control == ThinkingControl.PREFILL) {
        wire.add(ChatMessage("assistant", EMPTY_THINK_PREFILL))
    }
    return wire
}

class OpenAICompatProvider(
    override val name: String,
    baseUrl: String,
    val model: String,
    private val apiKey: String? = null,
    val thinkingControl: ThinkingControl = ThinkingControl.NONE,
    val reasoningEffort: String? = null,
    val timeoutS: Double = 60.0,
    extraBody: Map<String, Any?> = emptyMap(),
    private val transport: Transport = UrlTransport(),
    usage: Usage? = null,
) : StreamingProvider, ToolsProvider {

    val baseUrl: String = baseUrl.trimEnd('/')
    val extraBody: Map<String, Any?> = LinkedHashMap(extraBody)

    /**
     * The session's tokens over every call, shared with the provider's copies at other
     * efforts: one figure for the panel.
     */
    val usage: Usage = usage ?: Usage()

    private val headers: Map<String, String> =
        if (apiKey != null) mapOf("Authorization" to "Bearer $apiKey") else emptyMap()

    private val timeoutMs: Int = (timeoutS * 1000).toInt()

    /**
     * The same endpoint and model at another reasoning effort (an "effort" provider); any
     * other kind of provider is returned as it is.
     */
    fun withEffort(effort: String?): OpenAICompatProvider {
        if (thinkingControl != ThinkingControl.EFFORT || effort == reasoningEffort) return this
        return OpenAICompatProvider(
            name,
            baseUrl,
            model,
            apiKey = apiKey,
            thinkingControl = thinkingControl,
            reasoningEffort = effort,
            timeoutS = timeoutS,
            extraBody = extraBody,
            transport = transport,
            usage = usage,
        )
    }

    override fun healthcheck(): Pair<Boolean, String> {
        val ids: List<String> = try {
            val url = "$baseUrl/models"
            val reply = transport.get(url, headers, timeoutMs)
            if (reply.status !in 200..299) {
                return false to "$baseUrl unreachable: ${statusError(reply.status, reply.reason, url)}"
            }
            val data = JSONObject(reply.text).optJSONArray("data") ?: JSONArray()
            (0 until data.length()).map { data.optJSONObject(it)?.opt("id").toString() }
        } catch (exc: Exception) {
            return false to "$baseUrl unreachable: $exc"
        }
        if (model in ids) return true to "$model at $baseUrl"
        return false to "model '$model' not among ${ids.size} served models"
    }

    fun requestBody(
        system: String,
        messages: List<ChatMessage>,
        maxTokens: Int,
        temperature: Double,
        thinking: Boolean,
        tools: JSONArray? = null,
        turns: JSONArray? = null,
        toolChoice: String? = null,
    ): JSONObject {
        val body = JSONObject()
        body.put("model", model)
        val wire = JSONArray()
        val listed = buildMessages(system, messages, thinking, thinkingControl)
        // The tool calls and their results follow the conversation; the no-think prefill, when
        // used, stays last (`build_messages(turns=)`).
        val prefilled = !thinking && thinkingControl == ThinkingControl.PREFILL
        for (message in if (prefilled) listed.dropLast(1) else listed) {
            wire.put(JSONObject().put("role", message.role).put("content", message.content))
        }
        if (turns != null) for (index in 0 until turns.length()) wire.put(turns.get(index))
        if (prefilled) {
            val prefill = listed.last()
            wire.put(JSONObject().put("role", prefill.role).put("content", prefill.content))
        }
        body.put("messages", wire)
        for ((key, value) in extraBody) body.put(key, wrap(value))
        if (tools != null && tools.length() > 0) {
            body.put("tools", tools)
            if (!toolChoice.isNullOrEmpty()) body.put("tool_choice", toolChoice)
        }
        if (thinkingControl == ThinkingControl.EFFORT) {
            body.put(
                "max_completion_tokens",
                maxTokens + (if (!reasoningEffort.isNullOrEmpty()) REASONING_HEADROOM else 0),
            )
            if (!reasoningEffort.isNullOrEmpty()) body.put("reasoning_effort", reasoningEffort)
        } else {
            body.put("max_tokens", maxTokens)
            body.put("temperature", temperature)
        }
        if (thinkingControl == ThinkingControl.PARAM) body.put("enable_thinking", thinking)
        return body
    }

    override fun chat(
        system: String,
        messages: List<ChatMessage>,
        maxTokens: Int,
        temperature: Double,
        thinking: Boolean,
    ): Completion = send(requestBody(system, messages, maxTokens, temperature, thinking))

    /**
     * The reply streamed: every visible delta to [onDelta] as it arrives, the first one's
     * latency and the stream's usage in the stats. A callback returning false closes the
     * stream; the usage is then whatever the server had sent, usually nothing, and
     * finish_reason is "stopped".
     */
    override fun chatStream(
        system: String,
        messages: List<ChatMessage>,
        maxTokens: Int,
        temperature: Double,
        thinking: Boolean,
        onDelta: StreamCallback,
    ): Completion = stream(requestBody(system, messages, maxTokens, temperature, thinking), onDelta)

    /**
     * With tools: read whole, or streamed when [onDelta] is given. With tools LM Studio does not
     * continue the prefilled message: the model thinks again, and its thinking comes back as the
     * reply's first words, up to a closing </think> (Tai's puzzle replayed, 2026-09-28). Nothing
     * of it may reach the voice, so under "prefill" a reply with tools is read whole, its thinking
     * split off, and handed on at once.
     */
    override fun chatTools(
        system: String,
        messages: List<ChatMessage>,
        tools: JSONArray,
        turns: JSONArray,
        maxTokens: Int,
        temperature: Double,
        toolChoice: String?,
        onDelta: StreamCallback?,
    ): Completion {
        val body = requestBody(system, messages, maxTokens, temperature, false, tools, turns, toolChoice)
        if (onDelta == null || thinkingControl == ThinkingControl.PREFILL) {
            val completion = send(body)
            if (onDelta != null && completion.text.isNotEmpty()) onDelta(completion.text)
            return completion
        }
        return stream(body, onDelta)
    }

    private fun send(body: JSONObject): Completion {
        val started = System.nanoTime()
        val reply = transport.post("$baseUrl/chat/completions", headers, body.toString(), timeoutMs)
        if (reply.status != 200) {
            // The status code stays in the message: the callers' backoff keys on "429".
            throw RuntimeException("${reply.status}: ${reply.text.take(200)}")
        }
        val data = JSONObject(reply.text)
        val elapsed = (System.nanoTime() - started) / 1e9

        val choices = data.optJSONArray("choices")
        val choice = choices?.optJSONObject(0) ?: JSONObject()
        val message = choice.optJSONObject("message") ?: JSONObject()
        val usageJson = data.optJSONObject("usage") ?: JSONObject()
        val completionDetails = usageJson.optJSONObject("completion_tokens_details") ?: JSONObject()
        val promptDetails = usageJson.optJSONObject("prompt_tokens_details") ?: JSONObject()
        val stats = linkedMapOf<String, Any?>(
            "elapsed_s" to round3(elapsed),
            "prompt_tokens" to number(usageJson, "prompt_tokens"),
            "cached_tokens" to number(promptDetails, "cached_tokens"),
            "completion_tokens" to number(usageJson, "completion_tokens"),
            "reasoning_tokens" to number(completionDetails, "reasoning_tokens"),
            "finish_reason" to text(choice, "finish_reason"),
        )
        val (thought, answer) = unthought((text(message, "content") ?: "").trim())
        val reasoning = listOf((text(message, "reasoning_content") ?: "").trim(), thought)
            .filter { it.isNotEmpty() }
            .joinToString("\n")
        usage.record(
            stats,
            promptChars = chars(body),
            completionChars = answer.length + reasoning.length,
        )
        return Completion(
            text = answer,
            reasoning = reasoning,
            stats = stats,
            toolCalls = toolCalls(message.optJSONArray("tool_calls") ?: JSONArray()),
        )
    }

    /** [chatStream]'s body; tool calls arrive in pieces, by index, and are put together. */
    private fun stream(request: JSONObject, onDelta: StreamCallback): Completion {
        val body = request
        body.put("stream", true)
        body.put("stream_options", JSONObject().put("include_usage", true))
        val started = System.nanoTime()
        var first: Double? = null
        val parts = StringBuilder()
        val reasoning = StringBuilder()
        val calls = sortedMapOf<Int, JSONObject>()
        var usageJson = JSONObject()
        var finish: String? = null
        var stopped = false
        val reply = transport.postLines(
            "$baseUrl/chat/completions",
            headers,
            body.toString(),
            timeoutMs,
        ) line@{ line ->
            if (!line.startsWith("data: ")) return@line true
            val payload = line.substring(6).trim()
            if (payload == "[DONE]") return@line false
            val chunk = try {
                JSONObject(payload)
            } catch (_: JSONException) {
                return@line true
            }
            chunk.optJSONObject("usage")?.let { usageJson = it }
            val choices = chunk.optJSONArray("choices") ?: JSONArray()
            for (index in 0 until choices.length()) {
                val choice = choices.optJSONObject(index) ?: continue
                val delta = choice.optJSONObject("delta") ?: JSONObject()
                val thought = text(delta, "reasoning_content")
                if (!thought.isNullOrEmpty()) reasoning.append(thought)
                val reason = text(choice, "finish_reason")
                if (!reason.isNullOrEmpty()) finish = reason
                val pieces = delta.optJSONArray("tool_calls") ?: JSONArray()
                for (p in 0 until pieces.length()) {
                    val piece = pieces.optJSONObject(p) ?: continue
                    val call = calls.getOrPut(piece.optInt("index", 0)) {
                        JSONObject().put("id", "").put(
                            "function",
                            JSONObject().put("name", "").put("arguments", ""),
                        )
                    }
                    text(piece, "id")?.takeIf { it.isNotEmpty() }?.let { call.put("id", it) }
                    val function = piece.optJSONObject("function") ?: JSONObject()
                    val into = call.getJSONObject("function")
                    into.put("name", into.getString("name") + (text(function, "name") ?: ""))
                    into.put("arguments", into.getString("arguments") + (text(function, "arguments") ?: ""))
                }
                val piece = text(delta, "content")
                if (piece.isNullOrEmpty()) continue
                if (first == null) first = (System.nanoTime() - started) / 1e9
                parts.append(piece)
                if (!onDelta(piece)) {
                    stopped = true
                    break
                }
            }
            !stopped
        }
        if (reply.status != 200) {
            throw RuntimeException("${reply.status}: ${reply.text.take(200)}")
        }
        val elapsed = (System.nanoTime() - started) / 1e9
        val completionDetails = usageJson.optJSONObject("completion_tokens_details") ?: JSONObject()
        val promptDetails = usageJson.optJSONObject("prompt_tokens_details") ?: JSONObject()
        val stats = linkedMapOf<String, Any?>(
            "elapsed_s" to round3(elapsed),
            "first_token_s" to first?.let { round3(it) },
            "prompt_tokens" to number(usageJson, "prompt_tokens"),
            "cached_tokens" to number(promptDetails, "cached_tokens"),
            "completion_tokens" to number(usageJson, "completion_tokens"),
            "reasoning_tokens" to number(completionDetails, "reasoning_tokens"),
            "finish_reason" to if (stopped) "stopped" else finish,
            "streamed" to true,
        )
        val answer = parts.toString().trim()
        val thought = reasoning.toString().trim()
        // A stream closed early has no usage chunk: the count is then an estimate, and the
        // reasoning that was never streamed goes uncounted.
        usage.record(
            stats,
            promptChars = chars(body),
            completionChars = answer.length + thought.length,
        )
        return Completion(
            text = answer,
            reasoning = thought,
            stats = stats,
            toolCalls = toolCalls(JSONArray(calls.values.toList())),
        )
    }

    /** Single-turn convenience over [chat]. */
    fun complete(
        system: String,
        user: String,
        maxTokens: Int = 1500,
        temperature: Double = 0.7,
        thinking: Boolean = false,
    ): Completion =
        chat(system, listOf(ChatMessage("user", user)), maxTokens, temperature, thinking)

    companion object {
        /**
         * (the thinking, the reply) of a reply that carries the model's thinking before a closing
         * </think>, as a server that did not parse it hands it back; ("", text) otherwise.
         * `openai_compat._unthought`.
         */
        fun unthought(text: String): Pair<String, String> {
            val at = text.lastIndexOf("</think>")
            if (at < 0) return "" to text
            return text.substring(0, at).replace("<think>", "").trim() to
                text.substring(at + "</think>".length).trim()
        }

        /**
         * The model's calls as the wire gives them; arguments that are no JSON object come through
         * under "_raw", for the tool to say so. `openai_compat._tool_calls`.
         */
        fun toolCalls(raw: JSONArray): List<ToolCall> = (0 until raw.length()).mapNotNull { i ->
            val call = raw.optJSONObject(i) ?: return@mapNotNull null
            val function = call.optJSONObject("function") ?: JSONObject()
            val arguments = function.opt("arguments")
            // Strict, as json.loads is: org.json took unquoted keys, single quotes and
            // trailing text, and ran a tool the desktop told the arguments were unreadable
            // (the parity audit, 2026-09-29).
            val parsed = when (arguments) {
                is JSONObject -> arguments
                null, JSONObject.NULL -> JSONObject()
                else -> {
                    val written = arguments.toString().ifEmpty { "{}" }
                    val loaded = pyJsonOr(written, null)
                    if (loaded is Map<*, *>) JSONObject(loaded) else JSONObject().put("_raw", written)
                }
            }
            ToolCall(
                id = text(call, "id")?.ifEmpty { null } ?: "call_$i",
                name = text(function, "name") ?: "",
                arguments = parsed,
            )
        }

        /** The prompt's size in characters, for the estimate when no usage comes back. */
        fun chars(body: JSONObject): Int {
            val messages = body.optJSONArray("messages") ?: return 0
            var total = 0
            for (index in 0 until messages.length()) {
                val message = messages.optJSONObject(index) ?: continue
                total += (text(message, "content") ?: "").length
            }
            return total
        }

        private fun text(o: JSONObject, key: String): String? =
            if (o.isNull(key)) null else o.opt(key)?.toString()

        private fun number(o: JSONObject, key: String): Int? =
            if (o.isNull(key)) null else (o.opt(key) as? Number)?.toInt()

        private fun round3(value: Double): Double = Math.round(value * 1000.0) / 1000.0

        /** Whatever a settings file put in `extra_body`, as org.json wants it. */
        private fun wrap(value: Any?): Any = when (value) {
            null -> JSONObject.NULL
            is Map<*, *> -> JSONObject(value.entries.associate { it.key.toString() to it.value })
            is List<*> -> JSONArray(value)
            else -> value
        }
    }
}
