package com.sidecoach.llm

import com.sidecoach.coach.pyJsonDumps
import com.sidecoach.coach.pyJsonLoads
import com.sidecoach.coach.pyJsonOr
import com.sidecoach.coach.pyStr
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

/**
 * One provider for every OpenAI-compatible endpoint: LM Studio on a computer on the network,
 * DashScope, OpenAI itself, or anything else that serves `/v1/chat/completions`.
 * `llm/openai_compat.py`.
 *
 * What differs between servers is how reasoning is controlled, and that is the one thing a
 * provider is configured with ([thinkingControl]). Measured 2026-09-14 on the first two:
 *
 * - "prefill": LM Studio serving the Qwen 3.8 GGUFs ignores `/no_think` and every reasoning
 *   parameter, but honours a trailing assistant message holding an empty think block: the
 *   server continues that message and the model skips reasoning.
 * - "param": DashScope honours `enable_thinking` in the request body. With qwen3.8-max a
 *   turn took 9.7 s with thinking and 1.8 s without; qwen3.8-flash 6.1 s and 1.1 s.
 * - "effort": OpenAI's reasoning models take a reasoning effort (none, low, medium, high,
 *   ...) and reject `max_tokens` and `temperature`; the answer budget has headroom for the
 *   reasoning tokens that count against it. The effort is a property of the provider, not
 *   of the call: a coach at medium and a judge at none are two providers of the same model
 *   ([withEffort]).
 * - "none": nothing is sent and the model keeps its default.
 *
 * And which API it speaks ([wire]): "chat", the chat completions every such server serves, or
 * "responses", OpenAI's `/v1/responses` -- the only one on which its reasoning models take
 * function tools with an effort ("Function tools with reasoning_effort are not supported for
 * gpt-5.6-luna in /v1/chat/completions", 2026-09-29: at medium the coach's engine tools were
 * refused on every question). It is sent stateless (`store` false), and the model's reasoning
 * comes back encrypted in its output items, which go back with the tools' results so it
 * reasons on from where it was.
 *
 * On the chat wire, reasoning text, when the server returns it, comes back in
 * `message.reasoning_content`; the cached share of the prompt in
 * `usage.prompt_tokens_details.cached_tokens`.
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

/** `openai_compat.Wire`: the API a provider speaks, chat completions or OpenAI's responses. */
enum class Wire(val slug: String) {
    CHAT("chat"),
    RESPONSES("responses"),
}

const val EMPTY_THINK_PREFILL = "<think>\n\n</think>\n\n"

/**
 * Reasoning tokens count against the answer's token budget: a caller's budget is for the
 * answer, so the reasoning gets headroom of its own or a short answer comes back empty.
 */
const val REASONING_HEADROOM = 4000

/** The fewest output tokens the Responses API takes; fewer is refused. */
const val MIN_OUTPUT_TOKENS = 16

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

/**
 * A round's calls, each with what it returned: `zip(strict=True)`, which raises when the counts
 * differ.
 */
private fun answered(turn: ToolRound): List<Pair<ToolCall, String>> {
    val calls = turn.reply.toolCalls
    require(calls.size == turn.results.size) {
        "zip() argument 2 is ${if (turn.results.size < calls.size) "shorter" else "longer"} than argument 1"
    }
    return calls.zip(turn.results)
}

/**
 * Whether a failed call had reached the server, and is billed: a read that timed out or a
 * connection dropped mid-reply (`openai_compat.py` `_SENT_LOST`), not a connection refused,
 * a name not found or a connect timeout.
 */
fun lostAfterSending(exc: java.io.IOException): Boolean = when (exc) {
    // Never connected: refused, no route, no port, no local address -- the desktop's ConnectError,
    // which is no `_SENT_LOST`. As SocketExceptions they were counted as billed (the round-3
    // parity audit).
    is java.net.ConnectException, is java.net.UnknownHostException,
    is java.net.NoRouteToHostException, is java.net.PortUnreachableException,
    is java.net.BindException -> false
    is java.net.SocketTimeoutException -> !(exc.message ?: "").contains("connect", ignoreCase = true)
    is java.net.SocketException, is java.io.EOFException -> true
    else -> false
}

/**
 * The rounds of tool calls as chat messages: the reply that asked, then each result.
 * `openai_compat.chat_turns`.
 */
fun chatTurns(turns: List<ToolRound>): JSONArray {
    val wire = JSONArray()
    for (turn in turns) {
        val calls = JSONArray()
        for (call in turn.reply.toolCalls) {
            calls.put(
                JSONObject()
                    .put("id", call.id)
                    .put("type", "function")
                    .put("function", JSONObject().put("name", call.name).put("arguments", pyJsonDumps(call.arguments))),
            )
        }
        wire.put(
            JSONObject()
                .put("role", "assistant")
                .put("content", turn.reply.text.ifEmpty { JSONObject.NULL })
                .put("tool_calls", calls),
        )
        for ((call, result) in answered(turn)) {
            wire.put(JSONObject().put("role", "tool").put("tool_call_id", call.id).put("content", result))
        }
    }
    return wire
}

/**
 * The chat wire's message list: system, the conversation, and the no-think prefill if used.
 * The tool calls and their results ([chatTurns]) go in before the prefill, in
 * [OpenAICompatProvider.requestBody]: `build_messages(turns=)`.
 */
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

/**
 * The Responses wire's input: the conversation, then each round -- the reply's own output items
 * (its reasoning with them), or its text and calls when it has none -- and what each call
 * returned. `openai_compat.response_input`.
 */
fun responseInput(messages: List<ChatMessage>, turns: List<ToolRound> = emptyList()): JSONArray {
    val wire = JSONArray()
    for (message in messages) wire.put(JSONObject().put("role", message.role).put("content", message.content))
    for (turn in turns) {
        if (turn.reply.items.isNotEmpty()) {
            for (item in turn.reply.items) wire.put(item)
        } else {
            if (turn.reply.text.isNotEmpty()) {
                wire.put(JSONObject().put("role", "assistant").put("content", turn.reply.text))
            }
            for (call in turn.reply.toolCalls) {
                wire.put(
                    JSONObject()
                        .put("type", "function_call")
                        .put("call_id", call.id)
                        .put("name", call.name)
                        .put("arguments", pyJsonDumps(call.arguments)),
                )
            }
        }
        for ((call, result) in answered(turn)) {
            wire.put(JSONObject().put("type", "function_call_output").put("call_id", call.id).put("output", result))
        }
    }
    return wire
}

/**
 * Chat-completions function specs in the Responses form: the function's fields at the top, and
 * `strict` off -- on, every parameter would have to be required. `openai_compat.response_tools`.
 */
fun responseTools(tools: JSONArray): JSONArray {
    val out = JSONArray()
    for (index in 0 until tools.length()) {
        val function = tools.optJSONObject(index)?.optJSONObject("function") ?: JSONObject()
        val tool = JSONObject().put("type", "function")
        for (key in function.keys()) tool.put(key, function.opt(key))
        out.put(tool.put("strict", false))
    }
    return out
}

class OpenAICompatProvider(
    override val name: String,
    baseUrl: String,
    val model: String,
    private val apiKey: String? = null,
    val thinkingControl: ThinkingControl = ThinkingControl.NONE,
    val reasoningEffort: String? = null,
    val wire: Wire = Wire.CHAT,
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
    val usage: Usage = (usage ?: Usage()).also { if (it.model == null) it.model = model }

    private val headers: Map<String, String> =
        if (apiKey != null) mapOf("Authorization" to "Bearer $apiKey") else emptyMap()

    private val timeoutMs: Int = (timeoutS * 1000).toInt()

    private val endpoint: String
        get() = "$baseUrl/" + if (wire == Wire.RESPONSES) "responses" else "chat/completions"

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
            wire = wire,
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
        turns: List<ToolRound> = emptyList(),
        toolChoice: String? = null,
    ): JSONObject {
        if (wire == Wire.RESPONSES) {
            return responsesBody(system, messages, maxTokens, temperature, tools, turns, toolChoice)
        }
        val body = JSONObject()
        body.put("model", model)
        val sent = JSONArray()
        val listed = buildMessages(system, messages, thinking, thinkingControl)
        // The tool calls and their results follow the conversation; the no-think prefill, when
        // used, stays last (`build_messages(turns=)`).
        val prefilled = !thinking && thinkingControl == ThinkingControl.PREFILL
        for (message in if (prefilled) listed.dropLast(1) else listed) {
            sent.put(JSONObject().put("role", message.role).put("content", message.content))
        }
        val rounds = chatTurns(turns)
        for (index in 0 until rounds.length()) sent.put(rounds.get(index))
        if (prefilled) {
            val prefill = listed.last()
            sent.put(JSONObject().put("role", prefill.role).put("content", prefill.content))
        }
        body.put("messages", sent)
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

    /** [requestBody] on the Responses wire. `OpenAICompatProvider._responses_body`. */
    private fun responsesBody(
        system: String,
        messages: List<ChatMessage>,
        maxTokens: Int,
        temperature: Double,
        tools: JSONArray?,
        turns: List<ToolRound>,
        toolChoice: String?,
    ): JSONObject {
        val body = JSONObject()
        body.put("model", model)
        body.put("instructions", system)
        body.put("input", responseInput(messages, turns))
        // Nothing kept on the server: the reasoning travels encrypted with the items.
        body.put("store", false)
        for ((key, value) in extraBody) body.put(key, wrap(value))
        if (tools != null && tools.length() > 0) {
            body.put("tools", responseTools(tools))
            if (!toolChoice.isNullOrEmpty()) body.put("tool_choice", toolChoice)
        }
        val effort = thinkingControl == ThinkingControl.EFFORT && !reasoningEffort.isNullOrEmpty()
        body.put("max_output_tokens", maxOf(MIN_OUTPUT_TOKENS, maxTokens + (if (effort) REASONING_HEADROOM else 0)))
        if (thinkingControl == ThinkingControl.EFFORT) {
            if (!reasoningEffort.isNullOrEmpty()) {
                body.put("reasoning", JSONObject().put("effort", reasoningEffort))
                body.put("include", JSONArray().put("reasoning.encrypted_content"))
            }
        } else {
            body.put("temperature", temperature)
        }
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
        turns: List<ToolRound>,
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

    /**
     * [call], and a call that reached the server and then failed -- a read timeout, a
     * dropped connection -- counted all the same, as it is billed: estimated from what was
     * sent and what came back. A refused connection or a connect timeout never reached it.
     * `openai_compat.py` `_lost`.
     */
    private inline fun <T> sent(body: JSONObject, received: () -> Int, call: () -> T): T = try {
        call()
    } catch (exc: java.io.IOException) {
        if (lostAfterSending(exc)) {
            usage.record(
                mapOf("lost" to exc::class.java.simpleName),
                promptChars = chars(body),
                completionChars = received(),
            )
        }
        throw exc
    }

    private fun send(body: JSONObject): Completion {
        val started = System.nanoTime()
        val reply = sent(body, { 0 }) { transport.post(endpoint, headers, body.toString(), timeoutMs) }
        if (reply.status != 200) {
            // The status code stays in the message: the callers' backoff keys on "429".
            throw RuntimeException("${reply.status}: ${reply.text.take(200)}")
        }
        val data = JSONObject(reply.text)
        val elapsed = (System.nanoTime() - started) / 1e9
        if (wire == Wire.RESPONSES) {
            if (text(data, "status") == "failed") {
                val error = data.optJSONObject("error") ?: JSONObject()
                throw RuntimeException("failed: ${field(error, "code")}: ${field(error, "message")}")
            }
            val items = objects(data.optJSONArray("output"))
            val out = outputRead(items)
            val stats = linkedMapOf<String, Any?>("elapsed_s" to round3(elapsed))
            stats.putAll(responseStats(data, out))
            usage.record(
                stats,
                promptChars = chars(body),
                completionChars = out.text.length + out.summary.length,
            )
            return Completion(
                text = out.text,
                reasoning = out.summary,
                stats = stats,
                toolCalls = out.calls,
                items = items,
            )
        }

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
            "cache_write_tokens" to number(promptDetails, "cache_write_tokens"),
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
        if (wire == Wire.RESPONSES) return responsesStream(request.put("stream", true), onDelta)
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
        val reply = sent(body, { parts.length }) {
            transport.postLines(
                endpoint,
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
            "cache_write_tokens" to number(promptDetails, "cache_write_tokens"),
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

    /**
     * [stream] on the Responses wire: the text deltas to [onDelta], but for a preamble's
     * ([outputRead]); the output items, the usage and the end from the completed response (or,
     * stopped early, the items finished so far). An error in the stream raises, as a refused
     * request does, and so does a stream that ends unfinished: its text is cut off where the
     * connection was. `OpenAICompatProvider._responses_stream`.
     */
    private fun responsesStream(body: JSONObject, onDelta: StreamCallback): Completion {
        val started = System.nanoTime()
        var first: Double? = null
        val parts = StringBuilder()
        val items = sortedMapOf<Int, JSONObject>()
        val preambles = HashSet<Int>() // the output items that are preambles, by index
        var completed = JSONObject()
        var stopped = false
        val reply = sent(body, { parts.length }) {
            transport.postLines(endpoint, headers, body.toString(), timeoutMs) line@{ line ->
                if (!line.startsWith("data: ")) return@line true
                val event = try {
                    JSONObject(line.substring(6))
                } catch (_: JSONException) {
                    return@line true
                }
                when (text(event, "type")) {
                    "response.output_item.added" -> {
                        if (text(event.optJSONObject("item") ?: JSONObject(), "phase") == "commentary") {
                            preambles.add(event.optInt("output_index", 0))
                        }
                    }
                    "response.output_text.delta" -> {
                        val piece = text(event, "delta") ?: ""
                        if (piece.isEmpty() || event.optInt("output_index", 0) in preambles) return@line true
                        if (first == null) first = (System.nanoTime() - started) / 1e9
                        parts.append(piece)
                        if (!onDelta(piece)) {
                            stopped = true
                            return@line false
                        }
                    }
                    "response.output_item.done" -> {
                        items[event.optInt("output_index", 0)] = event.optJSONObject("item") ?: JSONObject()
                    }
                    "response.completed", "response.incomplete" -> {
                        completed = event.optJSONObject("response") ?: JSONObject()
                        return@line false
                    }
                    "response.failed" -> {
                        val error = (event.optJSONObject("response") ?: JSONObject()).optJSONObject("error") ?: JSONObject()
                        throw RuntimeException("failed: ${field(error, "code")}: ${field(error, "message")}")
                    }
                    "error" -> throw RuntimeException("error: ${field(event, "code")}: ${field(event, "message")}")
                }
                true
            }
        }
        if (reply.status != 200) {
            throw RuntimeException("${reply.status}: ${reply.text.take(200)}")
        }
        val done = completed.length() > 0
        if (!done && !stopped) throw RuntimeException("stream ended before the response completed")
        val elapsed = (System.nanoTime() - started) / 1e9
        val output = objects(completed.optJSONArray("output")).ifEmpty { items.values.toList() }
        val out = outputRead(output)
        // Stopped early, the text is what was handed on: the message has no item yet.
        val answer = if (done) out.text else parts.toString().trim()
        val stats = linkedMapOf<String, Any?>(
            "elapsed_s" to round3(elapsed),
            "first_token_s" to first?.let { round3(it) },
        )
        stats.putAll(responseStats(completed, out))
        stats["streamed"] = true
        if (stopped) stats["finish_reason"] = "stopped"
        usage.record(stats, promptChars = chars(body), completionChars = answer.length + out.summary.length)
        return Completion(
            text = answer,
            reasoning = out.summary,
            stats = stats,
            toolCalls = out.calls,
            items = output,
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
         * One call of the model's; arguments that are no JSON object come through under "_raw",
         * for the tool to say so. `openai_compat._call`.
         */
        private fun toolCall(callId: Any?, name: Any?, arguments: Any?, index: Int): ToolCall {
            val parsed = if (arguments is JSONObject) {
                arguments
            } else {
                val written = pyText(arguments).ifEmpty { "{}" }
                // Strict, as json.loads is: org.json took unquoted keys, single quotes and
                // trailing text, and ran a tool the desktop told the arguments were unreadable
                // (the parity audit, 2026-09-29).
                val loaded = pyJsonOr(written, null)
                if (loaded is Map<*, *>) JSONObject(loaded) else JSONObject().put("_raw", written)
            }
            return ToolCall(
                id = pyText(callId).ifEmpty { "call_$index" },
                name = pyText(name),
                arguments = parsed,
            )
        }

        /** The model's calls as the chat wire gives them. `openai_compat._tool_calls`. */
        fun toolCalls(raw: JSONArray): List<ToolCall> = (0 until raw.length()).mapNotNull { i ->
            val one = raw.optJSONObject(i) ?: return@mapNotNull null
            val function = one.optJSONObject("function") ?: JSONObject()
            toolCall(one.opt("id"), function.opt("name"), function.opt("arguments"), i)
        }

        /**
         * What a response's output items say: the reply, the reasoning's summary, the calls to
         * run, and a refusal's words. `openai_compat._Output`.
         */
        private class Output(val text: String, val summary: String, val calls: List<ToolCall>, val refusal: String)

        /**
         * The Responses wire's output items read. A preamble -- a message of phase "commentary",
         * written before tool calls in a tool-heavy turn -- is no part of the reply: it goes back
         * with the items, and is never said. A call not "completed" (the token budget spent inside
         * its arguments) is not run. `openai_compat._output_read`.
         */
        private fun outputRead(items: List<JSONObject>): Output {
            val replies = ArrayList<String>()
            val summary = ArrayList<String>()
            val refusals = ArrayList<String>()
            val calls = ArrayList<ToolCall>()
            for (item in items) {
                when (text(item, "type")) {
                    "message" -> {
                        val parts = objects(item.optJSONArray("content"))
                        for (part in parts) {
                            if (text(part, "type") == "refusal") refusals.add(pyText(part.opt("refusal")))
                        }
                        if (text(item, "phase") != "commentary") {
                            replies.add(
                                parts.filter { text(it, "type") == "output_text" }
                                    .joinToString("") { pyText(it.opt("text")) },
                            )
                        }
                    }
                    "reasoning" -> {
                        for (part in objects(item.optJSONArray("summary"))) summary.add(pyText(part.opt("text")))
                    }
                    "function_call" -> {
                        val status = text(item, "status")
                        if (status == null || status == "completed") {
                            calls.add(toolCall(item.opt("call_id"), item.opt("name"), item.opt("arguments"), calls.size))
                        }
                    }
                }
            }
            return Output(
                text = replies.map { it.trim() }.filter { it.isNotEmpty() }.joinToString("\n\n"),
                summary = summary.filter { it.isNotEmpty() }.joinToString("\n").trim(),
                calls = calls,
                refusal = refusals.filter { it.isNotEmpty() }.joinToString(" ").trim(),
            )
        }

        /**
         * A response's usage in the chat wire's terms, and its end as a finish reason: "stop",
         * "tool_calls", "length" for the token budget, or what else cut it short; a refusal's
         * words when the model refused. `openai_compat._response_stats`.
         */
        private fun responseStats(response: JSONObject, out: Output): Map<String, Any?> {
            val usageJson = response.optJSONObject("usage") ?: JSONObject()
            val details = usageJson.optJSONObject("input_tokens_details") ?: JSONObject()
            val status = text(response, "status")
            val finish = when (status) {
                "completed" -> if (out.calls.isNotEmpty()) "tool_calls" else "stop"
                "incomplete" -> {
                    val reason = text(response.optJSONObject("incomplete_details") ?: JSONObject(), "reason")
                    if (reason == "max_output_tokens") "length" else reason
                }
                else -> status
            }
            val stats = linkedMapOf<String, Any?>(
                "prompt_tokens" to number(usageJson, "input_tokens"),
                "cached_tokens" to number(details, "cached_tokens"),
                "cache_write_tokens" to number(details, "cache_write_tokens"),
                "completion_tokens" to number(usageJson, "output_tokens"),
                "reasoning_tokens" to number(usageJson.optJSONObject("output_tokens_details") ?: JSONObject(), "reasoning_tokens"),
                "finish_reason" to finish,
            )
            if (out.refusal.isNotEmpty()) stats["refusal"] = out.refusal
            return stats
        }

        /**
         * The prompt's size in characters, for the estimate when no usage comes back: the chat
         * wire's messages, or the Responses wire's instructions and input items (a message's
         * text, a call's result or its arguments).
         */
        fun chars(body: JSONObject): Int {
            val parts = body.optJSONArray("messages")?.takeIf { it.length() > 0 }
                ?: body.optJSONArray("input")
                ?: JSONArray()
            var total = pyText(body.opt("instructions")).length
            for (index in 0 until parts.length()) {
                val part = parts.optJSONObject(index) ?: continue
                val said = listOf("content", "output", "arguments").map { part.opt(it) }.firstOrNull { truthy(it) }
                total += pyText(said).length
            }
            return total
        }

        private fun text(o: JSONObject, key: String): String? =
            if (o.isNull(key)) null else o.opt(key)?.toString()

        private fun number(o: JSONObject, key: String): Int? =
            if (o.isNull(key)) null else (o.opt(key) as? Number)?.toInt()

        /** A field as Python's f-string writes `o.get(key)`: "None" when it is missing or null. */
        private fun field(o: JSONObject, key: String): String = text(o, key) ?: "None"

        /** The objects of a JSON list, none when there is no list. */
        private fun objects(array: JSONArray?): List<JSONObject> =
            if (array == null) emptyList() else (0 until array.length()).mapNotNull { array.optJSONObject(it) }

        /** Python's truthiness of an org.json value: null, false, 0, "", [] and {} are false. */
        private fun truthy(value: Any?): Boolean = when {
            value == null || value == JSONObject.NULL -> false
            value is String -> value.isNotEmpty()
            value is Boolean -> value
            value is Number -> value.toDouble() != 0.0
            value is JSONArray -> value.length() > 0
            value is JSONObject -> value.length() > 0
            else -> true
        }

        /** `str(value or "")` of an org.json value: a list or an object as Python's repr of it. */
        private fun pyText(value: Any?): String = when {
            !truthy(value) -> ""
            value is String -> value
            value is JSONObject || value is JSONArray -> pyStr(pyJsonLoads(value.toString()))
            value is Boolean -> "True"
            else -> value.toString()
        }

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
