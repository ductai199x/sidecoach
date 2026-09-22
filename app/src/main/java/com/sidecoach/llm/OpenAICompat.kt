package com.sidecoach.llm

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
) : StreamingProvider {

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
            val reply = transport.get("$baseUrl/models", headers, timeoutMs)
            if (reply.status !in 200..299) {
                return false to "$baseUrl unreachable: ${reply.status}: ${reply.text.take(200)}"
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
    ): JSONObject {
        val body = JSONObject()
        body.put("model", model)
        val wire = JSONArray()
        for (message in buildMessages(system, messages, thinking, thinkingControl)) {
            wire.put(JSONObject().put("role", message.role).put("content", message.content))
        }
        body.put("messages", wire)
        for ((key, value) in extraBody) body.put(key, wrap(value))
        if (thinkingControl == ThinkingControl.EFFORT) {
            body.put(
                "max_completion_tokens",
                maxTokens + (if (reasoningEffort != null) REASONING_HEADROOM else 0),
            )
            if (reasoningEffort != null) body.put("reasoning_effort", reasoningEffort)
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
    ): Completion {
        val body = requestBody(system, messages, maxTokens, temperature, thinking)
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
        val answer = (text(message, "content") ?: "").trim()
        val reasoning = (text(message, "reasoning_content") ?: "").trim()
        usage.record(
            stats,
            promptChars = chars(body),
            completionChars = answer.length + reasoning.length,
        )
        return Completion(text = answer, reasoning = reasoning, stats = stats)
    }

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
    ): Completion {
        val body = requestBody(system, messages, maxTokens, temperature, thinking)
        body.put("stream", true)
        body.put("stream_options", JSONObject().put("include_usage", true))
        val started = System.nanoTime()
        var first: Double? = null
        val parts = StringBuilder()
        val reasoning = StringBuilder()
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
        return Completion(text = answer, reasoning = thought, stats = stats)
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
