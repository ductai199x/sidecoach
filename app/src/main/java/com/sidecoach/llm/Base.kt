package com.sidecoach.llm

import org.json.JSONArray
import org.json.JSONObject

/**
 * Provider interface. Every LLM backend implements this and nothing more: `llm/base.py`.
 *
 * The desktop's `LLMProvider` is a Protocol with `name`, `healthcheck` and `chat`; a
 * provider may also stream, which the Speaker looks for with `getattr` and falls back to
 * `chat` when it is missing. Here the stream is a second interface a provider may add,
 * checked with `is` rather than by name.
 */

/** `llm.base.ChatMessage`: the role is "user" or "assistant". */
data class ChatMessage(val role: String, val content: String)

/** `llm.base.ToolCall`: a function the model asked to have run, and its arguments as it wrote them. */
data class ToolCall(val id: String, val name: String, val arguments: JSONObject)

/**
 * `llm.base.Completion`. `stats` carries whatever the backend reported about the call;
 * [toolCalls] the functions the model asked for instead of (or before) a reply -- empty when
 * it replied, and always empty from a provider that takes no tools. [items] is the reply as
 * its wire takes it back in a later request, where that is more than the text and the calls:
 * the Responses API's output items, the model's reasoning among them.
 */
data class Completion(
    val text: String,
    val reasoning: String = "",
    val stats: Map<String, Any?> = emptyMap(),
    val toolCalls: List<ToolCall> = emptyList(),
    val items: List<JSONObject> = emptyList(),
)

/**
 * `llm.base.ToolRound`: one round of the model's calls -- the reply that made them, and what
 * each returned, in the calls' order. Each provider puts it on its own wire.
 */
data class ToolRound(val reply: Completion, val results: List<String>)

private val STATUS = Regex("^(\\d{3}):")

/**
 * A request the server refused as it stands -- a 4xx other than 429 (bad request, too long, no
 * access): asking again costs time and changes nothing. Providers throw with the status code
 * heading the message. `llm.base.refused_for_good`.
 */
fun refusedForGood(error: Throwable): Boolean {
    val status = STATUS.find(error.message ?: "")?.groupValues?.get(1) ?: return false
    return status.startsWith("4") && status != "429"
}

interface LLMProvider {
    val name: String

    /** Whether the backend can be reached, and one line saying what was found. */
    fun healthcheck(): Pair<Boolean, String>

    fun chat(
        system: String,
        messages: List<ChatMessage>,
        maxTokens: Int = 1500,
        temperature: Double = 0.7,
        thinking: Boolean = false,
    ): Completion
}

/**
 * A callback returning false ends the stream early: the text so far is the completion,
 * and the usage is then unknown. `llm.base.StreamCallback`.
 */
typealias StreamCallback = (String) -> Boolean

/** The optional half of the protocol: the same result as [LLMProvider.chat], piece by piece. */
interface StreamingProvider : LLMProvider {
    fun chatStream(
        system: String,
        messages: List<ChatMessage>,
        maxTokens: Int = 1500,
        temperature: Double = 0.7,
        thinking: Boolean = false,
        onDelta: StreamCallback,
    ): Completion
}

/**
 * A provider that takes tools: the desktop's `tools_ok`, whose `chat` and `chat_stream` then
 * accept `tools` (function specs in the chat-completions form), `turns` (the rounds that follow
 * the conversation, [ToolRound]: the model's earlier calls and what each returned) and
 * `tool_choice`. Here one call: streamed to [onDelta] when given, read whole otherwise.
 */
interface ToolsProvider : LLMProvider {
    fun chatTools(
        system: String,
        messages: List<ChatMessage>,
        tools: JSONArray,
        turns: List<ToolRound>,
        maxTokens: Int = 1500,
        temperature: Double = 0.7,
        toolChoice: String? = null,
        onDelta: StreamCallback? = null,
    ): Completion
}
