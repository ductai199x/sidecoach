package com.sidecoach.llm

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

/** `llm.base.Completion`. `stats` carries whatever the backend reported about the call. */
data class Completion(
    val text: String,
    val reasoning: String = "",
    val stats: Map<String, Any?> = emptyMap(),
)

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
