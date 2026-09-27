package com.sidecoach.llm

/**
 * No model at all: the coach speaks from the board and its templates. `llm/offline.py`.
 *
 * (Tai, 2026-09-21: the app must work, free and offline, for whoever wants it that way.)
 * `llm.provider = "none"` gives the coach this provider. It answers no call; the Speaker
 * sees [offline] and takes each request's fallback -- the verdict written from the board
 * for a best, excellent or book move, the lesson and hint templates, the review's own
 * sentences -- without a model, a network or a key. The user's typed questions get the
 * template for their kind (the hint, the material, the standing) rather than a reply.
 */

/** Raised by any call: there is no model to call. */
class OfflineError(message: String) : RuntimeException(message)

class OfflineProvider : LLMProvider {
    override val name = "none"
    val model: String? = null
    val offline = true

    /** A ledger with nothing on it, as the desktop's has: the strip shows no calls, not none. */
    val usage = Usage()

    override fun healthcheck(): Pair<Boolean, String> =
        true to "offline: the coach speaks from the board and its templates, no model"

    override fun chat(
        system: String,
        messages: List<ChatMessage>,
        maxTokens: Int,
        temperature: Double,
        thinking: Boolean,
    ): Completion = throw OfflineError("no model: llm.provider is none")

    fun withEffort(effort: String?): OfflineProvider = this
}

/**
 * A provider's ledger, as the desktop reads it (`getattr(provider, "usage", None)`): the
 * endpoint's, or the offline provider's empty one; null for anything that keeps none.
 */
fun usageOf(provider: LLMProvider?): Usage? = when (provider) {
    is OpenAICompatProvider -> provider.usage
    is OfflineProvider -> provider.usage
    else -> null
}

/** A provider's model, as `getattr(provider, "model", None)` reads it. */
fun modelOf(provider: LLMProvider?): String? = when (provider) {
    is OpenAICompatProvider -> provider.model
    is OfflineProvider -> provider.model
    else -> null
}
