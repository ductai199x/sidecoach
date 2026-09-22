package com.sidecoach.llm

/**
 * What the session's model calls have cost: tokens, and dollars when the prices are known.
 * `llm/usage.py`.
 *
 * One [Usage] per endpoint, shared by every provider object that speaks to it (the coach's
 * model at its several reasoning efforts, the verifier, the profile's query writer), so the
 * panel can show one figure for the session. A call whose usage the server did not report
 * -- a stream closed early, before the usage chunk -- is estimated from the text at four
 * characters a token, and counted as such.
 */

/** The estimate for a call the server sent no usage for. */
const val CHARS_PER_TOKEN = 4.0

/**
 * USD per million tokens: fresh prompt, cached prompt, completion (reasoning included).
 * Null for any of them means the cost cannot be given.
 */
data class Prices(
    val prompt: Double? = null,
    val cached: Double? = null,
    val completion: Double? = null,
) {
    val known: Boolean get() = prompt != null && completion != null
}

/**
 * Prices the presets know, USD per million tokens (fresh prompt, cached prompt, completion,
 * reasoning billed as completion), as OpenAI's model page gave them on 2026-09-20 -- the
 * standard tier; a prompt over 272K tokens costs 2x/1.5x, which the coach never sends.
 * Settings override these.
 */
val KNOWN_PRICES: Map<String, Prices> = mapOf(
    "gpt-5.6-luna" to Prices(0.20, 0.02, 1.20),
)

/**
 * The prices from the LLM settings, or the known ones for [model] (the provider's, once the
 * preset has resolved it); unknown otherwise, and the cost is not given.
 */
fun pricesFor(llm: LLMSettings, model: String? = null): Prices {
    if (llm.pricePromptPerM != null || llm.priceCompletionPerM != null) {
        return Prices(llm.pricePromptPerM, llm.priceCachedPerM, llm.priceCompletionPerM)
    }
    return KNOWN_PRICES[model ?: llm.model ?: ""] ?: Prices()
}

class Usage {
    private val lock = Any()
    var calls = 0
        private set

    /** Calls whose tokens are an estimate. */
    var estimated = 0
        private set
    var prompt = 0
        private set

    /** Of the prompt tokens, the ones the server served from its cache. */
    var cached = 0
        private set
    var completion = 0
        private set

    /** Of the completion tokens, the reasoning ones. */
    var reasoning = 0
        private set

    /**
     * Add one call. [stats] is the provider's (prompt_tokens, cached_tokens,
     * completion_tokens, reasoning_tokens); missing counts are estimated from the
     * character counts.
     */
    fun record(stats: Map<String, Any?>, promptChars: Int = 0, completionChars: Int = 0) {
        var prompt = asInt(stats["prompt_tokens"])
        var completion = asInt(stats["completion_tokens"])
        val wasEstimated = prompt == null || completion == null
        if (prompt == null) prompt = (promptChars / CHARS_PER_TOKEN).toInt()
        if (completion == null) completion = (completionChars / CHARS_PER_TOKEN).toInt()
        synchronized(lock) {
            calls += 1
            if (wasEstimated) estimated += 1
            this.prompt += prompt
            this.cached += asInt(stats["cached_tokens"]) ?: 0
            this.completion += completion
            this.reasoning += asInt(stats["reasoning_tokens"]) ?: 0
        }
    }

    /** Dollars at [prices], or null when they are not known. */
    fun cost(prices: Prices): Double? {
        if (!prices.known) return null
        val promptPrice = prices.prompt!!
        val completionPrice = prices.completion!!
        synchronized(lock) {
            val fresh = maxOf(0, prompt - cached)
            val cachedPrice = prices.cached ?: promptPrice
            return fresh / 1e6 * promptPrice +
                cached / 1e6 * cachedPrice +
                completion / 1e6 * completionPrice
        }
    }

    /** For the panel: the counts, and the cost when it can be given. */
    fun summary(prices: Prices): Map<String, Any?> {
        val out = LinkedHashMap<String, Any?>()
        synchronized(lock) {
            out["calls"] = calls
            out["estimated"] = estimated
            out["prompt"] = prompt
            out["cached"] = cached
            out["completion"] = completion
            out["reasoning"] = reasoning
        }
        out["cost"] = cost(prices)
        return out
    }
}

/** A count the server may have sent as an int, a float or not at all. */
private fun asInt(value: Any?): Int? = when (value) {
    null -> null
    is Int -> value
    is Number -> value.toInt()
    is String -> value.toDoubleOrNull()?.toInt()
    else -> null
}

private val COUNT_KEYS =
    listOf("calls", "estimated", "prompt", "cached", "completion", "reasoning")

/**
 * Several ledgers' summaries as one (the game's coach and the puzzle rooms'): the counts
 * summed, the cost summed when every one is known, else null.
 */
fun combine(summaries: List<Map<String, Any?>>): Map<String, Any?> {
    val out = LinkedHashMap<String, Any?>()
    for (key in COUNT_KEYS) out[key] = 0
    var cost: Double? = 0.0
    for (summary in summaries) {
        for (key in COUNT_KEYS) {
            out[key] = (out[key] as Int) + (asInt(summary[key]) ?: 0)
        }
        val one = summary["cost"]
        cost = if (cost == null || one == null) null else cost + (one as Number).toDouble()
    }
    out["cost"] = cost
    return out
}
