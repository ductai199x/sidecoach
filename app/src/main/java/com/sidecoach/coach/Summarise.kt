package com.sidecoach.coach

import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.LLMProvider

/**
 * The earlier dialogue, folded into a summary so the harness stays flat in a long game.
 * `coach/summarise.py`.
 *
 * Every `every` plies the coach folds what was said earlier than the last `keep` plies into
 * one block at the top of the record. The summariser sees only the dialogue -- the previous
 * summary and the lines being folded, each with the move it followed -- never the position,
 * so it can add no fact; the record and the position stay lossless. The last `keep` plies
 * stay verbatim, so the coach still knows its own last words.
 */

const val SUMMARY_SYSTEM =
    "You keep a running summary of what a chess coach and a player have said to each other " +
        "during a game. You are given the summary so far and the newest exchanges, each marked " +
        "with the move it followed. Write the new summary: what the coach has taught or pointed " +
        "at (the ideas, the patterns, the moves it praised or faulted, with move numbers), what " +
        "the player asked, and what advice was given -- in the third person ('the coach', 'the " +
        "player'), at most 120 words, no new facts about the position, nothing that was not " +
        "said. Plain text, no headings."

/**
 * The user message the summariser is given: the summary so far and the newest exchanges,
 * each marked with the move it followed. Pure, so it is compared against the desktop's.
 */
fun summaryRequest(pending: List<Pair<String, Entry>>, summaryBefore: String): String {
    val lines = pending.map { (after, entry) ->
        val who = if (entry.kind == "ask") "PLAYER" else "COACH"
        "after $after, $who: ${entry.text}"
    }
    return "Summary so far:\n" +
        summaryBefore.ifEmpty { "(nothing yet)" } +
        "\n\nNewest exchanges:\n" +
        lines.joinToString("\n") +
        "\n\nThe new summary:"
}

/**
 * Fold the dialogue older than the last [keep] plies into the summary. Returns whether
 * anything was folded; a failed call folds nothing. [lock] guards the conversation while it
 * is read and folded; the model call runs outside it.
 */
fun summarise(
    convo: Conversation,
    provider: LLMProvider,
    keep: Int = 20,
    lock: Any? = null,
): Boolean {
    val guard = lock ?: Any()
    val pending: List<Pair<String, Entry>>
    val summaryBefore: String
    synchronized(guard) {
        pending = convo.foldable(keep)
        summaryBefore = convo.summary
    }
    if (pending.isEmpty()) return false
    val content = summaryRequest(pending, summaryBefore)
    val messages = listOf(ChatMessage("user", content))
    val completion = try {
        // A rewrite, not a reasoning task: no thinking, where the provider can turn it off.
        provider.chat(SUMMARY_SYSTEM, messages, maxTokens = 400, temperature = 0.1)
    } catch (_: Exception) {
        return false
    }
    val text = completion.text.trim()
    if (text.isEmpty()) return false
    synchronized(guard) {
        convo.fold(pending.map { it.second }, text)
    }
    return true
}
