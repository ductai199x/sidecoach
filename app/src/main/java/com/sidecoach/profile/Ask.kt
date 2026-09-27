package com.sidecoach.profile

import com.sidecoach.coach.ProfileAnswer
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.LLMProvider
import com.sidecoach.voice.Py
import java.math.BigDecimal

/**
 * Ask the profile: the coach writes its own SQL over the record (DESIGN section 5.6).
 * `sidecoach/profile/ask.py`.
 *
 * A question about the player over time -- "why do I keep losing in the middlegame?", "which
 * openings do I score worst in?" -- is answered from the record, not from memory. The model is
 * given the schema in words (SCHEMA_DOC) and a set of worked queries (EXAMPLES), and the
 * question. It answers with one ```sql block when it needs data; the rows come back as the
 * next message; after at most ROUNDS such exchanges, or as soon as it writes prose instead,
 * the prose is the answer. Every query is read-only by construction (Store.query). The
 * queries and their rows are kept with the answer, so what it was read from can be shown.
 */
object Ask {

    const val ROUNDS = 4
    const val MAX_TOKENS = 900
    const val CELL_WIDTH = 60

    private val FENCE = Regex("```(?:sql)?\\s*(.*?)```", setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL))

    /** One query the model wrote, and what came back. */
    class Exchange(val sql: String) {
        var columns: List<String> = emptyList()
        var rows: List<List<Any?>> = emptyList()
        var error: String? = null
    }

    fun systemMessage(rounds: Int = ROUNDS): String {
        val examples = EXAMPLES.joinToString("\n") { (q, sql) -> "Q: $q\n```sql\n$sql\n```" }
        return SYSTEM_TEMPLATE
            .replace("{rounds}", rounds.toString())
            .replace("{schema}", SCHEMA_DOC)
            .replace("{examples}", examples)
    }

    /** The one SQL block of a reply, or null when the reply is prose. */
    fun extractSql(text: String): String? = FENCE.find(text)?.groupValues?.get(1)?.trim()

    /**
     * A value as the rows show it: `ask._cell`, with Python's `str` of each value -- an
     * infinite or NaN float as "inf", "-inf" or "nan", a whole float as its integer however
     * large, a blob as `b'...'` -- and the width counted in characters, not UTF-16 units.
     */
    fun cell(value: Any?): String {
        if (value == null) return "NULL"
        if (value is Double) {
            // "SELECT 1e999" crashed the ask (the parity audit).
            if (value.isNaN()) return "nan"
            if (value.isInfinite()) return if (value > 0) "inf" else "-inf"
            if (value != Math.floor(value)) return Insights.fixed(value, 3).trimEnd('0').trimEnd('.')
            return BigDecimal(value).toBigInteger().toString()
        }
        val text = if (value is ByteArray) Py.bytesRepr(value) else value.toString()
        if (text.codePointCount(0, text.length) <= CELL_WIDTH) return text
        return text.substring(0, text.offsetByCodePoints(0, CELL_WIDTH - 1)) + "…"
    }

    /** The rows as a plain table the model can read. */
    fun renderRows(columns: List<String>, rows: List<List<Any?>>): String {
        if (rows.isEmpty()) return "(no rows)"
        val lines = ArrayList<String>()
        lines.add(columns.joinToString(" | "))
        for (row in rows) lines.add(row.joinToString(" | ") { cell(it) })
        if (rows.size >= Store.QUERY_ROWS) lines.add("(first ${Store.QUERY_ROWS} rows only; aggregate or add LIMIT)")
        return lines.joinToString("\n")
    }

    /** Answer the question from the record, the model writing the queries. */
    fun answer(
        question: String,
        store: Store,
        provider: LLMProvider,
        rounds: Int = ROUNDS,
        maxTokens: Int = MAX_TOKENS,
    ): ProfileAnswer {
        val system = systemMessage(rounds)
        val messages = ArrayList<ChatMessage>()
        messages.add(ChatMessage("user", "Question: ${question.trim()}"))
        val exchanges = ArrayList<Exchange>()
        var calls = 0
        for (roundIndex in 0..rounds) {
            val completion = provider.chat(system, messages, maxTokens = maxTokens, temperature = 0.2)
            calls++
            val reply = completion.text.trim()
            val sql = extractSql(reply)
            if (sql == null || roundIndex == rounds) {
                val prose = FENCE.replace(reply, "").trim()
                if (prose.isNotEmpty()) return ProfileAnswer(prose, exchanges, calls = calls)
                break
            }
            val exchange = Exchange(sql)
            val body = try {
                val (columns, rows) = store.query(sql)
                exchange.columns = columns
                exchange.rows = rows
                renderRows(columns, rows)
            } catch (exc: QueryRefused) {
                exchange.error = exc.message ?: exc.toString()
                "Query refused: ${exchange.error}. Write a different single SELECT."
            }
            exchanges.add(exchange)
            messages.add(ChatMessage("assistant", reply))
            val left = rounds - roundIndex - 1
            val hint = if (left > 0) {
                " You may query $left more time${if (left != 1) "s" else ""}."
            } else {
                " Answer now in prose from what you have."
            }
            messages.add(ChatMessage("user", "Rows:\n$body\n$hint"))
        }
        return ProfileAnswer(fallback(exchanges), exchanges, calls = calls, failed = true)
    }

    /** When the model never wrote prose: the last rows it saw, in a sentence. */
    fun fallback(exchanges: List<Exchange>): String {
        val good = exchanges.filter { it.error == null }
        if (good.isEmpty()) return "I could not read the record for that; ask me again in a moment."
        val last = good.last()
        if (last.rows.isEmpty()) return "The record has nothing on that yet."
        val first = last.columns.zip(last.rows[0]).joinToString(", ") { (c, v) -> "$c ${cell(v)}" }
        val more = if (last.rows.size > 1) " and ${last.rows.size - 1} more rows" else ""
        return "From the record: $first$more."
    }
}
