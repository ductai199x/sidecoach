package com.sidecoach.coach

import com.sidecoach.events.ReviewJson
import com.sidecoach.events.ReviewedGame
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.Completion
import com.sidecoach.llm.HttpReply
import com.sidecoach.llm.LLMProvider
import com.sidecoach.llm.OpenAICompatProvider
import com.sidecoach.llm.Prices
import com.sidecoach.llm.ThinkingControl
import com.sidecoach.llm.Transport
import org.json.JSONArray
import org.json.JSONObject
import org.junit.AfterClass
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.IOException
import java.util.concurrent.ConcurrentHashMap

/**
 * The AI coach's words on a reviewed game's key moves as the desktop's (`coach/review_words.py`,
 * `coach/review_links.py`): `tools/make_review_words_fixtures.py` records, on nine games (the
 * seven of profile/reviews.json, the desktop tests' short game, and that game with no key move),
 * the key moves, every prompt, the estimate at several prices, and runs of `write` with a scripted
 * model -- each request as the model was asked it, the canned replies it was handed (good ones,
 * ones the checks drop, odd heads, empty and unreadable answers, failed calls), each `on_answer`,
 * the result or the error raised, and the result's links -- with hand-written paragraphs that hit
 * every linking rule, and `to_student`, `parse`, `in_colours` and `checked` on their own.
 *
 * Here the same scripted model replays those replies. The calls run at once, so the requests are
 * compared by the ply they ask about, never by when they came. No model is called.
 */
class ReviewWordsParityTest {

    /**
     * The desktop's scripted model, replaying its replies: each ply's text and token counts, or
     * its error. Records every request by the ply it asks about.
     */
    private class Scripted(private val replies: JSONObject) : LLMProvider {
        override val name = "scripted"
        val requests = ConcurrentHashMap<Int, JSONObject>()
        val systems = ConcurrentHashMap<Int, String>()

        override fun healthcheck(): Pair<Boolean, String> = true to "scripted"

        override fun chat(
            system: String,
            messages: List<ChatMessage>,
            maxTokens: Int,
            temperature: Double,
            thinking: Boolean,
        ): Completion {
            val ply = plyOf(messages[0].content)
            systems[ply] = system
            requests[ply] = JSONObject()
                .put("messages", JSONArray(messages.map { JSONObject().put("role", it.role).put("content", it.content) }))
                .put("max_tokens", maxTokens)
                .put("temperature", temperature)
                .put("thinking", thinking)
            val reply = replies.optJSONObject("$ply") ?: throw IllegalStateException("no reply scripted for ply $ply")
            if (reply.has("error")) throw RuntimeException(reply.getString("error"))
            return Completion(text = reply.getString("text"), stats = statsOf(reply.getJSONObject("stats")))
        }
    }

    /**
     * The same replies on the chat-completions wire, for an [OpenAICompatProvider]: the text as
     * the message's content, the token counts where the server puts them.
     */
    private class ScriptedWire(private val replies: JSONObject) : Transport {
        val bodies = ConcurrentHashMap<Int, JSONObject>()

        override fun get(url: String, headers: Map<String, String>, timeoutMs: Int) = HttpReply(404, "")

        override fun post(url: String, headers: Map<String, String>, body: String, timeoutMs: Int): HttpReply {
            val sent = JSONObject(body)
            val messages = sent.getJSONArray("messages")
            val ply = plyOf(messages.getJSONObject(messages.length() - 1).getString("content"))
            bodies[ply] = sent
            val reply = replies.getJSONObject("$ply")
            val stats = reply.getJSONObject("stats")
            val usage = JSONObject()
            for (key in listOf("prompt_tokens", "completion_tokens")) if (stats.has(key)) usage.put(key, stats.get(key))
            val prompt = JSONObject()
            for (key in listOf("cached_tokens", "cache_write_tokens")) if (stats.has(key)) prompt.put(key, stats.get(key))
            usage.put("prompt_tokens_details", prompt)
            if (stats.has("reasoning_tokens")) {
                usage.put("completion_tokens_details", JSONObject().put("reasoning_tokens", stats.get("reasoning_tokens")))
            }
            val message = JSONObject().put("role", "assistant").put("content", reply.getString("text"))
            val choice = JSONObject().put("message", message).put("finish_reason", "stop")
            return HttpReply(200, JSONObject().put("choices", JSONArray().put(choice)).put("usage", usage).toString())
        }

        override fun postLines(
            url: String,
            headers: Map<String, String>,
            body: String,
            timeoutMs: Int,
            onLine: (String) -> Boolean,
        ): HttpReply = throw UnsupportedOperationException("the words are read whole")
    }

    @Test
    fun theKeyMovesPromptsAndEstimatesAreTheDesktops() {
        val failures = ArrayList<String>()
        var prompts = 0
        for ((g, game) in games) {
            val name = g.getString("name")
            same("$name key plies", JSONArray(ReviewWords.keyPlies(game)), g.getJSONArray("key_plies"), failures)
            val opening = str(g, "opening")
            val byUser = g.getJSONObject("prompts")
            for (key in byUser.keys()) {
                val user = if (key == "None") null else key
                val told = byUser.getJSONObject(key)
                for (ply in told.keys()) {
                    prompts += 1
                    val mine = ReviewWords.prompt(game, ply.toInt(), user, opening)
                    if (mine != told.getString(ply)) failures.add("$name prompt $ply for $key:\n  mine:   $mine\n  python: ${told.getString(ply)}")
                }
            }
            val estimates = g.getJSONArray("estimates")
            for (i in 0 until estimates.length()) {
                val e = estimates.getJSONObject(i)
                val answered = ints(e.getJSONArray("answered")).toSet()
                val cost = ReviewWords.estimate(game, str(e, "user"), str(e, "opening"), prices(e.getJSONArray("prices")), answered)
                same("$name estimate $i ($e)", cost, e.get("cost"), failures)
            }
        }
        assertTrue("$prompts prompts; ${failures.size} failures:\n" + failures.take(20).joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun theWordsAreTheDesktops() {
        val failures = ArrayList<String>()
        var requests = 0
        for ((g, game) in games) {
            val runs = g.getJSONArray("runs")
            for (i in 0 until runs.length()) {
                val r = runs.getJSONObject(i)
                val path = "${g.getString("name")} / ${r.getString("name")}"
                val user = str(r, "user")
                val fake = Scripted(r.getJSONObject("replies"))
                val kept = JSONObject()
                val keepFails = if (r.isNull("keep_fails")) null else r.getInt("keep_fails")
                val onAnswer: ((Int, String, Map<String, Any?>) -> Unit)? = if (r.getBoolean("keep")) {
                    { ply, text, stats ->
                        synchronized(kept) { kept.put("$ply", JSONObject().put("text", text).put("stats", JSONObject(stats))) }
                        if (ply == keepFails) throw IOException("the store is locked") // not kept only costs a resume
                    }
                } else {
                    null
                }
                try {
                    val words = ReviewWords.write(
                        game,
                        fake,
                        user = user,
                        opening = str(r, "opening"),
                        prices = prices(r.getJSONArray("prices")),
                        thinking = r.getBoolean("thinking"),
                        answered = answeredOf(r),
                        onAnswer = onAnswer,
                    )
                    if (r.has("raised")) {
                        failures.add("$path: python raised \"${r.getString("raised")}\", mine did not")
                    } else {
                        same("$path result", wordsJson(words), r.getJSONObject("result"), failures)
                        same("$path links", ReviewLinks.linked(textsJson(words), game, user), r.getJSONObject("links"), failures)
                    }
                } catch (e: Exception) {
                    if (!r.has("raised")) failures.add("$path: mine raised $e")
                    else if (e.message != r.getString("raised")) failures.add("$path: mine raised \"${e.message}\", python \"${r.getString("raised")}\"")
                }
                // The requests by the ply they ask about: the system prompt, the message, the limits.
                requests += fake.requests.size
                val system = data.getString("system")
                for ((ply, said) in fake.systems) if (said != system) failures.add("$path ply $ply system:\n  mine:   $said\n  python: $system")
                same("$path requests", JSONObject().apply { for ((ply, req) in fake.requests) put("$ply", req) }, r.getJSONObject("requests"), failures)
                same("$path on_answer", kept, r.getJSONObject("on_answer"), failures)
            }
        }
        assertTrue("$requests requests; ${failures.size} failures:\n" + failures.take(20).joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun theLinksAreTheDesktops() {
        val failures = ArrayList<String>()
        var paragraphs = 0
        for ((g, game) in games) {
            val links = g.getJSONArray("links")
            for (i in 0 until links.length()) {
                val l = links.getJSONObject(i)
                val texts = l.getJSONObject("texts")
                paragraphs += texts.keys().asSequence().sumOf { texts.getJSONObject(it).length() }
                val user = str(l, "user")
                same("${g.getString("name")} links for $user", ReviewLinks.linked(texts, game, user), l.getJSONObject("linked"), failures)
            }
        }
        assertTrue("$paragraphs paragraphs; ${failures.size} failures:\n" + failures.take(20).joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun theStudentsWordsTheHeadsAndTheChecksAreTheDesktops() {
        val failures = ArrayList<String>()
        val student = data.getJSONArray("to_student")
        for (i in 0 until student.length()) {
            val s = student.getJSONObject(i)
            val mine = ReviewWords.toStudent(s.getString("paragraph"), str(s, "user"))
            if (mine != s.getString("read")) {
                failures.add("to_student (${str(s, "user")}) \"${s.getString("paragraph")}\":\n  mine:   $mine\n  python: ${s.getString("read")}")
            }
        }
        val parsed = data.getJSONArray("parse")
        for (i in 0 until parsed.length()) {
            val p = parsed.getJSONObject(i)
            same("parse ${JSONObject.quote(p.getString("text"))}", JSONObject(ReviewWords.parse(p.getString("text"))), p.getJSONObject("parts"), failures)
        }
        val coloured = data.getJSONArray("in_colours")
        for (i in 0 until coloured.length()) {
            val c = coloured.getJSONObject(i)
            same("in_colours $i", json(ReviewWords.inColours(plain(c.get("value")), c.getString("mover"))), c.get("out"), failures)
        }
        val short = games.single { it.first.getString("name") == "the tests' short game" }.second
        val checks = data.getJSONArray("checks")
        for (i in 0 until checks.length()) {
            val c = checks.getJSONObject(i)
            val parts = strings(c.getJSONObject("parts"))
            val ply = c.getInt("ply")
            val user = str(c, "user")
            val (kept, dropped) = ReviewWords.checked(parts, short, ply, user)
            same("checked $i kept", JSONObject(kept), c.getJSONObject("kept"), failures)
            same("checked $i dropped", dropped, c.getString("dropped"), failures)
            val sides = JSONObject().apply { for ((part, text) in parts) put(part, ReviewWords.wrongSide(text, short, ply)) }
            same("checked $i wrong_side", sides, c.getJSONObject("wrong_side"), failures)
        }
        assertTrue("${failures.size} failures:\n" + failures.take(30).joinToString("\n"), failures.isEmpty())
    }

    /**
     * A model that reasons by its effort, on its own wire: told "none" for the words, its model's
     * name kept with them, and the words, the cost and the answers as the desktop's.
     */
    @Test
    fun aModelThatReasonsByItsEffortIsToldNone() {
        val failures = ArrayList<String>()
        val r = data.getJSONObject("effort")
        val game = games[r.getInt("game")].second
        val wire = ScriptedWire(r.getJSONObject("replies"))
        val provider = OpenAICompatProvider(
            "openai",
            "http://scripted",
            "scripted-luna",
            thinkingControl = ThinkingControl.EFFORT,
            reasoningEffort = "medium",
            transport = wire,
        )
        val kept = JSONObject()
        val words = ReviewWords.write(
            game,
            provider,
            user = str(r, "user"),
            opening = str(r, "opening"),
            prices = prices(r.getJSONArray("prices")),
            thinking = r.getBoolean("thinking"),
            answered = answeredOf(r),
            onAnswer = { ply, text, _ -> synchronized(kept) { kept.put("$ply", text) } },
        )
        val want = r.getJSONObject("result")
        val mine = wordsJson(words)
        for (key in listOf("texts", "dropped", "answers", "model", "cost")) same("effort $key", mine.get(key), want.get(key), failures)
        val texts = JSONObject().apply { val o = r.getJSONObject("on_answer"); for (k in o.keys()) put(k, o.getJSONObject(k).getString("text")) }
        same("effort on_answer", kept, texts, failures)
        same("effort efforts asked", JSONArray(wire.bodies.values.map { it.optString("reasoning_effort") }.distinct()), r.getJSONArray("efforts"), failures)
        val requests = r.getJSONObject("requests")
        same("effort plies asked", JSONArray(wire.bodies.keys.sorted()), JSONArray(requests.keys().asSequence().map { it.toInt() }.sorted().toList()), failures)
        for ((ply, body) in wire.bodies) {
            val messages = body.getJSONArray("messages")
            same("effort ply $ply system", messages.getJSONObject(0).getString("content"), data.getString("system"), failures)
            val content = requests.getJSONObject("$ply").getJSONArray("messages").getJSONObject(0).getString("content")
            same("effort ply $ply message", messages.getJSONObject(1).getString("content"), content, failures)
        }
        assertTrue("${failures.size} failures:\n" + failures.take(20).joinToString("\n"), failures.isEmpty())
    }

    companion object {
        /**
         * The fixture and its games, read once for the class rather than once a test, and let go
         * after it: the suite's other classes share the 192 MB heap.
         */
        private var read: JSONObject? = null
        private var reviewed: List<Pair<JSONObject, ReviewedGame>>? = null

        private val data: JSONObject get() = read ?: fixture("coach/review_words.json").also { read = it }

        private val games: List<Pair<JSONObject, ReviewedGame>>
            get() = reviewed ?: data.getJSONArray("games").let { all ->
                (0 until all.length()).map { all.getJSONObject(it) }.map { it to ReviewJson.reviewed(it.getJSONObject("reviewed")) }
            }.also { reviewed = it }

        @JvmStatic
        @AfterClass
        fun letGo() {
            read = null
            reviewed = null
        }

        private val PLY = Regex("\"key move\": \\{\"ply\": (\\d+)")

        /** The ply a request asks about, read from its message. */
        fun plyOf(content: String): Int =
            PLY.find(content)?.groupValues?.get(1)?.toInt() ?: throw IllegalStateException("no ply in $content")

        fun str(o: JSONObject, key: String): String? = if (o.isNull(key)) null else o.getString(key)

        fun ints(a: JSONArray): List<Int> = (0 until a.length()).map { a.getInt(it) }

        fun strings(o: JSONObject): Map<String, String> =
            LinkedHashMap<String, String>().apply { for (k in o.keys()) put(k, o.getString(k)) }

        fun prices(a: JSONArray): Prices {
            fun at(i: Int): Double? = if (a.isNull(i)) null else a.getDouble(i)
            return Prices(at(0), at(1), at(2))
        }

        fun answeredOf(r: JSONObject): Map<Int, String>? {
            if (r.isNull("answered")) return null
            val o = r.getJSONObject("answered")
            return LinkedHashMap<Int, String>().apply { for (k in o.keys()) put(k.toInt(), o.getString(k)) }
        }

        /** A reply's token counts as the provider hands them on: the JSON's own numbers. */
        fun statsOf(o: JSONObject): Map<String, Any?> =
            LinkedHashMap<String, Any?>().apply { for (k in o.keys()) put(k, o.get(k)) }

        /** The words as the fixture writes them (`words_json`): plies as strings. */
        fun wordsJson(words: ReviewWords.Words): JSONObject = JSONObject()
            .put("texts", textsJson(words))
            .put("dropped", JSONObject().apply { for ((p, why) in words.dropped) put("$p", why) })
            .put("answers", JSONObject().apply { for ((p, text) in words.answers) put("$p", text) })
            .put("model", words.model)
            .put("cost", words.cost ?: JSONObject.NULL)
            .put("calls", JSONArray(words.calls.map { JSONObject(it) }))

        fun textsJson(words: ReviewWords.Words): JSONObject =
            JSONObject().apply { for ((p, parts) in words.texts) put("$p", JSONObject(parts)) }

        /** A JSON value as Python's: a dict a map, a list a list, null None. */
        fun plain(value: Any?): Any? = when (value) {
            JSONObject.NULL, null -> null
            is JSONObject -> LinkedHashMap<String, Any?>().apply { for (k in value.keys()) put(k, plain(value.get(k))) }
            is JSONArray -> (0 until value.length()).map { plain(value.get(it)) }
            else -> value
        }

        /** And back. */
        fun json(value: Any?): Any? = when (value) {
            null -> JSONObject.NULL
            is Map<*, *> -> JSONObject().apply { for ((k, v) in value) put(k.toString(), json(v)) }
            is List<*> -> JSONArray().apply { for (v in value) put(json(v)) }
            else -> value
        }

        /** The two JSON values the same, numbers by value. */
        fun same(path: String, mine: Any?, theirs: Any?, failures: MutableList<String>) {
            if (failures.size >= 60) return
            val a = if (mine == JSONObject.NULL) null else mine
            val b = if (theirs == JSONObject.NULL) null else theirs
            when {
                a is JSONObject && b is JSONObject -> {
                    val keys = (a.keys().asSequence() + b.keys().asSequence()).toSortedSet()
                    for (k in keys) {
                        if (!a.has(k) || !b.has(k)) {
                            failures.add("$path.$k: ${if (a.has(k)) "only mine: ${a.get(k)}" else "only python's: ${b.get(k)}"}")
                        } else {
                            same("$path.$k", a.get(k), b.get(k), failures)
                        }
                    }
                }
                a is JSONArray && b is JSONArray -> {
                    if (a.length() != b.length()) failures.add("$path: ${a.length()} items, python ${b.length()}\n  mine:   $a\n  python: $b")
                    else for (i in 0 until a.length()) same("$path[$i]", a.get(i), b.get(i), failures)
                }
                a is Number && b is Number -> if (a.toDouble() != b.toDouble()) failures.add("$path: mine $a, python $b")
                else -> if (a != b) failures.add("$path:\n  mine:   $a\n  python: $b")
            }
        }
    }
}
