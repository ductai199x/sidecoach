package com.sidecoach.llm

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The wire, the ledger and the factory, held to the desktop's answer.
 *
 * Everything about `llm/openai_compat.py` except the socket is pure, so the request body
 * is compared as canonical JSON, the server's replies are the canned ones the generator
 * used -- fed through the real [OpenAICompatProvider.chat] and
 * [OpenAICompatProvider.chatStream] over a transport that answers from the fixture -- and
 * the ledger is driven record by record. `llm/factory.py` is resolved against the same
 * fake environment the generator used.
 */
@RunWith(AndroidJUnit4::class)
class LlmTest {

    private val failures = ArrayList<String>()

    private fun check(what: String, mine: Any?, theirs: Any?) {
        if (mine != theirs && failures.size < 10) {
            failures.add("$what:\n  kotlin: $mine\n  python: $theirs")
        }
    }

    @Test
    fun theWireAndTheLedgerMatchThePython() {
        val context = InstrumentationRegistry.getInstrumentation().context
        val text = context.assets.open("chess/llm.json").bufferedReader().use { it.readText() }
        val root = JSONObject(text)
        val began = System.currentTimeMillis()

        val conversations = root.getJSONArray("conversations").let { all ->
            (0 until all.length()).map { messagesFrom(all.getJSONArray(it)) }
        }
        val spec = root.getJSONObject("provider")

        // 1. The wire message list.
        val built = root.getJSONArray("build_messages")
        for (i in 0 until built.length()) {
            val one = built.getJSONObject(i)
            val control = ThinkingControl.of(one.getString("control"))!!
            val wire = buildMessages(
                "SYSTEM",
                conversations[one.getInt("conversation")],
                one.getBoolean("thinking"),
                control,
            )
            check(
                "build_messages ${one.getString("control")} " +
                    "thinking=${one.getBoolean("thinking")} #${one.getInt("conversation")}",
                canon(wire.map { linkedMapOf("role" to it.role, "content" to it.content) }),
                one.getString("wire"),
            )
        }

        // 2. The request body.
        val bodies = root.getJSONArray("request_bodies")
        for (i in 0 until bodies.length()) {
            val one = bodies.getJSONObject(i)
            val provider = providerOf(
                spec,
                one.getString("control"),
                if (one.isNull("effort")) null else one.getString("effort"),
                mapFrom(JSONObject(one.getString("extra"))),
                transportOf(200, "{}"),
            )
            val body = provider.requestBody(
                "SYSTEM",
                conversations[one.getInt("conversation")],
                one.getInt("max_tokens"),
                one.getDouble("temperature"),
                one.getBoolean("thinking"),
            )
            val label = "request_body ${one.getString("control")}/${one.optString("effort")} " +
                "thinking=${one.getBoolean("thinking")} max=${one.getInt("max_tokens")}"
            check(label, canon(body), one.getString("body"))
            check("$label chars", OpenAICompatProvider.chars(body), one.getInt("chars"))
        }

        // 3. What `chat` makes of each reply.
        val chats = root.getJSONArray("chats")
        for (i in 0 until chats.length()) {
            val one = chats.getJSONObject(i)
            val label = "chat ${one.getString("label")}"
            val provider = providerOf(
                spec,
                "none",
                null,
                emptyMap(),
                transportOf(one.getInt("status"), one.getString("reply")),
            )
            try {
                val completion = provider.chat(
                    "SYSTEM",
                    conversations[one.getInt("conversation")],
                    maxTokens = one.getInt("max_tokens"),
                )
                check("$label text", completion.text, one.optString("text"))
                check("$label reasoning", completion.reasoning, one.optString("reasoning"))
                check("$label stats", canon(withoutTimings(completion.stats)), one.getString("stats"))
                check("$label raised", null, if (one.has("error")) one.getString("error") else null)
            } catch (exc: Exception) {
                check(
                    "$label error",
                    "${exc::class.java.simpleName}: ${exc.message}",
                    pythonError(one.optString("error")),
                )
            }
            check("$label usage", usageOf(provider.usage), one.getJSONObject("usage").toMap())
        }

        // 4. What `chatStream` makes of each stream, including one closed early.
        val streams = root.getJSONArray("streams")
        for (i in 0 until streams.length()) {
            val one = streams.getJSONObject(i)
            val label = "stream ${one.getString("label")}"
            val provider = providerOf(
                spec,
                "none",
                null,
                emptyMap(),
                transportOf(one.getInt("status"), one.getString("reply")),
            )
            val keep = one.getInt("keep")
            val seen = ArrayList<String>()
            try {
                val completion = provider.chatStream(
                    "SYSTEM",
                    conversations[one.getInt("conversation")],
                    maxTokens = one.getInt("max_tokens"),
                ) { piece ->
                    seen.add(piece)
                    seen.size < keep
                }
                check("$label text", completion.text, one.optString("text"))
                check("$label reasoning", completion.reasoning, one.optString("reasoning"))
                check("$label stats", canon(withoutTimings(completion.stats)), one.getString("stats"))
                check("$label raised", null, if (one.has("error")) one.getString("error") else null)
            } catch (exc: Exception) {
                check(
                    "$label error",
                    "${exc::class.java.simpleName}: ${exc.message}",
                    pythonError(one.optString("error")),
                )
            }
            check("$label deltas", seen, strings(one.getJSONArray("deltas")))
            check("$label usage", usageOf(provider.usage), one.getJSONObject("usage").toMap())
        }

        // 4c. With tools (desktop c80b64b): the request with the engine tools' specs, the calls
        // and their results, and tool_choice; the calls read whole and streamed in pieces; the
        // thinking before </think> split off; under "prefill" a reply with tools read whole.
        val tools = root.getJSONObject("tools")
        val toolSpecs = com.sidecoach.coach.engineToolSpecs()
        check("tool specs", canon(toolSpecs), tools.getString("specs"))
        val turns = JSONArray(tools.getString("turns"))
        val toolBodies = tools.getJSONArray("bodies")
        for (i in 0 until toolBodies.length()) {
            val one = toolBodies.getJSONObject(i)
            val choice = if (one.isNull("tool_choice")) null else one.getString("tool_choice")
            val provider = providerOf(
                spec,
                one.getString("control"),
                if (one.isNull("effort")) null else one.getString("effort"),
                emptyMap(),
                transportOf(200, "{}"),
            )
            val body = provider.requestBody(
                "SYSTEM", conversations[1], 100, 0.1, false, toolSpecs, turns, choice,
            )
            check("tool body ${one.getString("control")} choice=$choice", canon(body), one.getString("body"))
        }
        fun calls(completion: Completion): List<List<String>> =
            completion.toolCalls.map { listOf(it.id, it.name, canon(it.arguments)) }
        fun expected(array: JSONArray): List<List<String>> = (0 until array.length()).map { k ->
            array.getJSONArray(k).let { listOf(it.getString(0), it.getString(1), it.getString(2)) }
        }
        val toolChats = tools.getJSONArray("chats")
        for (i in 0 until toolChats.length()) {
            val one = toolChats.getJSONObject(i)
            val label = "tool chat ${one.getString("label")}"
            val provider = providerOf(spec, "none", null, emptyMap(), transportOf(200, one.getString("reply")))
            val completion = provider.chatTools("SYSTEM", conversations[1], toolSpecs, JSONArray())
            check("$label text", completion.text, one.getString("text"))
            check("$label reasoning", completion.reasoning, one.getString("reasoning"))
            check("$label calls", calls(completion), expected(one.getJSONArray("calls")))
        }
        val toolStreams = tools.getJSONArray("streams")
        for (i in 0 until toolStreams.length()) {
            val one = toolStreams.getJSONObject(i)
            val label = "tool stream ${one.getString("control")}"
            val provider = providerOf(spec, one.getString("control"), null, emptyMap(), transportOf(200, one.getString("reply")))
            val heard = ArrayList<String>()
            val completion = provider.chatTools("SYSTEM", conversations[1], toolSpecs, JSONArray()) { piece ->
                heard.add(piece)
                true
            }
            check("$label text", completion.text, one.getString("text"))
            check("$label deltas", heard, (0 until one.getJSONArray("deltas").length()).map { one.getJSONArray("deltas").getString(it) })
            check("$label calls", calls(completion), expected(one.getJSONArray("calls")))
        }

        // 4b. What the healthcheck says of each answer to `GET /models`.
        val healths = root.getJSONArray("healthchecks")
        for (i in 0 until healths.length()) {
            val one = healths.getJSONObject(i)
            val provider = providerOf(spec, "none", null, emptyMap(), transportOf(one.getInt("status"), one.getString("reply")))
            check("healthcheck ${one.getString("label")}", provider.healthcheck(), one.getBoolean("ok") to one.getString("detail"))
        }

        // 5. The ledger, record by record.
        val priceSets = root.getJSONArray("prices").let { all ->
            (0 until all.length()).map { pricesFrom(all.getJSONObject(it)) }
        }
        for (i in 0 until priceSets.size) {
            val want = root.getJSONArray("prices").getJSONObject(i)
            check("prices #$i known", priceSets[i].known, want.getBoolean("known"))
        }
        val ledger = Usage()
        val steps = root.getJSONArray("ledger")
        for (i in 0 until steps.length()) {
            val step = steps.getJSONObject(i)
            ledger.record(
                mapFrom(JSONObject(step.getString("stats"))),
                promptChars = step.getInt("prompt_chars"),
                completionChars = step.getInt("completion_chars"),
            )
            check("ledger after #$i", usageOf(ledger), step.getJSONObject("after").toMap())
            val summaries = step.getJSONArray("summaries")
            for (k in priceSets.indices) {
                checkSummary(
                    "ledger summary #$i at prices #$k",
                    ledger.summary(priceSets[k]),
                    JSONObject(summaries.getString(k)),
                )
            }
        }

        // 6. Several ledgers as one.
        val combines = root.getJSONArray("combines")
        for (i in 0 until combines.length()) {
            val one = combines.getJSONObject(i)
            val parts = strings(one.getJSONArray("parts")).map { mapFrom(JSONObject(it)) }
            checkSummary("combine #$i", combine(parts), JSONObject(one.getString("out")))
        }

        // 7. Prices from the settings.
        val priced = root.getJSONArray("prices_for")
        for (i in 0 until priced.length()) {
            val one = priced.getJSONObject(i)
            val settings = settingsFrom(JSONObject(one.getString("settings")))
            val model = if (one.isNull("model")) null else one.getString("model")
            val prices = pricesFor(settings, model)
            val want = one.getJSONObject("prices")
            check(
                "prices_for ${one.getString("settings")} / $model",
                listOf(prices.prompt, prices.cached, prices.completion, prices.known),
                listOf(
                    if (want.isNull("prompt")) null else want.getDouble("prompt"),
                    if (want.isNull("cached")) null else want.getDouble("cached"),
                    if (want.isNull("completion")) null else want.getDouble("completion"),
                    want.getBoolean("known"),
                ),
            )
        }

        // 8. What the factory resolves, against the same fake environment.
        val env = mapFrom(root.getJSONObject("env"))
        val keys = KeyStore { name -> env[name]?.toString() }
        val settings = root.getJSONArray("settings")
        for (i in 0 until settings.length()) {
            val one = settings.getJSONObject(i)
            check(
                "settings ${one.getString("label")}",
                resolve(settingsFrom(JSONObject(one.getString("settings"))), keys),
                one.getJSONObject("resolved").toMap(),
            )
        }
        val specs = root.getJSONArray("specs")
        for (i in 0 until specs.length()) {
            val one = specs.getJSONObject(i)
            check(
                "spec ${one.getString("spec")}",
                resolveSpec(one.getString("spec"), keys),
                one.getJSONObject("resolved").toMap(),
            )
        }

        val took = System.currentTimeMillis() - began
        Log.i(TAG, "${built.length()} wires, ${bodies.length()} bodies, ${chats.length()} chats, " +
            "${streams.length()} streams, ${settings.length()} settings, $took ms")
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /**
     * The effort is the provider's: another effort is another provider of the same model and
     * endpoint, sharing the ledger, and the same effort is the provider itself; one that is no
     * "effort" provider is itself at any effort. The key goes out as `Authorization: Bearer`. The
     * offline provider's ledger is there and empty (`test_the_default_is_luna_an_effort_provider_and_local_is_a_preset`
     * and `test_the_factory_gives_the_offline_provider_and_it_reports_itself` in
     * tests/test_llm_factory.py, `test_usage_adds_up_over_calls_and_effort_copies` in
     * tests/test_llm_usage.py, `test_chat_parses_content_reasoning_and_usage` in
     * tests/test_llm_openai_compat.py; the parity audit, 2026-09-29).
     */
    @Test
    fun anEffortCopySharesTheLedgerAndTheKeyGoesAsABearer() {
        val seen = java.util.Collections.synchronizedList(ArrayList<Map<String, String>>())
        val reply = "{\"choices\": [{\"message\": {\"content\": \"Fine.\"}, \"finish_reason\": \"stop\"}], " +
            "\"usage\": {\"prompt_tokens\": 1000, \"prompt_tokens_details\": {\"cached_tokens\": 800}, " +
            "\"completion_tokens\": 50, \"completion_tokens_details\": {\"reasoning_tokens\": 30}}}"
        val transport = object : Transport {
            override fun get(url: String, headers: Map<String, String>, timeoutMs: Int): HttpReply {
                seen.add(headers)
                return HttpReply(200, "{\"data\": []}")
            }

            override fun post(url: String, headers: Map<String, String>, body: String, timeoutMs: Int): HttpReply {
                seen.add(headers)
                return HttpReply(200, reply)
            }

            override fun postLines(
                url: String,
                headers: Map<String, String>,
                body: String,
                timeoutMs: Int,
                onLine: (String) -> Boolean,
            ): HttpReply {
                seen.add(headers)
                return HttpReply(200, "")
            }
        }
        val provider = OpenAICompatProvider(
            "x", "http://h/v1", "m", apiKey = "secret",
            thinkingControl = ThinkingControl.EFFORT, reasoningEffort = "medium", transport = transport,
        )
        assertTrue("the same effort: the provider itself", provider.withEffort("medium") === provider)
        val routine = provider.withEffort("none")
        assertTrue(routine !== provider && routine.usage === provider.usage)
        assertEquals("none", routine.reasoningEffort)
        assertEquals(provider.model to provider.baseUrl, routine.model to routine.baseUrl)
        provider.chat("s", listOf(ChatMessage("user", "hi")))
        routine.chat("s", listOf(ChatMessage("user", "hi")))
        val u = provider.usage
        assertEquals(listOf(2, 2000, 1600, 100, 60), listOf(u.calls, u.prompt, u.cached, u.completion, u.reasoning))
        assertEquals(0, u.estimated)
        assertEquals(listOf("Bearer secret", "Bearer secret"), seen.map { it["Authorization"] })
        // Not an "effort" provider: itself at any effort.
        val local = OpenAICompatProvider("local", "http://h/v1", "m", thinkingControl = ThinkingControl.PREFILL, transport = transport)
        assertTrue(local.withEffort("none") === local)
        // No model at all: its ledger is there, and empty.
        val none = makeProvider(LLMSettings(provider = "none"))
        assertTrue(none is OfflineProvider)
        assertEquals(0, (none as OfflineProvider).usage.calls)
        assertTrue(none.withEffort("low") === none)
    }

    private fun checkSummary(label: String, mine: Map<String, Any?>, theirs: JSONObject) {
        check(
            "$label counts",
            canon(mine.filterKeys { it != "cost" }),
            canon(theirs.toMap().filterKeys { it != "cost" }),
        )
        check(
            "$label cost",
            mine["cost"],
            if (theirs.isNull("cost")) null else theirs.getDouble("cost"),
        )
    }

    // --- what the desktop's objects look like as plain maps -----------------------------------

    private fun resolve(settings: LLMSettings, keys: KeyStore): Map<String, Any?> = try {
        val provider = makeProvider(settings, keys = keys, transport = transportOf(200, "{}"))
        if (provider is OfflineProvider) {
            linkedMapOf<String, Any?>("name" to provider.name, "offline" to true)
        } else {
            val one = provider as OpenAICompatProvider
            linkedMapOf<String, Any?>(
                "name" to one.name,
                "base_url" to one.baseUrl,
                "model" to one.model,
                "thinking_control" to one.thinkingControl.slug,
                "reasoning_effort" to one.reasoningEffort,
                "timeout_s" to one.timeoutS,
                "extra_body" to canon(one.extraBody),
            )
        }
    } catch (exc: Exception) {
        linkedMapOf<String, Any?>("error" to "${pythonName(exc)}: ${exc.message}")
    }

    private fun resolveSpec(spec: String, keys: KeyStore): Map<String, Any?> = try {
        val one = providerFromSpec(spec, keys = keys, transport = transportOf(200, "{}"))
            as OpenAICompatProvider
        linkedMapOf<String, Any?>(
            "name" to one.name,
            "base_url" to one.baseUrl,
            "model" to one.model,
            "thinking_control" to one.thinkingControl.slug,
            "reasoning_effort" to one.reasoningEffort,
            "timeout_s" to one.timeoutS,
            "extra_body" to canon(one.extraBody),
        )
    } catch (exc: Exception) {
        linkedMapOf<String, Any?>("error" to "${pythonName(exc)}: ${exc.message}")
    }

    /** The exception python would have raised for the same refusal. */
    private fun pythonName(exc: Exception): String = when (exc) {
        is IllegalArgumentException -> "ValueError"
        is IllegalStateException -> "RuntimeError"
        else -> exc::class.java.simpleName
    }

    /** `RuntimeError: 429: ...` as the Kotlin writes it. */
    private fun pythonError(recorded: String): String =
        recorded.replaceFirst("RuntimeError: ", "RuntimeException: ")

    private fun usageOf(usage: Usage): Map<String, Any?> = linkedMapOf(
        "calls" to usage.calls,
        "estimated" to usage.estimated,
        "prompt" to usage.prompt,
        "cached" to usage.cached,
        "completion" to usage.completion,
        "reasoning" to usage.reasoning,
    )

    private fun withoutTimings(stats: Map<String, Any?>): Map<String, Any?> =
        stats.filterKeys { it != "elapsed_s" && it != "first_token_s" }

    private fun pricesFrom(o: JSONObject) = Prices(
        prompt = if (o.isNull("prompt")) null else o.getDouble("prompt"),
        cached = if (o.isNull("cached")) null else o.getDouble("cached"),
        completion = if (o.isNull("completion")) null else o.getDouble("completion"),
    )

    private fun settingsFrom(o: JSONObject): LLMSettings {
        val base = LLMSettings()
        return LLMSettings(
            provider = o.optString("provider", base.provider),
            model = if (o.has("model")) nullable(o, "model") else base.model,
            baseUrl = if (o.has("base_url")) nullable(o, "base_url") else base.baseUrl,
            apiKeyEnv = if (o.has("api_key_env")) nullable(o, "api_key_env") else base.apiKeyEnv,
            thinkingControl = if (o.has("thinking_control")) {
                nullable(o, "thinking_control")
            } else {
                base.thinkingControl
            },
            reasoningEffort = if (o.has("reasoning_effort")) {
                nullable(o, "reasoning_effort")
            } else {
                base.reasoningEffort
            },
            routineEffort = if (o.has("routine_effort")) {
                nullable(o, "routine_effort")
            } else {
                base.routineEffort
            },
            pricePromptPerM = if (o.has("price_prompt_per_m")) {
                o.getDouble("price_prompt_per_m")
            } else {
                null
            },
            priceCachedPerM = if (o.has("price_cached_per_m")) {
                o.getDouble("price_cached_per_m")
            } else {
                null
            },
            priceCompletionPerM = if (o.has("price_completion_per_m")) {
                o.getDouble("price_completion_per_m")
            } else {
                null
            },
            timeoutS = if (o.has("timeout_s")) o.getDouble("timeout_s") else base.timeoutS,
            extraBody = if (o.has("extra_body")) {
                mapFrom(o.getJSONObject("extra_body"))
            } else {
                base.extraBody
            },
        )
    }

    private fun nullable(o: JSONObject, key: String): String? =
        if (o.isNull(key)) null else o.getString(key)

    private fun providerOf(
        spec: JSONObject,
        control: String,
        effort: String?,
        extra: Map<String, Any?>,
        transport: Transport,
    ) = OpenAICompatProvider(
        spec.getString("name"),
        spec.getString("base_url"),
        spec.getString("model"),
        apiKey = spec.getString("api_key"),
        thinkingControl = ThinkingControl.of(control)!!,
        reasoningEffort = effort,
        extraBody = extra,
        transport = transport,
    )

    /** The canned reply, in place of the socket; the stream is the body split into lines. */
    private fun transportOf(status: Int, text: String) = object : Transport {
        override fun get(url: String, headers: Map<String, String>, timeoutMs: Int) =
            HttpReply(status, text)

        override fun post(
            url: String,
            headers: Map<String, String>,
            body: String,
            timeoutMs: Int,
        ) = HttpReply(status, text)

        override fun postLines(
            url: String,
            headers: Map<String, String>,
            body: String,
            timeoutMs: Int,
            onLine: (String) -> Boolean,
        ): HttpReply {
            if (status != 200) return HttpReply(status, text)
            if (text.isNotEmpty()) {
                for (line in text.split("\n")) if (!onLine(line)) break
            }
            return HttpReply(status, "")
        }
    }

    // --- json, the way python writes it -------------------------------------------------------

    private fun messagesFrom(array: JSONArray): List<ChatMessage> =
        (0 until array.length()).map { i ->
            val o = array.getJSONObject(i)
            ChatMessage(o.getString("role"), o.getString("content"))
        }

    private fun mapFrom(o: JSONObject): Map<String, Any?> {
        val out = LinkedHashMap<String, Any?>()
        for (key in o.keys()) out[key] = if (o.isNull(key)) null else o.opt(key)
        return out
    }

    private fun JSONObject.toMap(): Map<String, Any?> = mapFrom(this)

    private fun strings(array: JSONArray): List<String> =
        (0 until array.length()).map { array.getString(it) }

    /**
     * `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
     *
     * org.json's own writer escapes a forward slash, which python does not, so the
     * canonical form is written here rather than taken from `toString`.
     */
    private fun canon(value: Any?): String {
        val out = StringBuilder()
        write(value, out)
        return out.toString()
    }

    private fun write(value: Any?, out: StringBuilder) {
        when {
            value == null || value === JSONObject.NULL -> out.append("null")
            value is String -> quote(value, out)
            value is Boolean -> out.append(if (value) "true" else "false")
            value is Int || value is Long -> out.append(value.toString())
            value is Number -> out.append(value.toDouble().toString())
            value is Map<*, *> -> {
                out.append('{')
                val keys = value.keys.map { it.toString() }.sorted()
                for ((index, key) in keys.withIndex()) {
                    if (index > 0) out.append(',')
                    quote(key, out)
                    out.append(':')
                    write(value[key], out)
                }
                out.append('}')
            }
            value is List<*> -> {
                out.append('[')
                for ((index, one) in value.withIndex()) {
                    if (index > 0) out.append(',')
                    write(one, out)
                }
                out.append(']')
            }
            value is JSONObject -> {
                out.append('{')
                val keys = value.keys().asSequence().toList().sorted()
                for ((index, key) in keys.withIndex()) {
                    if (index > 0) out.append(',')
                    quote(key, out)
                    out.append(':')
                    write(if (value.isNull(key)) null else value.opt(key), out)
                }
                out.append('}')
            }
            value is JSONArray -> {
                out.append('[')
                for (index in 0 until value.length()) {
                    if (index > 0) out.append(',')
                    write(if (value.isNull(index)) null else value.opt(index), out)
                }
                out.append(']')
            }
            else -> quote(value.toString(), out)
        }
    }

    private fun quote(text: String, out: StringBuilder) {
        out.append('"')
        for (char in text) {
            when {
                char == '"' -> out.append("\\\"")
                char == '\\' -> out.append("\\\\")
                char == '\n' -> out.append("\\n")
                char == '\r' -> out.append("\\r")
                char == '\t' -> out.append("\\t")
                char == '\b' -> out.append("\\b")
                char == '\u000C' -> out.append("\\f")
                char < ' ' -> out.append(String.format("\\u%04x", char.code))
                else -> out.append(char)
            }
        }
        out.append('"')
    }

    companion object {
        private const val TAG = "sidecoach.llm"
    }
}
