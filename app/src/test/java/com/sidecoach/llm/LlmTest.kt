package com.sidecoach.llm

import com.sidecoach.TestAssets
import com.sidecoach.TestLog
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

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
class LlmTest {

    private val failures = ArrayList<String>()

    private fun check(what: String, mine: Any?, theirs: Any?) {
        if (mine != theirs && failures.size < 10) {
            failures.add("$what:\n  kotlin: $mine\n  python: $theirs")
        }
    }

    @Test
    fun theWireAndTheLedgerMatchThePython() {
        val text = TestAssets.text("chess/llm.json")
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
        // The rounds as the coach hands them on, each provider writing them on its own wire
        // (desktop ce06c4e: `ToolRound`, where the coach used to write the chat messages).
        val rounds = roundsFrom(tools.getJSONArray("rounds"))
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
                "SYSTEM", conversations[1], 100, 0.1, false, toolSpecs, rounds, choice,
            )
            val label = "tool body ${one.getString("control")} choice=$choice"
            check(label, canon(body), one.getString("body"))
            check("$label chars", OpenAICompatProvider.chars(body), one.getInt("chars"))
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
            val completion = provider.chatTools("SYSTEM", conversations[1], toolSpecs, emptyList())
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
            val completion = provider.chatTools("SYSTEM", conversations[1], toolSpecs, emptyList()) { piece ->
                heard.add(piece)
                true
            }
            check("$label text", completion.text, one.getString("text"))
            check("$label deltas", heard, (0 until one.getJSONArray("deltas").length()).map { one.getJSONArray("deltas").getString(it) })
            check("$label calls", calls(completion), expected(one.getJSONArray("calls")))
        }

        // 4d. OpenAI's Responses API (desktop ce06c4e): the body (instructions, input, the rounds
        // as the reply's own output items or as its text and calls, the tools flattened, store
        // false, the reasoning and its encrypted content only with an effort, never fewer than 16
        // output tokens), and what the provider makes of each reply, read whole and streamed.
        val responses = root.getJSONObject("responses")
        val responseRounds = roundsFrom(responses.getJSONArray("rounds"))
        val responseBodies = responses.getJSONArray("bodies")
        for (i in 0 until responseBodies.length()) {
            val one = responseBodies.getJSONObject(i)
            val tooled = one.getBoolean("tools")
            val choice = nullable(one, "tool_choice")
            val provider = providerOf(
                spec,
                one.getString("control"),
                nullable(one, "effort"),
                mapFrom(JSONObject(one.getString("extra"))),
                transportOf(200, "{}"),
                Wire.RESPONSES,
            )
            val body = provider.requestBody(
                "SYSTEM",
                conversations[one.getInt("conversation")],
                one.getInt("max_tokens"),
                one.getDouble("temperature"),
                one.getBoolean("thinking"),
                if (tooled) toolSpecs else null,
                if (tooled) responseRounds else emptyList(),
                choice,
            )
            val label = "responses body ${one.getString("control")}/${one.optString("effort")} " +
                "max=${one.getInt("max_tokens")} tools=$tooled choice=$choice extra=${one.getString("extra")}"
            check(label, canon(body), one.getString("body"))
            check("$label chars", OpenAICompatProvider.chars(body), one.getInt("chars"))
        }
        fun checkReply(label: String, completion: Completion, one: JSONObject) {
            check("$label text", completion.text, one.optString("text"))
            check("$label reasoning", completion.reasoning, one.optString("reasoning"))
            check("$label stats", canon(withoutTimings(completion.stats)), one.getString("stats"))
            check("$label calls", calls(completion), expected(one.getJSONArray("calls")))
            check("$label items", canon(completion.items), one.getString("items"))
        }
        val responseChats = responses.getJSONArray("chats")
        for (i in 0 until responseChats.length()) {
            val one = responseChats.getJSONObject(i)
            val label = "responses chat ${one.getString("label")}"
            val provider = providerOf(
                spec,
                "effort",
                "medium",
                emptyMap(),
                transportOf(one.getInt("status"), one.getString("reply")),
                Wire.RESPONSES,
            )
            try {
                val completion = provider.chatTools(
                    "SYSTEM",
                    conversations[one.getInt("conversation")],
                    toolSpecs,
                    emptyList(),
                    maxTokens = one.getInt("max_tokens"),
                )
                checkReply(label, completion, one)
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
        val responseStreams = responses.getJSONArray("streams")
        for (i in 0 until responseStreams.length()) {
            val one = responseStreams.getJSONObject(i)
            val label = "responses stream ${one.getString("label")}"
            val provider = providerOf(
                spec,
                "effort",
                "medium",
                emptyMap(),
                transportOf(one.getInt("status"), one.getString("reply")),
                Wire.RESPONSES,
            )
            val keep = one.getInt("keep")
            val seen = ArrayList<String>()
            try {
                val completion = provider.chatTools(
                    "SYSTEM",
                    conversations[one.getInt("conversation")],
                    toolSpecs,
                    emptyList(),
                    maxTokens = one.getInt("max_tokens"),
                ) { piece ->
                    seen.add(piece)
                    seen.size < keep
                }
                checkReply(label, completion, one)
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
        TestLog.i(TAG, "${built.length()} wires, ${bodies.length()} bodies, ${chats.length()} chats, " +
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

    /**
     * OpenAI's reasoning models take tools with an effort only on /v1/responses; every other
     * preset speaks chat completions, and an effort copy keeps the wire (the wire in
     * `test_the_default_is_luna_an_effort_provider_and_local_is_a_preset` in
     * tests/test_llm_factory.py, desktop ce06c4e).
     */
    @Test
    fun theOpenAIPresetSpeaksResponsesAndEveryOtherChat() {
        val keys = KeyStore { name -> if (name == "OPENAI_API_KEY") "sk-openai" else null }
        val provider = makeProvider(LLMSettings(provider = "openai"), keys = keys) as OpenAICompatProvider
        val quiet = provider.withEffort("none")
        assertTrue(quiet !== provider)
        assertEquals(Wire.RESPONSES to Wire.RESPONSES, provider.wire to quiet.wire)
        val local = makeProvider(LLMSettings(provider = "local"), keys = keys) as OpenAICompatProvider
        assertEquals(ThinkingControl.PREFILL to Wire.CHAT, local.thinkingControl to local.wire)
    }

    /**
     * GPT-5.6 and later bill a prompt written to the server's cache at 1.25x the fresh price
     * (OpenAI's prompt-caching guide, 2026-09-29): 1000 prompt tokens, 200 read from the cache,
     * 500 written to it, 300 fresh (`test_a_prompt_written_to_the_cache_costs_a_quarter_more` in
     * tests/test_llm_usage.py).
     */
    @Test
    fun aPromptWrittenToTheCacheCostsAQuarterMore() {
        val usage = Usage()
        usage.record(
            mapOf(
                "prompt_tokens" to 1000,
                "cached_tokens" to 200,
                "cache_write_tokens" to 500,
                "completion_tokens" to 0,
            ),
        )
        assertEquals(500, usage.written)
        // 300 fresh at $1, 500 written at $1.25, 200 cached at $0.1, per million.
        assertEquals(0.0003 + 0.000625 + 0.00002, usage.cost(Prices(1.0, 0.1, 8.0)) ?: 0.0, 1e-12)
    }

    // ---- OpenAI's Responses API: the items as the API returned them (2026-09-29) --------------

    /**
     * Luna at an effort takes function tools only on /v1/responses (chat completions refused
     * them, 2026-09-29): stateless, its reasoning comes back encrypted in its output items, which
     * go back before what each call returned
     * (`test_the_responses_wire_sends_the_reasoning_back_with_the_tools_results` in
     * tests/test_llm_openai_compat.py).
     */
    @Test
    fun theResponsesWireSendsTheReasoningBackWithTheToolsResults() {
        val call = ToolCall("call_1", "analyse", JSONObject().put("moves", JSONArray(listOf("e4", "e5", "Nf3"))))
        val asked = Completion(text = "", toolCalls = listOf(call), items = listOf(JSONObject(REASONING), JSONObject(FUNCTION_CALL)))
        val body = responses().requestBody(
            "sys",
            USER,
            10,
            0.1,
            false,
            tools = JSONArray().put(JSONObject(TOOL)),
            turns = listOf(ToolRound(asked, listOf("Standing: about equal."))),
            toolChoice = "none",
        )
        assertEquals("sys", body.getString("instructions"))
        assertFalse(body.has("messages"))
        assertEquals(
            canon(
                listOf(
                    mapOf("role" to "user", "content" to "hi"),
                    JSONObject(REASONING),
                    JSONObject(FUNCTION_CALL),
                    mapOf("type" to "function_call_output", "call_id" to "call_1", "output" to "Standing: about equal."),
                ),
            ),
            canon(body.getJSONArray("input")),
        )
        // The function's fields at the top of the tool, strict off as on the chat wire.
        assertEquals(
            canon(listOf(mapOf("type" to "function", "name" to "analyse", "parameters" to mapOf("type" to "object"), "strict" to false))),
            canon(body.getJSONArray("tools")),
        )
        assertEquals("none", body.getString("tool_choice"))
        assertEquals(false, body.getBoolean("store"))
        assertEquals(canon(mapOf("effort" to "medium")), canon(body.getJSONObject("reasoning")))
        assertEquals(canon(listOf("reasoning.encrypted_content")), canon(body.getJSONArray("include")))
        assertEquals(10 + 4000, body.getInt("max_output_tokens"))
        assertFalse(body.has("temperature"))
        // A reply with no items of its own (another provider's): its calls written out.
        val plain = responses().requestBody("sys", USER, 10, 0.1, false, turns = TURNS)
        val input = plain.getJSONArray("input")
        assertEquals(
            canon(
                listOf(
                    mapOf("type" to "function_call", "call_id" to "c1", "name" to "analyse", "arguments" to "{}"),
                    mapOf("type" to "function_call_output", "call_id" to "c1", "output" to "From the board now: ..."),
                ),
            ),
            canon((1 until input.length()).map { input.get(it) }),
        )
        assertFalse(plain.has("tools") || plain.has("tool_choice"))
    }

    /** `test_the_responses_wire_reads_the_calls_the_reply_and_the_usage` in tests/test_llm_openai_compat.py. */
    @Test
    fun theResponsesWireReadsTheCallsTheReplyAndTheUsage() {
        val replies = listOf(
            """{"status": "completed", "output": [$REASONING, $FUNCTION_CALL], "usage": $USAGE}""",
            """{"status": "completed", "output": [$MESSAGE], "usage": $USAGE}""",
            """{"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": [$REASONING], "usage": $USAGE}""",
        )
        val paths = ArrayList<String>()
        val provider = responses(answering(replies.map { 200 to it }, paths))
        val asked = provider.chatTools("sys", USER, JSONArray().put(JSONObject(TOOL)), emptyList())
        assertEquals(listOf("/v1/responses"), paths)
        assertEquals(
            listOf(listOf("call_1", "analyse", canon(mapOf("moves" to listOf("e4", "e5", "Nf3"))))),
            asked.toolCalls.map { listOf(it.id, it.name, canon(it.arguments)) },
        )
        assertEquals("", asked.text)
        assertEquals(canon(listOf(JSONObject(REASONING), JSONObject(FUNCTION_CALL))), canon(asked.items))
        assertTrue("elapsed_s" in asked.stats)
        assertEquals(
            canon(
                mapOf(
                    "prompt_tokens" to 372,
                    "cached_tokens" to 100,
                    "cache_write_tokens" to 0,
                    "completion_tokens" to 38,
                    "reasoning_tokens" to 9,
                    "finish_reason" to "tool_calls",
                ),
            ),
            canon(asked.stats.filterKeys { it != "elapsed_s" }),
        )
        val answered = provider.chat("sys", USER)
        assertEquals("About equal." to "stop", answered.text to answered.stats["finish_reason"])
        val cut = provider.chat("sys", USER)
        assertEquals("" to "length", cut.text to cut.stats["finish_reason"])
    }

    /** `test_the_responses_wire_streams_the_text_and_ends_on_the_completed_response` in tests/test_llm_openai_compat.py. */
    @Test
    fun theResponsesWireStreamsTheTextAndEndsOnTheCompletedResponse() {
        val provider = responses(answering(listOf(200 to STREAMED, 200 to STREAMED)))
        val heard = ArrayList<String>()
        val got = provider.chatStream("sys", USER) { piece ->
            heard.add(piece)
            true
        }
        assertEquals(listOf("About", " equal", "."), heard)
        assertEquals("About equal.", got.text)
        assertEquals(canon(listOf(JSONObject(MESSAGE))), canon(got.items))
        assertEquals(true, got.stats["streamed"])
        assertTrue((got.stats["first_token_s"] as Double) >= 0)
        assertEquals("stop" to 372, got.stats["finish_reason"] to got.stats["prompt_tokens"])
        // Closed after the first piece: the text so far, the usage unknown.
        val stopped = provider.chatStream("sys", USER) { false }
        assertEquals("About" to "stopped", stopped.text to stopped.stats["finish_reason"])
        assertNull(stopped.stats["prompt_tokens"])
    }

    /** `test_an_error_in_a_responses_stream_raises` in tests/test_llm_openai_compat.py. */
    @Test
    fun anErrorInAResponsesStreamRaises() {
        val failing = events(
            """{"type": "response.created", "response": {"status": "in_progress"}}""",
            """{"type": "error", "code": "rate_limit_exceeded", "message": "slow down"}""",
        )
        val said = raised { responses(answering(listOf(200 to failing))).chatStream("sys", USER) { true } }
        assertTrue(said, "rate_limit_exceeded: slow down" in said)
        val refused = raised { responses(answering(listOf(400 to "bad tools"))).chatStream("sys", USER) { true } }
        assertTrue(refused, refused.startsWith("400: bad tools"))
        // Cut off before the response completed: an error, not a reply that ends mid-sentence.
        val cut = events(
            """{"type": "response.created", "response": {"status": "in_progress"}}""",
            """{"type": "response.output_text.delta", "output_index": 0, "delta": "About"}""",
        )
        val dropped = raised { responses(answering(listOf(200 to cut))).chatStream("sys", USER) { true } }
        assertTrue(dropped, "stream ended before the response completed" in dropped)
        // A failed response read whole raises too, as the stream's failure does.
        val failed = """{"status": "failed", "error": {"code": "server_error", "message": "try again"}}"""
        val whole = raised { responses(answering(listOf(200 to failed))).chat("sys", USER) }
        assertTrue(whole, "failed: server_error: try again" in whole)
    }

    /**
     * A tool-heavy turn may open with a preamble (phase "commentary"): no part of the reply, and
     * never to the voice -- but it goes back with the round, phase and all (OpenAI's reasoning
     * guide; `test_a_preamble_before_the_calls_is_never_said_and_goes_back_with_them` in
     * tests/test_llm_openai_compat.py).
     */
    @Test
    fun aPreambleBeforeTheCallsIsNeverSaidAndGoesBackWithThem() {
        val stream = events(
            """{"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "phase": "commentary", "content": []}}""",
            """{"type": "response.output_text.delta", "output_index": 0, "delta": "Let me ask the engine."}""",
            """{"type": "response.output_item.done", "output_index": 0, "item": $PREAMBLE}""",
            """{"type": "response.output_item.done", "output_index": 1, "item": $FUNCTION_CALL}""",
            """{"type": "response.completed", "response": {"status": "completed", "output": [$PREAMBLE, $FUNCTION_CALL], "usage": $USAGE}}""",
        )
        val heard = ArrayList<String>()
        val got = responses(answering(listOf(200 to stream))).chatStream("sys", USER) { piece ->
            heard.add(piece)
            true
        }
        assertTrue(heard.toString(), heard.isEmpty())
        assertEquals("", got.text)
        assertEquals(listOf("analyse"), got.toolCalls.map { it.name })
        assertEquals(canon(listOf(JSONObject(PREAMBLE), JSONObject(FUNCTION_CALL))), canon(got.items))
        val whole = """{"status": "completed", "output": [$PREAMBLE, $MESSAGE], "usage": $USAGE}"""
        assertEquals("About equal.", responses(answering(listOf(200 to whole))).chat("sys", USER).text)
    }

    /** `test_a_call_cut_short_is_not_run_and_a_refusal_is_on_the_record` in tests/test_llm_openai_compat.py. */
    @Test
    fun aCallCutShortIsNotRunAndARefusalIsOnTheRecord() {
        val cutCall = JSONObject(FUNCTION_CALL).put("status", "incomplete").put("arguments", "{\"moves\":[\"e4\",")
        val refusal = JSONObject(MESSAGE)
            .put("content", JSONArray().put(JSONObject().put("type", "refusal").put("refusal", "I can't help with that.")))
        val replies = listOf(
            JSONObject()
                .put("status", "incomplete")
                .put("incomplete_details", JSONObject().put("reason", "max_output_tokens"))
                .put("output", JSONArray().put(cutCall))
                .put("usage", JSONObject(USAGE)),
            JSONObject().put("status", "completed").put("output", JSONArray().put(refusal)).put("usage", JSONObject(USAGE)),
        )
        val provider = responses(answering(replies.map { 200 to it.toString() }))
        val cut = provider.chatTools("sys", USER, JSONArray().put(JSONObject(TOOL)), emptyList())
        assertTrue(cut.toolCalls.isEmpty())
        assertEquals("length", cut.stats["finish_reason"])
        val refused = provider.chat("sys", USER)
        assertEquals("" to "I can't help with that.", refused.text to refused.stats["refusal"])
    }

    /**
     * Fewer output tokens than 16 the API refuses; a provider with no effort gets no headroom,
     * and a caller's tiny budget (a yes/no judge) is raised to the floor
     * (`test_the_responses_wire_asks_for_at_least_sixteen_tokens` in tests/test_llm_openai_compat.py).
     */
    @Test
    fun theResponsesWireAsksForAtLeastSixteenTokens() {
        val body = responses(effort = null).requestBody("sys", USER, 5, 0.1, false)
        assertEquals(16, body.getInt("max_output_tokens"))
        assertFalse(body.has("reasoning"))
    }

    /** `_responses`: Luna on the Responses wire at [effort], answering from [transport]. */
    private fun responses(transport: Transport = answering(emptyList()), effort: String? = "medium") =
        OpenAICompatProvider(
            "openai",
            "http://h/v1",
            "gpt-5.6-luna",
            thinkingControl = ThinkingControl.EFFORT,
            reasoningEffort = effort,
            wire = Wire.RESPONSES,
            transport = transport,
        )

    /**
     * The canned replies in turn, (status, body), in place of `httpx.MockTransport`: a stream is
     * the body split into lines. Each path asked for goes into [paths].
     */
    private fun answering(replies: List<Pair<Int, String>>, paths: MutableList<String>? = null) = object : Transport {
        private val queue = ArrayDeque(replies)

        override fun get(url: String, headers: Map<String, String>, timeoutMs: Int) = HttpReply(200, "{\"data\": []}")

        override fun post(url: String, headers: Map<String, String>, body: String, timeoutMs: Int): HttpReply {
            paths?.add(java.net.URI(url).path)
            val (status, text) = queue.removeFirst()
            return HttpReply(status, text)
        }

        override fun postLines(
            url: String,
            headers: Map<String, String>,
            body: String,
            timeoutMs: Int,
            onLine: (String) -> Boolean,
        ): HttpReply {
            paths?.add(java.net.URI(url).path)
            val (status, text) = queue.removeFirst()
            if (status != 200) return HttpReply(status, text)
            for (line in text.split("\n")) if (!onLine(line)) break
            return HttpReply(status, "")
        }
    }

    /** What [block] raised, as a provider raises (RuntimeError on the desktop); "" when nothing was. */
    private fun raised(block: () -> Unit): String = try {
        block()
        ""
    } catch (exc: RuntimeException) {
        exc.message ?: ""
    }

    /** The rounds of calls as the generator recorded them (`round_json`). */
    private fun roundsFrom(array: JSONArray): List<ToolRound> = (0 until array.length()).map { i ->
        val one = array.getJSONObject(i)
        val calls = one.getJSONArray("calls").let { all ->
            (0 until all.length()).map { k ->
                all.getJSONArray(k).let { ToolCall(it.getString(0), it.getString(1), JSONObject(it.getString(2))) }
            }
        }
        val items = strings(one.getJSONArray("items")).map { JSONObject(it) }
        ToolRound(Completion(text = one.getString("text"), toolCalls = calls, items = items), strings(one.getJSONArray("results")))
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
                "wire" to one.wire.slug,
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
            "wire" to one.wire.slug,
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
        "written" to usage.written,
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
        wire: Wire = Wire.CHAT,
    ) = OpenAICompatProvider(
        spec.getString("name"),
        spec.getString("base_url"),
        spec.getString("model"),
        apiKey = spec.getString("api_key"),
        thinkingControl = ThinkingControl.of(control)!!,
        reasoningEffort = effort,
        wire = wire,
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

        // tests/test_llm_openai_compat.py's conversation, tool and round.
        private val USER = listOf(ChatMessage("user", "hi"))
        private const val TOOL = """{"type": "function", "function": {"name": "analyse", "parameters": {"type": "object"}}}"""
        private val TURNS = listOf(
            ToolRound(
                Completion(text = "", toolCalls = listOf(ToolCall("c1", "analyse", JSONObject()))),
                listOf("From the board now: ..."),
            ),
        )

        // OpenAI's Responses API: the items as the API returned them (2026-09-29).
        private const val REASONING =
            """{"id": "rs_1", "type": "reasoning", "content": [], "encrypted_content": "gAAAAABqu-ri...", "summary": []}"""
        private const val FUNCTION_CALL =
            """{"id": "fc_1", "type": "function_call", "status": "completed", "arguments": "{\"moves\":[\"e4\",\"e5\",\"Nf3\"]}", "call_id": "call_1", "name": "analyse"}"""
        private const val MESSAGE =
            """{"id": "msg_1", "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": "About equal."}]}"""
        private const val PREAMBLE =
            """{"id": "msg_0", "type": "message", "status": "completed", "role": "assistant", "phase": "commentary", "content": [{"type": "output_text", "annotations": [], "text": "Let me ask the engine."}]}"""
        private const val USAGE =
            """{"input_tokens": 372, "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 100}, "output_tokens": 38, "output_tokens_details": {"reasoning_tokens": 9}, "total_tokens": 410}"""

        /** A stream as the API writes it: each event's type, then its data (`_events`). */
        private fun events(vararg events: String): String =
            events.joinToString("") { "event: ${JSONObject(it).getString("type")}\ndata: $it\n\n" }

        private val STREAMED = events(
            """{"type": "response.created", "response": {"status": "in_progress", "output": []}}""",
            """{"type": "response.output_text.delta", "output_index": 0, "delta": "About"}""",
            """{"type": "response.output_text.delta", "output_index": 0, "delta": " equal"}""",
            """{"type": "response.output_text.delta", "output_index": 0, "delta": "."}""",
            """{"type": "response.output_item.done", "output_index": 0, "item": $MESSAGE}""",
            """{"type": "response.completed", "response": {"status": "completed", "output": [$MESSAGE], "usage": $USAGE}}""",
        )
    }
}
