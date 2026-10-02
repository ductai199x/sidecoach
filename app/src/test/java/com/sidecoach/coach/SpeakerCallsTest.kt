package com.sidecoach.coach

import com.sidecoach.chess.Board
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.Trigger
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.Completion
import com.sidecoach.llm.HttpReply
import com.sidecoach.llm.LLMProvider
import com.sidecoach.llm.OpenAICompatProvider
import com.sidecoach.llm.Transport
import java.net.ConnectException
import java.net.SocketTimeoutException
import java.util.concurrent.CountDownLatch
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * What the Speaker does when a model call fails or the Speaker is closed: every paid call
 * counted, none made twice for one answer or after the session is gone. The desktop's
 * `tests/test_speaker_calls.py`.
 */
class SpeakerCallsTest {

    private val okReply = """{"choices": [{"message": {"content": "Develop the knight to f3."}}]}"""

    /** Answers each POST from [replies] in turn (a reply or a thrown failure), then [okReply]. */
    private class Scripted(val replies: MutableList<Any>, val ok: String) : Transport {
        val bodies = ArrayList<String>()

        private fun next(body: String): HttpReply {
            bodies.add(body)
            val reply = if (replies.isEmpty()) HttpReply(200, ok) else replies.removeAt(0)
            if (reply is Exception) throw reply
            return reply as HttpReply
        }

        override fun get(url: String, headers: Map<String, String>, timeoutMs: Int) = HttpReply(200, "{}")

        override fun post(url: String, headers: Map<String, String>, body: String, timeoutMs: Int) = next(body)

        override fun postLines(
            url: String,
            headers: Map<String, String>,
            body: String,
            timeoutMs: Int,
            onLine: (String) -> Boolean,
        ): HttpReply = next(body)
    }

    private fun provider(transport: Transport, timeoutS: Double = 60.0) =
        OpenAICompatProvider("x", "http://h/v1", "m", timeoutS = timeoutS, transport = transport)

    private fun request(trigger: Trigger = Trigger.USER_ASK) = CoachRequest(
        trigger = trigger,
        level = DisclosureLevel.LINE,
        system = "You are a coach.",
        messages = listOf(ChatMessage("user", "What now?")),
        facts = PositionFacts(fen = SpeakerTest.FEN, sideToMove = "white", phase = "opening", material = "level"),
        board = Board(SpeakerTest.FEN),
        fallback = "TEMPLATE",
        ply = 6,
        gameId = "g",
        candidates = 1,
    )

    /** A timeout, a refused key, a refused connection: one call, the template, no complaint. */
    @Test
    fun aFailedCallIsNotADraftAskedAgain() {
        val failures = listOf<Any>(
            HttpReply(401, """{"error": {"message": "Incorrect API key provided"}}"""),
            SocketTimeoutException("Read timed out"),
            ConnectException("Connection refused"),
        )
        for (failure in failures) {
            val transport = Scripted(mutableListOf(failure, failure), okReply)
            val spoken = Speaker(provider(transport)).speakNow(request())
            assertEquals("$failure", 1, transport.bodies.size)
            assertEquals("TEMPLATE", spoken.message.text)
            assertTrue(spoken.problems.isNotEmpty() && "empty reply" !in spoken.problems)
        }
    }

    /**
     * A rate limit is waited out once, inside the provider's timeout, with the same request. Time
     * is the test's: a clock it moves and a wait it records, no real delay.
     */
    @Test
    fun aRateLimitIsWaitedOutOnceWithinTheTimeout() {
        var now = 100_000_000_000L
        val waits = ArrayList<Double>()

        fun speaker(transport: Transport, timeoutS: Double = 60.0, closedMeanwhile: Boolean = false) =
            Speaker(provider(transport, timeoutS)).apply {
                clockNs = { now }
                pause = { seconds ->
                    waits.add(seconds)
                    now += (seconds * 1e9).toLong()
                    closedMeanwhile
                }
            }

        var transport = Scripted(mutableListOf(HttpReply(429, "Rate limit reached")), okReply)
        var spoken = speaker(transport).speakNow(request())
        assertEquals(listOf(RATE_BACKOFF_S), waits)
        assertEquals(2, transport.bodies.size)
        assertEquals(transport.bodies[0], transport.bodies[1])
        assertEquals("Develop the knight to f3.", spoken.message.text)
        // No room left in the provider's timeout for the wait: the template after one call.
        waits.clear()
        transport = Scripted(mutableListOf(HttpReply(429, "Rate limit reached")), okReply)
        spoken = speaker(transport, timeoutS = RATE_BACKOFF_S).speakNow(request())
        assertTrue(waits.isEmpty())
        assertEquals(1, transport.bodies.size)
        assertEquals("TEMPLATE", spoken.message.text)
        // Closed during the wait: the template, and no second call.
        transport = Scripted(mutableListOf(HttpReply(429, "Rate limit reached")), okReply)
        spoken = speaker(transport, closedMeanwhile = true).speakNow(request())
        assertEquals(1, transport.bodies.size)
        assertEquals("TEMPLATE", spoken.message.text)
    }

    /** A call lost after it reached the server is billed: counted; a refused connection is not. */
    @Test
    fun aCallLostAfterItWasSentIsCounted() {
        val lost = provider(Scripted(mutableListOf(SocketTimeoutException("Read timed out")), okReply))
        runCatching { lost.chat("sys", listOf(ChatMessage("user", "hi ".repeat(100)))) }
        assertEquals(1, lost.usage.calls)
        assertEquals(1, lost.usage.estimated)
        assertTrue(lost.usage.prompt > 0)
        val refused = provider(Scripted(mutableListOf(ConnectException("Connection refused")), okReply))
        runCatching { refused.chat("sys", listOf(ChatMessage("user", "hi"))) }
        assertEquals(0, refused.usage.calls)
        // Never connected either, though a SocketException: no route to the host.
        val unrouted = provider(Scripted(mutableListOf(java.net.NoRouteToHostException("No route")), okReply))
        runCatching { unrouted.chat("sys", listOf(ChatMessage("user", "hi"))) }
        assertEquals(0, unrouted.usage.calls)
    }

    /**
     * A model changed mid-session and a verifier on its own endpoint: every ledger still counts,
     * each priced as its own model (`test_replaced_providers_are_counted_and_closed`,
     * `test_each_ledger_is_priced_as_its_own_model`).
     */
    @Test
    fun replacedProvidersAreCountedAndEachLedgerPricedAsItsModel() {
        fun named(model: String) = OpenAICompatProvider(
            "x", "http://h/v1", model, transport = Scripted(mutableListOf(), okReply),
        )
        val first = named("local-model")
        val verifier = named("gpt-5.6-luna")
        val second = named("local-model")
        val speaker = Speaker(first, verifier = verifier)
        first.chat("s", listOf(ChatMessage("user", "hi")))
        verifier.usage.record(mapOf("prompt_tokens" to 1_000_000, "completion_tokens" to 0))
        speaker.replaceProviders(second)
        second.chat("s", listOf(ChatMessage("user", "hi")))
        assertEquals(3, speaker.usages().sumOf { it.calls })
        // The coach's local model has no price; the verifier's luna is priced as luna.
        val luna = com.sidecoach.llm.total(listOf(verifier.usage, verifier.usage), com.sidecoach.llm.LLMSettings(), "local-model")
        assertEquals(1, luna["calls"])
        assertEquals(com.sidecoach.llm.KNOWN_PRICES.getValue("gpt-5.6-luna").prompt!!, luna["cost"] as Double, 1e-9)
    }

    /** A model whose call waits for the test to let it finish. */
    private class Held : LLMProvider {
        override val name = "held"
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)

        @Volatile
        var calls = 0

        override fun healthcheck(): Pair<Boolean, String> = true to "held"

        override fun chat(
            system: String,
            messages: List<ChatMessage>,
            maxTokens: Int,
            temperature: Double,
            thinking: Boolean,
        ): Completion {
            calls += 1
            started.countDown()
            release.await()
            return Completion(text = "Fine move.")
        }
    }

    /** Closing drops what is queued: the call in flight finishes, nothing behind it is asked. */
    @Test
    fun closingDropsWhatIsQueued() {
        val model = Held()
        val speaker = Speaker(model)
        speaker.start()
        for (trigger in listOf(Trigger.POST_MOVE, Trigger.USER_ASK, Trigger.USER_ASK, Trigger.USER_ASK)) {
            speaker.submit(request(trigger))
        }
        model.started.await()
        val closer = Thread { speaker.close(60000) }
        closer.start()
        try {
            speaker.awaitClosing() // close has emptied the queue and waits on the call in flight
        } finally {
            model.release.countDown()
        }
        closer.join()
        assertEquals(1, model.calls)
        assertEquals(0, speaker.pending)
    }
}
