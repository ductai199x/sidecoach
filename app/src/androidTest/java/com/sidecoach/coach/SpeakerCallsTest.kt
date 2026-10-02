package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
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
import java.util.concurrent.TimeUnit
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * What the Speaker does when a model call fails or the Speaker is closed: every paid call
 * counted, none made twice for one answer or after the session is gone. The desktop's
 * `tests/test_speaker_calls.py`.
 */
@RunWith(AndroidJUnit4::class)
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

    /** A rate limit is waited out once, inside the provider's timeout, with the same request. */
    @Test
    fun aRateLimitIsWaitedOutOnceWithinTheTimeout() {
        val before = RATE_BACKOFF_S
        try {
            RATE_BACKOFF_S = 0.05
            var transport = Scripted(mutableListOf(HttpReply(429, "Rate limit reached")), okReply)
            var spoken = Speaker(provider(transport)).speakNow(request())
            assertEquals(2, transport.bodies.size)
            assertEquals(transport.bodies[0], transport.bodies[1])
            assertEquals("Develop the knight to f3.", spoken.message.text)
            // No time left for the wait: the template after one call.
            RATE_BACKOFF_S = 5.0
            transport = Scripted(mutableListOf(HttpReply(429, "Rate limit reached")), okReply)
            spoken = Speaker(provider(transport, timeoutS = 1.0)).speakNow(request())
            assertEquals(1, transport.bodies.size)
            assertEquals("TEMPLATE", spoken.message.text)
        } finally {
            RATE_BACKOFF_S = before
        }
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
    }

    /** A model that takes a while a call, and says when it has started one. */
    private class Slow : LLMProvider {
        override val name = "slow"
        val started = CountDownLatch(1)

        @Volatile
        var calls = 0

        override fun healthcheck(): Pair<Boolean, String> = true to "slow"

        override fun chat(
            system: String,
            messages: List<ChatMessage>,
            maxTokens: Int,
            temperature: Double,
            thinking: Boolean,
        ): Completion {
            calls += 1
            started.countDown()
            Thread.sleep(300)
            return Completion(text = "Fine move.")
        }
    }

    /** Closing drops what is queued: the call in flight finishes, nothing behind it is asked. */
    @Test
    fun closingDropsWhatIsQueued() {
        val model = Slow()
        val speaker = Speaker(model)
        speaker.start()
        for (trigger in listOf(Trigger.POST_MOVE, Trigger.USER_ASK, Trigger.USER_ASK, Trigger.USER_ASK)) {
            speaker.submit(request(trigger))
        }
        assertTrue(model.started.await(5, TimeUnit.SECONDS))
        speaker.close(5000)
        Thread.sleep(500)
        assertEquals(1, model.calls)
        assertFalse(speaker.pending > 0)
    }
}
