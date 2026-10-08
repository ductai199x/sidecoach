package com.sidecoach.llm

import com.sidecoach.profile.Ask
import com.sidecoach.profile.Store
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.net.SocketTimeoutException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Future
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException

/**
 * The timeout bounds the whole reply, not each read: the desktop's
 * `tests/test_llm_openai_compat.py` (the trickle tests) and `tests/test_profile.py`
 * (`test_a_record_question_ends_at_the_models_timeout_however_many_queries`), 1:1. The timer is
 * one on a clock the test moves ([StepTimer], the desktop's ManualClock), so nothing waits on
 * real time.
 */
class ReplyTimeoutTest {

    /** A [ReplyTimer] on a clock the test moves: a step fires when [advance] passes its time. */
    private class StepTimer : ReplyTimer {
        @Volatile var now = 1000.0
        private val steps = ArrayList<Step>()

        inner class Step(val due: Double, val run: () -> Unit) : Future<Unit> {
            @Volatile var cancelled = false
            @Volatile var fired = false
            override fun cancel(mayInterruptIfRunning: Boolean): Boolean {
                cancelled = true
                return true
            }
            override fun isCancelled() = cancelled
            override fun isDone() = cancelled || fired
            override fun get() = Unit
            override fun get(timeout: Long, unit: TimeUnit) = Unit
        }

        override fun schedule(seconds: Double, run: () -> Unit): Future<*> =
            synchronized(steps) { Step(now + seconds, run).also { steps.add(it) } }

        fun pending(): List<Step> = synchronized(steps) { steps.filter { !it.isDone } }

        fun advance(seconds: Double) {
            val until = now + seconds
            while (true) {
                val step = synchronized(steps) {
                    val due = steps.filter { !it.isDone && it.due <= until }.minByOrNull { it.due }
                    if (due == null) {
                        now = maxOf(now, until)
                        null
                    } else {
                        now = maxOf(now, due.due)
                        due.fired = true
                        due
                    }
                } ?: return
                step.run()
            }
        }
    }

    private val user = listOf(ChatMessage("user", "hi"))

    private fun provider(wire: Wire, timer: StepTimer, transport: Transport) = OpenAICompatProvider(
        "x",
        "http://h/v1",
        "m",
        wire = wire,
        timeoutS = 60.0,
        transport = transport,
        timer = timer,
    )

    private fun chatPiece(i: Int) =
        "data: " + JSONObject(mapOf("choices" to listOf(mapOf("delta" to mapOf("content" to "w$i "))))).toString()

    private fun responsesPiece(i: Int) =
        "data: " + JSONObject(mapOf("type" to "response.output_text.delta", "output_index" to 0, "delta" to "w$i ")).toString()

    /** A reply that trickles in: each line [gap] seconds after the last, on the test's clock. */
    private fun trickling(timer: StepTimer, lines: List<String>, gap: Double, whole: String = "") = object : Transport {
        override fun get(url: String, headers: Map<String, String>, timeoutMs: Int) = HttpReply(404, "")

        override fun post(url: String, headers: Map<String, String>, body: String, timeoutMs: Int) =
            post(url, headers, body, timeoutMs) {}

        override fun post(url: String, headers: Map<String, String>, body: String, timeoutMs: Int, sent: () -> Unit): HttpReply {
            sent()
            repeat(lines.size) { timer.advance(gap) }
            return HttpReply(200, whole)
        }

        override fun postLines(
            url: String,
            headers: Map<String, String>,
            body: String,
            timeoutMs: Int,
            onLine: (String) -> Boolean,
        ): HttpReply {
            for (line in lines) {
                if (!onLine(line)) break
                timer.advance(gap)
            }
            return HttpReply(200, "")
        }
    }

    @Test
    fun aReplyThatTricklesInEndsAtTheTimeoutNotAfterIt() {
        for (wire in listOf(Wire.CHAT, Wire.RESPONSES)) {
            val timer = StepTimer()
            val lines = if (wire == Wire.CHAT) {
                (0 until 100).map { chatPiece(it) } + "data: [DONE]"
            } else {
                (0 until 100).map { responsesPiece(it) } +
                    ("data: " + JSONObject(mapOf("type" to "response.completed", "response" to mapOf("status" to "completed"))))
            }
            val model = provider(wire, timer, trickling(timer, lines, 1.0))
            val heard = ArrayList<String>()
            val thrown = runCatching {
                model.chatStream("sys", user, onDelta = { heard.add(it); true })
            }.exceptionOrNull()
            assertTrue("$wire: $thrown", thrown is SocketTimeoutException && thrown.message!!.startsWith("Read timed out"))
            // The pieces sent at 0 s to 59 s, and none after the timeout.
            assertEquals(wire.toString(), (0 until 60).map { "w$it " }, heard)
            assertEquals(1, model.usage.calls)
            assertEquals(1, model.usage.estimated)
        }
    }

    @Test
    fun aWholeReplyThatTricklesInEndsAtTheTimeout() {
        val timer = StepTimer()
        val body = JSONObject(
            mapOf("choices" to listOf(mapOf("message" to mapOf("content" to "late"), "finish_reason" to "stop"))),
        ).toString()
        val model = provider(Wire.CHAT, timer, trickling(timer, List(100) { " " }, 1.0, body))
        val thrown = runCatching { model.chat("sys", user) }.exceptionOrNull()
        assertTrue("$thrown", thrown is SocketTimeoutException)
        assertEquals(1, model.usage.calls)
        // Within the timeout the same reply is read as before.
        val quick = provider(Wire.CHAT, timer, trickling(timer, List(100) { " " }, 0.5, body))
        assertEquals("late", quick.chat("sys", user).text)
    }

    @Test
    fun aReplyThatStopsComingEndsAtTheTimeoutAndNothingLaterIsHandedOn() {
        val timer = StepTimer()
        val resume = CountDownLatch(1)
        val two = CountDownLatch(1)
        val stalled = object : Transport {
            override fun get(url: String, headers: Map<String, String>, timeoutMs: Int) = HttpReply(404, "")

            override fun post(url: String, headers: Map<String, String>, body: String, timeoutMs: Int) = HttpReply(500, "")

            override fun postLines(
                url: String,
                headers: Map<String, String>,
                body: String,
                timeoutMs: Int,
                onLine: (String) -> Boolean,
            ): HttpReply {
                for (i in 0 until 4) {
                    if (i == 2) assertTrue(resume.await(HANG_MS, TimeUnit.MILLISECONDS))
                    if (!onLine(chatPiece(i))) break
                }
                return HttpReply(200, "")
            }
        }
        val model = provider(Wire.CHAT, timer, stalled)
        val heard = java.util.Collections.synchronizedList(ArrayList<String>())
        var thrown: Throwable? = null
        val caller = Thread {
            thrown = runCatching {
                model.chatStream("sys", user, onDelta = {
                    heard.add(it)
                    if (heard.size == 2) two.countDown()
                    true
                })
            }.exceptionOrNull()
        }
        caller.start()
        assertTrue(two.await(HANG_MS, TimeUnit.MILLISECONDS))
        timer.advance(60.0)
        caller.join(HANG_MS)
        assertFalse(caller.isAlive)
        assertTrue("$thrown", thrown is SocketTimeoutException)
        resume.countDown() // the server goes on: nobody hears it
        assertEquals(listOf("w0 ", "w1 "), heard.toList())
    }

    @Test
    fun aTimeoutBeforeTheRequestWasSentIsAConnectTimeoutAndNotBilled() {
        for (reached in listOf(false, true)) {
            val timer = StepTimer()
            val waiting = CountDownLatch(1)
            val go = CountDownLatch(1)
            val held = object : Transport {
                override fun get(url: String, headers: Map<String, String>, timeoutMs: Int) = HttpReply(404, "")

                override fun post(url: String, headers: Map<String, String>, body: String, timeoutMs: Int) =
                    post(url, headers, body, timeoutMs) {}

                override fun post(
                    url: String,
                    headers: Map<String, String>,
                    body: String,
                    timeoutMs: Int,
                    sent: () -> Unit,
                ): HttpReply {
                    if (reached) sent()
                    waiting.countDown()
                    assertTrue(go.await(HANG_MS, TimeUnit.MILLISECONDS))
                    return HttpReply(200, """{"choices": [{"message": {"content": "late"}}]}""")
                }

                override fun postLines(
                    url: String,
                    headers: Map<String, String>,
                    body: String,
                    timeoutMs: Int,
                    onLine: (String) -> Boolean,
                ) = HttpReply(500, "")
            }
            val model = provider(Wire.CHAT, timer, held)
            var thrown: Throwable? = null
            val caller = Thread { thrown = runCatching { model.chat("sys", user) }.exceptionOrNull() }
            caller.start()
            assertTrue(waiting.await(HANG_MS, TimeUnit.MILLISECONDS))
            timer.advance(60.0)
            caller.join(HANG_MS)
            go.countDown()
            assertFalse(caller.isAlive)
            val message = (thrown as SocketTimeoutException).message!!
            assertTrue(message, message.startsWith(if (reached) "Read timed out" else "connect timed out"))
            assertEquals(if (reached) 1 else 0, model.usage.calls)
        }
    }

    /**
     * A [Store] never opened: there is no SQLite on the JVM, and every query the model writes
     * here is refused before it would reach the database (Store.query: only SELECT), so the
     * rounds go on as they do with rows.
     */
    private fun unopenedStore(): Store {
        val unsafe = Class.forName("sun.misc.Unsafe").getDeclaredField("theUnsafe").apply { isAccessible = true }.get(null)
        val allocate = unsafe.javaClass.getMethod("allocateInstance", Class::class.java)
        return allocate.invoke(unsafe, Store::class.java) as Store
    }

    /** Each call takes 25 s of the test's clock, within a 60 s timeout of its own. */
    private fun slow(timer: StepTimer, replies: List<String>, asked: MutableList<String>) = object : Transport {
        val left = ArrayList(replies)

        override fun get(url: String, headers: Map<String, String>, timeoutMs: Int) = HttpReply(404, "")

        override fun post(url: String, headers: Map<String, String>, body: String, timeoutMs: Int): HttpReply {
            asked.add(body)
            timer.advance(25.0)
            val text = if (left.size > 1) left.removeAt(0) else left[0]
            return HttpReply(200, JSONObject(mapOf("choices" to listOf(mapOf("message" to mapOf("content" to text))))).toString())
        }

        override fun postLines(
            url: String,
            headers: Map<String, String>,
            body: String,
            timeoutMs: Int,
            onLine: (String) -> Boolean,
        ) = HttpReply(500, "")
    }

    @Test
    fun aRecordQuestionEndsAtTheModelsTimeoutHoweverManyQueries() {
        val store = unopenedStore()
        val asked = ArrayList<String>()
        // The model's own calls run on a real timer of their own, far off; the question's on the
        // test's clock.
        val calls = StepTimer()
        val model = OpenAICompatProvider("x", "http://h/v1", "m", timeoutS = 60.0, transport = slow(calls, listOf("```sql\nDELETE FROM games\n```"), asked))
        val thrown = runCatching { Ask.answer("how many games?", store, model, timer = calls) }.exceptionOrNull()
        assertTrue("$thrown", thrown is TimeoutException)
        assertEquals("the third call passed the timeout; no fourth started", 3, asked.size)
        // Within the timeout the question is answered as before.
        val quickAsked = ArrayList<String>()
        val quick = OpenAICompatProvider(
            "x",
            "http://h/v1",
            "m",
            timeoutS = 60.0,
            transport = slow(calls, listOf("```sql\nDELETE FROM games\n```", "One game so far."), quickAsked),
        )
        val got = Ask.answer("how many games?", store, quick, timer = calls)
        assertEquals("One game so far.", got.text)
        assertEquals(2, got.calls)
        assertTrue(calls.pending().isEmpty())
    }

    private companion object {
        const val HANG_MS = 30_000L
    }
}
