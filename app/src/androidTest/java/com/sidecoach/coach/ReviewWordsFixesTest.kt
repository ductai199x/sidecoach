package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.events.ReviewJson
import com.sidecoach.events.ReviewedGame
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.Completion
import com.sidecoach.llm.LLMProvider
import com.sidecoach.llm.Prices
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The AI coach's words on Game Review's key moves, the fixes of the review of 2026-10-01 as the
 * desktop's tests/test_review_words.py says them, on a real review of profile/reviews.json.
 * Logic only: who was asked, on how many threads, what was kept -- never how long anything took.
 */
@RunWith(AndroidJUnit4::class)
class ReviewWordsFixesTest {
    private val game: ReviewedGame by lazy {
        val assets = InstrumentationRegistry.getInstrumentation().context.assets
        val fixture = JSONObject(assets.open("profile/reviews.json").bufferedReader().use { it.readText() })
        ReviewJson.reviewed(fixture.getJSONArray("games").getJSONObject(0).getJSONObject("reviewed"))
    }

    /** Answers every key move's call, failing those in [failing]; records the plies and threads. */
    private inner class Model(private val failing: Set<Int> = emptySet()) : LLMProvider {
        override val name = "fake"
        val asked = ArrayList<Int>()
        val threads = HashSet<String>()
        private val plies by lazy { ReviewWords.keyPlies(game).associateBy { ReviewWords.prompt(game, it, "black", null) } }

        override fun healthcheck(): Pair<Boolean, String> = true to "fake"

        override fun chat(system: String, messages: List<ChatMessage>, maxTokens: Int, temperature: Double, thinking: Boolean): Completion {
            val ply = plies.getValue(messages[0].content)
            synchronized(this) {
                asked.add(ply)
                threads.add(Thread.currentThread().name + "#" + Thread.currentThread().id)
            }
            if (ply in failing) throw RuntimeException("429: slow down")
            return Completion(text = "### why\nA move.", stats = mapOf("prompt_tokens" to 10, "completion_tokens" to 10))
        }
    }

    /** `test_a_failed_call_drops_its_move_alone`: a failed call's move is left to ask again. */
    @Test
    fun aFailedCallLeavesItsMoveToAskAgain() {
        val plies = ReviewWords.keyPlies(game)
        assertTrue("a game with key moves", plies.size >= 2)
        val words = ReviewWords.write(game, Model(setOf(plies[0])), user = "black", opening = null, prices = Prices())
        assertEquals(setOf(plies[0]), words.failed)
        assertEquals("the call failed: 429: slow down", words.dropped[plies[0]])
    }

    /** `test_the_calls_go_a_few_at_a_time`: the calls go on PARALLEL threads at most. */
    @Test
    fun theCallsGoAFewAtATime() {
        val plies = ReviewWords.keyPlies(game)
        val was = ReviewWords.PARALLEL
        ReviewWords.PARALLEL = 1
        try {
            val model = Model()
            ReviewWords.write(game, model, user = "black", opening = null, prices = Prices())
            assertEquals(plies.sorted(), model.asked.sorted())
            assertEquals("one thread for every call", 1, model.threads.size)
        } finally {
            ReviewWords.PARALLEL = was
        }
    }
}
