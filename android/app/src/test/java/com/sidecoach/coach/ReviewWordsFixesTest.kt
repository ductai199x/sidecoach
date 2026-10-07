package com.sidecoach.coach

import com.sidecoach.TestAssets
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

/**
 * The AI coach's words on Game Review's key moves, the fixes of the review of 2026-10-01 as the
 * desktop's tests/test_review_words.py says them, on a real review of profile/reviews.json.
 * Logic only: who was asked, on how many threads, what was kept -- never how long anything took.
 */
class ReviewWordsFixesTest {
    private val game: ReviewedGame by lazy {
        val fixture = JSONObject(TestAssets.text("profile/reviews.json"))
        ReviewJson.reviewed(fixture.getJSONArray("games").getJSONObject(0).getJSONObject("reviewed"))
    }

    /** Answers every key move's call, failing those in [failing]; records the plies and threads. */
    private inner class Model(
        private val failing: Set<Int> = emptySet(),
        private val answers: Map<Int, String> = emptyMap(),
    ) : LLMProvider {
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
            return Completion(text = answers[ply] ?: "### why\nA move.", stats = mapOf("prompt_tokens" to 10, "completion_tokens" to 10))
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

    /** `test_no_call_is_made_once_told_to_stop`: the game removed while its words are written. */
    @Test
    fun noCallIsMadeOnceToldToStop() {
        val plies = ReviewWords.keyPlies(game)
        val model = Model()
        val words = ReviewWords.write(
            game, model, user = "black", opening = null, prices = Prices(),
            answered = mapOf(plies[0] to "### why\nA move."), stop = { true },
        )
        assertEquals(emptyList<Int>(), model.asked)
        assertEquals(plies.drop(1).toSet(), words.failed)
    }

    /**
     * `test_a_why_naming_the_better_move_is_shown_as_written`, `test_a_side_named_you_is_shown_as_
     * written`, `test_a_lesson_is_shown_as_written`: no rule over the words drops or holds one back
     * (Tai, 2026-10-02) -- a "why" that gives the better move away in words and says "your
     * opponent", and a lesson that names a square, are shown as written, and nothing is dropped.
     */
    @Test
    fun theWordsAreShownAsWritten() {
        val move = game.moves.first {
            val label = it.assessment.label?.let { l -> ReviewJson.labelName(l) }
            label in ReviewWords.BAD && it.assessment.best.isNotEmpty() &&
                !com.sidecoach.chess.Board(it.fenBefore).isCastling(com.sidecoach.chess.Move.fromUci(it.assessment.best[0]))
        }
        val board = com.sidecoach.chess.Board(move.fenBefore)
        val answer = com.sidecoach.chess.Move.fromUci(move.assessment.best[0])
        val piece = listOf("", "pawn", "knight", "bishop", "rook", "queen", "king")[board.pieceTypeAt(answer.from)]
        val square = com.sidecoach.chess.squareName(answer.to)
        val why = "Bringing the $piece over to $square was better, and your opponent knew it."
        val lesson = "Keep an eye on $square before you commit."
        val words = ReviewWords.write(
            game, Model(answers = mapOf(move.ply to "### why\n$why\n### lesson\n$lesson")),
            user = "black", opening = null, prices = Prices(),
        )
        assertEquals(mapOf("why" to why, "lesson" to lesson), words.texts[move.ply])
        assertEquals(null, words.dropped[move.ply])
    }
}
