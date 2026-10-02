package com.sidecoach.ui

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.events.ReviewJson
import com.sidecoach.profile.GameSummaries
import com.sidecoach.profile.Library
import com.sidecoach.profile.StatFacts
import com.sidecoach.profile.Store
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Game Review's fixes of the review of 2026-10-01 that need no engine, each as its desktop test
 * in tests/test_review_server.py says it, on the real reviews of profile/reviews.json.
 */
@RunWith(AndroidJUnit4::class)
class ReviewFixesTest {
    private val fixture: JSONObject by lazy {
        val assets = InstrumentationRegistry.getInstrumentation().context.assets
        JSONObject(assets.open("profile/reviews.json").bufferedReader().use { it.readText() })
    }

    /** `test_saved_reviews_are_summed_up_again_when_the_rule_changes`. */
    @Test
    fun savedReviewsAreSummedUpAgainWhenTheRuleChanges() {
        val g = fixture.getJSONArray("games").getJSONObject(0)
        val store = Store(null)
        try {
            val row = Library.fromPgn(g.getString("pgn"), listOf("nosegazm")).single()
            store.addLibraryGame(row)
            val reviewed = ReviewJson.reviewed(g.getJSONObject("reviewed"))
            val fresh = ReviewJson.summary(GameSummaries.summarize(reviewed)).toString()
            val stale = JSONObject(fresh).put("shape", "old").toString()
            store.saveReview(row.key, "fast", ReviewRoom.REVIEW_VERSION, 1.0, ReviewJson.reviewed(reviewed).toString(), stale)
            val (fact, moves) = StatFacts.factsOf(reviewed, row)
            val phases = moves.map { it.phase }
            store.saveStats("fast", fact, moves.map { it.copy(phase = "endgame") }) // as an older rule left them
            store.setSetting(ReviewRoom.DERIVED_KEY, "1")
            assertEquals(1, ReviewRoom.rederive(store))
            assertEquals(fresh, store.review(row.key, "fast")!!.summary)
            assertEquals(phases, store.statMoves(listOf(row.key)).map { it.phase })
            assertEquals("once", 0, ReviewRoom.rederive(store))
        } finally {
            store.close()
        }
    }

    /** `test_no_shape_says_a_side_kept_the_upper_hand_it_lost`, test_game_summary.py. */
    @Test
    fun noShapeSaysASideKeptTheUpperHandItLost() {
        val once = listOf(0.5, 0.55, 0.6, 0.7, 0.75, 0.74, 0.72, 0.6, 0.5, 0.4, 0.3, 0.25, 0.26, 0.25, 0.3, 0.27)
        assertEquals(null, GameSummaries.shapeOf(once, "0-1"))
        val led = listOf(0.5, 0.6, 0.72, 0.72, 0.73, 0.72, 0.71, 0.72, 0.72, 0.72, 0.6, 0.55)
        assertEquals(null, GameSummaries.shapeOf(led, "0-1"))
        val sharp = listOf(0.5, 0.75, 0.99, 0.71, 0.99, 0.71, 0.99, 0.71, 0.99, 0.71, 0.99, 0.71, 1.0)
        assertEquals("sharp", GameSummaries.shapeOf(sharp, "1-0"))
    }

    /** `test_how_a_game_ended_when_its_header_does_not_say`, test_stat_facts.py. */
    @Test
    fun howAGameEndedWhenItsHeaderDoesNotSay() {
        fun played(sans: List<String>, result: String): com.sidecoach.events.ReviewedGame {
            val g = JSONObject().put("headers", JSONObject().put("Result", result)).put("strength", "fast").put("depth", 14)
            val start = JSONObject().put("game_id", "g").put("fen", com.sidecoach.chess.STARTING_FEN).put("ply", 0)
                .put("lines", org.json.JSONArray()).put("decisive", false).put("only_move", false)
            g.put("start", start)
            val moves = org.json.JSONArray()
            val board = com.sidecoach.chess.Board()
            for ((i, san) in sans.withIndex()) {
                val move = board.parseSan(san)!!
                val side = if (board.turn) "white" else "black"
                val before = board.fen()
                board.push(move)
                moves.put(
                    JSONObject().put("ply", i + 1).put("san", san).put("side", side).put("fen_before", before)
                        .put("assessment", JSONObject().put("game_id", "g").put("ply", i + 1).put("uci", move.uci()).put("expected_loss", 0.0).put("severity", "none").put("best", org.json.JSONArray()))
                        .put("after", JSONObject(start.toString()).put("fen", board.fen()).put("ply", i + 1)),
                )
            }
            return ReviewJson.reviewed(g.put("moves", moves))
        }
        val mate = listOf("f3", "e5", "g4", "Qh4#")
        assertEquals("checkmate", StatFacts.endedBy("Normal", "0-1", played(mate, "0-1")))
        assertEquals("checkmate", StatFacts.endedBy(null, "0-1", played(mate, "0-1")))
        val short = listOf("e4", "e5")
        assertEquals("resignation", StatFacts.endedBy("Normal", "1-0", played(short, "1-0")))
        assertEquals("agreement", StatFacts.endedBy("Normal", "1/2-1/2", played(short, "1/2-1/2")))
        val shuffle = List(2) { listOf("Nf3", "Nf6", "Ng1", "Ng8") }.flatten()
        assertEquals("repetition", StatFacts.endedBy("Normal", "1/2-1/2", played(shuffle, "1/2-1/2")))
        assertEquals("nosegazm won on time", StatFacts.endedBy("nosegazm won on time", "1-0", played(short, "1-0")))
        assertEquals("nothing proves how", null, StatFacts.endedBy(null, "1-0", played(short, "1-0")))
    }

    /**
     * A stored move list with a move that is not legal proves nothing: the desktop's `ended_by`
     * pushes with `push_uci`, which raises on it, and gives the header's ending. The phone pushed
     * unchecked, so the queen's jump through its own pawn here mated (the review of 2026-10-01).
     */
    @Test
    fun aMoveListWithAnIllegalMoveProvesNoEnding() {
        fun stored(ucis: List<String>, result: String): com.sidecoach.events.ReviewedGame {
            val g = JSONObject().put("headers", JSONObject().put("Result", result)).put("strength", "fast").put("depth", 14)
            val start = JSONObject().put("game_id", "g").put("fen", com.sidecoach.chess.STARTING_FEN).put("ply", 0)
                .put("lines", org.json.JSONArray()).put("decisive", false).put("only_move", false)
            g.put("start", start)
            val moves = org.json.JSONArray()
            for ((i, uci) in ucis.withIndex()) {
                moves.put(
                    JSONObject().put("ply", i + 1).put("san", uci).put("side", if (i % 2 == 0) "white" else "black")
                        .put("fen_before", com.sidecoach.chess.STARTING_FEN)
                        .put("assessment", JSONObject().put("game_id", "g").put("ply", i + 1).put("uci", uci).put("expected_loss", 0.0).put("severity", "none").put("best", org.json.JSONArray()))
                        .put("after", JSONObject(start.toString()).put("ply", i + 1)),
                )
            }
            return ReviewJson.reviewed(g.put("moves", moves))
        }
        // d8h4 with the pawn still on e7: the fool's mate's final position, reached illegally.
        val jumped = listOf("f2f3", "d8h4", "g2g4", "a7a6")
        assertEquals("the header's ending", "Normal", StatFacts.endedBy("Normal", "0-1", stored(jumped, "0-1")))
        assertEquals("none", null, StatFacts.endedBy(null, "0-1", stored(jumped, "0-1")))
        val played = listOf("f2f3", "e7e5", "g2g4", "d8h4")
        assertEquals("the legal one mates", "checkmate", StatFacts.endedBy(null, "0-1", stored(played, "0-1")))
    }

    /**
     * For Tai #9 (decided 2026-10-02): the AI words paid for on a watched game whose record
     * differs from its import show in the import's review on the moves the two share -- the same
     * position, the same move, whatever the move numbers (a game joined late) -- and not on a
     * move the import does not have; the watched review keeps every word. The desktop's
     * test_words_go_by_position_and_move_whatever_the_move_numbers and
     * test_the_words_paid_on_a_watched_game_show_in_its_imports_review.
     */
    @Test
    fun theWordsPaidOnAWatchedGameShowWhereTheMovesMatch() {
        val g = fixture.getJSONArray("games").getJSONObject(0)
        val imported = ReviewJson.reviewed(g.getJSONObject("reviewed"))
        val misread = 7 // a move Sidecoach read otherwise: its position is not the import's
        val watched = com.sidecoach.events.ReviewedGame(
            headers = imported.headers,
            start = imported.start,
            moves = imported.moves.map { m ->
                val fields = m.fenBefore.split(" ").toMutableList()
                fields[5] = (fields[5].toInt() + 20).toString() // joined at move 21
                if (m.ply == misread) fields[0] = "8/8/8/8/8/8/8/K6k"
                com.sidecoach.events.ReviewedMove(m.ply, m.san, m.side, fields.joinToString(" "), m.assessment, m.after, m.clockMs, m.spentMs)
            },
            strength = imported.strength,
            depth = imported.depth,
            seconds = imported.seconds,
        )
        val store = Store(null)
        try {
            store.startGame("w", 1.0, "black", com.sidecoach.chess.STARTING_FEN)
            store.addLibraryGame(com.sidecoach.profile.LibraryRow("pgn:i", "pgn", 0.0, "*"))
            assertEquals(true, store.saveReview("watched:w", "fast", ReviewRoom.REVIEW_VERSION, 1.0, ReviewJson.reviewed(watched).toString(), null))
            val paid = JSONObject().put("model", "m").put(
                "texts",
                JSONObject().put("4", JSONObject().put("why", "W4")).put("$misread", JSONObject().put("why", "W$misread")),
            ).toString()
            store.setReviewWords("watched:w", "fast", paid)
            store.addTwin("watched:w", "pgn:i")
            val (carried, model) = ReviewRoom.carriedWords(store, "pgn:i", imported, "standard")
            assertEquals(mapOf(4 to "{\"why\":\"W4\"}"), carried.mapValues { it.value.toString() })
            assertEquals("m", model)
            assertEquals("the watched review keeps every word", paid, store.review("watched:w", "fast")!!.words)
            assertEquals("one way", emptyMap<Int, JSONObject>(), ReviewRoom.carriedWords(store, "watched:w", watched, "fast").first)
        } finally {
            store.close()
        }
    }

    /**
     * `test_the_words_paid_on_a_watched_game_show_in_its_imports_review`: the key moves with words
     * carried from the watched game are not asked (paid) again; an answer of the game's own, paid
     * for before, stays its own.
     */
    @Test
    fun theKeyMovesWithCarriedWordsAreNotAskedAgain() {
        val game = ReviewJson.reviewed(fixture.getJSONArray("games").getJSONObject(0).getJSONObject("reviewed"))
        val plies = com.sidecoach.coach.ReviewWords.keyPlies(game)
        assertEquals(true, plies.size >= 2)
        val asked = ArrayList<Int>()
        val model = object : com.sidecoach.llm.LLMProvider {
            override val name = "fake"
            override fun healthcheck(): Pair<Boolean, String> = true to "fake"
            override fun chat(system: String, messages: List<com.sidecoach.llm.ChatMessage>, maxTokens: Int, temperature: Double, thinking: Boolean): com.sidecoach.llm.Completion {
                val ply = JSONObject(messages[0].content).getJSONObject("key move").getInt("ply")
                synchronized(asked) { asked.add(ply) }
                return com.sidecoach.llm.Completion(text = "### why\nA move.", stats = emptyMap())
            }
        }
        val words = com.sidecoach.coach.ReviewWords.write(
            game, model, user = "black", opening = null, prices = com.sidecoach.llm.Prices(),
            carried = setOf(plies[0]),
        )
        assertEquals(plies.drop(1).sorted(), asked.sorted())
        assertEquals(false, plies[0] in words.texts)
        asked.clear()
        val own = com.sidecoach.coach.ReviewWords.write(
            game, model, user = "black", opening = null, prices = com.sidecoach.llm.Prices(),
            answered = mapOf(plies[0] to "### why\nOwn."), carried = setOf(plies[0]),
        )
        assertEquals(plies.drop(1).sorted(), asked.sorted())
        assertEquals(mapOf("why" to "Own."), own.texts[plies[0]])
    }

    /** `test_a_game_set_up_from_a_position_is_rated_from_where_it_starts`, test_performance.py. */
    @Test
    fun aGameSetUpFromAPositionIsRatedFromWhereItStarts() {
        assertEquals(45, com.sidecoach.profile.Performance.pliesBefore("4k3/8/8/8/8/8/4P3/4K3 b - - 0 23"))
        assertEquals(0, com.sidecoach.profile.Performance.pliesBefore(com.sidecoach.chess.STARTING_FEN))
    }
}
