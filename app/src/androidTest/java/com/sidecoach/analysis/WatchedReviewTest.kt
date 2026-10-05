package com.sidecoach.analysis

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.Lane
import com.sidecoach.chess.Board
import com.sidecoach.chess.PgnReader
import com.sidecoach.events.GameStarted
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.ReviewJson
import com.sidecoach.profile.Store
import com.sidecoach.vision.Pieces
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith

/**
 * `test_a_watched_games_review_searches_nothing_the_game_measured`, on the phone: a game followed
 * live had every position measured whole -- its own search and its side searches, at a review
 * strength's standard -- and its review measured them all again (the engine audit, 2026-10-04).
 * Kept as the game ends ([Assessing.keepForReview]), they are the review's: it searches none of
 * them. On the device for the store (SQLite).
 */
@Lane(Lane.Kind.FREE)
@RunWith(AndroidJUnit4::class)
class WatchedReviewTest {

    @Test
    fun aWatchedGamesReviewSearchesNothingTheGameMeasured() {
        val analyser = Analyser(GameAnnotator(ScriptEngine(), depth = 18, multipv = 6))
        val watch = AnalyserWatch(analyser)
        analyser.start()
        val assessing = Assessing(analyser)
        val store = Store(null)
        val known: Map<String, Measured>
        try {
            assessing.submit(GameStarted(gameId = "g", userSide = Pieces.Side.WHITE, startFen = Board().fen()))
            val board = Board()
            for ((i, san) in listOf("e4", "e5", "Nf3").withIndex()) {
                val before = board.fen()
                val move = board.parseSan(san)!!
                board.push(move)
                assessing.submit(MovePlayed("g", i + 1, san, move.uci(), before, board.fen(), byUser = i % 2 == 0))
            }
            watch.idle()
            assertEquals("the four positions, whole", 4, assessing.keepForReview(store, "g"))
            known = store.progress("${Store.WATCHED}g", "standard", 18, 6)
                .map { ReviewJson.measured(JSONObject(it)) }
                .associateBy { it.fen }
        } finally {
            store.close()
            analyser.close()
        }
        val reviewEngine = ScriptEngine()
        val game = PgnReader("1. e4 e5 2. Nf3 *").readGame()
        assertNotNull(game)
        val reviewed = GameReview.reviewGame(game!!, listOf(reviewEngine), depth = 18, multipv = 6, thresholds = Thresholds(), known = known)
        assertEquals(3, reviewed.moves.size)
        assertEquals("the review searched what the game had measured", emptyList<String>(), reviewEngine.calls.toList())
    }
}
