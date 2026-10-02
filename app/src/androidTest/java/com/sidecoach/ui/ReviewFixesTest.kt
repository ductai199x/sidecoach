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
}
