package com.sidecoach.profile

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Stats' fixes of the review of 2026-10-01, each as its desktop test in tests/test_stats.py
 * says it, on a store built by hand (no fixture).
 */
@RunWith(AndroidJUnit4::class)
class StatsReviewTest {

    private val now = 2_000_000_000.0

    private fun game(key: String, userSide: String? = "white", timeClass: String? = "rapid") = GameFact(
        key = key, source = "chesscom", timeClass = timeClass, playedAt = now - 3600, userSide = userSide,
        bot = false, result = "1-0", opening = "C50 Italian Game: Giuoco Piano", termination = "me won by resignation",
    )

    private fun move(
        ply: Int,
        side: String = "white",
        phase: String = "middlegame",
        label: String? = "best",
        loss: Double = 0.0,
        chance: String? = null,
        allowed: String? = null,
        fen: String = "8/8/8/8/8/8/8/8 w - - 0 1",
    ) = MoveFact(
        ply = ply, side = side, fen = fen, uci = "e2e4", san = "e4", phase = phase, loss = loss, label = label,
        accuracy = 90.0, score = 0.5, chance = chance, found = false, allowed = allowed, structure = "semi-open",
        themes = "", endgame = null, spentMs = 5000, clockMs = 300_000,
    )

    /** `test_work_on_this_waits_for_several_games`. */
    @Test
    fun workOnThisWaitsForSeveralGamesAndCountsMoves() {
        val store = Store(null)
        try {
            store.saveStats("standard", game("a"), (0 until 4).map { move(3 + 2 * it, phase = "opening", label = "blunder", loss = 0.5) })
            assertEquals(0, Stats.stats(store, Stats.Filter(), now).getJSONArray("work").length())
            store.saveStats("standard", game("b"), listOf(move(21)))
            store.saveStats("standard", game("c"), listOf(move(21, allowed = "fork", label = "mistake", loss = 0.4)))
            val work = Stats.stats(store, Stats.Filter(), now).getJSONArray("work")
            assertEquals(1, work.length())
            val w = work.getJSONObject(0)
            assertEquals(Triple("opening", 4, 3), Triple(w.getString("area"), w.getInt("moves"), w.getInt("games")))
        } finally {
            store.close()
        }
    }
}
