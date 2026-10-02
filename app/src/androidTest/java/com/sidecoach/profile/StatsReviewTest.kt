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

    /** `test_a_game_reviewed_before_the_username_counts_once_it_is_set`. */
    @Test
    fun aGameReviewedBeforeTheUsernameCountsOnceItIsSet() {
        val store = Store(null)
        try {
            store.addLibraryGame(LibraryRow(key = "pgn:1", source = "pgn", addedAt = 0.0, pgn = "1. e4 *", playedAt = now, white = "Tai"))
            store.saveStats("standard", game("pgn:1", userSide = null), listOf(move(1)))
            var shown = Stats.stats(store, Stats.Filter(), now)
            assertEquals(0 to 1, shown.getJSONObject("overview").getInt("games") to shown.getInt("unsided"))
            Library.refresh(store, listOf("tai"))
            shown = Stats.stats(store, Stats.Filter(), now)
            assertEquals(1 to 0, shown.getJSONObject("overview").getInt("games") to shown.getInt("unsided"))
            // Pasted again with the username known: the import's row has the side, and Stats follows.
            store.saveStats("standard", game("pgn:2", userSide = null), listOf(move(2, side = "black")))
            store.addLibraryGame(LibraryRow(key = "pgn:2", source = "pgn", addedAt = 0.0, pgn = "1. e4 *", playedAt = now, black = "Tai", userSide = "black"))
            assertEquals(2, Stats.stats(store, Stats.Filter(), now).getJSONObject("overview").getInt("games"))
        } finally {
            store.close()
        }
    }

    /**
     * `test_the_facts_kept_for_stats_are_the_strongest_reviews`: a Fast look after a Deep review
     * lowered Stats and Your games, and the review opened at the one strength always.
     */
    @Test
    fun statsAndTheReviewKeepTheStrongest() {
        val store = Store(null)
        try {
            store.saveStats("deep", game("a"), listOf(move(1)))
            store.saveStats("fast", game("a"), listOf(move(1, label = "blunder", loss = 0.4)))
            assertEquals(listOf("deep"), store.statGames().map { it.strength })
            assertEquals(0.0, store.statMoves(listOf("a")).single().loss, 0.0)
            store.saveStats("deep", game("a"), listOf(move(1, label = "inaccuracy", loss = 0.1)))
            assertEquals("as strong: replaced", 0.1, store.statMoves(listOf("a")).single().loss, 0.0)
            store.saveReview("b", "deep", 1, 1.0, "{}", null)
            store.saveReview("b", "fast", 1, 2.0, "{}", null)
            assertEquals(listOf("b" to "deep"), store.reviewsWithoutStats().map { it.key to it.strength })
            assertEquals("deep", store.bestReview("b")!!.strength)
        } finally {
            store.close()
        }
    }

    /**
     * `test_time_classes_by_the_expected_length_of_a_game` and
     * `test_games_kept_by_the_old_lines_are_classed_again`, test_library.py: Lichess's own lines.
     */
    @Test
    fun lichessGamesTakeLichesssTimeClasses() {
        assertEquals(
            listOf("bullet", "bullet", "rapid", "blitz", "rapid", "classical", "daily"),
            listOf("15", "60", "480", "180+2", "1200", "1500", "-").map { Library.timeClassOf(it, lichess = true) },
        )
        assertEquals("blitz" to null, Library.timeClassOf("480") to Library.timeClassOf("-"))
        val store = Store(null)
        try {
            store.addLibraryGame(LibraryRow("lichess:abc", "lichess", 0.0, "[Site \"https://lichess.org/abc\"]\n[TimeControl \"480\"]\n\n1. e4 *", timeClass = "blitz"))
            store.addLibraryGame(LibraryRow("pgn:xyz", "pgn", 0.0, "[Site \"https://lichess.org/xyz\"]\n[TimeControl \"-\"]\n\n1. e4 *"))
            store.addLibraryGame(LibraryRow("chesscom:1", "chesscom", 0.0, "[TimeControl \"480\"]\n\n1. e4 *", timeClass = "blitz"))
            Library.refresh(store, emptyList())
            assertEquals(
                mapOf("lichess:abc" to "rapid", "pgn:xyz" to "daily", "chesscom:1" to "blitz"),
                store.library().associate { it.key to it.timeClass },
            )
            store.setTimeClass("lichess:abc", "blitz")
            Library.refresh(store, emptyList())
            assertEquals("once", "blitz", store.libraryGame("lichess:abc")!!.timeClass)
        } finally {
            store.close()
        }
    }

    /**
     * `test_stats_read_the_store_in_a_few_queries`: the positions read the latest games a batch
     * at a time until they have enough -- the latest first, as many as before.
     */
    @Test
    fun thePositionsAreTheLatestReadABatchAtATime() {
        val store = Store(null)
        try {
            for (i in 0 until Stats.POSITIONS_BATCH * 3) {
                store.saveStats("standard", game("g%03d".format(i)).copy(playedAt = now - 3600.0 * (i + 1)), listOf(move(3, label = "blunder", loss = 0.4)))
            }
            store.addLibraryGame(LibraryRow("g000", "pgn", 0.0, "1. e4 *", whiteElo = 1500))
            val found = Stats.positions(store, Stats.Filter(), "gallery", "blunder", now)
            assertEquals(Stats.POSITIONS_MAX, found.length())
            assertEquals("the latest first", "g000", found.getJSONObject(0).getString("key"))
            assertEquals("g059", found.getJSONObject(Stats.POSITIONS_MAX - 1).getString("key"))
            assertEquals(mapOf("g000" to (1500 to null)), store.libraryElos())
        } finally {
            store.close()
        }
    }

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
