package com.sidecoach.ui

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.profile.GameFact
import com.sidecoach.profile.Library
import com.sidecoach.profile.Store
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Your games on the phone (ui/GamesApi.kt, `server._library` and `_import`): the player's record
 * as the list shows it and as Stats counts it, the fixes of the review of 2026-10-01. No engine,
 * no network: the stores are made by hand, as the desktop's tests make them.
 */
@RunWith(AndroidJUnit4::class)
class GamesApiTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    private fun dir(name: String): File = File(context.cacheDir, "games-$name").apply {
        deleteRecursively()
        mkdirs()
    }

    /** A game as Sidecoach records one it watched (the player Black): its moves as read. */
    private fun watch(store: Store, gameId: String, sans: List<String>, startedAt: Double): String {
        store.startGame(gameId, startedAt, "black", STARTING_FEN)
        sans.forEachIndexed { i, san ->
            val ply = i + 1
            store.addMove(gameId, 0, ply, (ply + 1) / 2, san, "0000", ply % 2 == 0, "-", "-", null, null, null, false, false, null, startedAt + ply)
        }
        store.endGame(gameId, startedAt + 900, "0-1", "resignation")
        return "watched:$gameId"
    }

    private fun fact(key: String) = GameFact(key, key.substringBefore(":"), "rapid", 0.0, "black", false, "0-1", null, null)

    private fun listed(files: File): Map<String, JSONObject> {
        val games = GamesApi(files, SettingsStore(context)).list().getJSONArray("games")
        return (0 until games.length()).map { games.getJSONObject(it) }.associateBy { it.getString("key") }
    }

    /**
     * A watched game whose record is not its import's position for position (a move misread)
     * is not carried whole: the moment the import came in, the game lost its accuracy in the list
     * and left Stats until the import was opened and reviewed. It keeps both until then. The
     * desktop's test_an_import_whose_record_differs_is_reviewed_from_the_watched_positions.
     */
    @Test
    fun aWatchedGamePairedWithAnImportThatDiffersKeepsItsAccuracyAndItsPlaceInStats() {
        val files = dir("pair")
        val store = Store(File(files, "profile.sqlite"))
        val row = Library.fromPgn(LONGER, listOf("nosegazm")).single()
        store.addLibraryGame(row)
        val watched = watch(store, "g1", SANS.take(6) + "d4" + SANS.drop(7), row.playedAt!! + 3600)
        val summary = JSONObject()
            .put("white", JSONObject().put("accuracy", 71.5))
            .put("black", JSONObject().put("accuracy", 83.25))
        assertTrue(store.saveReview(watched, "standard", ReviewRoom.REVIEW_VERSION, 1.0, "{}", summary.toString()))
        store.close()

        val games = listed(files)
        assertFalse("the import stands for the watched game", watched in games)
        val game = games.getValue(row.key)
        assertFalse("not reviewed: opening it reviews the import", game.getBoolean("reviewed"))
        assertEquals("the watched review's accuracy", 83.25, game.getJSONObject("accuracy").getDouble("black"), 0.0)

        val again = Store(File(files, "profile.sqlite"))
        assertEquals(mapOf("watched:g1" to row.key), again.twins())
        assertEquals("its facts wanted until the import's are kept", listOf(watched), again.reviewsWithoutStats().map { it.key })
        again.saveStats("standard", fact(watched), emptyList())
        assertEquals("counted, as the watched game", listOf(watched), again.statGames().map { it.key })
        again.saveStats("standard", fact(row.key), emptyList())
        assertEquals("once, as the import, once its facts are kept", listOf(row.key), again.statGames().map { it.key })
        assertEquals(emptyList<String>(), again.reviewsWithoutStats().map { it.key })
        again.close()
    }

    /**
     * The review of 2026-10-01: Your games listed at most 500 imports and 200 watched games while
     * Stats counted every one: a game past the caps could not be seen, opened or removed, nor a
     * watched one paired with its import. The desktop's test_every_game_is_listed_however_many.
     */
    @Test
    fun everyGameIsListedHoweverMany() {
        assertTrue(maxOf(Library.TWIN_BEFORE_S, Library.TWIN_AFTER_S) <= Library.DATE_WINDOW_S)
        val store = Store(null)
        val row = Library.fromPgn(LONGER, listOf("nosegazm")).single()
        val start = row.playedAt!!
        // The first game played: imported and watched, older than every other.
        store.addLibraryGame(row)
        val oldest = watch(store, "g0", SANS, start + 3600)
        // Then 500 imports and 200 watched games, days later and far apart: none pair.
        val later = start + 10 * 86400
        for (i in 0 until 500) store.addLibraryGame(row.copy(key = "chesscom:$i", playedAt = later + i * 86400.0))
        for (i in 0 until 200) watch(store, "w$i", SANS.take(10), later + 600 * 86400.0 + i * 86400)
        assertEquals(501, store.library().size)
        val watched = Library.fromWatched(store)
        assertEquals(201, watched.size)
        assertTrue(watched.any { it.key == oldest })
        assertTrue("opened", Library.find(store, oldest) != null)
        assertEquals("paired with its import", mapOf(oldest to row.key), Library.reconcile(store))
        store.close()
    }

    /**
     * The review of 2026-10-01: a game pasted as a PGN, then removed, came back when the player's
     * games were brought in from chess.com -- the same game under the site's key. It stays out;
     * pasted again by hand, it comes back. The desktop's
     * test_a_pasted_game_removed_stays_out_when_the_site_brings_it_in.
     */
    @Test
    fun aPastedGameRemovedStaysOutWhenTheSiteBringsItIn() {
        val store = Store(null)
        val pasted = Library.fromPgn(LONGER, listOf("nosegazm")).single()
        val fetched = Library.rowFor(
            Library.gamesIn(LONGER).first(),
            source = "chesscom",
            key = "chesscom:174300959956",
            userNames = listOf("nosegazm"),
            timeClass = "rapid",
        )
        assertEquals(pasted.key, Library.pastedKey(fetched))
        assertEquals(pasted.key, Library.pastedKey(pasted))
        store.addLibraryGame(pasted)
        store.removeGame(pasted.key)
        assertEquals("the site's copy stays out", setOf(fetched.key), Library.leftOut(store, listOf(fetched)))
        val other = Library.fromPgn(LONGER.replace("Anakin_737", "Someone"), listOf("nosegazm")).single()
        assertEquals("another game is not", emptySet<String>(), Library.leftOut(store, listOf(other)))
        store.unremove(listOf(pasted.key)) // pasted again by hand
        assertEquals(emptySet<String>(), Library.leftOut(store, listOf(fetched)))
        store.close()
    }

    /**
     * The review of 2026-10-01: a review saved whose Stats facts a kill cut short never reached
     * Stats on the phone -- the desktop's background review catches those up, and the phone has
     * none. Stats, and the review opened, catch it up. No engine: the review is the desktop's
     * (profile/reviews.json).
     */
    @Test
    fun aReviewWhoseStatsStepWasCutShortReachesStats() {
        val fixture = JSONObject(InstrumentationRegistry.getInstrumentation().context.assets.open("profile/reviews.json").bufferedReader().use { it.readText() })
        val game = fixture.getJSONArray("games").getJSONObject(0)
        val row = Library.fromPgn(game.getString("pgn"), listOf("nosegazm")).single()

        fun savedWithoutStats(files: File) = Store(File(files, "profile.sqlite")).apply {
            addLibraryGame(row)
            assertTrue(saveReview(row.key, game.getString("strength"), ReviewRoom.REVIEW_VERSION, 1.0, game.getJSONObject("reviewed").toString(), game.getJSONObject("summary").toString()))
            assertEquals(emptyList<String>(), statGames().map { it.key })
        }.close()

        val files = dir("stats-catch-up")
        savedWithoutStats(files)
        assertTrue(StatsApi(files).stats(emptyMap()).getBoolean("ok"))
        Store(File(files, "profile.sqlite")).apply {
            assertEquals("Stats opened", listOf(row.key), statGames().map { it.key })
        }.close()

        val other = dir("stats-catch-up-room")
        savedWithoutStats(other)
        val room = ReviewRoom(other, SettingsStore(context), row, File(other, "profile.sqlite"), game.getString("strength"), EngineMaker { _, _ -> throw AssertionError("a saved review needs no engine") })
        val end = android.os.SystemClock.elapsedRealtime() + 10_000
        while (room.status != "ready" && android.os.SystemClock.elapsedRealtime() < end) Thread.sleep(50)
        room.close()
        room.join()
        assertEquals("ready", room.status)
        Store(File(other, "profile.sqlite")).apply {
            assertEquals("the review opened", listOf(row.key), statGames().map { it.key })
        }.close()
    }

    private fun site(text: String, key: String) =
        Library.rowFor(Library.gamesIn(text).first(), source = "chesscom", key = key, userNames = listOf("nosegazm"), timeClass = "rapid")

    private fun shown(store: Store) = store.library().map { it.key }.sorted()

    /**
     * A game both pasted and imported is one game, whichever came first (the desktop's
     * test_a_game_both_pasted_and_imported_is_one_game_whichever_came_first): the site's record
     * stands for it in the list, Stats and the backlog; removing it takes its copy; Undo puts both back.
     */
    @Test
    fun aGameBothPastedAndImportedIsOneGameWhicheverCameFirst() {
        for (pastedFirst in listOf(true, false)) {
            val store = Store(null)
            val pasted = Library.fromPgn(LONGER, listOf("nosegazm")).single()
            val site = site(LONGER, "chesscom:1")
            for (row in if (pastedFirst) listOf(pasted, site) else listOf(site, pasted)) {
                store.addLibraryGame(row)
                Library.reconcile(store)
            }
            assertEquals(listOf(site.key), shown(store))
            assertEquals(mapOf(pasted.key to site.key), store.twins())
            assertEquals(listOf(site.key), store.unreviewed().map { it.key })
            store.saveStats("standard", fact(pasted.key), emptyList())
            assertEquals(listOf(pasted.key), store.statGames().map { it.key })
            store.saveStats("standard", fact(site.key), emptyList())
            assertEquals("once, as the site's", listOf(site.key), store.statGames().map { it.key })
            assertEquals(mapOf(pasted.key to site.key), Library.reconcile(store))
            assertEquals(setOf(site.key, pasted.key), store.removeGame(site.key).toSet())
            assertEquals(emptyList<String>(), shown(store))
            assertEquals(setOf(site.key, pasted.key), store.restoreGame(site.key).toSet())
            assertEquals(listOf(site.key), shown(store))
            store.close()
        }
    }

    /** Three copies are one, a better copy takes over the pairs of the one before, and what only looks alike is not merged. */
    @Test
    fun threeCopiesAreOneAndGamesThatOnlyLookAlikeAreNot() {
        val store = Store(null)
        val first = Library.fromPgn(LONGER, listOf("nosegazm")).single()
        val again = Library.fromPgn(LONGER.replace("2026.09.29", "2026.09.30"), listOf("nosegazm")).single()
        assertTrue(again.key != first.key)
        store.addLibraryGame(first)
        val watched = watch(store, "g1", SANS, first.playedAt!! + 60)
        assertEquals(mapOf(watched to first.key), Library.reconcile(store))
        store.addLibraryGame(again)
        Library.reconcile(store)
        assertEquals(listOf(first.key), shown(store))
        val best = site(LONGER, "chesscom:7")
        store.addLibraryGame(best)
        assertEquals(
            mapOf(first.key to best.key, again.key to best.key, watched to best.key),
            Library.reconcile(store),
        )
        assertEquals(listOf(best.key), shown(store))
        store.close()

        val other = Store(null)
        val a = site(LONGER, "chesscom:1")
        val b = site(LONGER, "chesscom:2") // a second game of the site with the same moves
        val lateB = LONGER.replace("8. axb3 c6", "8. axb3 d5")
        val pastedLate = Library.fromPgn(lateB, listOf("nosegazm")).single() // the last move differs
        val stranger = Library.fromPgn(LONGER.replace("Anakin_737", "Magnus"), listOf("nosegazm")).single() // other players
        val opening = Library.fromPgn(LONGER.replace("5. Nc3 d6", "5. c3 d6"), listOf("nosegazm")).single() // same opening
        for (row in listOf(a, b, pastedLate, opening)) other.addLibraryGame(row)
        Library.reconcile(other)
        assertEquals(listOf(a.key, b.key, pastedLate.key, opening.key).sorted(), shown(other))
        assertEquals(emptyMap<String, String>(), other.twins())
        // The same moves pasted with other players' names: a game of other players, two.
        other.addLibraryGame(stranger)
        Library.reconcile(other)
        assertEquals((listOf(a.key, b.key, pastedLate.key, opening.key, stranger.key)).sorted(), shown(other))
        assertEquals(emptyMap<String, String>(), other.twins())
        other.close()
    }

    /** Nothing paid for is lost: the pasted copy's Deep review and its words go to the site's record, which has a lower one. */
    @Test
    fun aCopysReviewAndPaidWordsGoToTheSurvivor() {
        val store = Store(null)
        val pasted = Library.fromPgn(LONGER, listOf("nosegazm")).single()
        val site = site(LONGER, "chesscom:9")
        store.addLibraryGame(pasted)
        val summary = JSONObject().put("white", JSONObject().put("accuracy", 71.5)).put("black", JSONObject().put("accuracy", 83.25))
        store.addLibraryGame(site)
        val twins = Library.reconcile(store)
        assertEquals(mapOf(pasted.key to site.key), twins)
        // Words the survivor lacks go to its review of the same strength; a review it lacks is carried only when
        // it can be read, so here the rows are bare: the words move, the unreadable report does not.
        assertTrue(store.saveReview(pasted.key, "deep", ReviewRoom.REVIEW_VERSION, 1.0, "{}", summary.toString()))
        store.setReviewWords(pasted.key, "deep", "{\"texts\":{\"4\":{\"why\":\"Paid for.\"}}}")
        assertTrue(store.saveReview(site.key, "deep", ReviewRoom.REVIEW_VERSION, 2.0, "{}", summary.toString()))
        ReviewRoom.carryTwins(store, twins)
        assertEquals("{\"texts\":{\"4\":{\"why\":\"Paid for.\"}}}", store.review(site.key, "deep")!!.words)
        assertTrue("the copy's own rows stay", store.review(pasted.key, "deep") != null)
        store.close()
    }

    companion object {
        // tests/test_review_server.py's LONGER: one move misread still leaves twelve to pair on.
        private const val LONGER = "[Event \"Live Chess\"]\n[Site \"Chess.com\"]\n[Date \"2026.09.29\"]\n" +
            "[White \"Anakin_737\"]\n[Black \"nosegazm\"]\n[Result \"0-1\"]\n[WhiteElo \"880\"]\n" +
            "[BlackElo \"1018\"]\n[TimeControl \"600\"]\n\n" +
            "1. e4 {[%clk 0:09:58]} e5 {[%clk 0:09:55]} 2. Nf3 {[%clk 0:09:50]} Nc6 {[%clk 0:09:40]}\n" +
            "3. Bc4 {[%clk 0:09:45]} Nf6 {[%clk 0:09:30]} 4. d3 Bc5 5. Nc3 d6 6. h3 Na5 7. Bb3 Nxb3 8. axb3 c6 0-1\n"
        private val SANS = "e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Bc5 Nc3 d6 h3 Na5 Bb3 Nxb3 axb3 c6".split(" ")
    }
}
