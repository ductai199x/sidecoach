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
