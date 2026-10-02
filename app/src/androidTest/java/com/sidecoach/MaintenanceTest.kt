package com.sidecoach

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.analysis.Engine
import com.sidecoach.analysis.GameReview
import com.sidecoach.chess.Board
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.events.GameStarted
import com.sidecoach.events.Line
import com.sidecoach.events.MovePlayed
import com.sidecoach.profile.Library
import com.sidecoach.profile.Profiler
import com.sidecoach.profile.PuzzleStore
import com.sidecoach.profile.Puzzles
import com.sidecoach.profile.Store
import com.sidecoach.ui.EngineMaker
import com.sidecoach.ui.PanelEngine
import com.sidecoach.ui.ReviewRoom
import com.sidecoach.ui.SettingsStore
import com.sidecoach.vision.Pieces
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Wiping the profile and the caches, `tests/test_maintenance.py` (parity st-T09): [Maintenance]
 * on its own, and through the panel ([PanelEngine.resetProfile], [PanelEngine.clearCaches]).
 * The phone's theme files are `theme-*.bin` where the desktop pickles, and Android's SQLite
 * keeps a rollback journal beside a closed store.
 */
@RunWith(AndroidJUnit4::class)
class MaintenanceTest {

    private val app = InstrumentationRegistry.getInstrumentation().targetContext

    private fun dir(name: String): File =
        File(app.cacheDir, "maintenance-$name").apply { deleteRecursively(); mkdirs() }

    /** A profile with a game and an attempt, a theme file, a puzzle store with its CSV. */
    private fun filled(files: File) {
        val store = Store(File(files, "profile.sqlite"))
        store.startGame("g", 1.0, "white", STARTING_FEN, "s")
        store.addAttempt("fork1", 2.0, "fork", true, listOf("Nxe5"))
        store.setSetting("puzzle_level", "1125")
        store.close()
        File(files, "themes").mkdirs()
        File(files, "themes/theme-abc.bin").writeBytes(byteArrayOf(1))
        File(files, "puzzles").mkdirs()
        File(files, "puzzles/sample.csv").writeText(CSV)
        val puzzles = Puzzles.openStore(files)
        assertNotNull(puzzles)
        assertEquals(1, puzzles!!.count())
        puzzles.close()
    }

    /**
     * What clearing the caches of [filled] says: the desktop's two lines, and the journal's when
     * there is one -- Android's SQLite keeps a closed database's journal file, Python's deletes it.
     */
    private fun removed(files: File): List<String> =
        listOf("removed theme-abc.bin", "removed puzzles.sqlite") +
            (if (File(files, "puzzles.sqlite-journal").exists()) listOf("removed puzzles.sqlite-journal") else emptyList())

    @Test
    fun resetProfileEmptiesEveryTableAndClearCachesRemovesTheFiles() {
        val files = dir("own")
        filled(files)
        assertEquals(File(files, "profile.sqlite"), Maintenance.resetProfile(files))
        val store = Store(File(files, "profile.sqlite"))
        assertNull(store.game("g"))
        assertTrue(store.attempts().isEmpty())
        assertNull(store.setting("puzzle_level"))
        assertEquals(0, store.gamesCount())
        store.close()
        // Android's SQLite keeps its rollback journal beside a closed store, emptied, where
        // Python's deletes it: the phone says one line more than the desktop.
        assertEquals(
            listOf("removed theme-abc.bin", "removed puzzles.sqlite", "removed puzzles.sqlite-journal"),
            Maintenance.clearCaches(files),
        )
        assertFalse(File(files, "themes/theme-abc.bin").exists())
        assertFalse(File(files, "puzzles.sqlite").exists())
        assertEquals("nothing left", emptyList<String>(), Maintenance.clearCaches(files))
        // With a store held open, the puzzles are rebuilt from the CSV rather than unlinked.
        val puzzles = PuzzleStore(File(files, "puzzles.sqlite"))
        assertEquals(0, puzzles.count())
        // The store, opened as `PuzzleStore` opens it, keeps a rollback journal and no
        // write-ahead log (the -wal and -shm the parity audit asked about, st-10): were that to
        // change, `clearCaches` would have two more files to remove.
        val mode = android.database.sqlite.SQLiteDatabase.openOrCreateDatabase(File(files, "puzzles.sqlite"), null).use { db ->
            db.rawQuery("PRAGMA journal_mode", null).use { it.moveToFirst(); it.getString(0) }
        }
        assertEquals("truncate", mode)
        File(files, "themes/theme-def.bin").writeBytes(byteArrayOf(1))
        assertEquals(
            listOf("removed theme-def.bin", "puzzle store rebuilt: 1 puzzles"),
            Maintenance.clearCaches(files, puzzles),
        )
        assertEquals(1, puzzles.count())
        assertNotNull(puzzles.get("fork1"))
        puzzles.close()
        // A missing profile is fine.
        assertTrue(File(files, "profile.sqlite").delete())
        assertEquals(File(files, "profile.sqlite"), Maintenance.resetProfile(files))
        files.deleteRecursively()
    }

    @Test
    fun thePanelResetsTheProfileAndClearsTheCaches() {
        val files = dir("panel")
        filled(files)
        val before = PanelEngine.profiler
        PanelEngine.profiler = null
        PanelEngine.practiceIds = listOf("fork1")
        PanelEngine.puzzleMotifs["fork1"] = "fork"
        try {
            val cursor = PanelEngine.poll(Int.MAX_VALUE).getInt("cursor")
            assertEquals(removed(files), PanelEngine.clearCaches(files))
            assertTrue(PanelEngine.resetProfile(files).endsWith("profile.sqlite"))
            val store = Store(File(files, "profile.sqlite"))
            assertTrue(store.gamesCount() == 0 && store.attempts().isEmpty())
            assertTrue(PanelEngine.practiceIds.isEmpty() && PanelEngine.puzzleMotifs.isEmpty())
            val lines = PanelEngine.poll(cursor).getJSONArray("lines")
            val texts = (0 until lines.length()).map { lines.getJSONObject(it).getString("text") }
            assertTrue(texts.toString(), texts.any { it.startsWith("profile reset:") })
            assertTrue(texts.toString(), texts.any { it.startsWith("cache: removed theme-abc.bin") })
            // A running profiler's game in progress is dropped with the reset.
            val profiler = Profiler(store, sessionId = "s", ts = 0.0)
            profiler.onEvent(GameStarted("g2", Pieces.Side.WHITE, STARTING_FEN), 1.0)
            PanelEngine.profiler = profiler
            PanelEngine.resetProfile(files)
            assertNull(store.game("g2"))
            val board = Board()
            val move = board.parseSan("e4")!!
            board.push(move)
            profiler.onEvent(MovePlayed("g2", 1, "e4", move.uci(), STARTING_FEN, board.fen(), byUser = true), 2.0)
            assertEquals("the dropped game takes no more moves", emptyList<Any>(), store.moves("g2"))
            store.close()
        } finally {
            PanelEngine.profiler = before
            PanelEngine.set("profile" to null, "review" to null, "practice" to null)
        }
    }

    /**
     * The review of 2026-10-01: Reset profile while a game was reviewed emptied the tables under
     * the review, which then saved itself and its Stats facts for a game no longer in the library
     * -- Stats counted a game nothing listed and nothing could remove. A game is in the record
     * while its import or its watched record is; a review saves only such a game. The desktop's
     * test_a_review_under_way_saves_nothing_for_a_profile_reset_meanwhile (the Pause it keeps is
     * the desktop's background review's: the phone has none).
     */
    @Test
    fun aReviewUnderWaySavesNothingForAProfileResetMeanwhile() {
        val files = dir("reset-review")
        val path = File(files, "profile.sqlite")
        val row = Library.fromPgn(PGN, listOf("nosegazm")).single()
        val store = Store(path)
        store.addLibraryGame(row)
        store.startGame("w1", 1.0, "black", STARTING_FEN)
        assertTrue(store.inRecord(row.key) && store.inRecord("watched:w1"))
        Maintenance.resetProfile(files)
        assertFalse("reset: no longer in the record", store.inRecord(row.key))
        assertFalse(store.inRecord("watched:w1"))
        store.addLibraryGame(row)
        store.removeGame(row.key)
        assertFalse("removed: no longer in the record", store.inRecord(row.key))
        store.close()

        // A review under way when the profile is reset saves nothing. Skipped without the engine.
        val binary = Engine.binaryIn(app.applicationInfo.nativeLibraryDir)
        val net = app.filesDir.listFiles { f: File -> f.name.endsWith(".nnue") }?.firstOrNull()
        assumeTrue("the engine or its network is missing", binary.exists() && net != null)
        val other = dir("reset-review-engine")
        val own = Store(File(other, "profile.sqlite"))
        own.addLibraryGame(row)
        var searched = 0
        val maker = EngineMaker { threads, multipv ->
            object : Engine(binary, net, threads = threads, hashMb = 64, multipv = multipv) {
                override fun analyse(
                    board: Board,
                    depth: Int?,
                    timeS: Double?,
                    multipv: Int?,
                    pvPlies: Int?,
                    carryBest: Boolean,
                    onDepth: ((List<Line>) -> Unit)?,
                ): List<Line> {
                    searched += 1
                    if (searched == 4) Maintenance.resetProfile(other) // pressed while it searches
                    return super.analyse(board, depth, timeS, multipv, pvPlies, carryBest, onDepth)
                }
            }
        }
        val fast = GameReview.STRENGTHS.getValue("fast")
        assertNull(ReviewRoom.reviewRow(SettingsStore(app), other, row, own, fast, 1, maker))
        assertNull(own.review(row.key, "fast"))
        assertTrue(own.statGames().isEmpty())
        assertTrue(own.progress(row.key, "fast", fast.depth, fast.multipv).isEmpty())
        own.close()
    }

    /**
     * The audit of 2026-10-02 (For Tai #14): Reset profile landing in the instant after the review
     * asked whether the game is in the record, or after its review was saved and before its Stats
     * facts, still saved them for a game no longer in the record. The saves ask it themselves, in
     * the same statement: staged at that instant -- the review looked, then the reset on another
     * connection -- neither is saved. The desktop's
     * test_a_reset_between_the_reviews_look_and_its_save_saves_nothing.
     */
    @Test
    fun aResetBetweenTheReviewsLookAndItsSaveSavesNothing() {
        val files = dir("reset-between")
        val store = Store(File(files, "profile.sqlite"))
        val row = Library.fromPgn(PGN, listOf("nosegazm")).single()
        store.addLibraryGame(row)
        store.startGame("w1", 1.0, "black", STARTING_FEN)
        for (key in listOf(row.key, "watched:w1")) assertTrue("the review's look", store.inRecord(key))
        Maintenance.resetProfile(files) // lands now, on another connection
        for (key in listOf(row.key, "watched:w1")) {
            assertFalse("no review of $key", store.saveReview(key, "fast", ReviewRoom.REVIEW_VERSION, 1.0, "{}", null))
            assertNull(store.review(key, "fast"))
            val game = com.sidecoach.profile.GameFact(
                key = key, source = "pgn", timeClass = null, playedAt = 1.0, userSide = "black",
                bot = false, result = "0-1", opening = null, termination = null,
            )
            store.saveStats("fast", game, emptyList(), held = true)
        }
        assertTrue("nor its Stats facts", store.statGames().isEmpty())
        // In the record, both are saved as before.
        store.addLibraryGame(row)
        assertTrue(store.saveReview(row.key, "fast", ReviewRoom.REVIEW_VERSION, 1.0, "{}", null))
        store.close()
    }

    private companion object {
        const val PGN = "[White \"Anakin_737\"]\n[Black \"nosegazm\"]\n[Result \"0-1\"]\n[TimeControl \"600\"]\n\n" +
            "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. d3 Bc5 5. Nc3 d6 6. h3 Na5 0-1\n"
        const val CSV = "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags\n" +
            "fork1,r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2,g8f6 f3e5 c6e5," +
            "1000,80,90,500,fork opening short,https://lichess.org/x#3,\n"
    }
}
