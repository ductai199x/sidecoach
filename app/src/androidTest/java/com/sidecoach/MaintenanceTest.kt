package com.sidecoach

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.events.GameStarted
import com.sidecoach.events.MovePlayed
import com.sidecoach.profile.Profiler
import com.sidecoach.profile.PuzzleStore
import com.sidecoach.profile.Puzzles
import com.sidecoach.profile.Store
import com.sidecoach.ui.PanelEngine
import com.sidecoach.vision.Pieces
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
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

    private companion object {
        const val CSV = "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags\n" +
            "fork1,r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2,g8f6 f3e5 c6e5," +
            "1000,80,90,500,fork opening short,https://lichess.org/x#3,\n"
    }
}
