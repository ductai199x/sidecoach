package com.sidecoach

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.profile.PuzzleStore
import com.sidecoach.profile.Puzzles
import com.sidecoach.profile.Store
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Wiping the profile and the caches ([Maintenance]): `tests/test_maintenance.py`'s first test
 * (parity st-T09), with the phone's theme files (`.bin` where the desktop pickles) and the
 * rollback journal Android's SQLite keeps beside a closed store. The desktop's second test is
 * the panel's API, which the phone has not.
 */
@RunWith(AndroidJUnit4::class)
class MaintenanceTest {

    private val csv = listOf(
        "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags",
        "fork1,r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2,g8f6 f3e5 c6e5," +
            "1000,80,90,500,fork opening short,https://lichess.org/x#3,",
    ).joinToString("\n") + "\n"

    private fun folder(): File {
        val dir = File(InstrumentationRegistry.getInstrumentation().targetContext.cacheDir, "maintenance-test")
        dir.deleteRecursively()
        dir.mkdirs()
        return dir
    }

    /** A profile with a game and an attempt, a theme file, a puzzle store with its CSV. */
    private fun filled(dir: File) {
        val store = Store(File(dir, "profile.sqlite"))
        store.startGame("g", 1.0, "white", STARTING_FEN, "s")
        store.addAttempt("fork1", 2.0, "fork", true, listOf("Nxe5"))
        store.setSetting("puzzle_level", "1125")
        store.close()
        File(dir, "themes").mkdirs()
        File(dir, "themes/theme-abc.bin").writeBytes(byteArrayOf(120))
        File(dir, "puzzles").mkdirs()
        File(dir, "puzzles/sample.csv").writeText(csv)
        val puzzles = Puzzles.openStore(dir)
        assertNotNull(puzzles)
        assertEquals(1, puzzles!!.count())
        puzzles.close()
    }

    @Test
    fun resetProfileEmptiesEveryTableAndClearCachesRemovesTheFiles() {
        val dir = folder()
        filled(dir)
        assertEquals(File(dir, "profile.sqlite"), Maintenance.resetProfile(dir))
        var store = Store(File(dir, "profile.sqlite"))
        assertNull(store.game("g"))
        assertTrue(store.attempts().isEmpty())
        assertNull(store.setting("puzzle_level"))
        assertEquals(0, store.gamesCount())
        store.close()
        // Android's SQLite keeps its rollback journal beside a closed store, emptied, where
        // Python's deletes it: the phone says one line more than the desktop.
        assertEquals(
            listOf("removed theme-abc.bin", "removed puzzles.sqlite", "removed puzzles.sqlite-journal"),
            Maintenance.clearCaches(dir),
        )
        assertFalse(File(dir, "themes/theme-abc.bin").exists())
        assertFalse(File(dir, "puzzles.sqlite").exists())
        assertEquals("nothing left", emptyList<String>(), Maintenance.clearCaches(dir))
        // With a store held open, the puzzles are rebuilt from the CSV rather than deleted.
        val puzzles = PuzzleStore(File(dir, "puzzles.sqlite"))
        assertEquals(0, puzzles.count())
        // The store, opened as `PuzzleStore` opens it, keeps a rollback journal and no
        // write-ahead log (the -wal and -shm the parity audit asked about, st-10): were that to
        // change, `clearCaches` would have two more files to remove.
        val mode = android.database.sqlite.SQLiteDatabase.openOrCreateDatabase(File(dir, "puzzles.sqlite"), null).use { db ->
            db.rawQuery("PRAGMA journal_mode", null).use { it.moveToFirst(); it.getString(0) }
        }
        assertEquals("truncate", mode)
        File(dir, "themes/theme-def.bin").writeBytes(byteArrayOf(120))
        assertEquals(
            listOf("removed theme-def.bin", "puzzle store rebuilt: 1 puzzles"),
            Maintenance.clearCaches(dir, puzzles),
        )
        assertEquals(1, puzzles.count())
        assertNotNull(puzzles.get("fork1"))
        puzzles.close()
        // A missing profile is fine.
        assertTrue(File(dir, "profile.sqlite").delete())
        assertEquals(File(dir, "profile.sqlite"), Maintenance.resetProfile(dir))
        dir.deleteRecursively()
    }
}
