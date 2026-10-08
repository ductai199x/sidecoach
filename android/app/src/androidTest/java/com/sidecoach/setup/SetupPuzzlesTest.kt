package com.sidecoach.setup

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.github.luben.zstd.Zstd
import java.io.ByteArrayInputStream
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The puzzle sample Setup fetches (the "puzzles" item): the Lichess file streamed and
 * decompressed, a few puzzles kept per theme, the progress in the item's own units.
 * `tests/test_setup_puzzles.py`, row for row.
 */
@RunWith(AndroidJUnit4::class)
class SetupPuzzlesTest {

    private val header = "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags"
    private val fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1"

    private fun row(i: Int, themes: String): String =
        "p${"%05d".format(i)},$fen,g8h8 a1a8,${900 + i % 700},70,90,100,$themes,https://lichess.org/x#1,"

    @Test
    fun theSampleKeepsAFewPerThemeAndReportsInBytes() {
        val lines = arrayListOf(header)
        // Three forks, two pins (one row is both): the pins never fill, so the scan runs on to
        // its limit through rows nothing wants.
        lines += listOf(row(0, "fork short"), row(1, "pin long"), row(2, "fork pin"), row(3, "fork"))
        for (i in 4 until 40_000) lines += row(i, "endgame")
        val blob = Zstd.compress((lines.joinToString("\n") + "\n").toByteArray(Charsets.UTF_8))

        val app = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = File(app.cacheDir, "setup-puzzles-${System.nanoTime()}").apply { mkdirs() }
        try {
            val item = items(app).first { it.id == "puzzles" }
            assertTrue(item.optional)
            val target = File(dir, "puzzles/sample.csv")
            assertFalse(target.exists())
            val seen = ArrayList<Pair<Long, Long?>>()
            fetchPuzzles(
                target,
                { done, total -> seen.add(done to total) },
                source = { ByteArrayInputStream(blob) },
                themes = listOf("fork", "pin"),
                perTheme = 3,
                maxScan = 40_000,
            )

            assertTrue(target.exists())
            val written = target.readText(Charsets.UTF_8).lines().filter { it.isNotEmpty() }
            // The forks, then the pins; a puzzle of both is written once, where it first came.
            assertEquals(header, written[0])
            assertEquals(listOf("p00000", "p00002", "p00003", "p00001"), written.drop(1).map { it.split(",")[0] })
            // csv.writer's line ending.
            assertTrue(target.readText(Charsets.UTF_8).startsWith("$header\r\n"))
            // The bar moves in the item's bytes, never past its size, and ends on it.
            assertEquals(listOf(item.size / 2 to item.size, item.size to item.size), seen)
            assertFalse("put in place whole, nothing left beside it", File(dir, "puzzles/sample.csv.part").exists())
        } finally {
            dir.deleteRecursively()
        }
    }

    /**
     * A write cut short left half a sample.csv that Setup took as done for good (audit
     * 2026-10-07, S4): it is written beside and put in place whole. The write is made to fail
     * here by a folder where the file beside would go. `test_a_sample_written_half_is_not_taken_as_done`.
     */
    @Test
    fun aSampleWrittenHalfIsNotTakenAsDone() {
        val lines = listOf(header, row(0, "fork"), row(1, "fork"))
        val blob = Zstd.compress((lines.joinToString("\n") + "\n").toByteArray(Charsets.UTF_8))
        val app = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = File(app.cacheDir, "setup-puzzles-${System.nanoTime()}").apply { mkdirs() }
        try {
            val target = File(dir, "puzzles/sample.csv")
            File(dir, "puzzles/sample.csv.part").mkdirs()
            val failed = runCatching {
                fetchPuzzles(target, { _, _ -> }, source = { ByteArrayInputStream(blob) }, themes = listOf("fork"), perTheme = 2, maxScan = 10)
            }
            assertTrue(failed.isFailure)
            assertFalse("half a sample is not the sample", target.exists())
        } finally {
            dir.deleteRecursively()
        }
    }
}
