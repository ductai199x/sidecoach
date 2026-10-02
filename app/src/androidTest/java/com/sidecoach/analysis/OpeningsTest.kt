package com.sidecoach.analysis

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.Lane
import com.sidecoach.Parallel
import com.sidecoach.chess.Move
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * The opening book held to `sidecoach/analysis/openings.py`: over the real Lichess tables,
 * every named line, prefixes, deviations, random games and transpositions -- the name, whether
 * the moves are in theory, and where they left it -- answered exactly as the Python did
 * (`tools/make_openings_fixtures.py`). Once compiled from the tables, once from the cache the
 * first load wrote.
 */
@Lane(Lane.Kind.FREE)
@RunWith(AndroidJUnit4::class)
class OpeningsTest {

    private val instrumentation = InstrumentationRegistry.getInstrumentation()

    private fun tables(): File {
        val folder = File(instrumentation.targetContext.cacheDir, "openings-test").apply {
            deleteRecursively()
            mkdirs()
        }
        val assets = instrumentation.context.assets
        for (name in assets.list("openings").orEmpty().filter { it.endsWith(".tsv") }) {
            assets.open("openings/$name").use { input ->
                File(folder, name).outputStream().use { input.copyTo(it) }
            }
        }
        return folder
    }

    private fun fixture(): JSONObject {
        val text = instrumentation.context.assets.open("openings/book.json").bufferedReader().use { it.readText() }
        return JSONObject(text)
    }

    private fun check(book: OpeningBook, data: JSONObject, pass: String) {
        assertEquals("$pass: named lines", data.getInt("names"), book.size)
        assertEquals("$pass: positions", data.getInt("positions"), book.positionCount)
        val cases = data.getJSONArray("cases")
        // The cases side by side on the cores: the book is only read, each trace on a board of
        // its own, and the first case that differs fails the test as before.
        Parallel.forEach((0 until cases.length()).toList()) { i ->
            val case = cases.getJSONObject(i)
            val ucis = case.getJSONArray("moves")
            val moves = (0 until ucis.length()).map { Move.fromUci(ucis.getString(it)) }
            val where = "$pass case $i ${(0 until ucis.length()).joinToString(" ") { ucis.getString(it) }}"
            val want = case.optJSONObject("identify")
            val got = book.identify(moves)
            if (want == null) {
                assertNull(where, got)
            } else {
                assertNotNull(where, got)
                assertEquals(where, want.getString("eco"), got!!.eco)
                assertEquals(where, want.getString("name"), got.name)
                assertEquals(where, want.getInt("ply"), got.ply)
            }
            assertEquals(where, case.getBoolean("in_book"), book.inBook(moves))
            val deviation = if (case.isNull("deviation")) null else case.getInt("deviation")
            assertEquals(where, deviation, book.deviation(moves))
        }
    }

    @Test
    fun everyAnswerMatchesThePythonCompiledAndCached() {
        val folder = tables()
        val data = fixture()
        val compiled = OpeningBook.load(folder)
        assertNotNull("the tables load", compiled)
        check(compiled!!, data, "compiled")
        assert(File(folder, OpeningBook.CACHE_FILE).isFile) { "the first load writes the cache" }
        val cached = OpeningBook.load(folder)
        assertNotNull("the cache loads", cached)
        check(cached!!, data, "cached")
    }

    /** SAN to the moves a game would hand the book: `test_openings.moves`. */
    private fun moves(line: String): List<Move> {
        val board = com.sidecoach.chess.Board()
        return line.split(" ").filter { it.isNotEmpty() && !it[0].isDigit() }.map { san ->
            board.parseSan(san)!!.also { board.push(it) }
        }
    }

    private fun folder(name: String): File =
        File(instrumentation.targetContext.cacheDir, name).apply {
            deleteRecursively()
            mkdirs()
        }

    /**
     * The cache beside the tables (`test_cache_is_written_and_reused`,
     * `test_changed_tables_recompile`, `test_corrupt_cache_is_ignored`): written by the first
     * load and answering the next, dropped when a table changes, ignored when it is corrupt.
     */
    @Test
    fun theCacheFollowsItsTables() {
        val folder = folder("openings-cache")
        instrumentation.context.assets.open("openings_sample/c.tsv").use { input ->
            File(folder, "c.tsv").outputStream().use { input.copyTo(it) }
        }
        val italian = moves("1. e4 e5 2. Nf3 Nc6 3. Bc4")
        val first = OpeningBook.load(folder)!!
        val cache = File(folder, OpeningBook.CACHE_FILE)
        assertTrue("the first load writes the cache", cache.isFile)
        assertEquals(first.identify(italian), OpeningBook.load(folder)!!.identify(italian))

        cache.writeText("{not json")
        assertEquals(Opening("C50", "Italian Game", 5), OpeningBook.load(folder)!!.identify(italian))

        File(folder, "c.tsv").writeText("eco\tname\tpgn\nA00\tAmar Opening\t1. Nh3\n")
        val rebuilt = OpeningBook.load(folder)!!
        assertEquals(Opening("A00", "Amar Opening", 1), rebuilt.identify(moves("1. Nh3")))
        assertNull(rebuilt.identify(moves("1. e4")))
    }

    private fun sample(name: String): OpeningBook {
        val folder = folder(name)
        instrumentation.context.assets.open("openings_sample/c.tsv").use { input ->
            File(folder, "c.tsv").outputStream().use { input.copyTo(it) }
        }
        return OpeningBook.load(folder)!!
    }

    /**
     * `test_a_game_that_transposes_back_into_theory_takes_that_name`: Lichess names the last named
     * position reached, wherever the game went before it -- 1.e4 e5 2.Bc4 leaves the tables, and
     * 2...Nc6 3.Nf3 comes back to the Italian Game (Tai, 2026-10-02). Still out of book from 2.Bc4.
     */
    @Test
    fun aGameThatTransposesBackIntoTheoryTakesThatName() {
        val book = sample("openings-transposed")
        val line = moves("1. e4 e5 2. Bc4 Nc6 3. Nf3")
        assertEquals(2, book.deviation(line))
        assertEquals(Opening("C50", "Italian Game", 5), book.identify(line))
    }

    /**
     * `test_the_name_is_read_from_the_first_forty_plies`: nine rounds of knights out and back (36
     * plies, nothing named), then 1.e4 e5 2.Nf3 at plies 37-39: the King's Knight Opening. Nc6
     * and Bc4 make the Italian position at ply 41, past Lichess's forty, so the name stays.
     */
    @Test
    fun theNameIsReadFromTheFirstFortyPlies() {
        val book = sample("openings-forty")
        val line = moves(List(9) { "Nf3 Nf6 Ng1 Ng8" }.joinToString(" ") + " e4 e5 Nf3 Nc6 Bc4")
        assertEquals(41, line.size)
        assertEquals(Opening("C40", "King's Knight Opening", 39), book.identify(line))
    }

    /** `test_unparsable_line_is_skipped`: a line that cannot be played costs only itself. */
    @Test
    fun anUnparsableLineIsSkipped() {
        val folder = folder("openings-unparsable")
        File(folder, "x.tsv").writeText("eco\tname\tpgn\nA00\tNonsense\t1. Qz9\nA00\tAmar Opening\t1. Nh3\n")
        assertEquals(Opening("A00", "Amar Opening", 1), OpeningBook.load(folder)!!.identify(moves("1. Nh3")))
    }

    @Test
    fun noTablesIsNoBook() {
        val empty = File(instrumentation.targetContext.cacheDir, "openings-none").apply {
            deleteRecursively()
            mkdirs()
        }
        assertNull(OpeningBook.load(empty))
        assertNull(OpeningBook.load(File(empty, "never-fetched")))
    }
}
