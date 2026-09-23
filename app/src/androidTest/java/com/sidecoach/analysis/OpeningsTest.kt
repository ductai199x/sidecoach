package com.sidecoach.analysis

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Move
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
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
        for (i in 0 until cases.length()) {
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
