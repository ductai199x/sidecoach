package com.sidecoach.ui

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.analysis.GameReviewParityTest
import com.sidecoach.chess.Board
import com.sidecoach.events.ReviewJson
import com.sidecoach.profile.LibraryParityTest
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.time.ZoneOffset

/**
 * Game Review's Analysis as the desktop's (ui/ReviewAnalysis.kt): `tools/make_analysis_fixtures.py`
 * records `review_analysis.py` on the seven real reviews and random games -- positions and their
 * legal moves, moves played and refused, the engine's lines (each position's own measurement
 * answered), the explorer of the games, each game's PGN with side lines grafted on (annotated
 * with the review and the AI coach's words, and plain), the file names, the move comments.
 */
@RunWith(AndroidJUnit4::class)
class ReviewAnalysisParityTest {
    private val fixture: JSONObject by lazy {
        val context = InstrumentationRegistry.getInstrumentation().context
        JSONObject(context.assets.open("ui/analysis.json").bufferedReader().use { it.readText() })
    }

    private fun same(path: String, mine: Any?, theirs: Any?, failures: MutableList<String>) =
        GameReviewParityTest.same(path, mine, theirs, failures)

    @Test
    fun positionsAndMovesAreTheDesktops() {
        val failures = ArrayList<String>()
        val positions = fixture.getJSONArray("positions")
        for (i in 0 until positions.length()) {
            val p = positions.getJSONObject(i)
            same("position $i", ReviewAnalysis.position(p.getString("fen")), p.getJSONObject("out"), failures)
        }
        val moves = fixture.getJSONArray("moves")
        for (i in 0 until moves.length()) {
            val m = moves.getJSONObject(i)
            val label = "move ${m.getString("uci")} in ${m.getString("fen")}"
            try {
                val got = ReviewAnalysis.move(m.getString("fen"), m.getString("uci"))
                if (m.has("error")) failures.add("$label: played, python refuses (${m.getString("error")})")
                else same(label, got, m.getJSONObject("out"), failures)
            } catch (exc: IllegalArgumentException) {
                if (!m.has("error")) failures.add("$label: refused (${exc.message}), python plays it")
                else if (exc.message != m.getString("error")) failures.add("$label: '${exc.message}', python '${m.getString("error")}'")
            }
        }
        assertTrue("${failures.size} failures:\n" + failures.take(30).joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun theEnginesLinesAreDrawnAsTheDesktops() {
        val failures = ArrayList<String>()
        val lines = fixture.getJSONArray("lines")
        for (i in 0 until lines.length()) {
            val l = lines.getJSONObject(i)
            val found = l.getJSONArray("found").let { f -> (0 until f.length()).map { ReviewJson.line(f.getJSONObject(it)) } }
            same("lines $i", ReviewAnalysis.linesJson(Board(l.getString("fen")), found), l.getJSONObject("out"), failures)
        }
        assertTrue("${lines.length()} positions; ${failures.size} failures:\n" + failures.take(30).joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun theExplorerIsTheDesktops() {
        val failures = ArrayList<String>()
        val rows = fixture.getJSONArray("rows").let { r -> (0 until r.length()).map { LibraryParityTest.libraryRow(r.getJSONObject(it)) } }
        val explorer = ReviewAnalysis.Explorer(rows)
        val explore = fixture.getJSONArray("explore")
        for (i in 0 until explore.length()) {
            val e = explore.getJSONObject(i)
            same("explore ${e.getString("fen")}", explorer.explore(e.getString("fen")), e.getJSONObject("out"), failures)
        }
        assertTrue("${failures.size} failures:\n" + failures.take(30).joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun theGameAsAPgnIsTheDesktops() {
        val failures = ArrayList<String>()
        val pgns = fixture.getJSONArray("pgn")
        for (i in 0 until pgns.length()) {
            val p = pgns.getJSONObject(i)
            val row = LibraryParityTest.libraryRow(p.getJSONObject("row"))
            val got = ReviewAnalysis.pgn(
                row,
                p.getJSONArray("moves"),
                p.getJSONArray("tree"),
                annotated = p.getBoolean("annotated"),
                words = p.optJSONObject("words"),
            )
            if (got != p.getString("out")) failures.add("pgn $i (annotated ${p.getBoolean("annotated")})\n  mine:   $got\n  python: ${p.getString("out")}")
        }
        val names = fixture.getJSONArray("names")
        for (i in 0 until names.length()) {
            val n = names.getJSONObject(i)
            val got = ReviewAnalysis.pgnName(LibraryParityTest.libraryRow(n.getJSONObject("row")), ZoneOffset.UTC)
            if (got != n.getString("out")) failures.add("name $i: '$got', python '${n.getString("out")}'")
        }
        val comments = fixture.getJSONArray("comments")
        for (i in 0 until comments.length()) {
            val c = comments.getJSONObject(i)
            val got = ReviewAnalysis.comment(
                if (c.isNull("label")) null else c.getString("label"),
                if (c.isNull("best")) null else c.getString("best"),
                c.getString("played"),
                c.optJSONObject("eval"),
            )
            if (got != c.getString("out")) failures.add("comment $i: '$got', python '${c.getString("out")}'")
        }
        assertTrue("${pgns.length()} PGNs; ${failures.size} failures:\n" + failures.take(10).joinToString("\n"), failures.isEmpty())
    }
}
