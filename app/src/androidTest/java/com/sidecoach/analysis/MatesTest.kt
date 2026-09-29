package com.sidecoach.analysis

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.MotifTag
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The mating-pattern namer held to `analysis/mates.py`, name for name and square for square.
 *
 * A wrong name teaches the player the wrong picture, so every checkmate the puzzle sets end
 * in is named here as the desktop names it (`tools/make_mates_fixtures.py`): the per-theme
 * set, with a hundred-odd of each of the nineteen patterns, the sample, and the desktop's own
 * positives and negatives.
 */
@RunWith(AndroidJUnit4::class)
class MatesTest {

    @Test
    fun everyMateIsNamedAsThePythonNamesIt() {
        val context = InstrumentationRegistry.getInstrumentation().context
        val text = context.assets.open("chess/mates.json").bufferedReader().use { it.readText() }
        val cases = JSONObject(text).getJSONArray("cases")
        val failures = ArrayList<String>()
        val seen = HashMap<String, Int>()
        val began = System.currentTimeMillis()
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val fen = case.getString("fen")
            val board = Board(fen)
            val mine = Mates.matingPatternTags(board, Move.fromUci(case.getString("move")))
            val wanted = case.getJSONArray("tags").let { a ->
                (0 until a.length()).map { k ->
                    val o = a.getJSONObject(k)
                    val squares = o.getJSONArray("squares").let { s -> List(s.length()) { s.getString(it) } }
                    MotifTag(o.getString("name"), squares, null)
                }
            }
            for (tag in mine) seen[tag.name] = (seen[tag.name] ?: 0) + 1
            if (mine != wanted && failures.size < 8) {
                failures.add(
                    "${case.getString("source")} $fen ${case.getString("move")}:\n" +
                        "  kotlin: ${mine.joinToString(" | ") { "${it.name}[${it.squares.joinToString(",")}]" }}\n" +
                        "  python: ${wanted.joinToString(" | ") { "${it.name}[${it.squares.joinToString(",")}]" }}",
                )
            }
            if (board.fen() != fen && failures.size < 8) failures.add("$fen: the board was changed")
        }
        val took = System.currentTimeMillis() - began
        Log.i(
            TAG,
            "${cases.length()} mates named in $took ms; " +
                Mates.MATE_PATTERNS.joinToString(", ") { "${it}x${seen[it] ?: 0}" },
        )
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /**
     * `test_a_check_that_is_not_mate_names_nothing`: Nd6+ gives check but the king walks out,
     * so no pattern is named -- the fixture's positions are all mates, so the guard that a
     * position that is not checkmate names nothing was never reached (the parity audit,
     * 2026-09-29).
     */
    @Test
    fun aCheckThatIsNotMateNamesNothing() {
        val board = Board("4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1")
        val move = Move.fromUci("e4d6")
        board.push(move)
        assertTrue("check, not mate", board.isCheck() && !board.isCheckmate())
        assertTrue(Mates.matingPatternTags(board, move).isEmpty())
    }

    @Test
    fun theTagOrderEndsWithTheMatePatterns() {
        // `motifs.TAG_ORDER` ends with `mates.MATE_PATTERNS`, in that order.
        val tail = Motifs.TAG_ORDER.takeLast(Mates.MATE_PATTERNS.size)
        assertTrue("$tail", tail == Mates.MATE_PATTERNS)
    }

    private companion object {
        const val TAG = "sidecoach.analysis"
    }
}
