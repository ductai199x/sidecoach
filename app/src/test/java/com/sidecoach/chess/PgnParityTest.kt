package com.sidecoach.chess

import com.sidecoach.TestAssets
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * PGN read and written as python-chess does (chess/Pgn.kt): `tools/make_pgn_fixtures.py`
 * records, for 62 texts -- the awkward ones by hand, three chess.com archive games, forty
 * random games with clocks, NAGs, comments and side lines -- every game `read_game` gives:
 * its headers in order, its main line, each move's clock, whether it has errors, and the text
 * `str(game)` writes. The one known difference: a variant game (Atomic), read by python-chess
 * on a board of its own and here as errored (no variant boards; the library leaves variants
 * out).
 */
class PgnParityTest {

    @Test
    fun everyGameReadsAndWritesAsThePythonDoes() {
        val text = TestAssets.text("chess/pgn.json")
        val cases = JSONObject(text).getJSONArray("cases")
        val failures = ArrayList<String>()
        var games = 0
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val name = case.getString("name")
            if (name == "a variant") continue
            val want = case.getJSONArray("games")
            val got = PgnReader.games(case.getString("text"))
            if (got.size != want.length()) {
                failures.add("$name: ${got.size} games, python reads ${want.length()}")
                continue
            }
            for (g in 0 until want.length()) {
                games += 1
                val w = want.getJSONObject(g)
                val game = got[g]
                fun check(what: String, mine: Any?, theirs: Any?) {
                    if (mine != theirs && failures.size < 30) failures.add("$name #$g: $what\n  mine:   $mine\n  python: $theirs")
                }
                val headers = w.getJSONArray("headers")
                check(
                    "headers",
                    game.headers.items().map { listOf(it.first, it.second) },
                    (0 until headers.length()).map { h -> headers.getJSONArray(h).let { listOf(it.getString(0), it.getString(1)) } },
                )
                val moves = w.getJSONArray("moves")
                check("moves", game.mainlineMoves().map { it.uci() }, (0 until moves.length()).map { moves.getString(it) })
                val clocks = w.getJSONArray("clocks")
                check("clocks", game.mainline().map { it.clock() }, (0 until clocks.length()).map { if (clocks.isNull(it)) null else clocks.getDouble(it) })
                check("errors", game.errors.isNotEmpty(), w.getBoolean("errors"))
                if (!w.getBoolean("errors")) check("str(game)", game.toString(), w.getString("text"))
            }
        }
        assertTrue("$games games; ${failures.size} failures:\n" + failures.joinToString("\n"), failures.isEmpty())
    }
}
