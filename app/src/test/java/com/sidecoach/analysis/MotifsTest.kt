package com.sidecoach.analysis

import com.sidecoach.TestAssets
import com.sidecoach.TestLog
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.MotifTag
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The motif tagger held to the desktop's, tag for tag and square for square.
 *
 * The tagger gives the coach its theme, its focus squares and the sentence it says about a
 * tactic, so a port that disagrees anywhere changes what the coach teaches. The positions
 * are the Lichess puzzle sample the desktop validates against -- real tactics, walked move
 * by move, which is where a tagger does its work.
 */
class MotifsTest {

    private fun fixtures(): JSONObject {
        val text = TestAssets.text("chess/motifs.json")
        return JSONObject(text)
    }

    @Test
    fun everyTagMatchesThePython() {
        val data = fixtures().getJSONArray("cases")
        val failures = ArrayList<String>()
        val seen = HashMap<String, Int>()
        var payoffs = 0
        var paid = 0
        val began = System.currentTimeMillis()
        for (i in 0 until data.length()) {
            val case = data.getJSONObject(i)
            val board = Board(case.getString("fen"))
            val move = Move.fromUci(case.getString("move"))
            val pv = case.getJSONArray("pv").let { a ->
                List(a.length()) { Move.fromUci(a.getString(it)) }
            }
            val mateIn = if (case.isNull("mate_in")) null else case.getInt("mate_in")

            val mine = Motifs.tagsForMove(board, move, pv, mateIn)
            compare("${case.getString("puzzle")} move", mine, case.getJSONArray("for_move"), failures)
            for (tag in mine) seen[tag.name] = (seen[tag.name] ?: 0) + 1

            val position = Motifs.tagsForPosition(board, listOf(move) + pv, mateIn)
            compare(
                "${case.getString("puzzle")} position",
                position,
                case.getJSONArray("for_position"),
                failures,
            )
            for (tag in position) seen[tag.name] = (seen[tag.name] ?: 0) + 1

            // `motifs.pays_off` at ply 0 of [move, *pv], walked as the annotator walks it.
            val wanted = case.optJSONArray("pays_off")
            if (wanted != null && wanted.length() > 0) {
                val line = listOf(move) + pv
                val boards = arrayListOf(board.copy(withStack = false))
                for (m in line) {
                    if (!boards.last().isLegal(m)) break
                    val nxt = boards.last().copy(withStack = false)
                    nxt.push(m)
                    boards.add(nxt)
                }
                val walked = line.subList(0, boards.size - 1)
                for (k in 0 until wanted.length()) {
                    val o = wanted.getJSONObject(k)
                    val squares = o.getJSONArray("squares").let { a -> List(a.length()) { a.getString(it) } }
                    val tag = MotifTag(o.getString("name"), squares, if (o.isNull("move")) null else o.getString("move"))
                    val mine = Motifs.paysOff(boards, walked, 0, tag)
                    payoffs += 1
                    if (mine) paid += 1
                    if (mine != o.getBoolean("pays") && failures.size < 8) {
                        failures.add("${case.getString("puzzle")} pays_off ${show(tag)}: kotlin $mine, python ${o.getBoolean("pays")}")
                    }
                }
            }
        }
        val took = System.currentTimeMillis() - began
        TestLog.i(
            TAG,
            "${data.length()} positions tagged in $took ms, $payoffs payoffs ($paid paid); " +
                seen.entries.sortedBy { it.key }.joinToString(", ") { "${it.key}x${it.value}" },
        )
        assertTrue(failures.take(8).joinToString("\n"), failures.isEmpty())
    }

    /**
     * The desktop's own hand-built positions: every tagger call `tests/test_motifs.py` makes --
     * the shapes each tag must and must not be given, the illegal moves that tag as nothing.
     */
    @Test
    fun theDesktopsHandBuiltPositionsTagAsThePythonTagsThem() {
        val data = fixtures().getJSONArray("hand")
        val failures = ArrayList<String>()
        for (i in 0 until data.length()) {
            val case = data.getJSONObject(i)
            val board = Board(case.getString("fen"))
            val pv = case.getJSONArray("pv").let { a -> List(a.length()) { Move.fromUci(a.getString(it)) } }
            val mateIn = if (case.isNull("mate_in")) null else case.getInt("mate_in")
            val kind = case.getString("kind")
            val mine = if (kind == "move") {
                Motifs.tagsForMove(board, Move.fromUci(case.getString("move")), pv, mateIn)
            } else {
                Motifs.tagsForPosition(board, pv, mateIn)
            }
            val move = if (case.isNull("move")) "" else case.getString("move")
            compare("${case.getString("fen")} $kind $move $pv $mateIn", mine, case.getJSONArray("tags"), failures)
        }
        TestLog.i(TAG, "${data.length()} hand-built tagger calls checked")
        assertTrue(failures.take(8).joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun thePhaseMatchesThePython() {
        val data = fixtures().getJSONArray("phases")
        val failures = ArrayList<String>()
        for (i in 0 until data.length()) {
            val o = data.getJSONObject(i)
            val mine = phaseOf(Board(o.getString("fen")))
            if (mine != o.getString("phase")) {
                failures.add("${o.getString("fen")}: $mine, python says ${o.getString("phase")}")
            }
        }
        TestLog.i(TAG, "${data.length()} phases checked")
        assertTrue(failures.take(8).joinToString("\n"), failures.isEmpty())
    }

    private fun compare(
        what: String,
        mine: List<MotifTag>,
        theirs: org.json.JSONArray,
        failures: MutableList<String>,
    ) {
        val wanted = (0 until theirs.length()).map { i ->
            val o = theirs.getJSONObject(i)
            val squares = o.getJSONArray("squares").let { a -> List(a.length()) { a.getString(it) } }
            MotifTag(o.getString("name"), squares, if (o.isNull("move")) null else o.getString("move"))
        }
        if (mine == wanted) return
        if (failures.size < 8) {
            failures.add(
                "$what:\n  kotlin: ${mine.joinToString(" | ") { show(it) }}" +
                    "\n  python: ${wanted.joinToString(" | ") { show(it) }}",
            )
        }
    }

    private fun show(tag: MotifTag): String =
        "${tag.name}[${tag.squares.joinToString(",")}]" + (tag.move?.let { "@$it" } ?: "")

    companion object {
        private const val TAG = "sidecoach.analysis"
    }
}
