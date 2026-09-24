package com.sidecoach.analysis

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.squareName
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The positional layer held to the desktop's, sentence for sentence.
 *
 * This is the half of the coach's knowledge that is not tactics -- the open file, the passed
 * pawn, the outpost, the Lucena and the Philidor, and what the engine's best move is *for*.
 * The coach reads these sentences out word for word, so the port has to produce them word for
 * word, with the same evidence squares and in the same order.
 *
 * The positions are real middlegames and endgames from the Lichess sample, random games for
 * the opening ideas, and a handful of built endings for the textbook detectors. The move
 * ideas are taken over every legal move of every position.
 */
@RunWith(AndroidJUnit4::class)
class PositionalTest {

    @Test
    fun everyFactMatchesThePython() {
        val context = InstrumentationRegistry.getInstrumentation().context
        val text = context.assets.open("chess/positional.json").bufferedReader().use { it.readText() }
        val data = JSONObject(text).getJSONArray("cases")
        val failures = ArrayList<String>()
        val kinds = HashMap<String, Int>()
        var moves = 0
        val began = System.currentTimeMillis()
        for (i in 0 until data.length()) {
            val case = data.getJSONObject(i)
            val board = Board(case.getString("fen"))
            val phase = case.getString("phase")
            val us = board.turn
            val them = !us

            val worst = worstPieceSquare(board, us)
            val wantWorst = if (case.isNull("worst")) null else case.getString("worst")
            if ((worst?.let { squareName(it) }) != wantWorst) {
                add(failures, "${case.getString("fen")}: worst piece ${worst?.let { squareName(it) }}, python $wantWorst")
            }

            compare(case, "features_us", features(board, us, phase), failures, kinds)
            compare(case, "features_them", features(board, them, phase), failures, kinds)
            compare(case, "top_us", topFeatures(board, us, phase), failures, kinds)
            compare(case, "top_them", topFeatures(board, them, phase), failures, kinds)

            val ideas = case.getJSONArray("ideas")
            for (k in 0 until ideas.length()) {
                val o = ideas.getJSONObject(k)
                val mine = bestMoveIdea(board, Move.fromUci(o.getString("move")), us, phase, worst, null)
                checkIdea(case, o, mine, failures, kinds)
                moves += 1
            }
            if (!case.isNull("threat")) {
                val threat = Move.fromUci(case.getString("threat"))
                val threatIdeas = case.getJSONArray("threat_ideas")
                for (k in 0 until threatIdeas.length()) {
                    val o = threatIdeas.getJSONObject(k)
                    val mine = bestMoveIdea(
                        board, Move.fromUci(o.getString("move")), us, phase, worst, threat,
                    )
                    checkIdea(case, o, mine, failures, kinds)
                    moves += 1
                }
            }
            // A threat that wins one of our pieces, tried against every legal move: where a
            // guard that still loses the piece stops nothing.
            if (case.has("capture_threat") && !case.isNull("capture_threat")) {
                val threat = Move.fromUci(case.getString("capture_threat"))
                val captureIdeas = case.getJSONArray("capture_threat_ideas")
                for (k in 0 until captureIdeas.length()) {
                    val o = captureIdeas.getJSONObject(k)
                    val mine = bestMoveIdea(
                        board, Move.fromUci(o.getString("move")), us, phase, worst, threat,
                    )
                    checkIdea(case, o, mine, failures, kinds)
                    moves += 1
                }
            }
        }
        val took = System.currentTimeMillis() - began
        Log.i(
            TAG,
            "${data.length()} positions, $moves move ideas, $took ms; " +
                kinds.entries.sortedBy { it.key }.joinToString(", ") { "${it.key}x${it.value}" },
        )
        assertTrue(failures.take(8).joinToString("\n"), failures.isEmpty())
    }

    private fun compare(
        case: JSONObject,
        key: String,
        mine: List<PositionalFact>,
        failures: MutableList<String>,
        kinds: MutableMap<String, Int>,
    ) {
        val wanted = facts(case.getJSONArray(key))
        for (fact in mine) kinds[fact.kind] = (kinds[fact.kind] ?: 0) + 1
        if (mine == wanted) return
        add(
            failures,
            "${case.getString("fen")} $key:\n  kotlin: ${mine.joinToString(" | ") { show(it) }}" +
                "\n  python: ${wanted.joinToString(" | ") { show(it) }}",
        )
    }

    private fun checkIdea(
        case: JSONObject,
        o: JSONObject,
        mine: PositionalFact?,
        failures: MutableList<String>,
        kinds: MutableMap<String, Int>,
    ) {
        val wanted = if (o.isNull("idea")) null else fact(o.getJSONObject("idea"))
        if (mine != null) kinds[mine.kind] = (kinds[mine.kind] ?: 0) + 1
        if (mine == wanted) return
        add(
            failures,
            "${case.getString("fen")} ${o.getString("move")}:\n  kotlin: ${mine?.let { show(it) }}" +
                "\n  python: ${wanted?.let { show(it) }}",
        )
    }

    private fun facts(array: JSONArray): List<PositionalFact> =
        (0 until array.length()).map { fact(array.getJSONObject(it)) }

    private fun fact(o: JSONObject): PositionalFact = PositionalFact(
        o.getString("kind"),
        o.getString("text"),
        o.getJSONArray("squares").let { a -> List(a.length()) { a.getString(it) } },
    )

    private fun show(fact: PositionalFact): String =
        "${fact.kind}[${fact.squares.joinToString(",")}] \"${fact.text}\""

    private fun add(failures: MutableList<String>, line: String) {
        if (failures.size < 8) failures.add(line)
    }

    companion object {
        private const val TAG = "sidecoach.analysis"
    }
}
