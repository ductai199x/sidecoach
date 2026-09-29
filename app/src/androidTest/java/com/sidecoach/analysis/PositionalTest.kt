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

    /**
     * The rescue rows of `test_the_best_move_s_idea` (desktop 3d9568d, one table since
     * ae6c91d): a piece that could be taken is in danger only when taking it is their threat
     * (the engine's best move for them, were it theirs): "a piece in danger comes first" was
     * said of a rook check that led to mate, and of moves whose threat was elsewhere (the lesson
     * audit, 2026-09-27). A pawn that steps on from an attack is a push, not a rescue. With the
     * threat, a piece that can be won with a safe square to go to is rescued.
     */
    @Test
    fun aRescueIsOfThePieceTheirThreatGoesAfter() {
        fun kind(board: Board, uci: String, threat: String?, phase: String = "middlegame"): String? =
            bestMoveIdea(board, Move.fromUci(uci), board.turn, phase = phase, threat = threat?.let { Move.fromUci(it) })?.kind
        val bishop = Board("rnbqkbnr/1ppp1ppp/p7/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 3")
        assertTrue(kind(bishop, "b5a4", "a6b5", phase = "opening") == "rescue")
        assertTrue(kind(bishop, "b5a4", null, phase = "opening") != "rescue")
        assertTrue("their threat is not the bishop", kind(bishop, "b5a4", "d8h4", phase = "opening") != "rescue")
        // The e4 pawn attacked by the d5 pawn steps on to e5: no rescue.
        val pawn = Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
        assertTrue(kind(pawn, "e4e5", "d5e4") != "rescue")
        // A knight a pawn attacks: Nb6, onto a square a pawn takes, is no rescue; Ne3 is.
        val knight = Board("4k3/p7/8/3p4/2N5/8/8/4K3 w - - 0 1")
        assertTrue(kind(knight, "c4b6", "d5c4") != "rescue")
        assertTrue(kind(knight, "c4e3", "d5c4") == "rescue")
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
