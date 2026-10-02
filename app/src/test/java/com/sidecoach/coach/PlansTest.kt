package com.sidecoach.coach

import com.sidecoach.TestAssets
import com.sidecoach.TestLog
import com.sidecoach.analysis.pawnStormSide
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.parseSquare
import com.sidecoach.chess.squareName
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The plan in the best line and the threat of the best move ([Plans]) held to the desktop's
 * `coach/plans.py`, word for word: the desktop's own tests, the engine's lines of the master
 * games and of Lichess puzzles, and every quiet move of some master positions asked for its
 * threat (`tools/make_plans_fixtures.py`). Then `tests/test_plans.py`'s cases written out here,
 * and the draw that saves nothing (`idea._drawn` with the next-best move's score).
 */
class PlansTest {

    private fun fixtures(): JSONObject {
        val text = TestAssets.text("chess/plans.json")
        return JSONObject(text)
    }

    private val failures = ArrayList<String>()
    private var mismatches = 0

    private fun check(what: String, mine: Any?, theirs: Any?) {
        if (mine != theirs) {
            mismatches += 1
            if (failures.size < 8) failures.add("$what:\n  kotlin: $mine\n  python: $theirs")
        }
    }

    private fun moves(array: JSONArray): List<Move> = (0 until array.length()).map { Move.fromUci(array.getString(it)) }

    private fun strings(array: JSONArray): List<String> = (0 until array.length()).map { array.getString(it) }

    private fun opt(o: JSONObject, key: String): String? = if (o.isNull(key)) null else o.getString(key)

    private fun planMap(plan: LinePlan?, board: Board): Map<String, Any?>? = plan?.let {
        linkedMapOf(
            "kind" to it.kind,
            "square" to it.square?.let { sq -> squareName(sq) },
            "where" to it.where,
            "pieces" to it.pieces.map { sq -> squareName(sq) },
            "sans" to it.sans,
            "detail" to it.detail,
            "name" to Plans.name(it),
            "mechanism" to Plans.mechanism(it),
            "target" to Plans.target(it),
            "pieces_text" to Plans.pieces(it, board),
            "why" to Plans.why(it, board),
            "question" to Plans.PLAN_QUESTION.getValue(it.kind),
            "nudge" to Plans.PLAN_NUDGE.getValue(it.kind),
        )
    }

    private fun planWanted(o: JSONObject?): Map<String, Any?>? = o?.let {
        linkedMapOf(
            "kind" to it.getString("kind"),
            "square" to opt(it, "square"),
            "where" to opt(it, "where"),
            "pieces" to strings(it.getJSONArray("pieces")),
            "sans" to strings(it.getJSONArray("sans")),
            "detail" to opt(it, "detail"),
            "name" to it.getString("name"),
            "mechanism" to it.getString("mechanism"),
            "target" to opt(it, "target"),
            "pieces_text" to opt(it, "pieces_text"),
            "why" to it.getString("why"),
            "question" to it.getString("question"),
            "nudge" to it.getString("nudge"),
        )
    }

    private fun threatMap(t: MoveThreat?, board: Board, move: Move): Map<String, Any?>? = t?.let {
        linkedMapOf(
            "kind" to it.kind,
            "square" to squareName(it.square),
            "follow" to it.follow.uci(),
            "gain" to it.gain,
            "why" to it.why,
            "mechanism" to Plans.threatMechanism(it),
            "target" to Plans.threatTarget(it, board),
            "why_text" to Plans.threatWhy(it, board, move),
            "goal" to Plans.THREAT_GOAL.getValue(it.kind),
            "goal_won" to Plans.THREAT_GOAL_WON.getValue(it.kind),
            "name" to Plans.THREAT_NAME.getValue(it.kind),
            "question" to Plans.THREAT_QUESTION.getValue(it.kind),
            "nudge" to Plans.THREAT_NUDGE.getValue(it.kind),
        )
    }

    private fun threatWanted(o: JSONObject?): Map<String, Any?>? = o?.let {
        linkedMapOf(
            "kind" to it.getString("kind"),
            "square" to it.getString("square"),
            "follow" to it.getString("follow"),
            "gain" to it.getInt("gain"),
            "why" to it.getString("why"),
            "mechanism" to it.getString("mechanism"),
            "target" to opt(it, "target"),
            "why_text" to it.getString("why_text"),
            "goal" to it.getString("goal"),
            "goal_won" to it.getString("goal_won"),
            "name" to it.getString("name"),
            "question" to it.getString("question"),
            "nudge" to it.getString("nudge"),
        )
    }

    @Test
    fun everyPlanAndThreatMatchesThePython() {
        val data = fixtures()
        val began = System.currentTimeMillis()
        val kinds = HashMap<String, Int>()

        // The desktop's own tests, and the engine's lines: the plan of the line and the threat of
        // its first move, read with the line's score.
        val cases = ArrayList<JSONObject>()
        val tests = data.getJSONArray("tests")
        for (i in 0 until tests.length()) cases.add(tests.getJSONObject(i))
        val lines = data.getJSONArray("cases")
        for (i in 0 until lines.length()) cases.add(lines.getJSONObject(i))
        for (case in cases) {
            val board = Board(case.getString("fen"))
            val pv = moves(case.getJSONArray("pv"))
            val score = if (case.isNull("score")) null else case.getDouble("score")
            val where = "${case.getString("source")} ${case.getString("fen")}"
            if (case.has("move")) {
                val move = Move.fromUci(case.getString("move"))
                val t = Plans.threat(board, move, pv, score)
                check("$where threat of ${case.getString("move")}", threatMap(t, board, move), threatWanted(case.optJSONObject("threat")))
                continue
            }
            val plan = Plans.read(board, pv, score)
            check("$where plan", planMap(plan, board), planWanted(case.optJSONObject("plan")))
            plan?.let { kinds["plan:${it.kind}"] = (kinds["plan:${it.kind}"] ?: 0) + 1 }
            val t = if (pv.isNotEmpty() && board.isLegal(pv[0])) Plans.threat(board, pv[0], pv, score) else null
            check("$where threat", pv.firstOrNull()?.let { threatMap(t, board, it) }, threatWanted(case.optJSONObject("threat")))
            t?.let { kinds["threat:${it.kind}"] = (kinds["threat:${it.kind}"] ?: 0) + 1 }
        }

        // Every legal quiet move of some master positions, asked for its threat with no line.
        val positions = data.getJSONArray("moves")
        var asked = 0
        for (i in 0 until positions.length()) {
            val o = positions.getJSONObject(i)
            val board = Board(o.getString("fen"))
            val list = o.getJSONArray("moves")
            for (k in 0 until list.length()) {
                val m = list.getJSONObject(k)
                val move = Move.fromUci(m.getString("move"))
                val t = Plans.threat(board, move)
                check("${o.getString("fen")} threat of ${m.getString("move")}", threatMap(t, board, move), threatWanted(m.optJSONObject("threat")))
                asked += 1
            }
        }
        val took = System.currentTimeMillis() - began
        TestLog.i(TAG, "${cases.size} lines, $asked move threats, $took ms; ${kinds.toSortedMap()}")
        assertTrue("$mismatches mismatches\n" + failures.joinToString("\n"), failures.isEmpty())
    }

    private fun line(fen: String, ucis: String): Pair<Board, List<Move>> =
        Board(fen) to ucis.split(" ").map { Move.fromUci(it) }

    /** `tests/test_plans.py`, case for case. */
    @Test
    fun theDesktopsPlanTests() {
        // Rooks brought to an open file are doubled.
        run {
            val (board, pv) = line("6k1/p4ppp/8/8/8/8/5PPP/R4RK1 w - - 0 1", "a1d1 a7a6 d1d2 a6a5 f1d1 a5a4")
            val plan = Plans.read(board, pv)!!
            assertEquals("double", plan.kind)
            assertEquals("d", plan.where)
            assertEquals("open", plan.detail)
            assertEquals("your rook on f1 and your rook on a1", Plans.pieces(plan, board))
            assertEquals("Rad1 and Rfd1 double your rooks on the open d-file", Plans.why(plan, board))
        }
        // A knight taken in two moves to an outpost.
        run {
            val (board, pv) = line("4k3/8/3p4/8/4P3/8/8/1N2K3 w - - 0 1", "b1c3 e8f7 c3d5 f7e6")
            val plan = Plans.read(board, pv)!!
            assertEquals("outpost", plan.kind)
            assertEquals(parseSquare("d5"), plan.square)
            assertEquals("the d5 square, an outpost for a knight", Plans.target(plan))
        }
        // A knight that goes out and back is no route.
        run {
            val (board, pv) = line("4k3/8/3p4/3N4/4P3/8/8/4K3 w - - 0 1", "d5c3 e8f7 c3d5 f7e6")
            assertNull(Plans.read(board, pv))
        }
        // A passed pawn pushed twice.
        run {
            val (board, pv) = line("4k3/7p/8/8/P7/8/8/4K3 w - - 0 1", "a4a5 h7h6 a5a6 h6h5")
            val plan = Plans.read(board, pv)!!
            assertEquals("passer", plan.kind)
            assertEquals(parseSquare("a4"), plan.square)
        }
        // Pawns rolled at a king on the other wing are a storm.
        run {
            val (board, pv) = line("6k1/p4ppp/8/8/8/8/6PP/2K5 w - - 0 1", "g2g4 a7a6 h2h4 a6a5 g4g5 a5a4")
            val plan = Plans.read(board, pv)!!
            assertEquals("storm", plan.kind)
            assertEquals("kingside", plan.where)
        }
        // Pawns in front of your own king are no storm: both kings behind the kingside pawns.
        run {
            val board = Board("6k1/p4ppp/8/8/8/8/5PPP/6K1 w - - 0 1")
            assertNull(pawnStormSide(board, Move.fromUci("h2h4"), WHITE))
            val (b, pv) = line(board.fen(), "g2g4 a7a6 h2h4 a6a5")
            assertNull(Plans.read(b, pv))
        }
        // Queenside pawns away from both kings gain space.
        run {
            val (board, pv) = line("6k1/p4ppp/8/8/8/8/PP3PPP/6K1 w - - 0 1", "b2b4 g8f8 a2a4 f8e7")
            val plan = Plans.read(board, pv)!!
            assertEquals("wing", plan.kind)
            assertEquals("queenside", plan.where)
        }
        // A break prepared by the best move.
        run {
            val (board, pv) = line("4k3/8/8/4p3/8/8/3P4/4K1N1 w - - 0 1", "g1f3 e8e7 d2d4 e7e6")
            val plan = Plans.read(board, pv)!!
            assertEquals("break", plan.kind)
            assertEquals(parseSquare("d4"), plan.square)
            assertEquals("Nf3 brings a piece to bear on d4, then d4 breaks open the centre", Plans.why(plan, board))
        }
        // Queen and rook lifted to their king are an attack; not for a side the engine calls worse.
        run {
            val (board, pv) = line("6k1/p4ppp/8/8/8/8/5PPP/3QR1K1 w - - 0 1", "d1d3 a7a6 e1e3 a6a5 e3h3 a5a4")
            val plan = Plans.read(board, pv)!!
            assertEquals("attack", plan.kind)
            assertEquals(parseSquare("g8"), plan.square)
            assertEquals(setOf(parseSquare("d1"), parseSquare("e1")), plan.pieces.toSet())
            // The lift and the swing both belong to it.
            assertEquals("Qd3, Re3 and Rh3 bring your queen and rook against their king", Plans.why(plan, board))
            assertNull(Plans.read(board, pv, score = 0.3))
        }
        // Captures make no plan: pawns traded on the queenside are an exchange.
        run {
            val (board, pv) = line("6k1/p4ppp/1p6/P7/8/8/1P3PPP/6K1 w - - 0 1", "a5b6 a7b6 b2b4 g8f8")
            assertNull(Plans.read(board, pv))
        }
        // A move that attacks a loose piece is a threat.
        run {
            val board = Board("4k3/8/8/1b6/8/8/8/1N2K3 w - - 0 1")
            val move = Move.fromUci("b1c3")
            val t = Plans.threat(board, move)!!
            assertEquals("win", t.kind)
            assertEquals(parseSquare("b5"), t.square)
            assertEquals("their bishop on b5", Plans.threatTarget(t, board))
            assertEquals("Nc3 goes after their bishop on b5: if they let it be, Nxb5 wins it", Plans.threatWhy(t, board, move))
        }
        // A move that gives its piece away is no threat: an offer.
        run {
            val board = Board("4k3/8/8/1b6/1p6/8/8/1N2K3 w - - 0 1")
            assertNull(Plans.threat(board, Move.fromUci("b1c3")))
        }
        // A quiet move that sets up mate threatens it.
        run {
            val board = Board("6k1/5ppp/8/8/8/8/1R3PPP/6K1 w - - 0 1")
            val move = Move.fromUci("b2e2")
            val t = Plans.threat(board, move)!!
            assertEquals("mate", t.kind)
            assertEquals(parseSquare("g8"), t.square)
            assertEquals("Re2 threatens Re8 mate", Plans.threatWhy(t, board, move))
        }
        // Like pieces face to face are a trade when the line makes it, and not when it does not.
        run {
            val board = Board("7k/8/8/8/8/8/RK6/7r w - - 0 1")
            val move = Move.fromUci("a2a1")
            val traded = listOf(move, Move.fromUci("h1a1"), Move.fromUci("b2a1"))
            val t = Plans.threat(board, move, traded)!!
            assertEquals("trade", t.kind)
            assertEquals(parseSquare("h1"), t.square)
            assertEquals("Ra1 puts your rook face to face with their rook on h1, to trade it off", Plans.threatWhy(t, board, move))
            val elsewhere = listOf(move, Move.fromUci("h8g8"), Move.fromUci("b2c2"))
            assertNull(Plans.threat(board, move, elsewhere))
        }
    }

    /** `test_a_level_line_that_repeats_is_a_draw_only_when_it_saves_the_game`. */
    @Test
    fun aLevelLineThatRepeatsIsADrawOnlyWhenItSavesTheGame() {
        val (board, pv) = line("6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1", "g1h1 g8h8 h1g1 h8g8")
        val boards = arrayListOf(board.copy(withStack = false))
        for (move in pv) {
            val nxt = boards.last().copy(withStack = false)
            nxt.push(move)
            boards.add(nxt)
        }
        assertEquals("repetition", IdeaReader.drawn(boards, pv, 0.5))
        assertEquals("repetition", IdeaReader.drawn(boards, pv, 0.5, second = 0.2))
        // Every other move holds as well: the engine marking time, not a rescue.
        assertNull(IdeaReader.drawn(boards, pv, 0.5, second = 0.48))
    }

    companion object {
        private const val TAG = "sidecoach.coach"
    }
}
