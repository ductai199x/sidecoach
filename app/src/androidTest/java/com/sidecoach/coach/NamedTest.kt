package com.sidecoach.coach

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.parseSquare
import com.sidecoach.chess.squareName
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.Line
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * What the player asks about, held to the desktop's answer.
 *
 * `coach/named.py` turns "what about Nf3?" into something the engine examines, and the
 * engine's answer into a line of the coach's object and then a sentence the coach says. The
 * engine here is the recorded one the generator used -- a deterministic stub whose lines are
 * in the fixture -- so the examination is compared exactly without a search.
 */
@RunWith(AndroidJUnit4::class)
class NamedTest {

    private val failures = ArrayList<String>()
    private var mismatches = 0

    private fun check(what: String, mine: Any?, theirs: Any?) {
        if (mine != theirs) {
            mismatches += 1
            if (failures.size < 8) failures.add("$what:\n  kotlin: $mine\n  python: $theirs")
        }
    }

    /**
     * Read under a locale that writes a decimal comma: the figures are Python's whatever the
     * phone's language ("0,53" reached the model and the voice: the parity audit cc-02).
     */
    @Test
    fun everyExaminationMatchesThePython() {
        val locale = java.util.Locale.getDefault()
        java.util.Locale.setDefault(java.util.Locale.GERMANY)
        try {
            examinations()
        } finally {
            java.util.Locale.setDefault(locale)
        }
    }

    private fun examinations() {
        val cases = fixture("chess/named.json").getJSONArray("cases")
        var asks = 0
        val began = System.currentTimeMillis()
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val board = Board(case.getString("fen"))
            val moved = Board(case.getString("moved"))
            val fenLabel = case.getString("fen")
            val us = board.turn
            val engine = recorded(case.getJSONObject("engine"))
            val list = case.getJSONArray("asks")
            for (k in 0 until list.length()) {
                val ask = list.getJSONObject(k)
                val words = ask.getString("text")
                asks += 1
                val probes = probesByWords(words, board)
                check("\"$words\" probes", probes.map { showProbe(it) }, probeStrings(ask.getJSONArray("probes")))
                check(
                    "\"$words\" pieces",
                    piecesByWords(words, board).map { showProbe(it) },
                    probeStrings(ask.getJSONArray("pieces")),
                )
                check(
                    "\"$words\" static",
                    // Only a move probe reaches examineStatic; a square probe is read by
                    // examinePiece, which is what analyseNamed does.
                    probes.filter { it.square == null }.map { examineStatic(it, board) },
                    strings(ask.getJSONArray("static")),
                )
                check(
                    "\"$words\" piece reads",
                    probes.filter { it.square != null }.map { examinePiece(it, board, us) },
                    strings(ask.getJSONArray("piece_reads")),
                )

                val before = engine.analyse(board, null)
                val best = before.firstOrNull()
                val levels = ask.getJSONObject("levels")
                for (levelKey in levels.keys()) {
                    val level = DisclosureLevel.of(levelKey.toInt())
                    val want = levels.getJSONObject(levelKey)
                    val examined = probes.map { p ->
                        if (p.square == null) {
                            examine(p, board, engine, best, us, level, 14)
                        } else {
                            examinePiece(p, board, us)
                        }
                    }
                    check("\"$words\" at $levelKey examined", examined, strings(want.getJSONArray("examined")))
                    check(
                        "\"$words\" at $levelKey spoken",
                        examined.map { spoken(it) },
                        strings(want.getJSONArray("spoken")),
                    )
                    check(
                        "\"$words\" at $levelKey frames",
                        disclosedFrames(board, examined).map { it.fen() },
                        strings(want.getJSONArray("frames")),
                    )
                    check(
                        "\"$words\" at $levelKey named",
                        analyseNamed(words, board, engine, us, level, depth = 14),
                        strings(want.getJSONArray("named")),
                    )
                    check(
                        "\"$words\" at $levelKey named static",
                        analyseNamed(words, board, null, us, level, static = true),
                        strings(want.getJSONArray("named_static")),
                    )
                    check(
                        "\"$words\" at $levelKey named none",
                        analyseNamed(words, board, null, us, level, static = false),
                        strings(want.getJSONArray("named_none")),
                    )
                    // The sheet's own lines, and the pieces read on the board as it stands (TG-04).
                    check(
                        "\"$words\" at $levelKey named known",
                        analyseNamed(words, board, engine, us, level, depth = 14, known = before),
                        strings(want.getJSONArray("named_known")),
                    )
                    check(
                        "\"$words\" at $levelKey named pieces",
                        analyseNamed(words, board, null, us, level, piecesBoard = moved),
                        strings(want.getJSONArray("named_pieces")),
                    )
                }
            }

            // Every square of the board as a piece or square probe: attack and defence by legal
            // moves, a controller pinned off the line marked so.
            val squares = case.optJSONArray("squares") ?: JSONArray()
            for (sq in 0 until squares.length()) {
                check(
                    "$fenLabel examinePiece ${squareName(sq)}",
                    examinePiece(Probe(label = squareName(sq), moves = emptyList(), square = sq), board, us),
                    squares.getString(sq),
                )
            }

            // The probes a planner would hand over, including the ones that go illegal
            // partway: what the words alone never reach.
            val planned = case.getJSONObject("planned")
            val probes = probeList(planned.getJSONArray("probes"), board)
            check(
                "planned probes",
                probes.map { showProbe(it) },
                probeStrings(planned.getJSONArray("probes")),
            )
            check(
                "planned static",
                probes.map { examineStatic(it, board) },
                strings(planned.getJSONArray("static")),
            )
            val best = engine.analyse(board, null).firstOrNull()
            val plannedLevels = planned.getJSONObject("levels")
            for (levelKey in plannedLevels.keys()) {
                val level = DisclosureLevel.of(levelKey.toInt())
                val want = plannedLevels.getJSONObject(levelKey)
                val examined = probes.map { examine(it, board, engine, best, us, level, 14) }
                check("planned at $levelKey", examined, strings(want.getJSONArray("examined")))
                check(
                    "planned spoken at $levelKey",
                    examined.map { spoken(it) },
                    strings(want.getJSONArray("spoken")),
                )
            }
        }
        val took = System.currentTimeMillis() - began
        Log.i(TAG, "${cases.length()} positions, $asks asks, $took ms")
        assertTrue("$mismatches mismatches\n" + failures.joinToString("\n"), failures.isEmpty())
    }

    /**
     * Every call the desktop's `tests/test_named.py` and `tests/test_inventory.py` make, replayed
     * with the desktop's answers (TG-04, TG-05): a pinned piece attacking and defending, a king in
     * check, a pinned controller, hanging and undefended, a move recited from the examined line,
     * the inventory's hanging pieces, pins, batteries, skewers and forks, a piece named with its
     * square. And the planner (TG-06): each reply read as Python reads it -- a null label is no
     * label, a null move "None", a reply Python cannot iterate raises -- and the request it is sent.
     */
    @Test
    fun theDesktopsNamedTestsAndThePlannerMatch() {
        val root = fixture("chess/named.json")
        val tests = root.getJSONObject("tests")
        assertTrue("the desktop's named tests ran", tests.getJSONArray("ran").length() >= 17)
        val calls = tests.getJSONArray("calls")
        for (i in 0 until calls.length()) {
            val call = calls.getJSONObject(i)
            val fn = call.getString("fn")
            val a = call.getJSONObject("args")
            val board = if (a.has("fen")) Board(a.getString("fen")) else Board()
            val engine = recorded(call.getJSONObject("engine"))
            val level = if (a.has("level")) DisclosureLevel.of(a.getInt("level")) else DisclosureLevel.LINE
            val out = call.opt("out")
            fun squares(pair: Pair<Int, Int>?) = pair?.let { listOf(squareName(it.first), squareName(it.second)) }
            val (mine, theirs) = when (fn) {
                "probes_by_words" -> probesByWords(a.getString("text"), board).map { showProbe(it) } to
                    probeStrings(out as JSONArray)
                "pieces_by_words" -> piecesByWords(a.getString("text"), board).map { showProbe(it) } to
                    probeStrings(out as JSONArray)
                "parse_probe" -> showProbe(parseProbe(a.getString("label"), strings(a.getJSONArray("moves")), board)) to
                    probeStrings(JSONArray().put(out)).single()
                "examine" -> examine(
                    probeFrom(a.getJSONObject("probe"), board), board, engine,
                    if (a.isNull("best")) null else lineFrom(a.getJSONObject("best")),
                    a.getBoolean("user_white"), level, a.getInt("depth"), known = lines(a.getJSONArray("known")),
                ) to out
                "examine_piece" -> examinePiece(probeFrom(a.getJSONObject("probe"), board), board, a.getBoolean("user_white")) to out
                "spoken" -> spoken(a.getString("fact")) to out
                "disclosed_frames" -> disclosedFrames(board, strings(a.getJSONArray("facts"))).map { it.fen() } to
                    strings(out as JSONArray)
                "analyse_named" -> analyseNamed(
                    a.getString("text"), board, if (a.getBoolean("engine")) engine else null,
                    a.getBoolean("user_white"), level, depth = a.getInt("depth"), static = a.getBoolean("static"),
                    piecesBoard = if (a.isNull("pieces_fen")) null else Board(a.getString("pieces_fen")),
                    known = lines(a.getJSONArray("known")),
                ) to strings(out as JSONArray)
                "check_draft" -> checkDraft(
                    a.getString("text"), board, strings(a.getJSONArray("after")).map { Board(it) },
                    if (a.isNull("us")) null else a.getBoolean("us"),
                ).map { "${it.text}|${it.reason}" } to (out as JSONArray).let { all ->
                    (0 until all.length()).map { all.getJSONObject(it).let { p -> "${p.getString("text")}|${p.getString("reason")}" } }
                }
                "inventory_text" -> inventoryText(board, a.getBoolean("white")) to out
                "pin_target" -> squares(pinTarget(board, parseSquare(a.getString("square")))) to
                    (if (out == JSONObject.NULL) null else strings(out as JSONArray))
                "skewer_target" -> squares(skewerTarget(board, parseSquare(a.getString("square")))) to
                    (if (out == JSONObject.NULL) null else strings(out as JSONArray))
                "batteries" -> batteries(board, a.getBoolean("white")).map { group -> group.map { squareName(it) } } to
                    (out as JSONArray).let { all -> (0 until all.length()).map { strings(all.getJSONArray(it)) } }
                "forks_in_place" -> forksInPlace(board, a.getBoolean("white"))
                    .map { (from, targets) -> listOf(squareName(from), targets.map { squareName(it) }) } to
                    (out as JSONArray).let { all ->
                        (0 until all.length()).map { k ->
                            all.getJSONArray(k).let { listOf(it.getString(0), strings(it.getJSONArray(1))) }
                        }
                    }
                else -> throw IllegalArgumentException("no replay for $fn")
            }
            check("${call.getString("test")} $fn ${a.toString().take(120)}", mine, theirs)
        }

        assertTrue("$mismatches mismatches\n" + failures.joinToString("\n"), failures.isEmpty())
    }

    private fun probeFrom(o: JSONObject, board: Board): Probe =
        if (o.isNull("square")) {
            parseProbe(o.getString("label"), strings(o.getJSONArray("moves")), board)
        } else {
            Probe(label = o.getString("label"), moves = emptyList(), square = parseSquare(o.getString("square")))
        }

    private fun lineFrom(o: JSONObject) = Line(
        moves = strings(o.getJSONArray("moves")),
        depth = o.getInt("depth"),
        expectedScore = o.getDouble("expected_score"),
        scoreCp = if (o.isNull("score_cp")) null else o.getInt("score_cp"),
    )

    private fun lines(array: JSONArray): List<Line> = (0 until array.length()).map { lineFrom(array.getJSONObject(it)) }

    /** The stub the generator used: the same lines for the same position, read from the file. */
    private fun recorded(seen: JSONObject): EngineLike = EngineLike { board, _ ->
        val entry = seen.optJSONArray(board.fen()) ?: JSONArray()
        (0 until entry.length()).map { i ->
            val o = entry.getJSONObject(i)
            Line(
                moves = o.getJSONArray("moves").let { a -> List(a.length()) { a.getString(it) } },
                depth = o.getInt("depth"),
                expectedScore = o.getDouble("expected_score"),
                scoreCp = if (o.isNull("score_cp")) null else o.getInt("score_cp"),
            )
        }
    }

    /** The probes rebuilt from the labels and move lists the generator recorded. */
    private fun probeList(array: JSONArray, board: Board): List<Probe> =
        (0 until array.length()).map { i ->
            val o = array.getJSONObject(i)
            val moves = o.getJSONArray("moves").let { a -> List(a.length()) { a.getString(it) } }
            parseProbe(o.getString("label"), moves, board)
        }

    private fun showProbe(p: Probe): String =
        "${p.label}|${p.moves.joinToString(" ")}|${p.parsed.joinToString(" ") { it.uci() }}|" +
            "${p.illegal ?: "-"}|${p.square?.let { squareName(it) } ?: "-"}"

    private fun probeStrings(array: JSONArray): List<String> = (0 until array.length()).map { i ->
        val o = array.getJSONObject(i)
        val moves = o.getJSONArray("moves").let { a -> List(a.length()) { a.getString(it) } }
        val parsed = o.getJSONArray("parsed").let { a -> List(a.length()) { a.getString(it) } }
        "${o.getString("label")}|${moves.joinToString(" ")}|${parsed.joinToString(" ")}|" +
            "${if (o.isNull("illegal")) "-" else o.getString("illegal")}|" +
            (if (o.isNull("square")) "-" else o.getString("square"))
    }

    private fun strings(array: JSONArray): List<String> =
        (0 until array.length()).map { array.getString(it) }

    companion object {
        private const val TAG = "sidecoach.coach"
        private const val ITALIAN_START = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    }
}
