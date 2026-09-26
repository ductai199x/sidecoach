package com.sidecoach.coach

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.squareName
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.Line
import org.json.JSONArray
import org.json.JSONObject
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

    @Test
    fun everyExaminationMatchesThePython() {
        val context = InstrumentationRegistry.getInstrumentation().context
        val text = context.assets.open("chess/named.json").bufferedReader().use { it.readText() }
        val cases = JSONObject(text).getJSONArray("cases")
        var asks = 0
        val began = System.currentTimeMillis()
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val board = Board(case.getString("fen"))
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

    /** `_Engine`: scripted lines keyed by the position's FEN board part. */
    private fun scripted(lines: Map<String, List<Line>>) = EngineLike { board, _ -> lines[board.boardFen()] ?: emptyList() }

    /**
     * Searched again after it is played, the best move came back a hair off its own score:
     * "12. Nc3: ... a hair behind the best (the best is Nc3)" (the simulated students; desktop
     * 797094c).
     */
    @Test
    fun theEnginesOwnBestMoveIsNeverBehindTheBest() {
        val board = Board(ITALIAN_START)
        val after = board.copy()
        after.push(after.parseSan("Bb5")!!)
        val engine = scripted(mapOf(after.boardFen() to listOf(Line(depth = 14, moves = listOf("a7a6"), expectedScore = 0.46))))
        val best = Line(depth = 14, moves = listOf("f1b5"), expectedScore = 0.55)
        val fact = examine(
            parseProbe("your Bb5", listOf("Bb5"), board), board, engine, best,
            userColour = WHITE, level = DisclosureLevel.LINE,
        )
        assertTrue(fact, "as good as the engine's best" in fact && "behind" !in fact)
    }

    /**
     * "if they take on b4" is the player passing, no move of theirs: it came out "after pass axb4
     * you are losing, a serious step down from the best (the best is Bxc3)" (desktop e7b5915).
     */
    @Test
    fun theirMoveFirstIsNotMeasuredAgainstTheBestAndIsSaidAsTheirMove() {
        val board = Board(ITALIAN_START)
        val after = board.copy()
        after.push(Move.NULL)
        after.push(after.parseSan("Nd4")!!)
        val engine = scripted(mapOf(after.boardFen() to listOf(Line(depth = 14, moves = listOf("d2d3"), expectedScore = 0.40))))
        val best = Line(depth = 14, moves = listOf("f1b5"), expectedScore = 0.55)
        val fact = examine(
            parseProbe("if they play Nd4", listOf("pass", "Nd4"), board), board, engine, best,
            userColour = WHITE, level = DisclosureLevel.LINE,
        )
        assertTrue(fact, "best" !in fact)
        val said = spoken(fact)
        assertTrue(said, said.startsWith("if they play Nd4: if they got to play Nd4 now, "))
        assertTrue(said, "pass" !in said)
    }

    /**
     * A piece or square is read on the board as it stands, whatever position the question is
     * about (desktop f85c2bf): the position before 3.Bb5 asked about, the board after it read.
     */
    @Test
    fun aPieceIsReadOnTheBoardGiven() {
        val before = Board(ITALIAN_START)
        val now = before.copy()
        now.push(now.parseSan("Bb5")!!)
        val there = analyseNamed("is my bishop on b5 safe?", before, null, WHITE, DisclosureLevel.LINE, piecesBoard = now)
        assertTrue(there.toString(), there.size == 1 && "the player's bishop on b5 stands there" in there[0])
        val past = analyseNamed("is my bishop on b5 safe?", before, null, WHITE, DisclosureLevel.LINE)
        assertTrue(past.toString(), past.size == 1 && "b5 is empty" in past[0])
    }

    /**
     * "they would answer 9... Bxg2 10. Kxg2 Qd7 11. d5 Na6 12. Nc3 O-O" came before every answer
     * the AI then gave in words: aloud, the reply's first two moves (desktop 6d9ba6f).
     */
    @Test
    fun aLineSaidAloudIsItsFirstReply() {
        val fact = "Engine on your Qc3, which the player asked about: after 6... Qc3 the expected score " +
            "is 0.45 from the player's side, 0.05 behind the engine's best; the opponent's reply " +
            "would be 7. Nf3 Nd7 8. Rc1 Qa5 9. d4; what Qc3 does: attacks the pawn on d3."
        val said = spoken(fact)
        assertTrue(said, "they would answer 7. Nf3 Nd7;" in said && "Rc1" !in said)
    }

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
