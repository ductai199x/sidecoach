package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.chess.Board
import com.sidecoach.events.Line
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The engine at the coach model's call, held to the desktop's answers (`coach/engine_tools.py`,
 * desktop c80b64b and ce06c4e; `tools/make_engine_tools_fixtures.py`). Both sides use an engine
 * that plays the first legal moves by UCI, so every answer is compared exactly.
 */
@RunWith(AndroidJUnit4::class)
class EngineToolsTest {

    private val failures = ArrayList<String>()
    private var mismatches = 0

    private fun check(what: String, mine: Any?, theirs: Any?) {
        if (mine != theirs) {
            mismatches += 1
            if (failures.size < 8) failures.add("$what:\n  kotlin: $mine\n  python: $theirs")
        }
    }

    /** `make_engine_tools_fixtures.FirstLegal`. */
    private val firstLegal = EngineLike { board, depth ->
        val replay = board.copy(withStack = false)
        val moves = ArrayList<String>()
        repeat(4) {
            val legal = replay.legalMoves().sortedBy { it.uci() }
            if (legal.isEmpty()) return@repeat
            moves.add(legal[0].uci())
            replay.push(legal[0])
        }
        if (moves.isEmpty()) {
            emptyList()
        } else {
            val score = 0.3 + (board.legalMoves().size % 7) / 10.0
            listOf(
                Line(moves = moves, depth = depth ?: 14, expectedScore = score),
                Line(moves = moves.take(2), depth = depth ?: 14, expectedScore = score - 0.05),
            )
        }
    }

    @Test
    fun everyToolAnswerMatchesThePython() {
        val root = fixture("chess/engine_tools.json")
        assertEquals(root.getString("tool_words"), TOOL_WORDS)
        assertEquals(root.getInt("tool_rounds"), TOOL_ROUNDS)
        // The tools as the model is told of them: names, descriptions, parameters.
        assertEquals(pyJsonLoads(root.getJSONArray("specs").toString()), pyJsonLoads(engineToolSpecs().toString()))
        val cases = root.getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val before = if (case.isNull("before")) null else Board(case.getString("before"))
            // The sheet's lines for each start: a move on one is read from it (desktop ce06c4e).
            val tools = EngineTools(
                Board(case.getString("now")), before, firstLegal, case.getBoolean("player_white"),
                knownNow = lines(case.getJSONArray("known_now")),
                knownBefore = lines(case.getJSONArray("known_before")),
            )
            val calls = case.getJSONArray("calls")
            for (k in 0 until calls.length()) {
                val call = calls.getJSONObject(k)
                val arguments = call.getJSONObject("arguments")
                check(
                    "case $i ${call.getString("name")} $arguments",
                    tools.run(call.getString("name"), JSONObject(arguments.toString())),
                    call.getString("answer"),
                )
            }
            val log = case.getJSONArray("log")
            check("case $i log", tools.log, (0 until log.length()).map { log.getString(it) })
            val frames = case.getJSONArray("frames")
            check("case $i frames", tools.frames.map { it.fen() }, (0 until frames.length()).map { frames.getString(it) })
        }
        assertTrue("$mismatches mismatches\n" + failures.joinToString("\n"), failures.isEmpty())
    }

    private fun lines(array: JSONArray): List<Line> = (0 until array.length()).map { i ->
        val o = array.getJSONObject(i)
        Line(
            moves = o.getJSONArray("moves").let { a -> List(a.length()) { a.getString(it) } },
            depth = o.getInt("depth"),
            expectedScore = o.getDouble("expected_score"),
        )
    }
}
