package com.sidecoach.coach

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.analysis.Motifs
import com.sidecoach.analysis.humanExpectedScore
import com.sidecoach.analysis.phaseOf
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Analysis
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.Severity
import com.sidecoach.events.Trigger
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs

/**
 * The coach's primitives held to the desktop's, sentence for sentence and mark for mark.
 *
 * Everything under `coach/` that is a pure function of the board and the analysis: who "you"
 * is, the disclosure filter and the leak check, the fact bundle, what a move does, what
 * stands on the board, where a line leads, the word on a sound move, when the coach may
 * speak, and the marks that go with a message. The coach reads these out, so a port that
 * differs anywhere says something different.
 */
@RunWith(AndroidJUnit4::class)
class CoachPrimitivesTest {

    private fun fixtures(): JSONObject {
        val context = InstrumentationRegistry.getInstrumentation().context
        val text = context.assets.open("chess/coach.json").bufferedReader().use { it.readText() }
        return JSONObject(text)
    }

    private val failures = ArrayList<String>()

    private fun check(what: String, mine: Any?, theirs: Any?) {
        if (mine != theirs && failures.size < 8) {
            failures.add("$what:\n  kotlin: $mine\n  python: $theirs")
        }
    }

    @Test
    fun everyPrimitiveMatchesThePython() {
        val data = fixtures()
        val began = System.currentTimeMillis()

        // Who "you" is.
        val persons = data.getJSONArray("persons")
        for (i in 0 until persons.length()) {
            val o = persons.getJSONObject(i)
            check(
                "thirdPerson(${o.getString("text")}, ${o.getString("side")})",
                Person.thirdPerson(o.getString("text"), o.getString("side")),
                o.getString("out"),
            )
        }

        // The disclosure state machine.
        val unlocks = data.getJSONArray("unlocks")
        for (i in 0 until unlocks.length()) {
            val o = unlocks.getJSONObject(i)
            val state = Disclosure(DisclosureLevel.of(o.getInt("start")))
            val result = state.apply(Unlock.entries.first { it.slug == o.getString("unlock") })
            check(
                "unlock ${o.getString("unlock")} from ${o.getInt("start")}",
                (result?.value ?: -1) to state.level.value,
                (if (o.isNull("result")) -1 else o.getInt("result")) to o.getInt("level"),
            )
        }

        // The time-control bands and the trigger priority.
        val controls = data.getJSONArray("controls")
        for (i in 0 until controls.length()) {
            val o = controls.getJSONObject(i)
            val initial = if (o.isNull("initial")) null else o.getInt("initial")
            check(
                "timeControlOf($initial, ${o.getInt("increment")})",
                timeControlOf(initial, o.getInt("increment")).slug,
                o.getString("control"),
            )
        }
        val priorities = data.getJSONArray("priorities")
        for (i in 0 until priorities.length()) {
            val o = priorities.getJSONObject(i)
            val input = o.getJSONArray("in").let { a ->
                List(a.length()) { k -> Trigger.entries.first { it.slug == a.getString(k) } }
            }
            check(
                "inPriority",
                inPriority(input).map { it.slug },
                o.getJSONArray("out").let { a -> List(a.length()) { a.getString(it) } },
            )
        }

        checkPolicy(data.getJSONArray("policy"))

        // The positions.
        val cases = data.getJSONArray("cases")
        var deepCases = 0
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val board = Board(case.getString("fen"))
            val us = board.turn
            val analysis = analysisFor(
                board,
                case.getJSONArray("line").let { a -> List(a.length()) { a.getString(it) } },
                case.getBoolean("decisive"),
                case.getBoolean("only_move"),
            )
            val facts = factsFromAnalysis(board, analysis, lastMove = "e4", inBook = false, user = us)
            checkFacts(case.getString("fen"), facts, case.getJSONObject("facts"))

            check("${case.getString("fen")} phase", phaseOf(board, false), case.getString("phase"))
            check("material us", materialText(board, us), case.getString("material_us"))
            check("material them", materialText(board, !us), case.getString("material_them"))
            check("position text", positionText(board), case.getString("position_text"))
            check("ascii", asciiBoard(board), case.getString("ascii"))
            check("inventory lines", inventoryLines(board, us), strings(case.getJSONArray("inventory_lines")))
            check("inventory text", inventoryText(board, us), case.getString("inventory_text"))
            check(
                "line outcome",
                lineOutcome(board, analysis.lines[0].moves, us),
                case.getString("line_outcome"),
            )

            val effects = case.getJSONArray("effects")
            for (k in 0 until effects.length()) {
                val o = effects.getJSONObject(k)
                val move = Move.fromUci(o.getString("move"))
                check("describeMove ${o.getString("move")}", describeMove(board, move), o.getString("described"))
                check(
                    "effects ${o.getString("move")}",
                    describeEffects(board, move, hazards = true),
                    strings(o.getJSONArray("hazards")),
                )
                check(
                    "effects no hazards ${o.getString("move")}",
                    describeEffects(board, move, hazards = false),
                    strings(o.getJSONArray("no_hazards")),
                )
                check("effects text ${o.getString("move")}", effectsText(board, move), o.getString("text"))
            }

            val leaks = case.getJSONArray("leaks")
            if (leaks.length() > 0) deepCases += 1
            for (k in 0 until leaks.length()) {
                val o = leaks.getJSONObject(k)
                val said = if (o.isNull("said")) null else o.getString("said")
                check(
                    "leak \"${o.getString("text")}\" at ${o.getInt("level")}",
                    leakCheck(o.getString("text"), facts, DisclosureLevel.of(o.getInt("level")), said),
                    strings(o.getJSONArray("leaks")),
                )
            }

            val marks = case.getJSONArray("marks")
            for (k in 0 until marks.length()) {
                val o = marks.getJSONObject(k)
                val played = if (o.isNull("played")) null else o.getString("played")
                val mine = annotationsFor(
                    facts,
                    DisclosureLevel.of(o.getInt("level")),
                    board,
                    Trigger.entries.first { it.slug == o.getString("trigger") },
                    played,
                    if (o.isNull("severity")) null else o.getString("severity"),
                )
                check(
                    "marks at ${o.getInt("level")} ${o.getString("trigger")}",
                    mine.map { showMark(it) },
                    annotations(o.getJSONArray("marks")),
                )
            }

            val verdicts = case.getJSONArray("verdicts")
            for (k in 0 until verdicts.length()) {
                val o = verdicts.getJSONObject(k)
                val move = Move.fromUci(o.getString("move"))
                val label = if (o.isNull("label")) null else o.getString("label")
                val last = if (o.isNull("last")) null else o.getString("last")
                val san = board.san(move)
                check(
                    "openers ${o.getString("move")} $label",
                    openersFor(san, label, last),
                    strings(o.getJSONArray("choices")),
                )
                check(
                    "clause ${o.getString("move")}",
                    whatItDoes(board, move, analysis.tags),
                    if (o.isNull("clause")) null else o.getString("clause"),
                )
                check(
                    "verdict ${o.getString("move")} $label",
                    verdict(board, move, label, analysis.tags, last) { it[0] },
                    o.getString("sentence"),
                )
            }

            val mentions = case.getJSONArray("mentions")
            for (k in 0 until mentions.length()) {
                val o = mentions.getJSONObject(k)
                check(
                    "mentions \"${o.getString("text")}\"",
                    mentionsIn(o.getString("text"), board).map { showMark(it) },
                    annotations(o.getJSONArray("marks")),
                )
            }
        }
        val took = System.currentTimeMillis() - began
        Log.i(
            TAG,
            "${cases.length()} positions ($deepCases with the whole sweep), " +
                "${persons.length()} rewrites, ${took} ms",
        )
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    private fun checkPolicy(runs: JSONArray) {
        for (i in 0 until runs.length()) {
            val run = runs.getJSONObject(i)
            val control = TimeControl.entries.first { it.slug == run.getString("control") }
            val policy = InterjectionPolicy(
                PolicySettings(quiet = run.getBoolean("quiet")), control,
            )
            val steps = run.getJSONArray("steps")
            val seen = ArrayList<Map<String, Any?>>()

            fun note(what: String, trigger: Trigger?) {
                seen.add(
                    mapOf(
                        "step" to what,
                        "trigger" to trigger?.slug,
                        "reason" to policy.lastReason,
                        "threshold" to policy.stuckThreshold(),
                        "scramble" to policy.inScramble,
                    ),
                )
            }

            fun assessment(severity: Severity, label: Label?) = MoveAssessment(
                gameId = "g", ply = 1, uci = "e2e4", expectedLoss = 0.2,
                severity = severity, best = emptyList(), label = label,
            )

            policy.newPosition(2, 0.0, userToMove = true)
            note("new position", null)
            note("tick 10", policy.tick(10.0))
            note("tick 60", policy.tick(60.0))
            note("analysed quiet", policy.onPositionAnalysed(2, decisive = false))
            note("analysed decisive", policy.onPositionAnalysed(2, decisive = true))
            policy.said(2)
            note("tick after said", policy.tick(120.0))
            policy.newPosition(4, 100.0, userToMove = true)
            note("analysed decisive again", policy.onPositionAnalysed(4, decisive = true))
            policy.newPosition(12, 200.0, userToMove = true)
            note("analysed decisive later", policy.onPositionAnalysed(12, decisive = true))
            policy.setRemaining(20_000)
            note("tick in scramble", policy.tick(400.0))
            policy.setRemaining(600_000)
            note("tick with clock", policy.tick(500.0))
            policy.newPosition(14, 500.0, userToMove = false)
            note("opponent to move", policy.tick(600.0))
            policy.newPosition(16, 600.0, userToMove = true)
            policy.covered = true
            note("covered", policy.tick(700.0))
            policy.covered = false
            note("blunder", policy.onUserMoveAssessed(assessment(Severity.BLUNDER, Label.BLUNDER)))
            note("fine move", policy.onUserMoveAssessed(assessment(Severity.NONE, Label.BEST)))
            note("miss", policy.onUserMoveAssessed(assessment(Severity.NONE, Label.MISS)))

            for (k in 0 until steps.length()) {
                val want = steps.getJSONObject(k)
                val got = seen[k]
                val label = "${run.getString("control")} quiet=${run.getBoolean("quiet")} " +
                    "step ${want.getString("step")}"
                check("$label trigger", got["trigger"], if (want.isNull("trigger")) null else want.getString("trigger"))
                check("$label reason", got["reason"], if (want.isNull("reason")) null else want.getString("reason"))
                val threshold = got["threshold"] as Double
                if (abs(threshold - want.getDouble("threshold")) > 1e-9) {
                    check("$label threshold", threshold, want.getDouble("threshold"))
                }
                check("$label scramble", got["scramble"], want.getBoolean("scramble"))
            }
        }
    }

    private fun checkFacts(fen: String, mine: PositionFacts, theirs: JSONObject) {
        val map = mine.asMap()
        for ((key, value) in map) {
            val want: Any? = when {
                theirs.isNull(key) -> null
                value is List<*> -> strings(theirs.getJSONArray(key))
                value is Boolean -> theirs.getBoolean(key)
                else -> theirs.getString(key)
            }
            check("$fen facts.$key", value, want)
        }
    }

    /** An analysis of the shape the generator built, so both sides start from the same one. */
    private fun analysisFor(
        board: Board,
        lineUcis: List<String>,
        decisive: Boolean,
        only: Boolean,
    ): Analysis {
        val moves = ArrayList<String>()
        val work = board.copy(withStack = false)
        for (uci in lineUcis) {
            val move = Move.fromUci(uci)
            if (!work.isLegal(move)) break
            moves.add(uci)
            work.push(move)
        }
        val lines = ArrayList<Line>()
        lines.add(Line(moves, 18, humanExpectedScore(120, null), 120))
        val others = board.legalMoves().filter { moves.isEmpty() || it.uci() != moves[0] }
        if (others.isNotEmpty()) {
            lines.add(Line(listOf(others[0].uci()), 18, humanExpectedScore(115, null), 115))
        }
        val threats = ArrayList<String>()
        if (!board.isCheck()) {
            val passed = nullMoved(board)
            passed.legalMoves().firstOrNull()?.let { threats.add(it.uci()) }
        }
        val pv = moves.map { Move.fromUci(it) }
        return Analysis(
            gameId = "g",
            fen = board.fen(),
            ply = 0,
            lines = lines,
            decisive = decisive,
            onlyMove = only,
            threats = threats,
            tags = Motifs.tagsForPosition(board, pv),
        )
    }

    private fun strings(array: JSONArray): List<String> =
        (0 until array.length()).map { array.getString(it) }

    private fun annotations(array: JSONArray): List<String> =
        (0 until array.length()).map { i ->
            val o = array.getJSONObject(i)
            val squares = o.getJSONArray("squares").let { a -> List(a.length()) { a.getString(it) } }
            mark(
                o.getString("kind"), squares, o.getString("color"), o.getDouble("opacity"),
                if (o.isNull("label")) null else o.getString("label"),
            )
        }

    private fun showMark(a: com.sidecoach.events.Annotation): String =
        mark(a.kind, a.squares, a.color, a.opacity, a.label)

    private fun mark(
        kind: String,
        squares: List<String>,
        color: String,
        opacity: Double,
        label: String?,
    ): String = "$kind ${squares.joinToString(",")} $color ${"%.4f".format(opacity)} ${label ?: "-"}"

    companion object {
        private const val TAG = "sidecoach.coach"
    }
}
