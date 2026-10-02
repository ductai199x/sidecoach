package com.sidecoach.analysis

import com.sidecoach.TestAssets
import com.sidecoach.TestLog
import com.sidecoach.chess.Board
import com.sidecoach.events.Line
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * The scoring held to the desktop's, number for number.
 *
 * Everything the coach says about a move comes through here: what it cost in expected score,
 * what that loss is called, how accurate it was, and whether the position was a moment where
 * the choice mattered. The thresholds were fitted to chess.com's own review over the user's
 * games, so a port that rounds differently would rename his mistakes. These functions are
 * pure, so unlike a search they can be compared exactly.
 */
class ScoreTest {

    private fun fixtures(): JSONObject {
        val text = TestAssets.text("chess/score.json")
        return JSONObject(text)
    }

    @Test
    fun theCurvesMatchThePython() {
        val data = fixtures().getJSONArray("curves")
        val failures = ArrayList<String>()
        for (i in 0 until data.length()) {
            val o = data.getJSONObject(i)
            val cp = if (o.isNull("score_cp")) null else o.getInt("score_cp")
            val mate = if (o.isNull("mate")) null else o.getInt("mate")
            val wdl = o.getJSONArray("wdl").let { Triple(it.getInt(0), it.getInt(1), it.getInt(2)) }
            near("human($cp, $mate)", humanExpectedScore(cp, mate), o.getDouble("human"), failures)
            near("engine($wdl)", engineExpectedScore(wdl), o.getDouble("engine"), failures)
        }
        TestLog.i(TAG, "${data.length()} curve points checked")
        assertTrue(failures.take(10).joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun everyMoveIsNamedAsThePythonNamesIt() {
        val data = fixtures().getJSONArray("labels")
        val failures = ArrayList<String>()
        for (i in 0 until data.length()) {
            val o = data.getJSONObject(i)
            val loss = o.getDouble("loss")
            val label = labelMove(
                loss,
                DEFAULT_THRESHOLDS,
                isBest = o.getBoolean("is_best"),
                inBook = o.getBoolean("in_book"),
                decisive = o.getBoolean("decisive"),
                sacrifice = o.getBoolean("sacrifice"),
                beforeScore = o.getDouble("before_score"),
                opponentBlundered = o.getBoolean("opponent_blundered"),
                secondScore = if (o.isNull("second_score")) null else o.getDouble("second_score"),
                chance = o.getBoolean("chance"),
                mateMissed = o.getBoolean("mate_missed"),
                mateAllowed = o.getBoolean("mate_allowed"),
                obvious = o.getBoolean("obvious"),
            )
            if (label.name.lowercase() != o.getString("label")) {
                failures.add("loss $loss: the label is ${label.name.lowercase()}, python's is ${o.getString("label")}")
            }
            val sev = severityOf(loss, DEFAULT_THRESHOLDS)
            if (sev.name.lowercase() != o.getString("severity")) {
                failures.add("loss $loss: the severity is $sev, python's is ${o.getString("severity")}")
            }
            near("accuracy($loss)", moveAccuracy(loss), o.getDouble("accuracy"), failures, 0.005)
        }
        TestLog.i(TAG, "${data.length()} labels checked")
        assertTrue(failures.take(10).joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun everyPositionIsJudgedAsThePythonJudgesIt() {
        val data = fixtures().getJSONArray("positions")
        val failures = ArrayList<String>()
        for (i in 0 until data.length()) {
            val o = data.getJSONObject(i)
            val board = Board(o.getString("fen"))
            val lines = lines(o.getJSONArray("lines"), o.getString("model"))
            val expected = o.getJSONArray("expected")
            for (k in lines.indices) {
                near("line $k", lines[k].expectedScore, expected.getDouble(k), failures)
            }
            val gap = bestGap(lines)
            if (o.isNull("best_gap")) {
                if (gap != null) failures.add("a gap of $gap where python has none")
            } else {
                near("best gap", gap ?: -1.0, o.getDouble("best_gap"), failures)
            }
            val only = isOnlyMove(board, lines, DEFAULT_THRESHOLDS)
            if (only != o.getBoolean("only_move")) {
                failures.add("${o.getString("fen")}: only move $only, python ${o.getBoolean("only_move")}")
            }
            val decisive = isDecisive(board, lines, DEFAULT_THRESHOLDS)
            if (decisive != o.getBoolean("decisive")) {
                failures.add("${o.getString("fen")}: decisive $decisive, python ${o.getBoolean("decisive")}")
            }
            near("position score", positionScore(board, lines), o.getDouble("position_score"), failures)
            val terminal = terminalExpectedScore(board)
            if (o.isNull("terminal")) {
                if (terminal != null) failures.add("${o.getString("fen")}: terminal $terminal, python none")
            } else {
                near("terminal", terminal ?: -1.0, o.getDouble("terminal"), failures)
            }
            if (!o.isNull("played")) {
                val loss = evalLoss(lines, o.getString("played"), o.getDouble("after_score"))
                near("eval loss", loss, o.getDouble("eval_loss"), failures)
            }
        }
        TestLog.i(TAG, "${data.length()} positions checked")
        assertTrue(failures.take(10).joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun aGameSumsUpAsThePythonSumsItUp() {
        val data = fixtures().getJSONArray("games")
        val failures = ArrayList<String>()
        for (i in 0 until data.length()) {
            val o = data.getJSONObject(i)
            val accuracies = o.getJSONArray("accuracies").let { a ->
                List(a.length()) { a.getDouble(it) }
            }
            near("game accuracy", gameAccuracy(accuracies) ?: -1.0, o.getDouble("game"), failures)
        }
        assertTrue(failures.take(10).joinToString("\n"), failures.isEmpty())
    }

    private fun lines(array: JSONArray, model: String): List<Line> = (0 until array.length()).map { i ->
        val o = array.getJSONObject(i)
        val cp = if (o.isNull("score_cp")) null else o.getInt("score_cp")
        val mate = if (o.isNull("mate")) null else o.getInt("mate")
        val wdl = o.getJSONArray("wdl").let { Triple(it.getInt(0), it.getInt(1), it.getInt(2)) }
        Line(
            moves = o.getJSONArray("moves").let { a -> List(a.length()) { a.getString(it) } },
            depth = o.getInt("depth"),
            expectedScore = expectedScore(model, cp, mate, wdl),
            scoreCp = cp,
            mate = mate,
            wdl = wdl,
        )
    }

    private fun near(
        what: String,
        mine: Double,
        theirs: Double,
        failures: MutableList<String>,
        tolerance: Double = 1e-9,
    ) {
        if (abs(mine - theirs) > tolerance) failures.add("$what: $mine, python's is $theirs")
    }

    companion object {
        private const val TAG = "sidecoach.analysis"
    }
}
