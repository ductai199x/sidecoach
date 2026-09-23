package com.sidecoach.analysis

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * The annotator with its collaborators held to `sidecoach/analysis/annotator.py`: four real
 * games, each played on one board so the moves that led to every position come along, and
 * every ply's label, severity, loss, best line, move tags, position tags, opening, decisive
 * and only-move calls and threats compared with what the desktop's annotator said
 * (`tools/make_annotator_fixtures.py`). The engine is scripted from the desktop's recording
 * of the real Stockfish, so what is compared is the annotator itself.
 */
@RunWith(AndroidJUnit4::class)
class AnnotatorParityTest {

    private val instrumentation = InstrumentationRegistry.getInstrumentation()

    /**
     * Answers the calls the desktop's annotator made, from its recording of them: per game,
     * and a position asked twice gets the answers in the order the engine gave them.
     */
    private class ScriptedEngine : Engine(File("scripted"), null) {
        val missed = ArrayList<String>()
        private var answers = JSONObject()
        private val asked = HashMap<String, Int>()

        fun play(game: JSONObject) {
            answers = game.getJSONObject("answers")
            asked.clear()
        }

        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?): List<Line> {
            val key = "${board.fen()}|$multipv"
            val calls = answers.optJSONArray(key)
            val n = asked.getOrDefault(key, 0)
            if (calls == null || n >= calls.length()) {
                missed.add(key)
                return emptyList()
            }
            asked[key] = n + 1
            val recorded = calls.getJSONArray(n)
            return (0 until recorded.length()).map { i ->
                val line = recorded.getJSONObject(i)
                val wdl = line.optJSONArray("wdl")
                Line(
                    moves = strings(line.getJSONArray("moves")),
                    depth = line.getInt("depth"),
                    expectedScore = line.getDouble("expected_score"),
                    scoreCp = if (line.isNull("score_cp")) null else line.getInt("score_cp"),
                    mate = if (line.isNull("mate")) null else line.getInt("mate"),
                    wdl = if (wdl == null) null else Triple(wdl.getInt(0), wdl.getInt(1), wdl.getInt(2)),
                )
            }
        }

        override fun close() = Unit
    }

    private fun book(): OpeningBook {
        val folder = File(instrumentation.targetContext.cacheDir, "openings-annotator").apply {
            deleteRecursively()
            mkdirs()
        }
        val assets = instrumentation.context.assets
        for (name in assets.list("openings").orEmpty().filter { it.endsWith(".tsv") }) {
            assets.open("openings/$name").use { input -> File(folder, name).outputStream().use { input.copyTo(it) } }
        }
        val loaded = OpeningBook.load(folder)
        assertNotNull("the tables load", loaded)
        return loaded!!
    }

    private fun tags(array: JSONArray): List<MotifTag> = (0 until array.length()).map { i ->
        val tag = array.getJSONObject(i)
        MotifTag(tag.getString("name"), strings(tag.getJSONArray("squares")), if (tag.isNull("move")) null else tag.getString("move"))
    }

    @Test
    fun everyPlyOfFourGamesMatchesThePython() {
        val text = instrumentation.context.assets.open("chess/annotator.json").bufferedReader().use { it.readText() }
        val data = JSONObject(text)
        val engine = ScriptedEngine()
        val book = book()
        val games = data.getJSONArray("games")
        var plies = 0
        for (g in 0 until games.length()) {
            val game = games.getJSONObject(g)
            val name = game.getString("name")
            engine.play(game)
            val annotator = GameAnnotator(
                engine,
                depth = data.getInt("depth"),
                multipv = data.getInt("multipv"),
                tagger = Motifs,
                book = book,
            )
            val board = Board()
            val expected = game.getJSONArray("plies")
            for (i in 0 until expected.length()) {
                val want = expected.getJSONObject(i)
                val move = Move.fromUci(want.getString("uci"))
                val ply = i + 1
                val (assessment, analysis) = annotator.assess(name, board, move, ply)
                board.push(move)
                val where = "$name ply $ply ${want.getString("uci")}"
                assertEquals("$where label", want.optString("label").ifEmpty { null }, assessment.label?.name?.lowercase())
                assertEquals("$where severity", want.getString("severity"), assessment.severity.name.lowercase())
                assertEquals("$where loss", want.getDouble("expected_loss"), assessment.expectedLoss, 1e-9)
                assertEquals("$where best", strings(want.getJSONArray("best")), assessment.best)
                assertEquals("$where move tags", tags(want.getJSONArray("move_tags")), assessment.tags)
                assertEquals("$where depth", want.getInt("depth"), assessment.depth)
                if (want.isNull("accuracy")) {
                    assertEquals("$where accuracy", null, assessment.accuracy)
                } else {
                    assertEquals("$where accuracy", want.getDouble("accuracy"), assessment.accuracy!!, 1e-9)
                }
                assertEquals("$where position tags", tags(want.getJSONArray("position_tags")), analysis.tags)
                assertEquals("$where opening", if (want.isNull("opening")) null else want.getString("opening"), analysis.opening)
                assertEquals("$where decisive", want.getBoolean("decisive"), analysis.decisive)
                assertEquals("$where only move", want.getBoolean("only_move"), analysis.onlyMove)
                assertEquals("$where threats", strings(want.getJSONArray("threats")), analysis.threats)
                plies++
            }
        }
        assertEquals("every engine call was one the desktop made", emptyList<String>(), engine.missed)
        assertEquals(64, plies)
    }

    private companion object {
        fun strings(array: JSONArray): List<String> = (0 until array.length()).map { array.getString(it) }
    }
}
