package com.sidecoach.analysis

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Label
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
     * and a position asked twice gets the answers in the order the engine gave them. A call is
     * keyed by its position, its number of lines and whether the best line is carried on
     * (`carry_best`), so a call the desktop did not make the same way finds no answer.
     */
    private class ScriptedEngine : Engine(File("scripted"), null) {
        val missed = ArrayList<String>()
        private var answers = JSONObject()
        private val asked = HashMap<String, Int>()

        fun play(game: JSONObject) {
            answers = game.getJSONObject("answers")
            asked.clear()
        }

        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean): List<Line> {
            val key = "${board.fen()}|$multipv|$carryBest"
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

    /** The desktop test's `FakeEngine`: scripted lines, a quiet move for anything else. */
    private class FakeEngine(private val answers: Map<String, List<Line>>) : Engine(File("fake"), null) {
        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean): List<Line> {
            if (board.isGameOver(claimDraw = true)) return emptyList()
            return answers[board.fen()] ?: listOf(Line(moves = listOf("a2a3"), depth = 1, expectedScore = 0.5))
        }

        override fun close() = Unit
    }

    /**
     * `test_a_capture_guarded_only_by_a_pinned_piece_is_a_free_grab`: a capture nobody can take
     * back is obvious, never great, and taking back is read by the rules -- the c3 knight
     * guarding e4 is pinned to its king by the b4 bishop, so ...Nxe4 is a grab. With the knight
     * free to take back, the one move that holds is great.
     */
    @Test
    fun aCaptureGuardedOnlyByAPinnedPieceIsAFreeGrab() {
        fun label(fen: String): Label? {
            val board = Board(fen)
            val fake = FakeEngine(
                mapOf(
                    board.fen() to listOf(
                        Line(moves = listOf("f6e4"), depth = 6, expectedScore = 0.80),
                        Line(moves = listOf("e8d7"), depth = 6, expectedScore = 0.40),
                    ),
                ),
            )
            val annotator = GameAnnotator(fake, depth = 6, withThreats = false)
            annotator.measure(board)  // measured whole, so the only-move call is made
            return annotator.assessMove("g", board, Move.fromUci("f6e4"), 1).label
        }

        assertEquals(Label.BEST, label("4k3/8/5n2/8/1b2B3/2N5/8/4K3 b - - 0 1"))
        assertEquals(Label.GREAT, label("4k3/8/5n2/8/4B3/2N5/8/4K3 b - - 0 1"))
    }

    /** The desktop test's `RecordingTagger`: the same tags for every move, the calls kept. */
    private class RecordingTagger : Tagger {
        val moveCalls = ArrayList<Pair<String, Int?>>()

        override fun tagsForMove(before: Board, move: Move, pv: List<Move>, mateIn: Int?): List<MotifTag> {
            moveCalls.add(move.uci() to mateIn)
            // A fork the line never cashes in, and a sacrifice the payoff rule does not judge:
            // the annotator keeps a move's tactic only when the line after it wins with it.
            return listOf(
                MotifTag("fork", listOf("a3", "b4", "c4"), move.uci()),
                MotifTag("sacrifice", listOf("a3"), move.uci()),
            )
        }

        override fun tagsForPosition(board: Board, pv: List<Move>, mateIn: Int?): List<MotifTag> =
            listOf(MotifTag("hangingPiece", listOf("d1"), null))
    }

    /**
     * `test_assessment_carries_the_loss_the_best_line_and_the_tags`: the loss, the best line,
     * and the move's tags as the tagger gave them, less a targeted tactic the line after the
     * move never cashes in (`motifs.pays_off`, desktop 5e80cdf).
     */
    @Test
    fun anAssessmentKeepsOnlyTheTacticsTheLineCashesIn() {
        val board = Board()
        val after = board.copy(withStack = false)
        after.push(Move.fromUci("a2a3"))
        val fake = FakeEngine(
            mapOf(
                board.fen() to listOf(
                    Line(moves = listOf("e2e4", "e7e5"), depth = 6, expectedScore = 0.60),
                    Line(moves = listOf("d2d4"), depth = 6, expectedScore = 0.45),
                ),
                after.fen() to listOf(
                    Line(moves = listOf("e7e5"), depth = 6, expectedScore = 0.55, mate = -3),
                    Line(moves = listOf("d7d5"), depth = 6, expectedScore = 0.20),
                ),
            ),
        )
        val tagger = RecordingTagger()
        val annotator = GameAnnotator(fake, depth = 6, tagger = tagger, withThreats = false)
        val (assessment, analysis) = annotator.assess("game1", board, Move.fromUci("a2a3"), 1)
        assertEquals(0.15, assessment.expectedLoss, 1e-9)
        assertEquals(listOf("e2e4", "e7e5"), assessment.best)
        assertEquals("the unpaid fork is dropped", listOf("sacrifice"), assessment.tags.map { it.name })
        assertEquals("mated in three more moves, four counting this", listOf("a2a3" to 4), tagger.moveCalls)
        assertEquals(listOf("hangingPiece"), analysis.tags.map { it.name })
    }

    private companion object {
        fun strings(array: JSONArray): List<String> = (0 until array.length()).map { array.getString(it) }
    }
}
