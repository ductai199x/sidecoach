package com.sidecoach.analysis

import com.sidecoach.TestAssets
import com.sidecoach.TestLog
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import com.sidecoach.events.Severity
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/**
 * The annotator with its collaborators held to `sidecoach/analysis/annotator.py`: four real
 * games, each played on one board so the moves that led to every position come along, and
 * every ply's label, severity, loss, best line, move tags, position tags, opening, decisive
 * and only-move calls, threats, traps, declined recaptures and what a quiet move threatens
 * compared with what the desktop's annotator said
 * (`tools/make_annotator_fixtures.py`). The engine is scripted from the desktop's recording
 * of the real Stockfish, so what is compared is the annotator itself.
 *
 * On the host: nothing here needs the Android runtime. The analyser on its workers is
 * AnalyserTest's.
 */
class AnnotatorParityTest {

    /** The opening tables' folder: the device's test used the app's cache, the host a folder of its own. */
    @get:Rule
    val temporary = TemporaryFolder()

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

        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
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
        val folder = temporary.newFolder("openings-annotator")
        for (name in TestAssets.list("openings").filter { it.endsWith(".tsv") }) {
            TestAssets.open("openings/$name").use { input -> File(folder, name).outputStream().use { input.copyTo(it) } }
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
        val data = JSONObject(TestAssets.text("chess/annotator.json"))
        val engine = ScriptedEngine()
        val book = book()
        val games = data.getJSONArray("games")
        var plies = 0
        var traps = 0
        var defences = 0
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
                // A JSON null is no label (Android's `optString` would read it as "null").
                assertEquals("$where label", if (want.isNull("label")) null else want.getString("label"), assessment.label?.name?.lowercase())
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
                assertEquals("$where traps", recordedLines(want.getJSONArray("traps")), analysis.traps)
                assertEquals("$where defences", recordedLines(want.getJSONArray("defences")), analysis.defences)
                assertEquals("$where threatens", recordedLines(want.getJSONArray("threatens")), analysis.threatens)
                assertEquals("$where in book", want.getBoolean("in_book"), analysis.inBook)
                traps += analysis.traps.size
                defences += analysis.defences.size
                plies++
            }
        }
        assertEquals("every engine call was one the desktop made", emptyList<String>(), engine.missed)
        assertEquals(64, plies)
        TestLog.i("sidecoach.analysis", "annotator parity: $plies plies, $traps traps, $defences defences")
    }

    private fun recordedLines(array: JSONArray): List<Line> = (0 until array.length()).map { i ->
        val line = array.getJSONObject(i)
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

    /** The desktop test's `FakeEngine`: scripted lines, a quiet move for anything else, the calls kept. */
    private open class FakeEngine(private val answers: Map<String, List<Line>> = emptyMap()) : Engine(File("fake"), null) {
        val calls = java.util.Collections.synchronizedList(ArrayList<String>())
        val multipvs = java.util.Collections.synchronizedList(ArrayList<Int>())

        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
            calls.add(board.fen())
            multipvs.add(multipv ?: 1)
            if (com.sidecoach.analysis.gameOver(board)) return emptyList()
            return answers[board.fen()] ?: listOf(Line(moves = listOf("a2a3"), depth = 1, expectedScore = 0.5))
        }

        override fun close() = Unit
    }

    /**
     * `test_each_position_is_measured_once`: whole once each; the verdict on a move searches the
     * position it led to for its best line alone first (a call of its own, one line), then the
     * whole; a new game measures again. The order of the first move's calls is new here (the
     * parity audit, 2026-09-29): the counts alone held with the whole search first.
     */
    @Test
    fun eachPositionIsMeasuredOnce() {
        val board = Board()
        val fake = FakeEngine()
        val annotator = GameAnnotator(fake, depth = 6, withThreats = false)
        val e2e4 = Move.fromUci("e2e4")
        val (assessment, _) = annotator.assess("g", board, e2e4, 1)
        val after = board.copy(withStack = false)
        after.push(e2e4)
        assertEquals("e2e4", assessment.uci)
        assertEquals(
            "the verdict from one line of the position after the move, before it is measured whole",
            listOf(board.fen() to 1, after.fen() to 1, after.fen() to 6),
            calls(fake).zip(synchronized(fake.multipvs) { fake.multipvs.toList() }),
        )
        annotator.assess("g", after, Move.fromUci("e7e5"), 2)
        assertEquals("after e4, after e5: each measured whole once", 2, fake.multipvs.count { it != 1 })
        assertEquals("one verdict search per position", 3, fake.multipvs.count { it == 1 })
        annotator.reset()
        annotator.assess("g", board, e2e4, 1)
        assertEquals("a new game measures again", 3, fake.multipvs.count { it != 1 })
    }

    /**
     * `test_the_mate_distance_counts_the_move_being_judged` and
     * `test_an_illegal_move_is_refused_rather_than_measured`.
     */
    @Test
    fun theMateDistanceCountsTheMoveAndAnIllegalMoveIsRefused() {
        val before = Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
        assertEquals(1, mateIn(before, Move.fromUci("h5f7"), emptyList()))
        val quiet = Move.fromUci("d2d3")
        // After the move the opponent is to move and is mated in two of the mover's moves.
        assertEquals(3, mateIn(before, quiet, listOf(Line(moves = emptyList(), depth = 9, expectedScore = 0.0, mate = -2))))
        assertEquals(-2, mateIn(before, quiet, listOf(Line(moves = emptyList(), depth = 9, expectedScore = 1.0, mate = 2))))
        assertEquals(null, mateIn(before, quiet, listOf(Line(moves = emptyList(), depth = 9, expectedScore = 0.5))))
        val refused = runCatching {
            GameAnnotator(FakeEngine(), depth = 6, withThreats = false).assess("g", Board(), Move.fromUci("e2e5"), 1)
        }.exceptionOrNull()
        assertTrue("$refused", refused is IllegalArgumentException && "not legal" in refused.message!!)
    }

    /** `_line`: a line of [moves] at depth 6, even. */
    private fun line(vararg moves: String) = Line(moves = moves.toList(), depth = 6, expectedScore = 0.5)

    /** `_after`: the position after [sans] from the start. */
    private fun after(vararg sans: String): String {
        val board = Board()
        for (san in sans) board.push(board.parseSan(san)!!)
        return board.fen()
    }

    private fun calls(fake: FakeEngine): List<String> = synchronized(fake.calls) { fake.calls.toList() }

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

    /**
     * `test_the_move_that_replaces_a_withdrawn_one_knows_the_move_before_it` (the state review,
     * 2026-10-01): after a takeback the move before the replacement is the one the game kept;
     * the last move judged was the withdrawn one, a ply too far, and the replacement was judged
     * as if nothing came before it -- the opponent's blunder unseen, the best reply to it merely
     * best, not the great move it is.
     */
    @Test
    fun theMoveThatReplacesAWithdrawnOneKnowsTheMoveBeforeIt() {
        val fake = FakeEngine(
            mapOf(
                Board().fen() to listOf(Line(moves = listOf("d2d4"), depth = 6, expectedScore = 0.55)),
                after("e4") to listOf(Line(moves = listOf("e7e5"), depth = 6, expectedScore = 0.90)),
                after("e4", "a6") to listOf(Line(moves = listOf("d2d4"), depth = 6, expectedScore = 0.80)),
            ),
        )
        val annotator = GameAnnotator(fake, depth = 6, withThreats = false)
        val board = Board()
        val first = annotator.assessMove("g", board, board.parseSan("e4")!!, 1)
        assertEquals("the opponent blundered", Severity.BLUNDER, first.severity)
        board.push(board.parseSan("e4")!!)
        val withdrawn = annotator.assessMove("g", board, board.parseSan("a6")!!, 2)
        val replacement = annotator.assessMove("g", board, board.parseSan("e5")!!, 2)
        // The withdrawn a6 let the opponent's blunder go: a miss (Tai, 2026-10-02).
        assertEquals(Label.MISS, withdrawn.label)
        assertEquals(Label.GREAT, replacement.label)
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
     * `test_assessment_carries_the_loss_the_best_line_the_tags_and_the_depth`: the loss and its
     * severity, the best line, the depth (the shallower of the two searches: the position after
     * the move hit the engine's time cap), and the move's tags as the tagger gave them, less a
     * targeted tactic the line after the move never cashes in (`motifs.pays_off`, desktop
     * 5e80cdf); the analysis of the position after it, decisive and an only move. The severity,
     * the analysis's ply and position, and the decisive and only-move calls are new here (the
     * parity audit, 2026-09-29); the depth was a test of its own until the desktop folded it in.
     */
    @Test
    fun assessmentCarriesTheLossTheBestLineTheTagsAndTheDepth() {
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
                    Line(moves = listOf("e7e5"), depth = 4, expectedScore = 0.55, mate = -3),
                    Line(moves = listOf("d7d5"), depth = 4, expectedScore = 0.20),
                ),
            ),
        )
        val tagger = RecordingTagger()
        val annotator = GameAnnotator(fake, depth = 6, tagger = tagger, withThreats = false)
        val (assessment, analysis) = annotator.assess("game1", board, Move.fromUci("a2a3"), 1)
        assertEquals(0.15, assessment.expectedLoss, 1e-9)
        assertEquals(Severity.MISTAKE, assessment.severity)
        assertEquals(listOf("e2e4", "e7e5"), assessment.best)
        assertEquals("the shallower of the two searches", 4, assessment.depth)
        assertEquals("the unpaid fork is dropped", listOf("sacrifice"), assessment.tags.map { it.name })
        assertEquals("mated in three more moves, four counting this", listOf("a2a3" to 4), tagger.moveCalls)
        assertEquals(1, analysis.ply)
        assertEquals(after.fen(), analysis.fen)
        assertEquals(listOf("hangingPiece"), analysis.tags.map { it.name })
        assertTrue("the replies differ by 0.35", analysis.decisive && analysis.onlyMove)
    }

    private companion object {
        /** 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6??, where 4. Qxf7# is mate (the desktop's `SCHOLAR`). */
        const val SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"

        fun strings(array: JSONArray): List<String> = (0 until array.length()).map { array.getString(it) }
    }
}
