package com.sidecoach.analysis

import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Event
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import com.sidecoach.events.MoveAssessment
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * The annotator with its collaborators held to `sidecoach/analysis/annotator.py`: four real
 * games, each played on one board so the moves that led to every position come along, and
 * every ply's label, severity, loss, best line, move tags, position tags, opening, decisive
 * and only-move calls, threats, traps, declined recaptures and what a quiet move threatens
 * compared with what the desktop's annotator said
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
        android.util.Log.i("sidecoach.analysis", "annotator parity: $plies plies, $traps traps, $defences defences")
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

        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean): List<Line> {
            calls.add(board.fen())
            multipvs.add(multipv ?: 1)
            if (board.isGameOver(claimDraw = true)) return emptyList()
            return answers[board.fen()] ?: listOf(Line(moves = listOf("a2a3"), depth = 1, expectedScore = 0.5))
        }

        override fun close() = Unit
    }

    /**
     * `test_each_position_is_measured_once`: whole once each; the verdict on a move searches the
     * position it led to for its best line alone first, then the whole; a new game measures again.
     */
    @Test
    fun eachPositionIsMeasuredOnce() {
        val board = Board()
        val fake = FakeEngine()
        val annotator = GameAnnotator(fake, depth = 6, withThreats = false)
        val e2e4 = Move.fromUci("e2e4")
        annotator.assess("g", board, e2e4, 1)
        val after = board.copy(withStack = false)
        after.push(e2e4)
        annotator.assess("g", after, Move.fromUci("e7e5"), 2)
        assertEquals("after e4, after e5: each measured whole once", 2, fake.multipvs.count { it != 1 })
        assertEquals("one verdict search per position", 3, fake.multipvs.count { it == 1 })
        annotator.reset()
        annotator.assess("g", board, e2e4, 1)
        assertEquals("a new game measures again", 3, fake.multipvs.count { it != 1 })
    }

    /** `test_the_assessment_says_how_deep_the_search_got`: the shallower of the two searches. */
    @Test
    fun theAssessmentSaysHowDeepTheSearchGot() {
        val board = Board()
        val after = board.copy()
        after.push(Move.fromUci("e2e4"))
        val fake = FakeEngine(
            mapOf(
                board.fen() to listOf(Line(moves = listOf("e2e4"), depth = 18, expectedScore = 0.55)),
                after.fen() to listOf(Line(moves = listOf("e7e5"), depth = 11, expectedScore = 0.45)),
            ),
        )
        val (assessment, _) = GameAnnotator(fake, depth = 18, withThreats = false).assess("g", board, Move.fromUci("e2e4"), 1)
        assertEquals(11, assessment.depth)
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

    /**
     * The worker: it answers off the caller's thread, a failing analysis does not end the
     * session, and a queue it cannot keep up with drops the oldest work
     * (`test_the_analyser_answers_off_the_caller_s_thread`, `test_a_failing_analysis_does_not_end_the_session`,
     * `test_the_queue_drops_the_oldest_work_when_it_cannot_keep_up`).
     */
    @Test
    fun theWorkerAnswersSurvivesAFailureAndDropsTheOldest() {
        val answered = Analyser(GameAnnotator(FakeEngine(), depth = 6, withThreats = false))
        answered.start()
        answered.submitMove("g", Board(), Move.fromUci("e2e4"), 1)
        val events = drain(answered)
        answered.close()
        assertEquals(listOf("MoveAssessment", "Analysis"), events.map { it::class.java.simpleName })
        assertEquals(0, answered.failures)

        val broken = object : FakeEngine() {
            override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean): List<Line> =
                throw RuntimeException("engine gone")
        }
        val failing = Analyser(GameAnnotator(broken, depth = 6, withThreats = false))
        failing.start()
        failing.submitMove("g", Board(), Move.fromUci("e2e4"), 1)
        val deadline = System.currentTimeMillis() + 5000
        while (failing.failures == 0 && System.currentTimeMillis() < deadline) Thread.sleep(10)
        failing.close()
        assertEquals(1, failing.failures)
        assertTrue("${failing.lastError}", failing.lastError?.contains("engine gone") == true)

        val stuck = Analyser(GameAnnotator(FakeEngine(), depth = 6, withThreats = false), maxPending = 2)  // never started
        repeat(5) { stuck.submitPosition("g", Board(), 0) }
        assertEquals(3, stuck.dropped)
        assertEquals(2, stuck.pending)
    }

    /**
     * `Analyser.reset` takes everything queued, the stop a closing analyser put there included,
     * as the desktop's `get_nowait` loop does: nothing is left pending (the phone put the stop
     * back and left it counted).
     */
    @Test
    fun aResetTakesTheQueueAndTheStopInIt() {
        val searching = java.util.concurrent.CountDownLatch(1)
        val go = java.util.concurrent.CountDownLatch(1)
        val held = object : FakeEngine() {
            override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean): List<Line> {
                searching.countDown()
                go.await(5, java.util.concurrent.TimeUnit.SECONDS)
                return super.analyse(board, depth, timeS, multipv, pvPlies, carryBest)
            }
        }
        val analyser = Analyser(GameAnnotator(held, depth = 6, withThreats = false))
        analyser.submitMove("g", Board(), Move.fromUci("e2e4"), 1)
        analyser.start()
        assertTrue("the worker never started", searching.await(5, java.util.concurrent.TimeUnit.SECONDS))
        val after = Board()
        after.push(Move.fromUci("e2e4"))
        analyser.submitMove("g", after, Move.fromUci("e7e5"), 2)
        // Closing while the worker is held: the stop goes in behind ply 2, then it waits.
        val closer = Thread { analyser.close() }.apply { start() }
        val until = System.currentTimeMillis() + 5000
        while (closer.state != Thread.State.TIMED_WAITING && System.currentTimeMillis() < until) Thread.sleep(5)
        analyser.reset()
        assertEquals("ply 1 in the engine, ply 2 and the stop taken", 0, analyser.pending)
        go.countDown()
    }

    /**
     * `test_behind_a_fast_game_the_analyser_keeps_the_verdicts_and_drops_stale_analyses`: moves
     * faster than the engine -- every ply still gets its verdict and an analysis, a thin one from
     * the verdict's own line for the plies the game has left, and the whole measurement only for
     * the position the player faces (2026-09-22: a queen ending at two seconds a move ran forty
     * plies behind).
     */
    @Test
    fun behindAFastGameTheVerdictsAreKeptAndStaleAnalysesThin() {
        val fake = FakeEngine()
        val analyser = Analyser(GameAnnotator(fake, depth = 6, withThreats = false))
        val board = Board()
        // Five plies queued before the worker starts: it is behind from the first one.
        for ((i, san) in listOf("e4", "e5", "Nf3", "Nc6", "Bb5").withIndex()) {
            val move = board.parseSan(san)!!
            analyser.submitMove("g", board, move, i + 1)
            board.push(move)
        }
        analyser.start()
        val got = ArrayList<com.sidecoach.events.Event>()
        val deadline = System.currentTimeMillis() + 5000
        while (System.currentTimeMillis() < deadline && got.count { it is com.sidecoach.events.Analysis } < 5) {
            got += analyser.poll()
            Thread.sleep(20)
        }
        Thread.sleep(100)
        got += analyser.poll()
        analyser.close()
        val assessments = got.filterIsInstance<com.sidecoach.events.MoveAssessment>()
        val analyses = got.filterIsInstance<com.sidecoach.events.Analysis>()
        assertEquals("a verdict for every ply", listOf(1, 2, 3, 4, 5), assessments.map { it.ply })
        assertEquals(listOf(1, 2, 3, 4, 5), analyses.map { it.ply })
        assertTrue("thin for stale plies", analyses.take(4).all { it.lines.size == 1 && it.threats.isEmpty() })
        assertEquals(4, analyser.skipped)
        assertEquals("one whole measurement: the position the player faces", 1, fake.multipvs.count { it == 6 })
    }

    private fun drain(analyser: Analyser): List<com.sidecoach.events.Event> {
        val deadline = System.currentTimeMillis() + 10_000
        val events = ArrayList<com.sidecoach.events.Event>()
        while (System.currentTimeMillis() < deadline) {
            events += analyser.poll()
            if (events.isNotEmpty() && analyser.pending == 0) return events
            Thread.sleep(10)
        }
        return events
    }

    /** `_line`: a line of [moves] at depth 6, even. */
    private fun line(vararg moves: String) = Line(moves = moves.toList(), depth = 6, expectedScore = 0.5)

    /** `_after`: the position after [sans] from the start. */
    private fun after(vararg sans: String): String {
        val board = Board()
        for (san in sans) board.push(board.parseSan(san)!!)
        return board.fen()
    }

    /** `_until`: whether [condition] holds within [timeoutS]. */
    private fun until(timeoutS: Double = 5.0, condition: () -> Boolean): Boolean {
        val deadline = SystemClock.elapsedRealtime() + (timeoutS * 1000).toLong()
        while (!condition() && SystemClock.elapsedRealtime() < deadline) Thread.sleep(10)
        return condition()
    }

    private fun calls(fake: FakeEngine): List<String> = synchronized(fake.calls) { fake.calls.toList() }

    /**
     * `test_while_no_move_waits_the_engine_looks_ahead_and_a_move_there_is_answered_at_once`
     * (desktop 9c0774d). Tai, 2026-09-27: "why is it not possible to do some form of SMART
     * caching of lines". With the position on the board measured and nothing queued, the
     * positions after its best moves, and after the best reply to each, are measured and
     * cached; the move played there is judged, and the position after it known, with no search
     * of its own.
     */
    @Test
    fun whileNoMoveWaitsTheEngineLooksAheadAndAMoveThereIsAnsweredAtOnce() {
        val start = Board()
        val fake = FakeEngine(
            mapOf(
                start.fen() to listOf(line("e2e4"), line("d2d4")),
                after("e4") to listOf(line("e7e5")),
                after("d4") to listOf(line("d7d5")),
            ),
        )
        val annotator = GameAnnotator(fake, depth = 6, withThreats = false)
        val analyser = Analyser(annotator, lookAhead = 2)
        analyser.start()
        val got: List<Event>
        val before: Int
        try {
            analyser.submitPosition("g", start, 0)
            assertTrue("${analyser.looked}", until { analyser.looked == 4 })
            val searched = calls(fake).toSet()
            for (fen in listOf(after("e4"), after("e4", "e5"), after("d4"), after("d4", "d5"))) {
                assertTrue("$fen was not looked at", fen in searched)
            }
            before = calls(fake).size
            analyser.submitMove("g", start, Move.fromUci("e2e4"), 1)
            got = drain(analyser)
        } finally {
            analyser.close()
        }
        assertEquals(listOf("Analysis", "MoveAssessment", "Analysis"), got.map { it::class.java.simpleName })
        assertTrue("the position after e4 was searched again", after("e4") !in calls(fake).drop(before))
        assertEquals(1, annotator.aheadHits)
    }

    /**
     * `_StoppableEngine`: a fake engine whose searches ahead run until `interrupt` stops them
     * (`_stopped.wait(5)`, then `Interrupted`).
     */
    private class StoppableEngine(answers: Map<String, List<Line>>) : FakeEngine(answers) {
        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean): List<Line> {
            if (interruptibleNow) {
                val until = SystemClock.elapsedRealtime() + 5000
                while (!stopped && SystemClock.elapsedRealtime() < until) Thread.sleep(1)
                throw Interrupted()
            }
            return super.analyse(board, depth, timeS, multipv, pvPlies, carryBest)
        }
    }

    /**
     * `test_a_move_played_stops_the_look_ahead_at_once` (desktop 9c0774d): a search ahead of the
     * game is let go the moment a move of the game's own comes: the move never waits for a
     * position it did not go to.
     */
    @Test
    fun aMovePlayedStopsTheLookAheadAtOnce() {
        val fake = StoppableEngine(mapOf(Board().fen() to listOf(line("e2e4"), line("d2d4"))))
        val annotator = GameAnnotator(fake, depth = 6, withThreats = false)
        val analyser = Analyser(annotator, lookAhead = 2)
        analyser.start()
        val got: List<Event>
        try {
            analyser.submitPosition("g", Board(), 0)
            assertTrue("never looked ahead", until { analyser.looking })
            val t = SystemClock.elapsedRealtime()
            analyser.submitMove("g", Board(), Move.fromUci("g1f3"), 1)
            assertTrue(until(1.0) { !analyser.looking && analyser.pending == 0 })
            assertTrue(SystemClock.elapsedRealtime() - t < 500)
            got = drain(analyser)
        } finally {
            analyser.close()
        }
        assertTrue(got.any { it is MoveAssessment && it.uci == "g1f3" })
        assertTrue("${analyser.looked} ${annotator.aheadHits}", analyser.looked == 0 && annotator.aheadHits == 0)
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
