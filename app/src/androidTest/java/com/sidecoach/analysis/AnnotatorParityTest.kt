package com.sidecoach.analysis

import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Analysis
import com.sidecoach.events.Event
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.Severity
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

    /**
     * The worker: it answers off the caller's thread (the sequence and `failures == 0` are in
     * `test_the_position_the_player_faces_is_told_early_and_then_whole` on the desktop now), a
     * failing analysis does not end the session, and a queue it cannot keep up with drops the
     * oldest work (`test_a_failing_analysis_does_not_end_the_session`,
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
            override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> =
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
            override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
                searching.countDown()
                go.await(5, java.util.concurrent.TimeUnit.SECONDS)
                return super.analyse(board, depth, timeS, multipv, pvPlies, carryBest, onDepth)
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
        assertEquals(listOf("Analysis(early)", "Analysis", "MoveAssessment", "Analysis"), kinds(got))
        assertTrue("the position after e4 was searched again", after("e4") !in calls(fake).drop(before))
        assertEquals(1, annotator.aheadHits)
    }

    /** `_DepthEngine`: a fake engine that answers with the depth it was asked for. */
    private class DepthEngine(answers: Map<String, List<Line>> = emptyMap()) : FakeEngine(answers) {
        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
            val lines = super.analyse(board, depth, timeS, multipv, pvPlies, carryBest, onDepth)
            return lines.map { it.copy(depth = depth ?: 0) }
        }
    }

    /** `_kinds`: each event's type, an early analysis marked. */
    private fun kinds(events: List<Event>): List<String> = events.map {
        val name = it::class.java.simpleName
        if (it is Analysis && it.provisional) "$name(early)" else name
    }

    /**
     * `test_the_position_the_player_faces_is_told_early_and_then_whole` (desktop 316026b). Tai,
     * 2026-09-28: "first usable answer is depth 12 ... the computation continues until max depth
     * of 18". After the opponent's move, and for a position begun from, the position the player
     * faces is measured at EARLY_DEPTH and delivered, provisional, before the verdict and the
     * full depth; after the player's own move the verdict comes first, as it did.
     */
    @Test
    fun thePositionThePlayerFacesIsToldEarlyAndThenWhole() {
        val board = Board()
        val fake = DepthEngine()
        val analyser = Analyser(GameAnnotator(fake, depth = 18, withThreats = false))
        analyser.start()
        val started: List<Event>
        val own: List<Event>
        val reply: List<Event>
        try {
            analyser.submitPosition("g", board, 0)
            started = drain(analyser)
            analyser.submitMove("g", board, Move.fromUci("e2e4"), 1, byUser = true)
            own = drain(analyser)
            board.push(board.parseSan("e4")!!)
            analyser.submitMove("g", board, Move.fromUci("e7e5"), 2, byUser = false)
            reply = drain(analyser)
        } finally {
            analyser.close()
        }
        assertEquals(listOf("Analysis(early)", "Analysis"), kinds(started))
        assertEquals(listOf("MoveAssessment", "Analysis"), kinds(own))
        assertEquals(0, analyser.failures)
        assertEquals(listOf("Analysis(early)", "MoveAssessment", "Analysis"), kinds(reply))
        val early = reply[0] as Analysis
        val full = reply[2] as Analysis
        assertTrue("${early.lines[0].depth} ${full.lines[0].depth}", early.lines[0].depth == EARLY_DEPTH && full.lines[0].depth == 18)
        assertEquals(early.fen, full.fen)
    }

    /**
     * `test_behind_a_fast_game_the_position_faced_is_told_early_and_still_measured_whole` (the
     * state review, 2026-10-01; Tai, 2026-09-28: "first usable answer is depth 12 ... the
     * computation continues until max depth of 18"). Behind a fast game -- the player's move
     * and the reply submitted together, as the puzzle room does -- the position the player then
     * faces is told at EARLY_DEPTH first, and then measured at the full depth and looked ahead
     * from. It was measured at EARLY_DEPTH alone and kept as the whole: never deepened, nothing
     * looked ahead at, and the next move's verdict set depth 12 against depth 18.
     */
    @Test
    fun behindAFastGameThePositionFacedIsToldEarlyAndStillMeasuredWhole() {
        val fake = DepthEngine()
        val annotator = GameAnnotator(fake, depth = 18)
        val board = Board()
        val analyser = Analyser(annotator, lookAhead = 2)
        analyser.submitMove("g", board, Move.fromUci("e2e4"), 1, byUser = true)
        board.push(board.parseSan("e4")!!)
        analyser.submitMove("g", board, Move.fromUci("e7e5"), 2, byUser = false)
        board.push(board.parseSan("e5")!!)
        analyser.start()
        val got: List<Event>
        try {
            got = drain(analyser)
            assertTrue("nothing looked ahead at", until { analyser.looked >= 1 })
        } finally {
            analyser.close()
        }
        val faced = got.filterIsInstance<Analysis>().filter { it.ply == 2 }
        assertTrue("${kinds(got)}", faced.isNotEmpty() && faced[0].provisional && faced[0].lines[0].depth == EARLY_DEPTH)
        assertTrue("${faced.last().lines[0].depth}", !faced.last().provisional && faced.last().lines[0].depth == 18)
        assertEquals(board.fen(), faced.last().fen)
        val asked = calls(fake).size
        assertEquals("kept at the early depth", 18, annotator.measure(board).depth)
        assertEquals("measured again", asked, calls(fake).size)
    }

    /**
     * The position the player faces measured ahead of the move: no early word, the whole one at
     * once (desktop 316026b; the second half of
     * `test_while_no_move_waits_the_engine_looks_ahead_and_a_move_there_is_answered_at_once`
     * since the desktop folded it in).
     */
    @Test
    fun aPositionLookedAtAheadIsWholeAtOnce() {
        val start = Board()
        val fake = DepthEngine(mapOf(start.fen() to listOf(line("e2e4")), after("e4") to listOf(line("e7e5"))))
        val annotator = GameAnnotator(fake, depth = 18, withThreats = false)
        val analyser = Analyser(annotator, lookAhead = 1)
        analyser.start()
        val reply: List<Event>
        try {
            analyser.submitPosition("g", start, 0)
            drain(analyser)
            assertTrue("${analyser.looked}", until { analyser.looked == 2 })
            analyser.submitMove("g", start, Move.fromUci("e2e4"), 1, byUser = true)
            drain(analyser)
            val after = Board()
            after.push(after.parseSan("e4")!!)
            analyser.submitMove("g", after, Move.fromUci("e7e5"), 2, byUser = false)
            reply = drain(analyser)
        } finally {
            analyser.close()
        }
        assertEquals(listOf("MoveAssessment", "Analysis"), kinds(reply))
        assertEquals(2, annotator.aheadHits)
    }

    /**
     * `_YieldOnceEngine`: a fake engine whose first search of the candidates (several lines: a
     * whole measure, not a verdict's one line) runs until `interrupt` stops it; the ones after
     * answer at once.
     */
    private class YieldOnceEngine : FakeEngine() {
        @Volatile var held = false

        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
            val several = (multipv ?: 1) > 1
            if (interruptibleNow && several && !held) {
                held = true
                val until = SystemClock.elapsedRealtime() + 5000
                while (!stopped && SystemClock.elapsedRealtime() < until) Thread.sleep(1)
                throw Interrupted()
            }
            return super.analyse(board, depth, timeS, multipv, pvPlies, carryBest, onDepth)
        }
    }

    /**
     * `test_the_position_the_player_left_gives_way_to_the_one_they_face` (desktop 316026b).
     * Tai's puzzles, 2026-09-28: the site replies 0.6 s after the player's move, and the
     * position they then faced waited behind the full measure of the one they had left. That
     * measure gives way to the reply, whose position is told first, and is taken up again after
     * it -- still owed until it is.
     */
    @Test
    fun thePositionThePlayerLeftGivesWayToTheOneTheyFace() {
        val fake = YieldOnceEngine()
        val board = Board()
        val analyser = Analyser(GameAnnotator(fake, depth = 18, withThreats = false))
        analyser.start()
        val got: List<Event>
        try {
            analyser.submitMove("g", board, Move.fromUci("e2e4"), 1, byUser = true)
            assertTrue("the full measure never started", until { analyser.looking })
            board.push(board.parseSan("e4")!!)
            analyser.submitMove("g", board, Move.fromUci("e7e5"), 2, byUser = false)
            got = drain(analyser)
        } finally {
            analyser.close()
        }
        val order = got.map { e ->
            val ply = when (e) {
                is MoveAssessment -> e.ply
                is Analysis -> e.ply
                else -> null
            }
            Triple(e::class.java.simpleName, ply, e is Analysis && e.provisional)
        }
        assertEquals(
            order.toString(),
            listOf(
                Triple("MoveAssessment", 1, false),
                Triple("Analysis", 2, true),
                Triple("MoveAssessment", 2, false),
                Triple("Analysis", 2, false),
                Triple("Analysis", 1, false),
            ),
            order,
        )
        assertEquals(0, analyser.pending)
    }

    /**
     * `_StoppableEngine`: a fake engine whose searches of the positions ahead ([ahead]) run
     * until `interrupt` stops them (`_stopped.wait(5)`, then `Interrupted`); the game's own are
     * answered at once.
     */
    private class StoppableEngine(answers: Map<String, List<Line>>, val ahead: Set<String>) : FakeEngine(answers) {
        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
            if (interruptibleNow && board.fen() in ahead) {
                val until = SystemClock.elapsedRealtime() + 5000
                while (!stopped && SystemClock.elapsedRealtime() < until) Thread.sleep(1)
                throw Interrupted()
            }
            return super.analyse(board, depth, timeS, multipv, pvPlies, carryBest, onDepth)
        }
    }

    /**
     * `test_a_move_played_stops_the_look_ahead_at_once` (desktop 9c0774d): a search ahead of the
     * game is let go the moment a move of the game's own comes: the move never waits for a
     * position it did not go to.
     */
    @Test
    fun aMovePlayedStopsTheLookAheadAtOnce() {
        val fake = StoppableEngine(
            mapOf(Board().fen() to listOf(line("e2e4"), line("d2d4"))),
            setOf(after("e4"), after("d4")),
        )
        val annotator = GameAnnotator(fake, depth = 6, withThreats = false)
        val analyser = Analyser(annotator, lookAhead = 2)
        analyser.start()
        val got: List<Event>
        try {
            analyser.submitPosition("g", Board(), 0)
            assertTrue("never looked ahead", until { analyser.looking })
            val t = SystemClock.elapsedRealtime()
            analyser.submitMove("g", Board(), Move.fromUci("g1f3"), 1, byUser = false)
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
