package com.sidecoach.state

import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.analysis.Analyser
import com.sidecoach.analysis.Assessing
import com.sidecoach.analysis.Engine
import com.sidecoach.analysis.GameAnnotator
import com.sidecoach.analysis.Interrupted
import com.sidecoach.chess.Board
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.coach.Coach
import com.sidecoach.events.Analysis
import com.sidecoach.events.Event
import com.sidecoach.events.GameStarted
import com.sidecoach.events.Line
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.Resync
import com.sidecoach.events.Takeback
import com.sidecoach.events.Unexplained
import com.sidecoach.llm.OfflineProvider
import com.sidecoach.vision.CellEmbedder
import com.sidecoach.vision.ChangeGate
import com.sidecoach.vision.FramePixels
import com.sidecoach.vision.Pieces
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Takebacks on the phone, as `tests/test_takebacks.py` and `test_analysis.py` hold them on the
 * desktop (Tai's game on the Pixel, 2026-09-23: "takebacks make the program lag behind by a
 * lot", and the game state in conflict after them).
 *
 * The analyser lets go of the withdrawn plies; the engine glue passes on nothing about a move
 * or position the game has left; and the recorded scenarios -- a takeback of two plies, ten
 * taken back quickly with the pieces sliding, a frame each, all at once, and a board left
 * unexplained for long before eleven are taken back -- run through the ported pipeline, the
 * glue and the coach, which must count the same plies as the reconciler throughout.
 */
@RunWith(AndroidJUnit4::class)
class TakebackTest {

    /** Every position answered with the same quiet line, as the desktop's FakeEngine. */
    private open class QuietEngine : Engine(File("quiet"), null) {
        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> =
            if (board.isGameOver(claimDraw = true)) emptyList()
            else listOf(Line(moves = listOf("a2a3"), depth = 1, expectedScore = 0.5))

        override fun close() = Unit
    }

    /** Its searches wait to be let go, so a test can act mid-search. */
    private class HeldEngine : QuietEngine() {
        val searching = CountDownLatch(1)
        val go = CountDownLatch(1)

        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
            searching.countDown()
            go.await(5, TimeUnit.SECONDS)
            return super.analyse(board, depth, timeS, multipv, pvPlies, carryBest, onDepth)
        }
    }

    /**
     * A moment per search, as a real engine takes, and stopped as a real one is
     * ([Engine.interrupt]): none begins once stopped, and one under way ends within a moment.
     * While [watching] -- from the moment the glue is told of a takeback -- it notes the
     * position each search begins on (`_SlowEngine.asked`).
     */
    private class SlowEngine(@Volatile var ms: Long) : QuietEngine() {
        @Volatile var watching = false
        val asked: MutableList<String> = java.util.Collections.synchronizedList(ArrayList<String>())

        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
            if (interruptibleNow && stopped) throw Interrupted()
            if (watching) asked.add(board.boardFen())
            val end = SystemClock.elapsedRealtime() + ms
            while (SystemClock.elapsedRealtime() < end) {
                if (interruptibleNow && stopped) throw Interrupted()
                Thread.sleep(5)
            }
            return super.analyse(board, depth, timeS, multipv, pvPlies, carryBest, onDepth)
        }
    }

    private fun played(board: Board, san: String, ply: Int): MovePlayed {
        val before = board.fen()
        val move = board.parseSan(san)!!
        board.push(move)
        return MovePlayed(
            gameId = "g", ply = ply, san = san, uci = move.uci(), fenBefore = before,
            fenAfter = board.fen(), byUser = ply % 2 == 1,
        )
    }

    @Test
    fun aTakebackLetsTheEngineGoOfTheWithdrawnMoves() {
        val engine = HeldEngine()
        val analyser = Analyser(GameAnnotator(engine, depth = 6, withThreats = false))
        val board = Board()
        val moves = "e4 e5 Nf3 Nc6".split(" ").mapIndexed { i, san ->
            val before = board.copy()
            val move = board.parseSan(san)!!
            board.push(move)
            Triple(before, move, i + 1)
        }
        analyser.submitMove("g", moves[0].first, moves[0].second, 1)
        analyser.start()
        assertTrue("the worker never started on ply 1", engine.searching.await(5, TimeUnit.SECONDS))
        for ((before, move, ply) in moves.drop(1)) analyser.submitMove("g", before, move, ply)
        // Plies 3 and 4 taken back while ply 1 is measured and 2 waits.
        analyser.withdraw("g", 2)
        assertEquals(2, analyser.withdrawn)
        assertEquals(2, analyser.pending)
        // Ply 1 taken back too, mid-search: its verdict is not delivered.
        analyser.withdraw("g", 0)
        assertEquals(4, analyser.withdrawn)
        engine.go.countDown()
        val got = ArrayList<Event>()
        val deadline = System.currentTimeMillis() + 5000
        while (System.currentTimeMillis() < deadline) {
            got += analyser.poll()
            if (analyser.pending == 0) break
            Thread.sleep(10)
        }
        Thread.sleep(100)
        got += analyser.poll()
        analyser.close()
        val plies = got.mapNotNull { (it as? MoveAssessment)?.ply ?: (it as? Analysis)?.ply }
        assertTrue("words on withdrawn plies: $plies", plies.none { it == 1 || it == 3 || it == 4 })
        assertEquals(0, analyser.pending)
    }

    /** A model that answers [reply] and remembers that it was asked (`ScriptedProvider`). */
    private class OneReply(private val reply: String) : com.sidecoach.llm.LLMProvider {
        override val name = "scripted"
        @Volatile var asked = 0

        override fun healthcheck(): Pair<Boolean, String> = true to "scripted"

        override fun chat(
            system: String,
            messages: List<com.sidecoach.llm.ChatMessage>,
            maxTokens: Int,
            temperature: Double,
            thinking: Boolean,
        ): com.sidecoach.llm.Completion {
            asked += 1
            return com.sidecoach.llm.Completion(text = reply)
        }
    }

    /**
     * `test_a_hint_asked_before_the_first_move_is_answered` (the state review, 2026-10-01): the
     * start position was never sent to the engine -- only a position begun from the middle was
     * -- so a hint asked before the first move was "One moment -- still looking" and nothing
     * after it, and no stuck hint came there. The engine glue and the coach on a model: the
     * model is asked, and answers.
     */
    @Test
    fun aHintAskedBeforeTheFirstMoveIsAnswered() {
        val hint = "Think about which piece you want in the centre first."
        val analyser = Analyser(GameAnnotator(QuietEngine(), depth = 6, withThreats = false))
        val assessing = Assessing(analyser)
        analyser.start()
        val model = OneReply(hint)
        val coach = Coach(
            model,
            policy = com.sidecoach.coach.InterjectionPolicy(com.sidecoach.coach.PolicySettings(stuckSeconds = 10.0)),
        )
        val drained = ArrayList<Event>()
        val said = ArrayList<String>()
        try {
            val started = GameStarted(gameId = "g", userSide = Pieces.Side.WHITE, startFen = STARTING_FEN)
            assessing.submit(started)
            coach.onEvent(started, 0.0)
            coach.ask("hint", 0.5)
            val deadline = System.currentTimeMillis() + 5000
            while (System.currentTimeMillis() < deadline) {
                val got = assessing.drain()
                drained += got
                for (event in got) coach.onEvent(event, 1.0)
                if (analyser.pending == 0 && got.isEmpty()) break
                Thread.sleep(10)
            }
            coach.speaker.runPending()
            said += coach.poll().map { it.text }
        } finally {
            assessing.close()
        }
        assertTrue("never measured", drained.any { it is Analysis && it.ply == 0 })
        assertTrue("the model was never asked: $said", model.asked > 0)
        assertTrue("$said", hint in said)
    }

    /**
     * `test_a_takeback_to_where_the_game_began_measures_it_again`: every move of a position
     * taken back, the position the game began from is the one the player faces again, and the
     * engine is asked for it again (at once, it is measured), so the look ahead goes on from
     * there and not from the moves withdrawn.
     */
    @Test
    fun aTakebackToWhereTheGameBeganMeasuresItAgain() {
        val analyser = Analyser(GameAnnotator(QuietEngine(), depth = 6, withThreats = false))
        val assessing = Assessing(analyser)
        analyser.start()
        try {
            val start = Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
            assessing.submit(GameStarted(gameId = "g", userSide = Pieces.Side.WHITE, startFen = start.fen(), kind = "position"))
            val board = start.copy()
            for ((i, san) in listOf("Bc4", "Bc5").withIndex()) {
                val before = board.fen()
                val move = board.parseSan(san)!!
                board.push(move)
                assessing.submit(
                    MovePlayed(
                        gameId = "g", ply = i + 1, san = san, uci = move.uci(), fenBefore = before,
                        fenAfter = board.fen(), byUser = i == 0,
                    ),
                )
            }
            assessing.wait(5.0)
            assessing.submit(Takeback(gameId = "g", plies = 2, fen = start.fen()))
            val again = assessing.wait(5.0)
            assertEquals(listOf(start.fen()), again.filterIsInstance<Analysis>().filter { it.ply == 0 }.map { it.fen })
        } finally {
            assessing.close()
        }
    }

    /**
     * `test_a_position_re_read_mid_measure_gets_no_word_on_the_old_reading` (the state review,
     * 2026-10-01): the side to move settled by the clock while the joined position is measured
     * whole. The measure, given way to the re-read position's, was put off after the reset had
     * cleared what was put off, measured again after the new reading and delivered last, and on
     * the same game at ply 0 it passed as the new position's -- the profile filed White's
     * analysis for a position with Black to move. The engine lets go of what it is on at a
     * reset, and the glue passes on nothing of a position the game no longer stands on, nor of
     * another game.
     */
    @Test
    fun aPositionReReadMidMeasureGetsNoWordOnTheOldReading() {
        val white = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
        val black = white.replace(" w ", " b ")
        val analyser = Analyser(GameAnnotator(SlowEngine(300), depth = 18, withThreats = false))
        val assessing = Assessing(analyser)
        analyser.start()
        try {
            assessing.submit(GameStarted(gameId = "g", userSide = Pieces.Side.WHITE, startFen = white, kind = "position"))
            val deadline = System.currentTimeMillis() + 5000
            while (!analyser.looking && System.currentTimeMillis() < deadline) Thread.sleep(5)
            assertTrue("the full measure never began", analyser.looking)
            val before = assessing.drain()
            assessing.submit(Resync(gameId = "g", fen = black, reason = "side to move settled by the clock"))
            val after = assessing.wait(10.0)
            assertEquals(listOf(white), before.filterIsInstance<Analysis>().map { it.fen })
            val told = after.filterIsInstance<Analysis>().map { it.fen }
            assertTrue("$told", told.isNotEmpty() && told.toSet() == setOf(black))
        } finally {
            assessing.close()
        }
        // The glue's own check, whatever the engine sends: a word at ply 0 on the old reading,
        // and one on another game, are not passed on.
        val quiet = Analyser(GameAnnotator(QuietEngine(), depth = 6, withThreats = false))
        val glue = Assessing(quiet)
        glue.submit(Resync(gameId = "g", fen = black, reason = "re-read"))
        quiet.reset()  // what that submitted, gone: only the words put below are left to drain
        quiet.results.add(Analysis(gameId = "g", fen = white, ply = 0, lines = emptyList(), decisive = false, onlyMove = false))
        quiet.results.add(Analysis(gameId = "h", fen = black, ply = 0, lines = emptyList(), decisive = false, onlyMove = false))
        assertEquals(emptyList<Event>(), glue.drain())
        glue.close()
    }

    @Test
    fun whatTheEngineSaysOfAWithdrawnMoveIsNotPassedOn() {
        // The engine has measured four plies and nothing has been drained; then 3.Nf3 Nc6 are
        // taken back and 3.Bc4 played. What it said of the withdrawn plies stays behind.
        val analyser = Analyser(GameAnnotator(QuietEngine(), depth = 6, withThreats = false))
        val assessing = Assessing(analyser)
        analyser.start()
        assessing.submit(GameStarted(gameId = "g", userSide = Pieces.Side.WHITE, startFen = STARTING_FEN))
        val board = Board()
        for ((i, san) in "e4 e5 Nf3 Nc6".split(" ").withIndex()) assessing.submit(played(board, san, i + 1))
        val until = System.currentTimeMillis() + 5000
        while (analyser.pending != 0 && System.currentTimeMillis() < until) Thread.sleep(10)
        board.pop()
        board.pop()
        assessing.submit(Takeback(gameId = "g", plies = 2, fen = board.fen()))
        assessing.submit(played(board, "Bc4", 3))
        val got = ArrayList<Event>()
        val deadline = System.currentTimeMillis() + 5000
        while (System.currentTimeMillis() < deadline &&
            got.none { it is MoveAssessment && it.ply == 3 && it.uci == "f1c4" }
        ) {
            got += assessing.drain()
            Thread.sleep(10)
        }
        assessing.close()
        val verdicts = got.filterIsInstance<MoveAssessment>().map { it.ply to it.uci }
        assertEquals(listOf(1 to "e2e4", 2 to "e7e5", 3 to "f1c4"), verdicts)
        assertTrue(got.filterIsInstance<Analysis>().all { it.ply <= 3 })
    }

    /**
     * The recorded scenarios through the ported pipeline, the glue and the coach, the engine
     * a phone's speed behind: the reconciler, the glue and the coach count the same plies, and
     * nothing drained once a takeback is known is about a withdrawn move or position.
     *
     * The three of `test_ten_plies_taken_back_quickly` (stepwise-sliding, a-frame-each,
     * one-jump) as the desktop plays them: searches of [FAST_MS] through the game, the engine
     * waited for, then [SEARCH_MS] a search for the last two plies and the takebacks; the
     * replacement's verdict must come within 2.2 searches of the takeback being told (the rest
     * of the search under way, then its own), and the withdrawn position is never searched
     * after it (the parity audit, 2026-09-29: neither was checked here). The desktop times it
     * from the frame the replacement is first shown on; here from the takeback being told, so
     * the phone's time to read a frame is not counted against the engine.
     */
    @Test
    fun theScenariosLeaveTheGameTheGlueAndTheCoachAgreeing() {
        val context = InstrumentationRegistry.getInstrumentation().context
        val runs = JSONObject(context.assets.open("pipeline/index.json").bufferedReader().use { it.readText() })
            .getJSONArray("runs")
        val quickly = setOf("quick-takebacks-green", "takebacks-a-frame-each-green", "takeback-jump-green")
        val wanted = setOf("takeback-brown", "unexplained-green") + quickly
        val failures = ArrayList<String>()
        var ran = 0
        CellEmbedder.load(InstrumentationRegistry.getInstrumentation().targetContext).use { net ->
            for (r in 0 until runs.length()) {
                val run = runs.getJSONObject(r)
                val name = run.getString("name")
                if (name !in wanted) continue
                ran += 1
                val quick = name in quickly
                val engine = SlowEngine(if (quick) FAST_MS else 250L)
                val analyser = Analyser(GameAnnotator(engine, depth = 6, withThreats = false))
                analyser.start()
                val assessing = Assessing(analyser)
                val coach = Coach(OfflineProvider())
                val pipeline = GamePipeline(net)
                val gate = ChangeGate()
                val frames = run.getJSONArray("frames")
                val decoded = HashMap<String, Mat>()
                val events = ArrayList<Event>()
                val drained = ArrayList<Event>()
                var knownAt = -1 // the drain index when the last takeback became known
                fun drain(ts: Double) {
                    for (said in assessing.drain()) {
                        drained += said
                        coach.onEvent(said, ts)
                    }
                }
                // The frame Qb5+ is first shown on, the 21st ply (the desktop's `c.play("Qb5+",
                // "c6")` after `c.wait_for_engine()`): the state the python first told it on.
                val slowFrom = if (quick) firstShown(frames, "MovePlayed 21 ") else -1
                var withdrawn: String? = null // the position after ply 22, which the takeback withdraws
                var told = -1L // when the glue was told of the takeback
                var waited: Double? = null
                for (f in 0 until frames.length()) {
                    val frame = frames.getJSONObject(f)
                    val file = frame.getString("file")
                    val mat = decoded.getOrPut(file) { load(file) }
                    val ts = frame.getDouble("ts")
                    if (f == slowFrom) {
                        // `c.wait_for_engine()`, then a phone's speed from here on.
                        val until = System.currentTimeMillis() + 30_000
                        while (analyser.pending != 0 && System.currentTimeMillis() < until) {
                            drain(ts)
                            Thread.sleep(10)
                        }
                        engine.ms = SEARCH_MS
                    }
                    val pixels = FramePixels.of(mat)
                    gate.region = ChangeGate.boardGate(pipeline.board, pixels.width, pixels.height)
                    val changed = gate.changed(pixels, ts)
                    var replaced = false
                    for (event in pipeline.process(mat, ts, changed, f, pixels)) {
                        events += event
                        assessing.submit(event)
                        coach.onEvent(event, ts)
                        if (event is Takeback) {
                            knownAt = drained.size
                            if (quick) {
                                // Recorded from the moment the analyser is told (`armed`, `watching`).
                                engine.watching = true
                                told = SystemClock.elapsedRealtime()
                            }
                        }
                        if (event is MovePlayed && event.ply == 22) withdrawn = event.fenAfter.split(" ")[0]
                        if (event is MovePlayed && event.ply == 11 && event.uci == "e4e5" && told >= 0) replaced = true
                    }
                    drain(ts)
                    if (replaced && waited == null) {
                        // 6.e5 in place of 6.cxd4: its verdict, waited for as the desktop's test waits.
                        val deadline = told + 10_000
                        while (drained.none { it is MoveAssessment && it.ply == 11 && it.uci == "e4e5" } &&
                            SystemClock.elapsedRealtime() < deadline
                        ) {
                            drain(ts)
                            Thread.sleep(5)
                        }
                        waited = (SystemClock.elapsedRealtime() - told) / 1000.0
                    }
                    // The frames come at the capture's own pace, not all at once.
                    Thread.sleep(15)
                }
                val until = System.currentTimeMillis() + 60_000
                while (analyser.pending != 0 && System.currentTimeMillis() < until) {
                    drain(99_999.0)
                    Thread.sleep(20)
                }
                Thread.sleep(100)
                drain(99_999.0)
                assessing.close()
                for (mat in decoded.values) mat.release()

                val rec = pipeline.reconciler
                if (rec == null) {
                    failures += "$name: no game"
                    continue
                }
                val game = rec.board.moveStack.mapIndexed { i, m -> i + 1 to m.uci() }.toMap()
                val fens = HashMap<Int, String>()
                val walked = Board()
                for ((i, m) in rec.board.moveStack.withIndex()) {
                    walked.push(m)
                    fens[i + 1] = walked.boardFen()
                }
                if (assessing.playedLine != game) failures += "$name: the glue's game ${assessing.playedLine} is not $game"
                val coachGame = coach.moves.mapValues { it.value.uci }
                if (coachGame != game) failures += "$name: the coach's game $coachGame is not $game"
                if (events.any { it is Resync }) failures += "$name: a resync"
                if (knownAt >= 0) {
                    for (said in drained.drop(knownAt)) {
                        if (said is MoveAssessment && game[said.ply] != said.uci) {
                            failures += "$name: a verdict on a withdrawn move, ply ${said.ply} ${said.uci}"
                        } else if (said is Analysis && said.ply > 0 && fens[said.ply] != said.fen.split(" ")[0]) {
                            failures += "$name: an analysis of a withdrawn position, ply ${said.ply}"
                        }
                    }
                }
                val takebacks = events.filterIsInstance<Takeback>().map { it.plies }
                if (quick) {
                    val took = waited
                    if (took == null || took >= 2.2 * SEARCH_MS / 1000.0) {
                        failures += "$name: the replacement's verdict took ${took ?: "forever"} s " +
                            "(searches of $SEARCH_MS ms, the takebacks $takebacks)"
                    }
                    if (withdrawn == null) {
                        failures += "$name: no 22nd ply to withdraw"
                    } else if (withdrawn in synchronized(engine.asked) { engine.asked.toList() }) {
                        failures += "$name: the withdrawn position $withdrawn was searched after the takeback"
                    }
                }
                val notes = events.count { it is Unexplained }
                Log.i(
                    "sidecoach.state",
                    "$name: takebacks $takebacks, $notes unexplained, ${game.size} plies agreed, " +
                        "${drained.size} drained, ${analyser.withdrawn} withdrawn" +
                        (waited?.let { ", the replacement's verdict in ${"%.2f".format(it)} s" } ?: ""),
                )
            }
        }
        assertEquals("the takeback scenarios in the fixtures", 5, ran)
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /**
     * The first frame of the state the python first told [prefix] on: the frame it was told on,
     * and back while the frames before it show the same file.
     */
    private fun firstShown(frames: org.json.JSONArray, prefix: String): Int {
        var at = (0 until frames.length()).firstOrNull { f ->
            val told = frames.getJSONObject(f).optJSONArray("events") ?: return@firstOrNull false
            (0 until told.length()).any { told.getString(it).startsWith(prefix) }
        } ?: return -1
        val file = frames.getJSONObject(at).optString("file")
        while (at > 0 && frames.getJSONObject(at - 1).optString("file") == file) at -= 1
        return at
    }

    /**
     * The move list's badges follow the game: a takeback's withdrawn moves take theirs with
     * them -- a replacement at the same ply showed the withdrawn move's -- and a resync, which
     * numbers the plies again, clears them all
     * (`test_the_move_list_badges_follow_takebacks_and_resyncs`).
     */
    @Test
    fun theMoveListBadgesFollowTakebacksAndResyncs() {
        com.sidecoach.ui.PanelEngine.onGameEvent(
            GameStarted(gameId = "g", userSide = Pieces.Side.WHITE, startFen = STARTING_FEN),
        )
        val board = Board()
        for ((i, san) in "e4 e5 Nf3 Nc6".split(" ").withIndex()) {
            val move = played(board, san, i + 1)
            com.sidecoach.ui.PanelEngine.onGameEvent(move)
            com.sidecoach.ui.PanelEngine.onEngineEvents(
                listOf(
                    MoveAssessment(
                        gameId = "g", ply = i + 1, uci = move.uci, expectedLoss = 0.0,
                        severity = com.sidecoach.events.Severity.NONE, best = emptyList(),
                        label = com.sidecoach.events.Label.BEST,
                    ),
                ),
                null,
            )
        }
        fun labels(): Set<String> {
            val state = com.sidecoach.ui.PanelEngine.poll(0).getJSONObject("state")
            val map = state.optJSONObject("labels") ?: return emptySet()
            return map.keys().asSequence().toSet()
        }
        assertEquals(setOf("1", "2", "3", "4"), labels())
        board.pop()
        board.pop()
        com.sidecoach.ui.PanelEngine.onGameEvent(Takeback(gameId = "g", plies = 2, fen = board.fen()))
        assertEquals("the withdrawn moves' badges went", setOf("1", "2"), labels())
        com.sidecoach.ui.PanelEngine.onGameEvent(Resync(gameId = "g", fen = board.fen(), reason = "position adopted"))
        assertEquals("a resync numbers the plies again", emptySet<String>(), labels())
    }

    private fun load(file: String): Mat {
        val context = InstrumentationRegistry.getInstrumentation().context
        val bitmap = context.assets.open("pipeline/$file").use {
            android.graphics.BitmapFactory.decodeStream(it)
        }
        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        val bgr = Mat()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        rgba.release()
        bitmap.recycle()
        return bgr
    }

    companion object {
        /** A search through the game, as the desktop's `Chain` gives it (0.03 s). */
        private const val FAST_MS = 30L

        /** A search from the last two plies on, a phone's (`test_ten_plies_taken_back_quickly`'s 0.4 s). */
        private const val SEARCH_MS = 400L

        @JvmStatic
        @BeforeClass
        fun loadOpenCv() {
            assertTrue("OpenCV did not load", OpenCVLoader.initLocal())
        }
    }
}
