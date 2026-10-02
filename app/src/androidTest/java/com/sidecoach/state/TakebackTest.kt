package com.sidecoach.state

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.analysis.Analyser
import com.sidecoach.analysis.AnalyserWatch
import com.sidecoach.analysis.Assessing
import com.sidecoach.analysis.EARLY_DEPTH
import com.sidecoach.analysis.Engine
import com.sidecoach.analysis.GameAnnotator
import com.sidecoach.analysis.HANG_MS
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
 *
 * No test here waits on a clock (Tai, 2026-10-01: a test is about logic, not timing): the
 * analyser's worker is watched through its own steps ([AnalyserWatch]), the fake engines' searches
 * run when the test lets them, and what a run did is counted from an ordered log. [HANG_MS]
 * bounds a hang and is never the expected path.
 */
@RunWith(AndroidJUnit4::class)
class TakebackTest {

    /** Every position answered with the same quiet line, as the desktop's FakeEngine. */
    private open class QuietEngine : Engine(File("quiet"), null) {
        /** Each position asked, with the depth asked for (`FakeEngine.calls`, `.depths`). */
        val calls: MutableList<Pair<String, Int?>> = java.util.Collections.synchronizedList(ArrayList())

        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
            calls.add(board.fen() to depth)
            return if (com.sidecoach.analysis.gameOver(board)) emptyList()
            else listOf(Line(moves = listOf("a2a3"), depth = 1, expectedScore = 0.5))
        }

        override fun close() = Unit
    }

    /** Its searches wait to be let go, so a test can act mid-search (`_HeldEngine`). */
    private class HeldEngine : QuietEngine() {
        val searching = CountDownLatch(1)
        val go = CountDownLatch(1)

        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
            searching.countDown()
            // A fake that timed out and carried on would pass a test that never let it go.
            if (!go.await(HANG_MS, TimeUnit.MILLISECONDS)) throw AssertionError("the test never let the search go")
            return super.analyse(board, depth, timeS, multipv, pvPlies, carryBest, onDepth)
        }
    }

    /**
     * `_GatedEngine`: searches run as the test lets them, never on a clock: answered at once while
     * the gate is open; after [hold], each search begun waits at the gate until [release] -- or
     * until it is interrupted, and then it stops, as a real one does ([Engine.interrupt]): none
     * begins once stopped, and one under way ends. ([Engine.interrupt] is not open: whoever
     * interrupts wakes the gate, [wake] -- the analyser's withdraw through its watch.)
     *
     * Every search begun is noted in [log], in order with what the analyser said; and the
     * positions begun while [watching] in [asked].
     */
    private class GatedEngine(private val log: MutableList<Pair<String, Any>>) : QuietEngine() {
        /** From the moment the analyser is told of a takeback: what it searched before was the game's. */
        @Volatile var watching = false
        private val asked = ArrayList<String>() // under [gate]
        private val ends = ArrayList<Pair<String, String>>() // each held search's position, and how it ended; under [gate]
        private val gate = Object()
        private var held = false
        var atGate = 0 // searches waiting at the gate now; under [gate]
            private set

        val isHeld: Boolean get() = synchronized(gate) { held }

        fun hold() = synchronized(gate) { held = true }

        fun release() = synchronized(gate) {
            held = false
            gate.notifyAll()
        }

        fun wake() = synchronized(gate) { gate.notifyAll() }

        fun askedNow(): List<String> = synchronized(gate) { ArrayList(asked) }

        private fun waitFor(what: String, condition: () -> Boolean) {
            val deadline = System.nanoTime() + HANG_MS * 1_000_000
            while (!condition()) {
                val left = (deadline - System.nanoTime()) / 1_000_000
                if (left <= 0) throw AssertionError(what)
                gate.wait(left)
            }
        }

        /** Until a search is held at the gate: the engine is busy on it. */
        fun waitAtGate() = synchronized(gate) { waitFor("no search came") { atGate > 0 } }

        /** Until [count] held searches have ended (let go or interrupted); how they ended. */
        fun waitEnded(count: Int): List<Pair<String, String>> = synchronized(gate) {
            waitFor("held searches ended: $ends") { ends.size >= count }
            ArrayList(ends)
        }

        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
            fun stopped() = interruptibleNow && stopped
            if (stopped()) throw Interrupted()
            synchronized(gate) {
                log.add("search" to board.boardFen())
                if (watching) asked.add(board.boardFen())
                if (held) {
                    atGate += 1
                    gate.notifyAll()
                    try {
                        waitFor("a held search never let go") { !held || stopped() }
                    } finally {
                        atGate -= 1
                    }
                    ends.add(board.boardFen() to if (stopped()) "interrupted" else "answered")
                    gate.notifyAll()
                }
                if (stopped()) throw Interrupted()
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
        val watch = AnalyserWatch(analyser)
        val board = Board()
        val moves = "e4 e5 Nf3 Nc6".split(" ").mapIndexed { i, san ->
            val before = board.copy()
            val move = board.parseSan(san)!!
            board.push(move)
            Triple(before, move, i + 1)
        }
        analyser.submitMove("g", moves[0].first, moves[0].second, 1)
        analyser.start()
        assertTrue("the worker never started on ply 1", engine.searching.await(HANG_MS, TimeUnit.MILLISECONDS))
        for ((before, move, ply) in moves.drop(1)) analyser.submitMove("g", before, move, ply)
        // Plies 3 and 4 taken back while ply 1 is measured and 2 waits.
        analyser.withdraw("g", 2)
        assertEquals(2, analyser.withdrawn)
        assertEquals(2, analyser.pending)
        // Ply 1 taken back too, mid-search: its verdict is not delivered.
        analyser.withdraw("g", 0)
        assertEquals(4, analyser.withdrawn)
        engine.go.countDown()
        // Nothing is delivered of ply 1, so wait for the work to be settled, not for words: a word
        // is put before its request settles, so none can come after `pending` is 0.
        val got = watch.settled()
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
        val watch = AnalyserWatch(analyser)
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
            // Everything owed answered (signalled by the worker), then what it said to the coach.
            watch.idle()
            drained += assessing.drain()
            for (event in drained) coach.onEvent(event, 1.0)
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
        val watch = AnalyserWatch(analyser)
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
            watch.idle() // both moves answered (signalled by the worker, not polled)
            assessing.drain()
            assessing.submit(Takeback(gameId = "g", plies = 2, fen = start.fen()))
            watch.idle()
            val again = assessing.drain()
            assertEquals(listOf(start.fen()), again.filterIsInstance<Analysis>().filter { it.ply == 0 }.map { it.fen })
        } finally {
            assessing.close()
        }
    }

    /**
     * `_WholeMeasureHeld`: answers at once, except the first whole measure (the full [depth]) of
     * [fen]: that search runs until it is interrupted, as a real one stops ([Engine.interrupt]),
     * so the test acts while it is under way, never on a clock. Whoever interrupts it [wake]s it.
     */
    private class WholeMeasureHeld(private val fen: String, private val depth: Int) : QuietEngine() {
        val holding = CountDownLatch(1) // the held search is under way
        val interrupted = CountDownLatch(1) // and was stopped
        private val lock = Object()

        fun wake() = synchronized(lock) { lock.notifyAll() }

        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
            val whole = depth == this.depth
            if (interruptibleNow && whole && board.fen() == fen && holding.count > 0) {
                synchronized(lock) {
                    holding.countDown()
                    val deadline = System.nanoTime() + HANG_MS * 1_000_000
                    while (!stopped) {
                        val left = (deadline - System.nanoTime()) / 1_000_000
                        if (left <= 0) throw AssertionError("the whole measure was never let go")
                        lock.wait(left)
                    }
                }
                interrupted.countDown()
                throw Interrupted()
            }
            return super.analyse(board, depth, timeS, multipv, pvPlies, carryBest, onDepth)
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
        val engine = WholeMeasureHeld(white, 18)
        val analyser = Analyser(GameAnnotator(engine, depth = 18, withThreats = false))
        val watch = AnalyserWatch(analyser)
        val assessing = Assessing(analyser)
        analyser.start()
        try {
            assessing.submit(GameStarted(gameId = "g", userSide = Pieces.Side.WHITE, startFen = white, kind = "position"))
            assertTrue("the full measure never began", engine.holding.await(HANG_MS, TimeUnit.MILLISECONDS))
            assertTrue("the whole measure, under way", analyser.looking)
            val before = assessing.drain()
            assessing.submit(Resync(gameId = "g", fen = black, reason = "side to move settled by the clock"))
            engine.wake() // the reset interrupted it before `submit` returned
            assertTrue("the reset let go of the measure under way", engine.interrupted.await(HANG_MS, TimeUnit.MILLISECONDS))
            watch.idle()
            val after = assessing.drain()
            assertEquals(listOf(white), before.filterIsInstance<Analysis>().map { it.fen })
            // The old reading's whole measure was never taken up again after the reset: its only
            // search that ran to an answer is the early one.
            val whiteDepths = synchronized(engine.calls) { engine.calls.filter { it.first == white }.map { it.second } }
            assertEquals(listOf<Int?>(EARLY_DEPTH), whiteDepths)
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
        val watch = AnalyserWatch(analyser)
        val assessing = Assessing(analyser)
        analyser.start()
        assessing.submit(GameStarted(gameId = "g", userSide = Pieces.Side.WHITE, startFen = STARTING_FEN))
        val board = Board()
        for ((i, san) in "e4 e5 Nf3 Nc6".split(" ").withIndex()) assessing.submit(played(board, san, i + 1))
        watch.idle() // all four measured, nothing drained
        board.pop()
        board.pop()
        assessing.submit(Takeback(gameId = "g", plies = 2, fen = board.fen()))
        assessing.submit(played(board, "Bc4", 3))
        // Every request answered; then the glue holds what came against the game as it stands.
        watch.idle()
        val got = assessing.drain()
        assessing.close()
        val verdicts = got.filterIsInstance<MoveAssessment>().map { it.ply to it.uci }
        assertEquals(listOf(1 to "e2e4", 2 to "e7e5", 3 to "f1c4"), verdicts)
        assertTrue(got.filterIsInstance<Analysis>().all { it.ply <= 3 })
    }

    /**
     * The recorded scenarios through the ported pipeline, the glue and the coach: the reconciler,
     * the glue and the coach count the same plies, and nothing drained once a takeback is known
     * is about a withdrawn move or position.
     *
     * As `test_ten_plies_taken_back_quickly` plays them, for every scenario: the engine keeps up
     * with the game -- each move's work settled before the next frame -- until the last two
     * plies before the takeback (Qb5+ c6 in the three quick ones), which it is held on, so it is
     * on the withdrawn moves when the takeback comes. "Prompt" is counted, not timed: every
     * search begun is logged in order with what the analyser says and each takeback it is told
     * of. The search under way when the takeback came is let go (interrupted, not run to its
     * end) and was of a withdrawn line; between the takeback and the replacement's verdict only
     * the replacement's own positions are searched, at most the two a verdict needs, and nothing
     * is said; no withdrawn position is searched from the takeback on (the parity audit,
     * 2026-09-29). The phone's test used to time it instead: searches that slept, 15 ms between
     * frames, and the verdict within 2.2 searches of the takeback by the stopwatch.
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
                failures += scenario(name, run.getJSONArray("frames"), net)
            }
        }
        assertEquals("the takeback scenarios in the fixtures", 5, ran)
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /** One recorded scenario played through ([theScenariosLeaveTheGameTheGlueAndTheCoachAgreeing]); what failed. */
    private fun scenario(name: String, frames: org.json.JSONArray, net: CellEmbedder): List<String> {
        val failures = ArrayList<String>()
        // In order: every search begun, every word the analyser said, and each takeback it was told of.
        val log: MutableList<Pair<String, Any>> = java.util.Collections.synchronizedList(ArrayList())
        val engine = GatedEngine(log)
        val analyser = Analyser(GameAnnotator(engine, depth = 6, withThreats = false))
        val watch = AnalyserWatch(
            analyser,
            object : Analyser.Watch {
                override fun said(event: Event) {
                    log.add("said" to event)
                }

                override fun withdrew(gameId: String, ply: Int) {
                    // After the analyser has let go: whatever begins after this is not of a
                    // withdrawn move (a search under way had begun before). Its interrupt is in:
                    // the gate is woken to see it.
                    log.add("takeback" to ply)
                    engine.watching = true
                    engine.wake()
                }
            },
        )
        analyser.start()
        val assessing = Assessing(analyser)
        val coach = Coach(OfflineProvider())
        val pipeline = GamePipeline(net)
        val gate = ChangeGate()
        val decoded = HashMap<String, Mat>()
        val letGo = lastUses(frames)
        val events = ArrayList<Event>()
        val drained = ArrayList<Event>()
        var knownAt = -1 // the drain index when the last takeback became known
        fun drain(ts: Double) {
            for (said in assessing.drain()) {
                drained += said
                coach.onEvent(said, ts)
            }
        }
        // The two plies before the takeback, which the engine is held on (`c.engine.hold()`,
        // `c.play("Qb5+", "c6")`): held from the frame the first of them is first shown on.
        val takebackAt = (0 until frames.length()).first { f -> told(frames, f).any { it.startsWith("Takeback ") } }
        val heldPlies = (0 until takebackAt).flatMap { f -> told(frames, f).filter { it.startsWith("MovePlayed ") } }
            .map { it.split(" ")[1].toInt() }
            .takeLast(2)
        val holdFrom = firstShown(frames, "MovePlayed ${heldPlies[0]} ")
        var withdrawn: String? = null // the position after the last held ply, which the takeback withdraws
        val line = HashMap<Int, String>() // each ply's position as the moves were told
        var oldLine: Map<Int, String> = emptyMap() // the game's positions when the takeback came
        var replaced = false
        try {
            for (f in 0 until frames.length()) {
                val frame = frames.getJSONObject(f)
                val file = frame.getString("file")
                val mat = decoded.getOrPut(file) { load(file) }
                val ts = frame.getDouble("ts")
                if (f == holdFrom) {
                    watch.idle() // `c.wait_for_engine()`
                    engine.hold()
                }
                val pixels = FramePixels.of(mat)
                gate.region = ChangeGate.boardGate(pipeline.board, pixels.width, pixels.height)
                val changed = gate.changed(pixels, ts)
                var moved = false
                var bothHeld = false
                var replacement: MovePlayed? = null
                for (event in pipeline.process(mat, ts, changed, f, pixels)) {
                    events += event
                    if (event is Takeback) {
                        // What the engine said before this was said of the game as it stood.
                        knownAt = drained.size
                        oldLine = HashMap(line)
                    }
                    assessing.submit(event)
                    coach.onEvent(event, ts)
                    if (event is MovePlayed) {
                        moved = true
                        line[event.ply] = event.fenAfter.split(" ")[0]
                        if (knownAt < 0 && event.ply == heldPlies.last()) {
                            withdrawn = event.fenAfter.split(" ")[0]
                            bothHeld = true
                        }
                        if (knownAt >= 0 && !replaced && replacement == null) replacement = event
                    }
                }
                if (bothHeld) {
                    // Both held plies submitted: the engine is on the first, the second waits.
                    engine.waitAtGate()
                    if (analyser.pending != 2) failures += "$name: ${analyser.pending} pending on the held plies, not 2"
                }
                if (replacement != null) {
                    replaced = true
                    failures += counted(name, engine, log, watch, replacement, oldLine, drained, ::drain, ts)
                } else if (moved && !engine.isHeld) {
                    watch.idle() // the engine keeps up: this move's work done
                }
                drain(ts)
                letGo[f]?.forEach { decoded.remove(it)?.release() }
            }
            watch.idle()
            drain(99_999.0)
        } finally {
            engine.release()
            assessing.close()
            for (mat in decoded.values) mat.release()
        }

        val rec = pipeline.reconciler ?: return failures + "$name: no game"
        val game = rec.board.moveStack.mapIndexed { i, m -> i + 1 to m.uci() }.toMap()
        val fens = fensOf(rec.board)
        if (assessing.playedLine != game) failures += "$name: the glue's game ${assessing.playedLine} is not $game"
        val coachGame = coach.moves.mapValues { it.value.uci }
        if (coachGame != game) failures += "$name: the coach's game $coachGame is not $game"
        if (events.any { it is Resync }) failures += "$name: a resync"
        if (!replaced) failures += "$name: no move replaced the withdrawn ones"
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
        if (takebacks.size != 1) failures += "$name: the takebacks $takebacks, not one"
        val toldAt = synchronized(log) { log.indices.filter { log[it].first == "takeback" } }
        if (toldAt.size != 1) failures += "$name: the analyser told of ${toldAt.size} takebacks"
        val asked = engine.askedNow()
        if (asked.isEmpty()) failures += "$name: nothing asked once the takeback was told"
        if (withdrawn == null) {
            failures += "$name: no ply ${heldPlies.last()} to withdraw"
        } else if (withdrawn in asked) {
            failures += "$name: the withdrawn position $withdrawn was searched after the takeback"
        }
        // Nothing withdrawn searched from the takeback on, to the end of the game.
        if (toldAt.isNotEmpty()) {
            val back = (log[toldAt[0]].second as Int)
            val gone = oldLine.filterKeys { it > back }.values.toSet()
            val after = synchronized(log) { log.drop(toldAt[0] + 1).filter { it.first == "search" }.map { it.second as String } }.toSet()
            if ((gone intersect after).isNotEmpty()) failures += "$name: withdrawn positions searched after the takeback: ${gone intersect after}"
        }
        val notes = events.count { it is Unexplained }
        Log.i(
            "sidecoach.state",
            "$name: takebacks $takebacks, $notes unexplained, ${game.size} plies agreed, " +
                "${drained.size} drained, ${analyser.withdrawn} withdrawn, ${analyser.skipped} skipped, " +
                "${synchronized(log) { log.count { it.first == "search" } }} searches",
        )
        return failures
    }

    /**
     * The replacement played after the takeback, [replacement]: the search under way when the
     * takeback came was interrupted and was of the line withdrawn; then the engine let go and
     * waited for; between the takeback and the replacement's verdict, only its own positions
     * searched (one or two) and nothing said.
     */
    private fun counted(
        name: String,
        engine: GatedEngine,
        log: MutableList<Pair<String, Any>>,
        watch: AnalyserWatch,
        replacement: MovePlayed,
        oldLine: Map<Int, String>,
        drained: List<Event>,
        drain: (Double) -> Unit,
        ts: Double,
    ): List<String> {
        val failures = ArrayList<String>()
        val first = engine.waitEnded(1)[0]
        if (first.second != "interrupted" || first.first !in oldLine.values) {
            failures += "$name: the search under way at the takeback $first was not let go, or not of the old line"
        }
        engine.release()
        watch.idle()
        drain(ts)
        val entries = synchronized(log) { log.toList() }
        val told = entries.indexOfFirst { it.first == "takeback" }
        val verdict = entries.indices.firstOrNull { i ->
            i > told && entries[i].first == "said" && (entries[i].second as? MoveAssessment)?.ply == replacement.ply
        }
        if (verdict == null) return failures + "$name: the replacement was never judged"
        val judged = entries[verdict].second as MoveAssessment
        if (judged.uci != replacement.uci) failures += "$name: the verdict at ply ${replacement.ply} is on ${judged.uci}, not ${replacement.uci}"
        if (drained.none { it is MoveAssessment && it.ply == replacement.ply && it.uci == replacement.uci }) {
            failures += "$name: the replacement's verdict never reached the coach"
        }
        val between = entries.subList(told + 1, verdict)
        val searched = between.filter { it.first == "search" }.map { it.second as String }
        val own = setOf(replacement.fenBefore.split(" ")[0], replacement.fenAfter.split(" ")[0])
        if (searched.size !in 1..2 || !own.containsAll(searched)) {
            failures += "$name: between the takeback and the replacement's verdict: $searched (its own: $own)"
        }
        // Nothing said before the verdict -- but, for the opponent's move, the early word on the
        // position the player then faces, its own (`Analyser.serve`: "measured early first").
        val words = between.filter { it.first == "said" }.map { it.second as Event }
        val early = words.filter {
            !replacement.byUser && it is Analysis && it.provisional && it.ply == replacement.ply &&
                it.fen.split(" ").take(4) == replacement.fenAfter.split(" ").take(4)
        }
        if (words.size != early.size || early.size > 1) {
            failures += "$name: a word before the verdict: " + words.map { e ->
                when (e) {
                    is Analysis -> "Analysis(ply ${e.ply}, ${e.fen}, provisional ${e.provisional})"
                    is MoveAssessment -> "MoveAssessment(ply ${e.ply}, ${e.uci})"
                    else -> e.toString()
                }
            }
        }
        return failures
    }

    private fun told(frames: org.json.JSONArray, f: Int): List<String> {
        val events = frames.getJSONObject(f).optJSONArray("events") ?: return emptyList()
        return (0 until events.length()).map { events.getString(it) }
    }

    /** Each ply's position (board part) along [board]'s moves. */
    private fun fensOf(board: Board): Map<Int, String> {
        val walked = Board()
        val out = HashMap<Int, String>()
        for ((i, m) in board.moveStack.withIndex()) {
            walked.push(m)
            out[i + 1] = walked.boardFen()
        }
        return out
    }

    /**
     * The first frame of the state the python first told [prefix] on: the frame it was told on,
     * and back while the frames before it show the same file.
     */
    private fun firstShown(frames: org.json.JSONArray, prefix: String): Int {
        var at = (0 until frames.length()).firstOrNull { f -> told(frames, f).any { it.startsWith(prefix) } } ?: return -1
        val file = frames.getJSONObject(at).optString("file")
        while (at > 0 && frames.getJSONObject(at - 1).optString("file") == file) at -= 1
        return at
    }

    /**
     * The files to let go after each frame: a file after the last frame that shows it (as
     * PipelineTest's `lastUses`), so a scenario's decoded frames are not all held to its end.
     */
    private fun lastUses(frames: org.json.JSONArray): Map<Int, List<String>> {
        val last = HashMap<String, Int>()
        for (f in 0 until frames.length()) last[frames.getJSONObject(f).getString("file")] = f
        return last.entries.groupBy({ it.value }, { it.key })
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
        @JvmStatic
        @BeforeClass
        fun loadOpenCv() {
            assertTrue("OpenCV did not load", OpenCVLoader.initLocal())
        }
    }
}
