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
 * taken back quickly with the pieces sliding, all at once, and a board left unexplained for
 * long before eleven are taken back -- run through the ported pipeline, the glue and the
 * coach, which must count the same plies as the reconciler throughout.
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
     */
    private class SlowEngine(private val ms: Long) : QuietEngine() {
        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
            if (interruptibleNow && stopped) throw Interrupted()
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
     */
    @Test
    fun theScenariosLeaveTheGameTheGlueAndTheCoachAgreeing() {
        val context = InstrumentationRegistry.getInstrumentation().context
        val runs = JSONObject(context.assets.open("pipeline/index.json").bufferedReader().use { it.readText() })
            .getJSONArray("runs")
        val wanted = setOf("takeback-brown", "quick-takebacks-green", "takeback-jump-green", "unexplained-green")
        val failures = ArrayList<String>()
        var ran = 0
        CellEmbedder.load(InstrumentationRegistry.getInstrumentation().targetContext).use { net ->
            for (r in 0 until runs.length()) {
                val run = runs.getJSONObject(r)
                val name = run.getString("name")
                if (name !in wanted) continue
                ran += 1
                val engine = SlowEngine(250)
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
                for (f in 0 until frames.length()) {
                    val frame = frames.getJSONObject(f)
                    val file = frame.getString("file")
                    val mat = decoded.getOrPut(file) { load(file) }
                    val ts = frame.getDouble("ts")
                    val pixels = FramePixels.of(mat)
                    gate.region = ChangeGate.boardGate(pipeline.board, pixels.width, pixels.height)
                    val changed = gate.changed(pixels, ts)
                    for (event in pipeline.process(mat, ts, changed, f, pixels)) {
                        events += event
                        assessing.submit(event)
                        coach.onEvent(event, ts)
                        if (event is Takeback) knownAt = drained.size
                    }
                    drain(ts)
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
                val notes = events.count { it is Unexplained }
                Log.i(
                    "sidecoach.state",
                    "$name: takebacks $takebacks, $notes unexplained, ${game.size} plies agreed, " +
                        "${drained.size} drained, ${analyser.withdrawn} withdrawn",
                )
            }
        }
        assertEquals("the takeback scenarios in the fixtures", 4, ran)
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /**
     * The move list's badges follow the game: a takeback's withdrawn moves take theirs with
     * them -- a replacement at the same ply showed the withdrawn move's -- and a resync, which
     * numbers the plies again, clears them all (`test_the_move_list_badges_follow_...`).
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
