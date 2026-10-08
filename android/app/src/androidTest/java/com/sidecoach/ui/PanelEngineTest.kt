package com.sidecoach.ui

import android.content.Context
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.TestNets
import com.sidecoach.analysis.Analyser
import com.sidecoach.analysis.Assessing
import com.sidecoach.analysis.Engine
import com.sidecoach.analysis.GameAnnotator
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.coach.CoachRunner
import com.sidecoach.events.GameStarted
import com.sidecoach.profile.Profiler
import com.sidecoach.profile.Store
import com.sidecoach.vision.CellEmbedder
import com.sidecoach.vision.GameWatcher
import com.sidecoach.vision.Pieces
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import java.io.File

/**
 * The panel's engine, as `tests/test_panel_engine.py` and `tests/test_debug_log.py` hold the
 * desktop's: the loop following a game from frames and reporting it, the poll, the controls
 * with no session, the degraded reading, a correction, the dump, and the stages the warm-up
 * builds once, the voice and mute controls. What is word for word the desktop's (the lines, the
 * saves) is `PanelParityTest`.
 */
@RunWith(AndroidJUnit4::class)
class PanelEngineTest {

    private val app = InstrumentationRegistry.getInstrumentation().targetContext

    private fun state(): JSONObject = PanelEngine.poll(Int.MAX_VALUE).getJSONObject("state")

    /**
     * The loop over a rendered game (the pipeline fixtures' scholar's mate): the board locked,
     * the start position learned, the moves on the page -- the position from the game, the side
     * from its start -- and the status strip in the loop's own ticks, all of it polled over
     * HTTP. An ask pressed before the session is drained by it; listening is left as it was.
     */
    @Test
    fun theLoopFollowsAGameAndReportsIt() {
        assertTrue(OpenCVLoader.initLocal())
        val context = InstrumentationRegistry.getInstrumentation().context
        val run = JSONObject(context.assets.open("pipeline/index.json").bufferedReader().use { it.readText() })
            .getJSONArray("runs").let { runs -> (0 until runs.length()).map { runs.getJSONObject(it) } }
            .first { it.getString("name") == "mate-green" }
        PanelEngine.commands.clear()
        PanelEngine.set("running" to true, "listen" to "ptt")
        PanelEngine.command("ask", "hello")
        val cursor = PanelEngine.poll(Int.MAX_VALUE).getInt("cursor")
        CellEmbedder.load(app, threads = TestNets.THREADS).use { net ->
            val watcher = GameWatcher(app, null, net, Assessing(null, "off"), CoachRunner(SettingsStore(app), enabled = false), profilePath = null)
            val frames = run.getJSONArray("frames")
            val bitmaps = HashMap<String, android.graphics.Bitmap>()
            try {
                for (f in 0 until frames.length()) {
                    val frame = frames.getJSONObject(f)
                    val file = frame.getString("file")
                    val bitmap = bitmaps.getOrPut(file) {
                        context.assets.open("pipeline/$file").use { BitmapFactory.decodeStream(it) }
                    }
                    watcher.read(bitmap, frame.getDouble("ts"))
                    watcher.coachTick(frame.getDouble("ts"))
                }
            } finally {
                watcher.close()
                for (bitmap in bitmaps.values) bitmap.recycle()
            }
        }
        // As the page gets it: the poll over HTTP (`test_engine_follows_a_game_and_reports_it`).
        val server = PanelServerTest.serve(app)
        val polled = try {
            JSONObject(PanelServerTest.request(server.start(), "GET", "/api/poll?since=$cursor").second)
        } finally {
            server.stop()
        }
        PanelEngine.set("running" to false)
        val lines = polled.getJSONArray("lines").let { a -> (0 until a.length()).map { a.getJSONObject(it) } }
        val texts = lines.map { it.getString("text") }
        val locked = lines.firstOrNull { it.getString("text").startsWith("board locked at (") }
        assertNotNull("never reported locking the board: $texts", locked)
        assertTrue(locked!!.getJSONObject("data").has("region") && locked.getJSONObject("data").has("fit"))
        assertTrue(texts.toString(), texts.any { it.startsWith("start position seen, white at the bottom; theme learned, ") })
        assertTrue("no moves in the log", lines.any { it.getString("chan") == "game" && "(you" in it.getString("text") })
        assertTrue("the ask pressed before the session was not drained", "no coach to ask: hello" in texts)
        val state = polled.getJSONObject("state")
        assertEquals(run.getString("final_fen"), state.getString("fen"))
        assertEquals(7, state.getInt("ply"))
        assertEquals("white", state.getString("user_side"))
        assertEquals("white", state.getString("bottom"))
        assertTrue(state.getBoolean("following") && state.getBoolean("locked"))
        assertEquals("ptt", state.getString("listen"))
        // The mate is seen, and the game reaches the profile, kept in memory for this session as a
        // replay's is: its review (no engine, so no verdicts) on the page and in the coach's
        // stream, and the session summed up -- every time, not only when the end was seen
        // (`test_engine_follows_a_game_and_reports_it`, which the desktop's profile test folded
        // into in 3deedfd: its "game over" branch had never run; the parity audit, 2026-09-29).
        assertTrue(texts.toString(), texts.any { it.startsWith("game over 1-0") })
        assertTrue(!state.isNull("profile"))
        assertEquals(1, state.getJSONObject("profile").getInt("games"))
        val review = state.getJSONObject("review")
        assertEquals(review.toString(), "1-0" to "checkmate", review.getString("result") to review.getString("reason"))
        assertEquals("no engine: no move judged", 0, review.getInt("moves"))
        assertTrue(lines.any { it.getString("chan") == "coach" && it.getJSONObject("data").has("review") })
        assertTrue(texts.toString(), texts.any { it.startsWith("session:") })
        val health = state.getJSONObject("health")
        assertEquals("following", health.getString("stage"))
        assertTrue(health.toString(), health.getString("lock").startsWith("locked, fit "))
        assertTrue(health.toString(), health.getDouble("fps") > 0 && health.getJSONArray("pipeline_ms").length() == 2)
        assertTrue(health.toString(), health.has("lag_ms") && !health.isNull("lag_ms") && health.isNull("llm"))
    }

    /**
     * The session ends with a game whose end was never seen, its final position the third
     * repetition, which the sites draw: a draw by repetition, as a game the next one replaced
     * is (Tai, 2026-10-02), not a game left unfinished
     * (`test_a_session_ended_on_a_repetition_played_records_the_draw`; the frames are the
     * pipeline fixtures' repetition-green).
     */
    @Test
    fun aSessionEndedOnARepetitionPlayedRecordsTheDraw() {
        assertTrue(OpenCVLoader.initLocal())
        val context = InstrumentationRegistry.getInstrumentation().context
        val run = JSONObject(context.assets.open("pipeline/index.json").bufferedReader().use { it.readText() })
            .getJSONArray("runs").let { runs -> (0 until runs.length()).map { runs.getJSONObject(it) } }
            .first { it.getString("name") == "repetition-green" }
        val cursor = PanelEngine.poll(Int.MAX_VALUE).getInt("cursor")
        CellEmbedder.load(app, threads = TestNets.THREADS).use { net ->
            val watcher = GameWatcher(app, null, net, Assessing(null, "off"), CoachRunner(SettingsStore(app), enabled = false), profilePath = null)
            val frames = run.getJSONArray("frames")
            val bitmaps = HashMap<String, android.graphics.Bitmap>()
            try {
                for (f in 0 until frames.length()) {
                    val frame = frames.getJSONObject(f)
                    val file = frame.getString("file")
                    val bitmap = bitmaps.getOrPut(file) {
                        context.assets.open("pipeline/$file").use { BitmapFactory.decodeStream(it) }
                    }
                    watcher.read(bitmap, frame.getDouble("ts"))
                }
            } finally {
                watcher.close()
                for (bitmap in bitmaps.values) bitmap.recycle()
            }
        }
        val polled = PanelEngine.poll(cursor)
        val texts = polled.getJSONArray("lines").let { a -> (0 until a.length()).map { a.getJSONObject(it).getString("text") } }
        assertTrue(texts.toString(), "game over 1/2-1/2 (repetition)" in texts)
        val review = polled.getJSONObject("state").getJSONObject("review")
        assertEquals(review.toString(), "1/2-1/2" to "repetition", review.getString("result") to review.getString("reason"))
    }

    /**
     * The poll's keys (`test_controls_and_static_page` in tests/test_panel_server.py), its
     * records and cursor (`test_records_are_structured_and_polled_incrementally`) and the ring
     * (`test_ring_keeps_only_the_most_recent`, both in tests/test_debug_log.py).
     */
    @Test
    fun thePollIsIncrementalAndTheRingKeepsTheLast3000() {
        val first = PanelEngine.poll(0)
        assertEquals(setOf("state", "lines", "cursor"), first.keys().asSequence().toSet())
        JSONObject(first.toString()) // plain JSON through and through
        PanelEngine.emit("sys", "info", "hello", JSONObject().put("detail", 42))
        PanelEngine.emit("coach", "debug", "quiet: opponent to move")
        val got = PanelEngine.poll(first.getInt("cursor"))
        val lines = got.getJSONArray("lines")
        assertEquals(2, lines.length())
        val record = lines.getJSONObject(0)
        assertEquals(listOf("sys", "info", "hello"), listOf(record.getString("chan"), record.getString("level"), record.getString("text")))
        assertEquals(42, record.getJSONObject("data").getInt("detail"))
        assertEquals("a record with nothing underneath carries {}", 0, lines.getJSONObject(1).getJSONObject("data").length())
        assertTrue(record.getDouble("t") >= 0)
        assertEquals(0, PanelEngine.poll(got.getInt("cursor")).getJSONArray("lines").length())
        repeat(3005) { PanelEngine.emit("sys", "debug", "line $it") }
        val all = PanelEngine.poll(0).getJSONArray("lines")
        assertEquals(3000, all.length())
        assertEquals("line 3004", all.getJSONObject(2999).getString("text"))
    }

    /**
     * `test_a_game_ended_by_the_next_one_is_not_offered_to_end_game` (the state review,
     * 2026-10-01): a game whose end was not read, ended because the start position came back
     * and a move was played from it. The page said "End game… can set it", and End game ends
     * the game the reconciler holds -- the new one: its moves were refused from then on and the
     * record kept a one-move win. Only a game still followed is offered.
     */
    @Test
    fun aGameEndedByTheNextOneIsNotOfferedToEndGame() {
        fun clean(board: com.sidecoach.chess.Board): com.sidecoach.state.Readings = (0 until 64).map { i ->
            val piece = board.pieceAt(com.sidecoach.vision.Squares.squareAt(i / 8, i % 8, true))
            if (piece == null) {
                Pieces.CellReading(Pieces.Kind.EMPTY)
            } else {
                val side = if (piece.colour) Pieces.Side.WHITE else Pieces.Side.BLACK
                Pieces.CellReading(Pieces.Kind.PIECE, symbol = piece.symbol(), colour = side)
            }
        }.toMutableList()

        val rec = com.sidecoach.state.Reconciler(bottom = Pieces.Side.WHITE, userSide = Pieces.Side.WHITE)
        var ts = 0.0
        val board = com.sidecoach.chess.Board()
        val events = ArrayList<com.sidecoach.events.Event>()
        events += rec.update(clean(board), ts)
        // A game: GAME_MIN_PLIES half-moves (fewer, replaced unread, is no game at all).
        for (san in listOf("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5")) {
            board.push(board.parseSan(san)!!)
            repeat(2) { ts += 0.1; events += rec.update(clean(board), ts) }
        }
        val first = rec.gameId
        // The start position back, and a first move played from it: the next game.
        val next = com.sidecoach.chess.Board()
        repeat(2) { ts += 0.1; events += rec.update(clean(next), ts) }
        next.push(next.parseSan("d4")!!)
        repeat(2) { ts += 0.1; events += rec.update(clean(next), ts) }
        assertTrue("no next game", rec.gameId != first)
        val ended = events.filterIsInstance<com.sidecoach.events.GameEnded>()
        assertEquals(listOf(first to "*"), ended.map { it.gameId to it.result })

        val cursor = PanelEngine.poll(Int.MAX_VALUE).getInt("cursor")
        for (event in events) PanelEngine.onGameEvent(event, rec.gameId)
        val lines = PanelEngine.poll(cursor).getJSONArray("lines")
        val over = (0 until lines.length()).map { lines.getJSONObject(it).getString("text") }
            .filter { it.startsWith("game over") }
        assertEquals(listOf("game over (new game started); the result was not read"), over)
        // The next game, one move in, replaced unread: no game, and nothing said of its end
        // (Tai, 2026-10-05: the puzzles page's start board, ended "a game" of one move).
        val quiet = PanelEngine.poll(Int.MAX_VALUE).getInt("cursor")
        PanelEngine.onGameEvent(com.sidecoach.events.GameEnded(gameId = rec.gameId!!, result = "*", reason = "board reoriented"), null)
        val after = PanelEngine.poll(quiet).getJSONArray("lines")
        assertTrue((0 until after.length()).none { after.getJSONObject(it).getString("text").startsWith("game over") })
        // A game ended with its result unknown that is still the one followed can be set.
        PanelEngine.gamePlies = 40 // a game followed to its middle
        val kept = PanelEngine.poll(Int.MAX_VALUE).getInt("cursor")
        PanelEngine.onGameEvent(com.sidecoach.events.GameEnded(gameId = "k", result = "*", reason = "abandoned"), "k")
        val said = PanelEngine.poll(kept).getJSONArray("lines")
        assertTrue(said.getJSONObject(said.length() - 1).getString("text").endsWith("End game… can set it"))
    }

    /**
     * With no session running, Flip, Re-read and End game wait for the next Start (the
     * desktop's command queue: `test_with_no_session_flip_re_read_and_end_wait_for_the_next_start`),
     * and an ask is answered at once, not queued with them. The voice's controls with no session
     * are `thePanelStartsWithTheSettingsVoiceAndItsControlsTakeEffect`'s (and PanelServerTest's
     * over HTTP).
     */
    /**
     * A board the reader fails on is never left in silence (Tai, 2026-10-03): the long
     * [com.sidecoach.events.Unexplained] note is said in the coach's pane, once, and again only
     * after the board has been read since (a move); the early note stays in the log
     * (`test_a_board_lost_track_of_is_told_once_until_it_is_read_again`).
     */
    @Test
    fun aBoardLostTrackOfIsToldOnceUntilItIsReadAgain() {
        fun lost() = PanelEngine.all().count { it.text.startsWith("board · lost") }
        val before = lost()
        for (told in listOf(false, true, true)) {
            PanelEngine.onGameEvent(com.sidecoach.events.Unexplained("g", STARTING_FEN, "unexplained: x", told))
        }
        assertEquals(before + 1, lost())
        PanelEngine.onGameEvent(
            com.sidecoach.events.MovePlayed(
                gameId = "g", ply = 1, san = "e4", uci = "e2e4", fenBefore = STARTING_FEN,
                fenAfter = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", byUser = true,
            ),
        )
        PanelEngine.onGameEvent(com.sidecoach.events.Unexplained("g", STARTING_FEN, "unexplained: x", true))
        assertEquals(before + 2, lost())
        assertEquals(LOST_TRACK, PanelEngine.all().last { it.text.startsWith("board · lost") }.text.substringAfter("— "))
    }

    @Test
    fun controlsWithoutASessionWaitForTheNextStart() {
        PanelEngine.commands.clear()
        PanelEngine.set("running" to false)
        try {
            PanelEngine.command("flip")
            PanelEngine.command("analyze")
            PanelEngine.endGame("resigned")
            PanelEngine.ask("hint")
            assertEquals(listOf("flip" to null, "analyze" to null, "end" to "resigned"), PanelEngine.commands.toList())
        } finally {
            PanelEngine.commands.clear()
        }
    }

    /**
     * A picked-up rook puts a dot on a dozen squares, each too small to be a piece and read as
     * unknown; the first live game logged "reading degraded" on nearly every endgame move for
     * it. Only covered cells and unknowns that are piece-sized or hovered count.
     */
    @Test
    fun legalMoveDotsAreNotADegradedReading() {
        fun cell(kind: Pieces.Kind, fraction: Double = 0.0, hover: Boolean = false, covered: Boolean = false) =
            Pieces.CellReading(kind, fgFraction = fraction, hover = hover, covered = covered)
        val dots = MutableList(64) { cell(Pieces.Kind.EMPTY) }
        for (c in 0 until 8) dots[3 * 8 + c] = cell(Pieces.Kind.UNKNOWN, 0.08)
        dots[3 * 8] = cell(Pieces.Kind.UNKNOWN, 0.3, hover = true)
        var degraded = false
        repeat(PanelEngine.DEGRADED_FRAMES + 2) { degraded = PanelEngine.checkDegraded(dots, degraded) }
        assertFalse(degraded)
        // A dialog fading in covers cells; that is degraded.
        val dialog = MutableList(64) { cell(Pieces.Kind.EMPTY) }
        for (r in 0 until 3) for (c in 0 until 3) dialog[r * 8 + c] = cell(Pieces.Kind.UNKNOWN, 0.5, covered = true)
        val cursor = PanelEngine.poll(Int.MAX_VALUE).getInt("cursor")
        repeat(PanelEngine.DEGRADED_FRAMES + 2) { degraded = PanelEngine.checkDegraded(dialog, degraded) }
        assertTrue(degraded)
        degraded = PanelEngine.checkDegraded(dots, degraded)
        assertFalse(degraded)
        val texts = PanelEngine.poll(cursor).getJSONArray("lines").let { a -> (0 until a.length()).map { a.getJSONObject(it).getString("text") } }
        assertEquals(listOf("reading degraded: 9 cells unreadable (an overlay or a dialog?)", "reading clean again"), texts)
    }

    /** The user's word on a mistake's cause, through the review's card to the profile. */
    @Test
    fun aCorrectionMovesThroughTheApi() {
        val kept = PanelEngine.profiler
        PanelEngine.profiler = null
        try {
            assertFalse("no session, no profile", PanelEngine.correct("g", 0, 3, "time pressure"))
            val store = Store(null)
            val profiler = Profiler(store, sessionId = "s")
            profiler.onEvent(GameStarted("g", Pieces.Side.WHITE, STARTING_FEN), 0.0)
            store.addEvidence(7L, "g", 0, 3, "opening", 0.12, "positional misjudgment", null, false)
            PanelEngine.profiler = profiler
            PanelEngine.set(
                "review" to JSONObject().put("game_id", "g").put(
                    "turning_points",
                    JSONArray().put(JSONObject().put("segment", 0).put("ply", 3).put("cause", "positional misjudgment")),
                ),
            )
            assertTrue(PanelEngine.correct("g", 0, 3, "time pressure"))
            assertEquals("time pressure", store.evidence("g")[0].cause)
            val review = PanelEngine.poll(Int.MAX_VALUE).getJSONObject("state").getJSONObject("review")
            assertEquals("time pressure", review.getJSONArray("turning_points").getJSONObject(0).getString("cause"))
            assertTrue(runCatching { PanelEngine.correct("g", 0, 3, "bad luck") }.exceptionOrNull() is IllegalArgumentException)
            val cursor = PanelEngine.poll(Int.MAX_VALUE).getInt("cursor")
            assertTrue(PanelEngine.correct("g", 0, 3, null))
            val line = PanelEngine.poll(cursor).getJSONArray("lines").getJSONObject(0).getString("text")
            assertEquals("a cleared cause is Python's None", "you corrected the cause of the mistake at ply 3: None", line)
            store.close()
        } finally {
            PanelEngine.profiler = kept
            PanelEngine.set("review" to null, "profile" to null)
        }
    }

    /**
     * Debug > Dump: the frames the ring holds, at half scale and named by their time, and every
     * record of the log, written to a folder of their own -- the only way a session reaches the
     * disk.
     */
    @Test
    fun aDumpWritesTheFramesAndTheLog() {
        assertTrue(OpenCVLoader.initLocal())
        val frame = Mat(80, 100, CvType.CV_8UC3, Scalar(40.0, 120.0, 200.0))
        // The ring filled with this test's own frames, each its own time: the ring is the app's,
        // and the frames other tests' replays left in it share their fixtures' times -- two
        // frames of one time are one file, and the count told nothing of the dump (main's
        // repetition test, 2026-10-02: 32 held, 22 files). The ring keeps its last 150.
        repeat(150) { i -> PanelEngine.frames.push(12.3456 + i, frame) }
        frame.release()
        val root = File(app.cacheDir, "dump-test").apply { deleteRecursively() }
        val held = PanelEngine.frames.size
        val records = PanelEngine.all().size
        val themes = File(root, "themes-in").apply { mkdirs() }
        File(themes, "theme-aa-bb.bin").writeBytes(byteArrayOf(1, 2, 3))
        File(themes, "theme-aa-bb.bin.part").writeBytes(byteArrayOf(9))
        val folder = File(PanelEngine.dump(root, themes))
        try {
            // The learned sets travel with the frames; a half-written one does not.
            assertEquals(listOf("theme-aa-bb.bin"), File(folder, "themes").list()!!.toList())
            assertEquals(3L, File(folder, "themes/theme-aa-bb.bin").length())
            val jpegs = folder.listFiles { f -> f.name.endsWith(".jpg") }!!.map { it.name }
            assertEquals(held, jpegs.size)
            assertTrue(jpegs.toString(), "0000000012.346.jpg" in jpegs)
            val small = BitmapFactory.decodeFile(File(folder, "0000000012.346.jpg").path)
            assertEquals(50 to 40, small.width to small.height)
            assertEquals(records, File(folder, "debug.jsonl").readLines().size)
            val said = PanelEngine.all().last()
            assertEquals("dumped $held frames and the log to ${folder.path}", said.text)
        } finally {
            root.deleteRecursively()
        }
    }

    /**
     * The engine and the coach are built once and kept
     * (`test_the_engine_coach_and_listening_outlive_a_session`,
     * `test_a_start_without_a_warm_up_builds_the_stages_once`): a second Start, or a session come
     * and gone on them, finds the same ones, announced once.
     */
    @Test
    fun theStagesAreBuiltOnceAndOutliveASession() {
        val cursor = PanelEngine.poll(Int.MAX_VALUE).getInt("cursor")
        val (_, engine, coach) = PanelWarm.forSession(app)
        val watcher = GameWatcher(app, null, null, engine, coach, profilePath = null)
        watcher.close()
        val (_, again, recoached) = PanelWarm.forSession(app)
        assertTrue("a session must use what was built, and its end keep it", engine === again && coach === recoached)
        val texts = PanelEngine.poll(cursor).getJSONArray("lines").let { a -> (0 until a.length()).map { a.getJSONObject(it).getString("text") } }
        assertTrue(texts.toString(), texts.count { it.startsWith("engine: ") && !it.startsWith("engine: primed") } <= 1)
        assertTrue(texts.toString(), texts.count { it.startsWith("coach: ") } <= 1)
    }

    /** The desktop test's engine: no process, never asked anything here. */
    private class NoProcess : Engine(File("fake"), null) {
        override fun close() = Unit
    }

    /** An engine stage that is on, as `_Stage` is when it found Stockfish. */
    private fun onStage() = Assessing(Analyser(GameAnnotator(NoProcess())), "Stockfish, depth 18")

    /**
     * The stages from a fresh start with [makeEngine] building the engine's stage, [found] the
     * network and the opening tables in [openings], as the desktop's tests monkeypatch
     * `cli._Assessing` and `config.find_stockfish`; everything put back after.
     */
    private fun withStages(
        makeEngine: (android.content.Context, SettingsStore) -> Assessing,
        found: () -> File?,
        openings: File,
        body: () -> Unit,
    ) {
        val seams = Triple(PanelWarm.makeEngine, PanelWarm.findEngine, PanelWarm.openings)
        PanelWarm.makeEngine = makeEngine
        PanelWarm.findEngine = { found() }
        PanelWarm.openings = { openings }
        val kept = PanelWarm.keepStages(null, null)
        try {
            body()
        } finally {
            runCatching { PanelWarm.coaching?.close() }
            runCatching { PanelWarm.assessing?.close() }
            PanelWarm.keepStages(kept.first, kept.second)
            PanelWarm.makeEngine = seams.first
            PanelWarm.findEngine = seams.second
            PanelWarm.openings = seams.third
        }
    }

    /**
     * A first run warms up before Setup has fetched the engine's network; the stage built then
     * has no engine. Once Setup has put one in place, the next Start builds the engine -- and
     * the coach, which was built knowing there was none -- rather than waiting for a restart
     * (`test_an_engine_installed_after_the_warm_up_is_built_at_the_next_start`).
     */
    @Test
    fun anEngineInstalledAfterTheWarmUpIsBuiltAtTheNextStart() {
        val built = ArrayList<Boolean>()
        var network: File? = null
        val none = File(app.cacheDir, "no-openings")
        withStages(
            makeEngine = { _, _ ->
                val on = network != null
                built.add(on)
                if (on) onStage() else Assessing(null, "no Stockfish network yet (see Setup)")
            },
            found = { network },
            openings = none,
        ) {
            val (_, first, coaching) = PanelWarm.forSession(app)
            assertFalse(first.on)
            assertTrue("nothing new to find: kept", PanelWarm.forSession(app).second === first)
            network = File(app.filesDir, "nn-installed.nnue")
            val (_, second, recoached) = PanelWarm.forSession(app)
            assertTrue("the engine Setup installed is built", second !== first && second.on)
            assertTrue("the coach is built again with the engine", recoached !== coaching)
            assertTrue("and kept from then on", PanelWarm.forSession(app).second === second)
            assertEquals(listOf(false, true), built)
        }
    }

    /**
     * The engine is built at the warm-up; Setup may fetch the opening tables after it. The next
     * Start gives the kept annotator the book, without building anything again
     * (`test_opening_tables_fetched_after_the_warm_up_reach_the_kept_engine`).
     */
    @Test
    fun openingTablesFetchedAfterTheWarmUpReachTheKeptEngine() {
        val folder = File(app.cacheDir, "openings-after-warm-up").apply { deleteRecursively() }
        try {
            withStages(makeEngine = { _, _ -> onStage() }, found = { null }, openings = folder) {
                val (_, first, _) = PanelWarm.forSession(app)
                assertEquals("no tables yet", null, first.analyser!!.annotator.book)
                folder.mkdirs()
                InstrumentationRegistry.getInstrumentation().context.assets.open("openings_sample/c.tsv").use { input ->
                    File(folder, "c.tsv").outputStream().use { input.copyTo(it) }
                }
                val (_, again, _) = PanelWarm.forSession(app)
                assertTrue("the engine is kept, not built again", again === first)
                assertNotNull("the book the tables make", again.analyser!!.annotator.book)
            }
        } finally {
            folder.deleteRecursively()
        }
    }

    /**
     * The warm-up opens listening, and a session only reads the screen: one that ends -- its
     * loop closed, its service's hold on the microphone let go -- leaves listening as it was
     * (`test_the_engine_coach_and_listening_outlive_a_session`: after a Stop, push-to-talk did
     * nothing while the page still showed it on, 2026-09-23), and closing turns it off. The
     * phone's own rule rides on it: the microphone is open while the panel is in front or a
     * session holds it.
     */
    /**
     * `test_the_board_reading_model_missing_is_said_and_only_then`: a session with no
     * board-reading model says so in the status line, and keeps saying it (the loop has nothing
     * to read with); with the model, the usual "waiting for a board".
     */
    @Test
    fun theBoardReadingModelMissingIsSaidAndOnlyThen() {
        GameWatcher(app, null, null, Assessing(null, "off"), CoachRunner(SettingsStore(app), enabled = false), profilePath = null)
            .use { watcher ->
                assertEquals(NO_BOARD_MODEL, state().getString("status"))
                watcher.coachTick(1.0)
                assertEquals(NO_BOARD_MODEL, state().getString("status"))
            }
        CellEmbedder.load(app, threads = TestNets.THREADS).use { net ->
            GameWatcher(app, null, net, Assessing(null, "off"), CoachRunner(SettingsStore(app), enabled = false), profilePath = null)
                .use { assertEquals("waiting for a board", state().getString("status")) }
        }
    }

    @Test
    fun listeningOutlivesASession() {
        InstrumentationRegistry.getInstrumentation().uiAutomation
            .grantRuntimePermission(app.packageName, android.Manifest.permission.RECORD_AUDIO)
        PanelListen.build(app)
        val before = PanelListen.listener?.mode ?: "off"
        try {
            PanelListen.visible(true)
            assertEquals("ptt", PanelListen.setListen("ptt"))
            org.junit.Assume.assumeTrue("no microphone here: ${PanelListen.mic!!.detail}", PanelListen.mic!!.isOpen)
            PanelListen.visible(false)
            PanelListen.sessionHolds(true) // Start
            assertTrue("a session holds the microphone", PanelListen.mic!!.isOpen)
            GameWatcher(app, null, null, Assessing(null, "off"), CoachRunner(SettingsStore(app), enabled = false), profilePath = null)
                .close()
            PanelListen.sessionHolds(false) // Stop
            assertEquals("the session closed the ear", "ptt", PanelListen.listener!!.mode)
            assertEquals("ptt", state().getString("listen"))
            assertFalse("behind another app, with no session, nothing is heard", PanelListen.mic!!.isOpen)
            PanelListen.visible(true)
            assertTrue(PanelListen.mic!!.isOpen)
        } finally {
            PanelListen.visible(true)
            PanelListen.setListen("off")
        }
        assertFalse(PanelListen.mic!!.isOpen)
        PanelListen.setListen(before)
    }

    /**
     * `test_the_panel_starts_with_the_settings_voice`, and the voice and mute controls of
     * `test_controls_and_static_page` (tests/test_panel_server.py, where the desktop's
     * flip-and-voice test went in 3deedfd): the saved voice is the one the panel speaks with from
     * the start (the field's old default always won, 2026-09-21); the voice and mute controls
     * take effect and are shown.
     */
    @Test
    fun thePanelStartsWithTheSettingsVoiceAndItsControlsTakeEffect() {
        val prefs = app.getSharedPreferences("settings", Context.MODE_PRIVATE)
        val saved = prefs.getString("json", null)
        val voice = PanelVoice.voice
        try {
            val settings = SettingsStore(app)
            val values = settings.values()
            values.getJSONObject("voice").put("voice", "bm_george")
            settings.save(values)
            PanelVoice.build(app, SettingsStore(app))
            assertEquals("bm_george", PanelVoice.voice)
            assertEquals("bm_george", PanelVoice.tts!!.voice)
            assertEquals("bm_george", state().getString("voice"))
            PanelVoice.setVoice("am_adam")
            assertEquals("am_adam", PanelVoice.voice)
            assertEquals("am_adam", PanelVoice.tts!!.voice)
            assertEquals("am_adam", state().getString("voice"))
            PanelVoice.setMuted(true)
            assertTrue(PanelVoice.muted && state().getBoolean("muted"))
            PanelVoice.setMuted(false)
            assertFalse(PanelVoice.muted || state().getBoolean("muted"))
        } finally {
            if (saved == null) prefs.edit().remove("json").commit() else prefs.edit().putString("json", saved).commit()
            PanelVoice.setVoice(voice)
        }
    }

    /**
     * For Tai #11 (decided 2026-10-02): the card said 78.0 and Game Review 78.3. The card says
     * the live accuracy while the game's review is not done, then Game Review's, as the poll
     * carries it: read again when a review is saved (by any store: a review page's), and not
     * asked of the store between changes. The desktop's
     * test_the_end_of_game_card_says_the_reviews_accuracy_once_it_is_done.
     */
    @Test
    fun theEndOfGameCardSaysTheReviewsAccuracyOnceItIsDone() {
        val files = File(app.cacheDir, "card-accuracy").apply { deleteRecursively(); mkdirs() }
        val path = File(files, "profile.sqlite")
        val store = Store(path)
        val profiler = Profiler(store, sessionId = "s", ts = 0.0)
        val before = PanelEngine.profiler
        PanelEngine.profiler = profiler
        try {
            val board = com.sidecoach.chess.Board()
            profiler.onEvent(GameStarted("card", Pieces.Side.BLACK, STARTING_FEN), 1.0)
            val sans = listOf("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "d3", "Bc5", "Nc3", "d6", "h3", "Na5")
            for ((i, san) in sans.withIndex()) {
                val move = board.parseSan(san)!!
                val fen = board.fen()
                board.push(move)
                val played = com.sidecoach.events.MovePlayed("card", i + 1, san, move.uci(), fen, board.fen(), byUser = i % 2 == 1)
                profiler.onEvent(played, 2.0 + i)
            }
            val cursor = PanelEngine.poll(Int.MAX_VALUE).getInt("cursor")
            PanelEngine.profile(profiler, listOf(com.sidecoach.events.GameEnded("card", "0-1", "resignation")), flush = true)
            val lines = PanelEngine.poll(cursor).getJSONArray("lines")
            val card = (0 until lines.length()).map { lines.getJSONObject(it) }
                .first { it.optJSONObject("data")?.has("review") == true }
            assertEquals("watched:card", card.getJSONObject("data").getString("review_key"))
            assertEquals("not reviewed: the live accuracy", 0, state().getJSONObject("review_accuracy").length())

            val other = Store(path) // a review page's own store
            val summary = JSONObject().put("white", JSONObject().put("accuracy", 90.1)).put("black", JSONObject().put("accuracy", 78.3))
            assertTrue(other.saveReview("watched:card", "standard", ReviewRoom.REVIEW_VERSION, 1.0, "{}", summary.toString()))
            other.close()
            assertEquals("Game Review's", 78.3, state().getJSONObject("review_accuracy").getDouble("watched:card"), 0.0)
        } finally {
            PanelEngine.profiler = before
            store.close()
        }
    }
}
