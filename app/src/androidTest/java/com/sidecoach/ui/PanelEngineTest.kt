package com.sidecoach.ui

import android.content.Context
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
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
        CellEmbedder.load(app).use { net ->
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
        // As the page gets it: the poll over HTTP (`test_poll_reports_a_game`).
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
        // The game reaches the profile, kept in memory for this session as a replay's is: its
        // review (no engine, so no verdicts) on the page and in the coach's stream, and the
        // session summed up (`test_the_profile_records_the_game_and_the_review_reaches_the_page`).
        assertTrue(!state.isNull("profile"))
        if (texts.any { "game over" in it }) {
            assertEquals(0, state.getJSONObject("review").getInt("moves"))
            assertTrue(lines.any { it.getString("chan") == "coach" && it.getJSONObject("data").has("review") })
            assertTrue(texts.toString(), texts.any { it.startsWith("session:") })
        }
        val health = state.getJSONObject("health")
        assertEquals("following", health.getString("stage"))
        assertTrue(health.toString(), health.getString("lock").startsWith("locked, fit "))
        assertTrue(health.toString(), health.getDouble("fps") > 0 && health.getJSONArray("pipeline_ms").length() == 2)
        assertTrue(health.toString(), health.has("lag_ms") && !health.isNull("lag_ms") && health.isNull("llm"))
    }

    /** `test_api_poll_is_incremental_and_serialisable`, with the ring of `test_debug_log`. */
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
     * With no session running, Flip, Re-read and End game wait for the next Start (the
     * desktop's command queue), an ask is answered at once, and the voice's controls take.
     */
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
            PanelVoice.setVoice("am_adam")
            assertEquals("am_adam", PanelVoice.voice)
            PanelVoice.setMuted(true)
            assertTrue(PanelVoice.muted)
        } finally {
            PanelVoice.setMuted(false)
            PanelVoice.setVoice(SettingsStore(app).voice())
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
        PanelEngine.frames.push(12.3456, frame)
        frame.release()
        val root = File(app.cacheDir, "dump-test").apply { deleteRecursively() }
        val held = PanelEngine.frames.size
        val records = PanelEngine.all().size
        val folder = File(PanelEngine.dump(root))
        try {
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
     * The engine and the coach are built once and kept (`test_the_engine_and_coach_outlive_a_
     * session`, `..._a_start_without_a_warm_up_builds_the_stages_once`): a second Start, or a
     * session come and gone on them, finds the same ones, announced once.
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
     * (`test_listening_outlives_a_session`: after a Stop, push-to-talk did nothing while the
     * page still showed it on, 2026-09-23), and closing turns it off. The phone's own rule rides
     * on it: the microphone is open while the panel is in front or a session holds it.
     */
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
     * The saved voice is the one the panel speaks with from the start (Tai, 2026-09-21: the
     * settings said bm_george, the panel came up with af_sarah).
     */
    @Test
    fun thePanelStartsWithTheSettingsVoice() {
        val store = SettingsStore(app)
        val saved = store.values()
        try {
            store.save(store.values().also { it.getJSONObject("voice").put("voice", "bm_george") })
            PanelVoice.build(app, store)
            assertEquals("bm_george", PanelVoice.voice)
            assertEquals("bm_george", PanelEngine.poll(Int.MAX_VALUE).getJSONObject("state").getString("voice"))
        } finally {
            store.save(saved)
            PanelVoice.build(app, store)
        }
    }

    /**
     * `test_flip_and_voice_controls_do_not_raise_without_a_game` and
     * `test_the_panel_starts_with_the_settings_voice`: the saved voice is the one the panel
     * speaks with from the start (the field's old default always won, 2026-09-21); the voice
     * and mute controls take effect and are shown.
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
}
