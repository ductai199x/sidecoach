package com.sidecoach.state

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.Lane
import com.sidecoach.Parallel
import com.sidecoach.events.BoardUnread
import com.sidecoach.events.Event
import com.sidecoach.events.GameEnded
import com.sidecoach.events.GameStarted
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.PositionChanged
import com.sidecoach.events.Resync
import com.sidecoach.events.ScreenStateChanged
import com.sidecoach.events.Takeback
import com.sidecoach.events.Unexplained
import com.sidecoach.vision.BoardGeometry
import com.sidecoach.vision.CellEmbedder
import com.sidecoach.vision.ChangeGate
import com.sidecoach.vision.Coordinates
import com.sidecoach.vision.FramePixels
import com.sidecoach.vision.Pieces
import com.sidecoach.vision.ThemeStore
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

/**
 * The whole pipeline held to the desktop's, frame for frame.
 *
 * `tools/make_pipeline_fixtures.py` renders games with the desktop's own renderer, writes the
 * frames out as PNGs, and runs the desktop's `GamePipeline` -- with its change gate on the
 * board's own region -- over exactly those pixels, recording what it said about each frame.
 * The phone reads the same PNGs through the ported pipeline and must agree: on which frames
 * counted as changed, and on every event, in order.
 *
 * Three games: the scholar's mate, which the board itself ends and the site's dialog then
 * covers; a game that ends in a resignation, which only the site's badges on the two kings
 * can end; and a game browsed back two plies and played on, which the record tells as a
 * takeback.
 *
 * A frame may carry the player's "Re-read the screen" (`asked`): the cold read they asked for,
 * taken from the tracker's lock as the panel takes it, and held to the desktop's the same way.
 * Between frames a scenario may do what the desktop's pipeline tests do to a pipeline (`call`:
 * a reset, Flip, the End game menu, a side forced, a clock as a reader left it, a reconciler's
 * tunable), and a frame may count as changed by hand (`forced`) where the desktop's test passes
 * it; every event must carry its tick's time and frame id. A frame may carry where the tracker's
 * lock stood after it (`lock`), as the desktop's test of a moved board asserts it; and a scenario
 * how often its last session read the theme cache (`reads`), as
 * `test_a_puzzle_left_still_is_read_once` counts it.
 */
@Lane(Lane.Kind.VISION)
@RunWith(AndroidJUnit4::class)
class PipelineTest {

    @Test
    fun everyFrameIsReadAsThePythonReadsIt() = readAll(sameFrameShortcut = false)

    /**
     * The phone's still screen: a frame the display sent before, read again. The fixtures
     * hold a frame over several ticks as a still screen does, and here every repeat takes the
     * shortcut the capture takes (`GameWatcher.read` with no new bitmap: the last pixels, the
     * gate's `changedAgain`, the pipeline's last grey image and the tracker's remembered
     * results). The answers must still be the desktop's, frame for frame.
     */
    @Test
    fun aStillScreenReadAgainIsReadAsThePythonReadsIt() = readAll(sameFrameShortcut = true)

    private fun readAll(sameFrameShortcut: Boolean) {
        val context = InstrumentationRegistry.getInstrumentation().context
        val text = context.assets.open("pipeline/index.json").bufferedReader().use { it.readText() }
        val all = JSONObject(text).getJSONArray("runs")
        // A scenario that pins a behaviour of the pipeline's is replayed once; the still screen's
        // shortcut is held to the others (`READ_ONCE` in the generator).
        val runs = (0 until all.length()).map { all.getJSONObject(it) }
            .filter { !(sameFrameShortcut && it.optBoolean("read_once")) }
        val failures = java.util.Collections.synchronizedList(ArrayList<String>())
        // The scenarios side by side: each its own pipeline, and each thread its own net, one
        // thread to a net. Two at a time: a scenario holds its frames' pixels and planes and,
        // while it learns a start position, its net's 38 MB input buffer (a direct buffer is on
        // the Java heap here); three or four of them ran the 192 MB heap out, and three were no
        // faster than two.
        // OpenCV's own threads kept to one meanwhile: the replays are the parallelism, and its
        // pool's idle workers waited by reading the clock, a trap to the host on the emulator,
        // a fifth of the CPU. Its answers do not depend on how many threads it uses.
        val target = InstrumentationRegistry.getInstrumentation().targetContext
        val openCvThreads = Core.getNumThreads()
        Core.setNumThreads(1)
        try {
            Parallel.forEach(runs, threads = 2, resource = { CellEmbedder.load(target, threads = 2) }) { run, net ->
                if (failures.size < 8) replay(run, net, sameFrameShortcut, failures)
            }
        } finally {
            Core.setNumThreads(openCvThreads)
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /** One scenario through a fresh pipeline, its differences from the desktop's into [failures]. */
    private fun replay(run: JSONObject, net: CellEmbedder, sameFrameShortcut: Boolean, failures: MutableList<String>) {
        val name = run.getString("name")
        // A scenario in sessions shares one theme cache between them, as the app's does:
        // each session a fresh pipeline on it, as a Stop and Start is.
        val themes = if (run.optBoolean("cache")) {
            java.io.File(InstrumentationRegistry.getInstrumentation().targetContext.cacheDir, "pipeline-themes/$name")
                .apply { deleteRecursively(); mkdirs() }
        } else {
            null
        }
        // Its listings counted: each read from the cache (`ThemeStore.candidates`) lists
        // the directory once, and a scenario may say how often its last session read it.
        val counted = themes?.let { CountedDir(it.path) }
        val store = counted?.let { ThemeStore(it) }
        var pipeline = GamePipeline(net, store)
        var gate = ChangeGate()
        val frames = run.getJSONArray("frames")
        val decoded = HashMap<String, Mat>()
        var moves = 0
        var lastFile: String? = null
        var lastPixels: FramePixels? = null
        var lastMat: Mat? = null
        var shortcuts = 0
        val began = System.currentTimeMillis()
        for (f in 0 until frames.length()) {
            val frame = frames.getJSONObject(f)
            if (frame.has("call")) {
                // What the desktop's test did to its pipeline between frames.
                val done = call(pipeline, frame.getJSONObject("call"), lastMat, frame.getDouble("ts"))
                    .map { line(it) }
                val wanted = frame.getJSONArray("events").let { a -> List(a.length()) { a.getString(it) } }
                if (done != wanted && failures.size < 8) {
                    failures.add(
                        "$name frame $f (${frame.getJSONObject("call")}):\n  kotlin: " +
                            "${done.joinToString(" | ")}\n  python: ${wanted.joinToString(" | ")}",
                    )
                }
                continue
            }
            if (frame.optBoolean("session")) {
                pipeline = GamePipeline(net, store)
                gate = ChangeGate()
                lastFile = null
                lastPixels = null
                counted?.listings = 0
            }
            val file = frame.getString("file")
            val mat = decoded.getOrPut(file) { load(file) }
            val same = sameFrameShortcut && file == lastFile
            if (same) shortcuts++
            val pixels = lastPixels.takeIf { same } ?: FramePixels.of(mat)
            val ts = frame.getDouble("ts")
            val changed = if (frame.optBoolean("forced")) {
                // Passed by hand, as the desktop's test passes it: the gate is not asked.
                frame.getBoolean("changed")
            } else {
                gate.region = ChangeGate.boardGate(pipeline.board, pixels.width, pixels.height)
                if (same) gate.changedAgain(pixels, ts) else gate.changed(pixels, ts)
            }
            lastFile = file
            lastPixels = pixels
            lastMat = mat
            if (changed != frame.getBoolean("changed") && failures.size < 8) {
                failures.add(
                    "$name frame $f ($file): the gate says changed=$changed, " +
                        "python says ${frame.getBoolean("changed")}",
                )
            }
            val id = frame.optInt("id", f)
            val events = pipeline.process(mat, ts, changed, id, pixels, sameFrame = same)
            // Every event carries the tick's time and frame (`_stamp`).
            if (events.any { it.ts != ts || it.sourceFrameId != id } && failures.size < 8) {
                failures.add("$name frame $f ($file): an event not stamped with the tick")
            }
            val mine = events.map { line(it) }
            val theirs = frame.getJSONArray("events").let { a ->
                List(a.length()) { a.getString(it) }
            }
            moves += mine.count { it.startsWith("MovePlayed") }
            if (mine != theirs && failures.size < 8) {
                failures.add(
                    "$name frame $f ($file):\n  kotlin: ${mine.joinToString(" | ")}" +
                        "\n  python: ${theirs.joinToString(" | ")}",
                )
            }
            if (frame.has("lock")) {
                // Where the tracker's lock stands after this frame (a board moved: desktop
                // 17c45a6's `test_a_board_moved_any_fraction_of_a_cell_is_found_again`).
                val region = pipeline.board?.region
                val mine = region?.let { listOf(it.x, it.y, it.w, it.h) }
                val theirs = if (frame.isNull("lock")) {
                    null
                } else {
                    frame.getJSONArray("lock").let { a -> List(a.length()) { a.getInt(it) } }
                }
                if (mine != theirs && failures.size < 8) {
                    failures.add("$name frame $f ($file): the lock is at $mine, python's at $theirs")
                }
            }
            if (frame.has("asked")) {
                // "Re-read the screen" on this frame: `GameWatcher.analyse`, from the lock.
                val lock = pipeline.board
                val asked = if (lock == null) {
                    listOf("no board locked to re-read")
                } else {
                    pipeline.coldStart(
                        mat, BoardGeometry(lock.region, lock.cellPx), null, frame.getDouble("ts"),
                    ).map { line(it) }
                }
                val wanted = frame.getJSONArray("asked").let { a ->
                    List(a.length()) { a.getString(it) }
                }
                if (asked != wanted && failures.size < 8) {
                    failures.add(
                        "$name frame $f ($file) re-read:\n  kotlin: ${asked.joinToString(" | ")}" +
                            "\n  python: ${wanted.joinToString(" | ")}",
                    )
                }
            }
        }
        val took = System.currentTimeMillis() - began
        for (mat in decoded.values) mat.release()
        val rec = pipeline.reconciler
        Log.i(
            "sidecoach.state",
            "$name: ${frames.length()} frames ($shortcuts read again), $moves moves, " +
                "${pipeline.bootstraps} bootstraps, $took ms — ${rec?.pgn()}",
        )
        // No game at the end is an empty record on both sides.
        if ((rec?.pgn() ?: "") != run.getString("pgn")) {
            failures.add("$name: pgn is '${rec?.pgn()}', python's is '${run.getString("pgn")}'")
        }
        if ((rec?.board?.fen() ?: "") != run.getString("final_fen")) {
            failures.add(
                "$name: final fen is ${rec?.board?.fen()}, " +
                    "python's is ${run.getString("final_fen")}",
            )
        }
        if (pipeline.bootstraps != run.getInt("bootstraps")) {
            failures.add(
                "$name: ${pipeline.bootstraps} bootstraps, " +
                    "python had ${run.getInt("bootstraps")}",
            )
        }
        if (run.has("reads") && counted?.listings != run.getInt("reads")) {
            // `test_a_puzzle_left_still_is_read_once`: a still board read from the cache
            // once, not every tick (the parity audit, 2026-09-29).
            failures.add("$name: read from the cache ${counted?.listings} times, python ${run.getInt("reads")}")
        }
        if (themes != null) {
            val learned = themes.listFiles { f -> f.name.startsWith("theme-") }?.size ?: 0
            if (learned != run.getInt("themes")) {
                failures.add("$name: $learned learned sets cached, python had ${run.getInt("themes")}")
            }
            themes.deleteRecursively()
        }
    }

    /**
     * What a desktop test does to a pipeline between frames (`_call` in the generator): a reset,
     * the panel's Flip re-reading the last frame from the lock (told, or a resync's), the End
     * game menu, a side forced, a clock as a reader would have left it, a set's labels cached
     * as solid blocks, a reconciler's tunable.
     */
    private fun call(pipeline: GamePipeline, call: JSONObject, last: Mat?, ts: Double): List<Event> {
        fun side(name: String) = if (name == "white") Pieces.Side.WHITE else Pieces.Side.BLACK
        return when {
            call.has("reset") -> { pipeline.reset(); emptyList() }
            call.has("reorient") -> {
                val lock = checkNotNull(pipeline.board) { "no board locked to re-read" }
                pipeline.reorient(
                    checkNotNull(last), BoardGeometry(lock.region, lock.cellPx), side(call.getString("reorient")), ts,
                    told = call.optBoolean("told"),
                )
            }
            call.has("solid_labels") -> {
                // The set's rank labels cached as solid blocks, as a start seen through a lock
                // off its board cached them (desktop 3d0a201).
                val store = checkNotNull(pipeline.cache) { "no cache to spoil" }
                val model = checkNotNull(pipeline.model) { "no set learned to spoil" }
                val (key, cached) = store.candidates(model.lightBg, model.darkBg).single()
                cached.rankLabels = (1..8).associateWith { BooleanArray(Coordinates.GLYPH * Coordinates.GLYPH) { true } }
                store.save(cached, key)
                emptyList()
            }
            call.has("end_game") -> {
                val given = call.getJSONArray("end_game")
                pipeline.endGame(given.getString(0), given.getString(1), ts)
            }
            call.has("forced_bottom") -> { pipeline.forcedBottom = side(call.getString("forced_bottom")); emptyList() }
            call.has("last_clock") -> {
                val given = call.getJSONArray("last_clock")
                fun ms(i: Int): Int? = if (given.isNull(i)) null else given.getInt(i)
                pipeline.lastClock = Triple(ms(0), ms(1), if (given.isNull(2)) null else side(given.getString(2)))
                pipeline.lastClockTs = if (call.isNull("last_clock_ts")) null else call.getDouble("last_clock_ts")
                emptyList()
            }
            call.has("reconciler") -> {
                // A reconciler's tunable, as the desktop's test sets it (`resync_hold_frames`).
                val rec = checkNotNull(pipeline.reconciler) { "no reconciler to tune" }
                val given = call.getJSONObject("reconciler")
                for (key in given.keys()) {
                    when (key) {
                        "resync_hold_frames" -> rec.resyncHoldFrames = given.getInt(key)
                        else -> throw IllegalArgumentException("a tunable this test does not set: $key")
                    }
                }
                emptyList()
            }
            else -> throw IllegalArgumentException("a call this test does not know: $call")
        }
    }

    /** A theme directory that counts its listings (`ThemeStore.candidates` lists it once a read). */
    private class CountedDir(path: String) : java.io.File(path) {
        var listings = 0

        override fun listFiles(filter: java.io.FileFilter?): Array<java.io.File>? {
            listings += 1
            return super.listFiles(filter)
        }
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

    /** The same line `make_pipeline_fixtures.py` writes; the game id is random, so it is out. */
    private fun line(event: Event): String = when (event) {
        is GameStarted -> "GameStarted ${event.kind} ${side(event.userSide)} ${event.startFen}"
        is PositionChanged -> "PositionChanged ${event.ply} ${event.fen}"
        is MovePlayed ->
            "MovePlayed ${event.ply} ${event.san} ${event.uci} " +
                (if (event.byUser) "user" else "them")
        is Takeback -> "Takeback ${event.plies} ${event.fen}"
        is Resync -> "Resync ${event.reason} | ${event.fen}"
        is Unexplained -> "Unexplained ${event.reason} | ${event.fen}"
        is BoardUnread -> "BoardUnread ${event.reason}"
        is GameEnded -> "GameEnded ${event.result} ${event.reason} | ${event.pgn}"
        is ScreenStateChanged -> "ScreenState ${event.state.name.lowercase()}"
        else -> event::class.java.simpleName
    }

    private fun side(side: Pieces.Side): String = if (side == Pieces.Side.WHITE) "white" else "black"

    companion object {
        @JvmStatic
        @BeforeClass
        fun loadOpenCv() {
            assertTrue("OpenCV did not load", OpenCVLoader.initLocal())
        }
    }
}
