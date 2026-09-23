package com.sidecoach.state

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.events.Event
import com.sidecoach.events.GameEnded
import com.sidecoach.events.GameStarted
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.PositionChanged
import com.sidecoach.events.Resync
import com.sidecoach.events.ScreenStateChanged
import com.sidecoach.events.Takeback
import com.sidecoach.vision.CellEmbedder
import com.sidecoach.vision.ChangeGate
import com.sidecoach.vision.FramePixels
import com.sidecoach.vision.Pieces
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
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
 */
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
        val runs = JSONObject(text).getJSONArray("runs")
        val failures = ArrayList<String>()
        CellEmbedder.load(InstrumentationRegistry.getInstrumentation().targetContext).use { net ->
            for (r in 0 until runs.length()) {
                val run = runs.getJSONObject(r)
                val name = run.getString("name")
                val pipeline = GamePipeline(net)
                val gate = ChangeGate()
                val frames = run.getJSONArray("frames")
                val decoded = HashMap<String, Mat>()
                var moves = 0
                var lastFile: String? = null
                var lastPixels: FramePixels? = null
                var shortcuts = 0
                val began = System.currentTimeMillis()
                for (f in 0 until frames.length()) {
                    val frame = frames.getJSONObject(f)
                    val file = frame.getString("file")
                    val mat = decoded.getOrPut(file) { load(file) }
                    val same = sameFrameShortcut && file == lastFile
                    if (same) shortcuts++
                    val pixels = lastPixels.takeIf { same } ?: FramePixels.of(mat)
                    gate.region = ChangeGate.boardGate(pipeline.board, pixels.width, pixels.height)
                    val changed = if (same) {
                        gate.changedAgain(pixels, frame.getDouble("ts"))
                    } else {
                        gate.changed(pixels, frame.getDouble("ts"))
                    }
                    lastFile = file
                    lastPixels = pixels
                    if (changed != frame.getBoolean("changed") && failures.size < 8) {
                        failures.add(
                            "$name frame $f ($file): the gate says changed=$changed, " +
                                "python says ${frame.getBoolean("changed")}",
                        )
                    }
                    val events = pipeline.process(mat, frame.getDouble("ts"), changed, f, pixels, sameFrame = same)
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
                }
                val took = System.currentTimeMillis() - began
                for (mat in decoded.values) mat.release()
                val rec = pipeline.reconciler
                Log.i(
                    "sidecoach.state",
                    "$name: ${frames.length()} frames ($shortcuts read again), $moves moves, " +
                        "${pipeline.bootstraps} bootstraps, $took ms — ${rec?.pgn()}",
                )
                if (rec?.pgn() != run.getString("pgn")) {
                    failures.add("$name: pgn is '${rec?.pgn()}', python's is '${run.getString("pgn")}'")
                }
                if (rec?.board?.fen() != run.getString("final_fen")) {
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
                if (failures.size >= 8) break
            }
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
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
