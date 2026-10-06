package com.sidecoach.state

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.vision.CellEmbedder
import com.sidecoach.vision.ChangeGate
import com.sidecoach.vision.FramePixels
import com.sidecoach.vision.Pieces
import com.sidecoach.vision.ThemeStore
import java.io.File
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

/**
 * A Dump's frames through the phone's own pipeline, ticked as the capture loop ticks: each
 * dumped frame (the ones the gate let through) and, between them, the same frame again seven
 * times a second. Logs every event, and on each dumped frame what holds a position resync off.
 * A tool, not a test: it runs only when `assets/dumpreplay/frames` is there (local, not
 * committed), with the learned sets in `assets/dumpreplay/themes` -- the dump's own `themes`
 * folder (a dump carries the phone's learned sets), or the desktop's, written as the phone's
 * by `tools/themes_to_phone.py`. Built for Tai's puzzle never adopted (2026-10-03): with the
 * desktop's sets the port adopted it at once, which put the fault in the phone's own set.
 *
 *     adb -s emulator-5556 shell am instrument -w -e class com.sidecoach.state.DumpReplayTest \
 *         -e from 683312 -e bottom white com.sidecoach.test/androidx.test.runner.AndroidJUnitRunner
 */
@RunWith(AndroidJUnit4::class)
class DumpReplayTest {

    @Test
    fun replay() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val assets = instrumentation.context.assets
        val names = assets.list("dumpreplay/frames")?.sorted().orEmpty()
        assumeTrue("no dump staged", names.isNotEmpty())
        check(OpenCVLoader.initLocal())
        val args = InstrumentationRegistry.getArguments()
        val from = args.getString("from")?.toDouble() ?: 0.0
        val fps = args.getString("fps")?.toDouble() ?: 7.0
        val target = instrumentation.targetContext
        val themes = File(target.cacheDir, "dumpreplay-themes").apply { deleteRecursively(); mkdirs() }
        for (name in assets.list("dumpreplay/themes").orEmpty()) {
            assets.open("dumpreplay/themes/$name").use { input -> File(themes, name).outputStream().use { input.copyTo(it) } }
        }
        val net = CellEmbedder.load(target, threads = 2)
        val pipeline = GamePipeline(net, ThemeStore(themes))
        when (args.getString("bottom")) {
            "white" -> pipeline.forcedBottom = Pieces.Side.WHITE
            "black" -> pipeline.forcedBottom = Pieces.Side.BLACK
        }
        val gate = ChangeGate()
        val frames = names.map { it.removeSuffix(".jpg").toDouble() to it }.filter { it.first >= from }
        var lastPixels: FramePixels? = null
        var lastMat: Mat? = null
        var id = 0
        for ((index, entry) in frames.withIndex()) {
            val (ts, name) = entry
            // The still ticks before this frame: the last frame again, as the loop reads it.
            val previous = lastMat
            if (previous != null) {
                var tick = frames[index - 1].first + 1.0 / fps
                while (tick < ts) {
                    gate.region = ChangeGate.boardGate(pipeline.board, lastPixels!!.width, lastPixels.height)
                    val changed = gate.changedAgain(lastPixels, tick)
                    report(tick, "tick", pipeline.process(previous, tick, changed, id++, lastPixels, sameFrame = true), null)
                    tick += 1.0 / fps
                }
            }
            val mat = load(assets.open("dumpreplay/frames/$name").use { android.graphics.BitmapFactory.decodeStream(it) })
            val pixels = FramePixels.of(mat)
            gate.region = ChangeGate.boardGate(pipeline.board, pixels.width, pixels.height)
            val changed = gate.changed(pixels, ts)
            val events = pipeline.process(mat, ts, changed, id++, pixels, sameFrame = false)
            report(ts, "frame changed=$changed", events, pipeline.reconciler)
            lastMat?.release()
            lastMat = mat
            lastPixels = pixels
        }
    }

    private fun report(ts: Double, what: String, events: List<com.sidecoach.events.Event>, rec: Reconciler?) {
        val said = events.map { e ->
            when (e) {
                is com.sidecoach.events.MovePlayed -> "MovePlayed ${e.san}"
                is com.sidecoach.events.Resync -> "Resync ${e.reason} ${e.fen}"
                is com.sidecoach.events.GameStarted -> "GameStarted ${e.kind} ${e.startFen}"
                is com.sidecoach.events.Unexplained -> "Unexplained ${e.reason}"
                is com.sidecoach.events.BoardUnread -> "BoardUnread ${e.reason}"
                else -> e::class.java.simpleName
            }
        }.filter { it != "PositionChanged" }
        if (rec == null) {
            if (said.isNotEmpty()) Log.i(TAG, "%.3f %s %s".format(ts, what, said))
            return
        }
        @Suppress("UNCHECKED_CAST")
        val obs = field(rec, "lastObs") as Map<Int, Pieces.CellReading>
        val cand = field(rec, "resyncCandidate") as Triple<*, *, *>?
        fun names(pick: (Pieces.CellReading) -> Boolean) =
            obs.filter { pick(it.value) }.keys.sorted().map { com.sidecoach.chess.squareName(it) }
        Log.i(
            TAG,
            "%.3f %s cells %d unsettled %s soft %s hover %s red %s cand %s queueTs %s fen %s %s".format(
                ts, what, obs.size,
                names { it.kind == Pieces.Kind.PIECE && it.symbol != null && !it.settled },
                names { it.kind == Pieces.Kind.PIECE && it.symbol == null },
                names { it.hover }, names { it.tint == "red" },
                cand?.second, field(rec, "queueTs"), rec.board.boardFen(), said,
            ),
        )
    }

    private fun field(target: Any, name: String): Any? =
        target.javaClass.getDeclaredField(name).apply { isAccessible = true }.get(target)

    private fun load(bitmap: android.graphics.Bitmap): Mat {
        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        val bgr = Mat()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        rgba.release()
        bitmap.recycle()
        return bgr
    }

    private companion object {
        const val TAG = "dumpreplay"
    }
}
