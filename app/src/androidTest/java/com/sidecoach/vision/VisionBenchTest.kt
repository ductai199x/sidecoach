package com.sidecoach.vision

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.InProcess
import com.sidecoach.Lane
import com.sidecoach.events.Event
import com.sidecoach.events.MovePlayed
import com.sidecoach.state.GamePipeline
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File

/**
 * How long the phone's vision takes to see a move: the cost of each tick and how many frames a
 * move takes to commit. The desktop's `tools/bench_vision.py`, on the phone's own code.
 *
 * The game comes from `bench_vision.py --width 672 --height 1496 --cell 82 --frames
 * app/build/bench/bench-vision` (the phone's capture size, half the screen each way), which the
 * build puts in the test APK's assets. Each frame
 * goes through what `GameWatcher.read` does on a tick: a new screen is converted to BGR and
 * measured, the change gate moved onto the board, then `GamePipeline.process`; the frames after
 * it are the same screen again, as the display sends nothing new while nothing moves. The net,
 * the theme store and the clock reader are the app's. Nothing else runs: no engine, no coach,
 * no voice. Times are wall time on this thread, one tick after another at the capture's rate
 * of the game's clock (the clock is the game's, the waiting between ticks is not simulated).
 *
 * A benchmark, not a test: it logs and asserts only that the game was followed, so it runs only
 * when asked (`-e bench true`), as [EmbedStagesTest] does.
 */
@Lane(Lane.Kind.VISION)
@InProcess(2)
@RunWith(AndroidJUnit4::class)
class VisionBenchTest {

    @Test
    fun theTicksOfAGameAreTimed() {
        assumeTrue(
            "a benchmark: run with -e bench true",
            InstrumentationRegistry.getArguments().getString("bench") == "true",
        )
        assertTrue(OpenCVLoader.initLocal())
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val assets = InstrumentationRegistry.getInstrumentation().context.assets
        val game = JSONObject(assets.open("$FOLDER/game.json").bufferedReader().use { it.readText() })
        val step = 1.0 / game.getDouble("fps")
        val patience = game.getInt("patience")
        val positions = game.getJSONArray("positions")

        val themes = File(context.cacheDir, "bench-vision-themes").apply { deleteRecursively(); mkdirs() }
        CellEmbedder.load(context).use { net ->
            val pipeline = GamePipeline(net, ThemeStore(themes)).apply {
                GlyphTemplates.load(context.assets)?.let { clock = ClockReader(it) }
            }
            val ticker = Ticker(pipeline, step)

            fun frame(i: Int): Bitmap = load(assets, "$FOLDER/${positions.getJSONObject(i).getString("file")}")
            val start = frame(0)
            var lockFrames: Int? = null
            var lockMs = 0.0
            for (n in 1..patience) {
                lockMs += ticker.tick(if (n == 1) start else null).ms
                if (pipeline.reconciler != null) {
                    lockFrames = n
                    break
                }
            }
            assertTrue("the start position was never locked", lockFrames != null)

            val moveTicks = ArrayList<Double>()
            val stillTicks = ArrayList<Double>()
            val commits = ArrayList<Int?>()
            for (i in 1 until positions.length()) {
                val frame = frame(i)
                var commit: Int? = null
                for (n in 1..patience) {
                    val tick = ticker.tick(if (n == 1) frame else null)
                    (if (n == 1) moveTicks else stillTicks).add(tick.ms)
                    if (tick.events.any { it is MovePlayed }) {
                        commit = n
                        break
                    }
                }
                commits.add(commit)
            }

            val committed = commits.filterNotNull()
            Log.i(TAG, "frame ${start.width}x${start.height}, ${commits.size} moves, ${1 / step} fps")
            Log.i(TAG, "first lock    frame $lockFrames, ${"%.0f".format(lockMs)} ms of compute")
            Log.i(TAG, "tick, move    ${summary(moveTicks)}")
            Log.i(TAG, "tick, still   ${summary(stillTicks)}")
            Log.i(
                TAG,
                "commit        on frame ${committed.minOrNull()}-${committed.maxOrNull()} showing the move, " +
                    "missed ${commits.size - committed.size}",
            )
            assertTrue("moves missed: $commits", committed.size == commits.size)
        }
    }

    /** `GameWatcher.read`'s tick over a new screen ([bitmap]) or the same one again (null), timed. */
    private class Ticker(private val pipeline: GamePipeline, private val step: Double) {
        private val gate = ChangeGate()
        private var clock = 0.0
        private var frameId = 0
        private var lastFrame: Mat? = null
        private var lastPixels: FramePixels? = null

        class Tick(val events: List<Event>, val ms: Double)

        fun tick(bitmap: Bitmap?): Tick {
            val began = System.nanoTime()
            val same = bitmap == null
            val bgr = if (same) lastFrame!! else toBgr(bitmap!!).also { fresh -> lastFrame?.release(); lastFrame = fresh }
            val pixels = lastPixels.takeIf { same } ?: FramePixels.of(bgr)
            gate.region = ChangeGate.boardGate(pipeline.board, pixels.width, pixels.height)
            val changed = if (same) gate.changedAgain(pixels, clock) else gate.changed(pixels, clock)
            lastPixels = pixels
            val events = pipeline.process(bgr, clock, changed, frameId++, pixels, sameFrame = same)
            clock += step
            return Tick(events, (System.nanoTime() - began) / 1e6)
        }

        private fun toBgr(bitmap: Bitmap): Mat {
            val rgba = Mat()
            Utils.bitmapToMat(bitmap, rgba)
            val bgr = Mat()
            Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
            rgba.release()
            return bgr
        }
    }

    private fun load(assets: AssetManager, path: String): Bitmap {
        val options = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        return assets.open(path).use { stream ->
            requireNotNull(BitmapFactory.decodeStream(stream, null, options)) { "cannot read $path" }
        }
    }

    private fun summary(ms: List<Double>): String {
        val ordered = ms.sorted()
        val median = if (ordered.size % 2 == 1) {
            ordered[ordered.size / 2]
        } else {
            (ordered[ordered.size / 2 - 1] + ordered[ordered.size / 2]) / 2
        }
        val p90 = ordered[maxOf(0, Math.round(0.9 * ordered.size).toInt() - 1)]
        return "median ${"%5.1f".format(median)} ms   p90 ${"%5.1f".format(p90)} ms   " +
            "max ${"%5.1f".format(ordered.last())} ms   (n=${ordered.size})"
    }

    private companion object {
        const val TAG = "sidecoach.bench"
        const val FOLDER = "bench-vision"
    }
}
