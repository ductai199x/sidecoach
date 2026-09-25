package com.sidecoach.vision

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

/**
 * What the phase landing ([Pieces.LANDING], desktop de58ca7) costs the recogniser per frame,
 * against the centroid it replaced: the slide scenario `slide-rook-green` read frame by frame on
 * one model, as a game is -- the first frame read whole, the frames a piece slides through
 * (busy: the changed cells embedded and landed), and the frames that repeat the one before
 * (quiet: nothing to read again). Not a parity test and not part of the suite: it runs only when
 * asked,
 *
 *     adb -s <serial> shell am instrument -w -e bench true \
 *         -e class com.sidecoach.vision.LandingCostTest com.sidecoach.test/androidx.test.runner.AndroidJUnitRunner
 *
 * and logs the numbers under the `sidecoach.vision` tag. `-e rounds N` reads the scenario N times
 * per rule (default 3). Then the phase correlation alone, per cell: a start crop against a piece
 * caught mid-slide, `-e calls N` times (default 2000).
 */
@RunWith(AndroidJUnit4::class)
class LandingCostTest {

    @Test
    fun whatThePhaseLandingCostsPerFrame() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue("a benchmark: run with -e bench true", args.getString("bench") == "true")
        val rounds = args.getString("rounds")?.toIntOrNull() ?: 3
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val index = JSONObject(
            instrumentation.context.assets.open("pipeline/index.json").bufferedReader().use { it.readText() },
        ).getJSONArray("runs")
        var files: List<String> = emptyList()
        for (r in 0 until index.length()) {
            val run = index.getJSONObject(r)
            if (run.getString("name") != "slide-rook-green") continue
            val frames = run.getJSONArray("frames")
            files = List(frames.length()) { "pipeline/" + frames.getJSONObject(it).getString("file") }
        }
        assumeTrue("the slide scenario is in the fixtures", files.isNotEmpty())
        val decoded = HashMap<String, Mat>()
        for (file in files.toSet()) decoded[file] = load(file)
        // The slide scenarios' board (`make_pipeline_fixtures.SLIDE_BOARD`).
        val geometry = BoardGeometry(Region(300, 120, 512, 512), 64.0)
        val landing = Pieces.LANDING
        try {
            CellEmbedder.load(instrumentation.targetContext).use { embedder ->
                for (rule in listOf("centroid", "phase", "centroid", "phase")) {
                    Pieces.LANDING = rule
                    val first = ArrayList<Long>()
                    val busy = ArrayList<Long>()
                    val quiet = ArrayList<Long>()
                    var landed = 0
                    repeat(rounds) {
                        val model = ThemeModel.fromStartPosition(decoded.getValue(files[0]), geometry, true, embedder)!!
                        var previous: String? = null
                        for ((k, file) in files.withIndex()) {
                            val began = System.nanoTime()
                            model.classifyFrame(decoded.getValue(file), geometry, embedder)
                            val took = (System.nanoTime() - began) / 1000
                            when {
                                k == 0 -> first.add(took)
                                file == previous -> quiet.add(took)
                                else -> {
                                    busy.add(took)
                                    landed += model.lastEmbedded
                                }
                            }
                            previous = file
                        }
                    }
                    Log.i(
                        TAG,
                        "$rule: whole first read ${stats(first)}; busy frames ${stats(busy)} " +
                            "(${landed / rounds} cells named in ${busy.size / rounds} frames a round); " +
                            "quiet frames ${stats(quiet)}",
                    )
                }
                // The phase correlation alone: the rook's start crop against the rook's cell on a
                // frame of its slide, as `fromEmbedding` measures it.
                val calls = args.getString("calls")?.toIntOrNull() ?: 2000
                val model = ThemeModel.fromStartPosition(decoded.getValue(files[0]), geometry, true, embedder)!!
                val reference = model.landingCrops.getValue("R")
                val sliding = geometry.crop(decoded.getValue(files[files.size / 3]), 3, 5, 2 * Squares.CELL)
                val each = ArrayList<Long>(calls)
                repeat(calls) {
                    val began = System.nanoTime()
                    model.phaseShift(reference, sliding)
                    each.add((System.nanoTime() - began) / 1000)
                }
                sliding.release()
                val sorted = each.sorted()
                Log.i(
                    TAG,
                    "phase correlation per cell: median %d us, mean %.0f us, p95 %d us (n=%d)".format(
                        sorted[sorted.size / 2], each.average(), sorted[sorted.size * 95 / 100], calls,
                    ),
                )
            }
        } finally {
            Pieces.LANDING = landing
            decoded.values.forEach { it.release() }
        }
    }

    /** Median and mean, and the worst, in milliseconds. */
    private fun stats(micros: List<Long>): String {
        if (micros.isEmpty()) return "none"
        val sorted = micros.sorted()
        val median = sorted[sorted.size / 2] / 1000.0
        val mean = micros.average() / 1000.0
        val worst = sorted.last() / 1000.0
        return "median %.2f ms, mean %.2f, max %.2f (n=%d)".format(median, mean, worst, micros.size)
    }

    private fun load(path: String): Mat {
        val context = InstrumentationRegistry.getInstrumentation().context
        val bitmap = context.assets.open(path).use { android.graphics.BitmapFactory.decodeStream(it) }
        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        val bgr = Mat()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        rgba.release()
        bitmap.recycle()
        return bgr
    }

    companion object {
        private const val TAG = "sidecoach.vision"

        @JvmStatic
        @BeforeClass
        fun loadOpenCv() {
            assertTrue("OpenCV did not load", OpenCVLoader.initLocal())
        }
    }
}
