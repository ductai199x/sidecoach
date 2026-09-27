package com.sidecoach.vision

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import kotlin.math.abs

/**
 * Where a piece sits, held to the desktop's phase landing (`pieces._phase_shift`, desktop
 * de58ca7): `tools/make_landing_fixtures.py` learned a model from a start frame and read frames
 * one after another -- pieces at rest on the phone-sized boards, pieces caught mid-slide with a
 * trail on the slide scenarios' frames, and the desktop's tests of a set learned at one place
 * and size and read at another (a king or a queen on the square colour its start never showed
 * it on, pieces at rest on a resized board, a pawn in flight on one, a rook on its labelled
 * corner: desktop aac94ac, cfaa416) -- and wrote every cell's kind, symbol, settled flag and
 * offset. The phone learns from the same frame, reads the same frames in the same order, and
 * must agree on every kind, symbol and settled flag, and on every piece's shift to 1e-3 of a
 * cell (its OpenCV is another build than the desktop's).
 */
@RunWith(AndroidJUnit4::class)
class LandingTest {

    @Test
    fun everyPieceLandsWhereThePythonSaysItDoes() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val text = instrumentation.context.assets.open("boards/landing.json").bufferedReader().use { it.readText() }
        val cases = JSONObject(text).getJSONArray("cases")
        val failures = ArrayList<String>()
        var worst = 0.0
        var worstAt = ""
        var shifts = 0
        var landing = 0
        assertTrue(Pieces.LANDING == "phase")
        CellEmbedder.load(instrumentation.targetContext).use { embedder ->
            for (k in 0 until cases.length()) {
                val case = cases.getJSONObject(k)
                val name = case.getString("name")
                // One geometry for the start and every frame; or, for the desktop's tests of a
                // set learned at one place and size and read at another (desktop aac94ac,
                // cfaa416), each frame's own, as the desktop's detector locked it.
                val geometry = if (case.has("start_region")) {
                    BoardGeometry(region(case.getJSONArray("start_region")), case.getDouble("start_cell"))
                } else {
                    val cell = case.getInt("cell")
                    BoardGeometry(Region(case.getInt("x"), case.getInt("y"), 8 * cell, 8 * cell), cell.toDouble())
                }
                val whiteAtBottom = case.optBoolean("white_at_bottom", true)
                val start = load(case.getString("start"))
                val model = ThemeModel.fromStartPosition(start, geometry, whiteAtBottom, embedder)
                start.release()
                if (model == null || model.landingCrops.isEmpty()) {
                    failures.add("$name: no model with landing crops")
                    continue
                }
                val frames = case.getJSONArray("frames")
                for (f in 0 until frames.length()) {
                    val frame = frames.getJSONObject(f)
                    val mat = load(frame.getString("file"))
                    val at = if (frame.has("region")) {
                        BoardGeometry(region(frame.getJSONArray("region")), frame.getDouble("cell"))
                    } else {
                        geometry
                    }
                    val readings = model.classifyFrame(mat, at, embedder)
                    mat.release()
                    val cells = frame.getJSONArray("cells")
                    for (i in 0 until 64) {
                        val want = cells.getJSONArray(i)
                        val kind = want.getString(0)
                        val symbol = if (want.isNull(1)) null else want.getString(1)
                        val settled = want.getBoolean(2)
                        val offset = want.getDouble(3)
                        val mine = readings[i]
                        val where = "$name ${frame.getString("file")} cell ${i / 8},${i % 8}"
                        if (mine.kind.name.lowercase() != kind || mine.symbol != symbol) {
                            if (failures.size < 8) failures.add("$where: python $kind $symbol, kotlin ${mine.kind} ${mine.symbol}")
                            continue
                        }
                        if (kind != "piece") continue
                        shifts++
                        if (!settled) landing++
                        val diff = abs(mine.offset - offset)
                        if (diff > worst) {
                            worst = diff
                            worstAt = "$where (python ${"%.5f".format(offset)}, kotlin ${"%.5f".format(mine.offset)})"
                        }
                        if (mine.settled != settled && failures.size < 8) {
                            failures.add(
                                "$where: python settled=$settled (${"%.5f".format(offset)}), " +
                                    "kotlin settled=${mine.settled} (${"%.5f".format(mine.offset)})",
                            )
                        }
                    }
                }
            }
        }
        Log.i(
            "sidecoach.vision",
            "landing parity: ${cases.length()} cases, $shifts piece shifts ($landing still landing), " +
                "max difference ${"%.2e".format(worst)} of a cell at $worstAt",
        )
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
        assertTrue("the worst shift differs by $worst at $worstAt", worst <= 1e-3)
    }

    private fun region(a: org.json.JSONArray) = Region(a.getInt(0), a.getInt(1), a.getInt(2), a.getInt(3))

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
        @JvmStatic
        @BeforeClass
        fun loadOpenCv() {
            assertTrue("OpenCV did not load", OpenCVLoader.initLocal())
        }
    }
}
