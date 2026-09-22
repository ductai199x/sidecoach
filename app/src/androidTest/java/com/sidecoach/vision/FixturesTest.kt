package com.sidecoach.vision

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType

/**
 * The fixtures the vision port is checked against: boards rendered by the desktop repo's
 * own renderer (tests/synth.py -- the real chess.com themes and the lichess sprite
 * sheets) at the size the capture service hands us on a phone, each with the region,
 * the cell and the FEN it was drawn from. See tools/make_fixtures.py.
 *
 * This test proves the harness: OpenCV loads, the assets are there, and a board becomes
 * a Mat of the size the index says.
 */
@RunWith(AndroidJUnit4::class)
class FixturesTest {

    @Test
    fun everyFixtureLoadsAsAMat() {
        val fixtures = Fixtures.all()
        assertTrue("no fixtures in the assets", fixtures.isNotEmpty())
        for (fixture in fixtures) {
            val mat = Fixtures.mat(fixture)
            assertEquals("${fixture.name}: width", fixture.width, mat.cols())
            assertEquals("${fixture.name}: height", fixture.height, mat.rows())
            assertEquals("${fixture.name}: three channels", CvType.CV_8UC3, mat.type())
            assertTrue("${fixture.name}: the board fits in the frame",
                fixture.x + fixture.side <= fixture.width && fixture.y + fixture.side <= fixture.height)
        }
    }

    companion object {
        @JvmStatic
        @BeforeClass
        fun loadOpenCv() {
            assertTrue("OpenCV did not load", OpenCVLoader.initLocal())
        }
    }
}

/** One rendered board and what it was drawn from. */
data class Fixture(
    val name: String,
    val file: String,
    val x: Int,
    val y: Int,
    val cell: Int,
    val side: Int,
    val fen: String,
    val bottom: String,
    val width: Int,
    val height: Int,
    /** What the Python measured on this frame, for the port to be held to. */
    val fractions: DoubleArray,
    val startShaped: Boolean,
    val orientation: String,
    /** The start fixture the Python bootstrapped from to classify this one, if any. */
    val classifiedFrom: String? = null,
    val symbols: List<String> = emptyList(),
    val kinds: List<String> = emptyList(),
)

object Fixtures {

    fun all(): List<Fixture> {
        val context = InstrumentationRegistry.getInstrumentation().context
        val text = context.assets.open("boards/index.json").bufferedReader().use { it.readText() }
        val array = JSONArray(text)
        return (0 until array.length()).map { i ->
            val o = array.getJSONObject(i)
            Fixture(
                name = o.getString("name"),
                file = o.getString("file"),
                x = o.getInt("x"),
                y = o.getInt("y"),
                cell = o.getInt("cell"),
                side = o.getInt("side"),
                fen = o.getString("fen"),
                bottom = o.getString("bottom"),
                width = o.getInt("width"),
                height = o.getInt("height"),
                fractions = o.getJSONArray("fractions").let { a ->
                    DoubleArray(a.length()) { a.getDouble(it) }
                },
                startShaped = o.getBoolean("start_shaped"),
                orientation = o.getString("orientation"),
                classifiedFrom = if (o.has("classified_from")) o.getString("classified_from") else null,
                symbols = if (o.has("symbols")) o.getJSONArray("symbols").let { a ->
                    List(a.length()) { a.getString(it) }
                } else emptyList(),
                kinds = if (o.has("kinds")) o.getJSONArray("kinds").let { a ->
                    List(a.length()) { a.getString(it) }
                } else emptyList(),
            )
        }
    }

    /** The fixture as BGR, the way a captured frame reaches the pipeline. */
    fun mat(fixture: Fixture): org.opencv.core.Mat {
        val context = InstrumentationRegistry.getInstrumentation().context
        val bitmap = context.assets.open("boards/${fixture.file}").use {
            android.graphics.BitmapFactory.decodeStream(it)
        }
        val rgba = org.opencv.core.Mat()
        Utils.bitmapToMat(bitmap, rgba)
        val bgr = org.opencv.core.Mat()
        org.opencv.imgproc.Imgproc.cvtColor(rgba, bgr, org.opencv.imgproc.Imgproc.COLOR_RGBA2BGR)
        rgba.release()
        bitmap.recycle()
        return bgr
    }
}
