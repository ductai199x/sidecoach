package com.sidecoach.vision

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.Lane
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader

/**
 * The recogniser held to the Python's own reading, cell for cell.
 *
 * `tools/make_fixtures.py` bootstraps a `ThemeModel` from each theme's start position and
 * records what `classify_frame` made of every cell of every board of that theme. The
 * Kotlin bootstraps the same way from the same frame and must name the same squares.
 */
@Lane(Lane.Kind.VISION)
@RunWith(AndroidJUnit4::class)
class ClassifyTest {

    @Test
    fun everyCellIsNamedAsThePythonNamesIt() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val fixtures = Fixtures.all().associateBy { it.name }
        val failures = ArrayList<String>()
        CellEmbedder.load(context).use { embedder ->
            for (fixture in fixtures.values) {
                val from = fixture.classifiedFrom ?: continue
                val start = fixtures.getValue(from)
                val startFrame = Fixtures.mat(start)
                val geometry = BoardGeometry(
                    Region(start.x, start.y, start.side, start.side), start.cell.toDouble(),
                )
                val cells = ThemeModel.occupancy(startFrame, geometry)
                val whiteAtBottom = ThemeModel.orientationFromStart(cells)
                val model = ThemeModel.fromStartPosition(startFrame, geometry, whiteAtBottom, embedder)
                startFrame.release()
                if (model == null) {
                    failures.add("${fixture.name}: could not bootstrap from $from")
                    continue
                }
                val frame = Fixtures.mat(fixture)
                val began = System.currentTimeMillis()
                val readings = model.classifyFrame(frame, geometry, embedder)
                val took = System.currentTimeMillis() - began
                frame.release()
                var wrong = 0
                val examples = ArrayList<String>()
                for (i in 0 until 64) {
                    val mine = readings[i].symbol ?: ""
                    val theirs = fixture.symbols[i]
                    if (mine != theirs) {
                        wrong++
                        if (examples.size < 4) {
                            examples.add("cell ${i / 8},${i % 8}: python '$theirs', kotlin '$mine'")
                        }
                    }
                }
                Log.i(
                    "sidecoach.vision",
                    "${fixture.name}: $wrong of 64 cells differ from the python, read in $took ms " +
                        "(${model.lastFresh} cells embedded)",
                )
                if (wrong > 0) failures.add("${fixture.name}: $wrong cells — ${examples.joinToString("; ")}")
            }
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    companion object {
        @JvmStatic
        @BeforeClass
        fun loadOpenCv() {
            assertTrue("OpenCV did not load", OpenCVLoader.initLocal())
        }
    }
}
