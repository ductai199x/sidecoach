package com.sidecoach.vision

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import kotlin.math.abs

/**
 * The silhouette port held to the Python's own numbers.
 *
 * `tools/make_fixtures.py` records, for every rendered board, what `pieces.occupancy`
 * measured for each of the 64 cells, what `looks_like_start_position` made of that, and
 * which side `orientation_from_start` put at the bottom. The Kotlin must agree: this is
 * what makes the port a port rather than a rewrite.
 */
@RunWith(AndroidJUnit4::class)
class SilhouetteTest {

    @Test
    fun theForegroundFractionsMatchThePython() {
        val failures = ArrayList<String>()
        for (fixture in Fixtures.all()) {
            val frame = Fixtures.mat(fixture)
            val geometry = BoardGeometry(
                Region(fixture.x, fixture.y, fixture.side, fixture.side),
                fixture.cell.toDouble(),
            )
            val cells = ThemeModel.occupancy(frame, geometry)
            var worst = 0.0
            var worstCell = -1
            for (i in 0 until 64) {
                val gap = abs(cells[i].fraction - fixture.fractions[i])
                if (gap > worst) { worst = gap; worstCell = i }
            }
            Log.i(
                "sidecoach.vision",
                "${fixture.name}: worst cell ${worstCell / 8},${worstCell % 8} off by " +
                    "${"%.4f".format(worst)} (python ${"%.4f".format(fixture.fractions[worstCell])}, " +
                    "kotlin ${"%.4f".format(cells[worstCell].fraction)})",
            )
            // A cell's fraction is a share of 2304 pixels; a pixel either way is 0.0004,
            // and the two runs differ only in how OpenCV rounds the resize.
            if (worst > 0.02) {
                failures.add("${fixture.name}: cell ${worstCell / 8},${worstCell % 8} off by ${"%.4f".format(worst)}")
            }
            val shaped = ThemeModel.startShaped(cells)
            if (shaped != fixture.startShaped) {
                failures.add("${fixture.name}: start shape read $shaped, python says ${fixture.startShaped}")
            }
            if (fixture.startShaped) {
                val white = ThemeModel.orientationFromStart(cells)
                val expected = fixture.orientation == "white"
                if (white != expected) {
                    failures.add("${fixture.name}: orientation read ${if (white) "white" else "black"}")
                }
            }
            frame.release()
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
