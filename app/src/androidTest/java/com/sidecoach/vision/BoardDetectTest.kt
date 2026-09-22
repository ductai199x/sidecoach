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
 * The port held to the Python's own ground truth: every fixture was rendered with a known
 * board region and cell, and the detector must find it to within a pixel or two, as the
 * desktop's suite demands of the Python.
 */
@RunWith(AndroidJUnit4::class)
class BoardDetectTest {

    @Test
    fun theBoardIsFoundWhereItWasDrawn() {
        val failures = ArrayList<String>()
        for (fixture in Fixtures.all()) {
            val bgr = Fixtures.mat(fixture)
            val gray = BoardDetect.toGray(bgr)
            bgr.release()
            val began = System.currentTimeMillis()
            val boards = BoardDetect.detect(gray)
            val took = System.currentTimeMillis() - began
            if (boards.isEmpty()) {
                failures.add("${fixture.name}: no board found (${took} ms)")
                continue
            }
            val board = boards.first()
            val dx = abs(board.region.x - fixture.x)
            val dy = abs(board.region.y - fixture.y)
            val dcell = abs(board.cellPx - fixture.cell)
            Log.i(
                "sidecoach.vision",
                "${fixture.name}: (${board.region.x},${board.region.y}) cell ${"%.2f".format(board.cellPx)} " +
                    "fit ${"%.2f".format(board.fit)} occupied ${board.occupied} in $took ms " +
                    "(truth ${fixture.x},${fixture.y} cell ${fixture.cell})",
            )
            if (dx > 2 || dy > 2) failures.add("${fixture.name}: off by ($dx,$dy) px")
            if (dcell > 0.5) failures.add("${fixture.name}: cell off by ${"%.2f".format(dcell)}")
            if (board.fit < 0.9) failures.add("${fixture.name}: fit ${"%.2f".format(board.fit)}")
            if (fixture.name.startsWith("start") && board.occupied < 24) {
                failures.add("${fixture.name}: only ${board.occupied} occupied cells; 32 pieces stand there")
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
