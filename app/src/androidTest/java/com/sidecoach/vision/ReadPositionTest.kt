package com.sidecoach.vision

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader

/**
 * The whole read, end to end on a rendered frame: find the board, learn the theme from the
 * start position, then read a later position off the same theme and check the placement
 * against the FEN it was drawn from.
 *
 * This is what the coach stands on. A wrong square here is a phantom move later.
 */
@RunWith(AndroidJUnit4::class)
class ReadPositionTest {

    @Test
    fun theStartPositionTeachesTheThemeAndTheThemeReadsTheGame() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        CellEmbedder.load(context).use { embedder ->
            val fixtures = Fixtures.all().associateBy { it.name }
            // The theme is learned from the green board's start position, then a later
            // position of the same game is read with it, as a session does.
            val start = fixtures.getValue("start-green")
            val startFrame = Fixtures.mat(start)
            val startGeometry = locate(startFrame, start)
            val model = ThemeModel.learn(startFrame, startGeometry, embedder, whiteAtBottom = true)
            assertNotNull("the start position did not read back as itself", model)

            val italian = fixtures.getValue("italian-green")
            val frame = Fixtures.mat(italian)
            val geometry = locate(frame, italian)
            val began = System.currentTimeMillis()
            val readings = model!!.readBoard(frame, geometry, embedder)
            val took = System.currentTimeMillis() - began
            val placement = model.placement(readings)
            val expected = italian.fen.substringBefore(' ')
            Log.i("sidecoach.vision", "read in $took ms: $placement")
            Log.i("sidecoach.vision", "expected:       $expected")
            val unsure = readings.count { it.label == null }
            assertEquals("the position read off the theme (unsure cells: $unsure)", expected, placement)
            frame.release()
            startFrame.release()
        }
    }

    @Test
    fun everyPieceSetAndThemeReadsItsOwnStartPosition() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        CellEmbedder.load(context).use { embedder ->
            val failures = ArrayList<String>()
            for (fixture in Fixtures.all()) {
                val frame = Fixtures.mat(fixture)
                val geometry = locate(frame, fixture)
                val whiteAtBottom = fixture.bottom == "white"
                // A theme learned from this very frame must read it back: that is the
                // bootstrap's own check, and it fails on a board the net cannot tell apart.
                val labels = if (fixture.name.startsWith("start")) {
                    ThemeModel.startLabels(whiteAtBottom)
                } else {
                    null
                }
                if (labels == null) { frame.release(); continue }
                val model = ThemeModel.learn(frame, geometry, embedder, whiteAtBottom)
                if (model == null) {
                    failures.add("${fixture.name}: the start position does not read back as itself")
                } else {
                    val placement = model.placement(model.readBoard(frame, geometry, embedder))
                    val expected = fixture.fen.substringBefore(' ')
                    if (placement != expected) failures.add("${fixture.name}: read $placement")
                }
                frame.release()
            }
            assertTrue(failures.joinToString("\n"), failures.isEmpty())
        }
    }

    /** The board as the detector finds it, checked against where it was drawn. */
    private fun locate(frame: org.opencv.core.Mat, fixture: Fixture): BoardGeometry {
        val boards = BoardDetect.detect(BoardDetect.toGray(frame))
        assertTrue("${fixture.name}: no board found", boards.isNotEmpty())
        val board = boards.first()
        return BoardGeometry(board.region, board.cellPx)
    }

    companion object {
        @JvmStatic
        @BeforeClass
        fun loadOpenCv() {
            assertTrue("OpenCV did not load", OpenCVLoader.initLocal())
        }
    }
}
