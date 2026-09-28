package com.sidecoach.vision

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

/**
 * Reading board orientation from the coordinate labels: `tests/test_coordinates.py`'s
 * synthetic board, drawn the same way -- rank digits in the top-left of each left-edge cell,
 * 8..1 down for White at the bottom, 1..8 for Black, as chess.com draws them -- with OpenCV's
 * own Hershey font, as the desktop's test draws it.
 */
@RunWith(AndroidJUnit4::class)
class CoordinatesTest {

    /**
     * `test_a_start_seen_through_a_lock_off_its_board_learns_no_labels` (desktop 3d0a201): a
     * lock a fifth of a cell out cuts each edge cell partly from its neighbour: the square's
     * colour is taken from the neighbour and the whole corner reads as ink. Kept, those solid
     * blocks matched every digit alike, and no board of the set had its side read again (Tai's
     * Pixel, 2026-09-27); none are learned, and the next start teaches them.
     */
    @Test
    fun aStartSeenThroughALockOffItsBoardLearnsNoLabels() {
        val board = makeBoard(Pieces.Side.WHITE)
        for ((dx, dy) in listOf(CELL / 5 to 0, 0 to CELL / 5, 0 to -CELL / 3)) {
            val off = BoardGeometry(Region(dx, dy, CELL * 8, CELL * 8), CELL.toDouble())
            assertEquals("($dx, $dy)", emptyMap<Int, BooleanArray>(), Coordinates.learnRankLabels(board, off, Pieces.Side.WHITE))
        }
        assertEquals((1..8).toList(), Coordinates.learnRankLabels(board, GEOM, Pieces.Side.WHITE).keys.sorted())
        board.release()
    }

    /** An 8x8 board; when [bottom] is set, rank digits are drawn in the left-edge corners. */
    private fun makeBoard(bottom: Pieces.Side?): Mat {
        val img = Mat(CELL * 8, CELL * 8, CvType.CV_8UC3)
        for (row in 0 until 8) {
            for (col in 0 until 8) {
                img.submat(Rect(col * CELL, row * CELL, CELL, CELL))
                    .setTo(if ((row + col) % 2 == 0) LIGHT else DARK)
            }
        }
        if (bottom != null) {
            for (row in 0 until 8) {
                val rank = if (bottom == Pieces.Side.WHITE) 8 - row else row + 1
                val square = if (row % 2 == 0) LIGHT_SUM else DARK_SUM
                val ink = if (square > 400) Scalar(30.0, 30.0, 30.0) else Scalar(240.0, 240.0, 240.0)
                Imgproc.putText(
                    img,
                    rank.toString(),
                    Point(3.0, (row * CELL + 17).toDouble()),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    ink,
                    1,
                    Imgproc.LINE_AA,
                )
            }
        }
        return img
    }

    companion object {
        private const val CELL = 48
        private val GEOM = BoardGeometry(Region(0, 0, CELL * 8, CELL * 8), CELL.toDouble())
        private val LIGHT = Scalar(210.0, 220.0, 190.0)
        private val DARK = Scalar(100.0, 140.0, 90.0)
        private const val LIGHT_SUM = 210 + 220 + 190
        private const val DARK_SUM = 100 + 140 + 90

        @JvmStatic
        @BeforeClass
        fun loadOpenCv() {
            assertTrue("OpenCV did not load", OpenCVLoader.initLocal())
        }
    }
}
