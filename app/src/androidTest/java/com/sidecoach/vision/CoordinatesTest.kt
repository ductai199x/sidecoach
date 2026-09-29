package com.sidecoach.vision

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
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
     * `test_orientation_is_read_from_the_rank_labels`: the labels learned from a start with White
     * at the bottom read either orientation from the edge digits alone -- Black at the bottom
     * included (the parity audit, 2026-09-29: no phone test read a board the other way up from
     * the side its labels were learned on).
     */
    @Test
    fun theOrientationIsReadFromTheRankLabels() {
        val white = makeBoard(Pieces.Side.WHITE)
        val black = makeBoard(Pieces.Side.BLACK)
        val labels = Coordinates.learnRankLabels(white, GEOM, Pieces.Side.WHITE)
        assertEquals((1..8).toList(), labels.keys.sorted())
        assertEquals(Pieces.Side.WHITE, Coordinates.readLabels(white, GEOM, labels))
        assertEquals(Pieces.Side.BLACK, Coordinates.readLabels(black, GEOM, labels))
        white.release(); black.release()
    }

    /**
     * `test_no_coordinates_read_no_side_so_the_caller_can_ask`: a board with its coordinates
     * turned off has nothing to read, labels learned or not, and no labels learned read nothing
     * themselves (the parity audit, 2026-09-29).
     */
    @Test
    fun noCoordinatesReadNoSideSoTheCallerCanAsk() {
        val white = makeBoard(Pieces.Side.WHITE)
        val bare = makeBoard(null)
        val labels = Coordinates.learnRankLabels(white, GEOM, Pieces.Side.WHITE)
        assertEquals(null, Coordinates.readBottomSide(bare, GEOM, labels))
        assertEquals(null, Coordinates.readBottomSide(bare, GEOM, emptyMap()))
        assertEquals(null, Coordinates.readLabels(white, GEOM, emptyMap()))
        white.release(); bare.release()
    }

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

    /**
     * `test_the_8_reads_the_side_with_nothing_learned` (desktop 495dee2), its real boards:
     * where the 8 is, at the top of the left edge or at its foot, says which side is at the
     * bottom with nothing learned -- read through labels cached as solid blocks, as Tai's
     * were: the board of his puzzle on the Pixel (84 px, Black at the bottom) and a chess.com
     * start (80 px, White), its board found by the detector as the desktop's test finds it (the
     * parity audit, 2026-09-29: the geometry was written in here by hand). The desktop's
     * rendered boards are read on the phone by the pipeline fixture `labels-read-past-by-8`.
     */
    @Test
    fun the8ReadsTheSideWithNothingLearned() {
        val solid = (1..8).associateWith { BooleanArray(Coordinates.GLYPH * Coordinates.GLYPH) { true } }
        val puzzle = load("chesscom_puzzle_black_84.png")
        val onPhone = BoardGeometry(Region(0, 0, 672, 672), 84.0)
        assertEquals(Pieces.Side.BLACK, Coordinates.eights(puzzle, onPhone))
        assertEquals(Pieces.Side.BLACK, Coordinates.readBottomSide(puzzle, onPhone, solid))
        val start = load("chesscom_start.png")
        val found = BoardDetect.detect(BoardDetect.toGray(start)).firstOrNull()
        assertTrue("no board found on the chess.com start", found != null)
        val board = BoardGeometry(found!!.region, found.cellPx)
        assertEquals(Pieces.Side.WHITE, Coordinates.readBottomSide(start, board, solid))
        puzzle.release(); start.release()
    }

    private fun load(file: String): Mat {
        val context = InstrumentationRegistry.getInstrumentation().context
        val bitmap = context.assets.open("coordinates/$file").use {
            android.graphics.BitmapFactory.decodeStream(it)
        }
        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        val bgr = Mat()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        rgba.release()
        bitmap.recycle()
        return bgr
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
