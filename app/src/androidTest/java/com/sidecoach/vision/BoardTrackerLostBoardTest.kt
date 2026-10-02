package com.sidecoach.vision

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import kotlin.math.abs

/**
 * The game board resized past the near search's range with another game's board at the old
 * size beside it: the near search found that board first and the lock stayed on it for good
 * (round 2, finding 1; desktop `test_a_board_resized_past_the_near_range_is_found_not_a_board_of_the_old_size`).
 * A board found near the old size is taken at once only when it shows the lost board's pieces.
 * The pages are grey planes drawn here: two checkerboards, pieces as dark squares inside cells.
 */
@RunWith(AndroidJUnit4::class)
class BoardTrackerLostBoardTest {

    @Test
    fun aBoardResizedPastTheNearRangeIsFoundNotABoardOfTheOldSize() {
        for (sameTheme in listOf(false, true)) {
            val tracker = BoardTracker()
            repeat(3) { tracker.update(page(gameCell = 40, thumbnail = false, sameTheme = sameTheme)) }
            var board: BoardCandidate? = null
            repeat(60) { board = tracker.update(page(gameCell = 100, thumbnail = true, sameTheme = sameTheme)) }
            val found = checkNotNull(board) { "same theme $sameTheme: no lock" }
            assertTrue("same theme $sameTheme: locked $found", abs(found.region.x - GAME_X) <= 2)
            assertTrue("same theme $sameTheme: locked $found", abs(found.cellPx - 100) <= 1.5)
        }
    }

    @Test
    fun aBoardMovedAtItsSizeIsRelockedByTheNearSearch() {
        val tracker = BoardTracker(maxMisses = 2)
        tracker.update(page(gameCell = 60, thumbnail = false, sameTheme = true))
        var board: BoardCandidate? = null
        repeat(4) { board = tracker.update(page(gameCell = 60, thumbnail = false, sameTheme = true, gameX = 700, gameY = 300)) }
        val found = board!!
        assertTrue("locked $found", abs(found.region.x - 700) <= 2)
        assertEquals("the first lock only", 1, tracker.fullSearches)
    }

    private fun page(gameCell: Int, thumbnail: Boolean, sameTheme: Boolean, gameX: Int = GAME_X, gameY: Int = 40): Plane {
        val data = FloatArray(W * H) { 32f }
        draw(data, gameX, gameY, gameCell, GAME, light = 200f, dark = 120f)
        if (thumbnail) {
            val (light, dark) = if (sameTheme) 200f to 120f else 225f to 150f
            draw(data, 1200, 60, 40, OTHER, light, dark)
        }
        return Plane(W, H, data)
    }

    /** A board at ([x0], [y0]) with [cell] px squares; a piece is a dark square in a cell's middle. */
    private fun draw(data: FloatArray, x0: Int, y0: Int, cell: Int, pieces: List<String>, light: Float, dark: Float) {
        for (r in 0 until 8) {
            for (c in 0 until 8) {
                val square = if ((r + c) % 2 == 0) light else dark
                val piece = pieces[r][c] != '.'
                val inset = cell * 3 / 10
                for (y in y0 + r * cell until y0 + (r + 1) * cell) {
                    for (x in x0 + c * cell until x0 + (c + 1) * cell) {
                        val inside = piece && y >= y0 + r * cell + inset && y < y0 + (r + 1) * cell - inset &&
                            x >= x0 + c * cell + inset && x < x0 + (c + 1) * cell - inset
                        data[y * W + x] = if (inside) 30f else square
                    }
                }
            }
        }
    }

    companion object {
        private const val W = 1600
        private const val H = 1000
        private const val GAME_X = 40

        // The desktop test's two positions, rank 8 first.
        private val GAME = listOf("r.bq.rk.", "..ppbppp", "p.n..n..", ".p..p...", "....P...", ".B...N..", "PPPP.PPP", "RNBQR.K.")
        private val OTHER = listOf("rnbqkb.r", "pp..pppp", "...p.n..", "........", "...NP...", "........", "PPP..PPP", "RNBQKB.R")

        @JvmStatic
        @BeforeClass
        fun loadOpenCv() {
            assertTrue("OpenCV did not load", OpenCVLoader.initLocal())
        }
    }
}
