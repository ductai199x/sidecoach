package com.sidecoach.vision

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.InProcess
import com.sidecoach.Lane
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
@Lane(Lane.Kind.FREE)
@InProcess(3)
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

    /**
     * A compact lost-board sweep: the game board at 50 px cells resized past the near range,
     * another game's board on the screen at the old size, the game played on while the board
     * was lost. With six plies played, a board found counted as the lost one only within eight
     * cells of its occupancy; the shrunk game board was not, the detector put the bigger board
     * first, and the other game was locked for good (round 3, finding 1's residual). Of the
     * boards found, the nearest in occupancy is taken (desktop
     * `test_a_resized_board_is_found_beside_another_board_of_its_old_size`).
     */
    @Test
    fun aResizedBoardIsFoundBesideAnotherBoardOfItsOldSize() {
        for (newCell in listOf(30, 37, 80)) {
            for (sameTheme in listOf(true, false)) {
                for (played in listOf(false, true)) {
                    val case = "new cell $newCell, same theme $sameTheme, six plies $played"
                    val before = FloatArray(W * H) { 32f }
                    draw(before, GAME_X, 40, 50, GAME, light = 200f, dark = 120f)
                    val after = FloatArray(W * H) { 32f }
                    draw(after, GAME_X, 40, newCell, if (played) GAME_PLAYED else GAME, light = 200f, dark = 120f)
                    val (light, dark) = if (sameTheme) 200f to 120f else 225f to 150f
                    draw(after, 1100, 60, 50, OTHER, light, dark)
                    val tracker = BoardTracker()
                    repeat(3) { tracker.update(Plane(W, H, before)) }
                    var board: BoardCandidate? = null
                    repeat(30) { board = tracker.update(Plane(W, H, after)) }
                    val found = checkNotNull(board) { "$case: no lock" }
                    assertTrue("$case: not verified", tracker.verified)
                    assertTrue("$case: locked $found", abs(found.region.x - GAME_X) <= 2)
                    assertTrue("$case: locked $found", abs(found.cellPx - newCell) <= 1.5)
                }
            }
        }
    }

    /**
     * The last frame that validated the lock was kept whole, for the lost board's pieces (round
     * 3): only the board's region is kept, and it reads the same pieces (desktop
     * `test_a_lock_keeps_its_board_s_pixels_not_the_frame`).
     */
    @Test
    fun aLockKeepsItsBoardsPixelsNotTheFrame() {
        val frame = page(gameCell = 50, thumbnail = false, sameTheme = true)
        val tracker = BoardTracker()
        val board = checkNotNull(tracker.update(frame))
        val (pixels, placed) = checkNotNull(tracker.seen)
        assertEquals(board.region.w, pixels.width)
        assertEquals(board.region.h, pixels.height)
        val whole = checkNotNull(BoardRefine.boardOccupancy(frame, board.region, board.cellPx))
        val kept = checkNotNull(BoardRefine.boardOccupancy(pixels, placed.region, placed.cellPx))
        assertTrue("no piece read", whole.any { it })
        assertTrue("the kept pixels read other pieces", whole.contentEquals(kept))
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

        // GAME after six plies played while the board was lost: d4 Kh8 g4 d5 exd5 Be6.
        private val GAME_PLAYED = listOf("r..q.r.k", "..p.bppp", "p.n.bn..", ".p.Pp...", "...P..P.", ".B...N..", "PPP..P.P", "RNBQR.K.")
        private val OTHER = listOf("rnbqkb.r", "pp..pppp", "...p.n..", "........", "...NP...", "........", "PPP..PPP", "RNBQKB.R")

        @JvmStatic
        @BeforeClass
        fun loadOpenCv() {
            assertTrue("OpenCV did not load", OpenCVLoader.initLocal())
        }
    }
}
