package com.sidecoach.state

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.InProcess
import com.sidecoach.Lane
import com.sidecoach.TestNets
import com.sidecoach.chess.Board
import com.sidecoach.events.ClockObservation
import com.sidecoach.events.ScreenState
import com.sidecoach.vision.BoardCandidate
import com.sidecoach.vision.BoardGeometry
import com.sidecoach.vision.CellEmbedder
import com.sidecoach.vision.ClockReader
import com.sidecoach.vision.ClockReading
import com.sidecoach.vision.Pieces.CellReading
import com.sidecoach.vision.Pieces.Kind
import com.sidecoach.vision.Pieces.Side
import com.sidecoach.vision.Region
import com.sidecoach.vision.Which
import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

/**
 * The pipeline's parts that need no game played through it: the desktop's tests that call them
 * directly. The clock path (`tests/test_clock_pipeline.py`: one observation then no repeat,
 * nothing until the board is oriented, a broken reader costs nothing, no reader wires nothing,
 * and a new game forgets the last one's clock -- on a reset, parity st-01, and on a recogniser
 * rebuilt from the next start position, st-02); why a game ended, from the board and the clock
 * (`test_result_badges.py`'s `test_the_reason_comes_from_the_board`); the screen state from the
 * game's life (`test_screen_state.py`); and a stray reading beside the start position
 * (`test_pipeline.py`). The clock's own mapping is `ClockTest`'s; games through the pipeline
 * are `PipelineTest`'s.
 */
@Lane(Lane.Kind.VISION)
@InProcess(2)
@RunWith(AndroidJUnit4::class)
class PipelinePartsTest {

    private val frame: Mat get() = Mat.zeros(8, 8, CvType.CV_8UC3)
    private val board = BoardCandidate(region = Region(0, 0, 80, 80), cellPx = 10.0, score = 1.0, contrast = 1.0)

    /** The start position of `make_pipeline_fixtures.py`'s games: the board at (8, 180), 50 px. */
    private val startGeometry = BoardGeometry(Region(8, 180, 400, 400), 50.0)

    private fun reading(top: Int?, bottom: Int?, running: Which? = null) =
        ClockReading(topMs = top, bottomMs = bottom, running = running, confidence = 0.9)

    /** A clock reader that returns a canned reading and counts its resets, no pixels read. */
    private class StubReader(private val out: ClockReading?) : ClockReader(templates = null) {
        var resets = 0

        override fun read(frame: Mat, board: BoardCandidate, ts: Double?): ClockReading? = out

        override fun reset() {
            resets += 1
        }
    }

    private fun oriented(reader: ClockReader?): GamePipeline {
        val pipe = GamePipeline(net)
        pipe.clock = reader
        pipe.bottom = Side.WHITE
        pipe.board = board
        return pipe
    }

    @Test
    fun theClockIsSaidOnceAndThenOnlyWhenItChanges() {
        val pipe = oriented(StubReader(reading(120_000, 95_000, Which.BOTTOM)))
        val first = pipe.readClock(frame, 1.0)
        assertTrue(first.size == 1 && first[0] is ClockObservation)
        // The identical reading on the next tick is not repeated into the event stream.
        assertTrue(pipe.readClock(frame, 1.1).isEmpty())
        // A changed reading (a second ticked off) is emitted again.
        pipe.clock = StubReader(reading(120_000, 94_000, Which.BOTTOM))
        assertEquals(1, pipe.readClock(frame, 2.0).size)
    }

    @Test
    fun noClockIsReadUntilTheBoardIsOrientedAndPresent() {
        val pipe = oriented(StubReader(reading(120_000, 95_000)))
        pipe.bottom = null  // no game oriented yet
        assertTrue(pipe.readClock(frame, 1.0).isEmpty())
        pipe.bottom = Side.WHITE
        pipe.board = null  // board lost this frame
        assertTrue(pipe.readClock(frame, 1.0).isEmpty())
    }

    @Test
    fun aBrokenAbsentOrMissingClockCostsOnlyTheClock() {
        val boom = object : ClockReader(templates = null) {
            override fun read(frame: Mat, board: BoardCandidate, ts: Double?): ClockReading? =
                throw RuntimeException("a dialog is drawn over the clock")
        }
        assertTrue(oriented(boom).readClock(frame, 1.0).isEmpty())  // an exception is swallowed
        assertTrue(oriented(StubReader(null)).readClock(frame, 1.0).isEmpty())  // no clock drawn
        val inert = oriented(null)  // built without a reader: the clock path does nothing
        assertNull(inert.clock)
        assertTrue(inert.readClock(frame, 1.0).isEmpty())
    }

    @Test
    fun aResetForgetsTheLastGamesClock() {
        // The same reading is news again, and the reader's own memory of who was running goes
        // too (the phone kept both: st-01).
        val reader = StubReader(reading(120_000, 95_000, Which.BOTTOM))
        val pipe = oriented(reader)
        assertEquals(1, pipe.readClock(frame, 1.0).size)
        pipe.reset()
        assertEquals(1, reader.resets)
        assertNull(pipe.lastClock)
        assertNull(pipe.lastClockTs)
        pipe.bottom = Side.WHITE
        pipe.board = board
        assertEquals(1, pipe.readClock(frame, 5.0).size)
    }

    @Test
    fun aRecogniserRebuiltFromTheNextStartPositionForgetsTheClock() {
        // `_rebuild_model`: the next game's start position, and its clock is not the last
        // game's (the phone kept it: st-02).
        val reader = StubReader(reading(120_000, 95_000, Which.BOTTOM))
        val pipe = oriented(reader)
        assertEquals(1, pipe.readClock(frame, 1.0).size)
        assertNotNull(pipe.lastClock)
        val start = load("mate-green/000.png")
        try {
            assertTrue("no recogniser learned", pipe.rebuildModel(start, startGeometry))
        } finally {
            start.release()
        }
        assertEquals(1, reader.resets)
        assertNull(pipe.lastClock)
        pipe.board = board
        assertEquals("the same reading is news again", 1, pipe.readClock(frame, 5.0).size)
    }

    @Test
    fun aBoardTurnedRoundForgetsTheClockReadTheOtherWayUp() {
        // Round 2, finding 3 (desktop `test_a_board_turned_round_forgets_the_clock_read_the_other_way_up`):
        // a game adopted upside down read the running bottom clock as White's; the Flip re-read
        // the board with Black at the bottom and left the side to move open, and the next tick
        // settled it from that stale observation.
        val reader = StubReader(reading(192_000, 295_000, Which.BOTTOM))
        val pipe = oriented(reader)
        val start = load("mate-green/000.png")
        try {
            assertTrue("no recogniser learned", pipe.rebuildModel(start, startGeometry))
        } finally {
            start.release()
        }
        pipe.board = board
        val learned = checkNotNull(pipe.bottom)
        assertEquals(1, pipe.readClock(frame, 1.0).size)
        assertEquals("the running bottom clock is the bottom side's", learned, pipe.lastClock?.third)
        val resets = reader.resets
        val empty = MutableList(64) { CellReading(Kind.EMPTY, confidence = 1.0) }
        val position = Board("r1bq1rk1/2ppbppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQR1K1 b - - 5 8")
        val model = checkNotNull(pipe.model)
        assertTrue(pipe.adoptCold(GamePipeline.ColdRead(model, learned.other, empty, position), 1.05).isNotEmpty())
        assertEquals(learned.other, pipe.bottom)
        assertNull(pipe.lastClock)
        assertNull(pipe.lastClockTs)
        assertEquals(resets + 1, reader.resets)
    }

    @Test
    fun theReasonTheGameEndedComesFromTheBoardAndTheClock() {
        val pipe = GamePipeline(net)
        val mate = Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        assertEquals("checkmate", pipe.endReason(mate, "0-1"))
        assertEquals("resignation", pipe.endReason(Board(), "1-0"))
        pipe.lastClock = Triple(5000, 0, null)
        assertEquals("lost on time", pipe.endReason(Board(), "1-0"))
        assertEquals("resignation", pipe.endReason(Board(), "0-1"))
    }

    @Test
    fun theScreenStateFollowsTheGamesLife() {
        val pipe = GamePipeline(net)
        // No board on screen at all.
        assertEquals(ScreenState.UNKNOWN, pipe.deriveScreenState(boardPresent = false))
        // A board, but no game recognised on it yet (a menu, a set-up board, the lobby).
        assertEquals(ScreenState.UNKNOWN, pipe.deriveScreenState(boardPresent = true))
        val rec = Reconciler(bottom = Side.WHITE, userSide = Side.WHITE)
        pipe.reconciler = rec
        assertEquals(ScreenState.UNKNOWN, pipe.deriveScreenState(boardPresent = true))  // not started
        rec.startedFlag = true
        assertEquals(ScreenState.LIVE_GAME, pipe.deriveScreenState(boardPresent = true))
        rec.finishedFlag = true
        assertEquals(ScreenState.GAME_OVER, pipe.deriveScreenState(boardPresent = true))
    }

    @Test
    fun aStrayReadingBesideTheStartPositionIsStillTheStart() {
        // A junk reading on an empty square next to the start position, and a king caught off
        // its centre: still the start, so browsing back to it is not a new game.
        val pipe = GamePipeline(net)
        val start = load("mate-green/000.png")
        try {
            assertTrue(pipe.rebuildModel(start, startGeometry))
            val readings = pipe.model!!.classifyFrame(start, startGeometry, net)
            readings[3 * 8 + 3] = CellReading(
                Kind.PIECE, null, Side.WHITE, 0.5, 0.4, 240.0, settled = false,
            )
            readings[0 * 8 + 4] = CellReading(
                Kind.PIECE, "k", Side.BLACK, 0.9, 0.4, 60.0, settled = false,
            )
            assertTrue(pipe.readsAsStart(readings))
        } finally {
            start.release()
        }
    }

    private fun load(file: String): Mat {
        val context = InstrumentationRegistry.getInstrumentation().context
        val bitmap = context.assets.open("pipeline/$file").use {
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

    companion object {
        private lateinit var net: CellEmbedder

        @JvmStatic
        @BeforeClass
        fun setUp() {
            assertTrue("OpenCV did not load", OpenCVLoader.initLocal())
            net = CellEmbedder.load(InstrumentationRegistry.getInstrumentation().targetContext, threads = TestNets.THREADS)
        }

        @JvmStatic
        @AfterClass
        fun close() {
            net.close()
        }
    }
}
