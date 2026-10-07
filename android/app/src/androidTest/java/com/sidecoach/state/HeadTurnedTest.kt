package com.sidecoach.state

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.InProcess
import com.sidecoach.Lane
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Event
import com.sidecoach.events.GameStarted
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.PositionChanged
import com.sidecoach.events.Resync
import com.sidecoach.vision.BoardGeometry
import com.sidecoach.vision.CellEmbedder
import com.sidecoach.vision.ChangeGate
import com.sidecoach.vision.FramePixels
import com.sidecoach.vision.ThemeStore
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

/**
 * `test_a_resync_on_its_head_keeps_the_opening_move_it_found_the_right_way_up`: a resync adopted
 * the next puzzle on its head, its opening move found between two boards on their heads, before
 * the labels turned it round. Replaced by the board read the right way, its move was kept -- a
 * "1... Nd5" no one played -- and the puzzle's own first move was lost (Tai's Pixel, 0.1.72). The
 * pipeline fixture `puzzle-flip-green` is read to its puzzle for Black, after White's Nf3, still
 * taken for White's side; [GamePipeline]'s resyncReoriented is handed what such a resync said.
 */
@Lane(Lane.Kind.VISION)
@InProcess(1)
@RunWith(AndroidJUnit4::class)
class HeadTurnedTest {

    @Test
    fun aResyncOnItsHeadKeepsTheOpeningMoveItFoundTheRightWayUp() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val index = instrumentation.context.assets.open("pipeline/index.json").bufferedReader().use { it.readText() }
        val runs = JSONObject(index).getJSONArray("runs")
        val run = (0 until runs.length()).map { runs.getJSONObject(it) }.first { it.getString("name") == "puzzle-flip-green" }
        val frames = run.getJSONArray("frames")
        val themes = java.io.File(instrumentation.targetContext.cacheDir, "head-turned-themes").apply {
            deleteRecursively()
            mkdirs()
        }
        CellEmbedder.load(instrumentation.targetContext, threads = 2).use { net ->
            val store = ThemeStore(themes)
            var pipeline = GamePipeline(net, store)
            var gate = ChangeGate()
            var last: Mat? = null
            // The fixture to the puzzle for Black's first frame: the White puzzle's game followed,
            // and the next board read once, still the White side's way up.
            val black = (0 until frames.length()).first {
                frames.getJSONObject(it).getString("file") == "puzzle-flip-green/003.png"
            }
            for (f in 0..black) {
                val frame = frames.getJSONObject(f)
                if (frame.optBoolean("session")) {
                    pipeline = GamePipeline(net, store)
                    gate = ChangeGate()
                }
                last?.release()
                val mat = load(frame.getString("file"))
                last = mat
                val pixels = FramePixels.of(mat)
                gate.region = ChangeGate.boardGate(pipeline.board, pixels.width, pixels.height)
                val changed = gate.changed(pixels, frame.getDouble("ts"))
                pipeline.process(mat, frame.getDouble("ts"), changed, f, pixels)
            }
            val lock = pipeline.board
            assertNotNull("no board locked", lock)
            val geometry = BoardGeometry(lock!!.region, lock.cellPx)
            val first = Board("r3k2r/pp3ppp/2n1b3/2bpp3/8/1PN1P3/PBP2PPP/R3KBNR w KQkq - 0 1")
            val puzzle = Board(first.fen())
            val opening = puzzle.parseSan("Nf3")!!
            puzzle.push(opening)
            // What the resync said: the first board and the move as read on their heads.
            val headFirst = turned(first)
            val headAfter = turned(puzzle)
            val headMove = Move(63 - opening.from, 63 - opening.to)
            val resynced = mutableListOf<Event>(
                Resync(gameId = "head", fen = headFirst.fen(), reason = "adopted; its first move seen"),
                PositionChanged(gameId = "head", fen = headFirst.fen(), ply = 0),
                MovePlayed(
                    gameId = "head", ply = 1, san = "Nc6", uci = headMove.uci(), fenBefore = headFirst.fen(),
                    fenAfter = headAfter.fen(), byUser = true,
                ),
                PositionChanged(gameId = "head", fen = headAfter.fen(), ply = 1),
            )
            val method = GamePipeline::class.java.getDeclaredMethod(
                "resyncReoriented", MutableList::class.java, Mat::class.java, BoardGeometry::class.java,
                Double::class.javaPrimitiveType,
            ).apply { isAccessible = true }
            @Suppress("UNCHECKED_CAST")
            val events = method.invoke(pipeline, resynced, last, geometry, frames.getJSONObject(black).getDouble("ts") + 0.1)
                as List<Event>
            assertEquals(events.toString(), listOf("g1f3"), events.filterIsInstance<MovePlayed>().map { it.uci })
            val started = events.filterIsInstance<GameStarted>()
            assertTrue(events.toString(), started.size == 1)
            assertEquals(first.boardFen(), started[0].startFen.split(" ")[0])
            last?.release()
        }
        themes.deleteRecursively()
    }

    /** A board turned round: square s at 63 - s, the colours as they were. */
    private fun turned(board: Board): Board {
        val out = Board(null)
        for ((square, piece) in board.pieceMap()) out.setPieceAt(63 - square, piece)
        out.turn = board.turn
        return out
    }

    private fun load(file: String): Mat {
        val bitmap = InstrumentationRegistry.getInstrumentation().context.assets.open("pipeline/$file").use {
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
        @JvmStatic
        @BeforeClass
        fun loadOpenCv() {
            assertTrue("OpenCV did not load", OpenCVLoader.initLocal())
        }
    }
}
