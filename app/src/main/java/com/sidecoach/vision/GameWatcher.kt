package com.sidecoach.vision

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.sidecoach.events.Event
import com.sidecoach.events.GameEnded
import com.sidecoach.events.GameStarted
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.Resync
import com.sidecoach.events.ScreenState
import com.sidecoach.events.ScreenStateChanged
import com.sidecoach.events.Takeback
import com.sidecoach.state.GamePipeline
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File

/**
 * A frame of the screen, watched: the board found and kept, its theme learned from the start
 * position, every square named, and the whole folded into a legal game.
 *
 * This is the desktop's live loop (`cli.py`'s watcher and `ui/panel.py`'s) rather than a
 * reader of positions: a frame goes through the change gate -- gated on the board's own
 * region, so a board that has not moved costs nothing however busy the rest of the screen
 * is -- and then through the pipeline, which answers with the game's events.
 */
class GameWatcher(context: Context, themeDirectory: File? = null) : AutoCloseable {

    private val embedder: CellEmbedder? = runCatching { CellEmbedder.load(context) }
        .onFailure { Log.w(TAG, "the piece net did not load: $it") }
        .getOrNull()

    private val store = themeDirectory?.let { ThemeStore(it) }
    private val pipeline: GamePipeline? = embedder?.let { GamePipeline(it, store) }
    private val gate = ChangeGate()
    private var frameId = 0
    private var reads = 0

    /** What one frame yielded, for the screen and the log. */
    data class Watch(
        val boardFound: Boolean,
        val region: Region? = null,
        val cellPx: Double = 0.0,
        val fit: Double = 0.0,
        val verified: Boolean = false,
        val changed: Boolean = true,
        /** The game as it stands: null until a position has been recognised. */
        val fen: String? = null,
        val ply: Int = 0,
        val pgn: String = "",
        val moves: List<String> = emptyList(),
        val userSide: String? = null,
        val state: ScreenState = ScreenState.UNKNOWN,
        val finished: String? = null,
        /** Cells the net had to embed on this frame: 64 at a bootstrap, a handful after. */
        val fresh: Int = 0,
        val unsure: Int = 0,
        val detectMs: Long = 0,
        val readMs: Long = 0,
    )

    private var state: ScreenState = ScreenState.UNKNOWN
    private val spoken = ArrayList<String>()

    fun read(bitmap: Bitmap, ts: Double): Watch? {
        val pipeline = pipeline ?: return null
        val bgr = toBgr(bitmap) ?: return null
        try {
            var began = SystemClock.elapsedRealtime()
            val pixels = FramePixels.of(bgr)
            // The desktop moves the gate onto the board before deciding a frame changed.
            gate.region = ChangeGate.boardGate(pipeline.board, pixels.width, pixels.height)
            val changed = gate.changed(pixels, ts)
            val gateMs = SystemClock.elapsedRealtime() - began
            began = SystemClock.elapsedRealtime()
            val events = pipeline.process(bgr, ts, changed, frameId++, pixels)
            val took = SystemClock.elapsedRealtime() - began
            for (event in events) note(event)
            if (++reads % 5 == 1) {
                Log.i(
                    TAG,
                    "frame $reads: gate $gateMs ms, board ${pipeline.detectMs} ms, " +
                        "cells ${pipeline.classifyMs} ms, cold ${pipeline.coldMs} ms, " +
                        "all $took ms, changed=$changed",
                )
            }
            val board = pipeline.board
            val rec = pipeline.reconciler
            return Watch(
                boardFound = board != null,
                region = board?.region,
                cellPx = board?.cellPx ?: 0.0,
                fit = board?.fit ?: 0.0,
                verified = pipeline.tracker.verified,
                changed = changed,
                fen = if (rec != null && rec.started) rec.board.fen() else null,
                ply = rec?.board?.moveStack?.size ?: 0,
                pgn = rec?.pgn() ?: "",
                moves = ArrayList(spoken),
                userSide = pipeline.bottom?.name?.lowercase(),
                state = state,
                finished = rec?.result,
                fresh = pipeline.model?.lastFresh ?: 0,
                unsure = pipeline.readings?.count { it.kind == Pieces.Kind.UNKNOWN } ?: 0,
                detectMs = pipeline.detectMs,
                readMs = took,
            )
        } finally {
            bgr.release()
        }
    }

    /** The events, as a line in the log and a line on the screen. */
    private fun note(event: Event) {
        when (event) {
            is GameStarted -> {
                spoken.clear()
                Log.i(TAG, "game started (${event.kind}), ${event.userSide} at the bottom")
            }
            is MovePlayed -> {
                spoken.add(event.san)
                if (spoken.size > 40) spoken.removeAt(0)
                Log.i(
                    TAG,
                    "${event.ply}. ${event.san} (${event.uci}) " +
                        "${if (event.byUser) "you" else "them"}, ${event.thinkMs ?: 0} ms",
                )
            }
            is Takeback -> Log.i(TAG, "takeback of ${event.plies} plies")
            is Resync -> Log.i(TAG, "resync: ${event.reason}")
            is GameEnded -> Log.i(TAG, "game over ${event.result}: ${event.reason} — ${event.pgn}")
            is ScreenStateChanged -> {
                state = event.state
                Log.i(TAG, "screen: ${event.state}")
            }
            else -> Unit
        }
    }

    private fun toBgr(bitmap: Bitmap): Mat? {
        if (!loaded) return null
        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        val bgr = Mat()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        rgba.release()
        return bgr
    }

    override fun close() {
        embedder?.close()
    }

    companion object {
        private const val TAG = "sidecoach.game"

        /** OpenCV's native library, loaded once for the process. */
        val loaded: Boolean by lazy {
            OpenCVLoader.initLocal().also { if (!it) Log.e(TAG, "OpenCV did not load") }
        }
    }
}
