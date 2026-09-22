package com.sidecoach.vision

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

/**
 * A frame of the screen, read: keep a lock on the board ([BoardTracker], as the desktop's
 * `vision/tracker.py` does), learn its theme when the start position is on it, and name
 * what stands on each square.
 *
 * The theme belongs to the lock: the desktop keeps the square colours a lock was taken
 * with, drops the lock when validation fails, and keys its theme cache by those colours.
 * Here the theme is dropped with the lock for the same reason -- exemplars from one board
 * say nothing about another.
 *
 * What is not here yet is the half that follows: `state/reconcile.py`, the legal game the
 * readings are folded into, and the coach after it. This reads a position, fresh, with no
 * memory of the game.
 */
class ScreenReader(context: Context) : AutoCloseable {

    private val embedder: CellEmbedder? = runCatching { CellEmbedder.load(context) }
        .onFailure { Log.w(TAG, "the piece net did not load: $it") }
        .getOrNull()

    private val tracker = BoardTracker()
    private var model: ThemeModel? = null
    private var startSeen = 0

    /** What one frame yielded. */
    data class Reading(
        val boardFound: Boolean,
        val region: Region? = null,
        val cellPx: Double = 0.0,
        val fit: Double = 0.0,
        val verified: Boolean = false,
        val placement: String? = null,
        val unsure: Int = 0,
        val learned: Boolean = false,
        val detectMs: Long = 0,
        val readMs: Long = 0,
    )

    fun read(bitmap: Bitmap): Reading {
        val bgr = toBgr(bitmap) ?: return Reading(boardFound = false)
        try {
            val gray = BoardDetect.toGray(bgr)
            val began = SystemClock.elapsedRealtime()
            val board = tracker.update(gray)
            val detectMs = SystemClock.elapsedRealtime() - began
            if (board == null) {
                if (model != null) {
                    // The lock is gone, and the exemplars were that board's.
                    Log.i(TAG, "the board is gone; the theme goes with it")
                    model = null
                }
                return Reading(boardFound = false, detectMs = detectMs)
            }
            val geometry = BoardGeometry(board.region, board.cellPx)
            val net = embedder ?: return Reading(
                boardFound = true, region = board.region, cellPx = board.cellPx,
                fit = board.fit, verified = tracker.verified, detectMs = detectMs,
            )

            var learned = false
            if (model == null) {
                // The desktop's order (state/pipeline.py): the cheap occupancy first, the
                // start position's shape from it, and only on START_FRAMES frames in a row
                // -- an animation passes through shapes that look like one -- is the net
                // asked for the 64 exemplars. The orientation comes from the same reading:
                // the brighter pieces are white.
                val cells = ThemeModel.occupancy(bgr, geometry)
                if (ThemeModel.startShaped(cells)) {
                    startSeen++
                    if (startSeen >= START_FRAMES) {
                        val whiteAtBottom = ThemeModel.orientationFromStart(cells)
                        val candidate = ThemeModel.learn(bgr, geometry, net, whiteAtBottom)
                        startSeen = 0
                        if (candidate != null) {
                            model = candidate
                            learned = true
                            Log.i(
                                TAG,
                                "theme learned, ${if (whiteAtBottom) "white" else "black"} at the bottom",
                            )
                        } else {
                            Log.i(TAG, "the start frame is spoiled; waiting for a clean one")
                        }
                    }
                } else {
                    startSeen = 0
                }
            }
            val theme = model ?: return Reading(
                boardFound = true, region = board.region, cellPx = board.cellPx,
                fit = board.fit, verified = tracker.verified, detectMs = detectMs,
            )
            val readBegan = SystemClock.elapsedRealtime()
            val readings = theme.readBoard(bgr, geometry, net)
            return Reading(
                boardFound = true,
                region = board.region,
                cellPx = board.cellPx,
                fit = board.fit,
                verified = tracker.verified,
                placement = theme.placement(readings),
                unsure = readings.count { it.label == null },
                learned = learned,
                detectMs = detectMs,
                readMs = SystemClock.elapsedRealtime() - readBegan,
            )
        } finally {
            bgr.release()
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
        private const val TAG = "sidecoach.vision"

        /** Frames of the start position's shape before the theme is learned from it. */
        private const val START_FRAMES = 2

        /** OpenCV's native library, loaded once for the process. */
        val loaded: Boolean by lazy {
            OpenCVLoader.initLocal().also { if (!it) Log.e(TAG, "OpenCV did not load") }
        }
    }
}
