package com.sidecoach.vision

import kotlin.math.abs

/**
 * Which frames count as changed: `sidecoach/replay/recording.py`'s gate, ported.
 *
 * On a busy desktop something changes every tick, so the difference is measured inside the
 * gated region -- the board plus a margin -- and a keyframe is taken every second regardless.
 * On a phone it matters more: the status bar's clock changes the frame every minute and the
 * screen's own animations change it constantly, and a board that has not moved must cost
 * nothing to look at. A frame the gate calls unchanged skips the recogniser entirely; the
 * tracker and the reconciler's timers still run, as they do on the desktop.
 *
 * `due` asks and `commit` moves the baseline; `changed` does both.
 */
class ChangeGate(
    private val threshold: Int = 12,
    private val minFraction: Double = 0.0005,
    private val keyframeIntervalS: Double? = 1.0,
) {

    /** The region the difference is measured in; the whole frame when null. */
    var region: Region? = null

    private var last: ByteArray? = null
    private var lastWidth = 0
    private var lastHeight = 0
    private var lastTs = -1e9

    /** Would this frame count as a change? The baseline is left where it is. */
    fun due(pixels: FramePixels, ts: Double): Boolean {
        val baseline = last
        if (baseline == null || lastWidth != pixels.width || lastHeight != pixels.height) return true
        var gate = region?.let { clampRegion(it, pixels.width, pixels.height) }
        if (gate != null && (gate.w == 0 || gate.h == 0)) gate = null
        measured = pixels
        measuredRegion = region
        return framesDiffer(baseline, pixels, gate) ||
            (keyframeIntervalS != null && ts - lastTs >= keyframeIntervalS)
    }

    // The pixels the difference was last measured on, and the region it was measured in.
    private var measured: FramePixels? = null
    private var measuredRegion: Region? = null

    /**
     * [changed] for a tick that brings the very pixels of the last one (the phone's display
     * sends nothing while the screen is still), measured in the same region. Their difference
     * from the baseline cannot be over the threshold: either the last frame was committed and
     * the baseline is these pixels, or it was not, and these pixels were measured against this
     * baseline and found below it. So only the keyframe clock can make them due -- the same
     * answer [changed] gives, without comparing the board's pixels again.
     */
    fun changedAgain(pixels: FramePixels, ts: Double): Boolean {
        if (pixels !== measured || region != measuredRegion) return changed(pixels, ts)
        val due = keyframeIntervalS != null && ts - lastTs >= keyframeIntervalS
        if (due) commit(pixels, ts)
        return due
    }

    /** Make this frame the baseline later frames are compared against. */
    fun commit(pixels: FramePixels, ts: Double) {
        // A baseline already copied from these very pixels holds them: only its time moves.
        if (pixels !== committed) {
            val into = last?.takeIf { it.size == pixels.bytes.size } ?: ByteArray(pixels.bytes.size)
            System.arraycopy(pixels.bytes, 0, into, 0, pixels.bytes.size)
            last = into
            lastWidth = pixels.width
            lastHeight = pixels.height
            committed = pixels
        }
        lastTs = ts
    }

    private var committed: FramePixels? = null

    fun changed(pixels: FramePixels, ts: Double): Boolean {
        val due = due(pixels, ts)
        if (due) commit(pixels, ts)
        return due
    }

    /**
     * True when more than `minFraction` of the compared pixels differ by over `threshold`. A
     * moved piece changes at least two cells; a cursor twitch changes a few dozen pixels.
     */
    private fun framesDiffer(baseline: ByteArray, pixels: FramePixels, gate: Region?): Boolean {
        val x0 = gate?.x ?: 0
        val y0 = gate?.y ?: 0
        val x1 = if (gate != null) gate.x + gate.w else pixels.width
        val y1 = if (gate != null) gate.y + gate.h else pixels.height
        val area = (x1 - x0) * (y1 - y0)
        if (area <= 0) return false
        // The desktop takes each pixel's largest channel difference, counts the pixels over
        // the threshold, and compares that with a share of the compared pixels.
        val needed = minFraction * area
        var changed = 0
        val bytes = pixels.bytes
        for (y in y0 until y1) {
            var at = (y * pixels.width + x0) * 3
            for (x in x0 until x1) {
                var worst = 0
                for (c in 0 until 3) {
                    val d = abs((bytes[at + c].toInt() and 0xFF) - (baseline[at + c].toInt() and 0xFF))
                    if (d > worst) worst = d
                }
                if (worst > threshold) {
                    changed += 1
                    if (changed > needed) return true
                }
                at += 3
            }
        }
        return changed > needed
    }

    companion object {

        fun clampRegion(region: Region, width: Int, height: Int): Region {
            val x0 = maxOf(0, region.x)
            val y0 = maxOf(0, region.y)
            val x1 = minOf(width, region.x + region.w)
            val y1 = minOf(height, region.y + region.h)
            return Region(x0, y0, maxOf(0, x1 - x0), maxOf(0, y1 - y0))
        }

        /** The board plus half a cell of margin, clamped to the frame; null without a board. */
        fun boardGate(board: BoardCandidate?, width: Int, height: Int): Region? {
            if (board == null) return null
            val pad = pyRound(board.cellPx / 2)  // `int(round(...))`: halves to even
            val r = board.region
            val gate = clampRegion(
                Region(r.x - pad, r.y - pad, r.w + 2 * pad, r.h + 2 * pad), width, height,
            )
            // The board lies off the frame; gate on the whole frame instead.
            if (gate.w == 0 || gate.h == 0) return null
            return gate
        }
    }
}
