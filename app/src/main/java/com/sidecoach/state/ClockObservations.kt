package com.sidecoach.state

import com.sidecoach.events.ClockObservation
import com.sidecoach.vision.ClockReading
import com.sidecoach.vision.Pieces
import com.sidecoach.vision.Which

/**
 * A clock reading, which is bottom/top relative to the board, keyed by colour:
 * `state/pipeline.py`'s `clock_observation`.
 *
 * The side at the bottom is the user, so the bottom box is the user's clock. chess.com draws
 * white's box light and black's dark, but the board orientation is the game's own truth, so
 * the colours come from [bottom] rather than the box tint.
 */

private const val CLOCK_MAX_MS = GamePipeline.CLOCK_MAX_MS

private fun plausibleMs(ms: Int?): Int? = if (ms != null && ms in 0..CLOCK_MAX_MS) ms else null

/**
 * Map a bottom-relative clock reading to a colour-keyed observation, or null.
 *
 * Null whenever there is nothing trustworthy to report: no reading, confidence below the
 * floor, or both boxes unreadable (an untimed game, the clocks scrolled off screen, a dialog
 * drawn over them). A single unreadable box passes through as that side's time being
 * unknown, which the consumer treats as no update for that side.
 */
fun clockObservation(reading: ClockReading?, bottom: Pieces.Side): ClockObservation? {
    if (reading == null || reading.confidence < GamePipeline.CLOCK_MIN_CONFIDENCE) return null
    val top = plausibleMs(reading.topMs)
    val bottomMs = plausibleMs(reading.bottomMs)
    if (top == null && bottomMs == null) return null
    val whiteMs: Int?
    val blackMs: Int?
    if (bottom == Pieces.Side.WHITE) {
        whiteMs = bottomMs
        blackMs = top
    } else {
        whiteMs = top
        blackMs = bottomMs
    }
    val running = when (reading.running) {
        Which.BOTTOM -> bottom
        Which.TOP -> if (bottom == Pieces.Side.WHITE) Pieces.Side.BLACK else Pieces.Side.WHITE
        null -> null
    }
    return ClockObservation(whiteMs = whiteMs, blackMs = blackMs, running = running)
}
