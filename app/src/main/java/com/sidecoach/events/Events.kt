package com.sidecoach.events

import com.sidecoach.vision.Pieces

/**
 * What one stage tells the next: the events of `sidecoach/events.py` that the state layer
 * produces. The stages talk only through these, on the phone as on the desktop.
 *
 * Only the game's own events are here. Analysis, coaching and the profile carry their own,
 * and come later.
 */
sealed class Event {
    var ts: Double = 0.0
    var sourceFrameId: Int? = null
}

/** `ScreenState`: what the screen is showing, as the pipeline can tell. */
enum class ScreenState { UNKNOWN, LIVE_GAME, ANALYSIS, PUZZLE, GAME_OVER, COVERED }

class ScreenStateChanged(val state: ScreenState) : Event()

class GameStarted(
    val gameId: String,
    val userSide: Pieces.Side,
    val startFen: String,
    /** "game" from the start position, "position" for a board joined from the middle. */
    val kind: String = "game",
) : Event()

class PositionChanged(val gameId: String, val fen: String, val ply: Int) : Event()

class MovePlayed(
    val gameId: String,
    val ply: Int,
    val san: String,
    val uci: String,
    val fenBefore: String,
    val fenAfter: String,
    val byUser: Boolean,
    val thinkMs: Int? = null,
) : Event()

class Takeback(val gameId: String, val plies: Int, val fen: String) : Event()

class Resync(val gameId: String, val fen: String, val reason: String) : Event()

class GameEnded(
    val gameId: String,
    val result: String,
    val reason: String? = null,
    val pgn: String? = null,
    val finalFen: String? = null,
) : Event()

class ClockObservation(
    val whiteMs: Int? = null,
    val blackMs: Int? = null,
    val running: Pieces.Side? = null,
) : Event()
