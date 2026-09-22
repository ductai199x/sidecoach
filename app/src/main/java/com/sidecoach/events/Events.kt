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

/** One line the engine gave, from the mover's point of view: `events.Line`. */
data class Line(
    val moves: List<String>,
    val depth: Int,
    val expectedScore: Double,
    val scoreCp: Int? = null,
    val mate: Int? = null,
    val wdl: Triple<Int, Int, Int>? = null,
)

/** A tactical motif the position or the move carries: `events.MotifTag`. */
data class MotifTag(
    val name: String,
    val squares: List<String> = emptyList(),
    val move: String? = null,
)

class Analysis(
    val gameId: String,
    val fen: String,
    val ply: Int,
    val lines: List<Line>,
    val decisive: Boolean,
    val onlyMove: Boolean,
    val threats: List<String> = emptyList(),
    val tags: List<MotifTag> = emptyList(),
    val opening: String? = null,
) : Event()

enum class Severity { NONE, INACCURACY, MISTAKE, BLUNDER }

/**
 * What a played move was, in the vocabulary the user already reads on chess.com.
 *
 * Severity says how much a move cost; the label says what kind of move it was, which needs
 * the move before it and the opening book as well: a miss is a chance the opponent handed
 * over and the mover did not take, a great move is one that took it or was the only move, a
 * brilliant move is a sound sacrifice, a book move is still in theory.
 */
enum class Label { BOOK, BRILLIANT, GREAT, BEST, EXCELLENT, GOOD, INACCURACY, MISTAKE, MISS, BLUNDER }

class MoveAssessment(
    val gameId: String,
    val ply: Int,
    val uci: String,
    val expectedLoss: Double,
    val severity: Severity,
    val best: List<String>,
    val tags: List<MotifTag> = emptyList(),
    /**
     * The shallower of the two searches the loss was measured from. A position that hit the
     * engine's time cap before reaching the depth asked for makes the comparison less exact.
     */
    val depth: Int = 0,
    val label: Label? = null,
    /** Per-move accuracy on the steepened curve, 0 to 100, so a game can be summed up. */
    val accuracy: Double? = null,
) : Event()

class ClockObservation(
    val whiteMs: Int? = null,
    val blackMs: Int? = null,
    val running: Pieces.Side? = null,
) : Event()
