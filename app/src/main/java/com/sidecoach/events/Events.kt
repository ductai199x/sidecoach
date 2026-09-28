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

/** `ScreenState`: what the screen is showing, as the pipeline can tell; the desktop's members. */
enum class ScreenState {
    LIVE_GAME,
    PUZZLES,
    GAME_OVER,

    /**
     * A dialog or a large overlay over the board of a live game: the game may be over (a
     * resignation, a flag, the site's game-over dialog) or merely interrupted; the board cannot
     * say which, so the game stays open and the coach stays quiet.
     */
    COVERED,
    UNKNOWN,
}

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

/**
 * Cells that have disagreed with the game for a long while, or a move refused as illegal: a
 * note for the log, the game kept as it stands. Not a resync -- the position did not jump and
 * the numbering goes on. Sent as a Resync, it made the coach and the engine restart their
 * numbering while the reconciler kept its own (Tai's game on the Pixel, 2026-09-23).
 */
class Unexplained(val gameId: String, val fen: String, val reason: String) : Event()

/**
 * A board on screen, no game followed on it, and nothing read from it for a while (Tai on
 * the Pixel, 2026-09-23: a puzzle up, "the program is not registering anything"). A note for
 * the person, said once while the board stays: "unlearned" when no learned board-and-piece
 * set reads it, so a start position shown once teaches it; "orientation" when a learned set
 * reads it but the coordinates cannot be, so Flip must say which side is at the bottom.
 */
class BoardUnread(val reason: String) : Event()

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
    /**
     * The captures that look free and are not ([com.sidecoach.analysis.Traps]): each the
     * capture, then the engine's best play after it, scored from the side to move's view.
     */
    val traps: List<Line> = emptyList(),
    /**
     * The recaptures the best line declines ([com.sidecoach.analysis.Defences]): each from the
     * root, the line up to the player's capture, the recapture, then the engine's best play
     * after it, scored from the side to move's view.
     */
    val defences: List<Line> = emptyList(),
    /**
     * What the best move threatens when it is quiet ([com.sidecoach.analysis.aimLines]): the side
     * to move's best line after it and a free move for the opponent, from that side's view.
     */
    val threatens: List<Line> = emptyList(),
    val tags: List<MotifTag> = emptyList(),
    val opening: String? = null,
    /**
     * The game is still in the opening book at this position ([opening] is the deepest line it
     * passed through, and stays after the game leaves the book).
     */
    val inBook: Boolean = false,
    /**
     * The position measured early (`annotator.EARLY_DEPTH`), the full depth to follow: what the
     * player asks is answered from it at once; the coach's own words, a verdict and the record
     * wait for the full depth.
     */
    val provisional: Boolean = false,
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
    /**
     * Per-move accuracy on a steepened Lichess curve (`analysis/Score.kt`), 0 to 100, so a game
     * can be summed up.
     */
    val accuracy: Double? = null,
) : Event()

// Coach and UI

/**
 * How much of what the analysis knows may reach the model, from DESIGN.md section 5.5.
 * Hiding is by construction: the level decides which facts are shown at all.
 *
 * Two hints, then the answer (Tai, 2026-09-23: "2 is probably the right number with the
 * third giving the answer with a good and correct explanation"). IDEA: the goal and the
 * pattern, and why it works. FOCUS: the target on the board and the piece that does it.
 * MOVE: the move, explained. LINE: the whole line.
 *
 * The desktop keeps the old seven-rung names as aliases (QUESTION = THEME = IDEA, REGION =
 * PIECE = FOCUS, CANDIDATE = MOVE) for its evaluation tools, which the phone does not have; a
 * Kotlin enum has no aliases, so only the five names are here.
 */
enum class DisclosureLevel(val value: Int) {
    SILENT(0),
    IDEA(1),
    FOCUS(2),
    MOVE(3),
    LINE(4),
    ;

    operator fun compareTo(other: Int): Int = value.compareTo(other)

    companion object {
        fun of(value: Int): DisclosureLevel = entries.first { it.value == value }
    }
}

enum class Trigger(val slug: String) {
    USER_ASK("user_ask"),
    POST_MISTAKE("post_mistake"),
    POST_MOVE("post_move"),
    DECISIVE_QUESTION("decisive_question"),
    STUCK("stuck"),
    REVIEW("review"),
}

class UserUtterance(val text: String, val source: String = "text") : Event()

/**
 * A mark on the board that goes with a coach message: what the words are about.
 *
 * `kind` is "square" (a highlight), "ring" (around a piece) or "arrow" (from, to); `squares`
 * holds one name, or two for an arrow. `color` is a role, not a hue -- good / warn / bad /
 * threat / focus / mention / region / line -- so the page keeps the palette. Marks are built
 * from the same disclosure-filtered facts the message was written from, so a mark never shows
 * more than the words do.
 */
data class Annotation(
    val kind: String,
    val squares: List<String>,
    val color: String = "focus",
    val opacity: Double = 1.0,
    val label: String? = null,
)

class CoachMessage(
    val text: String,
    val level: DisclosureLevel,
    val trigger: Trigger,
    val gameId: String? = null,
    val ply: Int? = null,
    /** The position the message is about, so a panel can show the marks on the right board. */
    val fen: String? = null,
    val annotations: List<Annotation> = emptyList(),
) : Event()

class ClockObservation(
    val whiteMs: Int? = null,
    val blackMs: Int? = null,
    val running: Pieces.Side? = null,
) : Event()

// The profile stage's events (`sidecoach/events.py`).

/**
 * One mistake of the player's, placed on the two axes: why (the cause) and what (the motif),
 * in which phase, and what it cost.
 */
class ProfileEvidence(
    val gameId: String,
    val ply: Int,
    val phase: String,
    val expectedLoss: Double,
    val motif: String? = null,
    val cause: String? = null,
    val timePressure: Boolean = false,
) : Event()

/**
 * A move of the player's that cost real ground, with what was better and why it happened,
 * for the review.
 */
data class TurningPoint(
    val ply: Int,
    val san: String,
    val label: String,
    val expectedLoss: Double,
    val segment: Int = 0,
    val bestSan: String? = null,
    var cause: String? = null,
    val motif: String? = null,
    val phase: String = "middlegame",
    val fenBefore: String? = null,
)

data class PhaseSummary(
    val phase: String,
    val moves: Int,
    val expectedLoss: Double,
    val mistakes: Int,
)

/**
 * One cell of the error taxonomy: a Beta-distributed error rate as counts. [errors] mistakes
 * out of [chances] occasions; the mean rate is (errors + 1) / (chances + 2).
 */
data class ProfileCell(
    val cause: String? = null,
    val motif: String? = null,
    val phase: String? = null,
    val errors: Int = 0,
    val chances: Int = 0,
    val expectedLoss: Double = 0.0,
) {
    val rate: Double get() = (errors + 1).toDouble() / (chances + 2)
}

/** A Lichess puzzle recommended: where to open it, how hard, what it trains. */
data class PuzzleRef(
    val id: String,
    val url: String,
    val rating: Int,
    val themes: List<String> = emptyList(),
)

/**
 * The game in numbers once it is over: the tally of the player's moves by label, their
 * accuracy, where the game turned, and how the phases went. Computed from the engine's
 * verdicts, so the coach can sum the game up without inventing it.
 */
class GameReview(
    val gameId: String,
    val userSide: Pieces.Side,
    val result: String,
    val reason: String? = null,
    val opening: String? = null,
    val moves: Int = 0,
    val accuracy: Double? = null,
    val tallies: Map<String, Int> = emptyMap(),
    val turningPoints: List<TurningPoint> = emptyList(),
    val phases: List<PhaseSummary> = emptyList(),
    /** Expected score lost by cause over the game, largest first. */
    val causes: Map<String, Double> = emptyMap(),
    /** The review in sentences, for the coach's summary and the panel's card. */
    var lines: List<String> = emptyList(),
) : Event()

/**
 * Practice for the player's weak spot after a game (DESIGN section 5.7): the motif the
 * profile says costs them most, and a few puzzles of it near their level.
 */
class Recommendation(
    val gameId: String,
    val motif: String,
    val motifWords: String,
    val reason: String,
    val level: Int,
    val puzzles: List<PuzzleRef> = emptyList(),
) : Event()

/**
 * A recommended (or any known) puzzle worked through on the screen: solved when every move
 * of the solution came, failed at the first wrong one.
 */
class PuzzleOutcome(
    val gameId: String,
    val puzzleId: String,
    val motif: String? = null,
    val solved: Boolean,
    /** The player's puzzle level after this outcome. */
    val level: Int,
) : Event()

/**
 * The player model as it stands: the cells with evidence, what they add up to in words, and
 * the last session's summary, for the coach's continuity.
 */
class ProfileSnapshot(
    val games: Int = 0,
    val cells: List<ProfileCell> = emptyList(),
    val insights: List<String> = emptyList(),
    val lastSession: String? = null,
) : Event()

/** A tagger name in words: hangingPiece -> hanging piece, mateIn2 -> mate in 2. */
fun insightWords(name: String?): String {
    if (name == null) return ""
    val out = StringBuilder()
    for (ch in name) {
        if (ch.isUpperCase()) {
            out.append(' ').append(ch.lowercaseChar())
        } else if (ch.isDigit() && out.isNotEmpty() && !out[out.length - 1].isDigit()) {
            out.append(' ').append(ch)
        } else {
            out.append(ch)
        }
    }
    return out.toString().replace("f 2 f 7", "f2/f7").replace("x ray", "x-ray")
}
