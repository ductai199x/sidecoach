package com.sidecoach.profile

import com.sidecoach.analysis.Exchange
import com.sidecoach.analysis.Motifs
import com.sidecoach.chess.BISHOP
import com.sidecoach.chess.Board
import com.sidecoach.chess.KNIGHT
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.events.MotifTag

/**
 * Why a mistake happened: the cause axis of the error taxonomy (DESIGN section 5.6).
 * `sidecoach/profile/causes.py`.
 *
 * The cause is read from engine evidence, in the order a coach would look: the clock first
 * (under ten seconds, it was time pressure whatever else it was), then what the move allowed
 * (a tactic for the opponent, a mate, a piece for free: an opponent's threat missed), then
 * what the move was (a tactic or a capture that did not work: a line miscalculated), then
 * what was there instead (a tactic the player had and did not play: a candidate never
 * considered), then the phase (opening theory, endgame technique), and positional misjudgment
 * for the rest. The motif axis comes from the tagger: the tactic allowed, missed or attempted.
 *
 * Self-reports refine this -- "I saw it but thought it failed" turns a candidate never
 * considered into a line miscalculated -- and every entry is the user's to correct.
 */
object Causes {

    const val MISSED_THREAT = "missed opponent threat"
    const val MISCALCULATED = "miscalculated line"
    const val NOT_CONSIDERED = "never considered the candidate"
    const val POSITIONAL = "positional misjudgment"
    const val OPENING_GAP = "opening knowledge gap"
    const val ENDGAME = "endgame technique"
    const val TIME_PRESSURE = "time pressure"

    val CAUSES: List<String> = listOf(
        MISSED_THREAT, MISCALCULATED, NOT_CONSIDERED, POSITIONAL, OPENING_GAP, ENDGAME, TIME_PRESSURE,
    )

    /** A mistake with less than this on the clock defaults to time pressure. */
    const val TIME_PRESSURE_MS = 10_000

    /**
     * The tagger's names that are tactics (what a move creates, what a line does, the mating
     * pattern), as against what a move merely is (a capture, a check, a quiet move).
     */
    val TACTICAL: Set<String> = Motifs.TAG_ORDER.drop(Motifs.TAG_ORDER.indexOf("fork")).toSet()

    private val VALUE = mapOf(PAWN to 1, KNIGHT to 3, BISHOP to 3, ROOK to 5, QUEEN to 9)

    /** The first tactical motif among the tags, or null. */
    fun tactical(tags: List<MotifTag>): String? = tags.firstOrNull { it.name in TACTICAL }?.name

    /**
     * Whether the reply takes something for less than it gives: a bigger piece, or one nobody
     * can take back -- by the rules, so a guard pinned to its own king, or a king that would
     * step into check, takes nothing back ([Exchange.canRetake]).
     */
    fun winsMaterial(board: Board, reply: Move): Boolean {
        if (!board.isCapture(reply) || !board.isLegal(reply)) return false
        val victim = board.pieceAt(reply.to)
        val attacker = board.pieceAt(reply.from) ?: return false
        val taken = if (victim != null) VALUE[victim.pieceType] ?: 1 else 1  // en passant
        if (taken > (VALUE[attacker.pieceType] ?: 0)) return true
        return !Exchange.canRetake(board, reply)
    }

    /**
     * The cause and the motif of a mistake.
     *
     * [before] is the position the player faced and [move] what they played; [playedTags] are
     * the tags of that move, [availableTags] those of the position they faced (what they
     * could have played), [allowedTags] those of the position after the move (what the
     * opponent now has); [reply] is the opponent's best answer and [replyMate] its mate
     * distance; [leftBook] says the move was the first out of theory; [clockMs] the player's
     * clock when the move was made.
     */
    fun classify(
        before: Board,
        move: Move,
        playedTags: List<MotifTag> = emptyList(),
        availableTags: List<MotifTag> = emptyList(),
        allowedTags: List<MotifTag> = emptyList(),
        reply: String? = null,
        replyMate: Int? = null,
        phase: String = "middlegame",
        leftBook: Boolean = false,
        clockMs: Int? = null,
    ): Pair<String, String?> {
        val allowed = tactical(allowedTags)
        val available = tactical(availableTags)
        val attempted = tactical(playedTags)
        if (clockMs != null && clockMs < TIME_PRESSURE_MS) {
            return TIME_PRESSURE to (allowed ?: available ?: attempted)
        }
        val after = before.copy(withStack = false)
        after.push(move)
        if (replyMate != null && replyMate > 0) return MISSED_THREAT to (allowed ?: "mate")
        if (allowed != null) return MISSED_THREAT to allowed
        if (reply != null) {
            val replyMove = Move.fromUciOrNull(reply)
            if (replyMove != null && winsMaterial(after, replyMove)) return MISSED_THREAT to "hangingPiece"
        }
        if (attempted != null) return MISCALCULATED to attempted
        if (before.isCapture(move) || before.givesCheck(move)) return MISCALCULATED to available
        if (available != null) return NOT_CONSIDERED to available
        if (phase == "endgame") return ENDGAME to null
        if (phase == "opening" || leftBook) return OPENING_GAP to null
        return POSITIONAL to null
    }

    /** What a player says about a mistake, when it says plainly what happened. */
    private val WORDS: List<Pair<Regex, String>> = listOf(
        Regex(
            "\\b(?:time|clock|rush(?:ed|ing)?|hurr(?:y|ied)|in a hurry|too fast|quick(?:ly)?|" +
                "premove[sd]?|no time)\\b",
            RegexOption.IGNORE_CASE,
        ) to TIME_PRESSURE,
        Regex(
            "\\b(?:(?:i )?(?:thought|figured|calculated|expected|assumed|believed|saw)\\b.*" +
                "\\b(?:but|work|would|could|was|wasn'?t|didn'?t)|misjudg|miscalculat|" +
                "(?:saw|knew) (?:it|that|the (?:threat|idea|move|fork|pin|mate)))",
            RegexOption.IGNORE_CASE,
        ) to MISCALCULATED,
        Regex(
            "\\b(?:didn'?t (?:see|notice|spot|consider|look|think)|missed|overlooked|" +
                "never (?:saw|considered|thought)|no idea|blind|forgot|" +
                "wasn'?t (?:looking|paying attention)|completely missed)\\b",
            RegexOption.IGNORE_CASE,
        ) to MISSED_THREAT,
    )

    /**
     * The cause a player's own words name, or null when they do not say. "I missed it" is a
     * threat missed; "I thought it worked" a line miscalculated; "I was rushing" time pressure.
     */
    fun fromWords(text: String): String? = WORDS.firstOrNull { it.first.containsMatchIn(text) }?.second
}
