package com.sidecoach.profile

import com.sidecoach.analysis.gameAccuracy
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.events.GameReview
import com.sidecoach.events.PhaseSummary
import com.sidecoach.events.TurningPoint
import com.sidecoach.vision.Pieces
import com.sidecoach.voice.Py

/**
 * The post-game review: the game in numbers, from the engine's verdicts on the player's
 * moves, and in sentences the coach can sum it up from. `sidecoach/profile/review.py`.
 *
 * Nothing here is opinion. The tally, the accuracy, where the game turned and what was
 * better, how each phase went and what the mistakes have in common are all read off the
 * assessments and the causes, so the coach's summary has something true to say.
 */
object Review {

    val ERRORS = setOf("mistake", "blunder")
    val GOOD = setOf("brilliant", "great", "best", "excellent")
    const val TURNING_POINTS = 3

    /** Mistakes of one cause in a game that make a pattern worth naming. */
    const val PATTERN_MIN = 2

    /** A judged move of the player's, as the profiler keeps it. */
    data class UserMove(
        val ply: Int,
        val san: String,
        val uci: String,
        val fenBefore: String,
        val label: String?,
        val severity: String,
        val expectedLoss: Double,
        val bestUci: String?,
        val accuracy: Double?,
        val phase: String,
        val cause: String? = null,
        val motif: String? = null,
        val segment: Int = 0,
    )

    fun numbered(fenBefore: String, san: String): String {
        val board = Board(fenBefore)
        val n = board.fullmoveNumber
        return if (board.turn == WHITE) "$n. $san" else "$n... $san"
    }

    fun bestSan(fenBefore: String, bestUci: String?): String? {
        if (bestUci == null) return null
        val board = Board(fenBefore)
        val move = Move.fromUciOrNull(bestUci) ?: return null
        return if (board.isLegal(move)) board.san(move) else null
    }

    private fun outcome(result: String, userSide: Pieces.Side): String {
        if (result == "1/2-1/2") return "drew"
        if (result != "1-0" && result != "0-1") return "played"
        val won = (result == "1-0") == userSide.color
        return if (won) "won" else "lost"
    }

    /** The review of one game from the player's judged moves. */
    fun buildReview(
        gameId: String,
        userSide: Pieces.Side,
        result: String,
        reason: String?,
        opening: String?,
        moves: List<UserMove>,
        fullmoves: Int,
    ): GameReview {
        val judged = moves
        val tallies = LinkedHashMap<String, Int>()
        for (m in judged) {
            val key = m.label ?: m.severity
            tallies[key] = (tallies[key] ?: 0) + 1
        }
        val accuracy = gameAccuracy(judged.mapNotNull { it.accuracy })
        val turning = judged.filter { it.severity in ERRORS }
            .sortedBy { -it.expectedLoss }
            .take(TURNING_POINTS)
        val turningPoints = turning.map { m ->
            TurningPoint(
                ply = m.ply,
                segment = m.segment,
                san = numbered(m.fenBefore, m.san),
                label = m.label ?: m.severity,
                expectedLoss = Insights.round(m.expectedLoss, 3),
                bestSan = bestSan(m.fenBefore, m.bestUci),
                cause = m.cause,
                motif = m.motif,
                phase = m.phase,
                fenBefore = m.fenBefore,
            )
        }
        val byPhase = LinkedHashMap<String, MutableList<UserMove>>()
        for (m in judged) byPhase.getOrPut(m.phase) { ArrayList() }.add(m)
        val phases = listOf("opening", "middlegame", "endgame").mapNotNull { phase ->
            val items = byPhase[phase]
            if (items.isNullOrEmpty()) {
                null
            } else {
                PhaseSummary(
                    phase = phase,
                    moves = items.size,
                    expectedLoss = Insights.round(Py.sum(items.map { it.expectedLoss }), 3),
                    mistakes = items.count { it.severity in ERRORS },
                )
            }
        }
        val causes = LinkedHashMap<String, Double>()
        val counts = LinkedHashMap<String, Int>()
        for (m in judged) {
            val cause = m.cause
            if (m.severity in ERRORS && !cause.isNullOrEmpty()) {
                causes[cause] = (causes[cause] ?: 0.0) + m.expectedLoss
                counts[cause] = (counts[cause] ?: 0) + 1
            }
        }
        val ordered = LinkedHashMap<String, Double>()
        for ((k, v) in causes.entries.map { it.key to Insights.round(it.value, 3) }.sortedBy { -it.second }) {
            ordered[k] = v
        }
        val review = GameReview(
            gameId = gameId,
            userSide = userSide,
            result = result,
            reason = reason,
            opening = opening,
            moves = judged.size,
            accuracy = accuracy?.let { Insights.round(it, 1) },
            tallies = tallies,
            turningPoints = turningPoints,
            phases = phases,
            causes = ordered,
        )
        review.lines = linesFor(review, fullmoves)
        val pattern = patternLine(counts, causes)
        if (pattern != null) {
            // Second, after the result: the offline review says the first four lines.
            review.lines = review.lines.toMutableList().apply { add(1, pattern) }
        }
        return review
    }

    /**
     * The one thing to work on, when one cause is behind two or more of the game's mistakes, and
     * a habit that helps: said as a skill to practise, not a count of the times it went wrong
     * (Tai, 2026-10-02). The cause is the most frequent, the larger loss breaking a tie, the
     * first seen among equals (Python's `max`).
     */
    fun patternLine(counts: Map<String, Int>, losses: Map<String, Double>): String? {
        if (counts.isEmpty()) return null
        var cause: String? = null
        var n = 0
        var loss = 0.0
        for ((k, v) in counts) {
            val l = losses[k] ?: 0.0
            if (cause == null || v > n || (v == n && l > loss)) {
                cause = k
                n = v
                loss = l
            }
        }
        val found = cause ?: return null
        if (n < PATTERN_MIN || !Causes.HABIT.containsKey(found)) return null
        val (words, habit) = Causes.HABIT.getValue(found)
        return "One thing to work on: $words. A habit that helps: $habit."
    }

    private val PLURAL = mapOf("inaccuracy" to "inaccuracies", "mistake" to "mistakes", "miss" to "misses", "blunder" to "blunders")

    private val ORDER = listOf(
        "brilliant", "great", "best", "excellent", "good", "book", "inaccuracy", "mistake", "miss", "blunder",
    )

    /**
     * How a game ended, as the review's first sentence says it: the reason where it reads as
     * English after "by", these where it does not ("lost by user resigned or lost on time",
     * "won by game over on board": the parity audit, 2026-09-26).
     */
    val HOW_IT_ENDED = mapOf(
        "draw agreed" to " by agreement",
        "lost on time" to " on time",
        "fifty-move rule" to " by the fifty-move rule",
        "user resigned or lost on time" to " by resignation or on time",
        "opponent resigned or lost on time" to " by resignation or on time",
        "game over on board" to "",
    )

    /** The review in sentences. */
    fun linesFor(review: GameReview, fullmoves: Int): List<String> {
        val out = ArrayList<String>()
        val reason = review.reason
        val how = if (!reason.isNullOrEmpty()) HOW_IT_ENDED[reason] ?: " by $reason" else ""
        val colour = review.userSide.name.lowercase()
        val outcome = outcome(review.result, review.userSide)
        if (review.result != "1-0" && review.result != "0-1" && review.result != "1/2-1/2") {
            out.add("You played $colour for $fullmoves moves; I did not see how it ended.")
        } else {
            out.add("You played $colour and $outcome$how in $fullmoves moves.")
        }
        if (!review.opening.isNullOrEmpty()) out.add("The opening was the ${review.opening}.")
        if (review.moves != 0) {
            // The kinds that are nouns take a plural; best, excellent, good and book do not.
            val parts = ORDER.filter { (review.tallies[it] ?: 0) != 0 }.map {
                val n = review.tallies.getValue(it)
                "$n ${if (n > 1) PLURAL[it] ?: it else it}"
            }
            val acc = review.accuracy?.let { "Accuracy ${Insights.fixed(it, 1)}. " } ?: ""
            out.add("${acc}Your ${review.moves} moves: " + parts.joinToString(", ") + ".")
        }
        for (tp in review.turningPoints) {
            val better = if (!tp.bestSan.isNullOrEmpty()) "; ${tp.bestSan} was better" else ""
            var why = ""
            if (!tp.cause.isNullOrEmpty()) {
                why = " (${tp.cause}"
                if (!tp.motif.isNullOrEmpty()) why += ": ${Insights.motifWords(tp.motif)}"
                why += ")"
            }
            out.add("Ground lost at ${tp.san}, a ${tp.label} costing ${Insights.fixed(tp.expectedLoss, 2)}$better$why.")
        }
        if (review.turningPoints.isEmpty() && review.moves != 0) out.add("No move cost you real ground.")
        if (review.phases.size > 1) {
            val bits = review.phases.map { "the ${it.phase} ${Insights.fixed(it.expectedLoss, 2)} over ${it.moves} moves" }
            out.add("Expected score lost by phase: " + bits.joinToString(", ") + ".")
        }
        if (review.causes.isNotEmpty()) {
            val (cause, loss) = review.causes.entries.first().toPair()
            out.add("Most ground went to $cause (${Insights.fixed(loss, 2)}).")
        }
        return out
    }
}
