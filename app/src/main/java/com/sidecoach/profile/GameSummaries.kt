package com.sidecoach.profile

import com.sidecoach.analysis.gameAccuracy
import com.sidecoach.analysis.gamePhases
import com.sidecoach.analysis.positionScore
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.events.GameSummary
import com.sidecoach.events.KeyMoment
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.PerformanceRating
import com.sidecoach.events.PhaseGrade
import com.sidecoach.events.ReviewJson
import com.sidecoach.events.ReviewedGame
import com.sidecoach.events.Side
import com.sidecoach.events.SideSummary
import com.sidecoach.voice.Py
import java.math.BigDecimal
import java.math.RoundingMode

/**
 * A reviewed game summed up (`profile/game_summary.py`, the summary screen): each side's
 * accuracy and count of every kind of move, the rating it played at ([Performance]), how each
 * side played each phase, the moves the review stops at, and the graph of the game -- all read
 * off the review's verdicts.
 *
 * The phase grade is chess.com's report card in our own vocabulary: the phase's accuracy drawn
 * as the kind of move it amounts to, with a great or brilliant move in a well-played phase
 * lifting it to that.
 */
object GameSummaries {
    val PHASES = listOf("opening", "middlegame", "endgame")

    // A phase needs this many of a side's moves to be graded.
    const val MIN_PHASE_MOVES = 3

    // The moves the review stops at, besides the last book move.
    val KEY_LABELS = setOf(Label.BRILLIANT, Label.GREAT, Label.INACCURACY, Label.MISTAKE, Label.MISS, Label.BLUNDER)

    // The moves that passed up something better: where the game's lessons are.
    val MISSED = setOf(Label.INACCURACY, Label.MISTAKE, Label.MISS, Label.BLUNDER)

    // A phase's accuracy as the kind of move it amounts to, from the top.
    private val GRADES = listOf(
        90.0 to Label.BEST,
        80.0 to Label.EXCELLENT,
        70.0 to Label.GOOD,
        55.0 to Label.INACCURACY,
        40.0 to Label.MISTAKE,
    )

    val SHAPES = listOf("smooth", "balanced", "sudden", "giveaway", "wild", "intense", "sharp")

    // White's expected score from which a side is winning, and from which it was clearly
    // winning (a giveaway throws that away); a game is close between the first two.
    private const val WINNING = 0.7
    private const val CLEARLY = 0.85
    private const val CLOSE_LOW = 0.35
    private const val CLOSE_HIGH = 0.65

    /**
     * The game's shape, chess.com's "game shapes" read off White's expected score after each
     * ply: wild, intense, giveaway, sudden, smooth, sharp (big swings that never changed who led,
     * the leader winning), balanced. Too short a game (under 10 plies), or one no shape fits, has
     * none. `shape_of`.
     */
    fun shapeOf(graph: List<Double>, result: String?): String? {
        if (graph.size < 11) return null
        val leaders = ArrayList<Int>() // +1 White winning, -1 Black, in order of the changes
        for (score in graph) {
            val now = if (score >= WINNING) 1 else if (score <= 1 - WINNING) -1 else 0
            if (now != 0 && (leaders.isEmpty() || leaders.last() != now)) leaders.add(now)
        }
        val changes = if (leaders.isEmpty()) 0 else leaders.size - 1
        val winner = if (result == "1-0") 1 else if (result == "0-1") -1 else 0
        if (changes >= 3) return "wild"
        if (changes == 2) return "intense"
        val clearly = graph.map { if (it >= CLEARLY) 1 else if (it <= 1 - CLEARLY) -1 else 0 }
        if (clearly.any { it != 0 && it != winner }) return "giveaway"
        val close = graph.map { CLOSE_LOW <= it && it <= CLOSE_HIGH }
        val threeQuarters = graph.size * 3 / 4
        if (close.take(threeQuarters).all { it } && winner != 0 && !close.last()) return "sudden"
        if (leaders.isNotEmpty() && winner == leaders.last() && changes == 0) {
            val swings = Py.sum(graph.zipWithNext { a, b -> Math.abs(b - a) })
            return if (swings > 2.0) "sharp" else "smooth"
        }
        if (close.count { it } >= 0.7 * graph.size) return "balanced"
        // None of the shapes: the lead changed once, or a side led and did not win. It was
        // "sharp", and the summary said the winner "kept the upper hand" of a game they came
        // back in (the review of 2026-10-01).
        return null
    }

    /**
     * A phase's grade: its accuracy as a kind of move; played that well with a brilliant or
     * great move in it, that move's name. `phase_grade`.
     */
    fun phaseGrade(accuracy: Double, labels: List<Label>): Label {
        if (accuracy >= GRADES[0].first) {
            if (Label.BRILLIANT in labels) return Label.BRILLIANT
            if (Label.GREAT in labels) return Label.GREAT
        }
        for ((floor, grade) in GRADES) if (accuracy >= floor) return grade
        return Label.BLUNDER
    }

    private class Moved(val phase: String, val accuracy: Double, val label: Label?)

    /** The summary of a reviewed game. `summarize`. */
    fun summarize(game: ReviewedGame): GameSummary {
        val perSide = mapOf(Side.WHITE to ArrayList<Moved>(), Side.BLACK to ArrayList())
        // What each side's moves cost from where they stood, for the rating they played at.
        val played = mapOf(Side.WHITE to ArrayList<Performance.Played>(), Side.BLACK to ArrayList())
        val graph = arrayListOf(whiteScore(Board(game.start.fen), game.start.lines))
        var lastBook: Int? = null
        val key = ArrayList<KeyMoment>()
        var opening = game.start.opening
        // The game divided as Lichess divides it: each move in the phase of the position it was
        // played from.
        val phases = gamePhases(game.moves.map { Board(it.fenBefore) })
        for ((index, move) in game.moves.withIndex()) {
            val before = Board(move.fenBefore)
            val phase = phases[index]
            val label = move.assessment.label
            perSide.getValue(move.side).add(Moved(phase, move.assessment.accuracy ?: 0.0, label))
            val standing = if (move.side == Side.WHITE) graph.last() else 1.0 - graph.last()
            played.getValue(move.side).add(Performance.Played(move.ply - 1, move.assessment.expectedLoss, standing))
            if (label == Label.BOOK) {
                lastBook = move.ply
            } else if (label != null && label in KEY_LABELS) {
                key.add(KeyMoment(move.ply, move.side, ReviewJson.labelName(label)))
            }
            val after = before.copy(withStack = false)
            after.push(Move.fromUci(move.assessment.uci))
            graph.add(whiteScore(after, move.after.lines))
            opening = move.after.opening?.takeIf { it.isNotEmpty() } ?: opening
        }
        if (lastBook != null) key.add(KeyMoment(lastBook, game.moves[lastBook - 1].side, ReviewJson.labelName(Label.BOOK)))
        key.sortBy { it.ply }
        val headers = game.headers
        return GameSummary(
            white = side(Side.WHITE, perSide.getValue(Side.WHITE), played.getValue(Side.WHITE), headers),
            black = side(Side.BLACK, perSide.getValue(Side.BLACK), played.getValue(Side.BLACK), headers),
            opening = opening,
            lastBookPly = lastBook,
            keyMoments = key,
            graph = graph.map { round(it, 4) },
            shape = shapeOf(graph, headers["Result"]),
        )
    }

    /**
     * The tactics the sides' mistakes passed up, most often first: for each inaccuracy, mistake,
     * miss or blunder, the motif of the position's best line (what the move could have done
     * instead) -- what the game says to practise. `missed_motifs`.
     */
    fun missedMotifs(game: ReviewedGame, sides: Set<Side>): List<Pair<String, Int>> {
        val counts = LinkedHashMap<String, Int>()
        for ((i, move) in game.moves.withIndex()) {
            if (move.side !in sides || move.assessment.label !in MISSED) continue
            val before = if (i == 0) game.start else game.moves[i - 1].after
            val motif = Causes.tactical(before.tags) ?: continue
            counts[motif] = (counts[motif] ?: 0) + 1
        }
        return counts.entries.map { it.key to it.value }.sortedWith(compareBy({ -it.second }, { it.first }))
    }

    private fun side(
        side: Side,
        moves: List<Moved>,
        played: List<Performance.Played>,
        headers: Map<String, String>,
    ): SideSummary {
        val colour = if (side == Side.WHITE) "White" else "Black"
        val counts = LinkedHashMap<String, Int>()
        for (m in moves) {
            val label = m.label ?: continue
            val name = ReviewJson.labelName(label)
            counts[name] = (counts[name] ?: 0) + 1
        }
        val phases = ArrayList<PhaseGrade>()
        for (phase in PHASES) {
            val inside = moves.filter { it.phase == phase }
            if (inside.size < MIN_PHASE_MOVES) continue
            // `fmean`, an exactly rounded sum over the count, as the desktop's: a plain sum over
            // the size put 1.8% of phases 0.1 apart, and a grade can flip on it (the review of
            // 2026-10-01).
            val accuracy = round(Stats.fmean(inside.map { it.accuracy }), 1)
            val labels = inside.mapNotNull { it.label }
            phases.add(PhaseGrade(phase, inside.size, accuracy, phaseGrade(accuracy, labels)))
        }
        val accuracy = gameAccuracy(moves.map { it.accuracy })
        var rating: PerformanceRating? = null
        if (played.size >= Performance.MIN_MOVES) {
            val estimate = Performance.estimate(played)
            val site = headers["Site"]
            rating = PerformanceRating(
                rating = Performance.forSite(estimate.rating.toDouble(), site),
                low = Performance.forSite(estimate.low.toDouble(), site),
                high = Performance.forSite(estimate.high.toDouble(), site),
            )
        }
        return SideSummary(
            side = side,
            name = headers[colour]?.takeIf { it.isNotEmpty() },
            rating = int(headers["${colour}Elo"]),
            accuracy = accuracy?.let { round(it, 1) },
            counts = counts,
            phases = phases,
            performance = rating,
        )
    }

    /** White's expected score in a position, from the lines measured for the side to move. */
    fun whiteScore(board: Board, lines: List<Line>): Double {
        val score = positionScore(board, lines)
        return if (board.turn == WHITE) score else 1.0 - score
    }

    /** Python's `round(value, places)`: the binary value rounded, halves to even. */
    fun round(value: Double, places: Int): Double = BigDecimal(value).setScale(places, RoundingMode.HALF_EVEN).toDouble()

    private fun int(value: String?): Int? =
        if (value == null || value == "" || value == "?") null else value.trim().toIntOrNull()
}
