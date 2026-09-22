package com.sidecoach.analysis

import com.sidecoach.chess.Board
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.Severity
import kotlin.math.exp
import kotlin.math.round

/**
 * Expected score, the loss a move costs, and what makes a moment decisive:
 * `sidecoach/analysis/score.py`, ported.
 *
 * Everything here is in expected score, never centipawns: 0.5 is a dead draw, 1.0 is a win
 * for the side to move. A 50-centipawn slip at +8 costs nothing and the same slip at level
 * costs the game, so a threshold in centipawns says nothing about how much a move mattered.
 *
 * Pure functions over engine lines, so they are cheap to test and carry no engine.
 */

const val DRAW = 0.5

/**
 * Lichess's curve from centipawns to winning chances, fitted to games between people.
 * Stockfish's own win-draw-loss numbers answer a different question: how often Stockfish
 * beats Stockfish from here, which is 100% from a knight up and says nothing about the rest
 * of a club game. A coach needs the human curve, where a knight up is about 0.88 and the
 * moves that follow can still cost something.
 */
const val HUMAN_K = 0.00368208
const val CP_CAP = 1000

/**
 * Per-move accuracy: 100 for no loss, falling off exponentially with the drop in winning
 * chances. The shape is Lichess's; the steepness is not. Lichess uses 0.04354, on which an
 * inaccuracy scores about 80, and the plain mean of that ran fifteen points above
 * chess.com's accuracy on every one of six games. At 0.25 an inaccuracy scores 26, a mistake
 * 5, and the mean lands within 4.4 points of chess.com's number over twelve game-sides.
 */
const val ACCURACY_A = 103.1668
const val ACCURACY_K = 0.25
const val ACCURACY_C = 3.1669

/** Expected score for the side to move, on the curve fitted to human games. */
fun humanExpectedScore(scoreCp: Int?, mate: Int?): Double {
    if (mate != null) return if (mate > 0) 1.0 else 0.0
    if (scoreCp == null) return DRAW
    val capped = scoreCp.coerceIn(-CP_CAP, CP_CAP)
    return 1.0 / (1.0 + exp(-HUMAN_K * capped))
}

/** Expected score from the engine's own win-draw-loss numbers: (W + D/2) / 1000. */
fun engineExpectedScore(wdl: Triple<Int, Int, Int>?): Double {
    if (wdl == null) return DRAW
    val total = wdl.first + wdl.second + wdl.third
    if (total == 0) return DRAW
    return (wdl.first + wdl.second / 2.0) / total
}

/** Expected score under the named model: "human" (the default) or "engine". */
fun expectedScore(model: String, scoreCp: Int?, mate: Int?, wdl: Triple<Int, Int, Int>?): Double =
    if (model == "engine") engineExpectedScore(wdl) else humanExpectedScore(scoreCp, mate)

/**
 * How much loss earns a name, and what makes a position a decisive moment.
 *
 * The three severity lines were fitted to chess.com's review tallies over six of the user's
 * games (671 plies, 2026-09-15): they put our inaccuracy, mistake, miss and blunder counts
 * at 55, 33, 29 and 9 against their 55, 38, 24 and 10. A blunder throws away more than a
 * quarter of the game.
 */
data class Thresholds(
    val inaccuracy: Double = 0.04,
    val mistake: Double = 0.10,
    val blunder: Double = 0.28,
    val decisiveGap: Double = 0.10,
    /**
     * An only move: everything else throws away this much. Wider than a decisive moment,
     * since "there is one move here" is a stronger claim than "this move matters".
     */
    val onlyMoveGap: Double = 0.30,
    /** Below this loss a move is "excellent" rather than merely "good". */
    val excellent: Double = 0.02,
    /** A position this far gone, either way, is decided: a sacrifice in it is not brilliant. */
    val decided: Double = 0.90,
)

val DEFAULT_THRESHOLDS = Thresholds()

/** The value of a finished position to the side to move, or null if it is not finished. */
fun terminalExpectedScore(board: Board): Double? {
    val outcome = board.outcome(claimDraw = true) ?: return null
    val winner = outcome.winner ?: return DRAW
    return if (winner == board.turn) 1.0 else 0.0
}

/** The position's expected score to the side to move, whether or not it is finished. */
fun positionScore(board: Board, lines: List<Line>): Double {
    terminalExpectedScore(board)?.let { return it }
    if (lines.isEmpty()) return DRAW  // nothing was measured; claim nothing
    return lines[0].expectedScore
}

/**
 * What the played move cost its mover, in expected score.
 *
 * `afterScore` is the position after the move, scored for the player who is to move there,
 * which is the opponent; the mover's own prospects are one minus that. Comparing the two
 * needs both to have been measured at the same depth, which is why the annotator analyses
 * every position of a game the same way.
 *
 * The engine's own first choice costs nothing by definition. Without that rule a search that
 * returns a slightly different number for the same line at the same depth would have the
 * coach calling the best move on the board an inaccuracy.
 */
fun evalLoss(before: List<Line>, playedUci: String, afterScore: Double): Double {
    if (before.isEmpty()) return 0.0
    val best = before[0]
    if (best.moves.isNotEmpty() && playedUci == best.moves[0]) return 0.0
    return maxOf(0.0, best.expectedScore - (1.0 - afterScore))
}

fun severityOf(loss: Double, thresholds: Thresholds = DEFAULT_THRESHOLDS): Severity = when {
    loss >= thresholds.blunder -> Severity.BLUNDER
    loss >= thresholds.mistake -> Severity.MISTAKE
    loss >= thresholds.inaccuracy -> Severity.INACCURACY
    else -> Severity.NONE
}

/**
 * What kind of move it was, in the vocabulary the user reads on chess.com.
 *
 * Two axes. The loss says how good the move was: best, excellent, good, then the three names
 * for losing ground. The moment says whether it mattered, which is what great, brilliant and
 * miss add. `opponentErred` says the move before this one was a mistake or worse, so this
 * move either took the chance (great, when it is the best move) or let it go (miss, whatever
 * the size of the loss); `onlyMove` says everything else lost, so finding it was great; a
 * sound `sacrifice` is brilliant unless the position was already decided, where giving
 * material back is just technique. Neither great nor brilliant applies to an `obvious` move,
 * nor in a lost position, where the best move changes nothing.
 */
fun labelMove(
    loss: Double,
    thresholds: Thresholds,
    isBest: Boolean,
    inBook: Boolean,
    onlyMove: Boolean,
    sacrifice: Boolean,
    beforeScore: Double,
    opponentErred: Boolean,
    obvious: Boolean = false,
): Label {
    val flagged = loss >= thresholds.inaccuracy
    if (inBook && !flagged) {
        // The opening tables hold dubious gambits too; a move that loses ground is judged by
        // what it lost, whatever a book calls it. 1. e4 f5 is in the tables.
        return Label.BOOK
    }
    val won = beforeScore >= thresholds.decided
    val lost = beforeScore <= 1.0 - thresholds.decided
    if (opponentErred && flagged) return Label.MISS
    if (!flagged) {
        if (!obvious && !lost) {
            if (sacrifice && !won) return Label.BRILLIANT
            if (onlyMove || (opponentErred && isBest)) return Label.GREAT
        }
        if (isBest) return Label.BEST
        return if (loss < thresholds.excellent) Label.EXCELLENT else Label.GOOD
    }
    return when (severityOf(loss, thresholds)) {
        Severity.BLUNDER -> Label.BLUNDER
        Severity.MISTAKE -> Label.MISTAKE
        else -> Label.INACCURACY
    }
}

/** 0 to 100 for one move, from its loss in expected score. */
fun moveAccuracy(loss: Double): Double {
    val raw = ACCURACY_A * exp(-ACCURACY_K * loss * 100.0) - ACCURACY_C
    return round(raw.coerceIn(0.0, 100.0) * 100.0) / 100.0
}

/**
 * One number for a side's game: the mean of the per-move accuracies. Lichess averages a
 * volatility-weighted mean with a harmonic mean; the harmonic mean collapses on a single
 * move that flipped the game, which club games have.
 */
fun gameAccuracy(accuracies: List<Double>): Double? {
    if (accuracies.isEmpty()) return null
    return accuracies.sum() / accuracies.size
}

/** How much better the best line is than the second best, or null without a second. */
fun bestGap(lines: List<Line>): Double? {
    if (lines.size < 2) return null
    return maxOf(0.0, lines[0].expectedScore - lines[1].expectedScore)
}

/** One move holds the position: either it is the only legal one, or the rest throw it away. */
fun isOnlyMove(board: Board, lines: List<Line>, thresholds: Thresholds): Boolean {
    val legal = board.legalMoves().size
    if (legal == 0) return false
    if (legal == 1) return true
    val gap = bestGap(lines)
    return gap != null && gap >= thresholds.onlyMoveGap
}

/**
 * A moment where the choice matters: the lines diverge, or there is only one move. This is
 * what the coach may say out loud before a move is played, because it points at the position
 * without naming the move.
 */
fun isDecisive(board: Board, lines: List<Line>, thresholds: Thresholds): Boolean {
    if (isOnlyMove(board, lines, thresholds)) return true
    val gap = bestGap(lines)
    return gap != null && gap >= thresholds.decisiveGap
}
