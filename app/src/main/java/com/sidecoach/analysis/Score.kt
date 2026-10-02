package com.sidecoach.analysis

import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.Severity
import com.sidecoach.voice.Py
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

/**
 * Whether the game is over on the board as the sites play it: mate, stalemate, no mating
 * material, or a draw by rule that has happened -- the third occurrence of a position, or fifty
 * moves without a capture or a pawn move. Chess.com and Lichess end the game there themselves.
 * A draw the side to move could merely claim with its next move is not one: that position is
 * played on, and a queen up is still a queen up. The repetition needs the board's moves; a
 * board without them knows no history and repeats nothing. `score.game_over`.
 */
fun gameOver(board: Board): Boolean =
    board.outcome(claimDraw = false) != null || board.isRepetition(3) || board.isFiftyMoves()

/** The value of a finished position to the side to move, or null if it is not finished. */
fun terminalExpectedScore(board: Board): Double? {
    if (!gameOver(board)) return null
    val outcome = board.outcome(claimDraw = false)
    val winner = outcome?.winner ?: return DRAW  // a draw by rule ([gameOver])
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

/**
 * A move this far behind the best is about as good as it: chess.com's Best or Excellent band, as
 * [labelMove] grades it. The coach's "about equally good" moves are these (Tai, 2026-10-02: the
 * bands, not an arbitrary margin). `score.about_as_good`.
 */
fun aboutAsGood(loss: Double, thresholds: Thresholds = DEFAULT_THRESHOLDS): Boolean = loss < thresholds.excellent

/**
 * A move this far behind the best is graded good or better -- Best, Excellent or Good -- and no
 * inaccuracy ([labelMove]'s flag). `score.graded_good`.
 */
fun gradedGood(loss: Double, thresholds: Thresholds = DEFAULT_THRESHOLDS): Boolean = loss < thresholds.inaccuracy

/**
 * A move this far behind the best is clearly worse: a mistake or a blunder, as [severityOf] grades
 * it. `score.clearly_worse`.
 */
fun clearlyWorse(loss: Double, thresholds: Thresholds = DEFAULT_THRESHOLDS): Boolean = loss >= thresholds.mistake

fun severityOf(loss: Double, thresholds: Thresholds = DEFAULT_THRESHOLDS): Severity = when {
    loss >= thresholds.blunder -> Severity.BLUNDER
    clearlyWorse(loss, thresholds) -> Severity.MISTAKE
    loss >= thresholds.inaccuracy -> Severity.INACCURACY
    else -> Severity.NONE
}

/*
 * How a side stands, by its expected score: the bands the coach speaks in (`standingWords`:
 * losing, worse, about equal, better, winning). "Better" starts at 0.62 and "winning" at 0.8, as
 * the idea reader counts a level and a won position (`IdeaReader.LEVEL`, `IdeaReader.WON`).
 */
const val WORSE_FROM = 0.2
const val EQUAL_FROM = 0.38
const val BETTER_FROM = 0.62
const val WINNING_FROM = 0.8

/** The band an expected score falls in: 0 losing, 1 worse, 2 about equal, 3 better, 4 winning. `standing`. */
fun standing(score: Double): Int = when {
    score >= WINNING_FROM -> 4
    score >= BETTER_FROM -> 3
    score > EQUAL_FROM -> 2
    score > WORSE_FROM -> 1
    else -> 0
}

/**
 * What kind of move it was, in the words chess.com's review uses, as its help pages define them
 * (2026-10-01) and as far as six of the user's reviewed games bear them out. `score.label_move`.
 *
 * - Best, "the engine's top choice" ([isBest]); excellent, "almost as good as the best move" (a
 *   loss under `excellent`); good, "a decent move but not the best".
 * - Inaccuracy, mistake, blunder: by the loss. A move that walks into a forced mate
 *   ([mateAllowed]) "loses the game" and is a blunder, unless the game was lost already (the
 *   curve puts a mate at 0.0 and a thousand centipawns at 0.025).
 * - Book, "a conventional opening move": the tables' move, unless it threw away a tenth of the
 *   game -- the tables hold dubious gambits too (1. e4 f5), and chess.com's book does not,
 *   calling 1...f5 a mistake.
 * - Miss, "a move that missed a tactical opportunity or a chance to punish the opponent": the
 *   position offered a winning chance ([chance], read by [verdict] from the engine's lines) and
 *   the move played gave ground (a loss of an inaccuracy or more), whatever the size of the
 *   loss (Tai, 2026-10-02). And a forced mate let go ([mateLetGo], read by [verdict]): the
 *   player had a mate in N and the move played leaves none, or only a longer one -- a miss
 *   whatever it cost, even while it still wins; a move that keeps a mate as short is labelled by
 *   its loss (Tai, 2026-10-02 12:16).
 * - Great, "a move that altered the course of the game": the best move where every other lost
 *   ground ([decisive], the only good move), or the best reply to the opponent's blunder.
 *   Merely the best move after any error was great 30 times in the six games where chess.com
 *   said so 20 times.
 * - Brilliant, "the best move -- and tricky to find too": a sound sacrifice, as good as the
 *   best, in a game still to be won. Whether it was won already is read from the second-best
 *   line ([secondScore]): the best is the line the sacrifice makes, and a sacrifice that is the
 *   only win was never brilliant.
 *
 * Neither great nor brilliant applies to an [obvious] move (a recapture, a free capture nobody
 * can take back, the only legal move), nor in a lost position, where the best move changes
 * nothing.
 */
fun labelMove(
    loss: Double,
    thresholds: Thresholds,
    isBest: Boolean,
    inBook: Boolean,
    decisive: Boolean,
    sacrifice: Boolean,
    beforeScore: Double,
    opponentBlundered: Boolean,
    secondScore: Double? = null,
    chance: Boolean = false,
    mateLetGo: Boolean = false,
    mateAllowed: Boolean = false,
    obvious: Boolean = false,
): Label {
    val lost = beforeScore <= 1.0 - thresholds.decided
    // Walking into a mate from a lost position is not the move that lost: chess.com leaves
    // those unmarked (eleven of them in the six games, the best move among them).
    val allowed = mateAllowed && !lost && !isBest
    // A mate let go is a miss of its own ([mateLetGo]), never flagged by what it cost: the
    // engine's own first move, or one that keeps a mate as short, is never an inaccuracy.
    val flagged = !gradedGood(loss, thresholds) || allowed
    if (inBook && loss < thresholds.mistake) return Label.BOOK
    if (mateLetGo && !isBest) return Label.MISS
    if (flagged && chance && !isBest) return Label.MISS
    if (allowed) return Label.BLUNDER
    if (!flagged) {
        if (!obvious && !lost) {
            val without = secondScore ?: beforeScore
            if (sacrifice && loss < thresholds.excellent && without < thresholds.decided) return Label.BRILLIANT
            if (isBest && (decisive || opponentBlundered)) return Label.GREAT
        }
        if (isBest) return Label.BEST
        return if (aboutAsGood(loss, thresholds)) Label.EXCELLENT else Label.GOOD
    }
    return when (severityOf(loss, thresholds)) {
        Severity.BLUNDER -> Label.BLUNDER
        Severity.MISTAKE -> Label.MISTAKE
        else -> Label.INACCURACY
    }
}

/**
 * The move before, as far as the next move's label needs it: what it cost its mover (the
 * opponent), where it landed and whether it captured. `score.Previous`.
 */
data class Previous(val loss: Double, val to: Int, val captured: Boolean)

/**
 * Whether the move from the position measured by [beforeLines] to [after] let a winning chance
 * go, read off the engine's own lines (Tai, 2026-10-02: chess.com's miss -- the opponent's error
 * or a tactic already on the board, whichever gave it). The chance: the best line stood better
 * or winning ([standing] 3 or more, from 0.62), and the move played leaves a lower band. The
 * engine's evaluation decides, not where the chance came from: an opponent's error that left the
 * player no better was no chance to win. A position with a forced mate is read by its mate
 * instead ([mateLetGo]). `score.winning_chance`.
 */
fun winningChance(after: Board, beforeLines: List<Line>, afterLines: List<Line>, beforeScore: Double): Boolean {
    val top = beforeLines.firstOrNull()
    if (top?.mate != null && top.mate > 0) return false
    val left = 1.0 - positionScore(after, afterLines)
    return standing(beforeScore) >= standing(BETTER_FROM) && standing(left) < standing(beforeScore)
}

/**
 * The player's shortest forced mate once [move] is played, counted as the engine counts a mate
 * before it -- in the player's own moves, the one played included -- or null when the move
 * leaves none. Like is compared with like: when the search before the move has a line for the
 * move played (its candidates), that line's mate is the answer, the same search and depth the
 * mate it is set against comes from; else the search after the move, whose mate against the
 * opponent (in their moves) is one of the player's longer. `score.mate_after`.
 */
fun mateAfter(move: Move, after: Board, beforeLines: List<Line>, afterLines: List<Line>): Int? {
    if (after.isCheckmate()) return 1
    for (line in beforeLines) {
        if (line.moves.firstOrNull() == move.uci()) return if (line.mate != null && line.mate > 0) line.mate else null
    }
    val top = afterLines.firstOrNull()
    return if (top?.mate != null && top.mate < 0) 1 - top.mate else null
}

/**
 * The player had a forced mate in N, and [move] leaves none or only a longer one (Tai,
 * 2026-10-02 12:16: a mate in 5 traded for a move that keeps the advantage is a miss, a mate in
 * 1 traded for a mate in 2 is a miss, another mate in 3 for a mate in 3 is fine). The engine's
 * own first move is never one ([labelMove] checks it). `score.mate_let_go`.
 */
fun mateLetGo(move: Move, after: Board, beforeLines: List<Line>, afterLines: List<Line>): Boolean {
    val top = beforeLines.firstOrNull()
    if (top?.mate == null || top.mate <= 0) return false
    val kept = mateAfter(move, after, beforeLines, afterLines)
    return kept == null || kept > top.mate
}

/**
 * The label of [move] from [before], read off the two positions' measured lines: the
 * annotator's as it judges a move, and a saved review's made again from what it kept
 * (`GameReview.relabel`), so the two cannot drift. [previous]: the move before, null when it
 * is not known (the game's first move, or a move that did not lead here). `score.verdict`.
 */
fun verdict(
    before: Board,
    move: Move,
    beforeLines: List<Line>,
    afterLines: List<Line>,
    loss: Double,
    beforeScore: Double,
    decisive: Boolean,
    inBook: Boolean,
    sacrifice: Boolean,
    previous: Previous?,
    thresholds: Thresholds,
): Label {
    val after = before.copy(withStack = false)
    after.push(move)
    val best = beforeLines.firstOrNull()?.moves ?: emptyList()
    val erred = previous?.loss ?: 0.0
    // Taking back on the square the opponent just captured on. Taking a piece that merely
    // moved there is not obvious: that is how a blunder gets punished.
    val captures = before.isCapture(move)
    val recapture = previous != null && previous.captured && captures && move.to == previous.to
    // A capture nobody can take back is a grab, not a find; a king's capture always is.
    // Taking back is read by the rules: a piece guarded only by one pinned to its own king
    // is taken for nothing.
    val freeGrab = captures && !Exchange.canRetake(before, move)
    val obvious = recapture || freeGrab || before.legalMoves().size == 1
    val top = beforeLines.firstOrNull()
    val afterTop = afterLines.firstOrNull()
    // A forced mate the move walks into. The mates are the lines' own: the curve puts a
    // thousand centipawns a fortieth of a game below a mate.
    val mateAllowed = afterTop?.mate != null && afterTop.mate > 0 && !(top?.mate != null && top.mate < 0)
    return labelMove(
        loss,
        thresholds,
        isBest = best.isNotEmpty() && best[0] == move.uci(),
        inBook = inBook,
        decisive = decisive,
        sacrifice = sacrifice,
        beforeScore = beforeScore,
        opponentBlundered = erred >= thresholds.blunder,
        secondScore = beforeLines.getOrNull(1)?.expectedScore,
        chance = winningChance(after, beforeLines, afterLines, beforeScore),
        mateLetGo = mateLetGo(move, after, beforeLines, afterLines),
        mateAllowed = mateAllowed,
        obvious = obvious,
    )
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
    return Py.sum(accuracies) / accuracies.size
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
