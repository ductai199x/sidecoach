package com.sidecoach.coach

import com.sidecoach.analysis.clearlyWorse
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Analysis
import com.sidecoach.events.Line

/**
 * The moves besides the best one: which are about as good, which are clearly worse, and what
 * goes wrong with the worse, from the engine's lines for the position. A port of
 * `sidecoach/coach/alternatives.py`.
 *
 * The best move is not always one move: in a quiet position several are within a few hundredths
 * of the best line's score and any of them is fine. A coach that names only line 1 tells a player
 * who found another that they were wrong. And the move that is clearly better than the one the
 * player considered is better for a reason the board shows: the mate it passes up, the piece the
 * line gives away. Only what the board and the engine's line settle is said (a line cut off in
 * the middle of an exchange says nothing of material), and never a figure.
 */

/*
 * A clearly worse move is one the move labels grade a mistake or a blunder ([clearlyWorse]; Tai,
 * 2026-10-02: the grading bands, not a margin of our own).
 */

/** How far a line is read: the sheet's own ([LINE_PLIES]). */
const val READ_PLIES = 8

/** The most worse moves named in an answer. */
const val MOST_NAMED = 2

/**
 * A move clearly behind the best, with what goes wrong with it when the board says: a verb phrase
 * of the move -- "gives up a rook for a pawn", "passes up the mate".
 */
data class Worse(val san: String, val drop: Double, val wrong: String)

private fun firstSan(board: Board, line: Line): String? {
    val move = line.moves.firstOrNull()?.let { Move.fromUciOrNull(it) } ?: return null
    return if (board.isLegal(move)) board.san(move) else null
}

/**
 * What the line changes, from the player's side, when it is settled: it ends on a move that takes
 * nothing and gives no check, so no exchange is cut off in the middle. Null when it is not, and
 * nothing is said of it then.
 */
private fun settled(board: Board, moves: List<String>, player: Boolean): List<String>? {
    val (after, played, _) = playOut(board, moves)
    if (played == 0) return null
    val last = board.copy(withStack = false)
    var taking = false
    for (uci in moves.take(played)) {
        val move = Move.fromUci(uci)
        taking = last.isCapture(move)
        last.push(move)
    }
    if (taking || after.isCheck()) return null
    return lineOutcome(board, moves.take(played), player).split("; ").filter { it.isNotEmpty() }
}

/** The rest of the outcome's part that begins [lead], if any. */
private fun part(parts: List<String>?, lead: String): String? {
    for (one in parts.orEmpty()) if (one.startsWith(lead)) return one.removePrefix(lead)
    return null
}

/**
 * What goes wrong with the move [line] starts, against the [best] line, or null when the board
 * does not say: the mate it passes up, the mate it walks into, what the best line wins that this
 * one does not, what this one gives up. `alternatives.wrong_with`.
 */
fun wrongWith(board: Board, best: Line, line: Line, player: Boolean): String? {
    val bestMate = best.mate
    val lineMate = line.mate
    if (bestMate != null && bestMate > 0 && !(lineMate != null && lineMate > 0)) return "passes up the mate"
    if (lineMate != null && lineMate < 0) return "walks into a mate"
    val ours = settled(board, line.moves.take(READ_PLIES), player)
    val theirs = settled(board, best.moves.take(READ_PLIES), player)
    val won = part(theirs, "the player wins ")
    if (won != null && part(ours, "the player wins ") != won && ours != null) return "passes up winning $won"
    val lost = part(ours, "the player gives up ")
    return if (lost != null) "gives up $lost" else null
}

/**
 * The moves clearly behind the best that something can be said of, the nearest to the best first
 * -- the tempting ones -- at most [limit]. Empty when the player is not the one to move, or no
 * line is clearly behind. `alternatives.worse_moves`.
 */
fun worseMoves(board: Board, analysis: Analysis, player: Boolean, limit: Int = MOST_NAMED): List<Worse> {
    val lines = analysis.lines.filter { it.moves.isNotEmpty() }
    if (board.turn != player || lines.size < 2) return emptyList()
    val best = lines[0]
    val named = ArrayList<Worse>()
    val seen = HashSet<String?>()
    seen.add(firstSan(board, best))
    for (line in lines.drop(1)) {
        val drop = best.expectedScore - line.expectedScore
        val san = firstSan(board, line)
        if (!clearlyWorse(drop) || san == null || san in seen) continue
        seen.add(san)
        val wrong = wrongWith(board, best, line, player)
        if (wrong != null) named.add(Worse(san, drop, wrong))
        if (named.size == limit) break
    }
    return named
}

/** The worse moves for the model: "Bd3 gives up a bishop; Qd1 passes up the mate". */
fun worseText(worse: List<Worse>): String = worse.joinToString("; ") { "${it.san} ${it.wrong}" }

/** The same as the coach says it to the player, for the template: a sentence per move. */
fun worseSaid(worse: List<Worse>, past: Boolean = false): String {
    // [past]: of the position the player has moved from ("was clearly worse there").
    val verb = if (past) "was clearly worse there" else "is clearly worse"
    return worse.joinToString(" ") { "${it.san} $verb: it ${it.wrong}." }
}

/** "Nf6, Bc5 and Be7 are all fine here -- about equally good.", or the one move that is. */
fun fineSaid(fine: List<String>, past: Boolean = false): String {
    // [past]: of the position the player has moved from ("were all fine there").
    if (fine.size == 1) return if (past) "${fine[0]} was the move there." else "${fine[0]} is the move here."
    val names = listed(fine)
    val every = if (fine.size == 2) "both" else "all"
    if (past) return "$names were $every fine there -- about equally good."
    return "$names are $every fine here -- about equally good."
}

/**
 * The moves graded Good, a little behind the best: "Be7 and d6 are good too, a little behind." --
 * "" when there are none. `alternatives.good_said`.
 */
fun goodSaid(good: List<String>, past: Boolean = false): String {
    if (good.isEmpty()) return ""
    val verb = if (past) (if (good.size == 1) "was" else "were") else (if (good.size == 1) "is" else "are")
    return "${listed(good)} $verb good too, a little behind."
}

private fun listed(moves: List<String>): String =
    if (moves.size == 1) moves[0] else moves.dropLast(1).joinToString(", ") + " and ${moves.last()}"
