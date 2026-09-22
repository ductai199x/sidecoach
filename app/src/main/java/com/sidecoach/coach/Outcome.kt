package com.sidecoach.coach

import com.sidecoach.analysis.isPassed
import com.sidecoach.chess.BLACK
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareName
import kotlin.math.abs

/**
 * Where a line leads: what is different on the board at the end of it, in a coach's words.
 * A port of `sidecoach/coach/outcome.py`.
 *
 * The engine's line is a move sequence with a score; the *point* of the line is what it
 * changes -- a pawn given up for a passed pawn, a file opened, a king brought to safety, a
 * piece won. Those are computed by playing the line out and comparing the two boards, so the
 * coach can say what the line is for without inventing it.
 */

private val OUTCOME_VALUES = intArrayOf(0, 1, 3, 3, 5, 9, 0)
private const val OUTCOME_FILES = "abcdefgh"

private fun material(board: Board, us: Boolean): Int {
    var total = 0
    for (piece in board.pieceMap().values) {
        val value = OUTCOME_VALUES[piece.pieceType]
        total += if (piece.colour == us) value else -value
    }
    return total
}

private fun passed(board: Board, colour: Boolean): Set<Int> =
    board.pieces(PAWN, colour).filter { isPassed(board, it) }.toSet()

private fun openFilesOf(board: Board): Set<Int> {
    val pawnFiles = HashSet<Int>()
    for (c in listOf(WHITE, BLACK)) for (s in board.pieces(PAWN, c)) pawnFiles.add(squareFile(s))
    return (0 until 8).toSet() - pawnFiles
}

/**
 * Material in pawns' worth: what a line wins is often a mix, so no piece is named for a sum
 * ("a knight and a pawn" would be a guess at which pieces).
 */
private fun amount(diff: Int): String {
    val n = abs(diff)
    if (n == 1) return "a pawn"
    if (n == 2) return "two pawns"
    if (n == 3) return "a piece's worth of material"
    return "material worth $n pawns"
}

/** What the board looks like after `moves` (UCI), how many were played, and who castled. */
fun playOut(board: Board, moves: List<String>): Triple<Board, Int, Map<Boolean, Boolean>> {
    val after = board.copy(withStack = false)
    val castled = hashMapOf(WHITE to false, BLACK to false)
    var played = 0
    for (uci in moves) {
        val move = runCatching { Move.fromUci(uci) }.getOrNull() ?: break
        if (!after.isLegal(move)) break
        if (after.isCastling(move)) castled[after.turn] = true
        after.push(move)
        played += 1
    }
    return Triple(after, played, castled)
}

/**
 * What the line changes, from the player's side, written for the harness: "the player wins a
 * pawn; the player gets a passed pawn on d5; the e-file opens; the player's king castles".
 * Empty when nothing of note changes.
 */
fun lineOutcome(board: Board, moves: List<String>, userColour: Boolean): String {
    val (after, played, castled) = playOut(board, moves)
    if (played == 0) return ""
    val us = userColour
    val them = !userColour
    val parts = ArrayList<String>()
    val diff = material(after, us) - material(board, us)
    if (diff > 0) {
        parts.add("the player wins ${amount(diff)}")
    } else if (diff < 0) {
        parts.add("the player gives up ${amount(diff)}")
    }
    val newOurs = passed(after, us) - passed(board, us)
    val newTheirs = passed(after, them) - passed(board, them)
    if (newOurs.isNotEmpty()) {
        parts.add(
            "the player gets a passed pawn on " +
                newOurs.sorted().joinToString(" and ") { squareName(it) },
        )
    }
    if (newTheirs.isNotEmpty()) {
        parts.add(
            "the opponent gets a passed pawn on " +
                newTheirs.sorted().joinToString(" and ") { squareName(it) },
        )
    }
    val opened = openFilesOf(after) - openFilesOf(board)
    if (opened.isNotEmpty()) {
        parts.add("the " + opened.sorted().joinToString(" and ") { "${OUTCOME_FILES[it]}-file" } + " opens")
    }
    for ((colour, whose) in listOf(us to "the player's", them to "the opponent's")) {
        if (castled[colour] == true) {
            parts.add("$whose king castles")
        } else {
            val beforeKing = board.king(colour)
            val afterKing = after.king(colour)
            if (beforeKing != null && afterKing != null && beforeKing != afterKing) {
                parts.add("$whose king goes to ${squareName(afterKing)}")
            }
        }
    }
    val bothQueens = board.pieces(QUEEN, us).isNotEmpty() && board.pieces(QUEEN, them).isNotEmpty()
    if (bothQueens && after.pieces(QUEEN, us).isEmpty() && after.pieces(QUEEN, them).isEmpty()) {
        parts.add("queens come off")
    }
    if (after.isCheckmate()) {
        parts.add(if (after.turn == them) "the opponent is mated" else "the player is mated")
    }
    return parts.joinToString("; ")
}
