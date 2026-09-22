package com.sidecoach.coach

import com.sidecoach.chess.BISHOP
import com.sidecoach.chess.Board
import com.sidecoach.chess.FILE_NAMES
import com.sidecoach.chess.KING
import com.sidecoach.chess.KNIGHT
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.Piece
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.chess.square
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareName

/**
 * What a move does on the board, computed, so the coach has a true mechanism to say. A port
 * of `sidecoach/coach/effects.py`.
 *
 * The coach's commonest invention was the reason a move was good or bad: "Qf5 hits the
 * knight", "b3 locks the queenside" -- claims the board contradicts. The messages gave it the
 * better move and a plan word, never what the move does. This module states that, from the
 * board alone. Every phrase is checked against the position the move leads to, so a coach
 * that repeats it is right.
 */

private val EFFECT_VALUE = intArrayOf(0, 1, 3, 3, 5, 9, 100)

private fun nameAt(board: Board, sq: Int): String {
    val piece = board.pieceAt(sq)
    val word = if (piece != null) COACH_PIECE_NAMES[piece.pieceType] else "piece"
    return "the $word on ${squareName(sq)}"
}

/** The enemy pieces `colour` attacks, as squares. */
private fun attackedBy(board: Board, colour: Boolean): Set<Int> {
    val out = HashSet<Int>()
    for (sq in 0 until 64) {
        val piece = board.pieceAt(sq) ?: continue
        if (piece.colour != colour) {
            if (board.isAttackedBy(colour, sq)) out.add(sq)
        }
    }
    return out
}

/** `colour`'s pieces (not the king, not pawns) with no defender. */
private fun undefended(board: Board, colour: Boolean): Set<Int> {
    val out = HashSet<Int>()
    for (sq in 0 until 64) {
        val piece = board.pieceAt(sq) ?: continue
        if (piece.colour == colour &&
            piece.pieceType != KING && piece.pieceType != PAWN &&
            !board.isAttackedBy(colour, sq)
        ) {
            out.add(sq)
        }
    }
    return out
}

/** Enemy pieces pinned to their king. */
private fun pinnedEnemies(board: Board, colour: Boolean): Set<Int> {
    val out = HashSet<Int>()
    for (sq in 0 until 64) {
        val piece = board.pieceAt(sq) ?: continue
        if (piece.colour != colour && board.isPinned(!colour, sq)) out.add(sq)
    }
    return out
}

/**
 * The move's effects on the board, most telling first, at most `limit` phrases.
 *
 * Computed on the position before and after the move; nothing here is inferred. The static
 * hazards -- a piece put or left where it is attacked and undefended -- are named only with
 * `hazards`: for a move the engine judged sound the tactics resolve them, and naming them
 * would mislead.
 */
fun describeEffects(board: Board, move: Move, limit: Int = 3, hazards: Boolean = true): List<String> {
    if (!board.isLegal(move)) return emptyList()
    val us = board.turn
    val them = !us
    val after = board.copy(withStack = false)
    after.push(move)
    val mover = board.pieceAt(move.from)
    val moverWord = if (mover != null) COACH_PIECE_NAMES[mover.pieceType] else "piece"
    val out = ArrayList<String>()

    if (after.isCheckmate()) return listOf("checkmate")
    if (board.isCastling(move)) out.add("castles, the king out of the centre")
    if (board.isCapture(move)) {
        val taken = board.pieceAt(move.to)
        if (board.isEnPassant(move)) {
            out.add("takes the pawn en passant")
        } else if (taken != null) {
            out.add("takes the ${COACH_PIECE_NAMES[taken.pieceType]} on ${squareName(move.to)}")
        }
    }
    if (after.isCheck()) out.add("gives check")

    // What the moved piece attacks from its new square, and what the move uncovers for the
    // rest of the army.
    val beforeAttacked = attackedBy(board, us)
    val afterAttacked = attackedBy(after, us)

    fun valueAt(sq: Int): Int {
        val piece = after.pieceAt(sq)
        return if (piece != null) EFFECT_VALUE[piece.pieceType] else 0
    }

    // By value, then by square: the desktop sorts the same way, because the operands are
    // sets and equal-valued pieces came out in CPython's hash order.
    val new = (afterAttacked - beforeAttacked)
        .sortedWith(compareByDescending<Int> { valueAt(it) }.thenBy { it })
    val byMover = new.filter { move.to in after.attackers(us, it) }
    val byOthers = new.filter { it !in byMover }
    val worth = byMover.filter { valueAt(it) >= 3 }
    if (worth.size >= 2) {
        out.add("forks " + worth.take(2).joinToString(" and ") { nameAt(after, it) })
    } else if (byMover.isNotEmpty()) {
        out.add("attacks " + nameAt(after, byMover[0]))
    }
    if (byOthers.isNotEmpty()) {
        val s = byOthers[0]
        val attacker = after.attackers(us, s).firstOrNull { it != move.to }
        if (attacker != null) {
            out.add("uncovers an attack on ${nameAt(after, s)} by ${nameAt(after, attacker)}")
        }
    }

    val pins = pinnedEnemies(after, us) - pinnedEnemies(board, us)
    for (s in pins.sorted()) {
        out.add("pins ${nameAt(after, s)} to the king")
        break
    }

    // The moved piece: was it attacked, and is it now?
    if (mover != null && mover.pieceType != KING) {
        val wasAttacked = board.isAttackedBy(them, move.from)
        val isAttacked = after.isAttackedBy(them, move.to)
        if (wasAttacked && !isAttacked) {
            out.add("gets the $moverWord out of attack")
        } else if (hazards && isAttacked && !after.isAttackedBy(us, move.to)) {
            out.add("puts the $moverWord where it is attacked and undefended")
        }
    }

    // Own pieces left undefended by the move, and attacked ones newly defended.
    val beforeLoose = undefended(board, us)
    val afterLoose = undefended(after, us)
    val left = if (hazards) (afterLoose - beforeLoose).filter { it != move.to } else emptyList()
    for (s in left.sortedWith(compareByDescending<Int> { valueAt(it) }.thenBy { it })) {
        val tail = if (after.isAttackedBy(them, s)) ", and it is attacked" else ""
        out.add("leaves ${nameAt(after, s)} undefended$tail")
        break
    }
    val covered = (beforeLoose - afterLoose).filter { it != move.from && after.isAttackedBy(them, it) }
    for (s in covered.sortedWith(compareByDescending<Int> { valueAt(it) }.thenBy { it })) {
        out.add("defends ${nameAt(after, s)}, which was attacked")
        break
    }

    // A rook or queen onto a file with no pawn of its own on it.
    if (mover != null && (mover.pieceType == ROOK || mover.pieceType == QUEEN)) {
        val fileIndex = squareFile(move.to)
        val ownPawn = (0 until 8).any { board.pieceAt(square(fileIndex, it)) == Piece(PAWN, us) }
        val theirPawn = (0 until 8).any { board.pieceAt(square(fileIndex, it)) == Piece(PAWN, them) }
        if (squareFile(move.from) != fileIndex && !ownPawn) {
            val kind = if (!theirPawn) "open" else "half-open"
            out.add("takes the $kind ${FILE_NAMES[fileIndex]}-file")
        }
    }

    return out.take(limit)
}

/**
 * The effects joined for a message, or an empty string when the move does nothing the board
 * can name (a quiet developing move).
 */
fun effectsText(board: Board, move: Move, limit: Int = 3, hazards: Boolean = true): String =
    describeEffects(board, move, limit, hazards).joinToString("; ")
