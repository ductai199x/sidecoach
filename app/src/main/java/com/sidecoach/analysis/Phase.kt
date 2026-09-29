package com.sidecoach.analysis

import com.sidecoach.chess.BISHOP
import com.sidecoach.chess.BLACK
import com.sidecoach.chess.Board
import com.sidecoach.chess.KING
import com.sidecoach.chess.KNIGHT
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.chess.WHITE

/** Which phase of the game a position is in: `sidecoach/analysis/phase.py`. */

private val PHASE_VALUES = intArrayOf(0, 1, 3, 3, 5, 9, 0)

/**
 * Opening, middlegame or endgame, from the book and what is left on the board. What is left
 * decides first: a puzzle's position starts at move 1, and a rook and a minor piece each were
 * read as the opening (Tai's puzzle, 2026-09-28).
 */
fun phaseOf(board: Board, inBook: Boolean = false): String {
    var material = 0
    for (piece in board.pieceMap().values) {
        if (piece.pieceType == PAWN || piece.pieceType == KING) continue
        material += PHASE_VALUES[piece.pieceType]
    }
    val queens = board.pieces(QUEEN, WHITE).size + board.pieces(QUEEN, BLACK).size
    if (material <= 13 || (queens == 0 && material <= 20)) return "endgame"
    if (inBook || board.fullmoveNumber <= 8) return "opening"
    return "middlegame"
}
