package com.sidecoach.analysis

import com.sidecoach.chess.BB_RANK_1
import com.sidecoach.chess.BB_RANK_8
import com.sidecoach.chess.Board
import com.sidecoach.chess.popcount

/**
 * Which phase of the game a position is in: opening, middlegame or endgame -- divided the way
 * Lichess divides a game (scalachess `Divider`, read 2026-10-01; Tai: "do openings the same way
 * chess.com or lichess does it"). `sidecoach/analysis/phase.py`.
 *
 * The middlegame begins at the first position where 10 pieces or fewer are left (kings and pawns
 * aside), or either side has fewer than 4 of its men on its own back rank, or the two sides' men
 * are mixed together on the board ([mixedness] over 150). The endgame begins at the first
 * position with 6 pieces or fewer. A game is divided once: a phase reached is never left
 * ([gamePhases]). Neither the move number nor the opening book decides it.
 */

val PHASES = listOf("opening", "middlegame", "endgame")

/** The 49 two-by-two squares of the board, a1-b2 first, rank by rank, with their row from 1. */
private val REGIONS: List<Pair<ULong, Int>> =
    (0 until 7).flatMap { y -> (0 until 7).map { x -> (0x0303uL shl (x + 8 * y)) to (y + 1) } }

/** The pieces on the board, kings and pawns aside. */
fun majorsAndMinors(board: Board): Int = popcount(board.occupied and (board.kings or board.pawns).inv())

/** Fewer than 4 of White's men on the first rank, or of Black's on the eighth. */
fun backrankSparse(board: Board): Boolean =
    popcount(BB_RANK_1 and board.occupiedCo[1]) < 4 || popcount(BB_RANK_8 and board.occupiedCo[0]) < 4

/** `Divider.score`: how mixed one two-by-two square is. */
private fun score(y: Int, white: Int, black: Int): Int = when (white) {
    0 -> when (black) {
        1 -> 1 + y
        2 -> if (y < 6) 2 + (6 - y) else 0
        3, 4 -> if (y < 7) 3 + (7 - y) else 0
        else -> 0
    }
    1 -> when (black) {
        0 -> 1 + (8 - y)
        1 -> 5 + kotlin.math.abs(4 - y)
        2 -> 4 + (7 - y)
        3 -> 5 + (7 - y)
        else -> 0
    }
    2 -> when (black) {
        0 -> if (y > 2) 2 + (y - 2) else 0
        1 -> 4 + (y - 1)
        2 -> 7
        else -> 0
    }
    3 -> when (black) {
        0 -> if (y > 1) 3 + (y - 1) else 0
        1 -> 5 + (y - 1)
        else -> 0
    }
    4 -> if (black == 0 && y > 1) 3 + (y - 1) else 0
    else -> 0
}

/** How far the two sides' men are mixed together, over every two-by-two square. */
fun mixedness(board: Board): Int {
    val white = board.occupiedCo[1]
    val black = board.occupiedCo[0]
    var acc = 0
    for ((region, y) in REGIONS) acc += score(y, popcount(region and white), popcount(region and black))
    return acc
}

/** The phase this position reads as, on its own. */
fun phaseOf(board: Board): String {
    val pieces = majorsAndMinors(board)
    if (pieces <= 6) return "endgame"
    if (pieces <= 10 || backrankSparse(board) || mixedness(board) > 150) return "middlegame"
    return "opening"
}

/** The later of two phases: the game's phase so far, and a position's. */
fun later(a: String, b: String): String = if (PHASES.indexOf(a) >= PHASES.indexOf(b)) a else b

/** Each position's phase in a game, in order: a phase reached is never left. */
fun gamePhases(boards: List<Board>): List<String> {
    val out = ArrayList<String>(boards.size)
    var now = "opening"
    for (board in boards) {
        now = later(now, phaseOf(board))
        out.add(now)
    }
    return out
}
