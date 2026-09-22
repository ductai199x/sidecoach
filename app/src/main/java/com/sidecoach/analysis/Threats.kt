package com.sidecoach.analysis

import com.sidecoach.chess.BLACK
import com.sidecoach.chess.Board
import com.sidecoach.events.Line

/**
 * What the opponent would do if it were their turn: `sidecoach/analysis/threats.py`.
 *
 * A null move hands them the move without changing anything else, so the engine's answer is
 * exactly the threat the position carries. This is what lets the coach ask "what is your
 * opponent trying to do?" instead of naming a move, which is the first rung of the hint
 * ladder in DESIGN.md section 5.5.
 */

/**
 * The opponent's best lines after a null move, from their point of view. Empty when passing
 * is not a question that can be asked: the side to move is in check (they cannot pass out of
 * it), or the game is over.
 */
fun threatLines(engine: Engine, board: Board, depth: Int, multipv: Int = 1): List<Line> {
    if (board.isCheck() || board.isGameOver(claimDraw = true)) return emptyList()
    val passed = board.copy(withStack = false)
    // python-chess's null move: the rights are cleaned, the en-passant square goes, the
    // clocks tick, and the turn changes hands.
    passed.castlingRights = passed.cleanCastlingRights()
    passed.epSquare = null
    passed.halfmoveClock += 1
    if (passed.turn == BLACK) passed.fullmoveNumber += 1
    passed.turn = !passed.turn
    // Re-read the position from its own description, as the desktop does: the position is
    // all the engine needs, and a history with a null move in it is not sendable.
    val reread = Board(passed.fen())
    if (!reread.isValid()) return emptyList()
    return engine.analyse(reread, depth = depth, multipv = multipv)
}

/** The first move of each threat line, in UCI. */
fun threatMoves(engine: Engine, board: Board, depth: Int, multipv: Int = 1): List<String> =
    threatLines(engine, board, depth, multipv).mapNotNull { it.moves.firstOrNull() }
