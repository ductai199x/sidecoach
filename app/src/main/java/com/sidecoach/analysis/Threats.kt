package com.sidecoach.analysis

import com.sidecoach.chess.BLACK
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
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
    if (board.isCheck() || gameOver(board)) return emptyList()
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

/** The search for what a quiet best move threatens: enough to see a mate or a piece won. */
const val AIM_DEPTH = 14

/**
 * Only a best line at least this good for the side to move is asked what it threatens: a quiet
 * move that wins, not one that holds.
 */
const val AIM_MIN = 0.6

/**
 * What the best move threatens, when it is quiet: the side to move's best line after it and a
 * null move, from the side to move's view. A quiet move -- no capture, no check, no promotion --
 * that wins has a threat as its point, and the threat is the lesson: "the first move is quiet,
 * but it threatens mate" where "their knight on c3 is the target" said nothing of how (the 28
 * wins of the puzzle sample whose first hint was bare, 2026-09-25). Empty for a move that is not
 * quiet, a line that does not win, and a move that gives check or ends the game (no free move can
 * be handed over then).
 */
fun aimLines(engine: Engine, board: Board, lines: List<Line>, depth: Int = AIM_DEPTH): List<Line> {
    if (lines.isEmpty() || lines[0].moves.isEmpty() || lines[0].expectedScore < AIM_MIN) return emptyList()
    val first = Move.fromUciOrNull(lines[0].moves[0]) ?: return emptyList()
    if (!board.isLegal(first) ||
        board.isCapture(first) ||
        board.givesCheck(first) ||
        first.promotion != 0
    ) {
        return emptyList()
    }
    val after = board.copy(withStack = false)
    after.push(first)
    return threatLines(engine, after, depth = depth)
}

/** The first move of each threat line, in UCI. */
fun threatMoves(engine: Engine, board: Board, depth: Int, multipv: Int = 1): List<String> =
    threatLines(engine, board, depth, multipv).mapNotNull { it.moves.firstOrNull() }
