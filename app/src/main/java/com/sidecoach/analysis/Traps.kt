package com.sidecoach.analysis

import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Line

/**
 * The captures that look free and are not. A port of `sidecoach/analysis/traps.py`.
 *
 * A player at the board sees a piece to take -- the exchange on its square nets something by
 * the rules -- and takes it. When the engine's good moves do not include that capture, it is
 * either merely worse or a trap: after it, the opponent has something much better than the
 * piece. Capablanca-Marshall, 14.Qf3: the knight on g4 looks free, and 14.hxg4? Qh4 wins for
 * Black -- the annotator's main point, and the lesson a stuck player most needs ("the knight
 * is bait").
 *
 * The engine's multi-PV lines already score the captures among them; the rest of the tempting
 * captures, at most [TRAPS_READ] of the most tempting, are searched on their own. A capture is
 * a trap when it scores [TRAP_DROP] or more below the best move, or walks into mate where the
 * best move does not: in a lost position the scores are all near nothing, and 34.Kxh3 in
 * Johner-Nimzowitsch, mate in six, scored 0.06 below 34.Bxe4.
 */
object Traps {

    /**
     * Below the best move's expected score, a tempting capture is a trap: 14.hxg4 in
     * Capablanca-Marshall scores 0.49 to the best move's 0.60 at the search depth here (the
     * refutation shows only at depth 22), and it is the annotator's main point.
     */
    const val TRAP_DROP = 0.10

    /** The search after the capture: enough to see the punishment. */
    const val TRAP_DEPTH = 14

    /** The most tempting captures searched per position. */
    const val TRAPS_READ = 2

    /**
     * The captures that look like they win something -- the exchange on the square nets
     * material, recaptures played out -- that are not among the good lines, the most tempting
     * first.
     */
    fun tempting(board: Board, lines: List<Line>): List<Move> {
        if (lines.isEmpty()) return emptyList()
        val good = lines.filter { it.moves.isNotEmpty() && !worse(it, lines[0]) }.map { it.moves[0] }.toSet()
        val found = ArrayList<Pair<Int, Move>>()
        for (move in board.legalMoves()) {
            if (!board.isCapture(move) || move.uci() in good) continue
            val gain = Exchange.exchange(board, move)
            if (gain > 0) found.add(gain to move)
        }
        // A stable sort, most first, as `found.sort(key=lambda item: -item[0])`.
        return found.sortedByDescending { it.first }.map { it.second }
    }

    /**
     * The tempting captures that are traps, each as a line from the player's side: the capture,
     * then the engine's best play after it, scored for the player.
     */
    fun trapLines(engine: Engine, board: Board, lines: List<Line>, depth: Int = TRAP_DEPTH): List<Line> {
        if (lines.isEmpty() || board.isGameOver()) return emptyList()
        val scored = LinkedHashMap<String, Line>()
        for (line in lines) if (line.moves.isNotEmpty()) scored[line.moves[0]] = line
        val out = ArrayList<Line>()
        var searched = 0
        for (move in tempting(board, lines)) {
            var line = scored[move.uci()]
            if (line == null) {
                if (searched >= TRAPS_READ) continue
                searched += 1
                val after = board.copy(withStack = false)
                after.push(move)
                if (after.isGameOver()) continue
                val replies = engine.analyse(Board(after.fen()), depth = depth, multipv = 1)
                if (replies.isEmpty()) continue
                val reply = replies[0]
                line = Line(
                    moves = listOf(move.uci()) + reply.moves,
                    depth = reply.depth,
                    expectedScore = 1.0 - reply.expectedScore,
                    mate = if (reply.mate != null) -reply.mate else null,
                )
            }
            if (worse(line, lines[0])) out.add(line)
        }
        return out
    }

    private fun mated(line: Line): Boolean = line.mate != null && line.mate < 0

    /** A capture's line is a trap's: [TRAP_DROP] below the best, or mate where the best is not. */
    fun worse(line: Line, best: Line): Boolean =
        best.expectedScore - line.expectedScore >= TRAP_DROP || (mated(line) && !mated(best))
}
