package com.sidecoach.analysis

import com.sidecoach.chess.BISHOP
import com.sidecoach.chess.Board
import com.sidecoach.chess.KING
import com.sidecoach.chess.KNIGHT
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.events.Line

/**
 * The recaptures a winning line declines, and what refutes them. A port of
 * `sidecoach/analysis/defences.py`.
 *
 * A player who sees a capture in the engine's line expects it to be taken back: Qxg3 is a queen
 * trade to anyone who sees fxg3. When the line's reply is not the recapture, taking back must
 * lose more, and that is the lesson -- "the pawn on f2 cannot take back: it shields their king".
 * Without it, "you can win their queen" read as a lie (Tai's chess.com puzzle, 2026-09-24:
 * 1...Qxg3 2.Rfe1, because 2.fxg3 Bc5+ is worse), and a rook that could not take back on b8
 * because it had to stop a pawn queening on f8 was taught as "a loose piece".
 *
 * Each of the player's captures in the first [PLAYER_MOVES] moves of the best line whose reply is
 * no recapture, while something could take back, has the natural recapture -- the cheapest piece
 * taking back -- played and searched. It is kept as a line from the root: the line up to the
 * capture, the recapture, and the engine's best play after it, scored for the player; kept only
 * when the recapture is no better for the opponent than the line's own reply.
 */
object Defences {

    /** The search after the recapture: enough to see the punishment. */
    const val DEFENCE_DEPTH = 14

    /** The player's first moves of the line looked at. */
    const val PLAYER_MOVES = 3

    /** Recaptures searched per position. */
    const val DEFENCES_READ = 2

    /**
     * The recapture is kept when it scores no better for the opponent than this below the line's
     * own reply (search noise between two depths).
     */
    const val DEFENCE_SLACK = 0.05

    private val VALUE = mapOf(
        PAWN to 1,
        KNIGHT to 3,
        BISHOP to 3,
        ROOK to 5,
        QUEEN to 9,
        KING to 100,
    )

    /**
     * The player's captures the line does not see taken back, as (the capture's index in the
     * line, the opponent's natural recapture): the cheapest piece that could take back.
     */
    fun declined(board: Board, line: Line): List<Pair<Int, Move>> {
        val out = ArrayList<Pair<Int, Move>>()
        val at = board.copy(withStack = false)
        val moves = line.moves.map { Move.fromUci(it) }
        for ((i, move) in moves.withIndex()) {
            if (i / 2 >= PLAYER_MOVES || !at.isLegal(move)) break
            if (i % 2 == 0 && at.isCapture(move) && i + 1 < moves.size) {
                val after = at.copy(withStack = false)
                after.push(move)
                val reply = moves[i + 1]
                val takers = after.legalMoves().filter { it.to == move.to }
                if (reply.to != move.to && takers.isNotEmpty()) {
                    // `min`: the first of the cheapest, in the order the moves are generated.
                    val cheapest = takers.minBy { m ->
                        val kind = after.pieceTypeAt(m.from)
                        VALUE[if (kind == 0) PAWN else kind] ?: 0
                    }
                    out.add(i to cheapest)
                }
            }
            at.push(move)
        }
        return out
    }

    /**
     * The declined recaptures of the best line, each as a line from the root: the line up to the
     * player's capture, the recapture, then the engine's best play, scored for the player.
     */
    fun defenceLines(engine: Engine, board: Board, lines: List<Line>, depth: Int = DEFENCE_DEPTH): List<Line> {
        if (lines.isEmpty() || lines[0].moves.isEmpty() || board.isGameOver()) return emptyList()
        val best = lines[0]
        val out = ArrayList<Line>()
        for ((i, recapture) in declined(board, best).take(DEFENCES_READ)) {
            val at = board.copy(withStack = false)
            // `push_uci`: the moves up to the capture are legal, as `declined` read them.
            for (uci in best.moves.take(i + 1)) at.push(Move.fromUci(uci))
            at.push(recapture)
            if (at.isGameOver()) continue
            val replies = engine.analyse(Board(at.fen()), depth = depth, multipv = 1)
            if (replies.isEmpty()) continue
            val reply = replies[0]
            val line = Line(
                moves = best.moves.take(i + 1) + recapture.uci() + reply.moves,
                depth = reply.depth,
                expectedScore = reply.expectedScore,
                mate = reply.mate,
            )
            val mates = line.mate != null && line.mate > 0
            if (mates || line.expectedScore >= best.expectedScore - DEFENCE_SLACK) out.add(line)
        }
        return out
    }
}
