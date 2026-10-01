package com.sidecoach.ui

import com.sidecoach.analysis.positionScore
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.squareName
import com.sidecoach.events.Line
import com.sidecoach.events.ReviewJson
import com.sidecoach.events.ReviewedGame
import com.sidecoach.profile.GameSummaries
import org.json.JSONArray
import org.json.JSONObject

/**
 * A reviewed game as the review page draws it (`review_room.move_views` and its helpers): every
 * move with its squares, label and cost, the best move and its line, what follows the move, and
 * the position after -- its FEN, the evaluation from White's side and White's expected score.
 */
object ReviewViews {
    // The lines a move's card plays on the board (the best, and what follows the move), in plies.
    const val LINE_PLIES = 6

    /** The opening the game reached, as the book last named it. `review_room._opening`. */
    fun opening(reviewed: ReviewedGame): String? =
        reviewed.moves.asReversed().firstOrNull { !it.after.opening.isNullOrEmpty() }?.after?.opening

    /** `move_views`: a line comes as steps the board plays -- each move in SAN and UCI and the position it leaves. */
    fun moveViews(reviewed: ReviewedGame): JSONArray {
        val out = JSONArray()
        for (move in reviewed.moves) {
            val before = Board(move.fenBefore)
            val played = Move.fromUci(move.assessment.uci)
            val after = before.copy(withStack = false)
            after.push(played)
            val best = move.assessment.best
            val line = steps(before, best.take(LINE_PLIES))
            val follow = if (move.after.lines.isNotEmpty()) move.after.lines[0].moves.take(LINE_PLIES) else emptyList()
            out.put(
                JSONObject()
                    .put("ply", move.ply)
                    .put("san", move.san)
                    .put("side", move.side.value)
                    .put("uci", move.assessment.uci)
                    .put("from", squareName(played.from))
                    .put("to", squareName(played.to))
                    .put("fen_before", move.fenBefore)
                    .put("fen_after", after.fen())
                    .put("label", move.assessment.label?.let { ReviewJson.labelName(it) } ?: JSONObject.NULL)
                    .put("accuracy", move.assessment.accuracy ?: JSONObject.NULL)
                    .put("loss", GameSummaries.round(move.assessment.expectedLoss, 4))
                    .put("best_uci", best.firstOrNull() ?: JSONObject.NULL)
                    .put("best_san", if (line.length() > 0) line.getJSONObject(0).getString("san") else JSONObject.NULL)
                    .put("best_line", line)
                    .put("follow_line", steps(after, follow))
                    // What the move threatens: the mover's best next move were it their turn again.
                    .put("idea", move.after.threats.firstOrNull() ?: JSONObject.NULL)
                    .put("eval", whiteEval(after, move.after.lines))
                    .put("white_score", GameSummaries.round(whiteScore(after, move.after.lines), 4))
                    .put("tags", JSONArray(move.assessment.tags.map { it.name }.toSortedSet().toList()))
                    .put("opening", move.after.opening ?: JSONObject.NULL)
                    .put("clock_ms", move.clockMs ?: JSONObject.NULL)
                    .put("spent_ms", move.spentMs ?: JSONObject.NULL),
            )
        }
        return out
    }

    /**
     * A line of moves from [board] as far as it stays legal, each in SAN and UCI with the
     * position it leaves. `_steps`.
     */
    fun steps(board: Board, ucis: List<String>): JSONArray {
        val b = board.copy(withStack = false)
        val out = JSONArray()
        for (uci in ucis) {
            val move = Move.fromUciOrNull(uci) ?: break
            if (!b.isLegal(move)) break
            val san = b.san(move)
            b.push(move)
            out.put(JSONObject().put("san", san).put("uci", uci).put("fen", b.fen()))
        }
        return out
    }

    fun whiteScore(board: Board, lines: List<Line>): Double {
        val score = positionScore(board, lines)
        return if (board.turn == WHITE) score else 1.0 - score
    }

    /**
     * The evaluation from White's side, as the eval bar and pill read it: a mate in moves (+
     * White mates, - Black mates) or centipawns; a finished game, its result instead. `_white_eval`.
     */
    fun whiteEval(board: Board, lines: List<Line>): JSONObject {
        val outcome = board.outcome(claimDraw = true)
        if (outcome != null) {
            val result = when (outcome.winner) {
                true -> "1-0"
                false -> "0-1"
                null -> "1/2-1/2"
            }
            return JSONObject().put("cp", JSONObject.NULL).put("mate", JSONObject.NULL).put("result", result)
        }
        if (lines.isEmpty()) return JSONObject().put("cp", 0).put("mate", JSONObject.NULL).put("result", JSONObject.NULL)
        val sign = if (board.turn == WHITE) 1 else -1
        val line = lines[0]
        val mate = line.mate
        if (mate != null) return JSONObject().put("cp", JSONObject.NULL).put("mate", sign * mate).put("result", JSONObject.NULL)
        return JSONObject().put("cp", sign * (line.scoreCp ?: 0)).put("mate", JSONObject.NULL).put("result", JSONObject.NULL)
    }
}
