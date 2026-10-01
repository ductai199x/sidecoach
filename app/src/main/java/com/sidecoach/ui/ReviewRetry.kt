package com.sidecoach.ui

import com.sidecoach.analysis.Book
import com.sidecoach.analysis.Engine
import com.sidecoach.analysis.GameAnnotator
import com.sidecoach.analysis.Measured
import com.sidecoach.analysis.Strength
import com.sidecoach.analysis.Tagger
import com.sidecoach.analysis.Thresholds
import com.sidecoach.analysis.positionScore
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.squareName
import com.sidecoach.coach.factsFromAnalysis
import com.sidecoach.coach.hintTemplate
import com.sidecoach.events.Analysis
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.ReviewJson
import com.sidecoach.events.ReviewedGame
import com.sidecoach.profile.GameSummaries
import org.json.JSONArray
import org.json.JSONObject

/**
 * Engines for a review, made as the room needs them: `make_engine(path=..., threads=...,
 * multipv=...)`, its memory the player's at most 128 MB (`min(engine.hash_mb, 128)`) and the
 * rest of the engine's settings the player's. Throws when there is no Stockfish to run
 * (`find_stockfish` found none).
 */
fun interface EngineMaker {
    fun make(threads: Int, multipv: Int): Engine
}

/**
 * Retry in a reviewed game (`ui/review_retry.py`, the plan's screen D): the player plays another
 * move in the position a move of the game was played from, and it is graded as the review
 * graded the game's own -- the same annotator and thresholds, the game's own measurements
 * adopted so the position is read as the review read it, the position the new move leads to
 * searched at the review's depth; and the coach's hints for the position, as in play (two hints,
 * then the move). The engine is made when a new move first needs a search, and closed with the
 * room.
 */
class Retries(
    private val reviewed: ReviewedGame,
    private val strength: Strength,
    private val gameId: String,
    private val thresholds: Thresholds,
    private val makeEngine: EngineMaker,
    private val tagger: Tagger?,
    private val book: () -> Book?,
) {
    private var engine: Engine? = null
    private var annotator: GameAnnotator? = null
    private val lock = Any()

    /**
     * The position move [ply] was played from: its FEN, whose move, and every legal move as the
     * board takes them (a square -> the squares its piece may go to). `position`.
     */
    fun position(ply: Int): JSONObject {
        val board = board(ply)
        val legal = LinkedHashMap<String, MutableList<String>>()
        for (move in board.legalMoves()) legal.getOrPut(squareName(move.from)) { ArrayList() }.add(squareName(move.to))
        return JSONObject()
            .put("ply", ply)
            .put("fen", board.fen())
            .put("side", if (board.turn == WHITE) "white" else "black")
            .put("legal", JSONObject().apply { for ((sq, to) in legal) put(sq, JSONArray(to.toSortedSet().toList())) })
    }

    /**
     * A new move from the position of move [ply], graded: its label, what it cost, its accuracy,
     * and the position it leads to. `attempt`.
     */
    fun attempt(ply: Int, uci: String): JSONObject {
        val board = board(ply)
        val move = Move.fromUciOrNull(uci)
        if (move == null || !board.isLegal(move)) throw IllegalArgumentException("$uci is not legal here")
        val assessment = synchronized(lock) {
            val annotator = ready()
            // The move before tells the annotator what it needs of it: whether the opponent had
            // just erred (a miss), whether this is a recapture. Both its positions are the
            // game's, adopted, so this searches nothing.
            if (ply > 1) {
                val before = reviewed.moves[ply - 2]
                annotator.assessMove(gameId, Board(before.fenBefore), Move.fromUci(before.assessment.uci), ply - 1)
            }
            annotator.assessMove(gameId, board, move, ply)
        }
        val after = board.copy(withStack = false)
        after.push(move)
        return JSONObject()
            .put("ply", ply)
            .put("uci", uci)
            .put("san", board.san(move))
            .put("fen", after.fen())
            .put("label", assessment.label?.let { ReviewJson.labelName(it) } ?: JSONObject.NULL)
            .put("loss", GameSummaries.round(assessment.expectedLoss, 4))
            .put("accuracy", assessment.accuracy ?: JSONObject.NULL)
            .put("only", analysisBefore(ply).onlyMove)
    }

    /**
     * The coach's hint at [level] for the position of move [ply] -- 1 the idea, 2 where to look,
     * 3 the move -- as the coach gives it in play (`coach.hint_template`). `hint`.
     */
    fun hint(ply: Int, level: Int): JSONObject {
        val board = board(ply)
        val analysis = analysisBefore(ply)
        val previous = if (ply > 1) reviewed.moves[ply - 2].assessment.uci else null
        val facts = factsFromAnalysis(board, analysis, lastMove = previous, inBook = analysis.inBook)
        val clamped = DisclosureLevel.of(maxOf(DisclosureLevel.IDEA.value, minOf(DisclosureLevel.MOVE.value, level)))
        val best = if (analysis.lines.isNotEmpty() && analysis.lines[0].moves.isNotEmpty()) analysis.lines[0].moves[0] else null
        return JSONObject()
            .put("ply", ply)
            .put("level", clamped.value)
            .put("text", hintTemplate(facts, clamped, board))
            .put("best", if (clamped.value >= DisclosureLevel.MOVE.value && best != null) best else JSONObject.NULL)
    }

    fun close() {
        synchronized(lock) {
            engine?.close()
            engine = null
            annotator = null
        }
    }

    private fun board(ply: Int): Board {
        if (ply < 1 || ply > reviewed.moves.size) throw IllegalArgumentException("no move $ply in the game")
        return Board(reviewed.moves[ply - 1].fenBefore)
    }

    private fun analysisBefore(ply: Int): Analysis = if (ply == 1) reviewed.start else reviewed.moves[ply - 2].after

    /** The annotator, with an engine and the game's own measurements (held under the lock). `_ready`. */
    private fun ready(): GameAnnotator {
        annotator?.let { return it }
        val made = makeEngine.make(threads = 2, multipv = 1)
        made.open()
        engine = made
        val annotator = GameAnnotator(
            made,
            depth = strength.depth,
            multipv = strength.multipv,
            thresholds = thresholds,
            tagger = tagger,
            book = book(),
            withThreats = false,
        )
        val analyses = listOf(reviewed.start) + reviewed.moves.map { it.after }
        annotator.adopt(analyses.map { measured(it) })
        this.annotator = annotator
        return annotator
    }

    companion object {
        /** A position of the game as the review measured it. `_measured`. */
        fun measured(analysis: Analysis): Measured = Measured(
            fen = analysis.fen,
            lines = analysis.lines,
            score = positionScore(Board(analysis.fen), analysis.lines),
            decisive = analysis.decisive,
            onlyMove = analysis.onlyMove,
            threats = analysis.threats.toList(),
            traps = analysis.traps.toList(),
            defences = analysis.defences.toList(),
            threatens = analysis.threatens.toList(),
        )
    }
}
