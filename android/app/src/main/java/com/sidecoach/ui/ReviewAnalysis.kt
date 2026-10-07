package com.sidecoach.ui

import com.sidecoach.analysis.Engine
import com.sidecoach.analysis.WholeSearcher
import com.sidecoach.analysis.Strength
import com.sidecoach.analysis.positionScore
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.PgnGame
import com.sidecoach.chess.PgnNode
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.squareName
import com.sidecoach.events.Line
import com.sidecoach.profile.GameSummaries
import com.sidecoach.profile.Library
import com.sidecoach.profile.LibraryRow
import org.json.JSONArray
import org.json.JSONObject
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale

/**
 * Analysis in a reviewed game (`ui/review_analysis.py`, the plan's screen F): the player's own
 * moves from any position of the game, the engine's best lines for the position on the board,
 * the moves the player has played from it in their other games (the explorer), and the game as
 * a PGN -- the review's verdicts on it, the player's side lines with it -- to copy or keep.
 *
 * The page holds the tree of moves; a move on the board is checked and named here, and each
 * position the board shows is asked of the engine and the explorer. The engine is the room's
 * own, made when the analysis first needs it.
 */
object ReviewAnalysis {
    // The engine's best lines for a position, and how far each is shown.
    const val LINES = 3
    const val LINE_PLIES = 10

    // A review's labels as PGN's move marks, and as the word its comment begins with. A miss has
    // no mark of its own: it takes its loss's (`severity`) -- "?" was put on every miss, one that
    // lost nothing (a mate let go, still winning) and a blunder-sized one alike.
    val NAGS = mapOf("brilliant" to 3, "great" to 1, "inaccuracy" to 6, "mistake" to 2, "blunder" to 4)

    /** The PGN mark of a reviewed move: its label's, a miss its loss's, else none. `nag`. */
    fun nag(label: String?, severity: String?): Int? = if (label == "miss") NAGS[severity] else NAGS[label]
    val WORDS = linkedMapOf(
        "brilliant" to "Brilliant",
        "great" to "Great move",
        "best" to "Best",
        "excellent" to "Excellent",
        "good" to "Good",
        "book" to "Book",
        "inaccuracy" to "Inaccuracy",
        "mistake" to "Mistake",
        "miss" to "Miss",
        "blunder" to "Blunder",
    )

    /** Every legal move as the board takes them: a square -> the squares its piece may go to. `legal_of`. */
    fun legalOf(board: Board): JSONObject {
        val legal = LinkedHashMap<String, MutableSet<String>>()
        for (m in board.legalMoves()) legal.getOrPut(squareName(m.from)) { sortedSetOf() }.add(squareName(m.to))
        return JSONObject().apply { for ((sq, to) in legal) put(sq, JSONArray(to.toList())) }
    }

    /**
     * A position as the analysis board takes it: whose move, its legal moves, and the result
     * when the game is over there. `position`.
     */
    fun position(fen: String): JSONObject {
        val board = Board(fen)
        return JSONObject()
            .put("fen", board.fen())
            .put("side", if (board.turn == WHITE) "white" else "black")
            .put("legal", legalOf(board))
            .put("over", if (board.isGameOver(claimDraw = false)) board.result() else JSONObject.NULL)
    }

    /**
     * A move played on the analysis board: named, and the position it leaves.
     * IllegalArgumentException when it is not legal there. `move`.
     */
    fun move(fen: String, uci: String): JSONObject {
        val board = Board(fen)
        val played = Move.fromUciOrNull(uci) ?: throw IllegalArgumentException("$uci is not a move")
        if (!board.isLegal(played)) throw IllegalArgumentException("$uci is not legal here")
        val san = board.san(played)
        board.push(played)
        val out = JSONObject().put("uci", played.uci()).put("san", san)
        val at = position(board.fen())
        for (k in at.keys()) out.put(k, at.get(k))
        return out
    }

    /** A line of moves from [board] as far as it stays legal, each in SAN and UCI with the position it leaves. `steps`. */
    fun steps(board: Board, ucis: List<String>): JSONArray = ReviewViews.steps(board, ucis)

    /** A line's evaluation from White's side: a mate in moves (+ White mates) or centipawns. `white_eval`. */
    fun whiteEval(board: Board, line: Line): JSONObject {
        val sign = if (board.turn == WHITE) 1 else -1
        val mate = line.mate
        if (mate != null) return JSONObject().put("cp", JSONObject.NULL).put("mate", sign * mate)
        return JSONObject().put("cp", sign * (line.scoreCp ?: 0)).put("mate", JSONObject.NULL)
    }

    /** The lines found for [board] as the page draws them. The engine's part of `Lines.lines`. */
    fun linesJson(board: Board, found: List<Line>): JSONObject {
        val lines = JSONArray()
        for (line in found) {
            val steps = steps(board, line.moves.take(LINE_PLIES))
            if (steps.length() > 0) lines.put(JSONObject().put("eval", whiteEval(board, line)).put("steps", steps))
        }
        val white = if (found.isNotEmpty()) positionScore(board, found) else null
        return JSONObject()
            .put("fen", board.fen())
            .put("depth", if (found.isNotEmpty()) found[0].depth else 0)
            .put(
                "white_score",
                white?.let { GameSummaries.round(if (board.turn == WHITE) it else 1.0 - it, 4) } ?: JSONObject.NULL,
            )
            .put("lines", lines)
    }

    /**
     * The engine's best lines for the positions the analysis board shows: one engine, made on
     * the first position asked of it and closed with the room; one position at a time, measured
     * whole the one-line way ([WholeSearcher]) -- three lines to the review's depth at once
     * exploded on decided and forcing positions, a minute and more for one position. `Lines`.
     */
    class Lines(private val strength: Strength, private val makeEngine: EngineMaker) {
        private var searcher: WholeSearcher? = null
        private val lock = Any()

        /** The best LINES lines from [fen] at the review's depth, each from White's side and played out. */
        fun lines(fen: String): JSONObject {
            val board = Board(fen)
            val found = synchronized(lock) { ready().analyse(board) }
            return linesJson(board, found)
        }

        fun close() {
            synchronized(lock) {
                searcher?.close()
                searcher = null
            }
        }

        private fun ready(): WholeSearcher {
            searcher?.let { return it }
            val engine = makeEngine.make(threads = 1, multipv = LINES)
            engine.open()
            return WholeSearcher(engine, lines = LINES, depth = strength.depth).also { searcher = it }
        }
    }

    // ---- the explorer: the player's own games ----

    /**
     * The moves played from each position of the player's games -- every game of the library,
     * imported or watched -- how often, and how those games went for the player. `Explorer`.
     */
    class Explorer(rows: Iterable<LibraryRow>) {
        private class Stats(val san: String) {
            var games = 0
            var won = 0
            var drawn = 0
            var lost = 0
            var lastKey: String? = null
            var lastAt = 0.0
        }

        // position -> uci -> its stats, in the order first seen
        private val moves = HashMap<String, LinkedHashMap<String, Stats>>()

        init {
            for (row in rows) {
                val game = Library.gamesIn(row.pgn).firstOrNull() ?: continue
                val outcome = outcome(row.result, row.userSide)
                val board = game.board()
                val seen = HashSet<Pair<String, String>>() // a position played twice in a game counts once
                for (m in game.mainlineMoves()) {
                    val at = board.epd()
                    val uci = m.uci()
                    if (seen.add(at to uci)) {
                        val stats = moves.getOrPut(at) { LinkedHashMap() }.getOrPut(uci) { Stats(board.san(m)) }
                        stats.games += 1
                        when (outcome) {
                            "won" -> stats.won += 1
                            "drawn" -> stats.drawn += 1
                            "lost" -> stats.lost += 1
                        }
                        val whenPlayed = row.playedAt ?: row.addedAt
                        if (stats.lastKey == null || whenPlayed > stats.lastAt) {
                            stats.lastKey = row.key
                            stats.lastAt = whenPlayed
                        }
                    }
                    board.push(m)
                }
            }
        }

        /** The moves the player's games went on with from [fen], the most played first. `explore`. */
        fun explore(fen: String): JSONObject {
            val found = moves[Board(fen).epd()].orEmpty()
            val sorted = found.entries.sortedWith(compareBy({ -it.value.games }, { it.value.san }))
            val out = JSONArray()
            for ((uci, s) in sorted) {
                out.put(
                    JSONObject().put("uci", uci).put("san", s.san).put("games", s.games).put("won", s.won)
                        .put("drawn", s.drawn).put("lost", s.lost)
                        .put("last", s.lastKey?.let { JSONObject().put("key", it).put("played_at", s.lastAt) } ?: JSONObject.NULL),
                )
            }
            return JSONObject().put("games", sorted.sumOf { it.value.games }).put("moves", out)
        }
    }

    /** How a game went for the player: won, drawn or lost; null when it is not known. `_outcome`. */
    fun outcome(result: String?, side: String?): String? {
        if (side != "white" && side != "black") return null
        if (result != "1-0" && result != "0-1" && result != "1/2-1/2") return null
        if (result == "1/2-1/2") return "drawn"
        return if ((result == "1-0") == (side == "white")) "won" else "lost"
    }

    // ---- the game as a PGN ----

    /**
     * A reviewed move's comment: its kind, the better move when it was not the best, and the
     * evaluation after it ([%eval], as Lichess and chess.com write it). `_comment`.
     */
    fun comment(label: String?, bestSan: String?, played: String, eval: JSONObject?): String {
        val words = ArrayList<String>()
        WORDS[label]?.let { words.add("$it.") }
        if (bestSan != null && bestSan != played && label in setOf("inaccuracy", "mistake", "miss", "blunder")) {
            words.add("$bestSan was best.")
        }
        var mark = ""
        if (eval != null) {
            if (!eval.isNull("mate")) mark = "[%eval #${eval.getInt("mate")}]"
            else if (!eval.isNull("cp")) mark = "[%eval ${String.format(Locale.ROOT, "%.2f", eval.getInt("cp") / 100.0)}]"
        }
        return (words + mark).filter { it.isNotEmpty() }.joinToString(" ")
    }

    /**
     * The game as a PGN: its headers and moves as they came, the player's side lines from the
     * analysis board, and -- [annotated], with a review -- each move's kind as PGN's mark and in
     * words with the evaluation after it, and the AI coach's words on the key moves when they
     * were written. [tree]: the analysis board's moves from the start, each {"uci",
     * "children"}, the game's own continuation first. `pgn`.
     */
    fun pgn(row: LibraryRow, moves: JSONArray?, tree: JSONArray, annotated: Boolean = true, words: JSONObject? = null): String {
        val game = Library.gamesIn(row.pgn).firstOrNull() ?: throw IllegalArgumentException("the game's PGN has no moves")
        val views = HashMap<Int, JSONObject>()
        if (moves != null) for (i in 0 until moves.length()) moves.getJSONObject(i).let { views[it.getInt("ply")] = it }
        val texts = words?.optJSONObject("texts")

        fun annotate(node: PgnNode, ply: Int) {
            val view = views[ply] ?: return
            if (!annotated) return
            val label = if (view.isNull("label")) null else view.getString("label")
            nag(label, if (view.isNull("severity")) null else view.optString("severity"))?.let { node.nags.add(it) }
            val said = texts?.optJSONObject(ply.toString())
            val parts = arrayListOf(
                comment(label, if (view.isNull("best_san")) null else view.getString("best_san"), view.getString("san"), view.optJSONObject("eval")),
            )
            for (p in listOf("why", "better", "lesson")) said?.optString(p)?.takeIf { it.isNotEmpty() }?.let { parts.add(it) }
            val mine = parts.filter { it.isNotEmpty() }.joinToString(" ")
            node.comment = if (node.comment.isNotEmpty()) "$mine ${node.comment}".trim() else mine
        }

        fun boardAt(node: PgnNode): Board {
            val path = ArrayList<Move>()
            var at: PgnNode? = node
            while (at != null && at !is PgnGame) {
                path.add(0, (at as com.sidecoach.chess.PgnChild).move)
                at = at.parent
            }
            val board = game.board()
            for (m in path) board.push(m)
            return board
        }

        fun graft(node: PgnNode, children: JSONArray) {
            for (i in 0 until children.length()) {
                val child = children.getJSONObject(i)
                val m = Move.fromUciOrNull(child.optString("uci")) ?: continue
                if (!boardAt(node).isLegal(m)) continue
                val on = node.variations.firstOrNull { it.move == m } ?: node.addVariation(m)
                graft(on, child.optJSONArray("children") ?: JSONArray())
            }
        }

        // The game's own moves first, annotated, then the player's lines grafted on.
        var node: PgnNode = game
        var ply = 0
        while (node.variations.isNotEmpty()) {
            node = node.variations[0]
            ply += 1
            annotate(node, ply)
        }
        graft(game, tree)
        if (annotated && moves != null && moves.length() > 0) game.headers["Annotator"] = "Sidecoach"
        return game.toString()
    }

    private val UNSAFE = Regex("[^\\p{L}\\p{N}_ .-]+")
    private val DATE = DateTimeFormatter.ofPattern("yyyy-MM-dd")

    /** A file name for the game's PGN: its players and date, as a file can be named. `pgn_name`. */
    fun pgnName(row: LibraryRow, zone: ZoneId = ZoneId.systemDefault()): String {
        val at = Instant.ofEpochMilli(((row.playedAt ?: row.addedAt) * 1000).toLong()).atZone(zone)
        val name = "${row.white?.ifEmpty { null } ?: "White"} vs ${row.black?.ifEmpty { null } ?: "Black"} ${DATE.format(at)}"
        return UNSAFE.replace(name, "_").trim() + ".pgn"
    }
}
