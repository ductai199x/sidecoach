package com.sidecoach.coach

import com.sidecoach.analysis.PIECE_NAMES
import com.sidecoach.analysis.gameOver
import com.sidecoach.analysis.atFullDepth
import com.sidecoach.analysis.restOfLine
import com.sidecoach.analysis.sanLine
import com.sidecoach.chess.BB_EMPTY
import com.sidecoach.chess.BB_SQUARES
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.parseSquare
import com.sidecoach.chess.squareName
import com.sidecoach.events.Line
import org.json.JSONArray
import org.json.JSONObject

/**
 * The engine at the coach model's call, while it answers a question of the player's:
 * `coach/engine_tools.py`.
 *
 * The coach's request carries what was computed before the question came: the engine's lines to
 * eight plies, the facts, the lists of what stands on the board. Asked "ok and after that?" past
 * the end of the line, or "whose bishop? ... he will just move away to the right, no?", the
 * model had nothing more to go on; a separate model call guessed beforehand what to examine and
 * guessed wrong ("is my bishop safe?"), and the answers came out repeated or invented (Tai's
 * puzzle, 2026-09-28). With these tools the model asks the engine itself, from the whole
 * conversation, for what it needs: the position after any moves -- who stands better and the
 * engine's lines with what each move does -- and what attacks a square.
 *
 * Everything returned is the engine's and the board's, in the harness's words ("the player",
 * "the opponent", expected scores from the player's side). The boards the tools went through are
 * kept ([frames]): the draft check then reads a reply about them against them.
 */

/** Rounds of tool calls a reply may make before it must answer with what it has. */
const val TOOL_ROUNDS = 3

/** What the Task says of them. */
const val TOOL_WORDS =
    "You can ask the engine yourself, with the tools: the position after any moves you give " +
        "it (who stands better, its best lines, what each move does) and what attacks a square. " +
        "Ask it for what the sections above do not show -- what comes after the end of a line, " +
        "a move they propose, whose piece a move takes, whether a square is safe -- before you " +
        "answer, and say only what it returns."

/** The engine lines an analysis gives, and the plies each. */
private const val LINES = 3
private const val LINE_PLIES = 8

private fun startSpec(): JSONObject = JSONObject()
    .put("type", "string")
    .put("enum", JSONArray(listOf("now", "before_last_move")))
    .put(
        "description",
        "now (the default): the board as it stands; before_last_move: the position the last " +
            "move was played from",
    )

private fun movesSpec(): JSONObject = JSONObject()
    .put("type", "array")
    .put("items", JSONObject().put("type", "string"))
    // "From the start position" was read as the game's first position: a model that knows a game
    // begins with 1. e4 could send the whole game (the code review, 2026-09-29).
    .put(
        "description",
        "Moves in standard notation, in order and for both sides, played from the position " +
            "`start` names (the board now by default) -- never the game's earlier moves; empty " +
            "for that position itself.",
    )

/** The tools' specs in the OpenAI form: `engine_tools.SPECS`. */
fun engineToolSpecs(): JSONArray = JSONArray()
    .put(
        JSONObject().put("type", "function").put(
            "function",
            JSONObject()
                .put("name", "analyse")
                .put(
                    "description",
                    "Stockfish on the position reached by playing `moves` from the board now (or " +
                        "from before the last move): whose move it is, who stands better, the " +
                        "engine's best lines with what each move does, and what hangs, is pinned or " +
                        "can be taken there. When moves are given, also how the first one rates " +
                        "against the engine's best. For what the messages do not already show: what " +
                        "comes after the end of a line, a move the player proposes, what a move " +
                        "would allow.",
                )
                .put(
                    "parameters",
                    JSONObject()
                        .put("type", "object")
                        .put("properties", JSONObject().put("moves", movesSpec()).put("start", startSpec()))
                        .put("required", JSONArray(listOf("moves"))),
                ),
        ),
    )
    .put(
        JSONObject().put("type", "function").put(
            "function",
            JSONObject()
                .put("name", "square")
                .put(
                    "description",
                    "What stands on a square, which pieces of each side attack it, and which " +
                        "pieces of the side to move can go there, in the position reached by " +
                        "playing `moves` (as for analyse): whether a piece there is defended, " +
                        "whether a square is safe to go to, whether a check can be blocked there.",
                )
                .put(
                    "parameters",
                    JSONObject()
                        .put("type", "object")
                        .put(
                            "properties",
                            JSONObject()
                                .put("square", JSONObject().put("type", "string").put("description", "a square, such as g1"))
                                .put("moves", movesSpec())
                                .put("start", startSpec()),
                        )
                        .put("required", JSONArray(listOf("square"))),
                ),
        ),
    )

/** A move as people write it, with its number or without: "4. Kg2", "4...Rg1+", "Kg2". */
private val NUMBER = Regex("^\\d+\\.+")
private val SQUARE_NAME = Regex("^[a-h][1-8]$")

/**
 * The tools of one request: the board now, the position before the last move, the engine, and
 * whose side "the player" is; and the lines the record shows for each of the two ([knownNow],
 * [knownBefore]). Moves on one of those lines are read from it, as the model was shown it, and
 * the rest go to the engine: a second search after a move the lines play can answer it
 * differently, and the model, shown both, believed each in turn (the dev audit, 2026-09-26; Tai,
 * 2026-09-29: "only then invoke a call to stockfish"). `engine_tools.EngineTools`.
 */
class EngineTools(
    now: Board,
    before: Board?,
    private val engine: EngineLike,
    private val player: Boolean,
    knownNow: List<Line> = emptyList(),
    knownBefore: List<Line> = emptyList(),
    private val depth: Int = 14,
) {
    // With their moves: Stockfish reads a repetition from them, and a move that repeats a
    // position the third time ends the game ([gameOver]).
    private val now: Board = now.copy()
    private val before: Board? = before?.copy()
    private val known: Map<String, List<Line>> =
        mapOf("now" to knownNow.toList(), "before_last_move" to knownBefore.toList())
    val frames: MutableList<Board> = ArrayList()
    val log: MutableList<String> = ArrayList() // what was asked, one entry a call, for the record
    val specs: JSONArray = engineToolSpecs()

    /** The tool's answer, as text for the model; a bad call is told what was wrong. */
    fun run(name: String, arguments: JSONObject): String {
        // The values as Python reads them: a null is None, falsy, and str() of anything
        // is Python's -- {"square": null} was 'null', and {"moves": false} a move
        // (the parity audit, 2026-09-29).
        val args = pyJsonOr(arguments.toString(), null) as? Map<*, *> ?: emptyMap<String, Any?>()
        if (args.containsKey("_raw")) {
            // Arguments that were no JSON object were analysed silently as the board now, and
            // the model never knew (the deep prompt review, 2026-10-01).
            log.add("$name(unreadable arguments)")
            return "The arguments were not a JSON object (${pyRepr(pyStr(args["_raw"]).take(80))}), so " +
                "nothing was analysed: call again with e.g. {\"moves\": [\"Nf3\"], \"start\": \"now\"}."
        }
        val start = pyStr(args["start"].takeIf { pyTruthy(it) } ?: "now")
        if (start != "now" && start != "before_last_move") {
            log.add("$name(unknown start ${pyRepr(start)})")
            return "No start named ${pyRepr(start)}, so nothing was analysed: start is \"now\" or " +
                "\"before_last_move\"."
        }
        val moves = tokens(args["moves"].takeIf { pyTruthy(it) } ?: emptyList<Any?>())
        val target = pyStr(args["square"].takeIf { pyTruthy(it) } ?: "")
        log.add(
            "$name(${moves.joinToString(" ")}" + (if (start != "now") "; $start" else "") +
                (if (name == "square") "; $target)" else ")"),
        )
        val board = (if (start == "before_last_move") before else now)
            ?: return "There is no last move to start from: the tools start from the board now."
        return when (name) {
            "analyse" -> analyse(board, moves, start)
            "square" -> square(board, moves, target, start)
            else -> "No tool named ${pyRepr(name)}: the tools are analyse and square."
        }
    }

    // ---- the tools --------------------------------------------------------------------------

    private fun analyse(start: Board, moves: List<String>, where: String): String {
        val (after, played, illegal) = play(start, moves)
        var head = heading(start, played, where)
        if (illegal.isNotEmpty()) head += " $illegal is not a legal move there, so the analysis stops before it."
        if (gameOver(after)) return "$head The game is over there: ${result(after)}."
        val known = this.known[where] ?: emptyList()
        val own = if (played.isNotEmpty()) restOfLine(known, played.map { it.uci() }) else null
        val lines: List<Line>
        val rating: String
        try {
            lines = when {
                own != null -> listOf(own)
                played.isEmpty() && known.isNotEmpty() -> known.take(LINES)
                else -> engine.analyse(after, depth).take(LINES)
            }
            rating = if (played.isNotEmpty()) rating(start, played[0], origin(where), known) else ""
        } catch (exc: Exception) { // the engine gone or a position it refused
            return "$head The engine could not look at that position (${exc.javaClass.simpleName})."
        }
        val parts = ArrayList<String>()
        parts.add("$head ${side(after.turn)} to move" + if (after.isCheck()) ", in check." else ".")
        if (rating.isNotEmpty()) parts.add(rating)
        if (lines.isNotEmpty()) {
            val score = fromPlayer(lines[0].expectedScore, after.turn)
            parts.add("Standing: ${standingWords(score, thirdPerson = true)} (${format2(score)} from the player's side).")
            // One line read off the sheet's is that line, not the engine's only move there.
            parts.add(
                if (own != null) "The rest of the engine's line through it, as given above:" else "The engine's lines, best first:",
            )
            for ((i, line) in lines.withIndex()) {
                val shown = line.moves.take(LINE_PLIES)
                var text = "  ${i + 1}. ${sanLine(after, shown)} (${format2(fromPlayer(line.expectedScore, after.turn))})"
                if (i == 0) {
                    val notes = lineNotes(after, shown, player = player)
                    if (notes.isNotEmpty()) text += " -- move by move: $notes"
                    walk(after, shown)
                }
                parts.add(text)
            }
        }
        parts.add("On the board there (complete lists):")
        for (line in inventoryLines(after, player)) parts.add("  $line")
        return parts.joinToString("\n")
    }

    private fun square(start: Board, moves: List<String>, name: String, where: String): String {
        val cleaned = name.trim().lowercase()
        if (!SQUARE_NAME.matches(cleaned)) return "${pyRepr(name)} is not a square: give one such as g1."
        val square = parseSquare(cleaned)
        val (after, played, illegal) = play(start, moves)
        var head = heading(start, played, where)
        if (illegal.isNotEmpty()) head += " $illegal is not a legal move there, so this is the position before it."
        val piece = after.pieceAt(square)
        val what = if (piece != null) "${owner(piece.colour)} ${PIECE_NAMES[piece.pieceType]}" else "nothing"
        val parts = arrayListOf("$head On ${squareName(square)}: $what.")
        for (colour in listOf(player, !player)) {
            val attackers = after.attackers(colour, square).map { at ->
                PIECE_NAMES[after.pieceTypeAt(at)] + " on ${squareName(at)}" +
                    if (after.pinMask(colour, at) and BB_SQUARES[square] == BB_EMPTY) " (pinned)" else ""
            }
            val role = if (piece != null && piece.colour == colour) "Defended" else "Attacked"
            parts.add("$role by ${owner(colour)}: " + (if (attackers.isNotEmpty()) attackers.joinToString(", ") else "nothing") + ".")
        }
        if (piece == null || piece.colour != after.turn) {
            val there = whoCanMoveThere(after, square, player)
            parts.add(there.replaceFirstChar { it.uppercaseChar() } + ".")
        }
        return parts.joinToString(" ")
    }

    // ---- the board --------------------------------------------------------------------------

    /** The board after [moves], the moves played, and the first one that could not be. */
    private fun play(start: Board, moves: List<String>): Triple<Board, List<Move>, String> {
        val board = start.copy()
        val played = ArrayList<Move>()
        for (token in moves) {
            val move = board.parseSan(token)
                ?: Move.fromUciOrNull(token)?.takeIf { board.isLegal(it) }
                ?: return Triple(board, played, token)
            played.add(move)
            board.push(move)
            frames.add(board.copy(withStack = false))
        }
        return Triple(board, played, "")
    }

    /** The boards along a line the model is shown, for the draft check. */
    private fun walk(board: Board, ucis: List<String>) {
        val replay = board.copy(withStack = false)
        for (uci in ucis) {
            val move = Move.fromUciOrNull(uci) ?: return
            if (!replay.isLegal(move)) return
            replay.push(move)
            frames.add(replay.copy(withStack = false))
        }
    }

    /**
     * How the first move rates against the engine's best from [start], said as the first move's:
     * "Kg2 is the engine's best move there", printed after the line's end, was read as the best
     * move there ("White's best is 8. Kg2", Tai's puzzle). A move one of [start]'s lines plays is
     * rated by that line's score, as the model was shown it; another by a search after it.
     */
    private fun rating(start: Board, first: Move, origin: String, known: List<Line>): String {
        // The lines shown, when they are the full depth's: the first answer (the hint's, at
        // EARLY_DEPTH) set against a reply searched to the full depth reads a move as good as
        // the best as worse by what the depths differ by (the audit of 2026-10-05, round 2).
        val best = if (atFullDepth(known)) known.toList() else engine.analyse(start, depth)
        if (best.isEmpty()) return ""
        val after = start.copy()
        after.push(first)
        val head = "The first move given, ${sanLine(start, listOf(first.uci()))}, is"
        if (best[0].moves.isNotEmpty() && best[0].moves[0] == first.uci()) return "$head the engine's best move from $origin."
        if (gameOver(after)) return ""
        val own = best.firstOrNull { it.moves.take(1) == listOf(first.uci()) }
        val score = if (own != null) {
            own.expectedScore
        } else {
            // At the best line's depth: set against a shallower search, a move as good as the
            // best read as worse by what the depths differ by.
            val reply = engine.best(after, best[0].depth.takeIf { it != 0 } ?: depth)
            if (reply.isEmpty()) return ""
            1.0 - reply[0].expectedScore
        }
        val gap = best[0].expectedScore - score
        val top = if (best[0].moves.isNotEmpty()) sanLine(start, best[0].moves.take(1)) else "?"
        if (gap <= 0.005) return "$head as good as the engine's best from $origin ($top)."
        return "$head ${format2(gap)} behind the engine's best from $origin, $top."
    }

    private fun heading(start: Board, played: List<Move>, where: String): String {
        if (played.isEmpty()) return "From ${origin(where)}:"
        return "After ${sanLine(start, played.map { it.uci() })} (from ${origin(where)}):"
    }

    private fun fromPlayer(expected: Double, toMove: Boolean): Double = if (toMove == player) expected else 1.0 - expected

    private fun owner(colour: Boolean): String = if (colour == player) "the player's" else "the opponent's"

    private companion object {
        /** The moves as tokens: one move each, their numbers taken off ("4. Kg2" -> "Kg2"). */
        fun tokens(moves: Any?): List<String> {
            val items: List<Any?> = if (moves is List<*>) moves else listOf(moves)
            val out = ArrayList<String>()
            for (item in items) {
                for (raw in pyStr(item).replace(",", " ").split(Regex("\\s+"))) {
                    val token = NUMBER.replace(raw.trim(), "")
                    if (token.isNotEmpty() && token != "..." && token != "..") {
                        out.add(token.replace("0-0-0", "O-O-O").replace("0-0", "O-O"))
                    }
                }
            }
            return out
        }

        fun origin(where: String): String =
            if (where == "before_last_move") "the position before the last move" else "the board now"

        fun side(colour: Boolean): String = if (colour == WHITE) "White" else "Black"

        fun result(board: Board): String = when {
            board.isCheckmate() -> "checkmate, ${side(!board.turn)} wins"
            board.isStalemate() -> "stalemate, a draw"
            else -> "a draw"
        }

        fun format2(value: Double): String = formatted(value, 2)
    }
}
