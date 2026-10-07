package com.sidecoach.chess

/**
 * PGN, read and written: a port of the part of `python-chess`'s `chess.pgn` that the game
 * library uses (profile/library.py on the desktop) -- `read_game` with its `GameBuilder`, the
 * game tree (the main line and any side lines, comments, NAGs), `Headers` with the Seven Tag
 * Roster first, a node's `clock()`, and `str(game)` (the `StringExporter` on one line).
 *
 * Literal where it matters: the same tokens (`MOVETEXT_REGEX`), the same handling of a comment
 * that runs over lines, of a move that cannot be played (the game's `errors`, the rest of that
 * line skipped), of a result token, of the headers' order -- so a game read here is the game the
 * desktop reads, and written out is the same text (the library keys a pasted game by it and
 * keeps the text for the review).
 *
 * Variants are not ported: a game whose `Variant` tag names one (anything but standard chess
 * or its aliases, and Chess960) is read as errored, where python-chess would read it on its own
 * board. The library leaves variants out anyway (a Lichess import skips them).
 */

private val TAG_REGEX = Regex("^\\[([A-Za-z0-9][A-Za-z0-9_+#=:-]*)\\s+\"([^\\r]*)\"\\]\\s*$")
private val TAG_NAME_REGEX = Regex("^[A-Za-z0-9][A-Za-z0-9_+#=:-]*$")
private val MOVETEXT_REGEX = Regex(
    "(" +
        "[NBKRQ]?[a-h]?[1-8]?[\\-x]?[a-h][1-8](?:=?[nbrqkNBRQK])?" +
        "|[PNBRQK]?@[a-h][1-8]" +
        "|--" +
        "|Z0" +
        "|0000" +
        "|@@@@" +
        "|O-O(?:-O)?" +
        "|0-0(?:-0)?" +
        ")" +
        "|(\\{.*)" +
        "|(;.*)" +
        "|(\\$[0-9]+)" +
        "|(\\()" +
        "|(\\))" +
        "|(\\*|1-0|0-1|1/2-1/2)" +
        "|([\\?!]{1,2})",
    RegexOption.DOT_MATCHES_ALL,
)
private val CLOCK_REGEX = Regex("(\\s?)\\[%clk\\s(\\d+):(\\d+):(\\d+(?:\\.\\d*)?)\\](\\s?)")

val TAG_ROSTER = listOf("Event", "Site", "Date", "Round", "White", "Black", "Result")
private val ROSTER_DEFAULTS = linkedMapOf(
    "Event" to "?", "Site" to "?", "Date" to "????.??.??", "Round" to "?",
    "White" to "?", "Black" to "?", "Result" to "*",
)
private val STANDARD_VARIANTS = setOf("standard", "chess", "classical", "normal", "illegal", "from position")
private val CHESS960 = setOf("chess960", "chess 960", "fischerandom", "fischerrandom", "fischer random")

private const val NAG_GOOD_MOVE = 1
private const val NAG_MISTAKE = 2
private const val NAG_BRILLIANT_MOVE = 3
private const val NAG_BLUNDER = 4
private const val NAG_SPECULATIVE_MOVE = 5
private const val NAG_DUBIOUS_MOVE = 6

/** `chess.pgn.Headers`: the Seven Tag Roster in its order, then the rest as they came. */
class PgnHeaders(defaults: Boolean = true) {
    private val roster = linkedMapOf<String, String>()
    private val others = linkedMapOf<String, String>()

    init {
        if (defaults) for ((k, v) in ROSTER_DEFAULTS) roster[k] = v
    }

    operator fun set(key: String, value: String) {
        if (key in TAG_ROSTER) roster[key] = value
        else {
            require(TAG_NAME_REGEX.matches(key)) { "invalid pgn header tag: '$key'" }
            require('\n' !in value && '\r' !in value) { "line break in pgn header $key" }
            others[key] = value
        }
    }

    operator fun get(key: String): String? = if (key in TAG_ROSTER) roster[key] else others[key]

    fun getOrDefault(key: String, default: String): String = get(key) ?: default

    operator fun contains(key: String): Boolean = get(key) != null

    fun remove(key: String) {
        if (key in TAG_ROSTER) roster.remove(key) else others.remove(key)
    }

    /** In python-chess's order: the roster's tags present, then the others. */
    fun items(): List<Pair<String, String>> =
        TAG_ROSTER.mapNotNull { k -> roster[k]?.let { k to it } } + others.map { it.key to it.value }

    fun isChess960(): Boolean = (get("Variant") ?: "").lowercase() in CHESS960

    /** Whether the game is standard chess (or Chess960, played on the same board). */
    fun isStandard(): Boolean {
        val variant = get("Variant") ?: return true
        return isChess960() || variant.lowercase() in STANDARD_VARIANTS
    }

    fun board(): Board = Board(get("FEN") ?: STARTING_FEN)
}

/** A node of the game tree: the root ([PgnGame]) or a move ([PgnChild]). */
abstract class PgnNode {
    val variations = mutableListOf<PgnChild>()
    var comment: String = ""
    var startingComment: String = ""
    val nags = sortedSetOf<Int>()
    abstract val parent: PgnNode?

    fun addVariation(move: Move, startingComment: String = ""): PgnChild {
        val child = PgnChild(this, move)
        child.startingComment = startingComment
        variations.add(child)
        return child
    }

    fun isEnd(): Boolean = variations.isEmpty()

    /** `clock()`: the first `[%clk h:mm:ss]` of this node's comment, in seconds. */
    fun clock(): Double? {
        val m = CLOCK_REGEX.find(comment) ?: return null
        return m.groupValues[2].toInt() * 3600 + m.groupValues[3].toInt() * 60 + m.groupValues[4].toDouble()
    }

    abstract fun game(): PgnGame
}

class PgnChild(override val parent: PgnNode, val move: Move) : PgnNode() {
    override fun game(): PgnGame = parent.game()
}

/** `chess.pgn.Game`: the headers, the root of the moves, what went wrong reading it. */
class PgnGame(val headers: PgnHeaders = PgnHeaders()) : PgnNode() {
    override val parent: PgnNode? = null
    val errors = mutableListOf<String>()

    override fun game(): PgnGame = this

    fun board(): Board = headers.board()

    /** The main line's moves, in order. */
    fun mainlineMoves(): List<Move> = mainline().map { it.move }

    /** The main line's nodes, in order (a node's comment holds its clock). */
    fun mainline(): List<PgnChild> {
        val out = mutableListOf<PgnChild>()
        var node: PgnNode = this
        while (node.variations.isNotEmpty()) {
            node = node.variations[0]
            out.add(node as PgnChild)
        }
        return out
    }

    /** `Game.setup(board)`: a starting position other than the usual in the FEN tag. */
    fun setup(board: Board) {
        val fen = board.fen()
        if (fen == STARTING_FEN) {
            headers.remove("FEN")
            headers.remove("SetUp")
        } else {
            headers["FEN"] = fen
            headers["SetUp"] = "1"
        }
        headers.remove("Variant")
    }

    /** `str(game)`: the headers, a blank line, the movetext on one line. */
    override fun toString(): String = Exporter().game(this)
}

/** `StringExporter(columns=None)`. */
private class Exporter {
    private val lines = mutableListOf<String>()
    private var current = StringBuilder()
    private var forceMoveNumber = true
    private var foundHeaders = false

    private fun flush() {
        if (current.isNotEmpty()) lines.add(current.toString().trimEnd())
        current = StringBuilder()
    }

    private fun token(t: String) {
        current.append(t)
    }

    private fun line(l: String = "") {
        flush()
        lines.add(l.trimEnd())
    }

    private fun comment(text: String) {
        token("{ " + text.replace("}", "").trim() + " } ")
        forceMoveNumber = true
    }

    private fun nag(n: Int) {
        token("$$n ")
    }

    private fun move(board: Board, move: Move) {
        if (board.turn == WHITE) token("${board.fullmoveNumber}. ")
        else if (forceMoveNumber) token("${board.fullmoveNumber}... ")
        token(board.san(move) + " ")
        forceMoveNumber = false
    }

    private fun acceptNode(node: PgnChild, board: Board) {
        if (node.startingComment.isNotEmpty()) comment(node.startingComment)
        move(board, node.move)
        for (n in node.nags) nag(n)
        if (node.comment.isNotEmpty()) comment(node.comment)
    }

    /** `ChildNode._accept`: the node, its side lines each in brackets, then the line on. */
    private fun accept(start: PgnChild, board: Board) {
        acceptNode(start, board)
        val parent = start.parent
        // The alternatives to this move: the parent's other variations.
        for (side in parent.variations.drop(1).takeIf { parent.variations.firstOrNull() === start } ?: emptyList()) {
            token("( ")
            forceMoveNumber = true
            accept(side, board)
            token(") ")
            forceMoveNumber = true
        }
        if (start.variations.isNotEmpty()) {
            board.push(start.move)
            accept(start.variations[0], board)
            board.pop()
        }
    }

    fun game(game: PgnGame): String {
        foundHeaders = false
        for ((k, v) in game.headers.items()) {
            foundHeaders = true
            line("[$k \"$v\"]")
        }
        if (foundHeaders) line()
        val board = game.board()
        if (game.comment.isNotEmpty()) comment(game.comment)
        if (game.variations.isNotEmpty()) accept(game.variations[0], board)
        token(game.headers.getOrDefault("Result", "*") + " ")
        line()
        return if (current.isNotEmpty()) (lines + current.toString().trimEnd()).joinToString("\n").trimEnd()
        else lines.joinToString("\n").trimEnd()
    }
}

/**
 * `chess.pgn.read_game` over the lines of a text: the next game, or null at the end. [reader]
 * keeps its place between calls, as a file handle does.
 */
class PgnReader(text: String) {
    private val lines: List<String>
    private var index = 0

    init {
        // Lines as `readline` gives them: each with its newline, the last one maybe without.
        val out = mutableListOf<String>()
        var start = 0
        for (i in text.indices) {
            if (text[i] == '\n') {
                out.add(text.substring(start, i + 1))
                start = i + 1
            }
        }
        if (start < text.length) out.add(text.substring(start))
        lines = out
    }

    private fun readline(): String = if (index < lines.size) lines[index++] else ""

    private fun String.isSpace(): Boolean = isNotEmpty() && all { it.isWhitespace() }

    fun readGame(): PgnGame? {
        var game: PgnGame? = null
        val variationStack = mutableListOf<PgnNode>()
        var startingComment = ""
        var inVariation = false

        var foundGame = false
        var line = readline().trimStart('﻿')
        while (line.isSpace() || line.startsWith("%") || line.startsWith(";")) line = readline()

        var consecutiveEmpty = 0
        while (line.isNotEmpty()) {
            if (line.startsWith("%") || line.startsWith(";")) {
                line = readline()
                continue
            }
            if (consecutiveEmpty < 1 && line.isSpace()) {
                consecutiveEmpty += 1
                line = readline()
                continue
            }
            if (!foundGame) {
                foundGame = true
                game = PgnGame()
                variationStack.add(game)
            }
            if (!line.startsWith("[")) break
            consecutiveEmpty = 0
            // `\s*$` takes the line's own newline, as `re.match` does on a `readline` line.
            val m = TAG_REGEX.find(line)
            if (m != null) runCatching { game!!.headers[m.groupValues[1]] = m.groupValues[2] }
            line = readline()
        }
        if (!foundGame || game == null) return null

        val boardStack = mutableListOf<Board>()
        if (!game.headers.isStandard()) {
            game.errors.add("variant not supported: ${game.headers["Variant"]}")
        }
        val start = runCatching { game.headers.board() }.getOrNull()
        if (start == null || game.errors.isNotEmpty()) {
            if (start == null) game.errors.add("invalid fen: ${game.headers["FEN"]}")
            skipGame(line)
            return game
        }
        boardStack.add(start)

        var skipDepth = 0
        var fresh = true
        while (line.isNotEmpty()) {
            if (fresh) {
                if (line.startsWith("%") || line.startsWith(";")) {
                    line = readline()
                    continue
                }
                if (line.isSpace()) return game
            }
            fresh = true
            var rest: String? = null
            for (match in MOVETEXT_REGEX.findAll(line)) {
                val token = match.value
                if (token.startsWith("{")) {
                    val startIndex = if (token.startsWith("{ ")) 2 else 1
                    var l = token.substring(startIndex)
                    val commentLines = mutableListOf<String>()
                    while (l.isNotEmpty() && '}' !in l) {
                        commentLines.add(l)
                        l = readline()
                    }
                    if (l.isNotEmpty()) {
                        val close = l.indexOf('}')
                        val end = if (close > 0 && l[close - 1] == ' ') close - 1 else close
                        commentLines.add(l.substring(0, end))
                        l = l.substring(close + 1)
                    }
                    if (skipDepth == 0) {
                        val text = commentLines.joinToString("")
                        val top = variationStack.last()
                        if (inVariation || (top.parent == null && top.isEnd())) {
                            top.comment = listOf(top.comment, text).filter { it.isNotEmpty() }.joinToString(" ")
                        } else {
                            startingComment = listOf(startingComment, text).filter { it.isNotEmpty() }.joinToString(" ")
                        }
                    }
                    rest = l
                    break
                } else if (token == "(") {
                    if (skipDepth > 0) skipDepth += 1
                    else if (boardStack.last().moveStack.isNotEmpty()) {
                        val parent = variationStack.last().parent!!
                        variationStack.add(parent)
                        inVariation = false
                        val b = boardStack.last().copy()
                        b.pop()
                        boardStack.add(b)
                    }
                } else if (token == ")") {
                    if (skipDepth == 1) {
                        skipDepth = 0
                        variationStack.removeAt(variationStack.lastIndex)
                    } else if (skipDepth > 0) skipDepth -= 1
                    else if (boardStack.size > 1) {
                        variationStack.removeAt(variationStack.lastIndex)
                        boardStack.removeAt(boardStack.lastIndex)
                    }
                } else if (skipDepth > 0) {
                    continue
                } else if (token.startsWith(";")) {
                    break
                } else if (token.startsWith("$")) {
                    variationStack.last().nags.add(token.substring(1).toInt())
                } else if (token in NAG_TOKENS) {
                    variationStack.last().nags.add(NAG_TOKENS.getValue(token))
                } else if (token in RESULTS && boardStack.size == 1) {
                    if (game.headers.getOrDefault("Result", "*") == "*") game.headers["Result"] = token
                } else {
                    val board = boardStack.last()
                    val move = board.parseSan(token)
                    if (move == null) {
                        game.errors.add("illegal or ambiguous san: $token in ${board.fen()}")
                        skipDepth = 1
                    } else {
                        val node = variationStack.last().addVariation(move, startingComment)
                        variationStack[variationStack.lastIndex] = node
                        startingComment = ""
                        inVariation = true
                        board.push(move)
                    }
                }
            }
            if (rest != null) {
                line = rest
                fresh = false
            } else {
                line = readline()
            }
        }
        return game
    }

    /** The rest of a game that is not read: to its blank line, comments minded. */
    private fun skipGame(first: String) {
        var line = first
        var inComment = false
        while (line.isNotEmpty()) {
            if (!inComment) {
                if (line.isSpace()) return
                if (line.startsWith("%")) {
                    line = readline()
                    continue
                }
            }
            for (ch in line) {
                if (ch == '{') inComment = true
                else if (!inComment && ch == ';') break
                else if (ch == '}') inComment = false
            }
            line = readline()
        }
    }

    companion object {
        private val RESULTS = setOf("1-0", "0-1", "1/2-1/2", "*")
        private val NAG_TOKENS = mapOf(
            "?" to NAG_MISTAKE, "??" to NAG_BLUNDER, "!" to NAG_GOOD_MOVE,
            "!!" to NAG_BRILLIANT_MOVE, "!?" to NAG_SPECULATIVE_MOVE, "?!" to NAG_DUBIOUS_MOVE,
        )

        /** Every game of a text, as `read_game` called until the end gives them. */
        fun games(text: String): List<PgnGame> {
            val reader = PgnReader(text)
            val out = mutableListOf<PgnGame>()
            while (true) out.add(reader.readGame() ?: return out)
        }
    }
}
