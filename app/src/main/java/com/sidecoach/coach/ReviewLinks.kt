package com.sidecoach.coach

import com.sidecoach.chess.BLACK
import com.sidecoach.chess.Board
import com.sidecoach.chess.FILE_NAMES
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.square
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareName
import com.sidecoach.chess.squareRank
import com.sidecoach.events.ReviewedGame
import org.json.JSONArray
import org.json.JSONObject

/**
 * The AI coach's words as the review page draws them (docs/game-review-plan.md, decision 3):
 * each move, piece, square, file and chess term in a paragraph a link to the board (Tai,
 * 2026-09-30: "hovering or tapping/clicking on the highlights must also interact with the
 * board"). `coach/review_links.py`.
 *
 * A move draws its arrow, coloured by whose it is -- the better move's line gold, the student's
 * blue, their opponent's red -- and lights its two squares, read in the positions the paragraph
 * can mean: before the move, after it, along the reply's line and the better line, and either
 * side's threat (a move for the side not to move). A piece on a square or a bare square lights
 * its square; a file spans it end to end (a double-ended arrow, as chess notation marks one); a
 * coordinate that is a move of those lines ("the move c4") is the move. A term carries its
 * meaning, and what it names on the board: the pawns of a pawn structure ("your doubled
 * c-pawns", read in the position after the move), the file it names ("the half-open d-file"),
 * else the squares of the move's own motif of that name when the review tagged one ("the fork"
 * -> the fork's squares). The paragraph comes back as its pieces: plain text as it is, a link as
 * an object the page draws -- JSON, as the page reads it.
 */
object ReviewLinks {
    const val LINE_PLIES = 6

    /** A term, its meaning, and the review's motif tags it names (their squares lit). */
    val TERMS: LinkedHashMap<String, Pair<String, List<String>>> = linkedMapOf(
        "fork" to ("One piece attacks two or more at once; only one can be saved." to listOf("fork")),
        "pin" to (
            "A piece that cannot move without exposing a more valuable piece behind it." to listOf("pin")
            ),
        "skewer" to (
            "A valuable piece attacked in a line: when it moves, the piece behind it falls." to listOf("skewer")
            ),
        "discovered attack" to (
            "Moving one piece uncovers an attack by another behind it." to
                listOf("discoveredAttack", "discoveredCheck")
            ),
        "double attack" to ("Two threats at once; only one can be met." to listOf("fork", "doubleCheck")),
        "back rank" to (
            "The king's first rank: a king shut in by its own pawns can be mated there." to listOf("backRankMate")
            ),
        "sacrifice" to (
            "Material given up on purpose for something worth more: an attack, a mate, a won ending." to
                listOf("sacrifice")
            ),
        "passed pawn" to ("A pawn no enemy pawn can stop on its way to promotion." to listOf("advancedPawn")),
        "isolated pawn" to (
            "A pawn with no pawn of its side on the files beside it: only pieces can defend it." to emptyList()
            ),
        "doubled pawns" to ("Two pawns of one side on one file: they cannot protect each other." to emptyList()),
        "backward pawn" to (
            "A pawn left behind its neighbours, which no pawn can protect; the square in front of " +
                "it is a hole." to emptyList()
            ),
        "hanging pawns" to (
            "Two pawns side by side with no pawns beside them: strong while they can advance, " +
                "targets once stopped." to emptyList()
            ),
        "hanging" to ("Attacked and not defended: it can be taken for free." to listOf("hangingPiece")),
        "trapped" to ("A piece with no safe square left: it will be won." to listOf("trappedPiece")),
        "outpost" to ("A square no enemy pawn can attack: a home for a knight." to emptyList()),
        "open file" to ("A file with no pawns on it: a road for the rooks." to emptyList()),
        "half-open file" to ("A file with only the opponent's pawns on it: pressure for the rooks." to emptyList()),
        "bishop pair" to ("Both bishops against a bishop and a knight, or two knights." to emptyList()),
        "bad bishop" to ("A bishop hemmed in by its own pawns on its colour." to emptyList()),
        "initiative" to (
            "Making the threats: the other side has to answer instead of carrying out its plans." to emptyList()
            ),
        "tempo" to ("A move's worth of time." to emptyList()),
        "zugzwang" to ("Every move makes things worse, and a move must be made." to listOf("zugzwang")),
        "prophylaxis" to ("A move that stops the opponent's plan before it starts." to emptyList()),
        "fianchetto" to ("A bishop developed on the long diagonal, from b2, g2, b7 or g7." to emptyList()),
        "king safety" to ("How well the king is sheltered from checks and attacks." to listOf("exposedKing")),
        "pawn break" to ("A pawn advance that challenges the opponent's pawns and opens lines." to emptyList()),
        "pawn majority" to ("More pawns than the opponent on one wing: a passed pawn in the making." to emptyList()),
        "promotion" to (
            "A pawn reaching the last rank becomes a queen, or another piece." to listOf("promotion")
            ),
        "deflection" to ("Forcing a defender away from what it guards." to listOf("deflection")),
        "decoy" to ("Luring a piece onto a square where it can be hit." to listOf("attraction")),
        "overloaded" to ("A piece with more jobs to do than it can do at once." to listOf("overloading")),
        "interference" to ("A piece put between an enemy piece and what it defends." to listOf("interference")),
        "in-between move" to (
            "An unexpected move played before the expected reply." to listOf("intermezzo")
            ),
        "quiet move" to ("A move with no check or capture that still carries a threat." to listOf("quietMove")),
        "counterplay" to ("Threats of one's own that make the opponent defend." to emptyList()),
        "closed centre" to ("Centre pawns locked against each other: the play moves to the wings." to emptyList()),
        "opposition" to ("Kings facing each other one square apart: the side not to move has it." to emptyList()),
        "weak square" to ("A square no pawn can defend any more: a post for the opponent's pieces." to emptyList()),
        "the exchange" to ("A rook for a bishop or a knight." to emptyList()),
    )

    /** Each term's spellings: plurals, verbs, the American centre. */
    private val SPELLINGS = mapOf(
        "fork" to "fork(?:s|ed|ing)?",
        "pin" to "pin(?:s|ned|ning)?",
        "skewer" to "skewer(?:s|ed|ing)?",
        "discovered attack" to "discovered (?:attack|check)s?",
        "double attack" to "double attacks?",
        "back rank" to "back[- ]rank",
        "sacrifice" to "sacrific(?:e|es|ed|ing)",
        "passed pawn" to "passed (?:pawns?|[a-h]-pawns?)|passers?",
        "isolated pawn" to "isolated (?:pawns?|[a-h]-pawns?)",
        "doubled pawns" to "doubled (?:pawns|[a-h]-pawns)",
        "backward pawn" to "backward (?:pawns?|[a-h]-pawns?)",
        "hanging pawns" to "hanging pawns",
        "hanging" to "hanging|loose",
        "trapped" to "trapped",
        "outpost" to "outposts?",
        "open file" to "open (?:[a-h]-)?files?",
        "half-open file" to "(?:half|semi)-open (?:[a-h]-)?files?",
        "bishop pair" to "bishop pair|pair of bishops",
        "bad bishop" to "bad bishops?",
        "initiative" to "initiative",
        "tempo" to "tempo|tempi",
        "zugzwang" to "zugzwang",
        "prophylaxis" to "prophyla(?:xis|ctic)",
        "fianchetto" to "fianchett(?:o|oed)",
        "king safety" to "king safety",
        "pawn break" to "pawn breaks?",
        "pawn majority" to "pawn majority|(?:queenside|kingside) majority",
        "promotion" to "promot(?:ion|e|es|ed|ing)|queening",
        "deflection" to "deflect(?:ion|s|ed|ing)?",
        "decoy" to "decoys?",
        "overloaded" to "overload(?:ed|ing)?",
        "interference" to "interference",
        "in-between move" to "in-between moves?|zwischenzug|intermezzo",
        "quiet move" to "quiet moves?",
        "counterplay" to "counterplay",
        "closed centre" to "closed cent(?:re|er)",
        "opposition" to "the opposition",
        "weak square" to "weak squares?|holes?",
        "the exchange" to "the exchange",
    )
    private val TERM_NAMES = TERMS.keys.toList()

    // The terms longest spelling first (a stable sort, as Python's), each its own group "t<i>".
    private val TERM_ORDER = TERM_NAMES.indices.sortedBy { -SPELLINGS.getValue(TERM_NAMES[it]).length }
    private val TERM = Regex(
        "\\b(?:" + TERM_ORDER.joinToString("|") { "(?<t$it>${SPELLINGS.getValue(TERM_NAMES[it])})" } + ")\\b",
        RegexOption.IGNORE_CASE,
    )

    private const val PIECE =
        "\\b(?:(?:your opponent's|your|their|the|White's|Black's|White|Black)\\s+)?" +
            "(?:(?:isolated|passed|backward|doubled|loose|hanging|pinned|trapped|advanced|extra|lone|" +
            "last|remaining|bad|good|light-squared|dark-squared|active|passive|strong|weak|exposed)\\s+)" +
            "{0,2}(?:[a-h]-)?\\b(?:king|queen|rook|bishop|knight|pawn)s?\\s+(?:on|at)\\s+(?<sq>[a-h][1-8])\\b"

    // Moves, squares and files in their own case: "b" is a file, "B" a bishop. `[\w-]` written
    // `[-\w]`, its hyphen first, as every regex syntax reads it.
    private const val SAN =
        "(?-i:(?<![-\\w])(?<san>O-O-O|O-O|[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?" +
            "|[a-h]x[a-h][1-8](?:=[QRBN])?[+#]?|[a-h][18]=[QRBN][+#]?)(?![-\\w]))"
    private const val COORD = "(?-i:(?<![-\\w])(?<coord>[a-h][1-8])(?![-\\w]))"
    private const val FILE = "(?-i:\\b(?<file>[a-h])-file\\b)"
    private val LINKS = Regex("(?<piece>$PIECE)|$SAN|$FILE|$COORD", RegexOption.IGNORE_CASE)
    private val ON_FILE = Regex("\\b([a-h])-(?:file|pawn)")

    /** A move's colour by whose it is: the student's blue, their opponent's red. `_side`. */
    private fun side(board: Board, user: String?): String {
        val side = if (board.turn == WHITE) "white" else "black"
        return if (user == null || side == user) "idea" else "threat"
    }

    /** A move's link but its text: its kind and its arrow [from, to, colour]. `_arrow`. */
    private class Arrow(val from: String, val to: String, val role: String)

    private fun arrow(move: Move, role: String) = Arrow(squareName(move.from), squareName(move.to), role)

    private fun moveLink(text: String, arrow: Arrow): JSONObject = JSONObject()
        .put("text", text)
        .put("kind", "move")
        .put("arrow", JSONArray().put(arrow.from).put(arrow.to).put(arrow.role))

    // The pawn structures a term names, and whose pawns it means by the word before it.
    private val STRUCTURES = setOf("passed pawn", "isolated pawn", "doubled pawns", "backward pawn", "hanging pawns")
    private val OWNER = Regex("(your opponent's|your|their|White's|Black's)\\s+(?:\\w+\\s+)?\$", RegexOption.IGNORE_CASE)

    /**
     * Whose pawns a term means, by the word before it ("your doubled pawns"); both sides' when it
     * names no one. `_owners`.
     */
    internal fun owners(before: String, user: String?): List<Boolean> {
        val found = OWNER.find(before)
        val word = found?.groupValues?.get(1)?.lowercase() ?: ""
        val us = if (user == null) null else user == "white"
        if (word.startsWith("white") || word.startsWith("black")) return listOf(word.startsWith("white"))
        if (us != null && word == "your") return listOf(us)
        if (us != null && (word == "your opponent's" || word == "their")) return listOf(!us)
        return listOf(WHITE, BLACK)
    }

    /**
     * The pawns of [colour] in a structure: doubled (another of theirs on the file), isolated
     * (none of theirs beside it), passed (no enemy pawn ahead on its file or beside it); a
     * backward or hanging pawn only by the file named. `_pawns`.
     */
    internal fun pawns(board: Board, colour: Boolean, structure: String, file: Int?): Set<Int> {
        val own = board.pieces(PAWN, colour)
        val enemy = board.pieces(PAWN, !colour).toSet()
        val files = own.map { squareFile(it) }
        val out = LinkedHashSet<Int>()
        for (sq in own) {
            val f = squareFile(sq)
            val rank = squareRank(sq)
            if (file != null && f != file) continue
            val ahead = if (colour == WHITE) (rank + 1 until 8) else (0 until rank)
            val fits = when (structure) {
                "doubled pawns" -> files.count { it == f } > 1
                "isolated pawn" -> files.none { Math.abs(it - f) == 1 }
                "passed pawn" -> (f - 1..f + 1).filter { it in 0..7 }.none { g -> ahead.any { r -> square(g, r) in enemy } }
                else -> file != null
            }
            if (fits) out.add(sq)
        }
        return out
    }

    /**
     * The moves a paragraph on move [ply] names, by SAN (checks left off): the move, the better
     * move's line (gold), the reply's line; the first naming wins. `_moves`.
     */
    private fun moves(game: ReviewedGame, ply: Int, user: String?): Map<String, Arrow> {
        val move = game.moves[ply - 1]
        val before = Board(move.fenBefore)
        val out = LinkedHashMap<String, Arrow>()

        fun walk(start: Board, line: List<String>, role: String?) {
            for ((board, played) in along(start, line)) {
                val san = board.san(played).trimEnd('+', '#')
                if (san !in out) out[san] = arrow(played, role ?: side(board, user))
            }
        }

        walk(before, listOf(move.assessment.uci), null)
        walk(before, move.assessment.best.take(LINE_PLIES), "best")
        val reply = if (move.after.lines.isNotEmpty()) move.after.lines[0].moves.take(LINE_PLIES) else emptyList()
        walk(Board(move.after.fen), reply, null)
        return out
    }

    /** Each move of a line with the position it is played from, as far as it stays legal. `_along`. */
    private fun along(start: Board, line: List<String>): List<Pair<Board, Move>> {
        val board = start.copy(withStack = false)
        val out = ArrayList<Pair<Board, Move>>()
        for (uci in line) {
            val played = Move.fromUci(uci)
            if (!board.isLegal(played)) break
            out.add(board.copy(withStack = false) to played)
            board.push(played)
        }
        return out
    }

    /**
     * The positions a move not of those lines is read in: before the move, after it, along both
     * lines, and in each of them the side not to move's threat ("Nd2 threatens Nxe4"). `_elsewhere`.
     */
    private fun elsewhere(game: ReviewedGame, ply: Int): List<Board> {
        val move = game.moves[ply - 1]
        val boards = arrayListOf(Board(move.fenBefore), Board(move.after.fen))
        val reply = if (move.after.lines.isNotEmpty()) move.after.lines[0].moves.take(LINE_PLIES) else emptyList()
        for ((start, line) in listOf(boards[0] to move.assessment.best.take(LINE_PLIES), boards[1] to reply)) {
            for ((board, played) in along(start, line)) {
                val after = board.copy(withStack = false)
                after.push(played)
                boards.add(after)
            }
        }
        for (board in boards.toList()) {
            if (!board.isCheck()) {
                val passed = board.copy(withStack = false)
                passed.push(Move.NULL)
                boards.add(passed)
            }
        }
        return boards
    }

    /**
     * A move's link: its arrow and colour, from the game's lines or the first position it is
     * legal in; null when it is legal nowhere the paragraph can mean. `_move`.
     */
    private fun move(san: String, named: Map<String, Arrow>, elsewhere: List<Board>, user: String?): JSONObject? {
        val known = named[san.trimEnd('+', '#')]
        if (known != null) return moveLink(san, known)
        for (board in elsewhere) {
            val move = board.parseSan(san) ?: continue
            return moveLink(san, arrow(move, side(board, user)))
        }
        return null
    }

    /**
     * [text] (a paragraph in the student's words, on move [ply]) as its pieces: plain text as it
     * is, each link an object -- "text", "kind" (move, piece, square, file, term) and what it does
     * on the board ("squares" lit, a move's "arrow" [from, to, colour], a file's "span" [end,
     * end]; a term's "meaning"). A [general] paragraph (the lesson) lights nothing of this game.
     * `links`.
     */
    fun links(text: String, game: ReviewedGame, ply: Int, user: String?, general: Boolean = false): JSONArray {
        val named = moves(game, ply, user)
        val elsewhere = elsewhere(game, ply)
        val tags = if (general) emptyList() else game.moves[ply - 1].assessment.tags
        val found = ArrayList<Triple<Int, Int, JSONObject>>()
        for (match in LINKS.findAll(text)) {
            val piece = match.groups["piece"]
            val san = match.groups["san"]
            val file = match.groups["file"]
            val coord = match.groups["coord"]
            val link: JSONObject? = when {
                piece != null && piece.value.isNotEmpty() -> JSONObject()
                    .put("text", match.value)
                    .put("kind", "piece")
                    .put("squares", JSONArray().put(match.groups["sq"]!!.value))
                san != null && san.value.isNotEmpty() -> move(san.value, named, elsewhere, user)
                file != null && file.value.isNotEmpty() -> {
                    val letter = file.value.lowercase()
                    JSONObject().put("text", match.value).put("kind", "file")
                        .put("span", JSONArray().put("${letter}1").put("${letter}8"))
                }
                coord != null && coord.value.isNotEmpty() -> {
                    val at = coord.value
                    val moved = if (at in named) move(at, named, emptyList(), user) else null
                    moved ?: JSONObject().put("text", at).put("kind", "square")
                        .put("squares", JSONArray().put(at.lowercase()))
                }
                else -> null
            }
            if (link != null) found.add(Triple(match.range.first, match.range.last + 1, link))
        }
        val move = game.moves[ply - 1]
        val shown = listOf(Board(move.after.fen), Board(move.fenBefore))
        for (match in TERM.findAll(text)) {
            val name = TERM_NAMES[TERM_ORDER.first { !match.groups["t$it"]?.value.isNullOrEmpty() }]
            val (meaning, motifs) = TERMS.getValue(name)
            val link = JSONObject().put("text", match.value).put("kind", "term").put("meaning", meaning)
            val onFile = ON_FILE.find(match.value)
            if (onFile != null && "file" in name) {
                // A file the term names: "the half-open d-file".
                val letter = onFile.groupValues[1]
                link.put("span", JSONArray().put("${letter}1").put("${letter}8"))
            } else if (name in STRUCTURES && !general) {
                // Its pawns as the board shows them (after the move), else as they stood before.
                val fileIndex = if (onFile != null) FILE_NAMES.indexOf(onFile.groupValues[1][0]) else null
                val whose = owners(text.substring(0, match.range.first), user)
                for (board in shown) {
                    val pawns = whose.flatMap { pawns(board, it, name, fileIndex) }.toSet()
                    if (pawns.isNotEmpty()) {
                        link.put("squares", JSONArray(pawns.map { squareName(it) }.sorted()))
                        break
                    }
                }
            } else {
                val squares = tags.filter { it.name in motifs }.flatMap { it.squares }.toSet().sorted()
                if (squares.isNotEmpty()) link.put("squares", JSONArray(squares))
            }
            found.add(Triple(match.range.first, match.range.last + 1, link))
        }
        // The earliest link wins where two overlap ("the passed pawn on e5" is a piece, not a term).
        val out = JSONArray()
        var at = 0
        for ((start, end, link) in found.sortedWith(compareBy<Triple<Int, Int, JSONObject>>({ it.first }, { -it.second }))) {
            if (start < at) continue
            if (start > at) out.put(text.substring(at, start))
            out.put(link)
            at = end
        }
        if (at < text.length) out.put(text.substring(at))
        return out
    }

    /**
     * Every paragraph of the saved words (by ply, then part: the "texts" of the saved words) as
     * its links. `linked`.
     */
    fun linked(texts: JSONObject, game: ReviewedGame, user: String?): JSONObject {
        val out = JSONObject()
        for (ply in texts.keys()) {
            val parts = texts.getJSONObject(ply)
            val linkedParts = JSONObject()
            for (part in parts.keys()) {
                linkedParts.put(part, links(parts.getString(part), game, ply.toInt(), user, general = part == "lesson"))
            }
            out.put(ply, linkedParts)
        }
        return out
    }
}
