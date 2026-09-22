package com.sidecoach.coach

import com.sidecoach.analysis.PositionalFact
import com.sidecoach.analysis.bestMoveIdea
import com.sidecoach.analysis.phaseOf
import com.sidecoach.analysis.sanLine
import com.sidecoach.analysis.topFeatures
import com.sidecoach.analysis.worstPieceSquare
import com.sidecoach.chess.BISHOP
import com.sidecoach.chess.BLACK
import com.sidecoach.chess.Board
import com.sidecoach.chess.KING
import com.sidecoach.chess.KNIGHT
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.parseSquare
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareName
import com.sidecoach.chess.squareRank
import com.sidecoach.events.Analysis
import com.sidecoach.events.MotifTag

/**
 * From the analysis of a position to the facts a coach may draw on: a port of
 * `sidecoach/coach/facts.py`.
 *
 * The disclosure machine filters these by level, so what goes in here decides what can ever
 * come out. Each field is written in the words the coach would use: the theme is a category
 * ("a loose piece is involved"), the region is a part of the board with one anchor square,
 * the piece is named but never its destination, and the candidates and the line are moves in
 * the notation a person reads.
 */

/**
 * The best move's idea, as a square-free strategic theme the coach may teach at the theme
 * level: it orients toward the plan without naming the move. Keyed by the idea's kind.
 */
private val PLAN_THEME = mapOf(
    "castle" to "getting your king to safety",
    "rescue" to "getting a piece out of danger",
    "prophylaxis" to "stopping your opponent's plan before your own",
    "activateKing" to "activating your King in the endgame",
    "develop" to "finishing your development",
    "outpostMove" to "planting a knight on an outpost",
    "centralise" to "centralising a piece",
    "takeFile" to "taking the open file",
    "rookBehindPasser" to "getting your rook behind the passed pawn",
    "break" to "breaking open their pawn structure",
    "centre" to "fighting for the centre",
    "pawnStorm" to "storming the pawns in front of their King",
    "pushPasser" to "pushing your passed pawn",
    "activate" to "activating your worst-placed piece",
    "space" to "gaining space on the flank",
    "simplify" to "trading down to convert your material",
)

/**
 * Tactical tags a coach would name as the theme, most specific first. Everything that a move
 * merely is (capture, check, quietMove) says nothing about the idea and is left out.
 */
val THEMES: List<Pair<String, String>> = listOf(
    "backRankMate" to "a weak back rank",
    "smotheredMate" to "a smothered mate",
    "mateIn1" to "a mating attack",
    "mateIn2" to "a mating attack",
    "mateIn3" to "a mating attack",
    "mateIn4" to "a mating attack",
    "mateIn5" to "a mating attack",
    "mate" to "a mating attack",
    "fork" to "a fork",
    "absolutePin" to "a pin",
    "pin" to "a pin",
    "skewer" to "a skewer",
    "discoveredAttack" to "a discovered attack",
    "doubleCheck" to "a double check",
    "hangingPiece" to "a loose piece",
    "trappedPiece" to "a trapped piece",
    "capturingDefender" to "a defender that can be removed",
    "deflection" to "a defender that can be pulled away",
    "attraction" to "a piece that can be lured onto a bad square",
    "sacrifice" to "a sacrifice",
    // The best move saves a piece that is attacked: the hint a stuck player needs most.
    "defensiveMove" to "a piece under attack",
    // The tagger's exposed king is always the enemy's.
    "exposedKing" to "your opponent's exposed king",
    "attackingF2F7" to "the weak point next to the king",
    "promotion" to "a passed pawn",
    "advancedPawn" to "a passed pawn",
    "xRayAttack" to "a piece attacked through another",
    "interference" to "a line that can be cut",
    "clearance" to "a square that can be cleared",
)

private val THEME_RANK = THEMES.withIndex().associate { (i, pair) -> pair.first to i }

val COACH_PIECE_NAMES = arrayOf("", "pawn", "knight", "bishop", "rook", "queen", "king")

private val FACT_VALUES = intArrayOf(0, 1, 3, 3, 5, 9, 0)

/** Candidates within this much of the best line are "the near-equal set". */
const val CANDIDATE_MARGIN = 0.02

/**
 * An expected score from the player's side, in words: as the coach says it to the player
 * ("you are better"), or as the harness writes it ("the player is better").
 */
fun standingWords(score: Double, thirdPerson: Boolean = false): String {
    val who = if (thirdPerson) "the player is" else "you are"
    if (score >= 0.85) return "$who winning"
    if (score >= 0.62) return "$who better"
    if (score > 0.38) return "it is about equal"
    if (score > 0.15) return "$who worse"
    return "$who losing"
}

private val CAMEL_SPLIT = Regex("(?<=[a-z])(?=[A-Z0-9])")

/**
 * A motif tag's name in plain words: the theme table's phrase when it has one, else the
 * camelCase name split into words ("discoveredAttack" -> "discovered attack").
 */
fun motifWords(name: String): String {
    for ((tag, words) in THEMES) if (tag == name) return words
    return CAMEL_SPLIT.replace(name, " ").lowercase()
}

private val COUNTED = intArrayOf(QUEEN, ROOK, BISHOP, KNIGHT, PAWN)
private val PLURALS = mapOf(
    "queen" to "queens", "rook" to "rooks", "bishop" to "bishops",
    "knight" to "knights", "pawn" to "pawns",
)
private val NUMBERS = mapOf(
    1 to "a", 2 to "two", 3 to "three", 4 to "four",
    5 to "five", 6 to "six", 7 to "seven", 8 to "eight",
)

/**
 * The surplus pieces of one side, most valuable first: "a rook and two pawns"; without the
 * article after a possessive ("your knight").
 */
private fun piecesWords(counts: Map<Int, Int>, article: Boolean = true): String {
    val parts = ArrayList<String>()
    for (pieceType in COUNTED) {
        val n = counts[pieceType] ?: 0
        if (n <= 0) continue
        val pieceName = COACH_PIECE_NAMES[pieceType]
        if (n == 1) {
            parts.add(if (article) "a $pieceName" else pieceName)
        } else {
            parts.add("${NUMBERS[n] ?: n.toString()} ${PLURALS[pieceName]}")
        }
    }
    if (parts.size <= 1) return parts.joinToString("")
    return parts.dropLast(1).joinToString(", ") + " and ${parts.last()}"
}

/**
 * The material balance as `perspective` (the user; by default the side to move) would say it,
 * by the pieces: "you are up a bishop for a pawn", "you are down the exchange".
 */
fun materialText(board: Board, perspective: Boolean? = null): String {
    val who = perspective ?: board.turn
    val surplus = LinkedHashMap<Int, Int>()
    for (t in COUNTED) surplus[t] = board.pieces(t, who).size - board.pieces(t, !who).size
    val ours = surplus.filterValues { it > 0 }
    val theirs = surplus.filterValues { it < 0 }.mapValues { -it.value }
    if (ours.isEmpty() && theirs.isEmpty()) return "level"
    var diff = 0
    for ((t, n) in surplus) diff += FACT_VALUES[t] * n
    if (ours.isNotEmpty() && theirs.isNotEmpty()) {
        val more = if (diff >= 0) ours else theirs
        val less = if (diff >= 0) theirs else ours
        // A rook against a bishop or knight, and nothing else: the exchange.
        val exchange = more.size == 1 && more[ROOK] == 1 &&
            less.values.sum() == 1 && !less.containsKey(PAWN)
        val what = if (exchange) "the exchange" else "${piecesWords(more)} for ${piecesWords(less)}"
        if (diff == 0) {
            return "level in value: your ${piecesWords(ours, article = false)} against their " +
                piecesWords(theirs, article = false)
        }
        return "you are ${if (diff > 0) "up" else "down"} $what"
    }
    return "you are ${if (ours.isNotEmpty()) "up" else "down"} " +
        piecesWords(ours.ifEmpty { theirs })
}

/**
 * Tags that never headline: they hold on most quiet moves, and as the theme they sent the
 * stuck hint asking "which line could be cut?" in positions about nothing.
 */
private val NOT_A_THEME = setOf("interference", "clearance", "attackingF2F7", "quietMove")

/** The tag a coach would name as the idea, most specific first. */
fun themeOf(tags: List<MotifTag>): MotifTag? {
    val ranked = tags.filter { it.name in THEME_RANK && it.name !in NOT_A_THEME }
    if (ranked.isEmpty()) return null
    return ranked.minByOrNull { THEME_RANK.getValue(it.name) }
}

/** A part of the board with one anchor: "the kingside, around f7". */
fun regionOf(squares: List<String>): String? {
    if (squares.isEmpty()) return null
    val files = squares.map { squareFile(parseSquare(it)) }
    val ranks = squares.map { squareRank(parseSquare(it)) }
    val kingside = files.count { it >= 4 }
    var central = 0
    for (i in files.indices) {
        if (files[i] in 2..5 && ranks[i] in 2..5) central += 1
    }
    val where = when {
        central >= files.size && files.size > 1 -> "the centre"
        kingside * 2 >= files.size + 1 -> "the kingside"
        kingside * 2 <= files.size - 1 -> "the queenside"
        else -> "the centre"
    }
    // The anchor lies in the region: "the queenside, around g4" was the first focus square
    // named for a region the other two squares had chosen.
    val inRegion: (Int) -> Boolean = when (where) {
        "the kingside" -> { f -> f >= 4 }
        "the queenside" -> { f -> f < 4 }
        else -> { f -> f in 2..5 }
    }
    var anchor = squares[0]
    for (i in squares.indices) {
        if (inRegion(files[i])) { anchor = squares[i]; break }
    }
    return "$where, around $anchor"
}

/**
 * A theme's focus squares as a true sentence, or null when the board does not bear it.
 *
 * Some themes are static on the board in front of the player (a pin, a loose piece); others
 * are made by the best move (a fork the move creates). So the description is tried on the
 * current board first, then on the board after `bestMove`, and only returned when every
 * square it names holds the piece it claims.
 */
fun describeTactic(
    board: Board,
    name: String,
    squares: List<String>,
    bestMove: Move? = null,
): String? {
    val frames = ArrayList<Board>()
    frames.add(board)
    if (bestMove != null && board.isLegal(bestMove)) {
        val after = board.copy(withStack = false)
        after.push(bestMove)
        frames.add(after)
    }
    for (frame in frames) {
        val text = tacticOn(frame, name, squares)
        if (text != null) return text
    }
    return null
}

private fun tacticOn(board: Board, name: String, squares: List<String>): String? {
    val sqs = squares.map { parseSquare(it) }
    val pieces = sqs.map { board.pieceAt(it) }

    fun named(i: Int): String {
        val piece = pieces[i]!!
        return "the ${COACH_PIECE_NAMES[piece.pieceType]} on ${squares[i]}"
    }

    // A pin or skewer aligns three pieces: an enemy attacker, then two of the mover's own.
    val aligned = sqs.size >= 3 &&
        pieces.take(3).all { it != null } &&
        pieces[0]!!.colour != pieces[1]!!.colour &&
        pieces[1]!!.colour == pieces[2]!!.colour
    if ((name == "pin" || name == "absolutePin") && aligned) {
        return "${named(1)} is pinned to ${named(2)} by ${named(0)}"
    } else if (name == "skewer" && aligned) {
        return "${named(0)} skewers ${named(1)} and ${named(2)} behind it"
    }

    fun enemyOf(i: Int, j: Int): Boolean =
        pieces[i] != null && pieces[j] != null && pieces[i]!!.colour != pieces[j]!!.colour

    if (name == "fork" && sqs.size >= 3 && pieces[0] != null && pieces.drop(1).all { it != null }) {
        // The forking piece must bear on every target, and every target must be the enemy's.
        if ((1 until sqs.size).all { enemyOf(0, it) && sqs[it] in board.attacks(sqs[0]) }) {
            // The king first, then the most valuable, at most three.
            fun worth(i: Int): Int {
                val kind = pieces[i]!!.pieceType
                return if (kind == KING) 100 else FACT_VALUES[kind]
            }
            val order = (1 until sqs.size).sortedByDescending { worth(it) }.take(3)
            val targets = order.map { named(it) }
            val listed = if (targets.size == 1) {
                targets[0]
            } else {
                targets.dropLast(1).joinToString(", ") + " and ${targets.last()}"
            }
            return "${named(0)} forks $listed"
        }
    } else if ((name == "hangingPiece" || name == "trappedPiece") && pieces.isNotEmpty() && pieces[0] != null) {
        return "${named(0)} is ${if (name == "hangingPiece") "loose" else "trapped"}"
    } else if (name == "discoveredAttack" && sqs.size >= 3) {
        // The tagger records the vacated square, the slider, then what it now attacks.
        if (enemyOf(1, 2) && sqs[2] in board.attacks(sqs[1])) {
            return "${named(1)} now bears on ${named(2)}"
        }
    } else if (name == "capturingDefender" && sqs.size >= 2 && enemyOf(0, 1)) {
        return "${named(0)} removes a defender of ${named(1)}"
    }
    return null
}

/**
 * The position as a model reads it best: the FEN, every piece by colour and type with its
 * squares, and the legal moves for the side to move.
 */
fun positionText(board: Board, legal: Boolean = true): String {
    val letters = mapOf(KING to "K", QUEEN to "Q", ROOK to "R", BISHOP to "B", KNIGHT to "N", PAWN to "P")
    val sides = ArrayList<String>()
    for ((colour, word) in listOf(WHITE to "White", BLACK to "Black")) {
        val parts = ArrayList<String>()
        for (pieceType in intArrayOf(KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN)) {
            val squares = board.pieces(pieceType, colour).map { squareName(it) }.sorted()
            if (squares.isNotEmpty()) parts.add("${letters[pieceType]} ${squares.joinToString(" ")}")
        }
        sides.add("$word: " + parts.joinToString(", "))
    }
    val lines = arrayListOf("FEN: ${board.fen()}", "Pieces -- " + sides.joinToString(" | "))
    if (legal && !board.isGameOver()) {
        lines.add("Legal moves: " + board.legalMoves().map { board.san(it) }.sorted().joinToString(" "))
    }
    return lines.joinToString("\n")
}

/** The position as a labelled grid, which a language model reads far better than a FEN. */
fun asciiBoard(board: Board): String {
    val rows = ArrayList<String>()
    for (rank in 7 downTo 0) {
        val cells = ArrayList<String>()
        for (file in 0 until 8) {
            val piece = board.pieceAt(com.sidecoach.chess.square(file, rank))
            cells.add(piece?.symbol() ?: ".")
        }
        rows.add("${rank + 1} " + cells.joinToString(" "))
    }
    rows.add("  a b c d e f g h")
    return rows.joinToString("\n")
}

/**
 * A move in words, so a model never has to guess which piece moved where: "the knight from g1
 * to f3", "the bishop takes the knight on c6, giving check", "castles kingside".
 */
fun describeMove(board: Board, move: Move): String {
    if (board.isCastling(move)) {
        return "castles " + if (squareFile(move.to) > 4) "kingside" else "queenside"
    }
    val piece = board.pieceAt(move.from)
    val pieceName = if (piece != null) COACH_PIECE_NAMES[piece.pieceType] else "piece"
    val frm = squareName(move.from)
    val to = squareName(move.to)
    var text: String
    if (board.isCapture(move)) {
        val taken = board.pieceAt(move.to)
        val what = when {
            board.isEnPassant(move) -> "the pawn en passant"
            taken != null -> "the ${COACH_PIECE_NAMES[taken.pieceType]}"
            else -> "a piece"
        }
        text = "the $pieceName takes $what on $to"
    } else {
        text = "the $pieceName from $frm to $to"
    }
    if (move.promotion != 0) text += ", promoting to a ${COACH_PIECE_NAMES[move.promotion]}"
    val after = board.copy(withStack = false)
    after.push(move)
    if (after.isCheckmate()) {
        text += ", checkmate"
    } else if (after.isCheck()) {
        text += ", giving check"
    }
    return text
}

private fun toMove(uci: String): Move? = runCatching { Move.fromUci(uci) }.getOrNull()

/**
 * A line whose play-out gains this much material (in pawns' worth) is a tactic even when its
 * first move is quiet: a discovered attack, a knight move that forks next.
 */
const val TACTIC_GAIN = 2

/**
 * Whether the answer is forcing, a quiet move that wins material, or quiet, in the words the
 * sheet uses. `line` is the engine's line from `best` (UCI), played out to see whether it
 * wins material.
 */
fun natureOf(board: Board, best: Move, line: List<String> = emptyList()): String {
    if (board.isCapture(best) || board.givesCheck(best) || best.promotion != 0) {
        return "forcing -- the answer is a check, a capture or a promotion"
    }
    if (line.isNotEmpty() && lineGain(board, line) >= TACTIC_GAIN) {
        return "a tactic -- the best move is quiet, but the line after it wins material"
    }
    return "quiet -- no tactic to find; the move improves your position"
}

/**
 * Material gained by the side to move over `line` (UCI), in pawns' worth; the line stops at
 * the first illegal move.
 */
private fun lineGain(board: Board, line: List<String>): Int {
    val us = board.turn
    val after = board.copy(withStack = false)
    for (uci in line) {
        val move = toMove(uci) ?: break
        if (!after.isLegal(move)) break
        after.push(move)
    }
    fun balance(b: Board): Int {
        var total = 0
        for (piece in b.pieceMap().values) {
            val value = FACT_VALUES[piece.pieceType]
            total += if (piece.colour == us) value else -value
        }
        return total
    }
    return balance(after) - balance(board)
}

/**
 * The fact bundle for `board`, which is the position `analysis` measured.
 *
 * `board` must be the position the side to move faces; the candidates, threats and plan are
 * the side to move's. `user` is the side the coach speaks to -- the material is read from
 * that side -- and by default the side to move.
 */
fun factsFromAnalysis(
    board: Board,
    analysis: Analysis,
    lastMove: String? = null,
    inBook: Boolean = false,
    user: Boolean? = null,
): PositionFacts {
    val lines = analysis.lines
    val bestUci = if (lines.isNotEmpty() && lines[0].moves.isNotEmpty()) lines[0].moves[0] else null
    var best = if (bestUci != null) toMove(bestUci) else null
    if (best != null && !board.isLegal(best)) best = null
    val phase = phaseOf(board, inBook)
    val us = board.turn

    // A tactic is the headline only when it is the point: a decisive or forced position, or
    // one where the best move plays into the very squares the tactic is on.
    val theme = themeOf(analysis.tags)
    val bestSquares = if (best != null) setOf(best.from, best.to) else emptySet()
    val tacticSquares = theme?.squares?.map { parseSquare(it) }?.toSet() ?: emptySet()
    val tacticRelevant = theme != null &&
        (analysis.decisive || analysis.onlyMove || bestSquares.intersect(tacticSquares).isNotEmpty())
    val region = if (tacticRelevant && theme != null) regionOf(theme.squares) else null
    val focus = if (tacticRelevant && theme != null) theme.squares else emptyList()
    val tactic = if (tacticRelevant && theme != null) {
        describeTactic(board, theme.name, theme.squares, best)
    } else {
        null
    }
    val themeText = if (tacticRelevant && theme != null) THEMES.toMap()[theme.name] else null

    // The opponent's threat, as the coach would name it, and the move behind it: the engine's
    // best reply after a pass.
    val threats = ArrayList<String>()
    var threatMove: Move? = null
    if (analysis.threats.isNotEmpty() && !board.isCheck()) {
        val passed = nullMoved(board)
        for (uci in analysis.threats) {
            val move = toMove(uci)
            if (move != null && passed.isLegal(move)) {
                threats.add(passed.san(move))
                if (threatMove == null) threatMove = move
            }
        }
    }

    // The idea half: the strategic point of the best move, and the few most instructive static
    // features. The plan only competes for the theme slot when no tactic leads.
    var plan: String? = null
    if (best != null) {
        val worst = worstPieceSquare(board, us)
        val idea: PositionalFact? = bestMoveIdea(board, best, us, phase, worst, threatMove)
        if (idea != null && !tacticRelevant) plan = PLAN_THEME[idea.kind]
    }
    val ideas = topFeatures(board, us, phase, limit = 2).map { it.text }

    var piece: String? = null
    var nature: String? = null
    if (best != null) {
        val moving = board.pieceAt(best.from)
        if (moving != null) piece = "your ${COACH_PIECE_NAMES[moving.pieceType]}"
        nature = natureOf(board, best, if (lines.isNotEmpty()) lines[0].moves else emptyList())
    }

    val candidates = ArrayList<String>()
    if (lines.isNotEmpty()) {
        val top = lines[0].expectedScore
        for (line in lines) {
            if (line.moves.isEmpty() || top - line.expectedScore > CANDIDATE_MARGIN) continue
            val move = toMove(line.moves[0])
            if (move != null && board.isLegal(move)) candidates.add(board.san(move))
        }
    }

    var line = if (lines.isNotEmpty()) {
        sanLine(board, lines[0].moves, limit = 8).split(" ").filter { it.isNotEmpty() }
    } else {
        emptyList()
    }
    line = line.filter { !it.endsWith(".") && !it.endsWith("...") }

    return PositionFacts(
        fen = board.fen(),
        sideToMove = if (board.turn == WHITE) "white" else "black",
        userSide = if (user == null) null else (if (user == WHITE) "white" else "black"),
        phase = phase,
        material = materialText(board, user),
        lastMove = lastMove,
        decisive = analysis.decisive,
        onlyMove = analysis.onlyMove,
        nature = nature,
        theme = themeText,
        region = region,
        piece = piece,
        focus = focus,
        tactic = tactic,
        plan = plan,
        ideas = ideas,
        candidates = candidates,
        threats = threats,
        line = line,
    )
}

/**
 * The position with the turn handed over, as `board.push(chess.Move.null())` leaves it: the
 * rights cleaned, the en-passant square gone, the clocks ticked.
 */
fun nullMoved(board: Board): Board {
    val passed = board.copy(withStack = false)
    passed.castlingRights = passed.cleanCastlingRights()
    passed.epSquare = null
    passed.halfmoveClock += 1
    if (passed.turn == BLACK) passed.fullmoveNumber += 1
    passed.turn = !passed.turn
    return passed
}
