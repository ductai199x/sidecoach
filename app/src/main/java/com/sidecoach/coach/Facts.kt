package com.sidecoach.coach

import com.sidecoach.analysis.Exchange
import com.sidecoach.analysis.PositionalFact
import com.sidecoach.analysis.bestMoveIdea
import com.sidecoach.analysis.phaseOf
import com.sidecoach.analysis.sanLine
import com.sidecoach.analysis.topFeatures
import com.sidecoach.analysis.worstPiece
import com.sidecoach.analysis.worstPieceSquare
import com.sidecoach.chess.BB_EMPTY
import com.sidecoach.chess.BB_SQUARES
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
    "passerPawnEnding" to "making a passed pawn",
)

/**
 * The best move's own idea that comes before the plan its line carries out: a king to get to
 * safety, a piece to save, their threat to stop, a piece still at home. Anything else gives way
 * to the plan of the player's next moves ([Plans]).
 */
private val BEFORE_THE_PLAN = setOf("castle", "rescue", "prophylaxis", "develop")

/**
 * The best move's own ideas that belong beside a threat to meet: the rest (a rook to an open
 * file, a knight to the centre) are what the move also does, and "stop it before it comes. The
 * key is taking the open file" named the wrong reason (the master games, 2026-09-24).
 */
private val DEFENDING = setOf("castle", "rescue", "prophylaxis")

/**
 * A threat to win less than this, in pawns, gives way to the move's own idea: "d4 goes after
 * their pawn on e5" is the fight for the centre, not the initiative.
 */
const val THREAT_MIN = 2

/**
 * In a pawn ending there is no king to storm and no space to take: a pawn move there is about
 * the passed pawn it makes ("storming the pawns in front of their King" was said of h5 in a
 * king-and-pawn ending, Lichess Yb1jJ).
 */
private val PAWN_ENDING_PLAN = mapOf(
    "break" to "passerPawnEnding",
    "space" to "passerPawnEnding",
    "pawnStorm" to "passerPawnEnding",
    "centre" to "passerPawnEnding",
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
    // A pawn on the sixth or seventh rank, passed or not: "a passed pawn" was said of b6+ with
    // their pawn on b7 in front (the test audit, 2026-09-26).
    "advancedPawn" to "a pawn deep in their half",
    // "A piece attacked through another" was read as "attacks your knight through another
    // piece" of a direct attack; the x-ray is the attack running on past its target.
    "xRayAttack" to "an x-ray: the attack runs on through one piece to the one behind it",
    "interference" to "a line that can be cut",
    "clearance" to "a square that can be cleared",
)

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
    // Won at 0.8, as the idea reader counts a won position (`IdeaReader.WON`): "better" of a
    // position four pawns up read wrong against ChessQA's bands (93% -> 96% at one line).
    if (score >= 0.8) return "$who winning"
    if (score >= 0.62) return "$who better"
    if (score > 0.38) return "it is about equal"
    if (score > 0.2) return "$who worse"
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
        // The forking piece must bear on every target, and every target must be the enemy's:
        // a fork read off a board that has moved on is otherwise built from the wrong pieces.
        val forkerColour = pieces[0]!!.colour
        val takes = HashSet<Int>()
        for (t in sqs.drop(1)) {
            val piece = pieces[sqs.indexOf(t)]
            if (piece != null &&
                (piece.pieceType == KING || sqs[0] in Exchange.capturers(board, forkerColour, t))
            ) {
                takes.add(t)
            }
        }
        if ((1 until sqs.size).all { enemyOf(0, it) && sqs[it] in board.attacks(sqs[0]) && sqs[it] in takes }) {
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
 * What makes the other side's move after a pass a threat, as the sheet names its effects: it
 * mates, takes, checks, or attacks something. A move that only defends, retreats or takes a file
 * is what they would do with a free move, not a threat ("They threaten Rc3." of a rook guarding
 * the third rank against a mate in three: dev audit, 2026-09-26).
 */
private val THREAT_EFFECTS = listOf(
    "checkmate",
    "takes ",
    "gives check",
    "forks ",
    "attacks ",
    "uncovers an attack",
    "frees ",
    "pins ",
)

/** `facts._threatens`. */
internal fun threatens(passed: Board, move: Move): Boolean {
    if (move.promotion != 0) return true
    val effects = describeEffects(passed, move, limit = 8, hazards = false)
    if (effects.any { e -> THREAT_EFFECTS.any { e.startsWith(it) } }) return true
    // A quiet move that threatens mate on an empty square attacks nothing yet.
    val again = passed.copy(withStack = false)
    again.push(move)
    if (again.isCheck() || again.isGameOver()) return false
    again.push(Move.NULL)
    for (reply in again.legalMoves().toList()) {
        again.push(reply)
        val mate = again.isCheckmate()
        again.pop()
        if (mate) return true
    }
    return false
}

/**
 * A line whose play-out gains this much material (in pawns' worth) is a tactic even when its
 * first move is quiet: a discovered attack, a knight move that forks next.
 */
const val TACTIC_GAIN = 2

/**
 * In check, the first hint's question and the nature of a quiet answer are the check: "a quiet
 * improvement -- which of your pieces has no job right now?" was the hint in check positions,
 * and the simulated students did not understand it (desktop 4cc6f11).
 */
const val IN_CHECK_QUESTION =
    "You are in check: can you take the checking piece, block it, or must your king move -- " +
        "and which way out leaves you safest?"
const val IN_CHECK_NATURE = "a check to answer -- take the checking piece, block it, or move the king"

/** A quiet first move of a tactic the idea reader finds, by its goal. `facts._QUIET_TACTIC`. */
private val QUIET_TACTIC = mapOf(
    "win" to "a tactic -- the best move is quiet, but the line after it wins material",
    "combination" to "a tactic -- the best move is quiet, but the line after it wins material",
    "mate" to "a tactic -- the best move is quiet, but the line after it mates",
    "promote" to "a tactic -- the best move is quiet, but the line after it queens a pawn",
    "attack" to "a tactic -- the best move is quiet, but the attack after it wins",
)

/**
 * Whether the answer is forcing, a quiet move that wins material, or quiet, in the words the
 * sheet uses. `line` is the engine's line from `best` (UCI), played out to see whether it
 * wins material.
 */
fun natureOf(board: Board, best: Move, line: List<String> = emptyList()): String {
    if (board.isCapture(best) || board.givesCheck(best) || best.promotion != 0) {
        return "forcing -- the answer is a check, a capture or a promotion"
    }
    // Counted as the idea counts it, net of what they just took ([IdeaReader.exchangeOwed]).
    val owed = IdeaReader.worth(IdeaReader.exchangeOwed(board))
    if (line.isNotEmpty() && lineGain(board, line) - owed >= TACTIC_GAIN) {
        return "a tactic -- the best move is quiet, but the line after it wins material"
    }
    return "quiet -- no tactic to find; the move improves your position"
}

/**
 * Material gained by the side to move over `line` (UCI), in pawns' worth, counted as the idea
 * reader counts its line ([IdeaReader.lineGain]): after the opponent's replies, a line that
 * stops mid-trade not counting what can still be taken back ("a3 Kd7 Nxe5" was a knight won,
 * with dxe5 to come). The line stops at the first illegal move.
 */
private fun lineGain(board: Board, line: List<String>): Int {
    val moves = ArrayList<Move>()
    val boards = arrayListOf(board.copy(withStack = false))
    for (uci in line) {
        val move = toMove(uci) ?: break
        if (!boards.last().isLegal(move)) break
        val after = boards.last().copy(withStack = false)
        after.push(move)
        moves.add(move)
        boards.add(after)
    }
    return if (moves.isNotEmpty()) IdeaReader.lineGain(boards, moves, moves.size).first else 0
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
    user: Boolean? = null,
    /** The live coach's bags, drawing the first hint's lead ([IdeaWords.opener]). */
    pick: ((String, List<String>) -> String)? = null,
): PositionFacts {
    val lines = analysis.lines
    val bestUci = if (lines.isNotEmpty() && lines[0].moves.isNotEmpty()) lines[0].moves[0] else null
    var best = if (bestUci != null) toMove(bestUci) else null
    if (best != null && !board.isLegal(best)) best = null
    val phase = phaseOf(board)
    val us = board.turn

    // What the best line is for, read through the player's own moves ([IdeaReader]): the goal,
    // the blow that delivers it on whichever move it comes, the target. The pattern is the
    // blow's, never a static feature elsewhere on the board: a static pin took the headline from
    // the fork on the player's second move, and a mate in one went out as "getting a piece out
    // of danger" (Tai's puzzles, 2026-09-23).
    val idea = IdeaReader.read(board, analysis)
    val tactical = idea.goal in listOf("mate", "win", "promote", "attack", "combination")
    val blow = if (tactical) idea.blow else null
    val themeText = if (tactical) IdeaWords.pattern(idea) else null
    // The squares the pattern turns on, as they stand now: the blow's when it is the first
    // move, else what it goes after.
    val focus = if (blow != null && idea.blowAt == 1) {
        blow.squares.toList()
    } else if (idea.target != null && tactical) {
        listOf(squareName(idea.target))
    } else {
        emptyList()
    }
    val region = if (focus.isNotEmpty()) regionOf(focus) else null
    val tactic = if (blow != null && idea.blowBoard != null) {
        describeTactic(idea.blowBoard, blow.name, blow.squares, idea.blowMove)
    } else {
        null
    }

    // The opponent's threat, as the coach would name it, and the move behind it: the engine's
    // best reply after a pass.
    val threats = ArrayList<String>()
    var threatMove: Move? = null
    if (analysis.threats.isNotEmpty() && !board.isCheck()) {
        val passed = nullMoved(board)
        for (uci in analysis.threats) {
            val move = toMove(uci)
            if (move != null && passed.isLegal(move)) {
                // Their best move if you passed stays the prophylactic question's ("if you let
                // them, their best is Kc2"); it is listed as a threat only when it is one.
                if (threatens(passed, move)) threats.add(passed.san(move))
                if (threatMove == null) threatMove = move
            }
        }
    }

    // The idea half: the strategic point of the best move (a square-free theme), and the few
    // most instructive static features. The plan only competes for the theme slot when no
    // tactic leads, so the coach is never handed two headlines at once.
    var plan: String? = null
    var planKind: String? = null
    if (best != null && (idea.goal == "plan" || idea.goal == "defend")) {
        val worst = worstPieceSquare(board, us)
        val planIdea: PositionalFact? = bestMoveIdea(board, best, us, phase, worst, threatMove)
        if (planIdea != null) {
            var kind = planIdea.kind
            val pawnEnding = board.pieceMap().values.all { it.pieceType == KING || it.pieceType == PAWN }
            if (pawnEnding) kind = PAWN_ENDING_PLAN[kind] ?: kind
            plan = PLAN_THEME[kind]
            planKind = if (plan != null) kind else null
            if (idea.goal == "defend" && (planKind == null || planKind !in DEFENDING)) {
                plan = null
                planKind = null
            }
        }
    }
    // The plan the best line carries out over the player's next moves: a king attack, rooks
    // doubled on a file, a knight's route to an outpost. What a grandmaster's note says of a
    // quiet position, where the best move alone reads as "a rook to an open file". And the
    // threat the best move makes, where the line wins nothing because they answer it: a threat
    // of mate comes before any plan, a threat to win material after the line's plan.
    var linePlan: LinePlan? = null
    var moveThreat: MoveThreat? = null
    // A move out of check is not a plan or a threat: it is the way out (20.Ke2 was "you can
    // take the initiative").
    if (best != null &&
        idea.goal == "plan" &&
        (planKind == null || planKind !in BEFORE_THE_PLAN) &&
        !board.isCheck()
    ) {
        val pv = lines[0].moves.mapNotNull { toMove(it) }
        linePlan = Plans.read(board, pv, lines[0].expectedScore)
        moveThreat = Plans.threat(board, best, pv, lines[0].expectedScore)
        if (moveThreat != null &&
            moveThreat.kind == "win" &&
            moveThreat.gain < THREAT_MIN &&
            planKind != null
        ) {
            moveThreat = null  // a pawn attacked is not the point when the move has its own
        }
        if (moveThreat != null && (moveThreat.kind == "mate" || linePlan == null)) {
            linePlan = null
            plan = Plans.THREAT_NAME.getValue(moveThreat.kind)
            planKind = null
        } else if (linePlan != null) {
            moveThreat = null
            plan = Plans.name(linePlan)
            planKind = null
        }
    }
    val ideas = topFeatures(board, us, phase, limit = 2).map { it.text }

    var piece: String? = IdeaWords.piece(idea, board)
    var nature: String? = null
    if (best != null) {
        nature = natureOf(board, best, if (lines.isNotEmpty()) lines[0].moves else emptyList())
        // The nature agrees with the idea: a tactic the idea reader finds -- a pawn won back, a
        // mate -- under TACTIC_GAIN was "quiet -- no tactic to find" beside "You have a tactic
        // that wins back a pawn" (the pre-spend audit, 2026-09-25).
        if (tactical && nature.startsWith("quiet")) {
            nature = QUIET_TACTIC[idea.goal] ?: QUIET_TACTIC.getValue("win")
        }
    }
    var mechanism = IdeaWords.mechanism(idea)
    if (mechanism == null && planKind != null && planKind in IdeaWords.PLAN_WHY) {
        mechanism = "$plan: ${IdeaWords.PLAN_WHY.getValue(planKind)}"
    }
    var question: String? = IdeaWords.question(idea, planKind, board)
    if (board.isCheck() && question in Situation.ANY_POSITION) {
        // In check the question is the check: "a quiet improvement -- which of your pieces has
        // no job right now?" was the hint in seven of ten AI sessions' check positions, and the
        // students listed it among what they did not understand (round 4, 2026-09-26).
        question = IN_CHECK_QUESTION
        if (nature != null && nature.startsWith("quiet")) nature = IN_CHECK_NATURE
    }
    var target = IdeaWords.target(idea, board)
    var nudge = IdeaWords.nudge(idea, board)
    var why = IdeaWords.why(idea, board, IdeaWords.Describe(::describeTactic))
    var goal = IdeaWords.goal(idea)
    if (moveThreat != null && best != null) {
        val won = if (idea.winning) Plans.THREAT_GOAL_WON else Plans.THREAT_GOAL
        goal = won.getValue(moveThreat.kind)
        mechanism = Plans.threatMechanism(moveThreat)
        question = Plans.THREAT_QUESTION.getValue(moveThreat.kind)
        target = Plans.threatTarget(moveThreat, board) ?: target
        nudge = Plans.THREAT_NUDGE.getValue(moveThreat.kind)
        why = Plans.threatWhy(moveThreat, board, best)
    }
    // A forcing capture with nothing else to point at: its focus is what it takes.
    if (target == null &&
        best != null &&
        idea.goal == "plan" &&
        moveThreat == null &&
        linePlan == null &&
        board.isCapture(best)
    ) {
        val taken = board.pieceAt(best.to)
        if (taken != null && taken.colour != us) {
            target = "their ${COACH_PIECE_NAMES[taken.pieceType]} on ${squareName(best.to)}"
        }
    }
    if (linePlan != null) {
        mechanism = Plans.mechanism(linePlan)
        question = Plans.PLAN_QUESTION.getValue(linePlan.kind)
        target = Plans.target(linePlan) ?: target
        piece = Plans.pieces(linePlan, board) ?: piece
        nudge = Plans.PLAN_NUDGE.getValue(linePlan.kind)
        why = Plans.why(linePlan, board)
    }

    // The first hint ([Situation]): the stakes, the key fact named on the board, the principle
    // behind it, the trap -- a short lesson on the position, never the move.
    val stakes = Situation.readStakes(board, analysis)
    var stakesText = if (stakes != null) Situation.stakesWords(board, stakes, analysis.opening) else null
    val choices = if (stakes != null && (stakes.kind == "many" || stakes.kind == "book")) stakes.choices.toList() else emptyList()
    val pv = if (lines.isNotEmpty()) lines[0].moves.mapNotNull { toMove(it) } else emptyList()
    var key: String? = null
    var principle: String? = null
    var tacticalGoal = idea.goal in listOf(
        "mate",
        "win",
        "promote",
        "attack",
        "combination",
        "draw",
        "convert",
        "king",
        "breakthrough",
    )
    if (idea.goal == "defend") {
        key = Situation.defendKey(board, idea, best)
    } else if (idea.goal == "mate" || idea.goal == "attack") {
        key = Situation.mateKey(board, idea)
        principle = IdeaWords.lesson(IdeaWords.mechanism(idea), idea)
    } else if (tacticalGoal) {
        key = IdeaWords.target(idea, board)
        if (key != null && ":" !in key && " is " !in key && "," !in key) {
            key = "$key is the target"  // "their rook on a8" alone said nothing
        } else if (key != null && idea.goal == "promote" && ", " in key) {
            key = key.replaceFirst(", ", " is ")  // "your pawn on b6 is two squares from queening"
        }
        if (key == null && idea.goal == "win") {
            key = Situation.winKey(board, idea)
            if (key == null && IdeaReader.endingOf(board)) key = Situation.kingKey(board, idea)  // the pawn falls to a king's march
        } else if (key == null && idea.goal == "convert") {
            key = Situation.convertKey(board, idea)
        } else if (key == null && idea.goal == "king") {
            key = Situation.kingKey(board, idea)
        }
        // The principle is the pattern's: a sacrifice on the way, with no pattern, is not what a
        // winning capture teaches (17.hxg4 was "a sacrifice").
        principle = if (idea.blow != null || idea.goal in listOf("mate", "combination", "attack", "promote")) {
            IdeaWords.lesson(IdeaWords.mechanism(idea), idea)
        } else {
            null
        }
    } else if (moveThreat != null) {
        key = Situation.threatKey(board, moveThreat)
        principle = if (moveThreat.kind == "trade") {
            Plans.TRADE_REASON.getValue(moveThreat.why)
        } else {
            Plans.THREAT_REASON.getValue(moveThreat.kind)
        }
    } else if (linePlan != null) {
        key = Situation.planKey(board, linePlan)
        principle = Plans.PLAN_REASON.getValue(linePlan.kind)
    } else if (planKind != null && best != null) {
        key = Situation.positionalKey(board, best, planKind, threatMove, phase)
        // The second hint's question follows from the fact the first named: without it, "which
        // pawn move lets it out?" asked of a position where nothing is shut in.
        if (nudge == null && key != null) {
            nudge = Situation.positionalNudge(board, planKind, key)
        } else if ((planKind == "centralise" || planKind == "activate") && key == null && why == null) {
            why = Situation.reachWhy(board, best)
        } else if (planKind == "centre" && key == null) {
            val said = Situation.centreMove(board, best)
            if (said != null) {
                nudge = if (!nudge.isNullOrEmpty()) nudge else said.first
                why = if (!why.isNullOrEmpty()) why else said.second
            }
        }
    }
    // One story ([Story]): a piece that simply hangs is said as that, before any pattern the
    // line also shows.
    val plain = if (tacticalGoal) Story.plainCapture(board, idea) else null
    if (plain != null && plain.whole) {
        key = plain.key
        principle = null
        question = null
        nudge = plain.nudge
        why = plain.why
    } else if (plain != null) {
        // The free piece first, then what the line wins after it.
        val named = "on ${squareName(plain.square)}"
        val rest = if (!key.isNullOrEmpty() && named !in key) ". After that, ${key.replaceFirstChar { it.lowercase() }}" else ""
        key = plain.key + rest
    }
    // The key the move itself answers, and so its reason at the move rung; the fallbacks below
    // (a static feature, their idea) are context, not the move's reason.
    val moveReason = key
    val inCheck = board.isCheck()
    val book = stakes != null && stakes.kind == "book"
    if (key == null &&
        best != null &&
        !tacticalGoal &&
        idea.goal != "defend" &&
        !inCheck &&
        !book
    ) {
        key = Situation.passerKey(board, best) ?: Situation.kingsKey(board, best)
        if (key == null) {
            val features = topFeatures(board, us, phase, limit = 4)
            key = Situation.relevantFeature(board, features, pv)
        }
    }
    if (inCheck && key == null) key = Situation.checkKey(board)
    if (key == null && !tacticalGoal && !book) {
        // Nothing else named: what they would do if you let them -- the prophylactic question,
        // answered on the board.
        val sharp = stakes != null && (stakes.kind == "only" || stakes.kind == "few" || stakes.kind == "lost")
        key = Situation.theirIdeaKey(board, threatMove, sharp = sharp)
    }
    if (inCheck) {
        nudge = "Which way out keeps you safest -- a king move, a block, or taking the checker?"
        if (why == null && best != null) why = Situation.evasionWhy(board, best)
    }
    // A recapture: the best move takes back what the line wins, at once (8.bxc3). Beside it, only
    // the wrong piece to take back with is a trap -- every other capture is worse only for not
    // taking back -- and 23.exd5? Re8 in Polugaevsky-Nezhmetdinov is the lesson.
    val recapture = idea.goal == "win" &&
        (idea.down || idea.back) &&
        best != null &&
        board.isCapture(best) &&
        best.to == idea.target
    val traps = analysis.traps.filter { t ->
        !recapture || (best != null && t.moves.isNotEmpty() && t.moves[0].substring(2, 4) == best.uci().substring(2, 4))
    }
    var trap: String? = null
    if (idea.goal != "mate") {
        // The trap that matters most and can be said: on the target, into mate, the worst.
        for (chosen in Situation.rankedTraps(traps, idea.target)) {
            trap = Situation.trapWords(
                board, chosen, idea.target, if (lines.isNotEmpty()) lines[0].expectedScore else null, best,
            )
            if (trap != null) break
        }
    }
    if (trap != null && idea.target != null && Situation.NOT_YET in trap && plain == null) {
        key = null  // the trap says it: the target can be won, but not at once
    }
    val parts = ArrayList<String>()
    val forced = board.isCheck() && board.legalMoves().size == 1
    val calm = stakes != null && (stakes.kind == "many" || stakes.kind == "open")
    if (!stakesText.isNullOrEmpty() && idea.goal == "defend" && calm) {
        stakesText = null  // "nothing is urgent" beside their threat said the opposite
    }
    if (!stakesText.isNullOrEmpty() &&
        (!tacticalGoal || (stakes != null && (stakes.kind == "check" || stakes.kind == "book")))
    ) {
        parts.add(stakesText)
    }
    // Where only one or a few moves hold, the move's reason is what it does on the board --
    // takes, checks, attacks, stops their idea -- never "The move is Bb5+." alone.
    val sharp = stakes != null &&
        stakesText != null &&
        (
            stakes.kind == "only" ||
                stakes.kind == "few" ||
                ((stakes.kind == "won" || stakes.kind == "check") && "only" in stakesText)
            )
    if (sharp && stakes != null && best != null && (why == null || "goes from" in why)) {
        val edge = Situation.edgeWords(stakes.top)
        val holds = if (stakes.kind == "few" && stakes.good > 1) {
            "it is one of the few moves that keep $edge"
        } else {
            "it is the only move that keeps $edge"
        }
        val (reason, ask) = Story.reason(board, best, threatMove, holds)
        why = reason
        if (why == "${board.san(best)}: $holds") why = Story.lineReason(board, pv, holds) ?: why
        if (nudge == null) nudge = ask
    }
    // In check, the piece giving check comes first, unless the capture that wins takes it.
    val checker = if (inCheck &&
        !forced &&
        !(plain != null && board.checkersMask() and BB_SQUARES[plain.square] != BB_EMPTY)
    ) {
        Situation.checkKey(board, if (tacticalGoal) idea.target else null)
    } else {
        null
    }
    var pawn = if (tacticalGoal && idea.goal != "promote" && !forced) Story.queening(board, idea) else null
    if (pawn != null && key != null && "queening" in key) pawn = null
    val said = listOf(checker, key, trap).filter { !it.isNullOrEmpty() }.joinToString(" ")
    val prize = if (tacticalGoal && !forced && (plain == null || !plain.whole)) Story.prize(board, idea, said) else null
    // After a free piece taken first, the prize comes where the line wins it: last.
    val prizeLast = plain != null
    // How a win with no named pattern comes about: what the first move does, and at the move
    // rung the line to it, never the result alone.
    val toldHow = if (tacticalGoal && !forced && plain == null) {
        Story.how(board, idea, threatMove, if (analysis.threatens.isNotEmpty()) analysis.threatens[0] else null)
    } else {
        null
    }
    var how = toldHow?.said
    if (toldHow != null && !toldHow.lesson.isNullOrEmpty()) {
        // The pattern the move carries, taught after it: "That's a trapped piece: ..."
        val taught = IdeaWords.lesson(toldHow.lesson, idea)?.ifEmpty { null } ?: toldHow.lesson
        how = "$how. ${taught.replaceFirstChar { it.uppercase() }}"
        nudge = toldHow.ask.ifEmpty { null } ?: nudge
    } else if (toldHow != null && toldHow.ask.isNotEmpty() && !inCheck) {
        // The second hint asks after what the first said the move is for: "which quiet move sets
        // up a mate?" after "the first move is quiet, but it threatens mate".
        nudge = toldHow.ask
    }
    if (tacticalGoal &&
        best != null &&
        idea.blow == null &&
        (why == null || Regex("${Regex.escape(board.san(best))}: you win [^;]*").matches(why))
    ) {
        why = Story.moveWhy(board, idea, threatMove) ?: why
    }
    // Why the obvious defence fails ([Story.defence]): the recapture the line does not play, and
    // the other job of the piece that would make it -- the overloaded defender, said in place of
    // "a loose piece" when the loose piece was only half the story (Tai's chess.com puzzles,
    // 2026-09-24: "you can win their queen" read as a lie beside fxg3, and a rook on f8 that had
    // to stop f8=Q was "not defended at all").
    val held = if (tacticalGoal && !forced && analysis.defences.isNotEmpty() && idea.goal != "mate") {
        Story.defence(board, analysis.defences, if (lines.isNotEmpty()) lines[0].moves else emptyList())
    } else {
        null
    }
    var defended: String? = null
    if (held != null) {
        val loose = idea.blow?.name == "hangingPiece"
        val known = key
        if (known == null || (loose && (held.square in known || held.target in known))) {
            // The loose piece was half the story: the defender is there, with another job.
            key = held.said
            principle = IdeaWords.lesson(held.lesson, idea)
            question = held.ask
            why = why?.let { Regex("the \\w+ on [a-h][1-8] is loose").replace(it) { held.clause } }
        } else {
            defended = held.said
            principle = if (!principle.isNullOrEmpty()) principle else IdeaWords.lesson(held.lesson, idea)
        }
        nudge = held.ask
        why = why?.let { "$it; ${held.line}" }
    }
    if (forced) {
        // One legal move: the rest of the reading is the move's, not a hint.
        tacticalGoal = false
        key = null
        principle = null
        trap = null
        moveThreat = null
        nudge = null
    }
    if (checker != null && !forced && checker != key) parts.add(checker)
    val lost = stakes != null && stakes.kind == "lost"
    // A line that wins by force -- not one that starts by taking a piece simply left hanging or
    // taking back ([Story.plainCapture]) -- is a tactic, and the first hint says there is one to
    // find (Tai, 2026-09-25).
    val isTactic = (
        idea.goal in listOf("win", "combination") ||
            (idea.goal == "promote" && (idea.blow != null || idea.sacrifices.isNotEmpty()))
        ) && plain == null
    if (tacticalGoal ||
        (moveThreat != null && !lost) ||
        (idea.winning && stakesText.isNullOrEmpty())
    ) {
        var lead = goal
        if (idea.goal == "plan" && idea.winning && moveThreat == null) {
            lead = "you are winning"
        } else if (moveThreat == null) {
            val seed = IdeaWords.openerSeed(board)
            lead = IdeaWords.opener(idea, goal, tactic = isTactic, seed = seed, pick = pick)
        }
        if (!lead.isNullOrEmpty()) parts.add(lead)
    }
    // Facts about the same piece or square said as one sentence, the way a coach links them.
    var shownKey = key
    if (held != null && !forced) {
        val heldAsKey = key == held.said
        val stops = held.stops

        // The pawn the defender has to stop, named in `text`.
        fun stopsIt(text: String?): Boolean =
            text != null && stops != null && "pawn on $stops " in text && "queening" in text

        val also = "${held.who} has to stop it, so it cannot also take back on ${held.target}"
        val known = key
        if (stopsIt(pawn)) {
            // "Your pawn on e6 is two squares from queening, and their rook on f8 has to stop it,
            // so it can't also take back on b8."
            pawn = "$pawn, and $also"
            shownKey = if (heldAsKey) null else known
            defended = null
        } else if (stopsIt(known) && !heldAsKey) {
            shownKey = "$known, and $also"
            defended = null
        } else if (defended != null && known != null && known.endsWith("on ${held.target} is the target")) {
            // "Their queen on g3 is the target, and their pawn on f2 can't take back there: it
            // also shields their king."
            shownKey = "$known, and ${held.who} cannot take back there: it also ${held.duty}"
            defended = null
        }
    }
    val keyShown = shownKey
    val targetSaid = if (!keyShown.isNullOrEmpty() && keyShown.endsWith(" is the target")) {
        Regex("[Tt]heir (\\w+) on [a-h][1-8]").matchEntire(keyShown.substring(0, keyShown.length - " is the target".length))
    } else {
        null
    }
    val howSaid = how
    if (!howSaid.isNullOrEmpty() && !keyShown.isNullOrEmpty() && targetSaid != null) {
        val their = "t" + keyShown.substring(1, keyShown.length - " is the target".length)  // "their rook on a1"
        if (howSaid.startsWith("The first move ")) {
            // "Their queen on d3 is the target, and the first move attacks it."
            var rest = howSaid.substring("The first move ".length)
            rest = Regex("^(\\w+) ${Regex.escape(their)}\\b").replace(rest, "\$1 it")
            if ("," in rest.substringBefore(". ")) {
                how = "The first move $rest"  // a clause of its own: said apart
            } else {
                shownKey = "$keyShown, and the first move $rest"
                how = null
            }
        }
        val now = how
        if (now != null && their in now) {
            // Named once: "... a check that also attacks the rook, which has no safe square".
            how = now.replace(their, "the ${targetSaid.groupValues[1]}")
        }
    }
    if (!pawn.isNullOrEmpty()) parts.add(pawn)
    if (!prize.isNullOrEmpty() && !prizeLast) parts.add(prize)
    if (!shownKey.isNullOrEmpty()) parts.add(shownKey)
    if (!defended.isNullOrEmpty() && !forced) parts.add(defended)
    if (!how.isNullOrEmpty()) parts.add(how)
    if (!prize.isNullOrEmpty() && prizeLast) parts.add(prize)
    if (!principle.isNullOrEmpty() && !key.isNullOrEmpty()) parts.add(principle)
    val warned = trap
    if (!warned.isNullOrEmpty()) {
        // The target named already (the piece, not only its square): the order is the lesson.
        var told = warned
        val target = idea.target
        val prey = if (target != null) board.pieceAt(target) else null
        if (prey != null && target != null) {
            val named = "${IdeaWords.PIECE_NAMES.getValue(prey.pieceType)} on ${squareName(target)}"
            if (named in parts.joinToString(" ")) told = Situation.inOrder(told)
        }
        if (parts.isEmpty()) told = Situation.firstSaid(told)
        trap = told
        parts.add(told)
    }
    if (parts.isEmpty() && !forced && best != null) {
        // Nothing else to say: what they would do with a free move, or the piece doing least --
        // never a question that fits any position ("which of your pieces has no job right now?"
        // at 32...Kh8 in Carlsen-Kramnik).
        var fallback = Situation.theirIdeaKey(board, threatMove, sharp = true)
        if (fallback == null) {
            val passive = worstPiece(board, us)
            fallback = if (passive.isNotEmpty()) passive[0].text else null
        }
        if (!fallback.isNullOrEmpty()) parts.add(fallback)
    }
    var situationText: String? = parts.joinToString(" ") { Situation.sentence(it) }.ifEmpty { null }
    // A tactic's first hint ends on the question that sets the player looking the right way
    // ("which check leaves their king no square at all?"); never one that fits any position.
    if (!situationText.isNullOrEmpty() &&
        tacticalGoal &&
        !question.isNullOrEmpty() &&
        !forced &&
        question !in Situation.ANY_POSITION
    ) {
        situationText += " $question"
    }
    situationText = situationText?.let { if (it.isNotEmpty()) Situation.spoken(Situation.withinBudget(it)) else it }
    // The principle as the hint says it, when it says it: the coach leaves it out while it
    // cools down.
    var principleSaid = if (!principle.isNullOrEmpty() && !key.isNullOrEmpty()) {
        Situation.spoken(Situation.sentence(principle))
    } else {
        null
    }
    if (!principleSaid.isNullOrEmpty() && (situationText.isNullOrEmpty() || principleSaid !in situationText)) {
        principleSaid = null
    }
    // A quiet move's own reason, said at the move rung: the key fact it answers.
    if (why == null && !moveReason.isNullOrEmpty() && best != null) {
        why = "${board.san(best)}: ${moveReason.replaceFirstChar { it.lowercase() }}"
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
        goal = goal,
        theme = themeText,
        mechanism = mechanism,
        question = question,
        region = region,
        target = target,
        piece = piece,
        why = why,
        nudge = nudge,
        focus = focus,
        tactic = tactic,
        plan = plan,
        ideas = ideas,
        candidates = candidates,
        threats = threats,
        line = line,
        stakes = stakesText,
        key = key,
        trap = trap,
        choices = choices,
        situation = situationText,
        principle = principleSaid,
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
