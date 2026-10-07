package com.sidecoach.coach

import com.sidecoach.analysis.Exchange
import com.sidecoach.analysis.bestMoveIdea
import com.sidecoach.analysis.worstPieceSquare
import com.sidecoach.chess.BISHOP
import com.sidecoach.chess.Board
import com.sidecoach.chess.KING
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.chess.parseSquare
import com.sidecoach.chess.square
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareRank
import com.sidecoach.events.MotifTag
import kotlin.random.Random

/**
 * The word on a sound move, written from the facts rather than by the model. A port of
 * `sidecoach/coach/verdicts.py`.
 *
 * A best, great, excellent, good or book move that cost nothing gets a word of praise in
 * chess.com's vocabulary, then -- when the board shows one -- what the move teaches: the mate,
 * the tactic it makes (the tagger's tag, described only when every square it names still holds
 * the piece it claims), the mate it threatens, or the strategic idea behind it ([bestMoveIdea],
 * from the board's geometry), put as the principle to keep -- and the principle only when it is
 * the move's reason: never of a move that checks, mates along the line, makes a tactic, or
 * threatens something ([lesson]; Tai, 2026-09-27: "If it says something it should be
 * relevant"). Nothing else: no advice, no guess at what it allows, no numbers, no question.
 *
 * Never the move itself, nor what it plainly does -- the capture, the check, the castling: the
 * player has just played it and sees it on the board. Tai, 2026-09-23: "the coach shouldn't be
 * starting the message with the move either since that's redundant ... either say something
 * instructive or don't say anything at all is okay or just 'That's a good move'". The praise
 * never repeats the last one, and a lesson taught in one of the last few words waits its turn.
 *
 * Tai, 2026-09-20: "if I made the best move or great move, we can synthesize the response
 * without hitting the API". A routine call was 2.7k prompt tokens and a second's wait; this
 * is free and immediate, and every word of it is true of the board.
 */

/** The praise, by the move's label. What the move teaches follows it after " -- ". */
val OPENERS: Map<String, List<String>> = mapOf(
    "book" to listOf("Book", "Still theory", "Known territory"),
    "brilliant" to listOf("Brilliant", "What a find", "Yes, that's the shot"),
    "great" to listOf("Great move", "Yes, you found it", "That's it"),
    "best" to listOf("Best", "That's the move", "Spot on", "Exactly", "Just what the position wanted"),
    "excellent" to listOf("Excellent", "Strong", "Very good", "Nice"),
    "good" to listOf("Good", "Fine", "Solid", "Sound"),
)

/** A mate needs no label: the game is won. */
val MATE = listOf("Checkmate -- well played.", "And that's mate. Well played.", "Mate. Nicely done.")

/**
 * The tags worth a lesson, in the order one is chosen; the rest (a capture, a check, a quiet
 * move) the player sees on the board.
 */
val VERDICT_TACTICS = listOf(
    "fork",
    "skewer",
    "pin",
    "absolutePin",
    "discoveredAttack",
    "doubleCheck",
    "capturingDefender",
    "trappedPiece",
    "hangingPiece",
)

/**
 * The idea behind a move ([bestMoveIdea]'s kinds) as the principle it follows: what the player
 * should keep, not what the move did. Two wordings each.
 */
val PRINCIPLES: Map<String, List<String>> = mapOf(
    "castle" to listOf(
        "king safety first: a castled king is much harder to attack",
        "the king tucked away, and a rook on its way to the centre",
    ),
    "rescue" to listOf(
        "a piece in danger comes first",
        "getting the attacked piece to safety before anything else",
    ),
    "prophylaxis" to listOf(
        "it takes away your opponent's idea before it starts",
        "prophylaxis: stop their plan before pushing your own",
    ),
    "activateKing" to listOf(
        "in the endgame the king is a fighting piece, and it belongs in the centre",
        "in the endgame the king walks to the centre and fights",
    ),
    "develop" to listOf(
        "every piece you bring out is one more in the fight",
        "develop first, attack later",
    ),
    "outpostMove" to listOf(
        "a knight where no pawn can chase it is a strong piece",
        "an outpost: no enemy pawn can drive that knight away",
    ),
    "centralise" to listOf(
        "a piece in the centre reaches both sides of the board",
        "pieces work best from the middle",
    ),
    "takeFile" to listOf("rooks belong on open files", "an open file is a road for your rook"),
    "rookBehindPasser" to listOf(
        "rooks belong behind passed pawns",
        "behind the passed pawn, the rook gains room with every step the pawn takes",
    ),
    "break" to listOf(
        "a pawn break opens lines for your pieces",
        "challenging their pawns opens the game for your pieces",
    ),
    "centre" to listOf(
        "pawns in the centre give your pieces room",
        "the fight is for the centre, and you joined it",
    ),
    "pawnStorm" to listOf(
        "pawns thrown at their king pry open lines for the attack",
        "the pawn storm: open lines toward their king",
    ),
    "pushPasser" to listOf("passed pawns must be pushed", "a passed pawn gets stronger as it advances"),
    "activate" to listOf(
        "improving your worst piece is rarely wrong",
        "your most passive piece joins the game",
    ),
    "space" to listOf(
        "space on the flank cramps their pieces",
        "more space, less room for their pieces",
    ),
    "simplify" to listOf(
        "when you are ahead, trades bring the win closer",
        "ahead in material, every trade helps you",
    ),
    // Prophylaxis against a threat to take or to check: not their plan stopped before it
    // starts but their threat met ("prophylaxis" was said of Nc3 guarding e4 against Nxe4).
    "defend" to listOf("their threat comes first", "first their threat, then your plan"),
)

/** A move that threatens mate: the point is the mate, whatever else the move does. */
val MATE_THREAT = listOf("that threatens mate", "and now mate is threatened")

/**
 * The tags that say what a move is for when it is not a principle: the check, the mate along the
 * line, the tactic it makes or starts. A move that carries one is a move with a concrete point,
 * and a principle said of it was not the reason ("rooks belong on open files" of a rook check
 * that led to mate, "passed pawns must be pushed" of the push that mated: Tai, 2026-09-27). The
 * weaker geometric tags -- an interference, an x-ray, an exposed king -- come with castling and
 * rook lifts alike and do not count.
 */
val CONCRETE = setOf(
    "doubleCheck",
    "mate",
    "sacrifice",
    "deflection",
    "attraction",
    "clearance",
    "capturingDefender",
    "discoveredAttack",
    "fork",
    "skewer",
    "pin",
    "absolutePin",
    "attackingF2F7",
    "trappedPiece",
    "hangingPiece",
    "promotion",
    "underPromotion",
)

/** The kinds that are about a threat met: a move that saves a piece is their point. */
private val DEFENSIVE = listOf("rescue", "prophylaxis", "castle")

/** The pawn moves whose principle is about pieces or a king to attack. */
private val PAWN_MOVES = listOf("break", "centre", "space", "pawnStorm")

/**
 * The terms of art in the wordings above: said to a player below 1400, the simulated 800-rated
 * student listed "prophylaxis: stop their plan before pushing your own" among the things it did
 * not understand (2026-09-26).
 */
val JARGON = Regex("\\b(?:prophylaxis|outpost|pawn storm)\\b", RegexOption.IGNORE_CASE)

/** The opponent's first threat (the fact sheet's SAN, read after a pass), or null. */
private fun threatOf(board: Board, threats: List<String>): Move? {
    if (threats.isEmpty() || board.isCheck()) return null
    return nullMoved(board).parseSan(threats[0])
}

/**
 * The pinned piece can be won as the board stands: a pin worth a word after the move. A pawn
 * pinned to its king with nothing to take it is true and teaches nothing.
 */
private fun pinWins(after: Board, tag: MotifTag): Boolean {
    if (tag.squares.size < 2) return false
    return Exchange.canBeWon(after, parseSquare(tag.squares[1]))
}

/**
 * Their threat (read after a pass) takes something or gives check: a threat to meet, not a plan
 * to forestall. `verdicts._direct`.
 */
private fun direct(board: Board, threat: Move): Boolean {
    val passed = nullMoved(board)
    return passed.isCapture(threat) || passed.givesCheck(threat)
}

/**
 * After a move that gives no check, the mover would mate in one were it their move again.
 * `verdicts.threatens_mate`.
 */
fun threatensMate(after: Board): Boolean {
    if (after.isCheck() || after.isGameOver()) return false
    val passed = nullMoved(after)
    for (reply in passed.legalMoves()) {
        passed.push(reply)
        val mate = passed.isCheckmate()
        passed.pop()
        if (mate) return true
    }
    return false
}

/**
 * The move makes a threat to win a piece of theirs, a minor or more, that could not be won
 * before it: the threat is its point. `verdicts._threatens_piece`.
 */
private fun threatensPiece(board: Board, after: Board, us: Boolean): Boolean {
    for ((square, piece) in after.pieceMap()) {
        if (piece.colour == us || Exchange.VALUE[piece.pieceType] < 3) continue
        if (Exchange.canBeWon(after, square) && !Exchange.canBeWon(board, square)) return true
    }
    return false
}

/**
 * The piece that moved to [to], a bishop, rook or queen, stands on a line with two of their
 * pieces one behind the other, a minor or more each: a pin (the one behind worth more, or the
 * king) or a skewer (the one in front worth more). Tagged or not, the line is the move's point:
 * the annotator keeps a pin only when the line cashes it in. A pawn in front is a file's usual
 * traffic, not a pin. `verdicts._lines_up`.
 */
private fun linesUp(after: Board, to: Int, us: Boolean): Boolean {
    val piece = after.pieceAt(to)
    if (piece == null || piece.pieceType !in listOf(BISHOP, ROOK, QUEEN)) return false
    val steps = ArrayList<Pair<Int, Int>>()
    if (piece.pieceType != BISHOP) steps += listOf(0 to 1, 0 to -1, 1 to 0, -1 to 0)
    if (piece.pieceType != ROOK) steps += listOf(1 to 1, 1 to -1, -1 to 1, -1 to -1)
    for ((df, dr) in steps) {
        val seen = ArrayList<com.sidecoach.chess.Piece>(2)
        var f = squareFile(to)
        var r = squareRank(to)
        while (seen.size < 2) {
            f += df
            r += dr
            if (f !in 0..7 || r !in 0..7) break
            val other = after.pieceAt(square(f, r)) ?: continue
            if (other.colour == us) break
            seen.add(other)
        }
        if (seen.size < 2) continue
        val front = Exchange.VALUE[seen[0].pieceType]
        val back = Exchange.VALUE[seen[1].pieceType]
        if (front >= 3 && back >= 3 && front != back) return true
    }
    return false
}

/**
 * What a sound move teaches, as its kind and the ways to say it: the mate, the tactic a tag
 * names when the squares bear it out, the mate it threatens, or the idea behind the move as its
 * principle (two wordings). Null when the board shows none, or when its kind is in [taught]
 * (said in one of the last few words).
 *
 * A principle only when it is the move's reason: never of a move that checks, mates along the
 * line, makes or starts a tactic ([CONCRETE]), lines up on two of their pieces, or threatens
 * mate or one of their pieces -- that is its point, and said or not, the principle beside it was
 * noise (Tai, 2026-09-27: "If it says something it should be relevant").
 */
fun lesson(
    board: Board,
    move: Move,
    tags: List<MotifTag> = emptyList(),
    phase: String = "middlegame",
    threats: List<String> = emptyList(),
    taught: List<String> = emptyList(),
): Pair<String, List<String>>? {
    val after = board.copy(withStack = false)
    after.push(move)
    if (after.isCheckmate()) return "mate" to listOf("checkmate")
    for (tag in tags) {
        if (tag.name in VERDICT_TACTICS && tag.squares.isNotEmpty() && tag.name !in taught) {
            if ((tag.name == "pin" || tag.name == "absolutePin") && !pinWins(after, tag)) continue
            val said = describeTactic(after, tag.name, tag.squares)
            if (said != null) return tag.name to listOf(said)
        }
    }
    // A capture that wins something -- a recapture most often -- is about the material: "that
    // lets your bishop out" was said of bxc3 taking back a knight, "that threatens mate" of Qxd4
    // taking a queen.
    val won = Exchange.exchange(board, move)
    if (threatensMate(after) && won < 3) {
        return if ("mateThreat" in taught) null else "mateThreat" to MATE_THREAT
    }
    val us = board.turn
    if (after.isCheck() ||
        tags.any { it.name in CONCRETE || it.name.startsWith("mateIn") } ||
        linesUp(after, move.to, us) ||
        threatensPiece(board, after, us) ||
        won > 0
    ) {
        return null
    }
    val threat = threatOf(board, threats)
    val idea = bestMoveIdea(
        board,
        move,
        us,
        phase = phase,
        worst = worstPieceSquare(board, us),
        threat = threat,
    )
    if (idea == null || idea.kind !in PRINCIPLES) return null
    if (idea.kind !in DEFENSIVE && tags.any { it.name == "defensiveMove" }) {
        return null // it saves a piece: that is its point, not the rook's file
    }
    var kind = idea.kind
    if (kind == "prophylaxis" && threat != null && direct(board, threat)) kind = "defend"
    if (kind in taught) return null
    if (kind in PAWN_MOVES && board.pieceMap().values.all { it.pieceType == KING || it.pieceType == PAWN }) {
        return null // a pawn ending: no pieces for the lines to open, no king to storm
    }
    if (kind == "centre" || kind == "space" || kind == "centralise") {
        // A principle with nothing on the board behind it is no lesson ("the fight is for the
        // centre" after every pawn move, Tai, 2026-09-24): only what the move opens.
        val opened = Situation.shutIn(board, move) ?: return null
        val home = opened.first
        val piece = board.pieceTypeAt(home)
        val name = COACH_PIECE_NAMES[if (piece == 0) PAWN else piece]
        return kind to listOf("that lets your $name out")
    }
    return kind to PRINCIPLES.getValue(kind)
}

/**
 * The praise still available for this label: the ones [last] does not already begin with, or
 * all of them when that would leave nothing.
 */
fun praiseFor(label: String?, last: String?): List<String> {
    val openers = OPENERS[label ?: ""] ?: OPENERS.getValue("good")
    return openers.filter { last == null || !last.startsWith(it) }.ifEmpty { openers }
}

/**
 * The sentence -- the praise for the label, then what the move teaches -- and the kind of lesson
 * it taught, null when it is praise alone. With [bags] (the live coach's) every wording is drawn
 * from its bag, each once before any comes again ([Bags]); without, [pick] makes every draw, one
 * that is never [last]'s praise, as the desktop's `rng.choice` does. The praise is drawn before
 * the wording. [plain] leaves out the wordings that name a term of art (a player below 1400:
 * [JARGON]); [strong] (1800 and up) the principles -- "rooks belong on open files" was what the
 * strong simulated students most often said they already knew -- keeping the tactics, which are
 * the board's own.
 */
fun verdict(
    board: Board,
    move: Move,
    label: String?,
    tags: List<MotifTag> = emptyList(),
    phase: String = "middlegame",
    threats: List<String> = emptyList(),
    taught: List<String> = emptyList(),
    last: String? = null,
    pick: (List<String>) -> String = { it[Random.nextInt(it.size)] },
    bags: Bags? = null,
    plain: Boolean = false,
    strong: Boolean = false,
): Pair<String, String?> {
    var found = lesson(board, move, tags, phase, threats, taught)
    if (strong && found != null && found.first in PRINCIPLES) found = null
    if (found != null && found.first == "mate") {
        if (bags != null) return bags.draw("mate", MATE) to "mate"
        return pick(MATE.filter { it != last }.ifEmpty { MATE }) to "mate"
    }
    val opener = if (bags != null) {
        val praise = if (label.isNullOrEmpty()) "good" else label
        bags.draw("praise:$praise", OPENERS[label ?: ""] ?: OPENERS.getValue("good"))
    } else {
        pick(praiseFor(label, last))
    }
    if (found == null) return "$opener." to null
    val kind = found.first
    var clauses = found.second
    if (plain) clauses = clauses.filter { !JARGON.containsMatchIn(it) }.ifEmpty { clauses }
    val clause = if (bags != null) {
        bags.draw("clause:$kind", clauses)
    } else if (clauses.size > 1) {
        pick(clauses)
    } else {
        clauses[0]
    }
    return "$opener -- $clause." to kind
}
