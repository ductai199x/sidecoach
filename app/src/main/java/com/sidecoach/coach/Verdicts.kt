package com.sidecoach.coach

import com.sidecoach.analysis.Exchange
import com.sidecoach.analysis.bestMoveIdea
import com.sidecoach.analysis.worstPieceSquare
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.parseSquare
import com.sidecoach.events.MotifTag
import kotlin.random.Random

/**
 * The word on a sound move, written from the facts rather than by the model. A port of
 * `sidecoach/coach/verdicts.py`.
 *
 * A best, great, excellent, good or book move that cost nothing gets a word of praise in
 * chess.com's vocabulary, then -- when the board shows one -- what the move teaches: the mate,
 * the tactic it makes (the tagger's tag, described only when every square it names still holds
 * the piece it claims), or the strategic idea behind it ([bestMoveIdea], from the board's
 * geometry), put as the principle to keep. Nothing else: no advice, no guess at what it
 * allows, no numbers, no question.
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
    "great" to listOf("Great move", "Yes, the one move that works", "That's it"),
    "best" to listOf("Best", "That's the move", "Spot on", "Exactly", "The engine agrees"),
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
)

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
 * What a sound move teaches, as its kind and the ways to say it: the mate, the tactic a tag
 * names when the squares bear it out, or the idea behind the move as its principle (two
 * wordings). Null when the board shows none, or when its kind is in [taught] (said in one of
 * the last few words).
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
    val us = board.turn
    val idea = bestMoveIdea(
        board,
        move,
        us,
        phase = phase,
        worst = worstPieceSquare(board, us),
        threat = threatOf(board, threats),
    )
    if (idea == null || idea.kind !in PRINCIPLES || idea.kind in taught) return null
    if (idea.kind == "centre" || idea.kind == "space" || idea.kind == "centralise") {
        // A principle with nothing on the board behind it is no lesson ("the fight is for the
        // centre" after every pawn move, Tai, 2026-09-24): only what the move opens.
        val opened = Situation.shutIn(board, move) ?: return null
        val home = opened.first
        val kind = board.pieceTypeAt(home)
        val name = COACH_PIECE_NAMES[if (kind == 0) PAWN else kind]
        return idea.kind to listOf("that lets your $name out")
    }
    return idea.kind to PRINCIPLES.getValue(idea.kind)
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
 * [JARGON]).
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
): Pair<String, String?> {
    val found = lesson(board, move, tags, phase, threats, taught)
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
