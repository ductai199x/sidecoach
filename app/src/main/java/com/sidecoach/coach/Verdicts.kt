package com.sidecoach.coach

import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareName
import com.sidecoach.events.MotifTag
import kotlin.random.Random

/**
 * The word on a sound move, written from the facts rather than by the model. A port of
 * `sidecoach/coach/verdicts.py`.
 *
 * A best, great, excellent, good or book move that cost nothing gets a sentence: the verdict
 * in chess.com's vocabulary, then what the move does when the board says so -- the capture,
 * the check, the castling, the promotion, the tactic it makes (the tagger's tag, described
 * only when every square it names still holds the piece it claims). Nothing else: no advice,
 * no guess at what it allows, no numbers, no question. The opener never repeats the last one.
 *
 * Tai, 2026-09-20: "if I made the best move or great move, we can synthesize the response
 * without hitting the API". A routine call was 2.7k prompt tokens and a second's wait; this
 * is free and immediate, and every word of it is true of the board.
 */

/** The verdict, by the move's label; "{san}" is the move. */
val OPENERS: Map<String, List<String>> = mapOf(
    "book" to listOf("{san}: book", "{san}, still theory", "{san}: known territory"),
    "brilliant" to listOf("{san}!! -- brilliant", "{san} -- what a find", "{san}: yes, that's the shot"),
    "great" to listOf("{san}: great move", "{san} -- yes, the one move that works", "{san}! -- that's it"),
    "best" to listOf(
        "{san}: best",
        "{san} -- that's the move",
        "{san}, spot on",
        "{san}: exactly",
        "{san}: the engine's choice too",
    ),
    "excellent" to listOf("{san}: excellent", "{san}: strong", "{san} -- very good", "{san}: nice"),
    "good" to listOf("{san}: good", "{san}: fine", "{san}, solid", "{san} works", "{san}: sound"),
)

/**
 * The tags worth a clause, in the order one is chosen; the rest (a capture, a check, a quiet
 * move) are said by the move itself.
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
 * A clause on the move, from the board: the mate, the check, the capture, the castling, the
 * promotion, or the tactic a tag names when the squares bear it out; null when the move is
 * quiet and no tag holds.
 */
fun whatItDoes(board: Board, move: Move, tags: List<MotifTag>): String? {
    val after = board.copy(withStack = false)
    after.push(move)
    if (after.isCheckmate()) return "checkmate"
    for (tag in tags) {
        if (tag.name in VERDICT_TACTICS && tag.squares.isNotEmpty()) {
            val said = describeTactic(after, tag.name, tag.squares)
            if (said != null) return said
        }
    }
    if (board.isCastling(move)) {
        val side = if (squareFile(move.to) > 4) "kingside" else "queenside"
        return "castling $side"
    }
    val parts = ArrayList<String>()
    if (board.isCapture(move)) {
        val taken = board.pieceAt(move.to)
        val what = when {
            board.isEnPassant(move) -> "the pawn en passant"
            taken != null -> "the ${COACH_PIECE_NAMES[taken.pieceType]}"
            else -> "a piece"
        }
        parts.add("taking $what on ${squareName(move.to)}")
    }
    if (move.promotion != 0) parts.add("promoting to a ${COACH_PIECE_NAMES[move.promotion]}")
    if (after.isCheck()) parts.add("with check")
    return if (parts.isNotEmpty()) parts.joinToString(", ") else null
}

/**
 * The openers still available for this label: the ones whose sentence `last` does not already
 * begin with, or all of them when that would leave nothing.
 */
fun openersFor(san: String, label: String?, last: String?): List<String> {
    val openers = OPENERS[label ?: ""] ?: OPENERS.getValue("good")
    val choices = openers.filter { last == null || !last.startsWith(it.replace("{san}", san)) }
    return choices.ifEmpty { openers }
}

/**
 * The sentence: the opener for the label (never `last`'s), then what the move does. `pick`
 * chooses among the openers still available, so a test can fix the draw.
 */
fun verdict(
    board: Board,
    move: Move,
    label: String?,
    tags: List<MotifTag> = emptyList(),
    last: String? = null,
    pick: (List<String>) -> String = { it[Random.nextInt(it.size)] },
): String {
    val san = board.san(move)
    val opener = pick(openersFor(san, label, last)).replace("{san}", san)
    val clause = whatItDoes(board, move, tags) ?: return "$opener."
    if (clause == "checkmate") return "$opener -- checkmate."
    val joiner = if (clause.startsWith("the ")) " -- " else ", "
    return "$opener$joiner$clause."
}
