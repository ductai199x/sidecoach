package com.sidecoach.coach

import com.sidecoach.chess.BISHOP
import com.sidecoach.chess.Board
import com.sidecoach.chess.KING
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.chess.square
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareName
import com.sidecoach.chess.squareRank
import com.sidecoach.chess.squaresOf

/**
 * What stands on the board right now, in the words a coach uses: `sidecoach/coach/inventory.py`.
 *
 * Which pieces hang, which are attacked or pinned, the checks and captures on offer, the
 * batteries and the forks in place. Computed on the board, worded from the player's side, so
 * the model reads the tactical state instead of working it out square by square -- the one
 * thing a language model does worst with a chess position.
 *
 * Everything here is a static fact of the position: no search, no judgement of whether a
 * capture is good. The engine's lines say what to do; this says what is there.
 */

private val INV_VALUE = intArrayOf(0, 1, 3, 3, 5, 9, 100)
private val SLIDERS = intArrayOf(BISHOP, ROOK, QUEEN)

/** Entries per list, most valuable first. */
const val INVENTORY_LIMIT = 8

fun nameOf(board: Board, sq: Int): String {
    val piece = board.pieceAt(sq)
    val word = if (piece != null) COACH_PIECE_NAMES[piece.pieceType] else "piece"
    return "$word on ${squareName(sq)}"
}

private fun sign(n: Int): Int = (if (n > 0) 1 else 0) - (if (n < 0) 1 else 0)

/**
 * (pinner, shielded) when the piece on `square` is pinned: an enemy slider attacks it along a
 * line and the next piece past it is a friendly piece worth more (the king most).
 */
fun pinTarget(board: Board, sq: Int): Pair<Int, Int>? {
    val piece = board.pieceAt(sq) ?: return null
    if (piece.pieceType == KING) return null
    for (attacker in board.attackers(!piece.colour, sq)) {
        val slider = board.pieceAt(attacker) ?: continue
        if (slider.pieceType !in SLIDERS.toList()) continue
        val df = sign(squareFile(sq) - squareFile(attacker))
        val dr = sign(squareRank(sq) - squareRank(attacker))
        var file = squareFile(sq) + df
        var rank = squareRank(sq) + dr
        while (file in 0..7 && rank in 0..7) {
            val behind = square(file, rank)
            val occupant = board.pieceAt(behind)
            if (occupant != null) {
                if (occupant.colour == piece.colour &&
                    INV_VALUE[occupant.pieceType] > INV_VALUE[piece.pieceType]
                ) {
                    return attacker to behind
                }
                break
            }
            file += df
            rank += dr
        }
    }
    return null
}

/**
 * (front, back) when the slider on `square` skewers: it attacks an enemy piece and, past it on
 * the same line, a less valuable enemy piece.
 */
fun skewerTarget(board: Board, sq: Int): Pair<Int, Int>? {
    val piece = board.pieceAt(sq) ?: return null
    if (piece.pieceType !in SLIDERS.toList()) return null
    for (front in board.attacks(sq)) {
        val target = board.pieceAt(front) ?: continue
        if (target.colour == piece.colour) continue
        val df = sign(squareFile(front) - squareFile(sq))
        val dr = sign(squareRank(front) - squareRank(sq))
        var file = squareFile(front) + df
        var rank = squareRank(front) + dr
        while (file in 0..7 && rank in 0..7) {
            val behind = square(file, rank)
            val occupant = board.pieceAt(behind)
            if (occupant != null) {
                if (occupant.colour != piece.colour &&
                    INV_VALUE[occupant.pieceType] < INV_VALUE[target.pieceType]
                ) {
                    return front to behind
                }
                break
            }
            file += df
            rank += dr
        }
    }
    return null
}

/**
 * Aligned sliders of one colour with nothing between: a rook behind a rook on a file, a queen
 * behind a bishop on a diagonal. Each battery once, squares in board order.
 */
fun batteries(board: Board, colour: Boolean): List<List<Int>> {
    val found = ArrayList<List<Int>>()
    val seen = HashSet<Set<Int>>()
    val sliders = (0 until 64).filter { sq ->
        val p = board.pieceAt(sq)
        p != null && p.colour == colour && p.pieceType in SLIDERS.toList()
    }
    for (a in sliders) {
        val pa = board.pieceAt(a)!!
        for (b in sliders) {
            if (b <= a) continue
            val pb = board.pieceAt(b)!!
            val df = sign(squareFile(b) - squareFile(a))
            val dr = sign(squareRank(b) - squareRank(a))
            val straight = df == 0 || dr == 0
            val diagonal = kotlin.math.abs(squareFile(b) - squareFile(a)) ==
                kotlin.math.abs(squareRank(b) - squareRank(a))
            if (!(straight || diagonal)) continue
            val kindsOk = listOf(pa, pb).all { p ->
                p.pieceType == QUEEN ||
                    (straight && p.pieceType == ROOK) ||
                    (diagonal && p.pieceType == BISHOP)
            }
            if (!kindsOk) continue
            var clear = true
            var file = squareFile(a) + df
            var rank = squareRank(a) + dr
            while (square(file, rank) != b) {
                if (board.pieceAt(square(file, rank)) != null) { clear = false; break }
                file += df
                rank += dr
            }
            val key = setOf(a, b)
            if (clear && key !in seen) {
                seen.add(key)
                found.add(listOf(a, b))
            }
        }
    }
    return found
}

/**
 * Each piece of `colour` that attacks two or more enemy units right now, the most valuable
 * targets first; the double attacks worth most first.
 */
fun forksInPlace(board: Board, colour: Boolean): List<Pair<Int, List<Int>>> {
    val out = ArrayList<Pair<Int, List<Int>>>()
    for (sq in 0 until 64) {
        val piece = board.pieceAt(sq) ?: continue
        if (piece.colour != colour) continue
        val hits = board.attacks(sq).filter { t ->
            val o = board.pieceAt(t)
            o != null && o.colour != colour
        }
        if (hits.size >= 2) {
            out.add(sq to hits.sortedByDescending { INV_VALUE[typeOrPawn(board, it)] })
        }
    }
    return out.sortedByDescending { entry -> entry.second.sumOf { INV_VALUE[typeOrPawn(board, it)] } }
}

private fun typeOrPawn(board: Board, sq: Int): Int {
    val t = board.pieceTypeAt(sq)
    return if (t == 0) PAWN else t
}

private fun piecesOf(board: Board, colour: Boolean): List<Int> =
    (0 until 64)
        .filter { sq ->
            val p = board.pieceAt(sq)
            p != null && p.colour == colour && p.pieceType != KING
        }
        .sortedByDescending { INV_VALUE[typeOrPawn(board, it)] }

private fun hitsBy(board: Board, sq: Int, colour: Boolean): List<Int> =
    board.attackers(colour, sq).sortedBy { INV_VALUE[typeOrPawn(board, it)] }

/**
 * The state of `colour`'s pieces: hanging (attacked, undefended), attacked by a lesser piece,
 * pinned; and `colour`'s batteries and forks in place. `own`/`their` are the possessives from
 * the reader's side ("your"/"their").
 */
fun sideLines(board: Board, colour: Boolean, own: String, their: String): List<String> {
    val out = ArrayList<String>()
    val hanging = ArrayList<String>()
    val attacked = ArrayList<String>()
    val pinned = ArrayList<String>()
    for (s in piecesOf(board, colour)) {
        val piece = board.pieceAt(s)!!
        val attackers = hitsBy(board, s, !colour)
        val defenders = board.attackers(colour, s)
        if (attackers.isNotEmpty() && defenders.isEmpty()) {
            hanging.add("${nameOf(board, s)} (attacked by the ${nameOf(board, attackers[0])})")
        } else if (attackers.isNotEmpty() &&
            INV_VALUE[typeOrPawn(board, attackers[0])] < INV_VALUE[piece.pieceType]
        ) {
            attacked.add("${nameOf(board, s)} by the ${nameOf(board, attackers[0])}")
        }
        val pin = pinTarget(board, s)
        if (pin != null) {
            pinned.add(
                "${nameOf(board, s)} to the ${nameOf(board, pin.second)} " +
                    "by the ${nameOf(board, pin.first)}",
            )
        }
    }
    if (hanging.isNotEmpty()) {
        out.add("$own hanging: " + hanging.take(INVENTORY_LIMIT).joinToString("; "))
    }
    if (attacked.isNotEmpty()) {
        out.add("$own attacked by a lesser piece: " + attacked.take(INVENTORY_LIMIT).joinToString("; "))
    }
    if (pinned.isNotEmpty()) {
        out.add("$own pinned: " + pinned.take(INVENTORY_LIMIT).joinToString("; "))
    }
    val forks = forksInPlace(board, colour)
    if (forks.isNotEmpty()) {
        out.add(
            "$own double attacks: " + forks.take(4).joinToString("; ") { (s, hits) ->
                "${nameOf(board, s)} hits the " +
                    hits.take(3).joinToString(" and the ") { nameOf(board, it) }
            },
        )
    }
    val skewers = piecesOf(board, colour).mapNotNull { s ->
        val sk = skewerTarget(board, s)
        if (sk == null) null else s to sk
    }
    if (skewers.isNotEmpty()) {
        out.add(
            "$own skewers: " + skewers.take(INVENTORY_LIMIT).joinToString("; ") { (s, pair) ->
                "${nameOf(board, s)} through the ${nameOf(board, pair.first)} " +
                    "to the ${nameOf(board, pair.second)}"
            },
        )
    }
    val bats = batteries(board, colour)
    if (bats.isNotEmpty()) {
        out.add(
            "$own batteries: " + bats.take(INVENTORY_LIMIT).joinToString("; ") { b ->
                b.joinToString(" and ") { nameOf(board, it) }
            },
        )
    }
    return out
}

/** What the side to move can do at once: checks and captures, in SAN, with what is taken. */
fun toMoveLines(board: Board, own: String): List<String> {
    val out = ArrayList<String>()
    val checks = ArrayList<String>()
    for (m in board.legalMoves()) {
        if (!board.givesCheck(m)) continue
        val after = board.copy(withStack = false)
        after.push(m)
        val checkers = squaresOf(after.checkersMask())
        val kind = when {
            checkers.size > 1 -> " (double check)"
            m.to !in checkers -> " (discovered check)"
            else -> ""
        }
        checks.add(board.san(m) + kind)
    }
    if (checks.isNotEmpty()) out.add("$own checks: " + checks.take(10).joinToString(", "))
    val caps = ArrayList<Pair<Int, String>>()
    for (m in board.legalMoves()) {
        if (!board.isCapture(m)) continue
        val victim = board.pieceAt(m.to)
        if (victim == null && board.isEnPassant(m)) {
            caps.add(1 to "${board.san(m)} (a pawn, en passant)")
            continue
        }
        if (victim == null) continue
        val defended = board.attackers(!board.turn, m.to)
        caps.add(
            INV_VALUE[victim.pieceType] to
                "${board.san(m)} (the ${COACH_PIECE_NAMES[victim.pieceType]}, " +
                "${if (defended.isNotEmpty()) "defended" else "undefended"})",
        )
    }
    if (caps.isNotEmpty()) {
        val sorted = caps.sortedByDescending { it.first }
        out.add("$own captures: " + sorted.take(10).joinToString(", ") { it.second })
    }
    return out
}

/**
 * The player's pieces, the opponent's, and what the side to move can do, one list per line;
 * one line saying so when nothing hangs, is pinned or is on offer.
 */
fun inventoryLines(board: Board, userColour: Boolean): List<String> {
    val mine = sideLines(board, userColour, "The player's", "the opponent's")
    val theirs = sideLines(board, !userColour, "The opponent's", "the player's")
    val mover = if (board.turn == userColour) "The player's" else "The opponent's"
    val lines = mine + theirs + toMoveLines(board, mover)
    return lines.ifEmpty { listOf("Nothing hangs, nothing is pinned, no check or capture is on.") }
}

/** The inventory as one block, for a message that is not sectioned. */
fun inventoryText(board: Board, userColour: Boolean): String {
    val lines = inventoryLines(board, userColour)
    if (lines.size == 1 && lines[0].startsWith("Nothing")) {
        return "On the board now: nothing hangs, nothing is pinned, no check or capture is on."
    }
    return "On the board now (complete lists): " + lines.joinToString(" | ") + "."
}
