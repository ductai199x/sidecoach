package com.sidecoach.analysis

import com.sidecoach.chess.BB_ALL
import com.sidecoach.chess.BB_EMPTY
import com.sidecoach.chess.BB_SQUARES
import com.sidecoach.chess.Board
import com.sidecoach.chess.KING
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.popcount
import com.sidecoach.chess.scanForward

/**
 * Who can take on a square, who can take back, and what a capture nets: by the rules. A port
 * of `sidecoach/analysis/exchange.py`.
 *
 * Every word the coach says about attack and defence -- "hanging", "defended", "safe", "can
 * be won", "attacks", "a real threat", "a fork", "a pin" -- comes down to these questions, and
 * `board.attackers()` answers them wrongly for a player: it counts a piece pinned to its own
 * king, which cannot take, and a king taking on a square the other side still guards. An
 * audit on 2026-09-23 reproduced 38 false sentences from that one gap ("their knight on e4 is
 * hanging, attacked by the knight on c3" with the knight pinned; "defended by the e8 king" of
 * f7 with the queen guarding it). So the answers here are read off legal moves: a capture is
 * a legal move, a recapture is a legal move on the board after the capture, and an exchange
 * is played out move by move, least valuable piece first -- which settles pins, checks, a
 * king that cannot take back, and the x-ray piece behind a capturer, with no special cases.
 *
 * `board.attackers()` stays right for king safety: a pinned piece still keeps the enemy king
 * off the squares it covers.
 *
 * The phone runs these per position, many times over (every detector of the tagger asks
 * them), so the moves onto one square are generated for that square only, as the desktop's
 * `_onto` does, and the exchange keeps the least valuable reply as it walks the generator
 * rather than listing them first -- the same first-of-the-cheapest `min()` picks.
 */
object Exchange {

    /** Standard 1/3/3/5/9, indexed by piece type; the king's 100 keeps comparisons right. */
    val VALUE = intArrayOf(0, 1, 3, 3, 5, 9, 100)

    /**
     * `board` with `colour` to move: the same position, the move handed over (no en passant
     * right carried across, no castling taken as a capture).
     */
    fun asMover(board: Board, colour: Boolean): Board {
        if (board.turn == colour) return board
        val view = board.copy(withStack = false)
        view.push(Move.NULL)
        return view
    }

    /** The legal moves of `colour` that take on `square` (were it `colour`'s move). */
    fun capturesOn(board: Board, colour: Boolean, square: Int): List<Move> =
        capturesOnView(asMover(board, colour), colour, square)

    /** [capturesOn] on a board that already has `colour` to move. */
    private fun capturesOnView(view: Board, colour: Boolean, square: Int): List<Move> {
        val target = view.colourAt(square)
        if (target == null || target == colour) return emptyList()
        return onto(view, square)
    }

    /**
     * The pieces of `colour` that can legally take on `square`: none pinned off the line to
     * their own king, the king only where it would not be taken back.
     */
    fun capturers(board: Board, colour: Boolean, square: Int): List<Int> {
        var from = BB_EMPTY
        for (m in capturesOn(board, colour, square)) from = from or BB_SQUARES[m.from]
        val out = ArrayList<Int>(popcount(from))
        scanForward(from) { out.add(it) }
        return out
    }

    /**
     * The pieces of `colour` that could legally take back on `square` were an enemy piece to
     * take there: none pinned to its own king off the line, and the king only when the capture
     * is the only enemy piece bearing on the square (a second one would guard it).
     */
    fun defenders(board: Board, colour: Boolean, square: Int): List<Int> {
        val enemy = board.attackersMask(!colour, square)
        val out = ArrayList<Int>()
        scanForward(board.attackersMask(colour, square)) { a ->
            if (board.pieceTypeAt(a) == KING) {
                if (popcount(enemy) <= 1) out.add(a)
            } else if (board.pinMask(colour, a) and BB_SQUARES[square] != BB_EMPTY) {
                out.add(a)
            }
        }
        return out
    }

    /**
     * What the capture `move` nets its side, in pawns, after the recaptures on its square
     * played out with legal moves, least valuable piece first, each side free to stop. A
     * non-capture scores 0. Promotions are counted as the pawn they were.
     */
    fun exchange(board: Board, move: Move): Int {
        val square = move.to
        val taken = if (board.isEnPassant(move)) {
            PAWN
        } else {
            val victim = board.pieceTypeAt(square)
            if (victim == 0) return 0
            victim
        }
        var onSquare = board.pieceTypeAt(move.from).let { if (it == 0) PAWN else it }
        // The gains, one per capture, grown as the exchange goes on: rarely past a handful.
        var gains = IntArray(8)
        var count = 0
        gains[count++] = VALUE[taken]
        val b = board.copy(withStack = false)
        b.push(move)
        while (true) {
            // The first of the cheapest capturers, as `min()` over the generated moves picks it.
            var reply: Move? = null
            var cheapest = Int.MAX_VALUE
            b.generateLegalMoves(BB_ALL, BB_SQUARES[square]) { m ->
                val cost = VALUE[b.pieceTypeAt(m.from).let { if (it == 0) PAWN else it }]
                if (cost < cheapest) {
                    cheapest = cost
                    reply = m
                }
            }
            val chosen = reply ?: break
            if (count == gains.size) gains = gains.copyOf(count * 2)
            gains[count] = VALUE[onSquare] - gains[count - 1]
            count += 1
            onSquare = b.pieceTypeAt(chosen.from).let { if (it == 0) PAWN else it }
            b.push(chosen)
        }
        while (count > 1) {
            val last = gains[--count]
            gains[count - 1] = -maxOf(-gains[count - 1], last)
        }
        return gains[0]
    }

    /**
     * The most `colour` nets by taking on `square` (were it `colour`'s move); 0 when it cannot
     * take or would lose by it.
     *
     * When the other side is to move and in check, it answers the check first: the most
     * `colour` nets after its best answer (0 when the piece steps away). Handing the move over
     * with a null move in check left every recapture illegal, and every defended piece of the
     * checked side read as loose.
     */
    fun bestCapture(board: Board, colour: Boolean, square: Int): Int {
        val replies = checkAnswers(board, !colour)
        if (replies != null) {
            var least = 0
            for ((i, b) in replies.withIndex()) {
                val nets = if (b.pieceAt(square) == null) 0 else bestCapture(b, colour, square)
                least = if (i == 0) nets else minOf(least, nets)
            }
            return least
        }
        val view = asMover(board, colour)
        var best = 0
        for (m in capturesOnView(view, colour, square)) best = maxOf(best, exchange(view, m))
        return best
    }

    /** The piece on `square` can be taken for profit by the other side. */
    fun canBeWon(board: Board, square: Int): Boolean {
        val piece = board.pieceAt(square) ?: return false
        if (piece.pieceType == KING) return false
        return bestCapture(board, !piece.colour, square) > 0
    }

    /**
     * The boards after each legal answer when `colour` is to move and in check; null
     * otherwise. A side in check cannot pass, so whether its piece can be taken is a question
     * about the board after its answer.
     */
    private fun checkAnswers(board: Board, colour: Boolean): List<Board>? {
        if (board.turn != colour || !board.isCheck()) return null
        return board.legalMoves().map { move -> board.copy(withStack = false).also { it.push(move) } }
    }

    /**
     * The piece on `square` is attacked by a legal capturer and nothing can take back.
     *
     * When its side is to move and in check, the piece hangs only if it does after every
     * answer to the check (one that moves the piece saves it).
     */
    fun hanging(board: Board, square: Int): Boolean {
        val piece = board.pieceAt(square) ?: return false
        if (piece.pieceType == KING) return false
        val replies = checkAnswers(board, piece.colour)
        if (replies != null) {
            return replies.isNotEmpty() &&
                replies.all { b -> b.pieceAt(square) == piece && hanging(b, square) }
        }
        val view = asMover(board, !piece.colour)
        for (move in capturesOnView(view, !piece.colour, square)) {
            val after = view.copy(withStack = false)
            after.push(move)
            if (!anyOnto(after, square)) return true
        }
        return false
    }

    /** After `capture`, the other side can take back on its square. */
    fun canRetake(board: Board, capture: Move): Boolean {
        val after = board.copy(withStack = false)
        after.push(capture)
        return anyOnto(after, capture.to)
    }

    /** The legal moves of the side to move onto `square`, generated for that square only. */
    private fun onto(board: Board, square: Int): List<Move> {
        val out = ArrayList<Move>(4)
        board.generateLegalMoves(BB_ALL, BB_SQUARES[square]) { out.add(it) }
        return out
    }

    private fun anyOnto(board: Board, square: Int): Boolean {
        var any = false
        board.generateLegalMoves(BB_ALL, BB_SQUARES[square]) { any = true }
        return any
    }
}
