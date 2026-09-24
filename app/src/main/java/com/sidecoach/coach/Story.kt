package com.sidecoach.coach

import com.sidecoach.analysis.Exchange
import com.sidecoach.analysis.isPassed
import com.sidecoach.chess.BB_EMPTY
import com.sidecoach.chess.BB_SQUARES
import com.sidecoach.chess.BISHOP
import com.sidecoach.chess.Board
import com.sidecoach.chess.KING
import com.sidecoach.chess.KNIGHT
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.square
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareName
import com.sidecoach.chess.squareRank
import com.sidecoach.chess.squaresOf
import kotlin.math.abs

/**
 * The story of the best line, told in the order a player meets it. A port of
 * `sidecoach/coach/story.py`.
 *
 * Tai, 2026-09-24, after the puzzles were read by hand: the first hint's parts -- what the line
 * wins, which tactic to name, which piece is the target, which plan -- were each chosen on
 * their own, and nothing made them agree with the move to find. "You can win their queen. Their
 * rook on f5, on a line one of your pieces is blocking" when a pawn simply takes the queen;
 * "their bishop on f4 is the target" to a player in check from a queen that can be taken. The
 * hints are told here as one story, the most immediate beat first:
 *
 * - in check, the piece giving check;
 * - a piece that simply hangs -- the capture on its own nets what the whole line does -- said as
 *   that, with no pattern beside it;
 * - the pawn about to queen, where the win rests on it;
 * - the prize, where it stands, when the pattern that wins it names other pieces;
 * - where only one or a few moves hold, what the move does: takes, checks, attacks, stops their
 *   idea -- never "The move is Bb5+." alone.
 *
 * The desktop's `tools/puzzles/story_checks.py` holds the hints to each beat over the Lichess
 * puzzles and the master games, read from the board and the engine's line, never from this code.
 */
object Story {

    private val NAMES = mapOf(
        PAWN to "pawn",
        KNIGHT to "knight",
        BISHOP to "bishop",
        ROOK to "rook",
        QUEEN to "queen",
        KING to "king",
    )
    private val TIMES = mapOf(1 to "once", 2 to "twice", 3 to "three times", 4 to "four times")

    /**
     * A free first capture worth this much (a minor piece) is the story's first beat even when
     * the line wins more after it.
     */
    const val FREE_MIN = 3
    private val NUMBER = mapOf(1 to "one square", 2 to "two squares", 3 to "three squares", 4 to "four squares")

    private fun sq(square: Int): String = squareName(square)

    private fun named(board: Board, square: Int): String {
        val piece = board.pieceAt(square) ?: return sq(square)
        return "${NAMES.getValue(piece.pieceType)} on ${sq(square)}"
    }

    /** A piece that simply hangs, as the three rungs say it. */
    data class Plain(
        val square: Int,
        /** The capture is the whole win; else the line wins more after it. */
        val whole: Boolean,
        /** The first hint: "Their queen on g4 can simply be taken: nothing defends it". */
        val key: String,
        /** The second hint's question. */
        val nudge: String,
        /** The third: "fxg4 takes their queen: nothing defends it". */
        val why: String,
    )

    /**
     * The first move takes a piece that simply hangs: the exchange on its square nets, on its
     * own, what the whole line wins (gNzJg: fxg4 and the queen is gone, no discovered attack
     * about it) -- or nets the whole piece, a minor one or more, and the line wins more after it
     * (DWvS7: Qxe8+ takes a free rook, and then the knight on e4 is loose).
     */
    fun plainCapture(board: Board, idea: Idea): Plain? {
        val first = idea.first
        if (idea.goal != "win" || first == null || !board.isLegal(first)) return null
        if (!board.isCapture(first) || board.isEnPassant(first)) return null
        val net = Exchange.exchange(board, first)
        val square = first.to
        val victim = board.pieceAt(square) ?: return null
        val whole = net >= maxOf(idea.gain, 1)
        val value = Exchange.VALUE[victim.pieceType]
        val free = net >= value && value >= FREE_MIN
        if (!(whole || free)) return null
        val us = board.turn
        val guards = Exchange.defenders(board, victim.colour, square)
        val hitters = Exchange.capturers(board, us, square)
        val cheapest = hitters.minOfOrNull { s ->
            val kind = board.pieceTypeAt(s)
            Exchange.VALUE[if (kind == 0) PAWN else kind]
        } ?: 0
        val worth = Exchange.VALUE[victim.pieceType]
        val reason = if (guards.isEmpty()) {
            "nothing defends it"
        } else if (cheapest < worth) {
            "it is worth more than what can take it"
        } else if (hitters.size > guards.size) {
            "it is attacked ${TIMES[hitters.size] ?: hitters.size.toString()} and defended " +
                (TIMES[guards.size] ?: guards.size.toString())
        } else {
            "the trades on its square come out in your favour"
        }
        val name = NAMES.getValue(victim.pieceType)
        val checker = board.isCheck() && board.checkersMask() and BB_SQUARES[square] != BB_EMPTY
        // "Back" only for what their last move took with: never guessed from the material count
        // ("their rook on h7 can simply be taken back" of a rook that had only moved there).
        val back = if (took(board, square)) " back" else ""
        var key = if (checker) {
            "Their $name on ${sq(square)} gives check, and it can be taken$back: $reason"
        } else {
            "Their $name on ${sq(square)} can simply be taken$back: $reason"
        }
        // They take something back later in the line: what is left is the win the goal names
        // (Nxg5 takes the queen, and after Bxe2 Nxe2 a bishop is what remains).
        if (net > idea.gain + 1 && !idea.won.isNullOrEmpty() && " for " !in idea.won) {
            key += "; after what they take back, you are ${idea.won} up"
        }
        return Plain(
            square = square,
            whole = whole,
            key = key,
            nudge = "Before you take, look at what can take back: is it safe?",
            why = "${board.san(first)} takes their $name: $reason${forcingReply(board, idea)}",
        )
    }

    /**
     * What their last move took, in pawns (0 when it took nothing or the board does not know its
     * moves): what the player wins next, up to that, is won back.
     */
    fun tookLast(board: Board): Int {
        if (board.moveStack.isEmpty()) return 0
        val before = board.copy()
        val last = before.pop()
        if (!before.isCapture(last)) return 0
        if (before.isEnPassant(last)) return Exchange.VALUE[PAWN]
        val kind = before.pieceTypeAt(last.to)
        return Exchange.VALUE[if (kind == 0) PAWN else kind]
    }

    /** Whether their last move captured on `square`, when the board knows its moves. */
    private fun took(board: Board, square: Int): Boolean {
        if (board.moveStack.isEmpty()) return false
        val last = board.moveStack.last()
        if (last.to != square) return false
        val before = board.copy()
        before.pop()
        return before.isCapture(last)
    }

    /**
     * Their reply and the player's answer, when the reply is a check or a capture: the
     * in-between move a plain capture must still meet.
     */
    private fun forcingReply(board: Board, idea: Idea): String {
        val line = idea.line.take(3)
        if (line.size < 3) return ""
        val b = board.copy(withStack = false)
        val said = ArrayList<String>()
        var forcing = false
        for ((j, move) in line.withIndex()) {
            if (!b.isLegal(move)) return ""
            if (j == 1) forcing = b.isCapture(move) || b.givesCheck(move)
            said.add(b.san(move))
            b.push(move)
        }
        return if (forcing) "; after ${said[1]}, ${said[2]}" else ""
    }

    /** The boards along `line` from `board`, to its first illegal move. */
    private fun boardsAlong(board: Board, line: List<Move>): List<Board> {
        val boards = arrayListOf(board.copy(withStack = false))
        for (move in line) {
            if (!boards.last().isLegal(move)) break
            val after = boards.last().copy(withStack = false)
            after.push(move)
            boards.add(after)
        }
        return boards
    }

    /**
     * The player's pawn the line pushes to the seventh or promotes, where the point rests on it:
     * "your pawn on f6 is two squares from queening" (uSAIW: f7, and the rook must give itself
     * up).
     */
    fun queening(board: Board, idea: Idea): String? {
        if (idea.goal !in listOf("win", "combination", "mate", "attack") || idea.line.isEmpty()) return null
        val boards = boardsAlong(board, idea.line)
        val line = idea.line.take(boards.size - 1)
        val last = if (idea.goal == "mate" || idea.goal == "attack") line.size else idea.gainPly + 1
        val us = board.turn
        val seventh = if (us == WHITE) 6 else 1
        for ((j, move) in line.take(last).withIndex()) {
            if (j % 2 != 0 || boards[j].pieceTypeAt(move.from) != PAWN) continue
            // Pushed now, or queening in the line: a pawn that only reaches the seventh later is
            // not the point, and naming it sent the player to push it (25.e6?? in Carlsen-Chao).
            if (!(move.promotion != 0 || (j == 0 && squareRank(move.to) == seventh))) continue
            val start = IdeaReader.origin(board, boards, j, move.from)
            if (start == null || board.pieceTypeAt(start) != PAWN) return null
            val rank = squareRank(start)
            val toGo = if (us == WHITE) 7 - rank else rank
            return "Your pawn on ${sq(start)} is ${NUMBER[toGo] ?: "$toGo squares"} from queening"
        }
        return null
    }

    /**
     * The piece the line wins, where it stands, when what is `said` does not name it: "the prize
     * is their queen on c7" beside a discovered attack whose line names their knight.
     */
    fun prize(board: Board, idea: Idea, said: String): String? {
        if ((idea.goal != "win" && idea.goal != "combination") || idea.target == null) return null
        val piece = board.pieceAt(idea.target) ?: return null
        if (piece.colour == board.turn || piece.pieceType == KING) return null
        if (namedAnyWon(board, idea, said)) return null
        return "The prize is their ${named(board, idea.target)}"
    }

    /** Whether `said` names a piece of the kind the line wins that it takes on the way. */
    private fun namedAnyWon(board: Board, idea: Idea, said: String): Boolean {
        val target = idea.target ?: return false
        val top = board.pieceTypeAt(target)
        if (top == 0) return false
        val boards = boardsAlong(board, idea.line)
        for ((j, move) in idea.line.take(boards.size - 1).withIndex()) {
            if (j % 2 != 0) continue
            val taken = boards[j].pieceAt(move.to)
            if (taken == null || taken.colour == board.turn || taken.pieceType != top) continue
            val start = IdeaReader.origin(board, boards, j, move.to)
            if (start != null && "${NAMES.getValue(top)} on ${sq(start)}" in said) return true
        }
        return "${NAMES.getValue(top)} on ${sq(target)}" in said
    }

    /**
     * `story.trapped`: the piece on [square], its side to move, is a trapped piece -- attacked
     * for profit where it stands, and every move of it lands where it is won too, or takes
     * something for less than itself. Tai's chess.com puzzle, 2026-09-24: Bg4 against a queen on
     * h5 that had nowhere to go was hinted as "the first move attacks their queen", not as the
     * trapped piece the puzzle teaches.
     */
    fun trapped(board: Board, square: Int): Boolean {
        val piece = board.pieceAt(square) ?: return false
        if (piece.pieceType == KING || piece.pieceType == PAWN) return false
        if (!Exchange.canBeWon(board, square)) return false
        for (move in board.legalMoves()) {
            if (move.from != square) continue
            if (board.isCapture(move)) {
                if (Exchange.exchange(board, move) >= 0) return false  // it escapes by taking something worth it
                continue
            }
            val after = board.copy(withStack = false)
            after.push(move)
            if (!Exchange.canBeWon(after, move.to)) return false
        }
        return true
    }

    /**
     * What a move does, in words for the third hint and a question for the second, and the
     * pattern's lesson when the move carries one.
     */
    data class Effect(
        /** "gives check", "takes their pawn on h5", "stops Nd6+". */
        val said: String,
        /** "Which check is it?". */
        val ask: String,
        val lesson: String? = null,
    )

    /**
     * What `move` does on the board, the most forcing first: a capture, a check, an attack on a
     * piece of theirs it did not hit before, their idea stopped, a passed pawn pushed.
     */
    fun effects(board: Board, move: Move, threat: Move?): List<Effect> {
        if (!board.isLegal(move)) return emptyList()
        val us = board.turn
        val out = ArrayList<Effect>()
        val after = board.copy(withStack = false)
        after.push(move)
        if (board.isCapture(move)) {
            var square = move.to
            if (board.isEnPassant(move)) square = square(squareFile(move.to), squareRank(move.from))
            out.add(Effect("takes their ${named(board, square)}", "What can you take?"))
        }
        if (after.isCheck()) out.add(Effect("gives check", "Which check is it?"))
        val before = board.attacksMask(move.from)
        for (square in squaresOf(after.attacksMask(move.to) and before.inv())) {
            val piece = after.pieceAt(square) ?: continue
            if (piece.colour == us || piece.pieceType == KING) continue
            val loose = Exchange.defenders(after, piece.colour, square).isEmpty()
            if (trapped(after, square)) {
                out.add(
                    Effect(
                        "attacks their ${named(after, square)}, and it has no safe square to go to",
                        "Which of their pieces has no safe square left?",
                        "${IdeaWords.PATTERN_NAMES.getValue("trappedPiece")}: " +
                            IdeaWords.PATTERN_WHY.getValue("trappedPiece"),
                    ),
                )
                break
            }
            if (piece.pieceType == QUEEN || piece.pieceType == ROOK || (loose && piece.pieceType != PAWN)) {
                out.add(Effect("attacks their ${named(after, square)}", "What can it attack?"))
                break
            }
        }
        if (threat != null) {
            val passed = board.copy(withStack = false)
            passed.push(Move.NULL)
            if (passed.isLegal(threat)) {
                val their = passed.san(threat)
                val worked = !passed.isCapture(threat) || Exchange.exchange(passed, threat) > 0
                val stopped = !after.isLegal(threat) || (
                    passed.isCapture(threat) &&
                        after.isCapture(threat) &&
                        Exchange.exchange(after, threat) <= 0 &&
                        worked
                    )
                if (stopped && threat.to == move.to) {
                    // The move takes the very square theirs wanted: said by their piece, or the
                    // square named would be the answer ("stops f3" when the move is f3).
                    val mover = "their ${named(passed, threat.from)}"
                    out.add(Effect("takes the square $mover wants", "Where does $mover want to go?"))
                } else if (stopped && threat.from != move.to) {
                    out.add(Effect("stops $their", "How do you stop $their?"))
                }
            }
        }
        val moved = board.pieceAt(move.from)
        if (moved != null &&
            moved.pieceType != KING &&
            Exchange.canBeWon(board, move.from) &&
            !Exchange.canBeWon(after, move.to)
        ) {
            out.add(
                Effect("takes your ${NAMES.getValue(moved.pieceType)} out of danger", "Which of your pieces is in danger?"),
            )
        }
        for (square in squaresOf(board.occupiedCo[if (us) 1 else 0])) {
            if (square == move.from || !Exchange.canBeWon(board, square)) continue
            if (after.pieceAt(square) != null && !Exchange.canBeWon(after, square)) {
                out.add(Effect("covers your ${named(board, square)}", "Which of your pieces needs cover?"))
                break
            }
        }
        val pawnEnding = board.pieceMap().values.all { it.pieceType == KING || it.pieceType == PAWN }
        if (moved != null && moved.pieceType == KING && pawnEnding) {
            val ours = after.king(us)
            val theirs = after.king(!us)
            if (ours != null && theirs != null) {
                val files = abs(squareFile(ours) - squareFile(theirs))
                val ranks = abs(squareRank(ours) - squareRank(theirs))
                if (listOf(files, ranks).sorted() == listOf(0, 2)) {
                    out.add(Effect("takes the opposition", "Where does your king face theirs?"))
                }
            }
        }
        val piece = board.pieceAt(move.from)
        if (piece != null && piece.pieceType == PAWN && !board.isCapture(move) && isPassed(board, move.from)) {
            out.add(Effect("pushes your passed pawn", "How far can your passed pawn go?"))
        }
        return out
    }

    /**
     * The third hint's reason and the second hint's question for a move that holds where little
     * else does (`holds`: "it is the only move that keeps your edge"): "Bb5+ gives check and
     * stops Nf6: it is the only move that keeps your edge".
     */
    fun reason(board: Board, move: Move, threat: Move?, holds: String): Pair<String, String> {
        val san = board.san(move)
        val found = effects(board, move, threat)
        if (found.isNotEmpty()) {
            val words = found.take(2).joinToString(" and ") { it.said }
            return "$san $words: $holds" to found[0].ask
        }
        return "$san: $holds" to "Which move keeps everything you have?"
    }

    /**
     * How a win with no named pattern comes about, for the first hint: what the first move does,
     * never the move. "Their pawn on h5 is the target" alone left the player to find the king
     * walk that wins it (Lichess jlCZP, a zugzwang); "the first move attacks their rook on d8"
     * says how.
     */
    fun how(board: Board, idea: Idea, threat: Move?): Effect? {
        if ((idea.goal != "win" && idea.goal != "combination") || idea.blow != null) return null
        val first = idea.first ?: return null
        if (idea.gainPly == 0 && board.isCapture(first)) return null  // the capture is the how
        if (idea.intermezzo) return Effect("A check comes first, before anything is taken", "")
        if (idea.sacrifices.isNotEmpty() && idea.sacrifices[0].first == 0) {
            val given = idea.sacrifices[0].second
            return Effect("It starts by giving up ${if (given == "exchange") "the" else "a"} $given", "")
        }
        var found = effects(board, first, threat)
        // Taking a piece is the point when it is a piece, or the one giving check; a pawn taken
        // on the way is not, and what else the move does comes first.
        val taken = if (board.isCapture(first)) board.pieceTypeAt(first.to) else 0
        val big = taken != 0 && (
            Exchange.VALUE[taken] >= FREE_MIN || board.checkersMask() and BB_SQUARES[first.to] != BB_EMPTY
            )
        if (!big) found = found.filter { !it.said.startsWith("takes their") }.ifEmpty { found }
        // "pushes your passed pawn" is the move itself, not how it wins.
        found = found.filter { !it.said.startsWith("pushes") }
        if (found.isEmpty()) return null
        val firstEffect = found[0]
        return Effect("The first move ${firstEffect.said}", firstEffect.ask, firstEffect.lesson)
    }

    /**
     * The third hint for a win with no named pattern: what the move does, the line's next two
     * moves, what it wins -- "Kf4 takes the opposition, and after Kf6, Kg5; you win two pawns",
     * never "Kf4: you win two pawns, a few moves on".
     */
    fun moveWhy(board: Board, idea: Idea, threat: Move?): String? {
        val first = idea.first
        if (first == null || !board.isLegal(first) || idea.won.isNullOrEmpty()) return null
        val san = board.san(first)
        val found = effects(board, first, threat)
        val did = found.take(2).joinToString(" and ") { it.said }
        val b = board.copy(withStack = false)
        val moves = ArrayList<String>()
        for (move in idea.line.take(3)) {
            if (!b.isLegal(move)) break
            moves.add(b.san(move))
            b.push(move)
        }
        val then = if (moves.size == 3) ", and after ${moves[1]}, ${moves[2]}" else ""
        val head = if (did.isNotEmpty()) "$san $did" else san
        if (did.isEmpty() && then.isEmpty()) return null
        val later = if (idea.gainPly > 2) ", a few moves on" else ""
        return "$head$then; you win ${idea.won}$later"
    }

    /**
     * The best line's next two moves, when nothing on the board says what the move does: "a3,
     * and after Kf6, Kd4 -- it is the only move that keeps your edge".
     */
    fun lineReason(board: Board, line: List<Move>, holds: String): String? {
        if (line.size < 3) return null
        val b = board.copy(withStack = false)
        val said = ArrayList<String>()
        for (move in line.take(3)) {
            if (!b.isLegal(move)) return null
            said.add(b.san(move))
            b.push(move)
        }
        return "${said[0]}, and after ${said[1]}, ${said[2]} -- $holds"
    }
}
