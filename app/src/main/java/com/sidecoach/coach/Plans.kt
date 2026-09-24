package com.sidecoach.coach

import com.sidecoach.analysis.Exchange
import com.sidecoach.analysis.isOutpost
import com.sidecoach.analysis.isPassed
import com.sidecoach.analysis.pawnStormSide
import com.sidecoach.chess.BB_EMPTY
import com.sidecoach.chess.BB_FILES
import com.sidecoach.chess.BB_KING_ATTACKS
import com.sidecoach.chess.BB_KNIGHT_ATTACKS
import com.sidecoach.chess.BB_SQUARES
import com.sidecoach.chess.BISHOP
import com.sidecoach.chess.Bitboard
import com.sidecoach.chess.Board
import com.sidecoach.chess.KING
import com.sidecoach.chess.KNIGHT
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.between
import com.sidecoach.chess.pawnAttacks
import com.sidecoach.chess.popcount
import com.sidecoach.chess.square
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareName
import com.sidecoach.chess.squareRank
import com.sidecoach.chess.squaresOf

/**
 * The plan in the best line: what the player's next few moves are for, read as a whole. A port
 * of `sidecoach/coach/plans.py`.
 *
 * The positional layer reads one move -- "this develops a piece", "this takes the file" -- and a
 * grandmaster explains a game in plans: "White begins operations on the h-file", "Black wants
 * to double on the c-file", "the knight is heading for c6". Against the master games' notes
 * (tools/eval/masters_notes.py, 2026-09-24) the coach named an attack on the king at 12% of the
 * moves the note did: it read the best move alone.
 *
 * So the player's first four moves in the engine's best line are read together, on the boards
 * the line passes through, each piece followed from where it stands now. A plan counts only
 * from quiet moves -- a capture or a recapture is an exchange, not a plan -- and the best move
 * must be part of it, so the coach never names a plan the move in hand does not start:
 *
 *   attack   pieces coming at their king: each move leaves the piece bearing on more of the
 *            squares around their king than before (a rook lifted on the way counts), two
 *            moves at least, and more pieces bearing there at the end than now
 *   storm    two pawn advances at their king, castled on the wing away from yours
 *   double   two heavy pieces lined up on a file with none of the player's pawns, not so now
 *   outpost  a knight taken in two moves or more to an outpost it could not reach in one
 *   passer   the same passed pawn pushed twice
 *   break    a pawn advance in the centre that meets one of theirs, the best move a piece
 *            brought to bear on that square first
 *   wing     two pawn advances on the wing away from both kings
 *
 * The first found in that order is the plan. An attack or a storm is not read for a side the
 * engine calls worse (below LEVEL_SCORE), where it would be a hope, not a plan.
 */

/**
 * A plan the best line carries out: its kind, the square it is about (their king, the outpost,
 * the passed pawn, the break), the file or wing, the player's pieces that carry it out as they
 * stand now, and the moves that do, in the order played. `plans.LinePlan`.
 */
data class LinePlan(
    val kind: String,
    val square: Int? = null,
    val where: String? = null,
    val pieces: List<Int> = emptyList(),
    val sans: List<String> = emptyList(),
    /** The doubled file: "open" or "half-open". */
    val detail: String? = null,
)

/**
 * What the best move threatens, were they to pass: mate (`square` their king), winning what
 * stands on `square` (`gain` pawns), or a trade of the moved piece for the like piece on
 * `square`, which the line then makes; `follow` is the move that would do it. `why` says what a
 * trade is for: "ahead" in material, "queens" off when the player is worse, else "plain".
 * `plans.MoveThreat`.
 */
data class MoveThreat(
    /** "mate", "win" or "trade". */
    val kind: String,
    val square: Int,
    val follow: Move,
    val gain: Int = 0,
    val why: String = "plain",
)

object Plans {

    /** The player's moves read together. */
    const val PLAN_MOVES = 4

    /** An attack or storm is read only for a side at least this well off. */
    const val LEVEL_SCORE = 0.45

    private val HEAVY = listOf(ROOK, QUEEN)
    private val NAMES = mapOf(
        PAWN to "pawn",
        KNIGHT to "knight",
        BISHOP to "bishop",
        ROOK to "rook",
        QUEEN to "queen",
        KING to "king",
    )

    /** The line walked from the board: `plans._Line`. */
    private class WalkedLine(
        val board: Board,
        val boards: List<Board>,
        val line: List<Move>,
        /** Indices of the player's moves in `line`. */
        val ours: List<Int>,
        /** The player's move -> where its piece stands now. */
        val origin: Map<Int, Int>,
        val us: Boolean,
    )

    private fun walk(board: Board, pv: List<Move>): WalkedLine {
        val us = board.turn
        val boards = arrayListOf(board.copy(withStack = false))
        val line = ArrayList<Move>()
        for (move in pv.take(2 * PLAN_MOVES)) {
            if (!boards.last().isLegal(move)) break
            val nxt = boards.last().copy(withStack = false)
            nxt.push(move)
            boards.add(nxt)
            line.add(move)
        }
        val ours = (0 until line.size step 2).toList()
        // Each of the player's pieces followed from where it stands now.
        val at = HashMap<Int, Int>()
        for (sq in squaresOf(board.occupiedCo[if (us) 1 else 0])) at[sq] = sq
        val origin = HashMap<Int, Int>()
        for (j in ours) {
            val move = line[j]
            val start = at.remove(move.from) ?: move.from
            origin[j] = start
            at[move.to] = start
            if (boards[j].isCastling(move)) {
                val (rookFrom, rookTo) = castleRook(boards[j], move)
                at[rookTo] = at.remove(rookFrom) ?: rookFrom
            }
            if (j + 1 < line.size) at.remove(line[j + 1].to)  // taken
        }
        return WalkedLine(board, boards, line, ours, origin, us)
    }

    private fun castleRook(board: Board, move: Move): Pair<Int, Int> {
        val rank = squareRank(move.from)
        if (squareFile(move.to) > squareFile(move.from)) return square(7, rank) to square(5, rank)
        return square(0, rank) to square(3, rank)
    }

    private fun quiet(b: Board, move: Move): Boolean = !b.isCapture(move) && move.promotion == 0

    private fun zone(king: Int): Bitboard = BB_KING_ATTACKS[king] or BB_SQUARES[king]

    /** How many of the player's pieces (not the king) bear on their king's zone, pawns too. */
    private fun bearing(b: Board, us: Boolean, king: Int): Int {
        val zone = zone(king)
        return b.pieceMap().count { (sq, piece) ->
            piece.colour == us && piece.pieceType != KING && b.attacksMask(sq) and zone != BB_EMPTY
        }
    }

    /**
     * The plan the player's first moves of `pv` carry out, or null. `score` is the player's
     * expected score in the line, when known.
     */
    fun read(board: Board, pv: List<Move>, score: Double? = null): LinePlan? {
        val w = walk(board, pv)
        if (w.ours.size < 2) return null
        val level = score == null || score >= LEVEL_SCORE
        val checks: List<((WalkedLine) -> LinePlan?)?> = listOf(
            if (level) this::attack else null,
            if (level) this::storm else null,
            this::double,
            this::outpost,
            this::passer,
            this::pawnBreak,
            this::wing,
        )
        for (check in checks) {
            if (check == null) continue
            val plan = check(w)
            if (plan != null) return plan
        }
        return null
    }

    private fun sans(w: WalkedLine, js: List<Int>): List<String> = js.map { w.boards[it].san(w.line[it]) }

    private fun plan(
        w: WalkedLine,
        kind: String,
        js: List<Int>,
        square: Int? = null,
        where: String? = null,
    ): LinePlan {
        val pieces = js.map { w.origin.getValue(it) }.distinct()
        return LinePlan(kind, square = square, where = where, pieces = pieces, sans = sans(w, js))
    }

    private fun wingOf(file: Int): String? = if (file <= 2) "queenside" else if (file >= 5) "kingside" else null

    /**
     * A pawn advance on the wing where their king stands, the kings apart (the positional
     * layer's storm, kept to the files of that wing).
     */
    private fun stormPush(w: WalkedLine, j: Int): Boolean {
        val b = w.boards[j]
        val move = w.line[j]
        val theirs = w.board.king(!w.us)
        if (theirs == null || b.pieceTypeAt(move.from) != PAWN || !quiet(b, move)) return false
        val wing = wingOf(squareFile(theirs))
        return wingOf(squareFile(move.to)) == wing && pawnStormSide(b, move, w.us) != null
    }

    /** The square's rank counted from the player's side, 1 to 8. */
    private fun rank(square: Int, us: Boolean): Int {
        val rank = squareRank(square)
        return if (us == WHITE) rank + 1 else 8 - rank
    }

    private fun attack(w: WalkedLine): LinePlan? {
        val king = w.board.king(!w.us) ?: return null
        val joins = ArrayList<Int>()
        val lifts = HashSet<Int>()
        for (j in w.ours) {
            val b = w.boards[j]
            val move = w.line[j]
            if (!quiet(b, move)) continue
            val piece = b.pieceTypeAt(move.from)
            val theirKing = b.king(!w.us)
            if (theirKing == null || piece == 0 || piece == KING) continue
            if (piece == PAWN) {
                if (stormPush(w, j)) joins.add(j)
                continue
            }
            val zone = zone(theirKing)
            val before = popcount(b.attacksMask(move.from) and zone)
            val after = popcount(w.boards[j + 1].attacksMask(move.to) and zone)
            if (after > before) {
                joins.add(j)
            } else if (piece == ROOK && squareFile(move.from) == squareFile(move.to)) {
                // A rook lift: up its file, to swing across to their king later.
                lifts.add(j)
                joins.add(j)
            }
        }
        // A lift counts only when the same rook swings in afterwards; the swing itself always
        // counts (Qd3 Re3 Rh3 was read as "Qd3 and Re3", the Kotlin port, 2026-09-24).
        val kept = joins.filter { j ->
            j !in lifts ||
                joins.any { k -> k > j && k !in lifts && w.origin.getValue(k) == w.origin.getValue(j) }
        }
        val pieces = kept
            .filter { w.boards[it].pieceTypeAt(w.line[it].from) != PAWN }
            .map { w.origin.getValue(it) }
            .toSet()
        if (0 !in kept || pieces.isEmpty() || kept.map { w.origin.getValue(it) }.toSet().size < 2) return null
        val end = w.boards[w.ours.last() + 1]
        val endKing = end.king(!w.us)
        if (endKing == null || bearing(end, w.us, endKing) <= bearing(w.board, w.us, king)) return null
        return plan(w, "attack", kept, square = king)
    }

    private fun storm(w: WalkedLine): LinePlan? {
        val king = w.board.king(!w.us) ?: return null
        val pushes = w.ours.filter { stormPush(w, it) }
        if (pushes.size < 2 || pushes[0] != 0) return null
        val wing = if (squareFile(king) >= 4) "kingside" else "queenside"
        return plan(w, "storm", pushes, square = king, where = wing)
    }

    private fun heavyOn(b: Board, us: Boolean, file: Int): List<Int> = squaresOf(BB_FILES[file]).filter { sq ->
        val p = b.pieceAt(sq)
        p != null && p.colour == us && p.pieceType in HEAVY
    }

    private fun double(w: WalkedLine): LinePlan? {
        val end = w.boards[w.ours.last() + 1]
        val first = w.line[0]
        if (!quiet(w.board, first) || w.board.pieceTypeAt(first.from) !in HEAVY) return null
        val file = squareFile(first.to)
        val heavy = heavyOn(end, w.us, file)
        if (heavy.size < 2 || heavyOn(w.board, w.us, file).size >= 2) return null
        if (end.piecesMask(PAWN, w.us) and BB_FILES[file] != BB_EMPTY) return null
        val theirs = end.piecesMask(PAWN, !w.us) and BB_FILES[file]
        val detail = if (theirs != BB_EMPTY) "half-open" else "open"
        // Lined up: nothing between the two.
        if (between(heavy[0], heavy[1]) and end.occupied != BB_EMPTY) return null
        val onto = w.ours.filter { j ->
            quiet(w.boards[j], w.line[j]) &&
                w.boards[j].pieceTypeAt(w.line[j].from) in HEAVY &&
                squareFile(w.line[j].to) == file &&
                squareFile(w.line[j].from) != file
        }
        if (0 !in onto) return null
        // The pieces that end up doubled, as they stand now.
        val now = heavy.map { sq ->
            val landed = w.ours.filter { w.line[it].to == sq }
            if (landed.isNotEmpty()) w.origin.getValue(landed.last()) else sq
        }
        return LinePlan(
            "double", where = "abcdefgh"[file].toString(), pieces = now, sans = sans(w, onto), detail = detail,
        )
    }

    private fun outpost(w: WalkedLine): LinePlan? {
        val first = w.line[0]
        if (w.board.pieceTypeAt(first.from) != KNIGHT || !quiet(w.board, first)) return null
        val knight = w.origin.getValue(0)
        val route = w.ours.filter { w.origin.getValue(it) == knight }
        if (route.size < 2 || !route.all { quiet(w.boards[it], w.line[it]) }) return null
        for ((index, j) in route.drop(1).withIndex()) {
            val n = index + 2
            val to = w.line[j].to
            if (to == knight || BB_KNIGHT_ATTACKS[knight] and BB_SQUARES[to] != BB_EMPTY) continue
            if (isOutpost(w.boards[j + 1], to, w.us)) return plan(w, "outpost", route.take(n), square = to)
        }
        return null
    }

    private fun passer(w: WalkedLine): LinePlan? {
        val first = w.line[0]
        if (w.board.pieceTypeAt(first.from) != PAWN || !quiet(w.board, first)) return null
        if (!isPassed(w.board, first.from)) return null
        val pawn = w.origin.getValue(0)
        val pushes = w.ours.filter { w.origin.getValue(it) == pawn && quiet(w.boards[it], w.line[it]) }
        if (pushes.size < 2) return null
        return plan(w, "passer", pushes, square = pawn)
    }

    /** A pawn advance that meets one of theirs: it attacks a pawn or a pawn attacks it. */
    private fun lever(b: Board, move: Move, us: Boolean): Boolean {
        val after = b.copy(withStack = false)
        after.push(move)
        val to = move.to
        val theirs = after.piecesMask(PAWN, !us)
        return after.attacksMask(to) and theirs != BB_EMPTY || pawnAttacks(us, to) and theirs != BB_EMPTY
    }

    /** `plans._break`. */
    private fun pawnBreak(w: WalkedLine): LinePlan? {
        val first = w.line[0]
        val mover = w.board.pieceTypeAt(first.from)
        if (mover == 0 || mover == PAWN || mover == KING) return null
        if (!quiet(w.board, first)) return null
        for (j in w.ours.drop(1)) {
            val b = w.boards[j]
            val move = w.line[j]
            if (b.pieceTypeAt(move.from) != PAWN || !quiet(b, move)) continue
            if (squareFile(move.to) !in 2..5) continue
            if (rank(move.to, w.us) !in listOf(4, 5) || !lever(b, move, w.us)) continue
            // The best move prepared it: its piece bears on the square the pawn goes to.
            if (w.boards[1].attacksMask(first.to) and BB_SQUARES[move.to] != BB_EMPTY) {
                return plan(w, "break", listOf(0, j), square = move.to)
            }
            return null
        }
        return null
    }

    private fun wing(w: WalkedLine): LinePlan? {
        val theirs = w.board.king(!w.us)
        val ours = w.board.king(w.us)
        if (theirs == null || ours == null) return null
        val kingSide = squareFile(theirs) >= 4
        if ((squareFile(ours) >= 4) != kingSide) return null  // kings on opposite wings: advancing is a storm, not expansion
        if (squareFile(theirs) in 3..4) return null
        val far = if (kingSide) 0 until 3 else 5 until 8
        val pushes = w.ours.filter { j ->
            w.boards[j].pieceTypeAt(w.line[j].from) == PAWN &&
                quiet(w.boards[j], w.line[j]) &&
                squareFile(w.line[j].to) in far &&
                rank(w.line[j].to, w.us) >= 4
        }
        if (pushes.size < 2 || pushes[0] != 0) return null
        return plan(w, "wing", pushes, where = if (kingSide) "queenside" else "kingside")
    }

    // The words, per plan: its name (square-free, for the idea rung), why it works, the question
    // that sets the player looking, and the focus rung's nudge.
    val PLAN_NAME: Map<String, String> = linkedMapOf(
        "attack" to "bringing more pieces against their king",
        "storm" to "a pawn storm against their king",
        "double" to "doubling your heavy pieces on {detail} file",
        "outpost" to "rerouting a knight to an outpost",
        "passer" to "pushing your passed pawn",
        "break" to "preparing a pawn break in the centre",
        "wing" to "gaining space with your pawns on the {where}",
    )
    val PLAN_REASON: Map<String, String> = linkedMapOf(
        "attack" to "an attack breaks through when more of your pieces bear on their king than they " +
            "can bring to defend it",
        "storm" to "with the kings on opposite wings, your pawns can roll at theirs without " +
            "exposing your own, and they pry open lines for your pieces",
        "double" to "two heavy pieces on the same file hit twice as hard and break into their position",
        "outpost" to "a knight on a square no pawn of theirs can ever attack stays there for good",
        "passer" to "a passed pawn ties their pieces down, and every step makes it stronger",
        "break" to "with your pieces trained on it, the pawn break opens the centre on your terms",
        "wing" to "on the wing away from the kings you can take space without weakening yours",
    )
    val PLAN_QUESTION: Map<String, String> = linkedMapOf(
        "attack" to "Which of your pieces is not yet helping against their king, and how does it " +
            "get there?",
        "storm" to "Which pawn starts the storm, and which file does it open?",
        "double" to "Which file has none of your pawns on it, and how do two of your heavy pieces " +
            "get onto it?",
        "outpost" to "Which square could no pawn of theirs ever drive a knight from, and how does " +
            "one of yours get there?",
        "passer" to "What stops your passed pawn, and how do you get it moving?",
        "break" to "Which pawn move would open the centre, and which piece should support it first?",
        "wing" to "Where can your pawns take space without weakening your king?",
    )
    val PLAN_NUDGE: Map<String, String> = linkedMapOf(
        "attack" to "Which piece joins the attack first?",
        "storm" to "Which pawn goes first?",
        "double" to "Which heavy piece goes to the file first?",
        "outpost" to "Which first step starts the knight on its way?",
        "passer" to "Which move pushes it, or clears its way?",
        "break" to "Which piece supports the break first?",
        "wing" to "Which pawn goes first?",
    )

    fun name(plan: LinePlan): String {
        val detail = if (plan.detail == "half-open") "a half-open" else "an open"
        return PLAN_NAME.getValue(plan.kind).replace("{where}", plan.where ?: "").replace("{detail}", detail)
    }

    fun mechanism(plan: LinePlan): String = "${name(plan)}: ${PLAN_REASON.getValue(plan.kind)}"

    /** `plan.detail or 'open'`. */
    private fun detailOrOpen(plan: LinePlan): String = if (plan.detail.isNullOrEmpty()) "open" else plan.detail

    /** The focus rung's target: their king, the outpost, the file, the passed pawn. */
    fun target(plan: LinePlan): String? {
        val sq = plan.square?.let { squareName(it) }
        if ((plan.kind == "attack" || plan.kind == "storm") && sq != null) return "their king on $sq"
        if (plan.kind == "outpost" && sq != null) return "the $sq square, an outpost for a knight"
        if (plan.kind == "double" && !plan.where.isNullOrEmpty()) return "the ${detailOrOpen(plan)} ${plan.where}-file"
        if (plan.kind == "passer" && sq != null) return "your passed pawn on $sq"
        if (plan.kind == "break" && sq != null) return "the $sq square, where the break comes"
        if (plan.kind == "wing" && !plan.where.isNullOrEmpty()) return "your pawns on the ${plan.where}"
        return null
    }

    private fun listed(names: List<String>): String =
        if (names.size == 1) names[0] else names.dropLast(1).joinToString(", ") + " and ${names.last()}"

    /**
     * The player's pieces that carry it out, where they stand now: "your rook on a1 and your
     * queen on d1".
     */
    fun pieces(plan: LinePlan, board: Board): String? {
        val named = plan.pieces.mapNotNull { sq ->
            val p = board.pieceAt(sq)
            if (p != null && p.colour == board.turn) "your ${NAMES.getValue(p.pieceType)} on ${squareName(sq)}" else null
        }.take(3)
        return if (named.isNotEmpty()) listed(named) else null
    }

    /** The plan played out, for the move rung: the moves that carry it out and what for. */
    fun why(plan: LinePlan, board: Board): String {
        val moves = listed(plan.sans)
        val one = plan.sans.size == 1

        fun verb(plural: String, singular: String): String = if (one) singular else plural

        val kinds = plan.pieces.mapNotNull { sq ->
            val p = board.pieceAt(sq)
            if (p != null && p.colour == board.turn) NAMES.getValue(p.pieceType) else null
        }
        // "your knights" for two knights, "your queen and rook" for one of each.
        val unique = kinds.distinct()
        val what = if (unique.size == 1 && kinds.size > 1) {
            "${unique[0]}s"
        } else if (unique.isNotEmpty()) {
            listed(unique)
        } else {
            "pieces"
        }
        val sq = plan.square?.let { squareName(it) } ?: ""
        if (plan.kind == "attack") return "$moves ${verb("bring", "brings")} your $what against their king"
        if (plan.kind == "storm") {
            return "$moves ${verb("roll", "rolls")} your pawns at their king on the ${plan.where}"
        }
        if (plan.kind == "double") {
            return "$moves ${verb("double", "doubles")} your $what on the ${detailOrOpen(plan)} " +
                "${plan.where}-file"
        }
        if (plan.kind == "outpost") return "$moves take your knight to $sq, an outpost no pawn of theirs can attack"
        if (plan.kind == "passer") return "$moves push your passed pawn"
        if (plan.kind == "break") {
            val first = plan.sans[0]
            val last = plan.sans.last()
            return "$first brings a piece to bear on $sq, then $last breaks open the centre"
        }
        return "$moves gain space on the ${plan.where}"
    }

    // The threat the best move makes. Where the line wins nothing -- they answer it -- the move
    // can still be a threat they must answer: "threatening f5", "hitting the loose knight", "now
    // Black threatens ...Bb3" (the master games' notes). The coach said "a quiet move is wanted
    // here" at 18% of their annotated moves; a threat is what many of those moves are.

    /**
     * The threat `move` makes that was not there before it: a mate in one, or the most a
     * capture would net on a square that could not be won before, or a trade it offers that the
     * line `pv` (starting with `move`) takes up. Null for a check (that is forcing, not a
     * threat), a capture (its point is the exchange) or a move that leaves its piece to be won
     * (an offer, not a threat).
     */
    fun threat(board: Board, move: Move, pv: List<Move> = emptyList(), score: Double? = null): MoveThreat? {
        val us = board.turn
        if (board.isCapture(move) || move.promotion != 0) return null
        val after = board.copy(withStack = false)
        after.push(move)
        if (after.isCheck() || after.isGameOver()) return null
        if (Exchange.canBeWon(after, move.to)) return null
        val passed = after.copy(withStack = false)
        passed.push(Move.NULL)
        val king = after.king(!us)
        for (follow in passed.legalMoves()) {
            passed.push(follow)
            val mate = passed.isCheckmate()
            passed.pop()
            if (mate && king != null) return MoveThreat("mate", king, follow)
        }
        val mover = after.pieceAt(move.to) ?: return null
        val toBb = BB_SQUARES[move.to]
        val view = Exchange.asMover(after, us)
        var best: Pair<Int, Int>? = null
        var trade: Int? = null
        for (sq in squaresOf(after.occupiedCo[if (!us) 1 else 0])) {
            val piece = after.pieceAt(sq)
            if (piece == null || piece.pieceType == KING) continue
            // Face to face with a like piece: they take first, or trade, and "if they let it be"
            // is no threat at all (Bf6 against a bishop on h4, the master games, 2026-09-24).
            // Two pawns face to face are a pawn break, which the positional layer names.
            val facing = after.attacksMask(sq) and toBb != BB_EMPTY &&
                after.attacksMask(move.to) and BB_SQUARES[sq] != BB_EMPTY
            val like = Exchange.VALUE[piece.pieceType] == Exchange.VALUE[mover.pieceType]
            if (facing && like && mover.pieceType != PAWN) {
                if (trade == null) trade = sq
                continue
            }
            val gain = Exchange.bestCapture(after, us, sq)
            if (gain <= 0 || gain <= Exchange.bestCapture(board, us, sq)) continue
            if (best == null || gain > best.first) best = gain to sq
        }
        if (best != null) {
            val (gain, sq) = best
            val follow = Exchange.capturesOn(after, us, sq).maxByOrNull { Exchange.exchange(view, it) }!!
            return MoveThreat("win", sq, follow, gain)
        }
        if (trade == null) return null
        // A trade only when the line makes it, on either square, within the next two moves.
        val squares = setOf(move.to, trade)
        val line = if (pv.isNotEmpty() && pv[0] == move) pv.subList(1, minOf(pv.size, 3)) else emptyList()
        if (line.none { it.to in squares }) return null
        val follow = Exchange.capturesOn(after, us, trade).firstOrNull() ?: Move(move.to, trade)
        val material = board.pieceMap().values
            .filter { it.pieceType != KING }
            .sumOf { Exchange.VALUE[it.pieceType] * (if (it.colour == us) 1 else -1) }
        val why = if (material >= 2) {
            "ahead"
        } else if (mover.pieceType == QUEEN && score != null && score < LEVEL_SCORE) {
            "queens"
        } else {
            "plain"
        }
        return MoveThreat("trade", trade, follow, 0, why)
    }

    val THREAT_GOAL: Map<String, String> = linkedMapOf(
        "mate" to "you can threaten mate",
        "win" to "you can take the initiative",
        "trade" to "you can offer a trade",
    )

    /** The same, when the engine calls the game won: said first, as the plan goal says it. */
    val THREAT_GOAL_WON: Map<String, String> = linkedMapOf(
        "mate" to "you are winning, and you can threaten mate",
        "win" to "you are winning: a threat keeps them on the back foot",
        "trade" to "you are winning, and a trade brings the win closer",
    )
    val THREAT_NAME: Map<String, String> = linkedMapOf(
        "mate" to "a threat of mate",
        "win" to "a threat",
        "trade" to "a trade",
    )
    val THREAT_REASON: Map<String, String> = linkedMapOf(
        "mate" to "they must spend their move stopping it, and have no time for their own plans",
        "win" to "a move that attacks something they must then defend keeps you a step ahead: " +
            "they answer you instead of playing their own moves",
    )
    val TRADE_REASON: Map<String, String> = linkedMapOf(
        "ahead" to "when you are ahead, every trade brings the win closer",
        "queens" to "with the queens off, their attack loses its sting",
        "plain" to "every trade changes the fight, and here it is the best way on",
    )
    val THREAT_QUESTION: Map<String, String> = linkedMapOf(
        "mate" to "Which move would leave you threatening mate next move?",
        "win" to "Which of their pieces could you attack with something worth less, or where it " +
            "is short of defenders?",
        "trade" to "Which of their pieces does the most for them, and which of yours could trade " +
            "itself off for it?",
    )
    val THREAT_NUDGE: Map<String, String> = linkedMapOf(
        "mate" to "Which move sets up the mate?",
        "win" to "Which move attacks it?",
        "trade" to "Which move puts your piece face to face with it?",
    )

    fun threatMechanism(t: MoveThreat): String {
        val reason = if (t.kind == "trade") TRADE_REASON.getValue(t.why) else THREAT_REASON.getValue(t.kind)
        return "${THREAT_NAME.getValue(t.kind)}: $reason"
    }

    /** The focus rung's target: their king, or what the threat goes after, as it stands. */
    fun threatTarget(t: MoveThreat, board: Board): String? {
        val piece = board.pieceAt(t.square)
        if (t.kind == "mate") return "their king on ${squareName(t.square)}"
        if (piece == null || piece.colour == board.turn) return null
        return "their ${NAMES.getValue(piece.pieceType)} on ${squareName(t.square)}"
    }

    /**
     * The move and its threat, for the move rung: "Nc4 goes after their pawn on e5: if they let
     * it be, Nxe5 wins it".
     */
    fun threatWhy(t: MoveThreat, board: Board, move: Move): String {
        val san = board.san(move)
        val after = board.copy(withStack = false)
        after.push(move)
        after.push(Move.NULL)
        val follow = after.san(t.follow)
        if (t.kind == "mate") return "$san threatens ${follow.trimEnd('#')} mate"
        val piece = board.pieceAt(t.square) ?: return "$san threatens $follow"
        val what = "their ${NAMES.getValue(piece.pieceType)} on ${squareName(t.square)}"
        if (t.kind == "trade") {
            val mover = board.pieceAt(move.from)
            val mine = if (mover != null) NAMES.getValue(mover.pieceType) else "piece"
            return "$san puts your $mine face to face with $what, to trade it off"
        }
        val won = if (t.gain >= Exchange.VALUE[piece.pieceType]) "wins it" else "wins material"
        return "$san goes after $what: if they let it be, $follow $won"
    }
}
