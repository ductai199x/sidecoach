package com.sidecoach.analysis

import com.sidecoach.chess.BB_ALL
import com.sidecoach.chess.BB_EMPTY
import com.sidecoach.chess.BB_KING_ATTACKS
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
import com.sidecoach.chess.between
import com.sidecoach.chess.parseSquare
import com.sidecoach.chess.ray
import com.sidecoach.chess.square
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareName
import com.sidecoach.chess.squareRank
import com.sidecoach.chess.squaresOf
import com.sidecoach.events.MotifTag
import kotlin.math.abs

/**
 * Motif tagger: pure functions over (board, move, principal variation). A port of
 * `sidecoach/analysis/motifs.py`.
 *
 * No engine calls, no I/O, no global state, and the boards handed in are never mutated.
 * Every tag is justified by evidence on the board or in the PV, never by a guess, because
 * the coach reads the evidence squares out loud ("look at f7 and the knight on g5"). Tag
 * names are the Lichess puzzle theme names, camelCase, spelled exactly as Lichess spells
 * them, so the profile and the recommendation engine can key on them.
 *
 * Two questions, two functions. [tagsForMove] answers "what does this move do", from the
 * mover's point of view. [tagsForPosition] answers "what is in this position" for the side
 * to move: the static features plus the tags of the engine's best move.
 *
 * The theme names are Lichess's, and so are the definitions: a motif is what Lichess would
 * tag, not a stricter thing of our own. Two rules keep the tags honest:
 *
 * 1. A tactic is named only when it is tactically real: it wins something, never as geometry
 *    alone (Tai, 2026-09-23). A tag that claims material has to gain it once defenders are
 *    counted: a knight forking two rooks while a pawn can take it is not a fork, and a loose
 *    piece must really be undefended. A pin is a claim too: a piece is pinned when leaving
 *    the line would lose what stands behind it -- the king, or a piece the pinner could take
 *    for profit -- and a pinner the enemy simply wins pins nothing. A queen "pinning" a knight
 *    to a defended rook pins nothing, whatever the geometry; Lichess tags it, the coach must
 *    not teach it. The same goes for the rest: a discovered attack reveals something the
 *    slider can win (or the king), a skewer still wins the piece behind whatever the front
 *    piece does, and the line motifs (deflection, attraction, removing the defender,
 *    interference, clearance) end in material won or mate. "Can take" and "defends" are read
 *    off legal moves ([Exchange]): a piece pinned to its own king takes nothing.
 * 2. [tagsForMove] reports only what the move brought about. A pin the bishop was already
 *    making before the move is not something the move did, so castling does not inherit it.
 *    Those features still show up in [tagsForPosition], where they belong.
 *
 * `mateIn` counts the mover's own moves including the one being judged: 1 means this move is
 * checkmate, 3 means this move plus two more, and a negative value means the mover gets
 * mated. A negative `mateIn` never produces a `mateIn*` tag.
 *
 * Output order is fixed by [TAG_ORDER]: what the move is, then what it creates, then what the
 * line does, then the mating pattern ([Mates]). Tags are deduplicated by name and a repeated
 * name merges its evidence squares in first-seen order.
 */
object Motifs : Tagger {

    val TAG_ORDER: List<String> = listOf(
        // A. What the move is: read off the move and the board.
        "capture",
        "enPassant",
        "promotion",
        "underPromotion",
        "castling",
        "advancedPawn",
        "check",
        "doubleCheck",
        "quietMove",
        "defensiveMove",
        // B. What the move creates: read off the attack maps after the move.
        "fork",
        "pin",
        "absolutePin",
        "skewer",
        "discoveredAttack",
        "xRayAttack",
        "interference",
        "hangingPiece",
        "trappedPiece",
        "attackingF2F7",
        "exposedKing",
        // C. What the line does: read off the PV and the mate distance.
        "sacrifice",
        "clearance",
        "capturingDefender",
        "deflection",
        "attraction",
        "mate",
        "mateIn1",
        "mateIn2",
        "mateIn3",
        "mateIn4",
        "mateIn5",
        // D. The mating pattern, only when the line actually ends in mate.
        "backRankMate",
        "smotheredMate",
        "arabianMate",
        "anastasiaMate",
        "hookMate",
        "killBoxMate",
        "vukovicMate",
        "bodenMate",
        "doubleBishopMate",
        "dovetailMate",
        "swallowstailMate",
        "epauletteMate",
        "triangleMate",
        "blindSwineMate",
        "cornerMate",
        "operaMate",
        "pillsburysMate",
        "morphysMate",
        "balestraMate",
    )

    private val ORDER_INDEX = TAG_ORDER.withIndex().associate { (i, name) -> name to i }

    /** Standard 1/3/3/5/9. The king is priceless; 100 keeps every comparison right. */
    private val VALUE = intArrayOf(0, 1, 3, 3, 5, 9, 100)

    private const val MINOR = 3

    /** A sacrifice must give up at least this much: a pawn gambit is not a sacrifice. */
    private const val SACRIFICE_THRESHOLD = 2

    // --- small helpers -----------------------------------------------------------------

    private fun value(pieceType: Int): Int = if (pieceType == 0) 0 else VALUE[pieceType]

    private fun tag(name: String, squares: List<Int>, move: Move? = null): MotifTag {
        val names = ArrayList<String>()
        for (square in squares) {
            val label = squareName(square)
            if (label !in names) names.add(label)
        }
        return MotifTag(name = name, squares = names, move = move?.uci())
    }

    private fun sliders(board: Board, colour: Boolean): List<Int> =
        squaresOf((board.bishops or board.rooks or board.queens) and board.occupiedCo[if (colour) 1 else 0])

    private fun sign(n: Int): Int = if (n > 0) 1 else if (n < 0) -1 else 0

    /** First occupied square past `far`, continuing along the `near` -> `far` ray. */
    private fun nextBeyond(board: Board, near: Int, far: Int): Int? {
        val df = sign(squareFile(far) - squareFile(near))
        val dr = sign(squareRank(far) - squareRank(near))
        var file = squareFile(far) + df
        var rank = squareRank(far) + dr
        while (file in 0..7 && rank in 0..7) {
            val sq = square(file, rank)
            if (board.occupied and BB_SQUARES[sq] != BB_EMPTY) return sq
            file += df
            rank += dr
        }
        return null
    }

    private fun epVictim(move: Move): Int = square(squareFile(move.to), squareRank(move.from))

    private fun balance(board: Board, colour: Boolean): Int {
        var total = 0
        for (pieceType in intArrayOf(PAWN, KNIGHT, BISHOP, ROOK, QUEEN)) {
            total += VALUE[pieceType] * board.pieces(pieceType, colour).size
            total -= VALUE[pieceType] * board.pieces(pieceType, !colour).size
        }
        return total
    }

    /**
     * `board` with `colour` to move, the way [Exchange] reads a threat: a null move hands the
     * turn over, so no en passant right is carried across.
     */
    private fun asSide(board: Board, colour: Boolean): Board {
        if (board.turn == colour) return board
        val view = board.copy(withStack = false)
        view.push(Move.NULL)
        return view
    }

    /**
     * The tactics aimed at a piece: the blow only when the line then takes that piece. A
     * discovered attack on a pawn the line never touches, a fork of two pieces that both get
     * away, was the headline of a promotion (Tai's puzzles, 2026-09-23).
     */
    val TARGETED: List<String> = listOf(
        "fork",
        "skewer",
        "discoveredAttack",
        "absolutePin",
        "pin",
        "trappedPiece",
        "hangingPiece",
        "capturingDefender",
    )

    /**
     * A fork or a skewer is the point only when what it wins is worth a minor piece: the king
     * and a pawn on a2 forked five moves after the queen was won was named as the idea.
     */
    const val MIN_TARGET_VALUE = 3

    /** How far on the line takes it: three of the player's moves. */
    const val PAYOFF_PLIES = 6

    /**
     * Whether the line cashes in the tactic at ply `i` within [PAYOFF_PLIES]: takes a piece it
     * goes after -- by the piece that does it, where the pattern says so -- with every piece
     * followed as it moves. The squares are the tagger's, per pattern: a fork's forker then its
     * targets; a skewer's and a pin's slider, front, back; a discovered attack's mover, slider,
     * target; a trapped piece then its hunters; a loose piece; a captured defender then what it
     * guarded. `motifs.pays_off`.
     */
    fun paysOff(boards: List<Board>, pv: List<Move>, i: Int, tag: MotifTag): Boolean {
        val us = boards[i].turn
        val sq = tag.squares.map { parseSquare(it) }
        if (sq.isEmpty()) return false
        val name = tag.name
        val after = boards[i + 1]

        fun theirs(square: Int): Boolean {
            val piece = after.pieceAt(square)
            return piece != null && piece.colour != us && piece.pieceType != KING
        }

        fun worth(square: Int): Int {
            val kind = after.pieceTypeAt(square)
            return Exchange.VALUE[if (kind == 0) PAWN else kind]
        }

        var actors: MutableSet<Int>? = null  // null: any of the player's pieces
        // The piece the pattern stands on: a pin is gone once its pinner is taken (a rook that
        // pins as it takes the queen, and is taken back at once, pinned nothing).
        var holder: Int? = null
        // A fork's piece taken by anything but one of its targets refutes it: "Nf3+ forks the
        // king, queen and rook" was said of a knight gxf3 simply took (Lichess 026RF).
        var forker: Int? = null
        val aimed: List<Int>
        if (name == "fork") {
            // Whoever collects it: the queen that takes back on e6 after Bxe6+ Qxe6 cashed in
            // the bishop's fork (Lichess 0BD4O).
            aimed = sq.drop(1).filter { theirs(it) && worth(it) >= MIN_TARGET_VALUE }
            forker = sq[0]
            // A bystander that can take the forking piece for profit where it lands refutes the
            // fork, whatever the line then does (Nf2+ with the rook on f1 able to take it: the
            // discovered attack behind it was the point, Lichess 01q4A). A pinned piece cannot
            // take; a target that takes and is taken back is the fork working.
            val targets = sq.drop(1)
            for (capture in Exchange.capturesOn(after, !us, forker)) {
                if (capture.from !in targets && Exchange.exchange(after, capture) > 0) return false
            }
        } else if (name == "skewer") {
            aimed = sq.drop(2).take(1).filter { theirs(it) && worth(it) >= MIN_TARGET_VALUE }
            actors = mutableSetOf(sq[0])
        } else if (name == "discoveredAttack" && sq.size >= 3) {
            val king = after.pieceAt(sq[2])
            if (king != null && king.pieceType == KING) {
                // A discovered check: the point is what the piece that moved goes on to take.
                aimed = (0 until 64).filter { theirs(it) }
                actors = mutableSetOf(sq[0])
            } else {
                // What it uncovers must be worth a minor piece, as a fork's target is: "the rook
                // now bears on the pawn on a7" taught nothing (the master games, 2026-09-24).
                aimed = if (theirs(sq[2]) && worth(sq[2]) >= MIN_TARGET_VALUE) listOf(sq[2]) else emptyList()
                actors = mutableSetOf(sq[0], sq[1])
            }
        } else if (name == "pin" || name == "absolutePin") {
            aimed = sq.drop(1).take(2).filter { theirs(it) }
            holder = sq[0]
        } else if (name == "trappedPiece" || name == "hangingPiece") {
            aimed = if (theirs(sq[0])) listOf(sq[0]) else emptyList()
        } else if (name == "capturingDefender") {
            // Taking their queen is its own point, not the removal of the pawn's guard: what was
            // guarded must be worth what was taken.
            val taken = boards[i].pieceAt(sq[0])
            val takenWorth = if (taken != null) Exchange.VALUE[taken.pieceType] else 0
            aimed = sq.drop(1).take(1).filter { theirs(it) && worth(it) >= takenWorth }
        } else {
            return true
        }
        val start = balance(boards[i], us)

        /**
         * The capture at ply `j` leaves the player up on the start, the captures on its square
         * that follow counted: a loose bishop taken after theirs took one of the player's, the
         * dark-squared bishops traded, won nothing ("the bishop on g7 is loose" was said of a
         * move whose point was that trade, the master games' notes, 2026-09-24). A loose piece
         * is won whole, give or take a pawn: queens traded and a pawn won is not a loose queen.
         * A trapped piece is often won for a lesser one, and that is its win.
         */
        fun wins(j: Int): Boolean {
            // The exchange on that square played out to its end, as the line plays it: Qxd1 Rxd1
            // Nxd1 wins the rook on the third capture (Lichess 01q4A).
            var k = j + 1
            while (k < pv.size &&
                k + 1 < boards.size &&
                pv[k].to == pv[j].to &&
                boards[k].isCapture(pv[k])
            ) {
                k += 1
            }
            if (k >= boards.size) return false
            val net = balance(boards[k], us) - start
            if (name == "hangingPiece") {
                val taken = boards[j].pieceTypeAt(pv[j].to)
                return net >= maxOf(1, Exchange.VALUE[if (taken == 0) PAWN else taken] - 1)
            }
            return net > 0
        }

        if (pv[i].to in aimed && boards[i].isCapture(pv[i]) && wins(i)) return true  // the move itself takes it
        var watched: MutableSet<Int> = aimed.toMutableSet()
        for (j in i + 1 until minOf(pv.size, i + 1 + PAYOFF_PLIES)) {
            val move = pv[j]
            if (boards[j].turn == us) {
                if (move.to in watched &&
                    boards[j].isCapture(move) &&
                    (actors == null || move.from in actors) &&
                    wins(j)
                ) {
                    return true
                }
                if (actors != null && move.from in actors) {
                    actors = (actors - move.from + move.to).toMutableSet()
                }
                if (holder != null && move.from == holder) holder = move.to
                if (forker != null && move.from == forker) forker = move.to
            } else {
                if (forker != null && move.to == forker) {
                    if (move.from !in watched) return false  // the fork's piece taken by a bystander
                    forker = null
                }
                if (move.from in watched) watched = (watched - move.from + move.to).toMutableSet()
                actors?.remove(move.to)  // a piece of the player's taken
                if (holder != null && move.to == holder) return false  // the pinner taken: the pin is gone
            }
        }
        return false
    }

    /**
     * What the piece of `colour` on `frm` nets by taking on `to`, the recaptures played out
     * with legal moves ([Exchange.exchange]); null when it cannot legally take there at all --
     * pinned to its own king, blocked, gone, or the king in check with the capture no answer.
     */
    private fun captureGain(board: Board, colour: Boolean, frm: Int, to: Int): Int? {
        val view = asSide(board, colour)
        var best: Int? = null
        for (m in Exchange.capturesOn(view, colour, to)) {
            if (m.from != frm) continue
            val gain = Exchange.exchange(view, m)
            if (best == null || gain > best) best = gain
        }
        return best
    }

    /** The line ends with the mover's opponent checkmated. */
    private fun mates(boards: List<Board>, mover: Boolean): Boolean =
        boards.last().isCheckmate() && boards.last().turn != mover

    /**
     * Whether one of the mover's later captures (from its second move on) leaves it ahead of
     * where it stood before the first move: the balance before that capture plus what the
     * capture nets by static exchange. Counting the exchange rather than the bare balance at
     * the end of the line means a line cut off mid-trade neither gains nor loses a piece.
     * `square` keeps to captures on that square; `upto` stops at that ply.
     */
    private fun winsMaterial(
        boards: List<Board>,
        played: List<Move>,
        mover: Boolean,
        square: Int? = null,
        upto: Int? = null,
    ): Boolean {
        val start = balance(boards[0], mover)
        val last = if (upto == null) played.size else minOf(played.size, upto + 1)
        for (k in 2 until last step 2) {
            val later = played[k]
            if (square != null && later.to != square) continue
            if (!boards[k].isCapture(later)) continue
            if (balance(boards[k], mover) + Exchange.exchange(boards[k], later) > start) return true
        }
        return false
    }

    /**
     * Walk the line, stopping at the first move that is not legal. `boards[i]` is the
     * position before `played[i]`; `boards[last]` is the end of the line.
     */
    private fun lineBoards(before: Board, moves: List<Move>): Pair<List<Board>, List<Move>> {
        val boards = arrayListOf(before.copy(withStack = false))
        val played = ArrayList<Move>()
        for (move in moves) {
            val current = boards[boards.size - 1]
            if (!current.isLegal(move)) break
            val next = current.copy(withStack = false)
            next.push(move)
            boards.add(next)
            played.add(move)
        }
        return boards to played
    }

    /** Deduplicate by name, merging evidence squares in first-seen order, then sort. */
    private fun finish(tags: List<MotifTag>): List<MotifTag> {
        val merged = LinkedHashMap<String, MutableList<String>>()
        val moves = LinkedHashMap<String, String?>()
        for (item in tags) {
            val current = merged[item.name]
            if (current == null) {
                merged[item.name] = ArrayList(item.squares)
                moves[item.name] = item.move
                continue
            }
            for (square in item.squares) if (square !in current) current.add(square)
            if (moves[item.name] == null) moves[item.name] = item.move
        }
        val fallback = ORDER_INDEX.size
        return merged.entries
            .map { MotifTag(it.key, it.value, moves[it.key]) }
            .sortedWith(compareBy({ ORDER_INDEX[it.name] ?: fallback }, { it.name }))
    }

    // --- group A: what the move is -----------------------------------------------------

    /**
     * A quiet move that saves material: it runs from an attack, or it defends an attacked
     * man.
     *
     * Running away means the piece could have been taken for profit where it stood and cannot
     * be on its new square. Defending means the move newly covers a friendly piece the
     * opponent could have won before the move, is still attacking after it, and can no longer
     * win. A guard that does not stop the loss is no rescue: a knight covering a rook that a
     * knight attacks still loses the exchange. Nor is one that is itself lost where it lands,
     * and adding a second guard to a piece that was already holding does not count either.
     */
    private fun defensiveEvidence(
        before: Board,
        after: Board,
        move: Move,
        mover: Boolean,
    ): List<Int>? {
        val enemy = !mover
        if (!isSafe(after, move.to)) return null
        if (!isSafe(before, move.from)) return listOf(move.from, move.to)
        val coveredBefore = before.attacksMask(move.from)
        for (sq in after.attacks(move.to)) {
            if (coveredBefore and BB_SQUARES[sq] != BB_EMPTY) continue
            val piece = after.pieceAt(sq) ?: continue
            if (piece.colour != mover) continue
            if (isSafe(before, sq) || Exchange.capturers(after, enemy, sq).isEmpty()) continue
            if (!isSafe(after, sq)) continue
            return listOf(move.to, sq)
        }
        return null
    }

    private fun groupA(before: Board, after: Board, move: Move, mover: Boolean): List<MotifTag> {
        val tags = ArrayList<MotifTag>()
        val frm = move.from
        val to = move.to
        val isEp = before.isEnPassant(move)
        val isCapture = before.isCapture(move)
        val givesCheck = after.isCheck()

        if (isCapture) {
            val squares = if (isEp) listOf(frm, to, epVictim(move)) else listOf(frm, to)
            tags.add(tag("capture", squares, move))
        }
        if (isEp) tags.add(tag("enPassant", listOf(frm, to, epVictim(move)), move))
        if (move.promotion != 0) {
            tags.add(tag("promotion", listOf(frm, to), move))
            if (move.promotion != QUEEN) tags.add(tag("underPromotion", listOf(frm, to), move))
        }
        if (before.isCastling(move)) {
            val rank = squareRank(frm)
            val kingside = before.isKingsideCastling(move)
            val rookFrom = square(if (kingside) 7 else 0, rank)
            val rookTo = square(if (kingside) 5 else 3, rank)
            tags.add(tag("castling", listOf(frm, to, rookFrom, rookTo), move))
        }
        if (before.pieceTypeAt(frm) == PAWN) {
            val rank = squareRank(to)
            if ((mover == WHITE && rank >= 5) || (mover == BLACK && rank <= 2)) {
                tags.add(tag("advancedPawn", listOf(frm, to), move))
            }
        }
        if (givesCheck) {
            val checkers = squaresOf(after.checkersMask())
            val king = after.king(!mover)
            val squares = if (king == null) checkers else checkers + king
            tags.add(tag("check", squares, move))
            if (checkers.size >= 2) tags.add(tag("doubleCheck", squares, move))
        }
        if (!isCapture && !givesCheck && move.promotion == 0) {
            // Bookkeeping, not a teaching claim: the move is not a capture, a check or a
            // promotion, and that is all. Lichess's quietMove is narrower (a quiet move that
            // makes no threat and still wins); the coach never names this one as a theme and
            // the panel hides it. It gates `defensiveMove` and records what kind of move it was.
            tags.add(tag("quietMove", listOf(frm, to), move))
            val evidence = defensiveEvidence(before, after, move, mover)
            if (evidence != null) tags.add(tag("defensiveMove", evidence, move))
        }
        return tags
    }

    // --- group B: what the move creates ------------------------------------------------

    /**
     * Would taking the piece on `square` with a `cost`-valued piece gain material?
     *
     * True when the piece is worth more than what takes it, or when nothing can legally take
     * back ([Exchange.defenders]: a defender pinned to its own king does not count, nor a king
     * that would step onto a guarded square). `ignore` drops one piece from the board first,
     * which is how a pinned or skewered piece stops counting as the guard of the piece behind
     * it: it is the one being driven off the line.
     */
    fun winnable(view: Board, sq: Int, cost: Int, ignore: Int? = null): Boolean {
        val piece = view.pieceAt(sq) ?: return false
        if (piece.pieceType == KING || value(piece.pieceType) > cost) return true
        var probe = view
        if (ignore != null) {
            probe = view.copy(withStack = false)
            probe.removePieceAt(ignore)
        }
        return Exchange.defenders(probe, piece.colour, sq).isEmpty()
    }

    /**
     * Can the piece on `square` be taken for profit? Safe means no, so it can stay. Read off
     * the static exchange with legal moves ([Exchange.canBeWon]): an attacker pinned to its own
     * king cannot take -- the knight forking king and queen is safe from the pawn that is
     * pinned (the "pin" of Lichess's fork puzzles) -- and an x-ray piece behind the first
     * attacker counts.
     */
    fun isSafe(view: Board, sq: Int): Boolean = !Exchange.canBeWon(view, sq)

    /**
     * Squares the mover captures an enemy piece on early in the line the engine expects.
     *
     * This lets a fork that wins its piece a move or two later (once a check has been
     * answered) be recognised on the board where the double attack is set up, using the
     * engine's own continuation rather than a guess -- the same line the multi-move motifs
     * already read.
     */
    private fun moverCaptures(
        boards: List<Board>,
        played: List<Move>,
        mover: Boolean,
        plies: Int = 6,
    ): Set<Int> {
        val squares = HashSet<Int>()
        for ((i, mv) in played.take(plies).withIndex()) {
            val board = boards[i]
            if (board.turn != mover) continue
            val victim = board.pieceAt(mv.to)
            if (victim != null && victim.colour != mover) squares.add(mv.to)
        }
        return squares
    }

    /**
     * The moved piece attacks two or more targets it could profitably take.
     *
     * A target is the enemy king, an enemy piece worth more than the moved piece, or an enemy
     * piece worth at least as much and undefended. Equal value plus a defender is a trade,
     * not a fork, which is what keeps pawn pushes out of this. The forker has to be able to
     * take a target by the rules: a piece pinned to its own king bears on the pieces around it
     * but may take none of them. And the forking piece must be safe where it stands, a check
     * included: a knight forking two rooks while a pawn can take it wins nothing, and a knight
     * checking king and queen that the g-pawn simply takes wins nothing either -- taking the
     * checker is a legal answer to the check, which the exchange with legal moves counts.
     *
     * A check is the exception to value: when the move gives check, a loose enemy piece the
     * forker also hits counts however cheap it is, because the enemy must answer the check
     * and cannot save the piece too. That single case is verified move by move
     * ([checkForkWins]) so a check that the enemy can meet while rescuing the piece is not
     * called a fork. This is the common queen-check-and-grab that plain value comparison
     * misses.
     */
    private fun fork(
        after: Board,
        move: Move,
        mover: Boolean,
        wonInLine: Set<Int> = emptySet(),
    ): MotifTag? {
        val piece = after.pieceAt(move.to) ?: return null
        if (piece.colour != mover) return null
        val enemy = !mover
        val mine = value(piece.pieceType)
        val givesCheck = after.isCheck()
        val targets = ArrayList<Int>()
        for (sq in after.attacks(move.to)) {
            val victim = after.pieceAt(sq) ?: continue
            if (victim.colour != enemy) continue
            val king = victim.pieceType == KING
            if (king || forkTarget(after, move.to, sq, mine, mover, givesCheck, wonInLine)) {
                targets.add(sq)
            }
        }
        if (targets.size < 2) return null
        if (!isSafe(after, move.to)) return null
        return tag("fork", listOf(move.to) + targets, move)
    }

    /**
     * Whether the enemy piece on `square` is one a fork from `forker` would win.
     *
     * Only a piece the forker can legally take ([Exchange.capturers]), and "undefended" means
     * nothing can legally take back ([Exchange.defenders]). A cheap loose piece counts when
     * the move gives check and either the piece cannot be saved against any reply
     * ([checkForkWins]) or the engine's own line actually captures it -- the latter settles
     * the cases where the enemy has a defence that only fails to deeper calculation, which is
     * the engine's job, not the geometry's.
     */
    private fun forkTarget(
        after: Board,
        forker: Int,
        sq: Int,
        mine: Int,
        mover: Boolean,
        givesCheck: Boolean,
        wonInLine: Set<Int>,
    ): Boolean {
        if (forker !in Exchange.capturers(after, mover, sq)) return false
        val enemy = !mover
        val theirs = value(after.pieceTypeAt(sq))
        val loose = Exchange.defenders(after, enemy, sq).isEmpty()
        if (theirs > mine || (theirs >= mine && loose)) return true
        return givesCheck && loose && (sq in wonInLine || checkForkWins(after, forker, sq, mover))
    }

    /**
     * After a checking move, is the loose enemy piece on `target` truly won?
     *
     * True only when no legal reply to the check rescues it: for every reply the piece is
     * still there, the forker can still legally take it, and taking it still gains material.
     * A reply that captures the forker, guards the piece, moves it to safety, or blocks with a
     * piece that pins the forker to its own king means the check-fork does not win it.
     */
    private fun checkForkWins(after: Board, forker: Int, target: Int, mover: Boolean): Boolean {
        val replies = after.legalMoves()
        if (replies.isEmpty()) return false  // checkmate: the mate tags cover this
        val enemy = !mover
        for (reply in replies) {
            val next = after.copy(withStack = false)
            next.push(reply)
            val victim = next.pieceAt(target)
            if (victim == null || victim.colour != enemy) return false  // moved, traded, taken
            val gain = captureGain(next, mover, forker, target)
                ?: return false  // the forker cannot take it now (captured, pinned, blocked)
            if (gain <= 0) return false  // it is defended now
        }
        return true
    }

    /**
     * Enemy pieces the mover can take with nothing to take back, one tag per piece.
     *
     * Read off legal moves ([Exchange.hanging]): a piece attacked only by a piece pinned to its
     * own king is not hanging, nor is one the mover cannot take because it has a check to
     * answer first. Pawns are excluded: loose pawns are everywhere and a coach calling one
     * "hanging" is noise. One tag per piece so that a piece that was already loose before a
     * move can be told apart from one the move just left en prise.
     */
    private fun hangingTags(view: Board, mover: Boolean): List<MotifTag> {
        val enemy = !mover
        val tags = ArrayList<MotifTag>()
        for (sq in view.occupiedSquares()) {
            val piece = view.pieceAt(sq) ?: continue
            if (piece.colour != enemy) continue
            if (piece.pieceType == PAWN || piece.pieceType == KING) continue
            if (Exchange.hanging(view, sq)) tags.add(tag("hangingPiece", listOf(sq)))
        }
        return tags
    }

    /**
     * Pins and skewers: one mover slider, one enemy piece, one enemy piece behind it.
     *
     * A pin is tactical, not only geometric: an enemy piece with a more valuable enemy piece
     * behind it on the slider's line is pinned when leaving the line would lose that piece --
     * the king always, another piece when the slider could legally take it for profit once
     * the front piece has gone. A pin against a defended piece worth no more than the pinner
     * wins nothing, and the coach said such pins made no sense (Tai, 2026-09-23). A pin also
     * has to hold something back: a pawn on the file whose only move is straight ahead is not
     * stopped from anything ([leavesLine]).
     *
     * Neither holds when the enemy simply wins the slider ([isSafe]): a rook "pinning" a rook
     * that takes it with check, or a bishop pinning a knight while a pawn takes the bishop. A
     * slider pinned to its own king can never take what is behind, so it pins nothing but the
     * king: a pin to the king stays, because the front piece still cannot move by the rules,
     * but a relative pin needs the slider to be able to cash it in.
     *
     * A skewer names a piece to go after, so it requires the piece behind to be winnable, and
     * it must still be winnable whatever the front piece does ([skewerHolds]): a queen that
     * steps off the line to guard the rook behind has answered it. The x-ray is a line motif,
     * not a static one, and lives in [xray].
     */
    private fun rayMotifs(view: Board, mover: Boolean): List<MotifTag> {
        val enemy = !mover
        val tags = ArrayList<MotifTag>()
        for (slider in sliders(view, mover)) {
            if (!isSafe(view, slider)) continue
            val cost = value(view.pieceTypeAt(slider))
            for (frontSq in view.attacks(slider)) {
                val front = view.pieceAt(frontSq) ?: continue
                if (front.colour != enemy) continue
                val backSq = nextBeyond(view, slider, frontSq) ?: continue
                val back = view.pieceAt(backSq) ?: continue
                if (back.colour != enemy) continue
                val squares = listOf(slider, frontSq, backSq)
                val frontValue = value(front.pieceType)
                val backValue = value(back.pieceType)
                if (backValue > frontValue) {
                    if (!leavesLine(view, slider, frontSq, backSq)) continue
                    if (back.pieceType == KING) {
                        tags.add(tag("pin", squares))
                        tags.add(tag("absolutePin", squares))
                        continue
                    }
                    val probe = view.copy(withStack = false)
                    probe.removePieceAt(frontSq)
                    val gain = captureGain(probe, mover, slider, backSq)
                    if (gain != null && gain > 0) tags.add(tag("pin", squares))
                    continue
                }
                if (frontValue > backValue) {
                    val mustMove = winnable(view, frontSq, cost)
                    val behindIsLoose = winnable(view, backSq, cost, ignore = frontSq)
                    if (mustMove && behindIsLoose && skewerHolds(view, mover, slider, frontSq, backSq)) {
                        tags.add(tag("skewer", squares))
                    }
                }
            }
        }
        return tags
    }

    /**
     * Whether the piece on `frontSq` has a move off the slider's line, the pin ignored.
     *
     * A pin restricts only a piece that had somewhere else to go: the pawn in front of the
     * queen on a file whose only move is a step down that file is held back from nothing.
     * Pseudo-legal moves are what the piece could do were it not pinned.
     */
    private fun leavesLine(view: Board, slider: Int, frontSq: Int, backSq: Int): Boolean {
        val front = view.pieceAt(frontSq) ?: return false
        val line = ray(slider, backSq)
        val side = asSide(view, front.colour)
        var off = false
        side.generatePseudoLegalMoves(BB_SQUARES[frontSq], BB_ALL) { m ->
            if (BB_SQUARES[m.to] and line == BB_EMPTY) off = true
        }
        return off
    }

    /**
     * Whether every way the front piece can step off the line still loses the piece behind.
     *
     * For each legal move of the front piece off the line, the slider must still be able to
     * take the piece behind for a gain ([Exchange.exchange] > 0). A queen that answers a rook
     * skewer by stepping to a square that guards the rook behind has met it: Rxb8 Qxb8 is only
     * a trade. A front piece with no way off the line is not skewered but simply lost, which
     * the loose and trapped pieces cover.
     */
    private fun skewerHolds(view: Board, mover: Boolean, slider: Int, frontSq: Int, backSq: Int): Boolean {
        val front = view.pieceAt(frontSq) ?: return false
        val line = ray(slider, backSq)
        val side = asSide(view, front.colour)
        val escapes = ArrayList<Move>()
        // The front piece's legal moves, in the order the whole list has them.
        side.generateLegalMoves(BB_SQUARES[frontSq], BB_ALL) { m ->
            if (BB_SQUARES[m.to] and line == BB_EMPTY) escapes.add(m)
        }
        if (escapes.isEmpty()) return false
        for (escape in escapes) {
            val next = side.copy(withStack = false)
            next.push(escape)
            val gain = captureGain(next, mover, slider, backSq)
            if (gain == null || gain <= 0) return false
        }
        return true
    }

    /**
     * The move steps off a line and a friendly slider now bears on something it can win.
     *
     * The geometry first: the vacated square lies strictly between the slider and an enemy
     * piece it now attacks and did not before. Then the claim: the revealed piece is the king
     * (a discovered check), or the slider could take it for a gain ([Exchange.exchange] > 0),
     * or the slider does take it later in the line -- Lichess tags a discovery only when that
     * capture happens. Revealing a bishop onto a pawn the g-pawn guards, as 2.d4 does to h6,
     * is a line opening, not a discovered attack.
     */
    private fun discoveredAttack(boards: List<Board>, played: List<Move>, mover: Boolean): MotifTag? {
        val before = boards[0]
        val after = boards[1]
        val move = played[0]
        val enemy = !mover
        // The mover's later captures, as (from, to) packed into one number.
        val later = HashSet<Int>()
        for (k in 2 until played.size step 2) {
            if (boards[k].isCapture(played[k])) later.add(played[k].from * 64 + played[k].to)
        }
        val squares = ArrayList<Int>()
        for (slider in sliders(after, mover)) {
            if (slider == move.to || before.pieceAt(slider) != after.pieceAt(slider)) continue
            val was = before.attacksMask(slider)
            for (target in after.attacks(slider)) {
                if (was and BB_SQUARES[target] != BB_EMPTY) continue
                val piece = after.pieceAt(target) ?: continue
                if (piece.colour != enemy) continue
                if (BB_SQUARES[move.from] and between(slider, target) == BB_EMPTY) continue
                if (piece.pieceType != KING && (slider * 64 + target) !in later) {
                    val gain = captureGain(after, mover, slider, target)
                    if (gain == null || gain <= 0) continue
                }
                if (slider !in squares) squares.add(slider)
                squares.add(target)
            }
        }
        if (squares.isEmpty()) return null
        return tag("discoveredAttack", listOf(move.from) + squares, move)
    }

    /**
     * Cutting a line of defence, the way Lichess tags it, read over the line and by either side.
     *
     * Two shapes. The mover interposes between an enemy slider and a piece it was defending,
     * cutting the guard and winning that piece ([moverInterference]). Or the mover forces the
     * enemy -- with a check, or by taking so that the reply is the recapture -- to answer with
     * a move that blocks one of the enemy's own sliders off a square it was guarding, a square
     * the mover takes or mates on with its very next move ([selfInterference]). The forced
     * self-block is the common case; the mover's own interposition is the textbook one.
     */
    private fun interference(boards: List<Board>, played: List<Move>, mover: Boolean): MotifTag? {
        for ((i, move) in played.withIndex()) {
            val before = boards[i]
            val after = boards[i + 1]
            val found = if (i % 2 == 0) {
                val taken = HashSet<Int>()
                for (k in i + 2 until played.size step 2) {
                    if (boards[k].isCapture(played[k])) taken.add(played[k].to)
                }
                moverInterference(before, after, move, mover, taken)
            } else {
                val recapture = before.isCapture(move) && move.to == played[i - 1].to
                if (!before.isCheck() && !recapture) continue  // a free reply interfered under no force
                val exploited = HashSet<Int>()
                if (i + 1 < played.size) {
                    val next = played[i + 1]
                    if (boards[i + 1].isCapture(next) || boards[i + 2].isCheckmate()) exploited.add(next.to)
                }
                selfInterference(before, after, move, !mover, exploited)
            }
            if (found != null) return found
        }
        return null
    }

    /**
     * The mover lands on the line between an enemy slider and the piece it was defending.
     *
     * Geometric plus a before/after test with legal moves: the defended piece loses its last
     * defender ([Exchange.defenders]) and the mover can now take it for a gain -- which a piece
     * pinned to its own king cannot, even standing on the line. And the interposer must hold
     * its square ([isSafe]), or the line must go on to take the piece it cut off (`taken`, the
     * squares the mover captures on later): a knight dropped between rook and pawn that the
     * rook simply takes has cut nothing.
     */
    private fun moverInterference(
        before: Board,
        after: Board,
        move: Move,
        mover: Boolean,
        taken: Set<Int>,
    ): MotifTag? {
        val enemy = !mover
        val holds = isSafe(after, move.to)
        for (slider in sliders(before, enemy)) {
            if (slider == move.to || before.pieceAt(slider) != after.pieceAt(slider)) continue
            for (guarded in before.attacks(slider)) {
                if (guarded == move.to) continue
                val piece = before.pieceAt(guarded) ?: continue
                if (piece.colour != enemy || piece.pieceType == KING) continue
                if (BB_SQUARES[move.to] and between(slider, guarded) == BB_EMPTY) continue
                if (after.attacksMask(slider) and BB_SQUARES[guarded] != BB_EMPTY ||
                    Exchange.defenders(after, enemy, guarded).isNotEmpty()
                ) {
                    continue
                }
                if (Exchange.bestCapture(after, mover, guarded) <= 0) continue
                if (!holds && guarded !in taken) continue
                return tag("interference", listOf(move.to, slider, guarded), move)
            }
        }
        return null
    }

    /**
     * A forced reply that blocks one of the replier's own sliders off a square it was guarding.
     *
     * The block lands on `reply.to`; a `defender` slider that reached that square cleanly and
     * guarded a square the mover takes or mates on next (`exploited`) no longer guards it
     * afterwards. That is the enemy interfering with its own defence. [interference] hands in
     * only forced replies -- a check evasion or a recapture -- and only the next move's square:
     * a knight that retreats of its own accord in front of its queen interferes with nothing.
     */
    private fun selfInterference(
        before: Board,
        after: Board,
        reply: Move,
        defender: Boolean,
        exploited: Set<Int>,
    ): MotifTag? {
        val block = reply.to
        for (slider in sliders(before, defender)) {
            if (slider == block || !sliderReaches(before.pieceTypeAt(slider), slider, block)) continue
            if (before.occupied and between(slider, block) != BB_EMPTY) continue
            for (target in beyond(slider, block)) {
                if (before.pieceAt(target) != null && target !in exploited) break
                if (target in exploited &&
                    slider in Exchange.defenders(before, defender, target) &&
                    after.attackersMask(defender, target) and BB_SQUARES[slider] == BB_EMPTY
                ) {
                    return tag("interference", listOf(target, block, slider), reply)
                }
                if (before.pieceAt(target) != null) break
            }
        }
        return null
    }

    /** Squares continuing along the frm -> through ray, past `through` to the board edge. */
    private fun beyond(frm: Int, through: Int): List<Int> {
        val df = sign(squareFile(through) - squareFile(frm))
        val dr = sign(squareRank(through) - squareRank(frm))
        var file = squareFile(through) + df
        var rank = squareRank(through) + dr
        val out = ArrayList<Int>()
        while (file in 0..7 && rank in 0..7) {
            out.add(square(file, rank))
            file += df
            rank += dr
        }
        return out
    }

    /**
     * An enemy piece the mover can win that no move of the enemy's saves.
     *
     * The piece can be taken for profit now ([Exchange.canBeWon], legal moves: a hunter pinned
     * to its own king hunts nothing), and after every legal enemy move it still can
     * ([rescues]): not by fleeing, not by guarding it, not by taking the hunter with something
     * else, not by a check. A flight square that is attacked but held is a flight square: the
     * knight on h6 hit by a pawn is not trapped while f5 and g4 are guarded. Minor pieces
     * count, not only rooks and queens. Needs the enemy to move, so it is skipped whenever
     * either king is in check: check evasions would make every piece look trapped. One tag per
     * piece, so a piece that was already trapped before a move can be told apart from one the
     * move just shut in.
     */
    private fun trappedTags(view: Board, mover: Boolean): List<MotifTag> {
        val enemy = !mover
        val enemyKing = view.king(enemy) ?: return emptyList()
        val moverKing = view.king(mover) ?: return emptyList()
        if (view.attackersMask(mover, enemyKing) != BB_EMPTY) return emptyList()
        if (view.attackersMask(enemy, moverKing) != BB_EMPTY) return emptyList()
        val replies = view.legalMoves()
        val trapped = ArrayList<MotifTag>()
        for (sq in view.occupiedSquares()) {
            val piece = view.pieceAt(sq) ?: continue
            if (piece.colour != enemy || value(piece.pieceType) < MINOR) continue
            if (!Exchange.canBeWon(view, sq)) continue
            // The piece's own moves first: fleeing is the usual rescue, and the search stops
            // there. The rest keep their order, as the desktop's stable sort keeps it.
            var saved = false
            for (reply in replies) {
                if (reply.from == sq && rescues(view, reply, sq, mover)) { saved = true; break }
            }
            if (!saved) {
                for (reply in replies) {
                    if (reply.from != sq && rescues(view, reply, sq, mover)) { saved = true; break }
                }
            }
            if (saved) continue
            val hunters = Exchange.capturers(view, mover, sq)
            trapped.add(tag("trappedPiece", listOf(sq) + hunters))
        }
        return trapped
    }

    /**
     * Whether the enemy `reply` saves the piece on `square` from being won.
     *
     * It does when, after the reply, the mover can no longer take the piece (wherever it went)
     * for more than the reply itself took: a piece that flees to a held square, a hunter taken,
     * a check to answer first, or a capture worth at least what is lost -- the old rule that a
     * trapped piece taking something as valuable on its way out is not trapped, and a queen
     * that wins a rook elsewhere has not lost her knight for nothing. A queen taking a rook on
     * a1 that the bishop takes back has still lost herself.
     */
    private fun rescues(view: Board, reply: Move, sq: Int, mover: Boolean): Boolean {
        val next = view.copy(withStack = false)
        next.push(reply)
        val where = if (reply.from == sq) reply.to else sq
        val taken = if (view.isEnPassant(reply)) PAWN else view.pieceTypeAt(reply.to)
        var gained = value(taken)
        if (reply.promotion != 0) gained += value(reply.promotion) - value(PAWN)
        return Exchange.bestCapture(next, mover, where) <= gained
    }

    /**
     * The move takes aim at the beginner's weak pawn: f7 as white, f2 as black.
     *
     * The pawn itself, not whatever stands on the square: a rook checking a king that has
     * walked to f7 in the endgame is a check, not this. And the aim must be real: the pawn can
     * be won ([Exchange.bestCapture] > 0) or the engine's line takes it (`wonInLine`). A bishop
     * eyeing f7 in a book Italian, with the king and queen holding it, only looks.
     */
    private fun attackingF2F7(
        before: Board,
        after: Board,
        move: Move,
        mover: Boolean,
        wonInLine: Set<Int> = emptySet(),
    ): MotifTag? {
        val target = if (mover == WHITE) 53 else 13  // f7, f2
        val piece = after.pieceAt(target) ?: return null
        if (piece.colour == mover || piece.pieceType != PAWN) return null
        if (after.attacksMask(move.to) and BB_SQUARES[target] == BB_EMPTY) return null
        if (before.attacksMask(move.from) and BB_SQUARES[target] != BB_EMPTY) return null
        if (Exchange.bestCapture(after, mover, target) <= 0 && target !in wonInLine) return null
        return tag("attackingF2F7", listOf(move.to, target), move)
    }

    /**
     * The enemy king has lost its pawn shelter and the mover has force bearing on it.
     *
     * Shelter is counted one pawn per file over the three files around the king, one or two
     * ranks in front. At most one such pawn, at least two of the mover's pieces actually
     * attacking the king's ring (one of them a queen or rook), and at least two ring squares
     * under fire. Counting pieces on the ring rather than pieces on the board is what keeps an
     * opening king that just pushed a pawn from looking besieged. Without a queen of the
     * mover's there is no attack to speak of: a king walking up the board in a rook and knight
     * ending is active, not exposed.
     */
    private fun exposedKing(view: Board, mover: Boolean): MotifTag? {
        val enemy = !mover
        val king = view.king(enemy) ?: return null
        if (view.piecesMask(QUEEN, mover) == BB_EMPTY) return null
        val kingFile = squareFile(king)
        val kingRank = squareRank(king)
        val step = if (enemy == WHITE) 1 else -1
        var shelter = 0
        for (df in intArrayOf(-1, 0, 1)) {
            val file = kingFile + df
            if (file !in 0..7) continue
            for (distance in intArrayOf(1, 2)) {
                val rank = kingRank + step * distance
                if (rank !in 0..7) continue
                val piece = view.pieceAt(square(file, rank))
                if (piece != null && piece.colour == enemy && piece.pieceType == PAWN) {
                    shelter += 1
                    break
                }
            }
        }
        if (shelter > 1) return null
        val ring = squaresOf(BB_KING_ATTACKS[king])
        val hit = ring.filter { view.isAttackedBy(mover, it) }
        if (hit.size < 2) return null
        // The force has to be bearing on the king's ring, not merely present on the board: an
        // opening king that has just lost a pawn is not exposed while every piece is at home.
        val ringAttackers = HashSet<Int>()
        var hasHeavy = false
        for (sq in ring) {
            for (attacker in view.attackers(mover, sq)) {
                val pieceType = view.pieceTypeAt(attacker)
                if (pieceType == QUEEN || pieceType == ROOK || pieceType == BISHOP || pieceType == KNIGHT) {
                    ringAttackers.add(attacker)
                    hasHeavy = hasHeavy || pieceType == QUEEN || pieceType == ROOK
                }
            }
        }
        if (ringAttackers.size < 2 || !hasHeavy) return null
        return tag("exposedKing", listOf(king) + hit)
    }

    /**
     * Everything readable from the attack maps alone, from `mover`'s point of view. `view`
     * must have the enemy to move, which is what the position after a move looks like.
     */
    private fun staticTags(view: Board, mover: Boolean): List<MotifTag> {
        val tags = ArrayList(rayMotifs(view, mover))
        tags.addAll(hangingTags(view, mover))
        tags.addAll(trappedTags(view, mover))
        exposedKing(view, mover)?.let { tags.add(it) }
        return tags
    }

    private fun enemyToMove(board: Board, mover: Boolean): Board {
        if (board.turn != mover) return board
        val view = board.copy(withStack = false)
        view.turn = !mover
        view.epSquare = null
        return view
    }

    private val RAY_NAMES = setOf("pin", "absolutePin", "skewer")

    /**
     * Identity of a static feature, so the same feature before and after a move matches. A
     * pin is which piece is shielding which, not which of my pieces happens to be doing the
     * pinning; the king being out in the open is the feature, not which ring squares are hit
     * this ply; and a trapped piece is the piece with nowhere to go, not its hunters.
     */
    private fun staticKey(item: MotifTag): Pair<String, List<String>> = when {
        item.name in RAY_NAMES -> item.name to item.squares.drop(1)
        item.name == "exposedKing" || item.name == "trappedPiece" -> item.name to item.squares.take(1)
        else -> item.name to item.squares
    }

    /**
     * Static features the move brought about, and only those. A pin the bishop was already
     * making before the move is not something the move did; it is still reported by
     * [tagsForPosition].
     */
    private fun newStaticTags(before: Board, after: Board, mover: Boolean): List<MotifTag> {
        val prior = staticTags(enemyToMove(before, mover), mover).map { staticKey(it) }.toSet()
        return staticTags(after, mover).filter { staticKey(it) !in prior }
    }

    // --- group C: what the line does ---------------------------------------------------

    /**
     * This move gives up at least two points to the opponent's reply, and the line still
     * wins. The judgement is about the move played, not about the line it leads to: the
     * balance is read after the move and the opponent's answer to it. Two points, so a pawn
     * gambit does not count; the mover must still be behind after its own next move (a knight
     * given for two pawns is a piece sacrificed: Tai's puzzle, 2026-10-06, Nxg5 hxg5 Bxg5);
     * and "still wins" is a mate for the mover, or the material back by the end of the line.
     *
     * Or the material is offered and the line declines it ([offered]): chess.com's
     * brilliant, a piece left to be taken whether or not it is (after Nxg5 the engine's
     * line declines the knight, hxg5 being a mistake). `_sacrifice`.
     */
    private fun sacrifice(
        boards: List<Board>,
        played: List<Move>,
        mover: Boolean,
        mateIn: Int?,
    ): MotifTag? {
        if (played.isEmpty()) return null
        val start = balance(boards[0], mover)
        val mating = mateIn != null && mateIn > 0
        if (!mating && balance(boards.last(), mover) < start) return null
        if (givenInLine(boards, played, mover, start)) {
            return tag("sacrifice", listOf(played[0].from, played[0].to, played[1].to), played[0])
        }
        val offered = offered(boards[0], boards[1], played[0], mover) ?: return null
        return tag("sacrifice", listOf(played[0].from, played[0].to, offered), played[0])
    }

    /**
     * The line takes what the move gives ([sacrifice]): two points down after the reply,
     * still behind after the mover's next move, and the reply's capture a gain once the
     * recaptures on its square are played out. `_given_in_line`.
     */
    private fun givenInLine(boards: List<Board>, played: List<Move>, mover: Boolean, start: Int): Boolean {
        if (played.size < 2) return false
        val answered = balance(boards[2], mover)
        if (start - answered < SACRIFICE_THRESHOLD) return false
        if (boards.size > 3 && balance(boards[3], mover) >= start) return false
        // The reply's capture wins it, the recaptures on its square played out: a knight taken
        // and taken back is a trade, however the line orders it (O-O Bxf3 Bb5+ Nc6 Bxc6+ bxc6
        // Qxf3, a check before the recapture, read as castling giving up a knight).
        return Exchange.exchange(boards[1], played[1]) >= SACRIFICE_THRESHOLD
    }

    /**
     * The square of a piece of [mover]'s the move left to be won ([sacrifice]): a piece, not a
     * pawn, the opponent could take for more than the move itself took -- the mover a point or
     * more down once the recaptures are played out -- and for more than the same piece could
     * have been won before the move. A capture taken back is a trade (Bxc3 bxc3); a knight
     * for two pawns is not (Nxg5 hxg5 Bxg5). Each piece against itself: a move that answers
     * one threat may offer something else at once. Null when the move offers nothing.
     * `_offered`.
     */
    private fun offered(before: Board, after: Board, move: Move, mover: Boolean): Int? {
        val took = if (before.isEnPassant(move)) {
            Exchange.VALUE[PAWN]
        } else {
            val taken = before.pieceTypeAt(move.to)
            if (taken != 0) Exchange.VALUE[taken] else 0
        }
        val mine = after.occupiedCo[if (mover) 1 else 0] and after.kings.inv() and after.pawns.inv()
        for (square in 0 until 64) {
            if (mine and BB_SQUARES[square] == BB_EMPTY) continue
            val won = Exchange.bestCapture(after, !mover, square)
            val kind = after.pieceTypeAt(square).let { if (it == 0) PAWN else it }
            if (won - took < 1 || Exchange.VALUE[kind] < MINOR) continue
            val was = if (square == move.to) move.from else square
            if (before.pieceAt(was) == null || Exchange.bestCapture(before, !mover, was) < won) {
                return square
            }
        }
        return null
    }

    /**
     * The move vacates a square with tempo and another piece soon uses that square with force.
     *
     * Tempo means the move is a capture or a check, which is what separates a clearance from a
     * piece that merely happened to be in the way. The square is cleared for another piece: a
     * queen that grabs a pawn and comes back to where it stood cleared nothing. Soon means one
     * of the mover's next two moves, and the use must itself be a capture, a check or mate, or
     * set up what the line goes on to win or mate with (Bxe5 Bxe5 Qd5, the queen's fork that
     * takes e5 a move later) -- a rook strolling down a file the knight left long ago is not
     * what the knight's move was for.
     */
    private fun clearance(boards: List<Board>, played: List<Move>): MotifTag? {
        val mover = boards[0].turn
        if (played.size < 3) return null
        val first = played[0]
        if (!boards[0].isCapture(first) && !boards[1].isCheck()) return null
        val vacated = first.from
        var clearedPiece = first.to
        for (i in 2 until minOf(played.size, 5) step 2) {
            val later = played[i]
            if (later.from == clearedPiece) {
                clearedPiece = later.to  // the piece that cleared the square, moving on
                continue
            }
            val uses = later.to == vacated ||
                BB_SQUARES[vacated] and between(later.from, later.to) != BB_EMPTY
            if (!uses) continue
            val forcing = boards[i].isCapture(later) || boards[i + 1].isCheck()
            if (forcing || mates(boards, mover) || winsMaterial(boards, played, mover)) {
                return tag("clearance", listOf(vacated, later.from, later.to), first)
            }
        }
        return null
    }

    /**
     * The move captures a defender, and a later move of the mover's wins what it defended.
     *
     * The captured piece must have been a legal defender of the later square at the start
     * ([Exchange.defenders]), and the piece there must be one the mover could not win before
     * ([Exchange.bestCapture] is 0) but wins at that point ([Exchange.exchange] > 0), with the
     * line ahead of where it started once that capture is counted: removing a knight that
     * guarded f6 when the g-pawn guards it too wins nothing, and giving a rook for the knight
     * that guarded a pawn loses more than the pawn. A move onto a square the defender guarded
     * counts without material only when it is a check and the line ends in mate.
     */
    private fun capturingDefender(boards: List<Board>, played: List<Move>, mover: Boolean): MotifTag? {
        if (played.isEmpty()) return null
        val first = played[0]
        if (!boards[0].isCapture(first)) return null
        val enemy = !mover
        val victimSquare = if (boards[0].isEnPassant(first)) epVictim(first) else first.to
        val start = balance(boards[0], mover)
        val mating = mates(boards, mover)
        for (i in 2 until played.size step 2) {
            val later = played[i]
            val sq = later.to
            if (victimSquare !in Exchange.defenders(boards[0], enemy, sq)) continue
            val position = boards[i]
            if (mating && boards[i + 1].isCheck()) {
                return tag("capturingDefender", listOf(victimSquare, sq), first)
            }
            if (!position.isCapture(later)) continue
            if (Exchange.bestCapture(boards[0], mover, sq) > 0) continue  // it could be won before
            val gain = Exchange.exchange(position, later)
            if (gain <= 0 || balance(position, mover) + gain <= start) continue
            return tag("capturingDefender", listOf(victimSquare, sq), first)
        }
        return null
    }

    /**
     * A forcing move pulls the only defender of a square away, and the mover wins there.
     *
     * Forcing means the move is a capture or a check, and the reply is what it forces: the
     * recapture on the move's square, or an answer to the check. A queen that walks away of
     * her own accord was not deflected. "Only defender" is checked on the position after the
     * move, by legal moves ([Exchange.defenders]). And the follow-up must win: a capture that
     * nets material ([Exchange.exchange] > 0) with the line ahead of where it started, or
     * mate. A rook check on the square the queen left, answered by a king step, wins nothing.
     */
    private fun deflection(boards: List<Board>, played: List<Move>, mover: Boolean): MotifTag? {
        if (played.size < 3) return null
        val first = played[0]
        val reply = played[1]
        val follow = played[2]
        if (!boards[0].isCapture(first) && !boards[1].isCheck()) return null
        val recapture = boards[1].isCapture(reply) && reply.to == first.to
        if (!recapture && !boards[1].isCheck()) return null
        val defender = reply.from
        val guards = Exchange.defenders(boards[1], !mover, follow.to)
        if (guards.size != 1 || defender !in guards) return null
        if (boards[1].attacksMask(defender) and BB_SQUARES[follow.to] == BB_EMPTY) return null
        if (boards[2].attacksMask(reply.to) and BB_SQUARES[follow.to] != BB_EMPTY) return null
        if (!boards[3].isCheckmate() && !mates(boards, mover)) {
            val gain = Exchange.exchange(boards[2], follow)
            if (gain <= 0 || balance(boards[2], mover) + gain <= balance(boards[0], mover)) return null
        }
        return tag("deflection", listOf(defender, reply.to, follow.to), first)
    }

    /**
     * A forcing move drags an enemy piece onto a square where the next move hits it.
     *
     * The enemy is dragged by taking the piece the move put there, so the reply must capture
     * on the move's square; the piece pulled is the king, a queen or a rook, as Lichess has it
     * (a knight taking back a knight is a trade, not a lure); and the mover's follow-up must
     * land on that square or attack it while capturing or checking. Then the lure has to pay:
     * the line ends in mate, or wins material -- by taking the lured queen or rook on its
     * square within the mover's next two moves, or, for a king, anywhere in the line. Material
     * is counted from before the first move, so the piece given as bait is paid back first.
     */
    private fun attraction(boards: List<Board>, played: List<Move>, mover: Boolean): MotifTag? {
        if (played.size < 3) return null
        val first = played[0]
        val reply = played[1]
        val follow = played[2]
        if (!boards[0].isCapture(first) && !boards[1].isCheck()) return null
        if (!boards[1].isCapture(reply) || reply.to != first.to) return null
        val landed = reply.to
        val attracted = boards[2].pieceAt(landed) ?: return null
        if (attracted.colour == mover) return null
        if (attracted.pieceType != KING && attracted.pieceType != QUEEN && attracted.pieceType != ROOK) return null
        if (!boards[2].isCapture(follow) && !boards[2].givesCheck(follow)) return null
        val hits = follow.to == landed || boards[3].attacksMask(follow.to) and BB_SQUARES[landed] != BB_EMPTY
        if (!hits) return null
        val pays = if (attracted.pieceType == KING) {
            winsMaterial(boards, played, mover)
        } else {
            winsMaterial(boards, played, mover, square = landed, upto = 4)
        }
        if (!pays && !mates(boards, mover)) return null
        return tag("attraction", listOf(reply.from, landed, follow.to), first)
    }

    private fun mateSquares(position: Board): List<Int> {
        val king = position.king(position.turn)
        val checkers = squaresOf(position.checkersMask())
        return if (king == null) checkers else checkers + king
    }

    private fun mateTags(after: Board, final: Board, mover: Boolean, mateIn: Int?): List<MotifTag> {
        val distance = if (after.isCheckmate()) 1 else mateIn
        if (distance == null || distance <= 0) return emptyList()
        val squares = when {
            after.isCheckmate() -> mateSquares(after)
            final.isCheckmate() && final.turn != mover -> mateSquares(final)
            else -> after.king(!mover)?.let { listOf(it) } ?: emptyList()
        }
        val tags = arrayListOf(tag("mate", squares))
        if (distance in 1..5) tags.add(tag("mateIn$distance", squares))
        return tags
    }

    // --- x-rays ------------------------------------------------------------------------

    /** Whether a slider of this type on `frm` moves along the straight line to `to`. */
    private fun sliderReaches(pieceType: Int, frm: Int, to: Int): Boolean {
        val sameRank = squareRank(frm) == squareRank(to)
        val sameFile = squareFile(frm) == squareFile(to)
        val onDiag = abs(squareFile(frm) - squareFile(to)) == abs(squareRank(frm) - squareRank(to))
        return when (pieceType) {
            ROOK -> sameRank || sameFile
            BISHOP -> onDiag
            QUEEN -> sameRank || sameFile || onDiag
            else -> false
        }
    }

    /** A mover slider bearing on `target` through exactly one enemy piece: (slider, blocker). */
    private fun xrayTo(board: Board, mover: Boolean, target: Int): Pair<Int, Int>? {
        for (slider in sliders(board, mover)) {
            if (!sliderReaches(board.pieceTypeAt(slider), slider, target)) continue
            val blockers = squaresOf(between(slider, target) and board.occupied)
            if (blockers.size != 1) continue
            val blocker = blockers[0]
            if (board.pieceAt(blocker)!!.colour != mover) return slider to blocker
        }
        return null
    }

    /**
     * Lichess's x-ray: a piece attacks a square through an enemy piece.
     *
     * Read over the line, the way the sacrifice and clearance motifs are: a square the mover
     * captures or mates on was, at some earlier point in the line, borne on by a mover slider
     * through exactly one enemy piece, and by the time the square is taken the slider is still
     * there with its ray to the square open, and either makes the capture itself or stands
     * behind the piece that does, guarding it. The ray opening is what turns a latent x-ray
     * into a real one, so a slider that merely looks through a piece that never moves is not
     * called one -- nor a rook whose own knight took the blocker's place on the file.
     */
    private fun xray(boards: List<Board>, played: List<Move>, mover: Boolean): MotifTag? {
        for (k in 0 until played.size step 2) {  // the mover's own moves
            val move = played[k]
            val boardK = boards[k]
            val afterK = boards[k + 1]
            if (!(boardK.isCapture(move) || afterK.isCheckmate())) continue
            val target = move.to
            for (j in 0..k step 2) {  // a mover position at or before this move
                val found = xrayTo(boards[j], mover, target) ?: continue
                val (slider, blocker) = found
                if (boardK.pieceAt(slider) != boards[j].pieceAt(slider)) continue  // moved or taken
                val takes = move.from == slider
                if (!takes && afterK.attacksMask(slider) and BB_SQUARES[target] == BB_EMPTY) {
                    continue  // the ray is still shut, by the blocker or by a piece of ours
                }
                return tag("xRayAttack", listOf(target, blocker, slider), move)
            }
        }
        return null
    }

    // --- public API --------------------------------------------------------------------

    /**
     * What the move does, from the mover's point of view.
     *
     * `pv` is the continuation the engine expects AFTER `move` (it does not include `move`).
     * `mateIn` counts the mover's own moves including this one. An illegal move returns an
     * empty list rather than raising: the caller feeds this from a vision pipeline that can
     * misread a board, and a bad read should not crash a live game.
     */
    override fun tagsForMove(
        before: Board,
        move: Move,
        pv: List<Move>,
        mateIn: Int?,
    ): List<MotifTag> {
        if (!before.isLegal(move)) return emptyList()
        val mover = before.turn
        val (boards, played) = lineBoards(before, listOf(move) + pv)
        val after = boards[1]

        val wonInLine = moverCaptures(boards, played, mover)
        val tags = ArrayList(groupA(before, after, move, mover))
        tags.addAll(newStaticTags(before, after, mover))
        for (found in listOf(
            fork(after, move, mover, wonInLine = wonInLine),
            discoveredAttack(boards, played, mover),
            xray(boards, played, mover),
            interference(boards, played, mover),
            attackingF2F7(before, after, move, mover, wonInLine = wonInLine),
            sacrifice(boards, played, mover, mateIn),
            clearance(boards, played),
            capturingDefender(boards, played, mover),
            deflection(boards, played, mover),
            attraction(boards, played, mover),
        )) {
            if (found != null) tags.add(found)
        }
        tags.addAll(mateTags(after, boards.last(), mover, mateIn))

        if (after.isCheckmate()) {
            tags.addAll(Mates.matingPatternTags(after, move))
        } else if (boards.last().isCheckmate() && boards.last().turn != mover) {
            tags.addAll(Mates.matingPatternTags(boards.last(), played.last()))
        }
        return finish(tags)
    }

    /**
     * What is in the position for the side to move: the static features plus the tags of the
     * engine's best move, which is `pv[0]` when a PV is given.
     */
    override fun tagsForPosition(
        board: Board,
        pv: List<Move>,
        mateIn: Int?,
    ): List<MotifTag> {
        val mover = board.turn
        val tags = ArrayList(staticTags(enemyToMove(board, mover), mover))

        if (pv.isNotEmpty() && board.isLegal(pv[0])) {
            tags.addAll(tagsForMove(board, pv[0], pv.drop(1), mateIn))
        } else if (mateIn != null && mateIn > 0) {
            val king = board.king(!mover)
            val squares = if (king == null) emptyList() else listOf(king)
            tags.add(tag("mate", squares))
            if (mateIn <= 5) tags.add(tag("mateIn$mateIn", squares))
        }
        return finish(tags)
    }
}
