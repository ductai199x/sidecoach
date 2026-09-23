package com.sidecoach.analysis

import com.sidecoach.chess.BB_EMPTY
import com.sidecoach.chess.BB_KING_ATTACKS
import com.sidecoach.chess.BB_SQUARES
import com.sidecoach.chess.BISHOP
import com.sidecoach.chess.BLACK
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
import com.sidecoach.chess.square
import com.sidecoach.chess.squareDistance
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareKnightDistance
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
 * 1. A tag that claims material has to gain material once defenders are counted: a knight
 *    forking two rooks while a pawn can take it is not a fork, and a loose piece must really
 *    be undefended. But a pin, a skewer's front piece, or a discovered attack is a
 *    restriction or a revealed line, not a material claim, so those follow the geometry the
 *    way Lichess does.
 * 2. [tagsForMove] reports only what the move brought about. A pin the bishop was already
 *    making before the move is not something the move did, so castling does not inherit it.
 *    Those features still show up in [tagsForPosition], where they belong.
 *
 * `mateIn` counts the mover's own moves including the one being judged: 1 means this move is
 * checkmate, 3 means this move plus two more, and a negative value means the mover gets
 * mated. A negative `mateIn` never produces a `mateIn*` tag.
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
    )

    private val ORDER_INDEX = TAG_ORDER.withIndex().associate { (i, name) -> name to i }

    /** Standard 1/3/3/5/9. The king is priceless; 100 keeps every comparison right. */
    private val VALUE = intArrayOf(0, 1, 3, 3, 5, 9, 100)

    private const val MINOR = 3

    /** A sacrifice must give up at least this much: a pawn gambit is not a sacrifice. */
    private const val SACRIFICE_THRESHOLD = 2

    private val CORNERS = intArrayOf(0, 56, 7, 63)

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
     * man. Running away means the piece could have been taken for profit where it stood and
     * its new square is not attacked at all. Defending means the move newly covers a friendly
     * piece the opponent could have won before the move and is still attacking after it.
     */
    private fun defensiveEvidence(
        before: Board,
        after: Board,
        move: Move,
        mover: Boolean,
    ): List<Int>? {
        val enemy = !mover
        val ran = !isSafe(before, move.from) && !after.isAttackedBy(enemy, move.to)
        if (ran) return listOf(move.from, move.to)
        val coveredBefore = before.attacks(move.from)
        for (sq in after.attacks(move.to)) {
            if (sq in coveredBefore) continue
            val piece = after.pieceAt(sq) ?: continue
            if (piece.colour != mover) continue
            if (isSafe(before, sq) || !after.isAttackedBy(enemy, sq)) continue
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
            tags.add(tag("quietMove", listOf(frm, to), move))
            val evidence = defensiveEvidence(before, after, move, mover)
            if (evidence != null) tags.add(tag("defensiveMove", evidence, move))
        }
        return tags
    }

    // --- group B: what the move creates ------------------------------------------------

    /**
     * Would taking the piece on `square` with a `cost`-valued piece gain material? True when
     * the piece is worth more than what takes it, or when nothing defends it. `ignore` drops
     * one defender, which is how a skewered piece stops counting as the guard of the piece
     * behind it.
     */
    private fun winnable(view: Board, sq: Int, cost: Int, ignore: Int? = null): Boolean {
        val piece = view.pieceAt(sq) ?: return false
        if (piece.pieceType == KING || value(piece.pieceType) > cost) return true
        val defenders = view.attackers(piece.colour, sq).toMutableList()
        if (ignore != null) defenders.remove(ignore)
        return defenders.isEmpty()
    }

    /** Can the piece on `square` be taken for profit? Safe means no, so it can stay. */
    private fun isSafe(view: Board, sq: Int): Boolean {
        val piece = view.pieceAt(sq) ?: return true
        val mine = value(piece.pieceType)
        val attackers = view.attackers(!piece.colour, sq)
        if (attackers.isEmpty()) return true
        if (attackers.any { value(view.pieceTypeAt(it)) < mine }) return false
        return view.attackers(piece.colour, sq).isNotEmpty()
    }

    /**
     * Squares the mover captures an enemy piece on early in the line the engine expects.
     * This lets a fork that wins its piece a move or two later be recognised on the board
     * where the double attack is set up, using the engine's own continuation.
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
     * not a fork. Unless the enemy king is one of the targets, the forking piece must also be
     * safe where it stands.
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
        var forksKing = false
        for (sq in after.attacks(move.to)) {
            val victim = after.pieceAt(sq) ?: continue
            if (victim.colour != enemy) continue
            if (victim.pieceType == KING) {
                forksKing = true
                targets.add(sq)
            } else if (forkTarget(after, move.to, sq, mine, mover, givesCheck, wonInLine)) {
                targets.add(sq)
            }
        }
        if (targets.size < 2) return null
        if (!forksKing && !isSafe(after, move.to)) return null
        return tag("fork", listOf(move.to) + targets, move)
    }

    /**
     * Whether the enemy piece on `square` is one a fork from `forker` would win. A cheap
     * loose piece counts when the move gives check and either the piece cannot be saved
     * against any reply or the engine's own line actually captures it.
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
        val enemy = !mover
        val theirs = value(after.pieceTypeAt(sq))
        if (theirs > mine || (theirs >= mine && after.attackers(enemy, sq).isEmpty())) return true
        return givesCheck &&
            after.attackers(enemy, sq).isEmpty() &&
            (sq in wonInLine || checkForkWins(after, forker, sq, mover))
    }

    /**
     * After a checking move, is the loose enemy piece on `target` truly won? True only when
     * no legal reply to the check rescues it.
     */
    private fun checkForkWins(after: Board, forker: Int, target: Int, mover: Boolean): Boolean {
        val replies = after.legalMoves()
        if (replies.isEmpty()) return false  // checkmate: the mate tags cover this
        val enemy = !mover
        val cost = value(after.pieceTypeAt(forker))
        for (reply in replies) {
            val next = after.copy(withStack = false)
            next.push(reply)
            val victim = next.pieceAt(target)
            if (victim == null || victim.colour != enemy) return false
            if (forker !in next.attackers(mover, target)) return false
            if (!winnable(next, target, cost)) return false
        }
        return true
    }

    /**
     * Enemy pieces attacked by the mover with no defender at all, one tag per piece. Pawns
     * are excluded: loose pawns are everywhere and a coach calling one "hanging" is noise.
     */
    private fun hangingTags(view: Board, mover: Boolean): List<MotifTag> {
        val enemy = !mover
        val tags = ArrayList<MotifTag>()
        for (sq in view.occupiedSquares()) {
            val piece = view.pieceAt(sq) ?: continue
            if (piece.colour != enemy) continue
            if (piece.pieceType == PAWN || piece.pieceType == KING) continue
            if (view.attackers(mover, sq).isNotEmpty() && view.attackers(enemy, sq).isEmpty()) {
                tags.add(tag("hangingPiece", listOf(sq)))
            }
        }
        return tags
    }

    /**
     * Pins and skewers: one mover slider, one enemy piece, one enemy piece behind it. A pin
     * is geometric, the way Lichess tags it. A skewer names a piece to go after, so it still
     * requires the piece behind to be winnable.
     */
    private fun rayMotifs(view: Board, mover: Boolean): List<MotifTag> {
        val enemy = !mover
        val tags = ArrayList<MotifTag>()
        for (slider in sliders(view, mover)) {
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
                    tags.add(tag("pin", squares))
                    if (back.pieceType == KING) tags.add(tag("absolutePin", squares))
                    continue
                }
                if (slider in view.attacks(frontSq) && winnable(view, slider, frontValue)) continue
                if (frontValue > backValue) {
                    val mustMove = winnable(view, frontSq, cost)
                    val behindIsLoose = winnable(view, backSq, cost, ignore = frontSq)
                    if (mustMove && behindIsLoose) tags.add(tag("skewer", squares))
                }
            }
        }
        return tags
    }

    /**
     * The move steps off a line and a friendly slider now bears on something it did not. The
     * proof is geometric, the way Lichess tags a discovery: the vacated square lies strictly
     * between the slider and an enemy piece it now attacks and did not before.
     */
    private fun discoveredAttack(
        before: Board,
        after: Board,
        move: Move,
        mover: Boolean,
    ): MotifTag? {
        val enemy = !mover
        val squares = ArrayList<Int>()
        for (slider in sliders(after, mover)) {
            if (slider == move.to || before.pieceAt(slider) != after.pieceAt(slider)) continue
            val was = before.attacks(slider)
            for (target in after.attacks(slider)) {
                if (target in was) continue
                val piece = after.pieceAt(target) ?: continue
                if (piece.colour != enemy) continue
                if (BB_SQUARES[move.from] and between(slider, target) == BB_EMPTY) continue
                if (slider !in squares) squares.add(slider)
                squares.add(target)
            }
        }
        if (squares.isEmpty()) return null
        return tag("discoveredAttack", listOf(move.from) + squares, move)
    }

    /**
     * Cutting a line of defence, read over the line and by either side. Two shapes: the mover
     * interposes between an enemy slider and a piece it was defending, or the mover forces
     * the enemy to answer with a move that blocks one of their own sliders.
     */
    private fun interference(
        boards: List<Board>,
        played: List<Move>,
        mover: Boolean,
    ): MotifTag? {
        val exploited = HashSet<Int>()
        var k = 0
        while (k < played.size) { exploited.add(played[k].to); k += 2 }
        for ((i, move) in played.withIndex()) {
            val before = boards[i]
            val after = boards[i + 1]
            val found = if (i % 2 == 0) {
                moverInterference(before, after, move, mover)
            } else {
                selfInterference(before, after, move, !mover, exploited)
            }
            if (found != null) return found
        }
        return null
    }

    /** The mover lands on the line between an enemy slider and the piece it was defending. */
    private fun moverInterference(
        before: Board,
        after: Board,
        move: Move,
        mover: Boolean,
    ): MotifTag? {
        val enemy = !mover
        for (slider in sliders(before, enemy)) {
            if (slider == move.to || before.pieceAt(slider) != after.pieceAt(slider)) continue
            for (guarded in before.attacks(slider)) {
                if (guarded == move.to) continue
                val piece = before.pieceAt(guarded) ?: continue
                if (piece.colour != enemy || piece.pieceType == KING) continue
                if (BB_SQUARES[move.to] and between(slider, guarded) == BB_EMPTY) continue
                if (guarded in after.attacks(slider) || after.attackers(enemy, guarded).isNotEmpty()) continue
                if (after.attackers(mover, guarded).isEmpty()) continue
                return tag("interference", listOf(move.to, slider, guarded), move)
            }
        }
        return null
    }

    /**
     * A forced reply that blocks one of the replier's own sliders off a square it was
     * guarding, a square the mover later lands on.
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
            if (squaresOf(between(slider, block)).any { before.pieceAt(it) != null }) continue
            for (target in beyond(slider, block)) {
                if (before.pieceAt(target) != null && target !in exploited) break
                if (target in exploited &&
                    slider in before.attackers(defender, target) &&
                    slider !in after.attackers(defender, target)
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
     * An enemy piece attacked by something cheaper with nowhere safe to go. Needs the enemy
     * to move, so it is skipped whenever either king is in check.
     */
    private fun trappedTags(view: Board, mover: Boolean): List<MotifTag> {
        val enemy = !mover
        val enemyKing = view.king(enemy) ?: return emptyList()
        val moverKing = view.king(mover) ?: return emptyList()
        if (view.attackers(mover, enemyKing).isNotEmpty()) return emptyList()
        if (view.attackers(enemy, moverKing).isNotEmpty()) return emptyList()
        val escapes = HashMap<Int, MutableList<Move>>()
        for (move in view.legalMoves()) escapes.getOrPut(move.from) { ArrayList() }.add(move)
        val trapped = ArrayList<MotifTag>()
        for (sq in view.occupiedSquares()) {
            val piece = view.pieceAt(sq) ?: continue
            if (piece.colour != enemy || value(piece.pieceType) < MINOR) continue
            val mine = value(piece.pieceType)
            val hunters = view.attackers(mover, sq)
            if (hunters.none { value(view.pieceTypeAt(it)) < mine }) continue
            var safe = false
            for (move in escapes[sq] ?: emptyList<Move>()) {
                val victim = view.pieceAt(move.to)
                if (victim != null && value(victim.pieceType) >= mine) { safe = true; break }
                if (!view.isAttackedBy(mover, move.to)) { safe = true; break }
            }
            if (!safe) trapped.add(tag("trappedPiece", listOf(sq) + hunters))
        }
        return trapped
    }

    /** The move takes aim at the beginner's square: f7 as white, f2 as black. */
    private fun attackingF2F7(
        before: Board,
        after: Board,
        move: Move,
        mover: Boolean,
    ): MotifTag? {
        val target = if (mover == WHITE) 53 else 13  // f7, f2
        val piece = after.pieceAt(target) ?: return null
        if (piece.colour == mover) return null
        if (piece.pieceType != PAWN && piece.pieceType != KING) return null
        if (target !in after.attacks(move.to)) return null
        if (target in before.attacks(move.from)) return null
        return tag("attackingF2F7", listOf(move.to, target), move)
    }

    /**
     * The enemy king has lost its pawn shelter and the mover has force bearing on it.
     * Counting pieces on the ring rather than pieces on the board is what keeps an opening
     * king that just pushed a pawn from looking besieged.
     */
    private fun exposedKing(view: Board, mover: Boolean): MotifTag? {
        val enemy = !mover
        val king = view.king(enemy) ?: return null
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
     * pinning.
     */
    private fun staticKey(item: MotifTag): Pair<String, List<String>> = when {
        item.name in RAY_NAMES -> item.name to item.squares.drop(1)
        item.name == "exposedKing" || item.name == "trappedPiece" -> item.name to item.squares.take(1)
        else -> item.name to item.squares
    }

    /** Static features the move brought about, and only those. */
    private fun newStaticTags(before: Board, after: Board, mover: Boolean): List<MotifTag> {
        val prior = staticTags(enemyToMove(before, mover), mover).map { staticKey(it) }.toSet()
        return staticTags(after, mover).filter { staticKey(it) !in prior }
    }

    // --- group C: what the line does ---------------------------------------------------

    /**
     * This move gives up at least two points to the opponent's reply, and the line still
     * wins. The judgement is about the move played, not about the line it leads to: the
     * balance is read after the move and the opponent's answer to it.
     */
    private fun sacrifice(
        boards: List<Board>,
        played: List<Move>,
        mover: Boolean,
        mateIn: Int?,
    ): MotifTag? {
        if (played.size < 2) return null
        val start = balance(boards[0], mover)
        val answered = if (boards.size > 2) balance(boards[2], mover) else balance(boards.last(), mover)
        if (start - answered < SACRIFICE_THRESHOLD) return null
        if (boards.size > 3 && start - balance(boards[3], mover) < SACRIFICE_THRESHOLD) return null
        val mating = mateIn != null && mateIn > 0
        if (!mating && balance(boards.last(), mover) < start) return null
        return tag("sacrifice", listOf(played[0].from, played[0].to, played[1].to), played[0])
    }

    /**
     * The move vacates a square with tempo and a later move of the mover's uses that square.
     * Tempo means the move is a capture or a check.
     */
    private fun clearance(boards: List<Board>, played: List<Move>): MotifTag? {
        if (played.size < 3) return null
        val first = played[0]
        if (!boards[0].isCapture(first) && !boards[1].isCheck()) return null
        val vacated = first.from
        var i = 2
        while (i < played.size) {
            val later = played[i]
            if (later.to == vacated) {
                return tag("clearance", listOf(vacated, later.from, later.to), first)
            }
            if (BB_SQUARES[vacated] and between(later.from, later.to) != BB_EMPTY) {
                return tag("clearance", listOf(vacated, later.from, later.to), first)
            }
            i += 2
        }
        return null
    }

    /** The move captures a piece and a later move of the mover's lands on a square it guarded. */
    private fun capturingDefender(boards: List<Board>, played: List<Move>): MotifTag? {
        if (played.isEmpty()) return null
        val first = played[0]
        if (!boards[0].isCapture(first)) return null
        val victimSquare = if (boards[0].isEnPassant(first)) epVictim(first) else first.to
        val guarded = boards[0].attacks(victimSquare)
        var i = 2
        while (i < played.size) {
            val later = played[i]
            if (later.to in guarded) {
                val position = boards[i]
                if (position.isCapture(later) || position.givesCheck(later)) {
                    return tag("capturingDefender", listOf(victimSquare, later.to), first)
                }
            }
            i += 2
        }
        return null
    }

    /** A forcing move pulls the only defender of a square away, and the mover takes the square. */
    private fun deflection(boards: List<Board>, played: List<Move>, mover: Boolean): MotifTag? {
        if (played.size < 3) return null
        val first = played[0]
        val reply = played[1]
        val follow = played[2]
        if (!boards[0].isCapture(first) && !boards[1].isCheck()) return null
        val defender = reply.from
        val guards = boards[1].attackers(!mover, follow.to)
        if (guards.size != 1 || defender !in guards) return null
        if (follow.to !in boards[1].attacks(defender)) return null
        if (follow.to in boards[2].attacks(reply.to)) return null
        if (!boards[2].isCapture(follow) && !boards[2].givesCheck(follow)) return null
        return tag("deflection", listOf(defender, reply.to, follow.to), first)
    }

    /** A forcing move drags an enemy piece onto a square where the next move hits it. */
    private fun attraction(boards: List<Board>, played: List<Move>, mover: Boolean): MotifTag? {
        if (played.size < 3) return null
        val first = played[0]
        val reply = played[1]
        val follow = played[2]
        if (!boards[0].isCapture(first) && !boards[1].isCheck()) return null
        if (!boards[1].isCapture(reply)) return null
        val landed = reply.to
        val attracted = boards[2].pieceAt(landed) ?: return null
        if (attracted.colour == mover || value(attracted.pieceType) < MINOR) return null
        if (!boards[2].isCapture(follow) && !boards[2].givesCheck(follow)) return null
        val hits = follow.to == landed || landed in boards[3].attacks(follow.to)
        if (!hits) return null
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

    // --- group D: mating patterns ------------------------------------------------------

    private fun diagonalKind(a: Int, b: Int): String? {
        val fa = squareFile(a)
        val ra = squareRank(a)
        val fb = squareFile(b)
        val rb = squareRank(b)
        if (fa - ra == fb - rb) return "main"
        if (fa + ra == fb + rb) return "anti"
        return null
    }

    private fun defenders(position: Board, colour: Boolean, sq: Int): List<Int> =
        position.attackers(colour, sq)

    /** Each pattern is emitted only when its shape matches exactly. */
    private fun matingPatternTags(position: Board): List<MotifTag> {
        val loser = position.turn
        val winner = !loser
        val king = position.king(loser) ?: return emptyList()
        val checkers = squaresOf(position.checkersMask())
        val ring = squaresOf(BB_KING_ATTACKS[king])
        val ownRing = ring.filter { position.pieceAt(it)?.colour == loser }
        val kingFile = squareFile(king)
        val kingRank = squareRank(king)
        val tags = ArrayList<MotifTag>()

        val backRank = if (loser == WHITE) 0 else 7
        if (kingRank == backRank) {
            val forward = backRank + (if (loser == WHITE) 1 else -1)
            val front = intArrayOf(kingFile - 1, kingFile, kingFile + 1)
                .filter { it in 0..7 }
                .map { square(it, forward) }
            val blocked = front.all { position.pieceAt(it)?.colour == loser }
            val heavy = checkers.filter {
                (position.pieceTypeAt(it) == ROOK || position.pieceTypeAt(it) == QUEEN) &&
                    squareRank(it) == backRank
            }
            if (blocked && heavy.isNotEmpty()) {
                tags.add(tag("backRankMate", listOf(king) + heavy + front))
            }
        }

        if (checkers.size == 1) {
            val checker = checkers[0]
            val kind = position.pieceTypeAt(checker)
            val adjacent = squareDistance(checker, king) == 1
            val knights = defenders(position, winner, checker)
                .filter { position.pieceTypeAt(it) == KNIGHT }

            if (kind == KNIGHT && ownRing.size == ring.size) {
                tags.add(tag("smotheredMate", listOf(king, checker) + ownRing))
            }

            if (kind == ROOK && adjacent && knights.isNotEmpty() && king in CORNERS) {
                tags.add(tag("arabianMate", listOf(king, checker, knights[0])))
            }

            if (kind == ROOK && adjacent && knights.isNotEmpty()) {
                val pawns = defenders(position, winner, knights[0])
                    .filter { position.pieceTypeAt(it) == PAWN }
                if (pawns.isNotEmpty()) {
                    tags.add(tag("hookMate", listOf(king, checker, knights[0], pawns[0])))
                }
            }

            if (kind == ROOK && adjacent) {
                val queens = defenders(position, winner, checker)
                    .filter { position.pieceTypeAt(it) == QUEEN }
                val box = queens.filter { squareKnightDistance(it, king) == 1 }
                if (box.isNotEmpty()) {
                    tags.add(tag("killBoxMate", listOf(king, checker, box[0])))
                }
            }

            val onEdge = kingFile == 0 || kingFile == 7 || kingRank == 0 || kingRank == 7
            if (kind == ROOK && adjacent && onEdge) {
                val others = defenders(position, winner, checker)
                    .filter { position.pieceTypeAt(it) != KNIGHT }
                val covers = ring.filter { s ->
                    position.pieceAt(s) == null &&
                        position.pieces(KNIGHT, winner).any { s in position.attacks(it) }
                }
                if (others.isNotEmpty() && covers.isNotEmpty()) {
                    tags.add(tag("vukovicMate", listOf(king, checker, others[0]) + covers))
                }
            }

            if (kind == ROOK && (kingFile == 0 || kingFile == 7) && squareFile(checker) == kingFile) {
                val inner = if (kingFile == 0) 1 else 6
                val above = kingRank + 1
                val below = kingRank - 1
                if (below >= 0 && above <= 7) {
                    val beside = square(inner, kingRank)
                    val blocker = position.pieceAt(beside)
                    val flights = intArrayOf(square(inner, above), square(inner, below))
                    val covered = position.pieces(KNIGHT, winner).any { n ->
                        flights.all { it in position.attacks(n) }
                    }
                    if (blocker != null && blocker.colour == loser && covered) {
                        tags.add(tag("anastasiaMate", listOf(king, checker, beside) + flights.toList()))
                    }
                }
            }

            if (kind == BISHOP) {
                val checkKind = diagonalKind(checker, king)
                for (other in position.pieces(BISHOP, winner)) {
                    if (other == checker) continue
                    for (sq in ring) {
                        if (sq !in position.attacks(other)) continue
                        val otherKind = diagonalKind(other, sq)
                        if (otherKind == null || checkKind == null) continue
                        val evidence = listOf(king, checker, other, sq)
                        if (otherKind == checkKind && onEdge) {
                            tags.add(tag("doubleBishopMate", evidence))
                        } else if (otherKind != checkKind && ownRing.size >= 2) {
                            tags.add(tag("bodenMate", evidence))
                        }
                        break
                    }
                }
            }

            if (kind == QUEEN && ring.size == 8 && defenders(position, winner, checker).isNotEmpty()) {
                var diagonal = abs(squareFile(checker) - kingFile) == 1
                diagonal = diagonal && abs(squareRank(checker) - kingRank) == 1
                val orthogonal = ownRing.filter {
                    squareFile(it) == kingFile || squareRank(it) == kingRank
                }
                if (diagonal && ownRing.size == 2 && orthogonal.size == 2) {
                    tags.add(tag("dovetailMate", listOf(king, checker) + ownRing))
                }
            }
        }

        return tags
    }

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
            val betweenSquares = squaresOf(between(slider, target))
            val blockers = betweenSquares.filter { board.pieceAt(it) != null }
            if (blockers.size != 1) continue
            val blocker = blockers[0]
            if (board.pieceAt(blocker)!!.colour != mover) return slider to blocker
        }
        return null
    }

    /**
     * Lichess's x-ray: a piece attacks a square through an enemy piece, read over the line.
     * The blocker clearing is what turns a latent x-ray into a real one.
     */
    private fun xray(boards: List<Board>, played: List<Move>, mover: Boolean): MotifTag? {
        var k = 0
        while (k < played.size) {
            val move = played[k]
            val boardK = boards[k]
            val afterK = if (k + 1 < boards.size) boards[k + 1] else null
            if (!(boardK.isCapture(move) || (afterK != null && afterK.isCheckmate()))) { k += 2; continue }
            val target = move.to
            var j = 0
            while (j <= k) {
                val found = xrayTo(boards[j], mover, target)
                if (found != null) {
                    val (slider, blocker) = found
                    val occupant = boardK.pieceAt(blocker)
                    if (occupant == null || occupant.colour == mover) {
                        return tag("xRayAttack", listOf(target, blocker, slider), move)
                    }
                }
                j += 2
            }
            k += 2
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

        val tags = ArrayList(groupA(before, after, move, mover))
        tags.addAll(newStaticTags(before, after, mover))
        for (found in listOf(
            fork(after, move, mover, wonInLine = moverCaptures(boards, played, mover)),
            discoveredAttack(before, after, move, mover),
            xray(boards, played, mover),
            interference(boards, played, mover),
            attackingF2F7(before, after, move, mover),
            sacrifice(boards, played, mover, mateIn),
            clearance(boards, played),
            capturingDefender(boards, played),
            deflection(boards, played, mover),
            attraction(boards, played, mover),
        )) {
            if (found != null) tags.add(found)
        }
        tags.addAll(mateTags(after, boards.last(), mover, mateIn))

        if (after.isCheckmate()) {
            tags.addAll(matingPatternTags(after))
        } else if (boards.last().isCheckmate() && boards.last().turn != mover) {
            tags.addAll(matingPatternTags(boards.last()))
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
