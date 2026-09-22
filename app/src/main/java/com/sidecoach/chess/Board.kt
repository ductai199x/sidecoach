package com.sidecoach.chess

import kotlin.math.abs

/**
 * The position, the moves and the notation: a port of `python-chess`'s `Board`.
 *
 * Only what the reconciler asks of it, but that literally -- including the order
 * `legalMoves()` returns moves in, because the desktop's missed-move search keeps the
 * first of several transposing stories and a different order would tell a different one.
 *
 * Castling is king-takes-rook internally, as python-chess stores it, and is converted back
 * on the way in and out (`fromChess960`/`toChess960` there). Chess960 itself is not ported:
 * the desktop watches a screen showing standard chess.
 */

data class Piece(val pieceType: Int, val colour: Boolean) {
    fun symbol(): String {
        val s = PIECE_SYMBOLS[pieceType]
        return if (colour) s.uppercase() else s
    }

    companion object {
        fun fromSymbol(symbol: Char): Piece =
            Piece(PIECE_SYMBOLS.indexOf(symbol.lowercaseChar().toString()), symbol.isUpperCase())

        fun fromSymbol(symbol: String): Piece = fromSymbol(symbol[0])
    }
}

data class Move(val from: Int, val to: Int, val promotion: Int = 0) {
    fun uci(): String =
        squareName(from) + squareName(to) + if (promotion != 0) PIECE_SYMBOLS[promotion] else ""

    override fun toString(): String = uci()

    companion object {
        fun fromUci(uci: String): Move = Move(
            parseSquare(uci.substring(0, 2)),
            parseSquare(uci.substring(2, 4)),
            if (uci.length > 4) PIECE_SYMBOLS.indexOf(uci.substring(4, 5)) else 0,
        )
    }
}

/** The bitboards and counters `pop()` puts back. */
private class BoardState(board: Board) {
    val pawns = board.pawns
    val knights = board.knights
    val bishops = board.bishops
    val rooks = board.rooks
    val queens = board.queens
    val kings = board.kings
    val occupiedW = board.occupiedCo[1]
    val occupiedB = board.occupiedCo[0]
    val occupied = board.occupied
    val promoted = board.promoted
    val turn = board.turn
    val castlingRights = board.castlingRights
    val epSquare = board.epSquare
    val halfmoveClock = board.halfmoveClock
    val fullmoveNumber = board.fullmoveNumber

    fun restore(board: Board) {
        board.pawns = pawns
        board.knights = knights
        board.bishops = bishops
        board.rooks = rooks
        board.queens = queens
        board.kings = kings
        board.occupiedCo[1] = occupiedW
        board.occupiedCo[0] = occupiedB
        board.occupied = occupied
        board.promoted = promoted
        board.turn = turn
        board.castlingRights = castlingRights
        board.epSquare = epSquare
        board.halfmoveClock = halfmoveClock
        board.fullmoveNumber = fullmoveNumber
    }
}

const val STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
const val STARTING_BOARD_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

class Board {

    var pawns: Bitboard = BB_EMPTY
    var knights: Bitboard = BB_EMPTY
    var bishops: Bitboard = BB_EMPTY
    var rooks: Bitboard = BB_EMPTY
    var queens: Bitboard = BB_EMPTY
    var kings: Bitboard = BB_EMPTY
    var promoted: Bitboard = BB_EMPTY
    var occupied: Bitboard = BB_EMPTY

    /** Indexed 0 = black, 1 = white, as python-chess indexes it by the colour bool. */
    val occupiedCo = ULongArray(2)

    var turn: Boolean = WHITE
    var castlingRights: Bitboard = BB_EMPTY
    var epSquare: Int? = null
    var halfmoveClock: Int = 0
    var fullmoveNumber: Int = 1

    val moveStack = ArrayList<Move>()
    private val stateStack = ArrayList<BoardState>()

    constructor(fen: String? = STARTING_FEN) {
        if (fen == null) clearBoard() else setFen(fen)
    }

    // Piece access

    fun pieceTypeAt(square: Int): Int {
        val mask = BB_SQUARES[square]
        return when {
            occupied and mask == BB_EMPTY -> 0
            pawns and mask != BB_EMPTY -> PAWN
            knights and mask != BB_EMPTY -> KNIGHT
            bishops and mask != BB_EMPTY -> BISHOP
            rooks and mask != BB_EMPTY -> ROOK
            queens and mask != BB_EMPTY -> QUEEN
            else -> KING
        }
    }

    fun pieceAt(square: Int): Piece? {
        val type = pieceTypeAt(square)
        if (type == 0) return null
        return Piece(type, occupiedCo[1] and BB_SQUARES[square] != BB_EMPTY)
    }

    fun colourAt(square: Int): Boolean? {
        val mask = BB_SQUARES[square]
        return when {
            occupiedCo[1] and mask != BB_EMPTY -> WHITE
            occupiedCo[0] and mask != BB_EMPTY -> BLACK
            else -> null
        }
    }

    fun king(colour: Boolean): Int? {
        val mask = occupiedCo[if (colour) 1 else 0] and kings and promoted.inv()
        return if (mask == BB_EMPTY) null else msb(mask)
    }

    fun piecesMask(pieceType: Int, colour: Boolean): Bitboard {
        val bb = when (pieceType) {
            PAWN -> pawns
            KNIGHT -> knights
            BISHOP -> bishops
            ROOK -> rooks
            QUEEN -> queens
            KING -> kings
            else -> BB_EMPTY
        }
        return bb and occupiedCo[if (colour) 1 else 0]
    }

    fun pieceMap(): Map<Int, Piece> {
        val out = LinkedHashMap<Int, Piece>()
        scanReversed(occupied) { sq -> out[sq] = pieceAt(sq)!! }
        return out
    }

    fun setPieceAt(square: Int, piece: Piece?, promotedPiece: Boolean = false) {
        if (piece == null) {
            removePieceAt(square)
            return
        }
        removePieceAt(square)
        val mask = BB_SQUARES[square]
        when (piece.pieceType) {
            PAWN -> pawns = pawns or mask
            KNIGHT -> knights = knights or mask
            BISHOP -> bishops = bishops or mask
            ROOK -> rooks = rooks or mask
            QUEEN -> queens = queens or mask
            KING -> kings = kings or mask
        }
        occupied = occupied or mask
        occupiedCo[if (piece.colour) 1 else 0] = occupiedCo[if (piece.colour) 1 else 0] or mask
        if (promotedPiece) promoted = promoted or mask
    }

    fun removePieceAt(square: Int): Int {
        val type = pieceTypeAt(square)
        if (type == 0) return 0
        val mask = BB_SQUARES[square]
        when (type) {
            PAWN -> pawns = pawns and mask.inv()
            KNIGHT -> knights = knights and mask.inv()
            BISHOP -> bishops = bishops and mask.inv()
            ROOK -> rooks = rooks and mask.inv()
            QUEEN -> queens = queens and mask.inv()
            KING -> kings = kings and mask.inv()
        }
        occupied = occupied and mask.inv()
        occupiedCo[0] = occupiedCo[0] and mask.inv()
        occupiedCo[1] = occupiedCo[1] and mask.inv()
        promoted = promoted and mask.inv()
        return type
    }

    private fun clearBoard() {
        pawns = BB_EMPTY; knights = BB_EMPTY; bishops = BB_EMPTY
        rooks = BB_EMPTY; queens = BB_EMPTY; kings = BB_EMPTY
        promoted = BB_EMPTY; occupied = BB_EMPTY
        occupiedCo[0] = BB_EMPTY; occupiedCo[1] = BB_EMPTY
        turn = WHITE
        castlingRights = BB_EMPTY
        epSquare = null
        halfmoveClock = 0
        fullmoveNumber = 1
        moveStack.clear()
        stateStack.clear()
    }

    // Attacks

    fun attacksMask(square: Int): Bitboard {
        val mask = BB_SQUARES[square]
        return when {
            pawns and mask != BB_EMPTY -> pawnAttacks(occupiedCo[1] and mask != BB_EMPTY, square)
            knights and mask != BB_EMPTY -> BB_KNIGHT_ATTACKS[square]
            kings and mask != BB_EMPTY -> BB_KING_ATTACKS[square]
            else -> {
                var attacks = BB_EMPTY
                if (bishops and mask != BB_EMPTY || queens and mask != BB_EMPTY) {
                    attacks = diagAttacks(square, occupied)
                }
                if (rooks and mask != BB_EMPTY || queens and mask != BB_EMPTY) {
                    attacks = attacks or rankAttacks(square, occupied) or fileAttacks(square, occupied)
                }
                attacks
            }
        }
    }

    fun attackersMask(colour: Boolean, square: Int, occupancy: Bitboard = occupied): Bitboard {
        val queensAndRooks = queens or rooks
        val queensAndBishops = queens or bishops
        val attackers = (BB_KING_ATTACKS[square] and kings) or
            (BB_KNIGHT_ATTACKS[square] and knights) or
            (rankAttacks(square, occupancy) and queensAndRooks) or
            (fileAttacks(square, occupancy) and queensAndRooks) or
            (diagAttacks(square, occupancy) and queensAndBishops) or
            (pawnAttacks(!colour, square) and pawns)
        return attackers and occupiedCo[if (colour) 1 else 0]
    }

    fun isAttackedBy(colour: Boolean, square: Int): Boolean =
        attackersMask(colour, square) != BB_EMPTY

    fun checkersMask(): Bitboard {
        val k = king(turn) ?: return BB_EMPTY
        return attackersMask(!turn, k)
    }

    fun isCheck(): Boolean = checkersMask() != BB_EMPTY

    fun wasIntoCheck(): Boolean {
        val k = king(!turn) ?: return false
        return isAttackedBy(turn, k)
    }

    private fun sliderBlockers(king: Int): Bitboard {
        val rooksAndQueens = rooks or queens
        val bishopsAndQueens = bishops or queens
        val snipers = (BB_RANK_RAYS[king] and rooksAndQueens) or
            (BB_FILE_RAYS[king] and rooksAndQueens) or
            (BB_DIAG_RAYS[king] and bishopsAndQueens)
        var blockers = BB_EMPTY
        scanReversed(snipers and occupiedCo[if (!turn) 1 else 0]) { sniper ->
            val b = between(king, sniper) and occupied
            if (b != BB_EMPTY && BB_SQUARES[msb(b)] == b) blockers = blockers or b
        }
        return blockers and occupiedCo[if (turn) 1 else 0]
    }

    fun pinMask(colour: Boolean, square: Int): Bitboard {
        val k = king(colour) ?: return BB_ALL
        val squareMask = BB_SQUARES[square]
        val tables = arrayOf(
            BB_FILE_RAYS to (rooks or queens),
            BB_RANK_RAYS to (rooks or queens),
            BB_DIAG_RAYS to (bishops or queens),
        )
        for ((rays, sliders) in tables) {
            val line = rays[k]
            if (line and squareMask != BB_EMPTY) {
                val snipers = line and sliders and occupiedCo[if (!colour) 1 else 0]
                var found: Bitboard? = null
                scanReversed(snipers) { sniper ->
                    if (found == null && between(sniper, k) and (occupied or squareMask) == squareMask) {
                        found = ray(k, sniper)
                    }
                }
                found?.let { return it }
                break
            }
        }
        return BB_ALL
    }

    private fun epSkewered(king: Int, capturer: Int): Boolean {
        val ep = epSquare!!
        val lastDouble = ep + (if (turn == WHITE) -8 else 8)
        val occupancy = (occupied and BB_SQUARES[lastDouble].inv() and BB_SQUARES[capturer].inv()) or
            BB_SQUARES[ep]
        val horizontal = occupiedCo[if (!turn) 1 else 0] and (rooks or queens)
        if (rankAttacks(king, occupancy) and horizontal != BB_EMPTY) return true
        val diagonal = occupiedCo[if (!turn) 1 else 0] and (bishops or queens)
        return diagAttacks(king, occupancy) and diagonal != BB_EMPTY
    }

    private fun attackedForKing(path: Bitboard, occupancy: Bitboard): Boolean {
        var hit = false
        scanReversed(path) { sq ->
            if (!hit && attackersMask(!turn, sq, occupancy) != BB_EMPTY) hit = true
        }
        return hit
    }

    private fun isSafe(king: Int, blockers: Bitboard, move: Move): Boolean {
        if (move.from == king) {
            return if (isCastling(move)) true else !isAttackedBy(!turn, move.to)
        }
        if (isEnPassant(move)) {
            return pinMask(turn, move.from) and BB_SQUARES[move.to] != BB_EMPTY &&
                !epSkewered(king, move.from)
        }
        return blockers and BB_SQUARES[move.from] == BB_EMPTY ||
            ray(move.from, king) and BB_SQUARES[move.to] != BB_EMPTY
    }

    // Move properties

    fun isCastling(move: Move): Boolean {
        if (kings and BB_SQUARES[move.from] == BB_EMPTY) return false
        val diff = squareFile(move.from) - squareFile(move.to)
        return abs(diff) > 1 ||
            (rooks and occupiedCo[if (turn) 1 else 0] and BB_SQUARES[move.to]) != BB_EMPTY
    }

    fun isEnPassant(move: Move): Boolean =
        epSquare == move.to &&
            pawns and BB_SQUARES[move.from] != BB_EMPTY &&
            (abs(move.to - move.from) == 7 || abs(move.to - move.from) == 9) &&
            occupied and BB_SQUARES[move.to] == BB_EMPTY

    /** `gives_check`: whether the move leaves the opponent in check. */
    fun givesCheck(move: Move): Boolean {
        push(move)
        val check = isCheck()
        pop()
        return check
    }

    fun isKingsideCastling(move: Move): Boolean =
        isCastling(move) && squareFile(move.to) > squareFile(move.from)

    fun isQueensideCastling(move: Move): Boolean =
        isCastling(move) && squareFile(move.to) < squareFile(move.from)

    fun hasKingsideCastlingRights(colour: Boolean): Boolean = castlingSide(colour, kingside = true)

    fun hasQueensideCastlingRights(colour: Boolean): Boolean = castlingSide(colour, kingside = false)

    private fun castlingSide(colour: Boolean, kingside: Boolean): Boolean {
        val backrank = if (colour == WHITE) BB_RANK_1 else BB_RANK_8
        val kingMask = kings and occupiedCo[if (colour) 1 else 0] and backrank and promoted.inv()
        if (kingMask == BB_EMPTY) return false
        var rights = cleanCastlingRights() and backrank
        while (rights != BB_EMPTY) {
            val rook = rights and (0uL - rights)
            if (if (kingside) rook > kingMask else rook < kingMask) return true
            rights = rights and (rights - 1uL)
        }
        return false
    }

    /** Whether the piece on `square` may not leave its line without exposing what is behind. */
    fun isPinned(colour: Boolean, square: Int): Boolean = pinMask(colour, square) != BB_ALL

    /** The squares the piece on `square` bears on, lowest first: `board.attacks`. */
    fun attacks(square: Int): List<Int> = squaresOf(attacksMask(square))

    /** The pieces of `colour` bearing on `square`, lowest first: `board.attackers`. */
    fun attackers(colour: Boolean, square: Int): List<Int> = squaresOf(attackersMask(colour, square))

    /** The squares of `colour`'s pieces of one type, lowest first: `board.pieces`. */
    fun pieces(pieceType: Int, colour: Boolean): List<Int> = squaresOf(piecesMask(pieceType, colour))

    /** The occupied squares, lowest first, as `sorted(board.piece_map())` gives them. */
    fun occupiedSquares(): List<Int> = squaresOf(occupied)

    fun isCapture(move: Move): Boolean {
        val touched = BB_SQUARES[move.from] xor BB_SQUARES[move.to]
        return touched and occupiedCo[if (!turn) 1 else 0] != BB_EMPTY || isEnPassant(move)
    }

    fun isZeroing(move: Move): Boolean {
        val touched = BB_SQUARES[move.from] xor BB_SQUARES[move.to]
        return touched and pawns != BB_EMPTY || touched and occupiedCo[if (!turn) 1 else 0] != BB_EMPTY
    }

    fun isIntoCheck(move: Move): Boolean {
        val k = king(turn) ?: return false
        val checkers = attackersMask(!turn, k)
        if (checkers != BB_EMPTY) {
            var found = false
            generateEvasions(k, checkers, BB_SQUARES[move.from], BB_SQUARES[move.to]) {
                if (it == move) found = true
            }
            if (!found) return true
        }
        return !isSafe(k, sliderBlockers(k), move)
    }

    fun isLegal(move: Move): Boolean {
        if (!isPseudoLegal(move)) return false
        return !isIntoCheck(move)
    }

    fun isPseudoLegal(move: Move): Boolean {
        val piece = pieceTypeAt(move.from)
        if (piece == 0) return false
        if (colourAt(move.from) != turn) return false
        if (move.promotion != 0) {
            if (piece != PAWN) return false
            if (turn == WHITE && squareRank(move.to) != 7) return false
            if (turn == BLACK && squareRank(move.to) != 0) return false
        }
        if (piece == KING) {
            var castles = false
            generateCastlingMoves(BB_ALL, BB_ALL) { if (it == move) castles = true }
            if (castles) return true
        }
        if (occupiedCo[if (turn) 1 else 0] and BB_SQUARES[move.to] != BB_EMPTY) return false
        if (piece == PAWN) {
            var ok = false
            generatePseudoLegalMoves(BB_SQUARES[move.from], BB_SQUARES[move.to]) { if (it == move) ok = true }
            return ok
        }
        return attacksMask(move.from) and BB_SQUARES[move.to] != BB_EMPTY
    }

    // Move generation, in python-chess's order

    fun generatePseudoLegalMoves(
        fromMask: Bitboard = BB_ALL,
        toMask: Bitboard = BB_ALL,
        yieldMove: (Move) -> Unit,
    ) {
        val ourPieces = occupiedCo[if (turn) 1 else 0]

        val nonPawns = ourPieces and pawns.inv() and fromMask
        scanReversed(nonPawns) { from ->
            val moves = attacksMask(from) and ourPieces.inv() and toMask
            scanReversed(moves) { to -> yieldMove(Move(from, to)) }
        }

        if (fromMask and kings != BB_EMPTY) generateCastlingMoves(fromMask, toMask, yieldMove)

        val ourPawns = pawns and ourPieces and fromMask
        if (ourPawns == BB_EMPTY) return

        scanReversed(ourPawns) { from ->
            val targets = pawnAttacks(turn, from) and occupiedCo[if (!turn) 1 else 0] and toMask
            scanReversed(targets) { to ->
                if (squareRank(to) == 0 || squareRank(to) == 7) {
                    yieldMove(Move(from, to, QUEEN))
                    yieldMove(Move(from, to, ROOK))
                    yieldMove(Move(from, to, BISHOP))
                    yieldMove(Move(from, to, KNIGHT))
                } else {
                    yieldMove(Move(from, to))
                }
            }
        }

        var singleMoves: Bitboard
        var doubleMoves: Bitboard
        if (turn == WHITE) {
            singleMoves = (ourPawns shl 8) and occupied.inv()
            doubleMoves = (singleMoves shl 8) and occupied.inv() and (BB_RANKS[2] or BB_RANKS[3])
        } else {
            singleMoves = (ourPawns shr 8) and occupied.inv()
            doubleMoves = (singleMoves shr 8) and occupied.inv() and (BB_RANKS[5] or BB_RANKS[4])
        }
        singleMoves = singleMoves and toMask
        doubleMoves = doubleMoves and toMask

        scanReversed(singleMoves) { to ->
            val from = to + (if (turn == BLACK) 8 else -8)
            if (squareRank(to) == 0 || squareRank(to) == 7) {
                yieldMove(Move(from, to, QUEEN))
                yieldMove(Move(from, to, ROOK))
                yieldMove(Move(from, to, BISHOP))
                yieldMove(Move(from, to, KNIGHT))
            } else {
                yieldMove(Move(from, to))
            }
        }

        scanReversed(doubleMoves) { to ->
            val from = to + (if (turn == BLACK) 16 else -16)
            yieldMove(Move(from, to))
        }

        if (epSquare != null) generatePseudoLegalEp(fromMask, toMask, yieldMove)
    }

    fun generatePseudoLegalEp(
        fromMask: Bitboard = BB_ALL,
        toMask: Bitboard = BB_ALL,
        yieldMove: (Move) -> Unit,
    ) {
        val ep = epSquare ?: return
        if (BB_SQUARES[ep] and toMask == BB_EMPTY) return
        if (BB_SQUARES[ep] and occupied != BB_EMPTY) return
        val capturers = pawns and occupiedCo[if (turn) 1 else 0] and fromMask and
            pawnAttacks(!turn, ep) and BB_RANKS[if (turn) 4 else 3]
        scanReversed(capturers) { capturer -> yieldMove(Move(capturer, ep)) }
    }

    /** `clean_castling_rights`, including its shortcut: after a push the stored rights stand. */
    fun cleanCastlingRights(): Bitboard {
        if (stateStack.isNotEmpty()) return castlingRights
        val castling = castlingRights and rooks
        var whiteCastling = castling and BB_RANK_1 and occupiedCo[1]
        var blackCastling = castling and BB_RANK_8 and occupiedCo[0]
        whiteCastling = whiteCastling and (BB_SQUARES[A1] or BB_SQUARES[H1])
        blackCastling = blackCastling and (BB_SQUARES[A8] or BB_SQUARES[H8])
        if (occupiedCo[1] and kings and promoted.inv() and BB_SQUARES[E1] == BB_EMPTY) {
            whiteCastling = BB_EMPTY
        }
        if (occupiedCo[0] and kings and promoted.inv() and BB_SQUARES[E8] == BB_EMPTY) {
            blackCastling = BB_EMPTY
        }
        return whiteCastling or blackCastling
    }

    fun generateCastlingMoves(
        fromMask: Bitboard = BB_ALL,
        toMask: Bitboard = BB_ALL,
        yieldMove: (Move) -> Unit,
    ) {
        val backrank = if (turn == WHITE) BB_RANK_1 else BB_RANK_8
        var kingBb = occupiedCo[if (turn) 1 else 0] and kings and promoted.inv() and backrank and fromMask
        kingBb = kingBb and (0uL - kingBb)  // the lowest set bit, as `king &= -king`
        if (kingBb == BB_EMPTY) return
        val kingSquare = msb(kingBb)

        val bbC = BB_FILES[2] and backrank
        val bbD = BB_FILES[3] and backrank
        val bbF = BB_FILES[5] and backrank
        val bbG = BB_FILES[6] and backrank

        scanReversed(cleanCastlingRights() and backrank and toMask) { candidate ->
            val rook = BB_SQUARES[candidate]
            val aSide = rook < kingBb
            val kingTo = if (aSide) bbC else bbG
            val rookTo = if (aSide) bbD else bbF
            val kingPath = between(kingSquare, msb(kingTo))
            val rookPath = between(candidate, msb(rookTo))
            val blocked = (occupied xor kingBb xor rook) and
                (kingPath or rookPath or kingTo or rookTo) != BB_EMPTY
            if (!blocked &&
                !attackedForKing(kingPath or kingBb, occupied xor kingBb) &&
                !attackedForKing(kingTo, occupied xor kingBb xor rook xor rookTo)
            ) {
                yieldMove(fromChess960(kingSquare, candidate))
            }
        }
    }

    private fun generateEvasions(
        king: Int,
        checkers: Bitboard,
        fromMask: Bitboard,
        toMask: Bitboard,
        yieldMove: (Move) -> Unit,
    ) {
        val sliders = checkers and (bishops or rooks or queens)
        var attacked = BB_EMPTY
        scanReversed(sliders) { checker -> attacked = attacked or (ray(king, checker) and BB_SQUARES[checker].inv()) }

        if (BB_SQUARES[king] and fromMask != BB_EMPTY) {
            scanReversed(
                BB_KING_ATTACKS[king] and occupiedCo[if (turn) 1 else 0].inv() and attacked.inv() and toMask,
            ) { to -> yieldMove(Move(king, to)) }
        }

        val checker = msb(checkers)
        if (BB_SQUARES[checker] == checkers) {
            val target = between(king, checker) or checkers
            generatePseudoLegalMoves(kings.inv() and fromMask, target and toMask, yieldMove)
            val ep = epSquare
            if (ep != null && ep != 0 && BB_SQUARES[ep] and target == BB_EMPTY) {
                val lastDouble = ep + (if (turn == WHITE) -8 else 8)
                if (lastDouble == checker) generatePseudoLegalEp(fromMask, toMask, yieldMove)
            }
        }
    }

    fun generateLegalMoves(
        fromMask: Bitboard = BB_ALL,
        toMask: Bitboard = BB_ALL,
        yieldMove: (Move) -> Unit,
    ) {
        val kingMask = kings and occupiedCo[if (turn) 1 else 0]
        if (kingMask != BB_EMPTY) {
            val king = msb(kingMask)
            val blockers = sliderBlockers(king)
            val checkers = attackersMask(!turn, king)
            if (checkers != BB_EMPTY) {
                generateEvasions(king, checkers, fromMask, toMask) { move ->
                    if (isSafe(king, blockers, move)) yieldMove(move)
                }
            } else {
                generatePseudoLegalMoves(fromMask, toMask) { move ->
                    if (isSafe(king, blockers, move)) yieldMove(move)
                }
            }
        } else {
            generatePseudoLegalMoves(fromMask, toMask, yieldMove)
        }
    }

    fun legalMoves(): List<Move> {
        val out = ArrayList<Move>(48)
        generateLegalMoves { out.add(it) }
        return out
    }

    fun anyLegalMove(): Boolean {
        var any = false
        generateLegalMoves { any = true }
        return any
    }

    fun hasLegalEnPassant(): Boolean {
        if (epSquare == null) return false
        var any = false
        generatePseudoLegalEp { move -> if (!isIntoCheck(move)) any = true }
        return any
    }

    // Making and unmaking

    private fun toChess960(move: Move): Move {
        if (move.from == E1 && kings and BB_SQUARES[E1] != BB_EMPTY) {
            if (move.to == G1 && rooks and BB_SQUARES[G1] == BB_EMPTY) return Move(E1, H1)
            if (move.to == C1 && rooks and BB_SQUARES[C1] == BB_EMPTY) return Move(E1, A1)
        } else if (move.from == E8 && kings and BB_SQUARES[E8] != BB_EMPTY) {
            if (move.to == G8 && rooks and BB_SQUARES[G8] == BB_EMPTY) return Move(E8, H8)
            if (move.to == C8 && rooks and BB_SQUARES[C8] == BB_EMPTY) return Move(E8, A8)
        }
        return move
    }

    private fun fromChess960(from: Int, to: Int, promotion: Int = 0): Move {
        if (promotion == 0) {
            if (from == E1 && kings and BB_SQUARES[E1] != BB_EMPTY) {
                if (to == H1) return Move(E1, G1)
                if (to == A1) return Move(E1, C1)
            } else if (from == E8 && kings and BB_SQUARES[E8] != BB_EMPTY) {
                if (to == H8) return Move(E8, G8)
                if (to == A8) return Move(E8, C8)
            }
        }
        return Move(from, to, promotion)
    }

    fun push(given: Move) {
        val move = toChess960(given)
        val state = BoardState(this)
        castlingRights = cleanCastlingRights()
        moveStack.add(fromChess960(move.from, move.to, move.promotion))
        stateStack.add(state)

        val ep = epSquare
        epSquare = null

        halfmoveClock += 1
        if (turn == BLACK) fullmoveNumber += 1

        if (isZeroing(move)) halfmoveClock = 0

        val fromBb = BB_SQUARES[move.from]
        val toBb = BB_SQUARES[move.to]

        var promotedPiece = promoted and fromBb != BB_EMPTY
        var pieceType = removePieceAt(move.from)
        var captureSquare = move.to
        var capturedPieceType = pieceTypeAt(captureSquare)

        castlingRights = castlingRights and toBb.inv() and fromBb.inv()
        if (pieceType == KING && !promotedPiece) {
            castlingRights = castlingRights and (if (turn == WHITE) BB_RANK_1.inv() else BB_RANK_8.inv())
        } else if (capturedPieceType == KING && promoted and toBb == BB_EMPTY) {
            if (turn == WHITE && squareRank(move.to) == 7) {
                castlingRights = castlingRights and BB_RANK_8.inv()
            } else if (turn == BLACK && squareRank(move.to) == 0) {
                castlingRights = castlingRights and BB_RANK_1.inv()
            }
        }

        if (pieceType == PAWN) {
            val diff = move.to - move.from
            if (diff == 16 && squareRank(move.from) == 1) {
                epSquare = move.from + 8
            } else if (diff == -16 && squareRank(move.from) == 6) {
                epSquare = move.from - 8
            } else if (ep != null && move.to == ep && (abs(diff) == 7 || abs(diff) == 9) &&
                capturedPieceType == 0
            ) {
                val down = if (turn == WHITE) -8 else 8
                captureSquare = ep + down
                capturedPieceType = removePieceAt(captureSquare)
            }
        }

        if (move.promotion != 0) {
            promotedPiece = true
            pieceType = move.promotion
        }

        val castling = pieceType == KING && occupiedCo[if (turn) 1 else 0] and toBb != BB_EMPTY
        if (castling) {
            val aSide = squareFile(move.to) < squareFile(move.from)
            removePieceAt(move.from)
            removePieceAt(move.to)
            if (aSide) {
                setPieceAt(if (turn == WHITE) C1 else C8, Piece(KING, turn))
                setPieceAt(if (turn == WHITE) D1 else D8, Piece(ROOK, turn))
            } else {
                setPieceAt(if (turn == WHITE) G1 else G8, Piece(KING, turn))
                setPieceAt(if (turn == WHITE) F1 else F8, Piece(ROOK, turn))
            }
        }

        if (!castling) setPieceAt(move.to, Piece(pieceType, turn), promotedPiece)

        turn = !turn
    }

    fun pop(): Move {
        val move = moveStack.removeAt(moveStack.size - 1)
        stateStack.removeAt(stateStack.size - 1).restore(this)
        return move
    }

    fun copy(withStack: Boolean = true): Board {
        val board = Board(null)
        board.pawns = pawns; board.knights = knights; board.bishops = bishops
        board.rooks = rooks; board.queens = queens; board.kings = kings
        board.promoted = promoted; board.occupied = occupied
        board.occupiedCo[0] = occupiedCo[0]; board.occupiedCo[1] = occupiedCo[1]
        board.turn = turn
        board.castlingRights = castlingRights
        board.epSquare = epSquare
        board.halfmoveClock = halfmoveClock
        board.fullmoveNumber = fullmoveNumber
        if (withStack) {
            board.moveStack.addAll(moveStack)
            board.stateStack.addAll(stateStack)
        }
        return board
    }

    // Outcomes

    fun isCheckmate(): Boolean = isCheck() && !anyLegalMove()

    fun isStalemate(): Boolean = !isCheck() && !anyLegalMove()

    fun isInsufficientMaterial(): Boolean =
        hasInsufficientMaterial(WHITE) && hasInsufficientMaterial(BLACK)

    fun hasInsufficientMaterial(colour: Boolean): Boolean {
        val ours = occupiedCo[if (colour) 1 else 0]
        val theirs = occupiedCo[if (!colour) 1 else 0]
        if (ours and (pawns or rooks or queens) != BB_EMPTY) return false
        // A knight mates only by the opponent's help, and only when they have nothing
        // else to move; bishops of one colour cannot mate at all.
        if (ours and knights != BB_EMPTY) {
            return popcount(ours) <= 2 && theirs and kings.inv() and queens.inv() == BB_EMPTY
        }
        if (ours and bishops != BB_EMPTY) {
            val sameColour = bishops and BB_DARK_SQUARES == BB_EMPTY ||
                bishops and BB_LIGHT_SQUARES == BB_EMPTY
            return sameColour && pawns == BB_EMPTY && knights == BB_EMPTY
        }
        return true
    }

    fun isSeventyfiveMoves(): Boolean = halfmoveClock >= 150 && anyLegalMove()

    fun isFiftyMoves(): Boolean = halfmoveClock >= 100 && anyLegalMove()

    /** `is_repetition`: the same position, side and rights seen `count` times. */
    fun isRepetition(wanted: Int = 3): Boolean {
        var count = wanted
        // The cheap check first: the same squares occupied, whatever stands on them.
        var maybeRepetitions = 1
        for (i in stateStack.indices.reversed()) {
            if (stateStack[i].occupied == occupied) {
                maybeRepetitions += 1
                if (maybeRepetitions >= count) break
            }
        }
        if (maybeRepetitions < count) return false

        val transpositionKey = transpositionKey()
        val switchyard = ArrayList<Move>()
        var repeated = false
        while (true) {
            if (count <= 1) { repeated = true; break }
            if (moveStack.size < count - 1) break
            val move = pop()
            switchyard.add(move)
            if (isIrreversible(move)) break
            if (transpositionKey() == transpositionKey) count -= 1
        }
        while (switchyard.isNotEmpty()) push(switchyard.removeAt(switchyard.size - 1))
        return repeated
    }

    private fun reducesCastlingRights(move: Move): Boolean {
        val cr = cleanCastlingRights()
        val touched = BB_SQUARES[move.from] xor BB_SQUARES[move.to]
        return touched and cr != BB_EMPTY ||
            (cr and BB_RANK_1 != BB_EMPTY && touched and kings and occupiedCo[1] and promoted.inv() != BB_EMPTY) ||
            (cr and BB_RANK_8 != BB_EMPTY && touched and kings and occupiedCo[0] and promoted.inv() != BB_EMPTY)
    }

    fun isIrreversible(move: Move): Boolean =
        isZeroing(move) || reducesCastlingRights(move) || hasLegalEnPassant()

    private fun transpositionKey(): String =
        boardFen() + " " + (if (turn) "w" else "b") + " " +
            cleanCastlingRights().toString(16) + " " +
            (if (hasLegalEnPassant()) squareName(epSquare!!) else "-")

    fun canClaimThreefoldRepetition(): Boolean {
        val transpositionKey = transpositionKey()
        val counts = HashMap<String, Int>()
        counts[transpositionKey] = 1
        val switchyard = ArrayList<Move>()
        while (moveStack.isNotEmpty()) {
            val move = pop()
            switchyard.add(move)
            if (isIrreversible(move)) break
            val key = transpositionKey()
            counts[key] = (counts[key] ?: 0) + 1
        }
        for (i in switchyard.indices.reversed()) push(switchyard[i])
        if ((counts[transpositionKey] ?: 0) >= 3) return true
        // A move that brings the position about for the third time can be claimed as well.
        for (move in legalMoves()) {
            push(move)
            val third = (counts[transpositionKey()] ?: 0) >= 2
            pop()
            if (third) return true
        }
        return false
    }

    fun canClaimFiftyMoves(): Boolean {
        if (isFiftyMoves()) return true
        if (halfmoveClock >= 99) {
            // The list is taken first: the generator reads the board as it walks it, and
            // these trials change the board.
            for (move in legalMoves()) {
                if (isZeroing(move)) continue
                push(move)
                val fifty = isFiftyMoves()
                pop()
                if (fifty) return true
            }
        }
        return false
    }

    fun isFivefoldRepetition(): Boolean = isRepetition(5)

    /** How a game ended. `winner` is null for a draw, as python-chess's `Outcome` has it. */
    data class Outcome(val winner: Boolean?)

    /**
     * The outcome, or null while the game is on. `claimDraw` adds the draws a player has to
     * claim -- the fifty-move rule and a threefold repetition -- to the ones that end a game
     * by themselves.
     */
    fun outcome(claimDraw: Boolean = false): Outcome? {
        if (isCheckmate()) return Outcome(!turn)
        if (isInsufficientMaterial()) return Outcome(null)
        if (!anyLegalMove()) return Outcome(null)
        if (isSeventyfiveMoves()) return Outcome(null)
        if (isFivefoldRepetition()) return Outcome(null)
        if (claimDraw) {
            if (canClaimFiftyMoves()) return Outcome(null)
            if (canClaimThreefoldRepetition()) return Outcome(null)
        }
        return null
    }

    fun isGameOver(claimDraw: Boolean = false): Boolean = outcome(claimDraw) != null

    fun result(claimDraw: Boolean = false): String {
        val outcome = outcome(claimDraw) ?: return "*"
        val winner = outcome.winner ?: return "1/2-1/2"
        return if (winner == WHITE) "1-0" else "0-1"
    }

    /** `status() == STATUS_VALID`, bar the variant checks the desktop never meets. */
    fun isValid(): Boolean {
        if (occupied == BB_EMPTY) return false
        if (occupiedCo[1] and kings == BB_EMPTY) return false
        if (occupiedCo[0] and kings == BB_EMPTY) return false
        if (popcount(occupied and kings) > 2) return false
        if (popcount(occupiedCo[1]) > 16) return false
        if (popcount(occupiedCo[0]) > 16) return false
        if (popcount(occupiedCo[1] and pawns) > 8) return false
        if (popcount(occupiedCo[0] and pawns) > 8) return false
        if (pawns and BB_BACKRANKS != BB_EMPTY) return false
        if (castlingRights != cleanCastlingRights()) return false
        val validEp = validEpSquare()
        if (epSquare != validEp) return false
        if (wasIntoCheck()) return false
        val checkers = checkersMask()
        val ourKings = kings and occupiedCo[if (turn) 1 else 0] and promoted.inv()
        if (checkers != BB_EMPTY) {
            if (popcount(checkers) > 2) return false
            if (validEp != null) {
                val pushedTo = validEp xor 8
                val pushedFrom = validEp xor 24
                val occupiedBefore = (occupied and BB_SQUARES[pushedTo].inv()) or BB_SQUARES[pushedFrom]
                if (popcount(checkers) > 1 ||
                    (msb(checkers) != pushedTo && attackedForKing(ourKings, occupiedBefore))
                ) {
                    return false
                }
            } else {
                if (popcount(checkers) > 2 ||
                    (popcount(checkers) == 2 && ray(lsb(checkers), msb(checkers)) and ourKings != BB_EMPTY)
                ) {
                    return false
                }
            }
        }
        return true
    }

    private fun validEpSquare(): Int? {
        val ep = epSquare ?: return null
        val epRank: Int
        val pawnMask: Bitboard
        val seventhRankMask: Bitboard
        if (turn == WHITE) {
            epRank = 5
            pawnMask = BB_SQUARES[ep] shr 8
            seventhRankMask = BB_SQUARES[ep] shl 8
        } else {
            epRank = 2
            pawnMask = BB_SQUARES[ep] shl 8
            seventhRankMask = BB_SQUARES[ep] shr 8
        }
        if (squareRank(ep) != epRank) return null
        if (pawns and occupiedCo[if (!turn) 1 else 0] and pawnMask == BB_EMPTY) return null
        if (occupied and BB_SQUARES[ep] != BB_EMPTY) return null
        if (occupied and seventhRankMask != BB_EMPTY) return null
        return ep
    }

    // Notation

    fun san(move: Move): String {
        val san = algebraicWithoutSuffix(move)
        push(move)
        val check = isCheck()
        val mate = check && isCheckmate()
        pop()
        return san + if (mate) "#" else if (check) "+" else ""
    }

    private fun algebraicWithoutSuffix(move: Move): String {
        if (isCastling(move)) {
            return if (squareFile(move.to) < squareFile(move.from)) "O-O-O" else "O-O"
        }
        val pieceType = pieceTypeAt(move.from)
        val capture = isCapture(move)
        val san = StringBuilder()
        if (pieceType != PAWN) san.append(PIECE_SYMBOLS[pieceType].uppercase())

        if (pieceType != PAWN) {
            var others = BB_EMPTY
            var fromMask = piecesMask(pieceType, turn)
            fromMask = fromMask and BB_SQUARES[move.from].inv()
            val toMask = BB_SQUARES[move.to]
            generateLegalMoves(fromMask, toMask) { candidate -> others = others or BB_SQUARES[candidate.from] }
            if (others != BB_EMPTY) {
                var row = false
                var column = false
                if (others and BB_RANKS[squareRank(move.from)] != BB_EMPTY) column = true
                if (others and BB_FILES[squareFile(move.from)] != BB_EMPTY) row = true else column = true
                if (column) san.append(FILE_NAMES[squareFile(move.from)])
                if (row) san.append(RANK_NAMES[squareRank(move.from)])
            }
        } else if (capture) {
            san.append(FILE_NAMES[squareFile(move.from)])
        }

        if (capture) san.append('x')
        san.append(squareName(move.to))
        if (move.promotion != 0) san.append('=').append(PIECE_SYMBOLS[move.promotion].uppercase())
        return san.toString()
    }

    /**
     * A move from its SAN, or null when the text names no legal move or names more than one.
     *
     * A parser, not a comparison with what [san] would write: python-chess accepts "Nxf7"
     * for a knight moving to an empty f7, and the coach's own prose is read back through
     * this, so a mark drawn from a sentence has to resolve the same way.
     */
    fun parseSan(text: String): Move? {
        val san = text.trim()
        if (san in listOf("O-O", "O-O+", "O-O#", "0-0", "0-0+", "0-0#")) {
            var found: Move? = null
            generateCastlingMoves { if (found == null && isKingsideCastling(it)) found = it }
            return found
        }
        if (san in listOf("O-O-O", "O-O-O+", "O-O-O#", "0-0-0", "0-0-0+", "0-0-0#")) {
            var found: Move? = null
            generateCastlingMoves { if (found == null && isQueensideCastling(it)) found = it }
            return found
        }
        val match = SAN_REGEX.matchEntire(san) ?: return null

        // The target square. Our own pieces are masked out to exclude castling moves.
        val toSquare = parseSquare(match.groupValues[4])
        val toMask = BB_SQUARES[toSquare] and occupiedCo[if (turn) 1 else 0].inv()

        val promotionText = match.groupValues[5]
        val promotion = if (promotionText.isEmpty()) {
            0
        } else {
            PIECE_SYMBOLS.indexOf(promotionText.last().lowercaseChar().toString())
        }

        var fromMask = BB_ALL
        val fileText = match.groupValues[2]
        val rankText = match.groupValues[3]
        if (fileText.isNotEmpty()) fromMask = fromMask and BB_FILES[FILE_NAMES.indexOf(fileText[0])]
        if (rankText.isNotEmpty()) fromMask = fromMask and BB_RANKS[rankText.toInt() - 1]

        val pieceText = match.groupValues[1]
        if (pieceText.isNotEmpty()) {
            val pieceType = PIECE_SYMBOLS.indexOf(pieceText.lowercase())
            fromMask = fromMask and piecesMask(pieceType, turn)
        } else if (fileText.isNotEmpty() && rankText.isNotEmpty()) {
            // A fully specified move, even one that is not a pawn move.
            val from = square(FILE_NAMES.indexOf(fileText[0]), rankText.toInt() - 1)
            val move = findMove(from, toSquare, promotion)
            return if (move != null && move.promotion == promotion) move else null
        } else {
            fromMask = fromMask and pawns
            // A pawn capture needs its file; without one the pawn is on the target's file.
            if (fileText.isEmpty()) fromMask = fromMask and BB_FILES[squareFile(toSquare)]
        }

        var matched: Move? = null
        var ambiguous = false
        generateLegalMoves(fromMask, toMask) { move ->
            if (move.promotion == promotion) {
                if (matched != null) ambiguous = true
                if (matched == null) matched = move
            }
        }
        if (ambiguous) return null
        return matched
    }

    /** `find_move`: the legal move between two squares, promoting to a queen by default. */
    fun findMove(from: Int, to: Int, promotion: Int = 0): Move? {
        var wanted = promotion
        if (wanted == 0 && pawns and BB_SQUARES[from] != BB_EMPTY &&
            BB_SQUARES[to] and BB_BACKRANKS != BB_EMPTY
        ) {
            wanted = QUEEN
        }
        val move = fromChess960(from, to, wanted)
        return if (isLegal(move)) move else null
    }

    fun boardFen(): String {
        val builder = StringBuilder()
        var empty = 0
        for (square in SQUARES_180) {
            val piece = pieceAt(square)
            if (piece == null) {
                empty += 1
            } else {
                if (empty > 0) { builder.append(empty); empty = 0 }
                builder.append(piece.symbol())
            }
            if (BB_SQUARES[square] and BB_FILES[7] != BB_EMPTY) {
                if (empty > 0) { builder.append(empty); empty = 0 }
                if (square != H1) builder.append('/')
            }
        }
        return builder.toString()
    }

    private fun castlingXfen(): String {
        val builder = StringBuilder()
        for (colour in booleanArrayOf(WHITE, BLACK)) {
            val kingSquare = king(colour) ?: continue
            val kingFile = squareFile(kingSquare)
            val backrank = if (colour == WHITE) BB_RANK_1 else BB_RANK_8
            scanReversed(cleanCastlingRights() and backrank) { rookSquare ->
                val rookFile = squareFile(rookSquare)
                val aSide = rookFile < kingFile
                val otherRooks = occupiedCo[if (colour) 1 else 0] and rooks and backrank and
                    BB_SQUARES[rookSquare].inv()
                var ambiguous = false
                scanReversed(otherRooks) { other -> if ((squareFile(other) < rookFile) == aSide) ambiguous = true }
                val ch = if (ambiguous) FILE_NAMES[rookFile].toString() else if (aSide) "q" else "k"
                builder.append(if (colour == WHITE) ch.uppercase() else ch)
            }
        }
        return if (builder.isEmpty()) "-" else builder.toString()
    }

    fun fen(): String {
        val ep = if (hasLegalEnPassant()) squareName(epSquare!!) else "-"
        return "${boardFen()} ${if (turn == WHITE) "w" else "b"} ${castlingXfen()} $ep $halfmoveClock $fullmoveNumber"
    }

    fun setFen(fen: String) {
        val parts = fen.trim().split(Regex("\\s+"))
        clearBoard()
        var square = 56
        for (ch in parts[0]) {
            when {
                ch == '/' -> square -= 16
                ch.isDigit() -> square += ch - '0'
                else -> { setPieceAt(square, Piece.fromSymbol(ch)); square += 1 }
            }
        }
        turn = parts.size < 2 || parts[1] == "w"
        castlingRights = BB_EMPTY
        if (parts.size > 2 && parts[2] != "-") {
            for (ch in parts[2]) {
                when (ch) {
                    'K' -> castlingRights = castlingRights or (BB_RANK_1 and rooks and occupiedCo[1] and BB_FILES[7])
                    'Q' -> castlingRights = castlingRights or (BB_RANK_1 and rooks and occupiedCo[1] and BB_FILES[0])
                    'k' -> castlingRights = castlingRights or (BB_RANK_8 and rooks and occupiedCo[0] and BB_FILES[7])
                    'q' -> castlingRights = castlingRights or (BB_RANK_8 and rooks and occupiedCo[0] and BB_FILES[0])
                    else -> {
                        val file = FILE_NAMES.indexOf(ch.lowercaseChar())
                        if (file >= 0) {
                            val backrank = if (ch.isUpperCase()) BB_RANK_1 else BB_RANK_8
                            val side = if (ch.isUpperCase()) 1 else 0
                            castlingRights = castlingRights or
                                (backrank and rooks and occupiedCo[side] and BB_FILES[file])
                        }
                    }
                }
            }
        }
        epSquare = if (parts.size > 3 && parts[3] != "-") parseSquare(parts[3]) else null
        halfmoveClock = if (parts.size > 4) parts[4].toIntOrNull() ?: 0 else 0
        fullmoveNumber = if (parts.size > 5) maxOf(parts[5].toIntOrNull() ?: 1, 1) else 1
    }

    companion object {
        /** python-chess's own SAN pattern, so the same texts parse and the same ones do not. */
        private val SAN_REGEX =
            Regex("([NBKRQ])?([a-h])?([1-8])?[\\-x]?([a-h][1-8])(=?[nbrqkNBRQK])?[+#]?")

        /** a8..h8, a7..h7, ... a1..h1: the order a FEN is written in. */
        val SQUARES_180 = IntArray(64) { i -> (7 - i / 8) * 8 + (i % 8) }
    }
}
