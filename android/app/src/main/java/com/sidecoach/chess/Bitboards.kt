package com.sidecoach.chess

/**
 * Squares, rays and attack tables: the board arithmetic `python-chess` does, in Kotlin.
 *
 * The desktop reconciler is written against python-chess, down to the order its move
 * generator yields moves in (a story of missed moves keeps the first of several
 * transpositions). So this is a port of that library's bitboard layer rather than a chess
 * engine of our own: same square numbering (a1 = 0, h8 = 63), same rays, same sliding
 * attacks. `tools/make_chess_fixtures.py` records what the Python answers for thousands of
 * positions and `ChessTest` holds this code to it.
 */

typealias Bitboard = ULong

const val BB_EMPTY: Bitboard = 0uL
const val BB_ALL: Bitboard = 0xFFFF_FFFF_FFFF_FFFFuL

const val A1 = 0
const val C1 = 2
const val D1 = 3
const val E1 = 4
const val F1 = 5
const val G1 = 6
const val H1 = 7
const val A8 = 56
const val C8 = 58
const val D8 = 59
const val E8 = 60
const val F8 = 61
const val G8 = 62
const val H8 = 63

val BB_SQUARES = ULongArray(64) { 1uL shl it }

val BB_FILES = ULongArray(8) { f -> (0 until 8).fold(BB_EMPTY) { acc, r -> acc or BB_SQUARES[r * 8 + f] } }
val BB_RANKS = ULongArray(8) { r -> 0xFFuL shl (r * 8) }

val BB_RANK_1 = BB_RANKS[0]
val BB_RANK_8 = BB_RANKS[7]
val BB_BACKRANKS = BB_RANK_1 or BB_RANK_8

const val BB_LIGHT_SQUARES: Bitboard = 0x55aa_55aa_55aa_55aauL
const val BB_DARK_SQUARES: Bitboard = 0xaa55_aa55_aa55_aa55uL

const val WHITE = true
const val BLACK = false

const val PAWN = 1
const val KNIGHT = 2
const val BISHOP = 3
const val ROOK = 4
const val QUEEN = 5
const val KING = 6

val PIECE_SYMBOLS = arrayOf("", "p", "n", "b", "r", "q", "k")
val FILE_NAMES = charArrayOf('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')
val RANK_NAMES = charArrayOf('1', '2', '3', '4', '5', '6', '7', '8')

fun squareFile(square: Int): Int = square and 7
fun squareRank(square: Int): Int = square shr 3
fun square(file: Int, rank: Int): Int = rank * 8 + file

fun squareName(square: Int): String = "${FILE_NAMES[squareFile(square)]}${RANK_NAMES[squareRank(square)]}"

fun parseSquare(name: String): Int =
    square(FILE_NAMES.indexOf(name[0]), RANK_NAMES.indexOf(name[1]))

/** Chebyshev distance, as `chess.square_distance`. */
fun squareDistance(a: Int, b: Int): Int =
    maxOf(kotlin.math.abs(squareFile(a) - squareFile(b)), kotlin.math.abs(squareRank(a) - squareRank(b)))

fun msb(bb: Bitboard): Int = 63 - bb.countLeadingZeroBits()

fun lsb(bb: Bitboard): Int = bb.countTrailingZeroBits()

fun popcount(bb: Bitboard): Int = bb.countOneBits()

/** `chess.SquareSet`'s own order: set squares from the lowest index up. */
inline fun scanForward(bb: Bitboard, body: (Int) -> Unit) {
    var rest = bb
    while (rest != BB_EMPTY) {
        val square = rest.countTrailingZeroBits()
        body(square)
        rest = rest and (rest - 1uL)
    }
}

/** The set squares, lowest first: what iterating a `SquareSet` gives. */
fun squaresOf(bb: Bitboard): List<Int> {
    val out = ArrayList<Int>(popcount(bb))
    scanForward(bb) { out.add(it) }
    return out
}

/** `chess.scan_reversed`: set squares from the highest index down. */
inline fun scanReversed(bb: Bitboard, body: (Int) -> Unit) {
    var rest = bb
    while (rest != BB_EMPTY) {
        val square = 63 - rest.countLeadingZeroBits()
        body(square)
        rest = rest xor BB_SQUARES[square]
    }
}

private val DIAG_DELTAS = intArrayOf(-9, -7, 7, 9)
private val FILE_DELTAS = intArrayOf(-8, 8)
private val RANK_DELTAS = intArrayOf(-1, 1)

/** `chess._sliding_attacks`: walk each ray until it leaves the board or hits a piece. */
private fun slidingAttacks(square: Int, occupied: Bitboard, deltas: IntArray): Bitboard {
    var attacks = BB_EMPTY
    for (delta in deltas) {
        var sq = square
        while (true) {
            val previous = sq
            sq += delta
            if (sq !in 0..63 || squareDistance(sq, previous) > 2) break
            attacks = attacks or BB_SQUARES[sq]
            if (occupied and BB_SQUARES[sq] != BB_EMPTY) break
        }
    }
    return attacks
}

fun diagAttacks(square: Int, occupied: Bitboard): Bitboard = slidingAttacks(square, occupied, DIAG_DELTAS)

fun fileAttacks(square: Int, occupied: Bitboard): Bitboard = slidingAttacks(square, occupied, FILE_DELTAS)

fun rankAttacks(square: Int, occupied: Bitboard): Bitboard = slidingAttacks(square, occupied, RANK_DELTAS)

/** The empty-board attacks python-chess reads as `BB_*_ATTACKS[square][0]`. */
val BB_DIAG_RAYS = ULongArray(64) { diagAttacks(it, BB_EMPTY) }
val BB_FILE_RAYS = ULongArray(64) { fileAttacks(it, BB_EMPTY) }
val BB_RANK_RAYS = ULongArray(64) { rankAttacks(it, BB_EMPTY) }

private fun stepAttacks(square: Int, deltas: IntArray): Bitboard =
    slidingAttacks(square, BB_ALL, deltas)

val BB_KNIGHT_ATTACKS = ULongArray(64) { stepAttacks(it, intArrayOf(17, 15, 10, 6, -17, -15, -10, -6)) }
val BB_KING_ATTACKS = ULongArray(64) { stepAttacks(it, intArrayOf(9, 8, 7, 1, -9, -8, -7, -1)) }

/** Indexed [colour][square] with white first, as `BB_PAWN_ATTACKS[not color]` is read. */
val BB_PAWN_ATTACKS = arrayOf(
    ULongArray(64) { stepAttacks(it, intArrayOf(-7, -9)) },  // black, index 0
    ULongArray(64) { stepAttacks(it, intArrayOf(7, 9)) },    // white, index 1
)

fun pawnAttacks(colour: Boolean, square: Int): Bitboard = BB_PAWN_ATTACKS[if (colour) 1 else 0][square]

private val BB_RAYS: Array<ULongArray> = Array(64) { a ->
    ULongArray(64) { b ->
        val bbB = BB_SQUARES[b]
        when {
            BB_DIAG_RAYS[a] and bbB != BB_EMPTY -> (BB_DIAG_RAYS[a] and BB_DIAG_RAYS[b]) or BB_SQUARES[a] or bbB
            BB_RANK_RAYS[a] and bbB != BB_EMPTY -> BB_RANK_RAYS[a] or BB_SQUARES[a]
            BB_FILE_RAYS[a] and bbB != BB_EMPTY -> BB_FILE_RAYS[a] or BB_SQUARES[a]
            else -> BB_EMPTY
        }
    }
}

/** `chess.ray`: the whole line through both squares, empty when they are not aligned. */
fun ray(a: Int, b: Int): Bitboard = BB_RAYS[a][b]

/**
 * `chess.square_knight_distance`: the moves a knight needs to get from one square to
 * another. The mating patterns use it for the kill-box.
 */
fun squareKnightDistance(a: Int, b: Int): Int {
    val dx = kotlin.math.abs(squareFile(a) - squareFile(b))
    val dy = kotlin.math.abs(squareRank(a) - squareRank(b))
    if (dx + dy == 1) return 3
    if (dx == 2 && dy == 2) return 4
    if (dx == 1 && dy == 1) {
        // The corner is the special case: a knight cannot triangulate against the edge.
        val corners = BB_SQUARES[A1] or BB_SQUARES[H1] or BB_SQUARES[A8] or BB_SQUARES[H8]
        return if (BB_SQUARES[a] and corners != BB_EMPTY || BB_SQUARES[b] and corners != BB_EMPTY) 4 else 2
    }
    val m = kotlin.math.ceil(
        maxOf(dx / 2.0, dy / 2.0, (dx + dy) / 3.0),
    ).toInt()
    return m + ((m + dx + dy) % 2)
}

/** `chess.between`: the squares strictly between two aligned squares. */
fun between(a: Int, b: Int): Bitboard {
    val bb = BB_RAYS[a][b] and ((BB_ALL shl a) xor (BB_ALL shl b))
    return bb and (bb - 1uL)
}
