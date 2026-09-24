package com.sidecoach.analysis

import com.sidecoach.chess.BB_EMPTY
import com.sidecoach.chess.BB_KING_ATTACKS
import com.sidecoach.chess.BB_SQUARES
import com.sidecoach.chess.BISHOP
import com.sidecoach.chess.Bitboard
import com.sidecoach.chess.Board
import com.sidecoach.chess.KNIGHT
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.square
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareKnightDistance
import com.sidecoach.chess.squareName
import com.sidecoach.chess.squareRank
import com.sidecoach.chess.squaresOf
import com.sidecoach.events.MotifTag
import kotlin.math.abs

/**
 * Named mating patterns: which textbook mate a checkmate is. A port of
 * `sidecoach/analysis/mates.py`.
 *
 * A coach that says "It's an Anastasia mate" is teaching a shape the player will meet again,
 * so the name has to be right every time: a name put on a mate that lacks the shape teaches
 * the wrong shape, which is worse than saying "a mating attack". Tag names are the Lichess
 * puzzle theme names, camelCase, spelled exactly as Lichess spells them, and the definitions
 * are the textbook ones as Lichess describes its themes. Two rules sit on top of them:
 *
 * 1. Every piece named as evidence is necessary. The knight of a Vukovic mate, the second
 *    bishop of a Boden, the queen of a box: each is named only if taking it off the board
 *    leaves no mate. A mate that still works without the piece is not made by it, however the
 *    pieces happen to stand (the 2026-09-23 audit found Boden, Vukovic and box mates named on
 *    positions that mate just as well with the "supporting" piece removed).
 * 2. A name that describes a picture needs the picture. A corner mate has the king in the
 *    corner, a dovetail has the two squares the queen cannot reach blocked by the king's own
 *    men, an epaulette has them on both shoulders. Lichess tags some of these more loosely (a
 *    "corner mate" of a king on f3); the coach does not follow it there. The mates named after
 *    people (Morphy, Pillsbury, Opera, Balestra) say which pieces do what, not where, and are
 *    read that way.
 *
 * Pure functions over a position that is already checkmate: no engine, no I/O, and the board
 * handed in is never mutated.
 */
object Mates {

    /**
     * The order the patterns are reported in, and the order they belong in
     * `Motifs.TAG_ORDER`: the Lichess patterns motifs already knew first, then the nine it
     * lacked.
     */
    val MATE_PATTERNS: List<String> = listOf(
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

    private val CORNERS = intArrayOf(0, 56, 7, 63)  // a1, a8, h1, h8

    private fun isHeavy(pieceType: Int): Boolean = pieceType == ROOK || pieceType == QUEEN

    // --- geometry --------------------------------------------------------------------------

    /** File difference from `a` to `b`; with [dRank], `_step`. */
    private fun dFile(a: Int, b: Int): Int = squareFile(b) - squareFile(a)

    private fun dRank(a: Int, b: Int): Int = squareRank(b) - squareRank(a)

    private fun at(sq: Int, df: Int, dr: Int): Int? {
        val file = squareFile(sq) + df
        val rank = squareRank(sq) + dr
        return if (file in 0..7 && rank in 0..7) square(file, rank) else null
    }

    /** Next to each other along a rank or a file. */
    private fun beside(a: Int, b: Int): Boolean = abs(dFile(a, b)) + abs(dRank(a, b)) == 1

    /** Next to each other on a diagonal. */
    private fun touchingCorners(a: Int, b: Int): Boolean = abs(dFile(a, b)) == 1 && abs(dRank(a, b)) == 1

    private fun onEdge(sq: Int): Boolean =
        squareFile(sq) == 0 || squareFile(sq) == 7 || squareRank(sq) == 0 || squareRank(sq) == 7

    /** Which way the diagonal through `a` and `b` runs, or null when they share none. */
    private fun diagonalKind(a: Int, b: Int): String? {
        val df = dFile(a, b)
        val dr = dRank(a, b)
        if (df == 0 || abs(df) != abs(dr)) return null
        return if (df == dr) "rising" else "falling"
    }

    private fun tag(name: String, squares: List<Int>): MotifTag {
        val names = ArrayList<String>()
        for (sq in squares) {
            val label = squareName(sq)
            if (label !in names) names.add(label)
        }
        return MotifTag(name = name, squares = names, move = null)
    }

    // --- the mated position, read once -----------------------------------------------------

    /**
     * The mated king, the piece that mates, and who guards each square around the king.
     *
     * `mater` is the piece that gives the mate: the one that moved when it gives check, else
     * the piece whose line it opened. The king's squares split into `own` (held by its own
     * men) and `free` (empty, or holding an enemy piece it might take). A square is guarded
     * by the pieces that attack it with the king lifted off the board, so a rook checking
     * along a file also guards the square behind the king on that file, which is the square
     * the king cannot step back to.
     */
    private class Mate(val board: Board, move: Move, val king: Int) {
        val loser = board.turn
        val winner = !board.turn
        val checkers: List<Int> = squaresOf(board.checkersMask())
        val mater: Int = if (move.to in checkers) move.to else checkers[0]
        val kind: Int = board.pieceTypeAt(mater)
        val ring: List<Int> = squaresOf(BB_KING_ATTACKS[king])
        val own: List<Int> = ring.filter { board.colourAt(it) == loser }
        val free: List<Int> = ring.filter { board.colourAt(it) != loser }

        /** Guards per square, zero off the ring, as the desktop's `.get(square, SquareSet())`. */
        private val guardsOf = ULongArray(64)
        private val neededOf = HashMap<Bitboard, Boolean>()

        init {
            val lifted = board.copy(withStack = false)
            lifted.removePieceAt(king)
            for (s in ring) guardsOf[s] = lifted.attackersMask(winner, s)
        }

        fun piece(sq: Int): Int = board.pieceTypeAt(sq)

        /** The mating side's pieces of one type. */
        fun mine(pieceType: Int): List<Int> = board.pieces(pieceType, winner)

        /** Who guards a square next to the king; for the mater's square, who defends it. */
        fun guards(sq: Int): Bitboard = guardsOf[sq]

        fun guardList(sq: Int): List<Int> = squaresOf(guardsOf[sq])

        /** The king's free squares this piece guards. */
        fun covers(piece: Int): List<Int> = free.filter { guardsOf[it] and BB_SQUARES[piece] != BB_EMPTY }

        /** The king's free squares no piece but this one guards. */
        fun sole(piece: Int): List<Int> = free.filter { guardsOf[it] == BB_SQUARES[piece] }

        /** Whether the mate falls apart once these pieces are taken off the board. */
        fun needed(vararg pieces: Int): Boolean {
            var key = BB_EMPTY
            for (sq in pieces) key = key or BB_SQUARES[sq]
            return neededOf.getOrPut(key) {
                val without = board.copy(withStack = false)
                for (sq in pieces) without.removePieceAt(sq)
                !without.isCheckmate()
            }
        }

        /**
         * Whether the mater and these pieces between them guard every free square but the
         * mater's own, the shape being theirs rather than a third piece's.
         */
        fun madeBy(vararg pieces: Int): Boolean {
            var named = BB_SQUARES[mater]
            for (sq in pieces) named = named or BB_SQUARES[sq]
            return free.all { it == mater || guardsOf[it] and named != BB_EMPTY }
        }

        fun materBeside(): Boolean = beside(king, mater)
    }

    // --- the patterns ------------------------------------------------------------------------

    /**
     * A rook or queen checks along the back rank, the king walled in by its own men in front.
     * Every square in front of the king holds one of its own pieces: a king held in by an enemy
     * piece covering its luft is mated on the back rank, but not by its own pawns.
     */
    private fun backRank(m: Mate): List<Int>? {
        val home = if (m.loser == WHITE) 0 else 7
        if (squareRank(m.king) != home) return null
        val heavy = m.checkers.filter { isHeavy(m.piece(it)) && squareRank(it) == home }
        if (heavy.isEmpty()) return null
        val ahead = if (m.loser == WHITE) 1 else -1
        val front = listOf(-1, 0, 1).mapNotNull { at(m.king, it, ahead) }
        if (front.any { m.board.colourAt(it) != m.loser }) return null
        return listOf(m.king) + heavy + front
    }

    /** A knight checks a king whose every neighbouring square holds one of its own men. */
    private fun smothered(m: Mate): List<Int>? {
        val knights = m.checkers.filter { m.piece(it) == KNIGHT }
        if (knights.isEmpty() || m.free.isNotEmpty()) return null
        return listOf(m.king, knights[0]) + m.own
    }

    /**
     * A rook mates beside a cornered king, held there by a knight that is needed. Rook and
     * knight take every square between them: a second knight or a bishop covering the last
     * square makes it some other mate.
     */
    private fun arabian(m: Mate): List<Int>? {
        if (m.kind != ROOK || m.king !in CORNERS || !m.materBeside()) return null
        for (knight in m.guardList(m.mater)) {
            if (m.piece(knight) == KNIGHT && m.needed(knight) && m.madeBy(knight)) {
                return listOf(m.king, m.mater, knight)
            }
        }
        return null
    }

    /**
     * A rook or queen mates down the edge file, the king boxed in by its own man and a knight.
     * The king stands on the a- or h-file, off the corner; its own piece stands beside it, one
     * file in; a knight takes both squares diagonally inward, above and below. That knight
     * must be needed: with a queen giving the check it can be doing nothing at all.
     */
    private fun anastasia(m: Mate): List<Int>? {
        val kingFile = squareFile(m.king)
        if ((kingFile != 0 && kingFile != 7) || squareRank(m.king) == 0 || squareRank(m.king) == 7) return null
        if (!isHeavy(m.kind) || squareFile(m.mater) != kingFile) return null
        val inward = if (kingFile == 0) 1 else -1
        val besideSq = at(m.king, inward, 0) ?: return null
        if (m.board.colourAt(besideSq) != m.loser) return null
        val flights = listOfNotNull(at(m.king, inward, 1), at(m.king, inward, -1))
        for (knight in m.mine(KNIGHT)) {
            val reach = m.board.attacksMask(knight)
            if (flights.all { reach and BB_SQUARES[it] != BB_EMPTY } && m.needed(knight)) {
                return listOf(m.king, m.mater, knight, besideSq)
            }
        }
        return null
    }

    /**
     * A rook beside the king, held by a knight, the knight held by a pawn; the king's own pawn
     * takes one of its squares. That is the textbook hook: rook, knight and pawn against one
     * enemy pawn. The knight must be needed, and so must the pawn behind it (it is when the
     * knight stands next to the king and would otherwise be taken). The enemy pawn must stand
     * on one of the king's squares that nothing of ours guards, so it is the pawn, not our
     * pieces, that shuts that square.
     */
    private fun hook(m: Mate): List<Int>? {
        if (m.kind != ROOK || !m.materBeside()) return null
        val blockers = m.own.filter { m.piece(it) == PAWN && m.guards(it) == BB_EMPTY }
        if (blockers.isEmpty()) return null
        for (knight in m.guardList(m.mater)) {
            if (m.piece(knight) != KNIGHT || !m.needed(knight)) continue
            for (pawn in m.board.attackers(m.winner, knight)) {
                if (m.piece(pawn) == PAWN && m.needed(pawn)) {
                    return listOf(m.king, m.mater, knight, pawn, blockers[0])
                }
            }
        }
        return null
    }

    /**
     * Rook and queen on opposite corners of a three-by-three box with the king inside it. They
     * guard each other along the box's diagonal, one of them gives the check, and the other
     * must be needed and must be the only one of the two to guard some empty square of the
     * king's: a queen that only backs up the rook's check is not making a box.
     */
    private fun killBox(m: Mate): List<Int>? {
        val kingFile = squareFile(m.king)
        val kingRank = squareRank(m.king)
        for (rook in m.mine(ROOK)) {
            for (queen in m.mine(QUEEN)) {
                if (m.mater != rook && m.mater != queen) continue
                val df = dFile(rook, queen)
                val dr = dRank(rook, queen)
                if (abs(df) != 2 || abs(dr) != 2) continue
                val middle = at(rook, df / 2, dr / 2)
                if (middle == null || (middle != m.king && m.board.pieceAt(middle) != null)) continue
                val files = minOf(squareFile(rook), squareFile(queen)) to maxOf(squareFile(rook), squareFile(queen))
                val ranks = minOf(squareRank(rook), squareRank(queen)) to maxOf(squareRank(rook), squareRank(queen))
                if (kingFile !in files.first..files.second || kingRank !in ranks.first..ranks.second) continue
                val other = if (m.mater == rook) queen else rook
                val alone = m.covers(other).filter {
                    m.board.pieceAt(it) == null && m.guards(it) and BB_SQUARES[m.mater] == BB_EMPTY
                }
                if (alone.isNotEmpty() && m.needed(other)) return listOf(m.king, m.mater, other)
            }
        }
        return null
    }

    /**
     * A rook mates beside a king on the edge, backed by a third piece, a knight taking squares.
     * The knight must be needed, and the rook's backing, the pieces other than that knight
     * defending it, must be needed too: a rook the king could not take anyway is not
     * "supported", it is just there.
     */
    private fun vukovic(m: Mate): List<Int>? {
        if (m.kind != ROOK || !m.materBeside() || !onEdge(m.king)) return null
        for (knight in m.mine(KNIGHT)) {
            if (m.covers(knight).none { it != m.mater } || !m.needed(knight)) continue
            val backing = m.guardList(m.mater).filter { it != knight }
            if (backing.isNotEmpty() && m.needed(*backing.toIntArray())) {
                return listOf(m.king, m.mater, knight) + backing
            }
        }
        return null
    }

    /**
     * A bishop mates and a second, needed bishop takes squares: Boden's or the double bishop.
     * Boden's diagonals cross, the second bishop working the other way from the check; in the
     * double-bishop mate they run side by side. The direction is read off the squares only the
     * second bishop guards, the reason it is needed. Both need the king hemmed in by at least
     * one of its own men, and the two bishops take every square between them.
     */
    private fun twoBishops(m: Mate): Pair<String, List<Int>>? {
        if (m.kind != BISHOP || m.own.isEmpty()) return null
        val along = diagonalKind(m.mater, m.king)
        for (other in m.mine(BISHOP)) {
            if (other == m.mater) continue
            val squares = m.sole(other).ifEmpty { m.covers(other) }
            if (squares.isEmpty() || !m.needed(other) || !m.madeBy(other)) continue
            val kinds = squares.map { diagonalKind(other, it) }.toSet()
            val evidence = listOf(m.king, m.mater, other) + squares
            if (kinds == setOf(along)) return "doubleBishopMate" to evidence
            if (along !in kinds) return "bodenMate" to evidence
        }
        return null
    }

    private fun boden(m: Mate): List<Int>? {
        val found = twoBishops(m)
        return if (found != null && found.first == "bodenMate") found.second else null
    }

    private fun doubleBishop(m: Mate): List<Int>? {
        val found = twoBishops(m)
        return if (found != null && found.first == "doubleBishopMate") found.second else null
    }

    /** The mating queen guards every free square of the king's but its own. */
    private fun queenAlone(m: Mate): Boolean =
        m.free.all { it == m.mater || m.guards(it) and BB_SQUARES[m.mater] != BB_EMPTY }

    /**
     * A queen mates from a square diagonally next to the king; the two squares it cannot
     * reach, beside and behind the king, hold the king's own men, and the queen covers the rest.
     */
    private fun dovetail(m: Mate): List<Int>? {
        if (m.kind != QUEEN || !touchingCorners(m.king, m.mater)) return null
        val df = dFile(m.king, m.mater)
        val dr = dRank(m.king, m.mater)
        val tail = listOf(at(m.king, -df, 0), at(m.king, 0, -dr))
        val blocked = tail.filterNotNull().filter { m.board.colourAt(it) == m.loser }
        if (blocked.size != 2 || !queenAlone(m)) return null
        return listOf(m.king, m.mater) + blocked
    }

    /**
     * A queen mates from the square straight next to the king; the two squares diagonally
     * behind the king, which it cannot reach, hold the king's own men, and it covers the rest.
     */
    private fun swallowsTail(m: Mate): List<Int>? {
        if (m.kind != QUEEN || !m.materBeside()) return null
        val df = dFile(m.king, m.mater)
        val dr = dRank(m.king, m.mater)
        val acrossF = dr
        val acrossR = df
        val tail = listOf(
            at(m.king, -df + acrossF, -dr + acrossR),
            at(m.king, -df - acrossF, -dr - acrossR),
        )
        val blocked = tail.filterNotNull().filter { m.board.colourAt(it) == m.loser }
        if (blocked.size != 2 || !queenAlone(m)) return null
        return listOf(m.king, m.mater) + blocked
    }

    /**
     * A rook or queen checks along a rank or file; the king's own men stand on both of its
     * shoulders, the two squares beside it across the line of the check.
     */
    private fun epaulette(m: Mate): List<Int>? {
        if (!isHeavy(m.kind)) return null
        val df = dFile(m.king, m.mater)
        val dr = dRank(m.king, m.mater)
        if (df != 0 && dr != 0) return null
        val acrossF = if (df == 0) 1 else 0
        val acrossR = if (df == 0) 0 else 1
        val shoulders = listOf(at(m.king, acrossF, acrossR), at(m.king, -acrossF, -acrossR))
        val blocked = shoulders.filterNotNull().filter { m.board.colourAt(it) == m.loser }
        if (blocked.size != 2) return null
        return listOf(m.king, m.mater) + blocked
    }

    /**
     * A queen mates from diagonally next to the king with a rook on the other near corner of
     * the same rank or file: king, queen and rook make the triangle. The rook must be needed.
     */
    private fun triangle(m: Mate): List<Int>? {
        if (m.kind != QUEEN || !touchingCorners(m.king, m.mater)) return null
        for (rook in m.mine(ROOK)) {
            if (!touchingCorners(m.king, rook)) continue
            val df = dFile(m.mater, rook)
            val dr = dRank(m.mater, rook)
            if ((df == 0 || dr == 0) && m.needed(rook)) return listOf(m.king, m.mater, rook)
        }
        return null
    }

    /**
     * Two rooks mate side by side next to the king, the three of them in one 2x2 square. One
     * rook checks from straight beside the king, the other stands next to it on the king's
     * diagonal, backing it up and taking squares; that second rook must be needed.
     */
    private fun blindSwine(m: Mate): List<Int>? {
        if (m.kind != ROOK || !m.materBeside()) return null
        for (rook in m.mine(ROOK)) {
            if (rook == m.mater) continue
            if (touchingCorners(m.king, rook) && beside(m.mater, rook) && m.needed(rook)) {
                return listOf(m.king, m.mater, rook)
            }
        }
        return null
    }

    /**
     * A knight mates a king in the corner that a rook or queen has shut in. The rook or queen
     * must guard a free square, be needed, and with the knight take every square. A corner
     * king hemmed in by its own men alone is a smothered mate, not this.
     */
    private fun corner(m: Mate): List<Int>? {
        if (m.kind != KNIGHT || m.king !in CORNERS || m.free.isEmpty()) return null
        for (heavy in m.mine(ROOK) + m.mine(QUEEN)) {
            if (m.covers(heavy).isNotEmpty() && m.needed(heavy) && m.madeBy(heavy)) {
                return listOf(m.king, m.mater, heavy)
            }
        }
        return null
    }

    /** A rook mates from straight next to the king, defended by a bishop that is needed. */
    private fun opera(m: Mate): List<Int>? {
        if (m.kind != ROOK || !m.materBeside()) return null
        for (bishop in m.guardList(m.mater)) {
            if (m.piece(bishop) == BISHOP && m.needed(bishop)) return listOf(m.king, m.mater, bishop)
        }
        return null
    }

    /**
     * A rook mates and a bishop, not the rook's defender, takes the king's squares. The bishop
     * must be needed. A bishop that only defends a rook next to the king makes the Opera mate
     * instead.
     */
    private fun pillsbury(m: Mate): List<Int>? {
        if (m.kind != ROOK) return null
        val defenders = if (m.materBeside()) m.guards(m.mater) else BB_EMPTY
        for (bishop in m.mine(BISHOP)) {
            if (defenders and BB_SQUARES[bishop] != BB_EMPTY || bishop in m.checkers) continue
            if (m.covers(bishop).isNotEmpty() && m.needed(bishop)) return listOf(m.king, m.mater, bishop)
        }
        return null
    }

    /**
     * A bishop mates and a rook takes the king's squares; the rook must be needed. A rook that
     * only defends a bishop next to the king is backing, not confining, so it has to guard a
     * square other than the bishop's.
     */
    private fun morphy(m: Mate): List<Int>? {
        if (m.kind != BISHOP) return null
        for (rook in m.mine(ROOK)) {
            if (rook in m.checkers) continue
            if (m.covers(rook).any { it != m.mater } && m.needed(rook)) return listOf(m.king, m.mater, rook)
        }
        return null
    }

    /**
     * A bishop mates from a distance and a queen a knight's move from the king takes every
     * square the bishop leaves. The queen gives no check itself and must be needed.
     */
    private fun balestra(m: Mate): List<Int>? {
        if (m.kind != BISHOP || m.materBeside() || touchingCorners(m.king, m.mater)) return null
        for (queen in m.mine(QUEEN)) {
            if (queen in m.checkers || squareKnightDistance(queen, m.king) != 1) continue
            val rest = m.free.filter { m.guards(it) and BB_SQUARES[m.mater] == BB_EMPTY }
            if (rest.all { m.guards(it) and BB_SQUARES[queen] != BB_EMPTY } && m.needed(queen)) {
                return listOf(m.king, m.mater, queen)
            }
        }
        return null
    }

    private val DETECTORS: Map<String, (Mate) -> List<Int>?> = mapOf(
        "backRankMate" to ::backRank,
        "smotheredMate" to ::smothered,
        "arabianMate" to ::arabian,
        "anastasiaMate" to ::anastasia,
        "hookMate" to ::hook,
        "killBoxMate" to ::killBox,
        "vukovicMate" to ::vukovic,
        "bodenMate" to ::boden,
        "doubleBishopMate" to ::doubleBishop,
        "dovetailMate" to ::dovetail,
        "swallowstailMate" to ::swallowsTail,
        "epauletteMate" to ::epaulette,
        "triangleMate" to ::triangle,
        "blindSwineMate" to ::blindSwine,
        "cornerMate" to ::corner,
        "operaMate" to ::opera,
        "pillsburysMate" to ::pillsbury,
        "morphysMate" to ::morphy,
        "balestraMate" to ::balestra,
    )

    // --- public API ----------------------------------------------------------------------

    /**
     * The named patterns of a checkmate, in [MATE_PATTERNS] order.
     *
     * `boardAfterMate` is the position with the mated side to move and `move` is the move
     * that mated. A position that is not checkmate names no pattern. Evidence squares start
     * with the king and the mating piece, then the pieces that help, then the squares that
     * make the shape.
     */
    fun matingPatternTags(boardAfterMate: Board, move: Move): List<MotifTag> {
        if (!boardAfterMate.isCheckmate()) return emptyList()
        val king = boardAfterMate.king(boardAfterMate.turn) ?: return emptyList()
        val mate = Mate(boardAfterMate, move, king)
        val tags = ArrayList<MotifTag>()
        for (name in MATE_PATTERNS) {
            val evidence = DETECTORS.getValue(name)(mate) ?: continue
            // In a double check the second checker is part of the mate whenever it is needed (it
            // stops the capture of the first), so a pattern that leaves it out is not the mate.
            if (mate.checkers.any { it != mate.mater && it !in evidence && mate.needed(it) }) continue
            tags.add(tag(name, evidence))
        }
        return tags
    }
}
