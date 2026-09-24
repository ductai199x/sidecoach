package com.sidecoach.analysis

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
import com.sidecoach.chess.square
import com.sidecoach.chess.squareDistance
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareName
import com.sidecoach.chess.squareRank
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Positional facts: the strategic ideas a coach teaches, grounded on the board. A port of
 * `sidecoach/analysis/positional.py`.
 *
 * The motif tagger answers "what tactic is here". This module answers the other half of the
 * teaching goal: "what is the plan". Every fact is computed from the board (and, for the
 * plan, the engine's own best move), never guessed, and carries the squares it stands on, so
 * the coach can point at them and the draft checker can verify the claim.
 *
 * Two kinds. [features] are static truths about the position (an open file, a passed pawn, a
 * knight on an outpost, an uncastled king). [bestMoveIdea] reads the strategic intent of the
 * engine's recommended move, which is the highest-value fact because it teaches why the move
 * is good.
 *
 * Nothing here evaluates: the engine already did.
 */

val PIECE_NAMES = arrayOf("", "pawn", "knight", "bishop", "rook", "queen", "king")

private val POS_VALUE = intArrayOf(0, 1, 3, 3, 5, 9, 0)
private const val FILES = "abcdefgh"
private val HOME_MINORS_WHITE = intArrayOf(1, 2, 5, 6)      // b1, c1, f1, g1
private val HOME_MINORS_BLACK = intArrayOf(57, 58, 61, 62)  // b8, c8, f8, g8

private fun homeMinors(colour: Boolean) = if (colour == WHITE) HOME_MINORS_WHITE else HOME_MINORS_BLACK

/**
 * One strategic truth about the position, in a coach's words, with its evidence squares.
 * `kind` is a stable slug (used to rank and to verify); `text` is the sentence a coach would
 * say; `squares` are the squares it rests on.
 */
data class PositionalFact(
    val kind: String,
    val text: String,
    val squares: List<String> = emptyList(),
)

private fun name(sq: Int): String = squareName(sq)

private fun forward(colour: Boolean): Int = if (colour == WHITE) 1 else -1

// --- static features -----------------------------------------------------------------

/** No enemy pawn can stop or capture the pawn on `square` on its way to promotion. */
fun isPassed(board: Board, sq: Int): Boolean {
    val pawn = board.pieceAt(sq) ?: return false
    if (pawn.pieceType != PAWN) return false
    val enemy = !pawn.colour
    val step = forward(pawn.colour)
    val file = squareFile(sq)
    val rank = squareRank(sq)
    for (df in -1..1) {
        val f = file + df
        if (f !in 0..7) continue
        for (other in board.pieces(PAWN, enemy)) {
            if (squareFile(other) != f) continue
            val ahead = (squareRank(other) - rank) * step
            if (ahead > 0) return false
        }
    }
    return true
}

fun passedPawns(board: Board, us: Boolean): List<PositionalFact> {
    val facts = ArrayList<PositionalFact>()
    for (sq in board.pieces(PAWN, us)) {
        if (isPassed(board, sq)) {
            val file = FILES[squareFile(sq)]
            facts.add(
                PositionalFact(
                    "passedPawn",
                    "your $file-pawn on ${name(sq)} is a passed pawn",
                    listOf(name(sq)),
                ),
            )
        }
    }
    return facts
}

/**
 * The square of the pawn (endgame): can the defending king catch a running passed pawn on its
 * own? The rule matches the Lichess tablebase on pure races.
 */
private fun kingCatchesPawn(
    pawn: Int,
    pawnColour: Boolean,
    king: Int,
    defenderToMove: Boolean,
): Boolean {
    val pf = squareFile(pawn)
    val pr = squareRank(pawn)
    val kf = squareFile(king)
    val kr = squareRank(king)
    val moves: Int
    val promotionRank: Int
    if (pawnColour == WHITE) {
        moves = (7 - pr) - (if (pr == 1) 1 else 0)
        promotionRank = 7
    } else {
        moves = pr - (if (pr == 6) 1 else 0)
        promotionRank = 0
    }
    val kingDistance = max(abs(kf - pf), abs(kr - promotionRank))
    return kingDistance <= moves + (if (defenderToMove) 1 else 0)
}

/** Every square ahead of the pawn on its file is empty: a clean run to promotion. */
private fun clearRun(board: Board, sq: Int, us: Boolean): Boolean {
    val step = forward(us)
    var rank = squareRank(sq) + step
    while (rank in 0..7) {
        if (board.pieceAt(square(squareFile(sq), rank)) != null) return false
        rank += step
    }
    return true
}

/**
 * Decisive passed-pawn races in the endgame: a pawn that queens untouched, or one that
 * outruns the enemy king. Gated to the endgame, where a lone king really must chase a pawn.
 */
fun squareOfThePawn(board: Board, us: Boolean, phase: String = "middlegame"): List<PositionalFact> {
    if (phase != "endgame") return emptyList()
    val facts = ArrayList<PositionalFact>()
    val them = !us
    val enemyKing = board.king(them)
    val ourKing = board.king(us)
    if (enemyKing != null) {
        for (sq in board.pieces(PAWN, us)) {
            if (isPassed(board, sq) && clearRun(board, sq, us) &&
                !kingCatchesPawn(sq, us, enemyKing, board.turn == them)
            ) {
                val letter = FILES[squareFile(sq)]
                facts.add(
                    PositionalFact(
                        "runningPawn",
                        "your passed $letter-pawn on ${name(sq)} runs to queen on its own -- " +
                            "the enemy king is outside its square",
                        listOf(name(sq)),
                    ),
                )
            }
        }
    }
    if (ourKing != null) {
        for (sq in board.pieces(PAWN, them)) {
            if (isPassed(board, sq) && clearRun(board, sq, them) &&
                !kingCatchesPawn(sq, them, ourKing, board.turn == us)
            ) {
                val letter = FILES[squareFile(sq)]
                facts.add(
                    PositionalFact(
                        "outrunPawn",
                        "their passed $letter-pawn on ${name(sq)} is outside your king's " +
                            "square -- you cannot catch it alone, so stop it another way",
                        listOf(name(sq)),
                    ),
                )
            }
        }
    }
    return facts
}

private fun onlyKingsAndPawns(board: Board): Boolean =
    board.pieceMap().values.all { it.pieceType == KING || it.pieceType == PAWN }

/**
 * Who holds the opposition in a king-and-pawn endgame: a true geometric fact, stated
 * neutrally, since the opposition only wins in the right pawn structure.
 */
fun opposition(board: Board, us: Boolean): List<PositionalFact> {
    if (!onlyKingsAndPawns(board) || board.pawns == com.sidecoach.chess.BB_EMPTY) return emptyList()
    val whiteKing = board.king(WHITE) ?: return emptyList()
    val blackKing = board.king(BLACK) ?: return emptyList()
    val files = abs(squareFile(whiteKing) - squareFile(blackKing))
    val ranks = abs(squareRank(whiteKing) - squareRank(blackKing))
    val aligned = (files == 0 && ranks in intArrayOf(2, 4, 6).toList()) ||
        (ranks == 0 && files in intArrayOf(2, 4, 6).toList()) ||
        (files == ranks && files in intArrayOf(2, 4, 6).toList())
    if (!aligned) return emptyList()
    val label = if (max(files, ranks) == 2) "opposition" else "distant opposition"
    val squares = listOf(name(whiteKing), name(blackKing))
    // The side to move must give way; the other holds the opposition.
    val holder = !board.turn
    if (holder == us) {
        return listOf(
            PositionalFact(
                "opposition",
                "you have the $label: the enemy king must give way first",
                squares,
            ),
        )
    }
    return listOf(
        PositionalFact(
            "oppositionAgainst",
            "your opponent has the $label, so your king must give way -- look to seize it back",
            squares,
        ),
    )
}

/**
 * The king's key squares for a non-rook pawn: reach one and the pawn queens. The ranks were
 * checked against the Lichess tablebase. A rook pawn has none (usually a draw).
 */
private fun keySquaresOf(pawn: Int, us: Boolean): List<Int> {
    val pf = squareFile(pawn)
    val pr = squareRank(pawn)
    if (pf == 0 || pf == 7) return emptyList()
    val ranks = if (us == WHITE) {
        if (pr <= 3) listOf(pr + 2) else listOf(pr + 1, pr + 2)
    } else {
        if (pr >= 4) listOf(pr - 2) else listOf(pr - 1, pr - 2)
    }
    val out = ArrayList<Int>()
    for (kr in ranks) {
        for (f in intArrayOf(pf - 1, pf, pf + 1)) {
            if (f in 0..7 && kr in 0..7) out.add(square(f, kr))
        }
    }
    return out
}

/**
 * The king's key squares in a king-and-pawn-versus-king ending. Only the clean textbook case
 * -- your lone non-rook pawn, bare kings -- where the rule is exact.
 */
fun keySquares(board: Board, us: Boolean): List<PositionalFact> {
    if (!onlyKingsAndPawns(board)) return emptyList()
    val ours = board.pieces(PAWN, us)
    if (ours.size != 1 || board.pieces(PAWN, !us).isNotEmpty()) return emptyList()
    val keys = keySquaresOf(ours[0], us)
    val king = board.king(us)
    if (keys.isEmpty() || king == null) return emptyList()
    val letter = FILES[squareFile(ours[0])]
    if (king in keys) {
        return listOf(
            PositionalFact(
                "keySquare",
                "your king holds a key square on ${name(king)} -- your $letter-pawn promotes, so " +
                    "push it through",
                listOf(name(king), name(ours[0])),
            ),
        )
    }
    val sortedKeys = keys.sorted()
    val names = sortedKeys.joinToString(", ") { name(it) }
    return listOf(
        PositionalFact(
            "keySquareGoal",
            "march your king to a key square ($names) to force your $letter-pawn through",
            listOf(name(ours[0])) + sortedKeys.map { name(it) },
        ),
    )
}

/** Exactly a rook-and-pawn versus rook ending: your K+R+one pawn against their K+R. */
private fun rookAndPawnVsRook(board: Board, us: Boolean): Boolean {
    val them = !us
    fun n(pieceType: Int, colour: Boolean) = board.pieces(pieceType, colour).size
    return n(PAWN, us) == 1 &&
        n(PAWN, them) == 0 &&
        n(ROOK, us) == 1 &&
        n(ROOK, them) == 1 &&
        n(QUEEN, WHITE) == 0 && n(QUEEN, BLACK) == 0 &&
        n(BISHOP, WHITE) == 0 && n(BISHOP, BLACK) == 0 &&
        n(KNIGHT, WHITE) == 0 && n(KNIGHT, BLACK) == 0
}

/** The Lucena position: a won rook-and-pawn ending, won by building a bridge. */
fun lucena(board: Board, us: Boolean): List<PositionalFact> {
    if (!rookAndPawnVsRook(board, us)) return emptyList()
    val pawn = board.pieces(PAWN, us).first()
    val pf = squareFile(pawn)
    val pr = squareRank(pawn)
    val seventh = if (us == WHITE) 6 else 1
    val promotion = square(pf, if (us == WHITE) 7 else 0)
    val king = board.king(us)
    val defender = board.king(!us)
    if (pf == 0 || pf == 7 || pr != seventh || king == null || king != promotion || defender == null) {
        return emptyList()
    }
    // The enemy king is not cut off from the pawn.
    if (abs(squareFile(defender) - pf) < 1 || squareDistance(defender, pawn) < 2) return emptyList()
    return listOf(
        PositionalFact(
            "lucena",
            "a Lucena position: this rook-and-pawn ending is won -- build a bridge with your rook " +
                "to shelter your king from the checks, then bring it out from in front of the " +
                "${FILES[pf]}-pawn",
            listOf(name(king), name(pawn)),
        ),
    )
}

/** The Philidor position: the drawing defence in a rook-and-pawn ending. */
fun philidor(board: Board, us: Boolean): List<PositionalFact> {
    val them = !us
    // They own the pawn, we defend with K + R.
    if (!rookAndPawnVsRook(board, them)) return emptyList()
    val pawn = board.pieces(PAWN, them).first()
    val pf = squareFile(pawn)
    val pr = squareRank(pawn)
    val king = board.king(us) ?: return emptyList()
    val rook = board.pieces(ROOK, us).firstOrNull() ?: return emptyList()
    if (squareFile(king) != pf) return emptyList()
    val inFront = if (them == WHITE) squareRank(king) > pr else squareRank(king) < pr
    val thirdRank = if (us == WHITE) 2 else 5
    val onThird = squareRank(rook) == thirdRank
    val notYetSixth = if (them == WHITE) pr <= 4 else pr >= 3
    if (inFront && onThird && notYetSixth) {
        return listOf(
            PositionalFact(
                "philidor",
                "hold the Philidor draw: keep your rook on the third rank to stop their " +
                    "king reaching the sixth, then check from behind once the pawn advances",
                listOf(name(king), name(rook)),
            ),
        )
    }
    return emptyList()
}

// Space (Silman's rules of space): central manoeuvring room, measured the way an engine
// counts it -- the safe central squares behind your front that no enemy pawn attacks and your
// own pawn does not block.
private val CENTRAL_FILES = intArrayOf(2, 3, 4, 5)  // c, d, e, f
const val SPACE_MARGIN = 3

private fun spaceCount(board: Board, colour: Boolean): Int {
    val ranks = if (colour == WHITE) intArrayOf(1, 2, 3) else intArrayOf(6, 5, 4)
    val enemyPawnAttacks = HashSet<Int>()
    for (sq in board.pieces(PAWN, !colour)) enemyPawnAttacks.addAll(board.attacks(sq))
    var count = 0
    for (file in CENTRAL_FILES) {
        for (rank in ranks) {
            val sq = square(file, rank)
            val piece = board.pieceAt(sq)
            if (piece != null && piece.pieceType == PAWN && piece.colour == colour) continue
            if (sq in enemyPawnAttacks) continue
            count += 1
        }
    }
    return count
}

/** A space advantage (avoid trades) or a cramp (seek them), when it is pronounced. */
fun spacePlan(board: Board, us: Boolean): List<PositionalFact> {
    val advantage = spaceCount(board, us) - spaceCount(board, !us)
    val advanced: (Int) -> Boolean = if (us == WHITE) { r -> r >= 3 } else { r -> r <= 4 }
    val pawns = board.pieces(PAWN, us)
        .filter { squareFile(it) in CENTRAL_FILES.toList() && advanced(squareRank(it)) }
        .sorted()
    val squares = pawns.map { name(it) }
    if (advantage >= SPACE_MARGIN) {
        return listOf(
            PositionalFact(
                "spaceAdvantage",
                "you have more space: keep pieces on and improve your worst-placed one -- a " +
                    "space edge is a lasting plus, so avoid trades",
                squares,
            ),
        )
    }
    if (advantage <= -SPACE_MARGIN) {
        return listOf(
            PositionalFact(
                "spaceCramp",
                "you are cramped for space: trade a pair of pieces to give the rest room",
                squares,
            ),
        )
    }
    return emptyList()
}

// Wings for a pawn majority: the flanks, where a majority makes an outside passed pawn.
private val QUEENSIDE = intArrayOf(0, 1, 2)
private val KINGSIDE = intArrayOf(5, 6, 7)

/** A wing where you outnumber the opponent's pawns and can make a passed pawn. */
fun pawnMajority(board: Board, us: Boolean): List<PositionalFact> {
    val facts = ArrayList<PositionalFact>()
    val them = !us
    for ((wing, files) in listOf("queenside" to QUEENSIDE, "kingside" to KINGSIDE)) {
        val ours = board.pieces(PAWN, us).filter { squareFile(it) in files.toList() }
        val theirs = board.pieces(PAWN, them).count { squareFile(it) in files.toList() }
        val distinct = ours.map { squareFile(it) }.toSet().size
        if (theirs >= 1 && ours.size > theirs && distinct > theirs) {
            facts.add(
                PositionalFact(
                    "pawnMajority",
                    "your $wing pawn majority: advance it to make a passed pawn",
                    ours.sorted().map { name(it) },
                ),
            )
        }
    }
    return facts
}

private fun pawnsOnFile(board: Board, colour: Boolean, file: Int): List<Int> =
    board.pieces(PAWN, colour).filter { squareFile(it) == file }

/** A pawn with no friendly pawn on either adjacent file: it can never be defended by one. */
fun isIsolated(board: Board, sq: Int): Boolean {
    val pawn = board.pieceAt(sq) ?: return false
    if (pawn.pieceType != PAWN) return false
    val file = squareFile(sq)
    return intArrayOf(file - 1, file + 1)
        .filter { it in 0..7 }
        .none { pawnsOnFile(board, pawn.colour, it).isNotEmpty() }
}

/**
 * A pawn its neighbours have advanced past, whose advance square an enemy pawn guards: it
 * cannot advance safely and no friendly pawn is placed to defend it forward.
 */
fun isBackward(board: Board, sq: Int): Boolean {
    val pawn = board.pieceAt(sq) ?: return false
    if (pawn.pieceType != PAWN) return false
    val colour = pawn.colour
    val step = forward(colour)
    val file = squareFile(sq)
    val rank = squareRank(sq)
    if (rank + step !in 0..7) return false
    val advance = square(file, rank + step)
    if (board.attackers(!colour, advance).none { board.pieceTypeAt(it) == PAWN }) return false
    val neighbours = intArrayOf(file - 1, file + 1)
        .filter { it in 0..7 }
        .flatMap { pawnsOnFile(board, colour, it) }
    if (neighbours.isEmpty()) return false  // no neighbours at all is isolated, another weakness
    // If any neighbour is behind or level, it can come up to defend.
    return neighbours.all { (squareRank(it) - rank) * step > 0 }
}

private fun doubledFiles(board: Board, colour: Boolean): List<Pair<Int, List<Int>>> {
    val out = ArrayList<Pair<Int, List<Int>>>()
    for (file in 0 until 8) {
        val pawns = pawnsOnFile(board, colour, file)
        if (pawns.size >= 2) out.add(file to pawns.sorted())
    }
    return out
}

/** Structural pawn weaknesses, the enemy's first (targets), then your own (cautions). */
fun pawnWeaknesses(board: Board, us: Boolean): List<PositionalFact> {
    val them = !us
    val facts = ArrayList<PositionalFact>()

    fun note(colour: Boolean, kind: String) {
        val target = colour == them
        val attacker = !colour
        for (sq in board.pieces(PAWN, colour)) {
            val file = squareFile(sq)
            val letter = FILES[file]
            // A lone d-pawn is the isolated queen's pawn, its own structure.
            if (isIsolated(board, sq) && file != 3) {
                val text = if (target) {
                    "their isolated $letter-pawn on ${name(sq)} is a target to pile up on"
                } else {
                    "your isolated $letter-pawn on ${name(sq)} needs pieces to defend it"
                }
                facts.add(PositionalFact(kind, text, listOf(name(sq))))
            } else if (isBackward(board, sq) && pawnsOnFile(board, attacker, file).isEmpty()) {
                // A backward pawn is a real weakness only on a half-open file.
                val text = if (target) {
                    "their backward $letter-pawn on ${name(sq)} is a lasting target"
                } else {
                    "your backward $letter-pawn on ${name(sq)} is a long-term weakness"
                }
                facts.add(PositionalFact(kind, text, listOf(name(sq))))
            }
        }
        for ((file, squares) in doubledFiles(board, colour)) {
            val letter = FILES[file]
            // Doubled pawns are not purely a weakness: the file the capture vacated is
            // half-open for the owner's rooks.
            val halfOpen = intArrayOf(file - 1, file + 1)
                .filter { it in 0..7 && pawnsOnFile(board, colour, it).isEmpty() }
                .map { FILES[it] }
                .firstOrNull()
            val text = when {
                target -> "their doubled $letter-pawns are a lasting weakness to work against"
                halfOpen != null ->
                    "your doubled $letter-pawns are a long-term weakness, though they give " +
                        "your rooks the half-open $halfOpen-file"
                else -> "your doubled $letter-pawns are a long-term weakness"
            }
            facts.add(PositionalFact(kind, text, squares.map { name(it) }))
        }
    }

    note(them, "targetPawn")
    note(us, "weakPawn")
    return facts
}

/** The isolated queen's pawn, named for both sides and by phase. */
fun isolatedDPawn(board: Board, us: Boolean, phase: String = "middlegame"): List<PositionalFact> {
    val facts = ArrayList<PositionalFact>()
    for (colour in listOf(!us, us)) {  // the enemy's IQP is the plan, so it comes first
        for (sq in pawnsOnFile(board, colour, 3)) {
            if (!isIsolated(board, sq)) continue
            val label = name(sq)
            val file = squareFile(sq)
            val rank = squareRank(sq)
            val ahead = rank + forward(colour)
            val front = name(square(file, ahead))
            val outposts = intArrayOf(file - 1, file + 1)
                .filter { it in 0..7 }
                .map { name(square(it, ahead)) }
            if (colour == us) {
                if (phase == "endgame") {
                    facts.add(
                        PositionalFact(
                            "iqp",
                            "your isolated d-pawn on $label is a target now: its middlegame " +
                                "activity is spent, so hold it and trade rooks to ease the pressure",
                            listOf(label),
                        ),
                    )
                } else {
                    facts.add(
                        PositionalFact(
                            "iqp",
                            "your isolated d-pawn on $label, an IQP: play on its energy -- the " +
                                "${outposts.joinToString(" and ")} outposts it supports, the c- and e-files " +
                                "for your rooks, the break to $front -- keep pieces on and go for " +
                                "the attack; it turns into a target only in an endgame",
                            listOf(label) + outposts,
                        ),
                    )
                }
            } else {
                facts.add(
                    PositionalFact(
                        "iqpTarget",
                        "their isolated d-pawn on $label, an IQP, is a long-term target: " +
                            "blockade it on $front, trade the minor pieces (that ends its attack) " +
                            "and steer for an endgame where it is just a weakness",
                        listOf(label, front),
                    ),
                )
            }
        }
    }
    return facts
}

/** Two friendly pawns abreast on half-open files (the c- and d- or d- and e-pawns). */
fun hangingPawns(board: Board, us: Boolean): List<PositionalFact> {
    val facts = ArrayList<PositionalFact>()
    for (colour in listOf(!us, us)) {
        val rank = if (colour == WHITE) 3 else 4
        val pawns = board.pieces(PAWN, colour).filter { squareRank(it) == rank }.sorted()
        for (i in 0 until maxOf(0, pawns.size - 1)) {
            val first = pawns[i]
            val second = pawns[i + 1]
            val fa = squareFile(first)
            val fb = squareFile(second)
            if (fb != fa + 1 || !(fa >= 1 && fb <= 6)) continue
            if (pawnsOnFile(board, colour, fa - 1).isNotEmpty()) continue
            if (pawnsOnFile(board, colour, fb + 1).isNotEmpty()) continue
            val pair = "${FILES[fa]}- and ${FILES[fb]}-pawns"
            val flanks = "${FILES[fa - 1]}- and ${FILES[fb + 1]}-files"
            if (colour == us) {
                facts.add(
                    PositionalFact(
                        "hangingPawns",
                        "your $pair are hanging pawns: they hold the centre and give your " +
                            "rooks the $flanks; keep them mobile -- advancing one can fix an " +
                            "enemy pawn -- but once blockaded they are targets",
                        listOf(name(first), name(second)),
                    ),
                )
            } else {
                facts.add(
                    PositionalFact(
                        "hangingPawnsTarget",
                        "their $pair are hanging pawns: press them until one must advance and " +
                            "leaves a hole, or blockade the pair and work against the fixed pawn",
                        listOf(name(first), name(second)),
                    ),
                )
            }
        }
    }
    return facts
}

/** A locked centre: a central pawn ram, with both sides' d- and e-pawns committed forward. */
private fun closedCenter(board: Board): Boolean {
    var ram = false
    for (file in intArrayOf(3, 4)) {
        for (whitePawn in pawnsOnFile(board, WHITE, file)) {
            for (blackPawn in pawnsOnFile(board, BLACK, file)) {
                if (squareRank(blackPawn) == squareRank(whitePawn) + 1) ram = true
            }
        }
    }
    if (!ram) return false
    for (colour in listOf(WHITE, BLACK)) {
        val home = if (colour == WHITE) 1 else 6
        for (file in intArrayOf(3, 4)) {
            val pawns = pawnsOnFile(board, colour, file)
            if (pawns.isEmpty() || pawns.all { squareRank(it) == home }) return false
        }
    }
    return true
}

/** With a closed centre, which wing to play on -- the way your pawns point. */
fun pawnPointing(board: Board, us: Boolean): List<PositionalFact> {
    if (!closedCenter(board)) return emptyList()
    val dPawns = pawnsOnFile(board, us, 3)
    val ePawns = pawnsOnFile(board, us, 4)
    if (dPawns.isEmpty() || ePawns.isEmpty()) return emptyList()
    val dAdvance = dPawns.maxOf { squareRank(it) * forward(us) }
    val eAdvance = ePawns.maxOf { squareRank(it) * forward(us) }
    val wing: String
    val push: String
    when {
        eAdvance > dAdvance -> { wing = "kingside"; push = "f" }
        dAdvance > eAdvance -> { wing = "queenside"; push = "c" }
        else -> return emptyList()  // a broad front, leaning neither way
    }
    val squares = (dPawns + ePawns).sorted().map { name(it) }
    return listOf(
        PositionalFact(
            "pawnPointing",
            "the centre is closed, so play where your pawns point -- the $wing; advance your " +
                "$push-pawn to open lines there",
            squares,
        ),
    )
}

/** Open and half-open files that one of your rooks or queens already stands on. */
fun openFiles(board: Board, us: Boolean): List<PositionalFact> {
    val facts = ArrayList<PositionalFact>()
    val heavy = HashSet<Int>()
    for (pieceType in intArrayOf(ROOK, QUEEN)) heavy.addAll(board.pieces(pieceType, us))
    for (file in 0 until 8) {
        val squares = (0 until 8).map { square(file, it) }
        val pawns = squares.map { board.pieceAt(it) }
        val ourPawns = pawns.any { it != null && it.pieceType == PAWN && it.colour == us }
        val theirPawns = pawns.any { it != null && it.pieceType == PAWN && it.colour != us }
        // Sorted, as the desktop now sorts: with two heavy pieces on one file, which one
        // the coach names used to come out of CPython's set hashing.
        val onFile = heavy.filter { squareFile(it) == file }.sorted()
        if (ourPawns || onFile.isEmpty()) continue
        val letter = FILES[file]
        val where = name(onFile[0])
        val piece = PIECE_NAMES[board.pieceTypeAt(onFile[0])]
        if (theirPawns) {
            facts.add(
                PositionalFact(
                    "halfOpenFile",
                    "your $piece on $where has the half-open $letter-file",
                    listOf(where),
                ),
            )
        } else {
            facts.add(
                PositionalFact(
                    "openFile",
                    "your $piece on $where holds the open $letter-file",
                    listOf(where),
                ),
            )
        }
    }
    return facts
}

/** No enemy pawn can ever advance to attack `square` (the outpost pawn-safety test). */
private fun pawnSafe(board: Board, sq: Int, us: Boolean): Boolean {
    val enemy = !us
    val step = forward(us)
    val file = squareFile(sq)
    val rank = squareRank(sq)
    for (other in board.pieces(PAWN, enemy)) {
        if (abs(squareFile(other) - file) != 1) continue
        if ((squareRank(other) - rank) * step >= 1) return false
    }
    return true
}

/** A square in enemy territory, defended by one of your pawns, no enemy pawn can hit. */
internal fun isOutpost(board: Board, sq: Int, us: Boolean): Boolean {
    val rank = squareRank(sq)
    val reach = if (us == WHITE) rank >= 4 else rank <= 3
    if (!reach) return false
    if (board.attackers(us, sq).none { board.pieceTypeAt(it) == PAWN }) return false
    return pawnSafe(board, sq, us)
}

/** A knight or bishop of yours already sitting on an outpost -- a durable strong square. */
fun outposts(board: Board, us: Boolean): List<PositionalFact> {
    val facts = ArrayList<PositionalFact>()
    for (pieceType in intArrayOf(KNIGHT, BISHOP)) {
        for (sq in board.pieces(pieceType, us)) {
            if (isOutpost(board, sq, us)) {
                facts.add(
                    PositionalFact(
                        "outpost",
                        "your ${PIECE_NAMES[pieceType]} on ${name(sq)} sits on an outpost, " +
                            "a square no enemy pawn can challenge",
                        listOf(name(sq)),
                    ),
                )
            }
        }
    }
    return facts
}

/** Pawns of both colours on the d- and e-files: a simple read of how open the centre is. */
private fun centerPawnCount(board: Board): Int {
    var count = 0
    for (colour in listOf(WHITE, BLACK)) {
        for (sq in board.pieces(PAWN, colour)) {
            if (squareFile(sq) == 3 || squareFile(sq) == 4) count += 1
        }
    }
    return count
}

/** You keep both bishops while the opponent does not. */
fun bishopPair(board: Board, us: Boolean): List<PositionalFact> {
    val ours = board.pieces(BISHOP, us)
    val theirs = board.pieces(BISHOP, !us)
    if (!(ours.size >= 2 && theirs.size < 2)) return emptyList()
    val centre = centerPawnCount(board)
    val text = when {
        centre <= 1 ->
            "you have the bishop pair and the position is open -- the two bishops cover both " +
                "colours here, so keep lines open and don't trade one off"
        centre >= 3 ->
            "you have the bishop pair but the centre is closed -- open lines to bring the two " +
                "bishops to life"
        else -> "you have the bishop pair, so keep the position open where you can"
    }
    return listOf(PositionalFact("bishopPair", text, ours.sorted().map { name(it) }))
}

/** A bishop hemmed in by a wall of your own pawns fixed on its colour of square. */
fun badBishop(board: Board, us: Boolean): List<PositionalFact> {
    val facts = ArrayList<PositionalFact>()
    for (sq in board.pieces(BISHOP, us)) {
        val light = (squareFile(sq) + squareRank(sq)) % 2 == 1
        val blockers = board.pieces(PAWN, us).count {
            ((squareFile(it) + squareRank(it)) % 2 == 1) == light
        }
        val mobility = board.attacks(sq).count { board.colourAt(it) != us }
        if (blockers >= 4 && mobility <= 3) {
            facts.add(
                PositionalFact(
                    "badBishop",
                    "your bishop on ${name(sq)} is a bad bishop, blocked by your own " +
                        "pawns on its colour: trade it, push those pawns off its colour, or get it " +
                        "outside the pawn chain",
                    listOf(name(sq)),
                ),
            )
        }
    }
    return facts
}

/**
 * Past this move a minor piece on its home square is as likely to have gone back there as
 * never to have left, so it is a piece on its home square, not "not yet in the game".
 */
const val DEVELOPMENT_MOVES = 20

/** Minor pieces still on their home squares, and whether the king has castled. */
fun development(board: Board, us: Boolean): List<PositionalFact> {
    val home = homeMinors(us)
        .filter { board.pieceAt(it) != null && board.colourAt(it) == us }
        .filter { board.pieceTypeAt(it) == KNIGHT || board.pieceTypeAt(it) == BISHOP }
    val facts = ArrayList<PositionalFact>()
    if (home.isNotEmpty()) {
        val names = home.joinToString(", ") { name(it) }
        val count = if (home.size == 1) "a piece" else "${home.size} pieces"
        if (board.fullmoveNumber <= DEVELOPMENT_MOVES) {
            facts.add(
                PositionalFact(
                    "undeveloped",
                    "you still have $count at home ($names), not yet in the game",
                    home.map { name(it) },
                ),
            )
        } else {
            val where = if (home.size == 1) "its home square" else "their home squares"
            facts.add(
                PositionalFact(
                    "homeSquare",
                    "you have $count on $where ($names)",
                    home.map { name(it) },
                ),
            )
        }
    }
    return facts
}

/** The king still in the centre with pieces about, an idea worth naming past the opening. */
fun uncastledKing(board: Board, us: Boolean): List<PositionalFact> {
    val king = board.king(us) ?: return emptyList()
    val file = squareFile(king)
    val homeRank = if (us == WHITE) 0 else 7
    val inCentre = file in 2..5 && squareRank(king) == homeRank
    val hasRights = board.hasKingsideCastlingRights(us) || board.hasQueensideCastlingRights(us)
    if (inCentre && hasRights) {
        return listOf(
            PositionalFact(
                "uncastled",
                "your king is still in the centre on ${name(king)} and can still castle",
                listOf(name(king)),
            ),
        )
    }
    return emptyList()
}

/** A developed minor or rook of yours that is nearly stuck: worth naming to bring to life. */
fun worstPiece(board: Board, us: Boolean): List<PositionalFact> {
    val back = if (us == WHITE) setOf(0, 7) else setOf(56, 63)
    val home = homeMinors(us).toSet() + back
    var worst: Pair<Int, Int>? = null  // mobility to square
    for (pieceType in intArrayOf(KNIGHT, BISHOP, ROOK)) {
        for (sq in board.pieces(pieceType, us)) {
            if (sq in home) continue
            val mobility = board.attacks(sq).count { board.colourAt(it) != us }
            if (worst == null || mobility < worst!!.first) worst = mobility to sq
        }
    }
    val found = worst ?: return emptyList()
    if (found.first > 1) return emptyList()
    val sq = found.second
    return listOf(
        PositionalFact(
            "passivePiece",
            "your ${PIECE_NAMES[board.pieceTypeAt(sq)]} on ${name(sq)} is your least " +
                "active piece, with almost nowhere to go",
            listOf(name(sq)),
        ),
    )
}

private fun materialDiff(board: Board, us: Boolean): Int {
    var total = 0
    for (piece in board.pieceMap().values) {
        val value = POS_VALUE[piece.pieceType]
        total += if (piece.colour == us) value else -value
    }
    return total
}

/** Up a piece means simplify; down a piece means keep the pieces on. */
fun materialPlan(board: Board, us: Boolean): List<PositionalFact> {
    val diff = materialDiff(board, us)
    if (diff >= 3) {
        return listOf(
            PositionalFact(
                "convert",
                "you are up material, so trade pieces where you can and steer for the endgame",
            ),
        )
    }
    if (diff <= -3) {
        return listOf(
            PositionalFact(
                "complicate",
                "you are down material, so keep pieces on and look for complications",
            ),
        )
    }
    return emptyList()
}

/** Chebyshev distance from a square to the central d4-e5 box: 0 in the centre, 3 on the rim. */
private fun centerDistance(sq: Int): Int {
    val file = squareFile(sq)
    val rank = squareRank(sq)
    return max(min(abs(file - 3), abs(file - 4)), min(abs(rank - 3), abs(rank - 4)))
}

/** In the endgame, a King left on the rim: it should be marching toward the centre. */
fun kingActivity(board: Board, us: Boolean, phase: String = "middlegame"): List<PositionalFact> {
    if (phase != "endgame") return emptyList()
    val king = board.king(us) ?: return emptyList()
    if (centerDistance(king) < 3) return emptyList()
    return listOf(
        PositionalFact(
            "passiveKing",
            "your King on ${name(king)} is passive; in the endgame the King is a fighting " +
                "piece, so bring it toward the centre",
            listOf(name(king)),
        ),
    )
}

/** The static positional facts that hold in the position, most instructive first. */
fun features(board: Board, us: Boolean, phase: String = "middlegame"): List<PositionalFact> {
    val facts = ArrayList<PositionalFact>()
    if (phase != "endgame") {
        facts += development(board, us)
        facts += uncastledKing(board, us)
        facts += spacePlan(board, us)
        facts += pawnPointing(board, us)
    }
    facts += isolatedDPawn(board, us, phase)
    facts += hangingPawns(board, us)
    facts += pawnWeaknesses(board, us)
    facts += passedPawns(board, us)
    facts += pawnMajority(board, us)
    facts += outposts(board, us)
    facts += openFiles(board, us)
    facts += badBishop(board, us)
    facts += bishopPair(board, us)
    facts += kingActivity(board, us, phase)
    facts += squareOfThePawn(board, us, phase)
    facts += opposition(board, us)
    facts += keySquares(board, us)
    facts += lucena(board, us)
    facts += philidor(board, us)
    facts += materialPlan(board, us)
    facts += worstPiece(board, us)
    return facts
}

// --- the idea behind the best move ----------------------------------------------------

/** A pawn move that strikes at or occupies the centre. */
private fun isCentralBreak(board: Board, move: Move): Boolean {
    if (board.pieceTypeAt(move.from) != PAWN) return false
    val toFile = squareFile(move.to)
    val toRank = squareRank(move.to)
    val centralFile = toFile in 2..5
    val centralRank = toRank in 3..4
    return centralFile && (centralRank || board.isCapture(move))
}

/** If a rook moves onto a file with no friendly pawns, the file letter, else null. */
private fun rookToFileIsOpen(board: Board, move: Move, us: Boolean): String? {
    if (board.pieceTypeAt(move.from) != ROOK) return null
    val file = squareFile(move.to)
    if (squareFile(move.from) == file) return null  // already on this file
    val squares = (0 until 8).map { square(file, it) }
    if (squares.any { sq ->
            val piece = board.pieceAt(sq)
            piece != null && piece.pieceType == PAWN && board.colourAt(sq) == us
        }
    ) {
        return null
    }
    return FILES[file].toString()
}

/** A square in the central box (c3-f6) that no enemy pawn attacks now. */
private fun isCentralStrong(board: Board, sq: Int, us: Boolean): Boolean {
    val file = squareFile(sq)
    val rank = squareRank(sq)
    if (!(file in 2..5 && rank in 2..5)) return false
    return board.attackers(!us, sq).none { board.pieceTypeAt(it) == PAWN }
}

/** The wing a file belongs to: a to c the queenside, f to h the kingside, else null. */
private fun wing(file: Int): String? = if (file <= 2) "queenside" else if (file >= 5) "kingside" else null

/** The square's rank counted from your side of the board, 1 to 8. */
private fun relativeRank(sq: Int, us: Boolean): Int {
    val rank = squareRank(sq)
    return if (us == WHITE) rank + 1 else 8 - rank
}

/**
 * The wing being stormed if the move pushes a pawn at the enemy King, else null.
 *
 * A pawn advance -- not a capture -- toward an enemy King that has castled to a wing, on the
 * files around it: the classic way to prise open a castled position. Only with your own King
 * off that wing, and only a pawn that reaches its fourth rank: with both Kings behind the same
 * pawns, h3 or g6 is a flight square or a guard, not a storm (the master games' notes, where
 * "storming the pawns in front of their King" was said of h3 at move 15, 2026-09-24).
 */
internal fun pawnStormSide(board: Board, move: Move, us: Boolean): String? {
    if (board.pieceTypeAt(move.from) != PAWN || board.isCapture(move)) return null
    val enemyKing = board.king(!us) ?: return null
    val ownKing = board.king(us)
    val kingFile = squareFile(enemyKing)
    if (!(kingFile <= 2 || kingFile >= 5)) return null  // a King in the centre is no wing
    if (ownKing != null && wing(squareFile(ownKing)) == wing(kingFile)) return null  // storming your own King's cover
    if (abs(squareFile(move.to) - kingFile) > 2) return null
    if (relativeRank(move.to, us) < 4) return null
    val advancing = (squareRank(move.to) - squareRank(move.from)) * forward(us) > 0
    return if (advancing) (if (kingFile <= 2) "queenside" else "kingside") else null
}

/** The coaching line if the rook has landed behind a passed pawn on its file, else null. */
private fun rookBehindPasser(after: Board, rookSquare: Int, us: Boolean): String? {
    val file = squareFile(rookSquare)
    val rookRank = squareRank(rookSquare)
    for (rank in 0 until 8) {
        val sq = square(file, rank)
        val pawn = after.pieceAt(sq) ?: continue
        if (pawn.pieceType != PAWN) continue
        if (!isPassed(after, sq)) continue
        val promotesUp = pawn.colour == WHITE
        val behind = if (promotesUp) rookRank < rank else rookRank > rank
        if (!behind) continue
        val letter = FILES[file]
        return if (pawn.colour == us) {
            "put the rook behind your passed $letter-pawn to escort it forward"
        } else {
            "put the rook behind their passed $letter-pawn to hold it back"
        }
    }
    return null
}

/**
 * Whether your piece on `square` is in danger: the opponent can win it, taking with a legal
 * capture and coming out ahead once the recaptures are played out.
 *
 * Read by the rules ([Exchange.canBeWon]), not off `board.attackers()`, which called a knight
 * guarded by a pawn "in danger" from the enemy king beside it (the king was worth nothing,
 * and it cannot take a guarded piece anyway), and a knight attacked only by a piece pinned to
 * its own king.
 */
private fun pieceIsAttacked(board: Board, sq: Int, us: Boolean): Boolean {
    val piece = board.pieceAt(sq)
    return piece != null && piece.colour == us && Exchange.canBeWon(board, sq)
}

/**
 * Whether the move that led to `after` stops the opponent's threatened move.
 *
 * Stopped when the threat is no longer a legal move at all, or when it was a capture of one
 * of your pieces that won material and no longer does: played out by the rules, recaptures
 * included, it now nets nothing ([Exchange.exchange]). A guard alone is not enough -- a bishop
 * guarding a knight a pawn attacks leaves the pawn winning it, and a guard pinned to its own
 * king takes nothing back.
 */
private fun preventsThreat(board: Board, after: Board, threat: Move, us: Boolean): Boolean {
    if (!after.isLegal(threat)) return true
    val victim = after.pieceAt(threat.to)
    if (victim == null || victim.colour != us) return false
    val passed = board.copy(withStack = false)
    passed.push(Move.NULL)  // the position before the move, the opponent to move
    return passed.isLegal(threat) &&
        Exchange.exchange(passed, threat) > 0 &&
        Exchange.exchange(after, threat) <= 0
}

/**
 * The strategic point of the engine's best move, in a coach's words, or null. Priority order,
 * most decisive idea first. The move is the engine's; this only names what kind of idea it is.
 */
fun bestMoveIdea(
    board: Board,
    move: Move,
    us: Boolean,
    phase: String = "middlegame",
    worst: Int? = null,
    threat: Move? = null,
): PositionalFact? {
    val pieceType = board.pieceTypeAt(move.from)
    if (pieceType == 0) return null
    val after = board.copy(withStack = false)
    after.push(move)
    val to = name(move.to)

    if (board.isCastling(move)) {
        return PositionalFact("castle", "castle to tuck your king away and connect the rooks", listOf(to))
    }

    // A rescue needs the square the piece goes to to be safe as well: a knight that runs from
    // one pawn onto a square another pawn takes is not brought to safety.
    if (pieceIsAttacked(board, move.from, us) &&
        !board.isCapture(move) &&
        !pieceIsAttacked(after, move.to, us)
    ) {
        val pieceName = PIECE_NAMES[pieceType]
        return PositionalFact(
            "rescue", "your $pieceName was in danger; moving to $to brings it to safety", listOf(to),
        )
    }

    if (threat != null && !board.isCapture(move) && !after.isCheck() &&
        preventsThreat(board, after, threat, us)
    ) {
        return PositionalFact(
            "prophylaxis",
            "prophylaxis: this quietly stops your opponent's idea before you push your own plan",
            listOf(to),
        )
    }

    if (phase == "endgame" && pieceType == KING &&
        centerDistance(move.to) < centerDistance(move.from)
    ) {
        return PositionalFact(
            "activateKing",
            "march your King to $to, toward the centre; in the endgame it is a strong piece",
            listOf(to),
        )
    }

    // A minor piece leaving its home square is development while the game is young.
    if (phase != "endgame" && board.fullmoveNumber <= DEVELOPMENT_MOVES &&
        (pieceType == KNIGHT || pieceType == BISHOP)
    ) {
        if (move.from in homeMinors(us) && !board.isCapture(move)) {
            val pieceName = PIECE_NAMES[pieceType]
            return PositionalFact(
                "develop", "finish developing: the $pieceName comes into the game on $to", listOf(to),
            )
        }
    }

    // What a piece move is for, when it takes nothing: a capture's point is the exchange, and
    // "centralise the knight" was said of Nxe5 winning a pawn (the master games, 2026-09-24).
    val quiet = !board.isCapture(move)

    if (quiet && pieceType == KNIGHT && isOutpost(after, move.to, us)) {
        return PositionalFact(
            "outpostMove",
            "the knight lands on $to, an outpost no enemy pawn can chase it from",
            listOf(to),
        )
    }

    if (quiet && (pieceType == KNIGHT || pieceType == BISHOP) && isCentralStrong(after, move.to, us)) {
        val pieceName = PIECE_NAMES[pieceType]
        return PositionalFact(
            "centralise", "centralise the $pieceName on $to, a strong square in the middle", listOf(to),
        )
    }

    val fileLetter = if (quiet) rookToFileIsOpen(board, move, us) else null
    if (fileLetter != null) {
        return PositionalFact(
            "takeFile", "your rook takes the $fileLetter-file, where it has room to work", listOf(to),
        )
    }

    if (pieceType == ROOK) {
        val behind = rookBehindPasser(after, move.to, us)
        if (behind != null) return PositionalFact("rookBehindPasser", behind, listOf(to))
    }

    if (pieceType == PAWN && !board.isCapture(move)) {
        val enemy = !us
        if (after.attacks(move.to).any {
                after.pieceTypeAt(it) == PAWN && after.colourAt(it) == enemy
            }
        ) {
            val letter = FILES[squareFile(move.to)]
            return PositionalFact(
                "break", "the $letter-pawn break challenges their pawns and opens lines", listOf(to),
            )
        }
    }

    if (isCentralBreak(board, move)) {
        return PositionalFact("centre", "strike in the centre with the pawn to $to", listOf(to))
    }

    if (pieceType == PAWN && isPassed(board, move.from)) {
        val file = FILES[squareFile(move.from)]
        return PositionalFact("pushPasser", "push your passed $file-pawn toward $to", listOf(to))
    }

    if (quiet && worst != null && move.from == worst) {
        val pieceName = PIECE_NAMES[pieceType]
        return PositionalFact(
            "activate", "this brings your most passive piece, the $pieceName, to life on $to", listOf(to),
        )
    }

    val stormSide = pawnStormSide(board, move, us)
    if (stormSide != null) {
        return PositionalFact(
            "pawnStorm",
            "storm their King with the pawn to $to, prising open the $stormSide",
            listOf(to),
        )
    }

    // Space is a pawn that goes forward and stays: not a capture, and past its third rank (a3
    // and h3 make room for the King or keep a piece out; they take no space).
    if (phase != "opening" && pieceType == PAWN && quiet && relativeRank(move.to, us) >= 4) {
        val file = squareFile(move.to)
        if (file <= 1 || file >= 6) {
            val side = if (file <= 1) "queenside" else "kingside"
            return PositionalFact("space", "gain space on the $side with the pawn to $to", listOf(to))
        }
    }

    if (board.isCapture(move) && materialDiff(board, us) >= 3) {
        return PositionalFact(
            "simplify", "you are up material, so this trade simplifies toward the win", listOf(to),
        )
    }

    return null
}

/**
 * How instructive each static feature is, most first, so the sheet leads with the one that
 * teaches the most and never turns into a laundry list of every open file on the board.
 */
private val FEATURE_RANK = mapOf(
    "uncastled" to 0,
    "undeveloped" to 1,
    "lucena" to 1,
    "philidor" to 1,
    "runningPawn" to 1,
    "outrunPawn" to 1,
    "passiveKing" to 2,
    "keySquare" to 2,
    "keySquareGoal" to 2,
    "opposition" to 2,
    "oppositionAgainst" to 3,
    "iqpTarget" to 3,
    "hangingPawnsTarget" to 4,
    "targetPawn" to 5,
    "passedPawn" to 6,
    "iqp" to 7,
    "hangingPawns" to 8,
    "pawnMajority" to 9,
    "pawnPointing" to 9,
    "convert" to 9,
    "complicate" to 9,
    "spaceAdvantage" to 10,
    "spaceCramp" to 11,
    "homeSquare" to 12,
    "outpost" to 10,
    "badBishop" to 11,
    "openFile" to 12,
    "bishopPair" to 13,
    "weakPawn" to 14,
    "passivePiece" to 15,
    "halfOpenFile" to 16,
)

/** The few most instructive static features, one per kind, ranked. The coach's context. */
fun topFeatures(
    board: Board,
    us: Boolean,
    phase: String = "middlegame",
    limit: Int = 2,
): List<PositionalFact> {
    val seen = HashSet<String>()
    val unique = ArrayList<PositionalFact>()
    for (fact in features(board, us, phase)) {
        if (!seen.add(fact.kind)) continue
        unique.add(fact)
    }
    // Python's sort is stable, so ties keep the order `features` produced them in.
    return unique.sortedBy { FEATURE_RANK[it.kind] ?: 99 }.take(limit)
}

/** The square of the least-active minor or rook, or null; shared by features and the idea. */
fun worstPieceSquare(board: Board, us: Boolean): Int? {
    val facts = worstPiece(board, us)
    if (facts.isEmpty() || facts[0].squares.isEmpty()) return null
    return com.sidecoach.chess.parseSquare(facts[0].squares[0])
}

fun factsText(facts: List<PositionalFact>): List<String> = facts.map { it.text }
