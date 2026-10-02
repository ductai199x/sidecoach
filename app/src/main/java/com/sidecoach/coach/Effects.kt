package com.sidecoach.coach

import com.sidecoach.analysis.Exchange
import com.sidecoach.chess.BB_EMPTY
import com.sidecoach.chess.BB_SQUARES
import com.sidecoach.chess.Board
import com.sidecoach.chess.FILE_NAMES
import com.sidecoach.chess.KING
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.Piece
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.chess.between
import com.sidecoach.chess.square
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareName

/**
 * What a move does on the board, computed, so the coach has a true mechanism to say. A port
 * of `sidecoach/coach/effects.py`.
 *
 * The coach's commonest invention was the reason a move was good or bad: "Qf5 hits the
 * knight", "b3 locks the queenside" -- claims the board contradicts. The messages gave it the
 * better move and a plan word, never what the move does. This module states that, from the
 * board alone. Every phrase is checked against the position the move leads to, so a coach
 * that repeats it is right.
 *
 * Attack and defence are read off legal moves ([Exchange]), not off the lines the pieces
 * cover: a knight pinned to its king attacks and defends nothing, a king takes nothing the
 * other side still guards. An audit on 2026-09-23 found "attacks the knight on e4" said of a
 * pinned knight and of a king that could not take, "defends the knight, which was attacked"
 * when a pawn still won it, and "forks" two knights that pawns defended. The king alone is
 * attacked by the lines: a pinned piece still gives check.
 */

private val EFFECT_VALUE = Exchange.VALUE

private fun nameAt(board: Board, sq: Int): String {
    val piece = board.pieceAt(sq)
    val word = if (piece != null) COACH_PIECE_NAMES[piece.pieceType] else "piece"
    return "the $word on ${squareName(sq)}"
}

/** The pieces on [squares], the first two: "the rook on g1 and the pawn on e5". */
private fun namesAt(board: Board, squares: List<Int>): String =
    squares.take(2).joinToString(" and ") { nameAt(board, it) }

/**
 * `colour`'s pieces that attack the enemy piece on `square`: those that could take it by a legal
 * move (were it `colour`'s move); on the king, those that give it check.
 *
 * A check `colour` is in is set aside: it limits the move that answers it, not what stands
 * attacked. Read through legal moves, a side in check attacked nothing but the checker, and
 * every attack it had came back after the answer as new ("uncovers an attack on the bishop by
 * the knight" after taking the checking queen). So then a piece attacks what it could take once
 * the check is gone: a pinned one only along its line, the king only onto a square nothing
 * guards once it has stepped off its own.
 */
private fun hitters(board: Board, colour: Boolean, square: Int): List<Int> {
    if (board.pieceTypeAt(square) == KING) return board.attackers(colour, square).sorted()
    if (board.turn != colour || !board.isCheck()) return Exchange.capturers(board, colour, square)
    val out = ArrayList<Int>()
    for (a in board.attackers(colour, square)) {
        if (board.pieceTypeAt(a) == KING) {
            val stepped = board.copy(withStack = false)
            stepped.removePieceAt(a)
            if (!stepped.isAttackedBy(!colour, square)) out.add(a)
        } else if (board.pinMask(colour, a) and BB_SQUARES[square] != BB_EMPTY) {
            out.add(a)
        }
    }
    return out.sorted()
}

/**
 * The enemy pieces `colour` attacks, as squares. The lines are read first, the legal moves only
 * where a piece bears on the square: nothing takes where nothing bears.
 */
private fun attackedBy(board: Board, colour: Boolean): Set<Int> {
    val out = HashSet<Int>()
    for (sq in 0 until 64) {
        val piece = board.pieceAt(sq) ?: continue
        if (piece.colour != colour &&
            board.isAttackedBy(colour, sq) &&
            hitters(board, colour, sq).isNotEmpty()
        ) {
            out.add(sq)
        }
    }
    return out
}

/** `colour`'s pieces (not the king, not pawns) with no legal defender. */
private fun undefended(board: Board, colour: Boolean): Set<Int> {
    val out = HashSet<Int>()
    for (sq in 0 until 64) {
        val piece = board.pieceAt(sq) ?: continue
        if (piece.colour == colour &&
            piece.pieceType != KING && piece.pieceType != PAWN &&
            Exchange.defenders(board, colour, sq).isEmpty()
        ) {
            out.add(sq)
        }
    }
    return out
}

/** Enemy pieces pinned to their king. */
private fun pinnedEnemies(board: Board, colour: Boolean): Set<Int> {
    val out = HashSet<Int>()
    for (sq in 0 until 64) {
        val piece = board.pieceAt(sq) ?: continue
        if (piece.colour != colour && board.isPinned(!colour, sq)) out.add(sq)
    }
    return out
}

/**
 * The move's effects on the board, most telling first, at most `limit` phrases.
 *
 * Computed on the position before and after the move; nothing here is inferred. The static
 * hazards -- a piece put or left where it is attacked and undefended -- are named only with
 * `hazards`: for a move the engine judged sound the tactics resolve them, and naming them
 * would mislead.
 */
fun describeEffects(board: Board, move: Move, limit: Int = 3, hazards: Boolean = true): List<String> {
    if (!board.isLegal(move)) return emptyList()
    val us = board.turn
    val them = !us
    val after = board.copy(withStack = false)
    after.push(move)
    val mover = board.pieceAt(move.from)
    val moverWord = if (mover != null) COACH_PIECE_NAMES[mover.pieceType] else "piece"
    val out = ArrayList<String>()

    if (after.isCheckmate()) return listOf("checkmate")
    if (board.isCastling(move)) out.add("castles, the king out of the centre")
    if (board.isCapture(move)) {
        val taken = board.pieceAt(move.to)
        if (board.isEnPassant(move)) {
            out.add("takes the pawn en passant")
        } else if (taken != null) {
            out.add("takes the ${COACH_PIECE_NAMES[taken.pieceType]} on ${squareName(move.to)}")
        }
    }
    if (after.isCheck()) out.add("gives check")

    // What the moved piece attacks from its new square, and what the move uncovers for the
    // rest of the army: enemy pieces attacked after that were not attacked before. Attacked
    // means takeable by a legal move, as though it were the mover's turn again.
    val beforeAttacked = attackedBy(board, us)
    val afterAttacked = attackedBy(after, us)

    fun valueAt(sq: Int): Int {
        val piece = after.pieceAt(sq)
        return if (piece != null) EFFECT_VALUE[piece.pieceType] else 0
    }

    // By value, then by square: the desktop sorts the same way, because the operands are
    // sets and equal-valued pieces came out in CPython's hash order.
    val new = (afterAttacked - beforeAttacked)
        .sortedWith(compareByDescending<Int> { valueAt(it) }.thenBy { it })
    val byMover = new.filter { move.to in hitters(after, us, it) }
    val byOthers = new.filter { it !in byMover }
    val worth = byMover.filter { valueAt(it) >= 3 }
    // A fork wins something: two targets it can win -- the king, or a piece that can be taken
    // for profit (worth more than the forker, or not defended enough) -- from a square where the
    // forker cannot be taken for profit. A queen touching two knights that pawns guard forks
    // nothing; it attacks one.
    val targets = worth.filter { valueAt(it) == Exchange.VALUE[KING] || Exchange.canBeWon(after, it) }
    if (targets.size >= 2 && !Exchange.canBeWon(after, move.to)) {
        out.add("forks " + targets.take(2).joinToString(" and ") { nameAt(after, it) })
    } else if (byMover.isNotEmpty()) {
        out.add("attacks " + nameAt(after, byMover[0]))
    }
    if (byOthers.isNotEmpty()) {
        val s = byOthers[0]
        val attacker = hitters(after, us, s).firstOrNull { it != move.to }
        if (attacker != null) {
            // Uncovered only where the moved piece stood between them; otherwise the move let a
            // pinned piece loose ("Ke7 uncovers an attack ... by the knight on d6", of a knight
            // unpinned from the d-file: the v2 sample check, 2026-09-26).
            if (between(attacker, s) and BB_SQUARES[move.from] != BB_EMPTY) {
                out.add("uncovers an attack on ${nameAt(after, s)} by ${nameAt(after, attacker)}")
            } else {
                out.add("frees ${nameAt(after, attacker)} to attack ${nameAt(after, s)}")
            }
        }
    }

    val pins = pinnedEnemies(after, us) - pinnedEnemies(board, us)
    for (s in pins.sorted()) {
        // A pinned piece that takes its pinner for profit is not held: its only captures run
        // along the line, onto the pinner (Re1 against a rook on e7, with e1 unguarded).
        if (after.legalMoves().any { m -> m.from == s && after.isCapture(m) && Exchange.exchange(after, m) > 0 }) {
            continue
        }
        out.add("pins ${nameAt(after, s)} to the king")
        break
    }

    // The moved piece: was it attacked, and is it now? Attacked is a legal capture onto its
    // square, defended a legal recapture: a pinned piece neither threatens nor guards it.
    // The attacker is named: "gets the knight out of attack" alone, the model guessed it, and
    // guessed the pawn that had just moved rather than the rook on g1 (Luna on Tai's puzzle,
    // 2026-09-29).
    if (mover != null && mover.pieceType != KING) {
        val wasAttacked = Exchange.capturers(board, them, move.from)
        val isAttacked = Exchange.capturers(after, them, move.to)
        if (wasAttacked.isNotEmpty() && isAttacked.isEmpty()) {
            out.add("gets the $moverWord out of attack from ${namesAt(board, wasAttacked)}")
        } else if (hazards && isAttacked.isNotEmpty() && Exchange.defenders(after, us, move.to).isEmpty()) {
            out.add("puts the $moverWord where it is undefended and attacked by " + namesAt(after, isAttacked))
        }
    }

    // Own pieces left undefended by the move, and attacked ones newly defended -- only when the
    // defence saves the piece: still attacked, and no longer to be won by the exchange (a
    // bishop guarding a knight that a pawn attacks saves nothing).
    val beforeLoose = undefended(board, us)
    val afterLoose = undefended(after, us)
    val left = if (hazards) (afterLoose - beforeLoose).filter { it != move.to } else emptyList()
    for (s in left.sortedWith(compareByDescending<Int> { valueAt(it) }.thenBy { it })) {
        val hitters = Exchange.capturers(after, them, s)
        val tail = if (hitters.isNotEmpty()) ", and it is attacked by ${namesAt(after, hitters)}" else ""
        out.add("leaves ${nameAt(after, s)} undefended$tail")
        break
    }
    val covered = (beforeLoose - afterLoose).filter {
        it != move.from && Exchange.capturers(after, them, it).isNotEmpty() && !Exchange.canBeWon(after, it)
    }
    for (s in covered.sortedWith(compareByDescending<Int> { valueAt(it) }.thenBy { it })) {
        val by = namesAt(after, Exchange.capturers(after, them, s))
        if (move.to in Exchange.defenders(after, us, s)) {
            out.add("defends ${nameAt(after, s)}, which is attacked by $by")
        } else {
            // Saved by a defence the move uncovers, or by what it takes: not "defends",
            // which names the moved piece (the facts audit, 2026-10-01).
            out.add("makes ${nameAt(after, s)} safe from $by")
        }
        break
    }

    // A rook or queen onto a file with no pawn of its own on it, the file as the move leaves
    // it: a capture of their pawn opens it ("takes the half-open file" of Rxd6, 2026-10-01).
    if (mover != null && (mover.pieceType == ROOK || mover.pieceType == QUEEN)) {
        val fileIndex = squareFile(move.to)
        val ownPawn = (0 until 8).any { after.pieceAt(square(fileIndex, it)) == Piece(PAWN, us) }
        val theirPawn = (0 until 8).any { after.pieceAt(square(fileIndex, it)) == Piece(PAWN, them) }
        if (squareFile(move.from) != fileIndex && !ownPawn) {
            val kind = if (!theirPawn) "open" else "half-open"
            out.add("takes the $kind ${FILE_NAMES[fileIndex]}-file")
        }
    }

    return out.take(limit)
}

/**
 * The effects joined for a message, or an empty string when the move does nothing the board
 * can name (a quiet developing move).
 */
fun effectsText(board: Board, move: Move, limit: Int = 3, hazards: Boolean = true): String =
    describeEffects(board, move, limit, hazards).joinToString("; ")

/**
 * A line move by move with what each move does, computed from the board: "Rh4+ (checks the
 * king on h5), Kg5 (only a king move gets out of check), ...". A reply to a check says whether
 * it was the only legal move, whether the king had to move, or that it was one of several ways
 * out. [moves] in UCI or SAN. `effects.line_notes`.
 *
 * Asked to explain a line, the model gave its moves purposes the board denies: "the check
 * forced the King away" where a block was legal, "Rad8 keeps the b- and e-files open" where
 * both held pawns (the bench's gold labels, 2026-09-25: 4 of the 5 false claims were in
 * explanations of lines). The notes are what a line may be explained by. With [player], each
 * move says whose it is ("the player's Qxa7", "the opponent's c3"): without it the model gave
 * the player the opponent's moves and the reverse (the test audit, 2026-09-26: "advancing b4"
 * of their pawn, "your queen checks" of theirs).
 */
fun lineNotes(board: Board, moves: List<String>, limit: Int = 8, player: Boolean? = null): String {
    val current = board.copy(withStack = false)
    val out = ArrayList<String>()
    for (text in moves.take(limit)) {
        // UCI when it is a legal move here (python-chess: the null move never is), else SAN.
        val uci = Move.fromUciOrNull(text)
        val move = if (uci != null && !uci.isNull && current.isLegal(uci)) {
            uci
        } else {
            current.parseSan(text) ?: break
        }
        val notes = ArrayList<String>()
        if (current.isCheck()) {
            val legal = current.legalMoves()
            if (legal.size == 1) {
                notes.add("the only legal move")
            } else if (legal.all { current.pieceTypeAt(it.from) == KING }) {
                // The king moves named: "only a king move gets out of check" was read as "Kh8 is
                // the only king move" where there were four (the test audit, 2026-09-26).
                val sans = legal.map { current.san(it) }.sorted()
                val listed = sans.dropLast(1).joinToString(", ") + " or ${sans.last()}"
                notes.add("the king must move: $listed")
            } else {
                notes.add("one of ${legal.size} ways out of check")
            }
        }
        val does = effectsText(current, move, hazards = false)
        if (does.isNotEmpty()) notes.add(does)
        var san = current.san(move)
        if (player != null) {
            san = (if (current.turn == player) "the player's " else "the opponent's ") + san
        }
        out.add(if (notes.isNotEmpty()) "$san (${notes.joinToString("; ")})" else san)
        current.push(move)
    }
    return out.joinToString(", ")
}

// What a move does, said as it is played: "Qxa7, taking the pawn on a7 and attacking the rook on
// b8". The effects' own verbs, in the -ing form.
private val GERUND = mapOf(
    "attacks" to "attacking",
    "takes" to "taking",
    "gives" to "giving",
    "gets" to "getting",
    "uncovers" to "uncovering",
    "frees" to "freeing",
    "pins" to "pinning",
    "defends" to "defending",
    "forks" to "forking",
    "skewers" to "skewering",
    "castles" to "castling",
    "stops" to "stopping",
    "threatens" to "threatening",
    "covers" to "covering",
    "blocks" to "blocking",
)

private fun gerund(phrase: String): String {
    val space = phrase.indexOf(' ')
    val verb = if (space < 0) phrase else phrase.substring(0, space)
    val rest = if (space < 0) "" else phrase.substring(space + 1)
    val said = GERUND[verb]
    return if (said != null) "$said $rest".trim() else phrase
}

/**
 * A line as the coach says it, each move with what it does, computed: "Qxa7, taking the pawn on
 * a7 and attacking the rook on b8; c3, getting the pawn out of attack; Rf2". The template's
 * answer to "show me the line": asked to explain a line, the model gave its moves to the wrong
 * side or purposes the board denies in a third of its answers (the locked test set,
 * 2026-09-26). [moves] in UCI or SAN. `effects.line_spoken`.
 */
fun lineSpoken(board: Board, moves: List<String>, limit: Int = 6): String {
    val current = board.copy(withStack = false)
    val parts = ArrayList<String>()
    for (text in moves.take(limit)) {
        // UCI when it is a legal move here (python-chess: the null move never is), else SAN.
        val uci = Move.fromUciOrNull(text)
        val move = if (uci != null && !uci.isNull && current.isLegal(uci)) {
            uci
        } else {
            current.parseSan(text) ?: break
        }
        val san = current.san(move)
        val effects = describeEffects(current, move, limit = 2, hazards = false)
        val after = current.copy(withStack = false)
        after.push(move)
        if (after.isCheckmate()) {
            parts.add("$san, checkmate")
            break
        }
        if (effects.isNotEmpty()) {
            parts.add("$san, " + effects.joinToString(" and ") { gerund(it) })
        } else {
            parts.add(san)
        }
        current.push(move)
    }
    return parts.joinToString("; ")
}
