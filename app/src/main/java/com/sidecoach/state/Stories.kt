package com.sidecoach.state

import com.sidecoach.chess.BB_SQUARES
import com.sidecoach.chess.Board
import com.sidecoach.chess.KING
import com.sidecoach.chess.KNIGHT
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.Piece
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.square
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareRank
import com.sidecoach.vision.Pieces.CellReading
import com.sidecoach.vision.Pieces.Kind

/*
 * Stories: the ways the game may have gone on past the record, kept side by side -- the
 * desktop's `state/stories.py`, line for line.
 *
 * docs/reconciler-model.md (desktop repo) is the design. The screen is drawn from a hidden
 * state: the game, the user's premove queue and the ply shown. A story is one such state past
 * the record -- the moves played since, the queue as it stands, the ply shown -- and what it
 * costs to believe. Every frame each story is extended by the cheapest events that make it
 * draw what the frame shows; the record takes only what every story agrees on. Nothing here
 * reads a colour: a story is checked against the pieces on the squares, and nothing else.
 */

/** What each premove edit adds to a story's prior ([Story.prior]). */
const val EDIT_PRIOR = 0.1

/**
 * What the opponent's reply the instant the user's move landed adds: common (a puzzle's reply,
 * a bot's, a premove of theirs), yet less likely than a frame that shows one thing.
 */
const val REPLY_PRIOR = 0.05

/** Positions the premove search may try for one way the queue stood ([Search.edits]). */
const val EDIT_BUDGET = 300

/**
 * `premove_picture`: the pieces a site that draws premoves shows while [queue] waits: the
 * position with every step applied in queue order, each lifting the piece off its square and
 * putting it down on the step's square over whatever stood there, of either side (chess.com's
 * board bundle, `v`, read 2026-10-05). A step whose square no longer holds a piece of the
 * queue's side is skipped; a promotion puts down the promoted piece; castling puts the king on
 * g/c and the rook on f/d.
 */
fun premovePicture(board: Board, queue: List<Move>): Board {
    if (queue.isEmpty()) return board
    val shown = board.copy(withStack = false)
    var side: Boolean? = null
    for (move in queue) {
        var piece = shown.pieceAt(move.from)
        if (side == null) side = board.pieceAt(move.from)?.colour
        if (piece == null || piece.colour != side) continue
        shown.removePieceAt(move.from)
        val rank = squareRank(move.from)
        val files = squareFile(move.to) - squareFile(move.from)
        if (piece.pieceType == KING && kotlin.math.abs(files) == 2) {
            val (rookFrom, rookTo) = if (files > 0) 7 to 5 else 0 to 3
            val rook = shown.pieceAt(square(rookFrom, rank))
            if (rook != null) {
                shown.removePieceAt(square(rookFrom, rank))
                shown.setPieceAt(square(rookTo, rank), rook)
            }
        }
        if (move.promotion != 0) piece = Piece(move.promotion, piece.colour)
        shown.setPieceAt(move.to, piece)
    }
    return shown
}

/**
 * `lone_moves`: the premoves a site allows the piece on [square]: its moves alone on an empty
 * board -- through anything, onto anything -- a pawn's forward diagonals always (it may premove
 * a capture of a piece not there yet), and a king's castling the position still allows
 * (chess.com's generator `_`; lichess's chessground premoves the same way).
 */
fun loneMoves(board: Board, square: Int): List<Move> {
    val piece = board.pieceAt(square) ?: return emptyList()
    val rights = if (piece.pieceType == KING) board.castlingRights else 0uL
    return LONE_MOVES.getOrPut(Triple(piece, square, rights)) { loneMovesOf(piece, square, rights) }
}

/** [loneMoves], once for each piece, square and castling rights: they depend on nothing else. */
private val LONE_MOVES = java.util.concurrent.ConcurrentHashMap<Triple<Piece, Int, ULong>, List<Move>>()

private fun loneMovesOf(piece: Piece, sq: Int, rights: ULong): List<Move> {
    val lone = Board(null)
    lone.turn = piece.colour
    lone.setPieceAt(sq, piece)
    if (piece.pieceType == PAWN) {
        val rank = squareRank(sq) + (if (piece.colour == WHITE) 1 else -1)
        for (file in listOf(squareFile(sq) - 1, squareFile(sq) + 1)) {
            if (file in 0..7 && rank in 0..7) {
                lone.setPieceAt(square(file, rank), Piece(KNIGHT, !piece.colour))
            }
        }
    }
    if (piece.pieceType == KING) {
        val home = if (piece.colour == WHITE) 0 else 7
        for (file in listOf(7, 0)) {
            val rook = square(file, home)
            if (sq == square(4, home) && (rights and BB_SQUARES[rook]) != 0uL) {
                lone.setPieceAt(rook, Piece(ROOK, piece.colour))
                lone.castlingRights = lone.castlingRights or BB_SQUARES[rook]
            }
        }
    }
    // A premoved promotion is to a queen, the sites' own choice.
    val out = ArrayList<Move>()
    lone.generatePseudoLegalMoves(BB_SQUARES[sq]) { move ->
        if (move.promotion == 0 || move.promotion == QUEEN) out.add(move)
    }
    return out
}

/** What a story did to the record before anything was played on it ([Story.head]). */
sealed class HeadOp {
    /** A new game from the start position. */
    object NewGame : HeadOp() { override fun toString() = "('newgame',)" }

    /** A piece of a position read from the screen was of another kind. */
    data class Fix(val square: Int, val kind: Int) : HeadOp()

    /** A takeback, seen as a move from an earlier position. */
    data class Back(val plies: Int) : HeadOp()

    /** The record's side to move read the other way. */
    object Flip : HeadOp() { override fun toString() = "('flip',)" }
}

/**
 * One way the game may have gone on past the record (`stories.Story`): [head] done to the
 * record first, then [moves] played; [queue] the user's premove queue as it stands; [view] the
 * ply of the record shown while the user browses; [cost] the pictures the story needs that no
 * frame showed; [prior] orders stories every frame showed as well.
 */
data class Story(
    val moves: List<Move> = emptyList(),
    val queue: List<Move> = emptyList(),
    val head: List<HeadOp> = emptyList(),
    val view: Int? = null,
    val cost: Double = 0.0,
    val prior: Double = 0.0,
    /** Whether the story holds an opponent's move no picture showed ([Search.unseenReplies]). */
    val hidden: Boolean = false,
    /** Whether it was charged for catching a piece in flight (`landed`). */
    val slide: Boolean = false,
    /** Its moves that were premoves played. Drawn at once, never slid. */
    val premoved: Set<Move> = emptySet(),
) {
    /** What the record must do to become this story's game, in order. */
    val ops: List<Any> get() = head + moves

    /** The position this story's game stands on. */
    fun board(record: Board): Board {
        var board = record
        for (op in head) {
            when (op) {
                is HeadOp.NewGame -> board = Board()
                is HeadOp.Fix -> {
                    board = board.copy()
                    val piece = board.pieceAt(op.square)!!
                    board.setPieceAt(op.square, Piece(op.kind, piece.colour))
                }
                is HeadOp.Back -> {
                    board = board.copy()
                    repeat(op.plies) { board.pop() }
                }
                is HeadOp.Flip -> {
                    board = board.copy(withStack = false)
                    board.turn = !board.turn
                    board.epSquare = null
                }
            }
        }
        if (moves.isNotEmpty()) {
            board = board.copy()
            for (move in moves) board.push(move)
        }
        return board
    }

    /** What the screen will go on from: the position, its side to move, the queue, the ply shown. */
    fun key(record: Board): List<Any?> {
        val board = board(record)
        return listOf(board.boardFen(), board.turn, board.castlingRights, queue, view)
    }

    /** This story told from a record that has done its first [done] ops. */
    fun rebased(done: Int): Story = copy(
        head = head.drop(done),
        moves = moves.drop(maxOf(0, done - head.size)),
    )
}

/**
 * What a frame reads, as bitboards, for comparing pictures fast (`stories.Evidence`): the
 * squares read empty, the squares read occupied, and for each piece read by its kind the
 * squares it is read on. Cells that carry no evidence are in none.
 */
class Evidence(val empty: ULong, val occupied: ULong, val kinds: List<Triple<Int, Boolean, ULong>>) {
    /** The squares the frame reads otherwise than [picture], as a bitboard. */
    fun wrong(picture: Board): ULong {
        var out = (picture.occupied and empty) or (occupied and picture.occupied.inv())
        for ((kind, colour, mask) in kinds) out = out or (mask and picture.piecesMask(kind, colour).inv())
        return out
    }

    companion object {
        fun of(obs: Observation, ignore: (Int, CellReading) -> Boolean): Evidence {
            var empty = 0uL
            var occupied = 0uL
            val kinds = LinkedHashMap<Pair<Int, Boolean>, ULong>()
            for ((sq, reading) in obs) {
                if (reading.kind == Kind.UNKNOWN || ignore(sq, reading)) continue
                val mask = BB_SQUARES[sq]
                if (reading.kind == Kind.EMPTY) {
                    empty = empty or mask
                    continue
                }
                occupied = occupied or mask
                val symbol = reading.symbol ?: continue
                val piece = Piece.fromSymbol(symbol)
                val k = piece.pieceType to piece.colour
                kinds[k] = (kinds[k] ?: 0uL) or mask
            }
            return Evidence(empty, occupied, kinds.map { (k, m) -> Triple(k.first, k.second, m) })
        }
    }
}

/** `unseen`: the pictures [events] made between two frames that neither frame showed. */
fun unseen(events: Int): Int = maxOf(0, events - 1)

/**
 * The cheapest stories that make one story draw the frame (`stories.Search`; see the desktop
 * for the full account of what a way costs and how the search goes).
 */
class Search(
    val record: Board,
    val user: Boolean,
    val obs: Observation,
    val ignore: (Int, CellReading) -> Boolean,
    val witness: (Board, Move) -> Boolean,
    val lifted: Boolean = false,
    /** The squares a piece in the user's hand may have been lifted from. */
    val liftable: Set<Int> = emptySet(),
    val maxPlies: Int = 4,
    val maxSteps: Int = 3,
    val budget: Int = 4000,
    val margin: Double = 1.0,
    /** Pictures this frame's events may need that neither frame showed. */
    val maxUnseen: Int = 1,
) {
    private val evidence = Evidence.of(obs, ignore)

    private fun unread(square: Int): Boolean {
        val reading = obs[square]
        return reading == null || reading.kind == Kind.UNKNOWN || ignore(square, reading)
    }

    /** The squares the frame reads otherwise than [picture]; during a drag, less one lifted piece. */
    fun wrong(picture: Board): MutableSet<Int> {
        val mask = evidence.wrong(picture)
        if (mask == 0uL) return LinkedHashSet()
        val out = LinkedHashSet<Int>()
        for (sq in 0 until 64) if ((mask and BB_SQUARES[sq]) != 0uL) out.add(sq)
        if (lifted) {
            for (sq in out.sorted()) {
                val piece = picture.pieceAt(sq)
                if (sq in liftable && piece != null && piece.colour == user && obs[sq]?.kind == Kind.EMPTY) {
                    out.remove(sq)
                    break
                }
            }
        }
        return out
    }

    private class Node(
        val board: Board,
        val queue: List<Move>,
        val moves: List<Move>,
        val events: Int,
        val plies: Int,
        val limit: Int,
        val edits: Boolean,
        val prior: Double,
        val blind: Boolean = false,
        val premoved: Set<Move> = emptySet(),
        val paired: Boolean = false,
    )

    /** The ways [story] goes on to draw the frame, none dearer than [ceiling]. */
    fun extend(story: Story, ceiling: Double = Double.POSITIVE_INFINITY): List<Story> {
        val found = ArrayList<Story>()
        var best = Double.POSITIVE_INFINITY
        val budgetLeft = intArrayOf(budget)

        fun allowed(): Double = minOf(ceiling, best + (if (best > story.cost) margin else 0.0))

        fun perfect(): Boolean = found.any { it.cost <= story.cost && it.prior <= story.prior }

        fun add(new: Story) {
            if (new.cost <= allowed() || new.cost < best) {
                found.add(new)
                best = minOf(best, new.cost)
            }
        }

        fun node(n: Node) {
            val cost = story.cost + unseen(n.events)
            if (cost > allowed() || unseen(n.events) > maxUnseen || budgetLeft[0] <= 0) return
            budgetLeft[0] -= 1
            val wrong = wrong(premovePicture(n.board, n.queue))
            if (wrong.isEmpty()) {
                add(story.copy(moves = n.moves, queue = n.queue, view = null, premoved = n.premoved, cost = cost, prior = n.prior))
                return
            }
            if (n.edits && n.board.turn != user && n.plies == n.limit) {
                // Once for each position, at the deepest level of the pass that reaches it.
                edits(story, n.board, n.queue, n.moves, n.events, n.prior, wrong, ::add, budgetLeft)
            }
            // A move changes four squares at most (castling), and a queue hides its own.
            val hidden = HashSet<Int>()
            for (step in n.queue) { hidden.add(step.from); hidden.add(step.to) }
            if (n.plies >= n.limit || (wrong - hidden).size > 4 * (n.limit - n.plies)) return
            val reach = wrong + hidden
            val children = ArrayList<Triple<Int, Int, Node>>()
            for (move in n.board.legalMoves()) {
                val touched = touchedSquares(n.board, move)
                // One move of a frame may live on cells nobody can read; a second is invented.
                val touchesReach = touched.any { it in reach }
                val unreadMove = !touchesReach && !n.blind && touched.all { unread(it) }
                if (!(touchesReach || unreadMove) || !witness(n.board, move)) continue
                if (n.plies >= 2 && move.from == n.moves[n.moves.size - 2].to &&
                    move.to == n.moves[n.moves.size - 2].from
                ) continue // there and straight back: no trace, no story
                val after = n.board.copy(withStack = false)
                after.push(move)
                var told = n.moves + move
                var rest = n.queue
                var played = n.premoved
                if (n.board.turn != user && n.queue.isNotEmpty()) {
                    val step = n.queue[0]
                    if (after.isLegal(step)) {
                        after.push(step)
                        told = told + step
                        rest = n.queue.drop(1)
                        played = played + step
                    } else {
                        rest = emptyList()
                    }
                }
                // A move the instant the other side's lands is drawn in one picture with it.
                val instant = n.plies > 0 && !n.paired
                val premove = instant && n.board.turn == user
                if (premove) played = played + move
                val left = wrong(premovePicture(after, rest)).size
                val child = Node(
                    after, rest, told,
                    n.events + (if (instant) 0 else 1), n.plies + 1, n.limit, n.edits,
                    n.prior + (if (premove) EDIT_PRIOR else if (instant) REPLY_PRIOR else 0.0),
                    n.blind || unreadMove, played, instant,
                )
                children.add(Triple(left, children.size, child))
            }
            // The moves that leave the least wrong first.
            for ((_, _, child) in children.sortedWith(compareBy({ it.first }, { it.second }))) node(child)
        }

        val start = story.board(record)
        for (limit in 0..maxPlies) {
            node(Node(start, story.queue, story.moves, 0, 0, limit, false, story.prior, false, story.premoved))
            // Past a fit that needs a premove, a story a picture longer is looked for in moves alone.
            if (found.isEmpty()) {
                node(Node(start, story.queue, story.moves, 0, 0, limit, true, story.prior, false, story.premoved))
            }
            // A picture more is two plies: the opponent's move and the premove it sets off.
            if (found.isNotEmpty() && (perfect() || limit > firstFit(found, story) + 1)) break
        }
        if (!story.hidden && story.queue.isNotEmpty() && start.turn != user &&
            wrong(premovePicture(start, story.queue)).isEmpty()
        ) {
            found.addAll(unseenReplies(story, start))
        }
        val cap = allowed()
        return found.filter { it.cost <= cap }
    }

    /** `_unseen_replies`: the opponent's moves a queue drawn over their squares hides whole. */
    private fun unseenReplies(story: Story, start: Board): List<Story> {
        val hidden = HashSet<Int>()
        for (step in story.queue) { hidden.add(step.from); hidden.add(step.to) }
        val out = ArrayList<Story>()
        for (move in start.legalMoves()) {
            if (!hidden.containsAll(touchedSquares(start, move))) continue
            val after = start.copy(withStack = false)
            after.push(move)
            var told = story.moves + move
            var rest = story.queue.drop(1)
            var premoved = story.premoved
            if (after.isLegal(story.queue[0])) {
                after.push(story.queue[0])
                told = told + story.queue[0]
                premoved = premoved + story.queue[0]
            } else {
                rest = emptyList()
            }
            if (wrong(premovePicture(after, rest)).isNotEmpty()) continue
            out.add(story.copy(moves = told, queue = rest, view = null, premoved = premoved, hidden = true))
        }
        return out
    }

    /** `_edits`: the queue cancelled, grown, or cancelled and begun again, so the picture fits. */
    internal fun edits(
        story: Story,
        board: Board,
        queue: List<Move>,
        moves: List<Move>,
        events: Int,
        prior: Double,
        @Suppress("UNUSED_PARAMETER") wrongNow: Set<Int>,
        add: (Story) -> Unit,
        budgetLeft: IntArray,
    ) {
        if (queue.isNotEmpty() && wrong(board).isEmpty() && unseen(events + 1) <= maxUnseen) {
            val cost = story.cost + unseen(events + 1)
            add(story.copy(moves = moves, queue = emptyList(), view = null, cost = cost, prior = prior + EDIT_PRIOR))
        }
        val free = obs.filter { (_, r) -> r.kind == Kind.EMPTY }.keys
        val bases = ArrayList<Pair<List<Move>, Int>>()
        bases.add(queue to events)
        if (queue.isNotEmpty()) bases.add(emptyList<Move>() to events + 1)
        for ((base, paid) in bases) {
            val start = premovePicture(board, base)
            val off = wrong(start)
            if (!off.all { editable(start, it) }) continue
            val movable = ArrayList<Int>()
            for (sq in off.sorted()) {
                val p = start.pieceAt(sq)
                if (p != null && p.colour == user) movable.add(sq)
            }
            if (base.isNotEmpty() && base.last().to !in movable) {
                val last = start.pieceAt(base.last().to)
                if (last != null && last.colour == user) movable.add(base.last().to)
            }
            // Where a piece moved by the queue now stands, the frame shows it.
            val targets = off.filter { obs[it]?.kind == Kind.PIECE }.toSet()
            if (targets.isEmpty()) continue
            // One base's premoves get a share of the budget, not all of it.
            val share = intArrayOf(minOf(budgetLeft[0], EDIT_BUDGET))

            fun grow(picture: Board, steps: List<Move>) {
                if (budgetLeft[0] <= 0 || share[0] <= 0) return
                budgetLeft[0] -= 1
                share[0] -= 1
                if (unseen(paid + steps.size) > maxUnseen) return
                if (steps.isNotEmpty() && steps.last().to in targets && wrong(picture).isEmpty()) {
                    val cost = story.cost + unseen(paid + steps.size)
                    val more = EDIT_PRIOR * (steps.size + (if (base != queue) 1 else 0))
                    add(story.copy(moves = moves, queue = base + steps, view = null, cost = cost, prior = prior + more))
                    return
                }
                val left = maxSteps - steps.size
                if (left == 0) return
                for (sq in if (steps.isNotEmpty()) listOf(steps.last().to) else movable) {
                    val walkedSet = walked(base + steps, sq, moves)
                    for (step in loneMoves(picture, sq)) {
                        val to = step.to
                        if (to in walkedSet) continue // back over its own way
                        if (to !in targets) {
                            if (left == 1 || (to !in off && to !in free)) continue
                            if (left == 2 && targets.none { it in loneReach(picture, sq, to) }) continue
                        }
                        val after = picture.copy(withStack = false)
                        applyStep(after, step)
                        grow(after, steps + step)
                    }
                }
            }
            grow(start, emptyList())
        }
    }

    /** `_editable`: whether a queue edit could put [square] right. */
    private fun editable(picture: Board, square: Int): Boolean {
        val piece = picture.pieceAt(square)
        if (piece != null && piece.colour == user) return true
        val reading = obs[square]!!
        if (reading.kind == Kind.EMPTY) return piece != null
        val symbol = reading.symbol ?: return true
        return Piece.fromSymbol(symbol).colour == user
    }
}

/** `_walked`: the squares the piece queued to [square] has stood on, its own way. */
internal fun walked(queue: List<Move>, square: Int, moves: List<Move> = emptyList()): Set<Int> {
    val walked = LinkedHashSet<Int>()
    walked.add(square)
    for (step in queue.asReversed()) if (step.to in walked) walked.add(step.from)
    val origin = origin(queue, square)
    for (move in moves.asReversed()) {
        if (move.to == origin) {
            walked.add(move.from)
            break
        }
    }
    return walked
}

/** `_origin`: where the piece queued to [square] stood before its queue. */
internal fun origin(queue: List<Move>, square: Int): Int {
    var origin = square
    for (step in queue.asReversed()) if (step.to == origin) origin = step.from
    return origin
}

/** `_apply_step`: one queued step drawn on [board], over whatever is there. */
internal fun applyStep(board: Board, move: Move) {
    var piece = board.pieceAt(move.from) ?: return
    board.removePieceAt(move.from)
    val rank = squareRank(move.from)
    val files = squareFile(move.to) - squareFile(move.from)
    if (piece.pieceType == KING && kotlin.math.abs(files) == 2) {
        val (rookFrom, rookTo) = if (files > 0) 7 to 5 else 0 to 3
        val rook = board.pieceAt(square(rookFrom, rank))
        if (rook != null) {
            board.removePieceAt(square(rookFrom, rank))
            board.setPieceAt(square(rookTo, rank), rook)
        }
    }
    if (move.promotion != 0) piece = Piece(move.promotion, piece.colour)
    board.setPieceAt(move.to, piece)
}

/** `_lone_reach`: where the piece on [square] could step next from [to], alone on the board. */
internal fun loneReach(picture: Board, square: Int, to: Int): Set<Int> {
    val after = picture.copy(withStack = false)
    applyStep(after, Move(square, to))
    return loneMoves(after, to).map { it.to }.toSet()
}

/** `_first_fit`: the fewest moves past [story] of the stories found. */
internal fun firstFit(found: List<Story>, story: Story): Int = found.minOf { it.moves.size - story.moves.size }
