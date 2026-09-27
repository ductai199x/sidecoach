package com.sidecoach.state

import com.sidecoach.chess.BB_EMPTY
import com.sidecoach.chess.BB_SQUARES
import com.sidecoach.chess.BLACK
import com.sidecoach.chess.Bitboard
import com.sidecoach.chess.Board
import com.sidecoach.chess.KING
import com.sidecoach.chess.Move
import com.sidecoach.chess.Piece
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.chess.BISHOP
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.between
import com.sidecoach.chess.scanReversed
import com.sidecoach.chess.square
import com.sidecoach.chess.squareDistance
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareName
import com.sidecoach.chess.squareRank
import com.sidecoach.events.Event
import com.sidecoach.events.GameEnded
import com.sidecoach.events.GameStarted
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.PositionChanged
import com.sidecoach.events.Resync
import com.sidecoach.events.Takeback
import com.sidecoach.events.Unexplained
import com.sidecoach.vision.Pieces
import com.sidecoach.vision.Pieces.CellReading
import com.sidecoach.vision.Pieces.Kind
import com.sidecoach.vision.Pieces.Side
import com.sidecoach.vision.Squares.squareAt
import com.sidecoach.vision.pyRound
import java.util.UUID

/**
 * Per-frame cell readings reconciled into a legal game: a port of
 * `sidecoach/state/reconcile.py`, line for line.
 *
 * The core idea from DESIGN.md section 5.3: detect moves, not positions. Given the current
 * legal position, a frame is explained by the legal move whose touched squares match what
 * the cells now show, or, when moves were missed, by a short path of moves. Everything else
 * is held: a piece lifted mid-drag empties one square and matches no move, a promotion
 * dialog hides cells, an arrow makes a cell unreadable.
 *
 * A move is committed only after it has explained `settleFrames` consecutive frames in
 * which every square it touches holds a settled piece or is empty, none of them is under
 * something drawn, and no square carries the hover outline of a drag. Highlight colours are
 * never consulted.
 *
 * Browsing history is normal; a takeback is recorded only when a different move is played
 * from the earlier position, which is the only way the two can be told apart on the board.
 * Cells that keep disagreeing go stale and are ignored, so one unreadable cell cannot stall
 * the game; resync fires only when several stay unexplained for a long time.
 */

/**
 * The readings of one frame, indexed row * 8 + column in screen order. Mutable because
 * the pipeline replaces a badged king's reading in place, as `pipeline._badged_kings` does.
 */
typealias Readings = MutableList<CellReading>

typealias Observation = LinkedHashMap<Int, CellReading>

/** One way a premove ghost could have been queued. */
private data class QueueWay(
    val origin: Int,
    val visited: Set<Int>,
    val passesOver: Boolean = false,
)

/** What a pending explanation is: the same tuple python keys `_pending` with. */
private data class PendingKey(val kind: String, val extra: Any?, val moves: List<Move>)

private val STORY_KEYS = setOf("moves", "moves-cautious")

/** The fields a stale count is keyed by; a changed reading resets it. */
private data class StaleKey(
    val kind: Kind,
    val symbol: String?,
    val colour: Side?,
    val highlighted: Boolean,
)

/** `reconcile.highlighted_squares`: the squares the site marks as its last move (or a
 * selection): tinted, not red. */
fun highlightedSquares(obs: Observation): Set<Int> =
    obs.filter { (_, r) -> r.highlighted && r.tint != "red" }.keys

/**
 * `reconcile._came_from`: whether the piece on [to] could have come from the empty [start] in
 * one move -- its own geometry on the board as it stands, the pawn's pushes and captures,
 * castling's king.
 */
private fun cameFrom(board: Board, start: Int, to: Int): Boolean {
    val piece = board.pieceAt(to) ?: return false
    if (board.pieceAt(start) != null) return false
    if (piece.pieceType == PAWN) {
        val step = if (piece.colour == WHITE) 8 else -8
        if (start == to - step) return true
        if (start == to - 2 * step && board.pieceAt(to - step) == null) {
            return squareRank(to) == (if (piece.colour == WHITE) 3 else 4)
        }
        return (start == to - step - 1 || start == to - step + 1) && squareDistance(start, to) == 1
    }
    if (piece.pieceType == KING && squareRank(start) == squareRank(to) &&
        kotlin.math.abs(squareFile(start) - squareFile(to)) == 2
    ) {
        return true  // castling
    }
    val back = board.copy(withStack = false)
    back.removePieceAt(to)
    back.setPieceAt(start, piece)
    return back.attacksMask(start) and BB_SQUARES[to] != 0uL
}

/**
 * `reconcile.LastMoveSeen`: what the site's last-move highlight tells a position read from
 * pixels -- who moved, and the en-passant square when the move was a pawn's two-square advance.
 */
data class LastMoveSeen(val mover: Boolean, val epSquare: Int? = null)

/**
 * `reconcile.last_move_seen`: the last move as the site highlights it -- an empty marked square
 * and a marked square whose piece could have come from it. Every such pair must name the same
 * mover, or nothing is known. The highlight is the history a position read from pixels lacks:
 * the side to move (a puzzle adopted with it open could not tell the solver's move and the
 * instant reply from the same two moves in the other order, and the record stayed empty), and
 * a pawn's two-square advance taken in passing (after c2-c4 the coach did not see bxc3 e.p.;
 * Tai's chess.com puzzles, 2026-09-24).
 */
fun lastMoveSeen(board: Board, highlighted: Set<Int>): LastMoveSeen? {
    val movers = HashSet<Boolean>()
    var ep: Int? = null
    for (to in highlighted) {
        val piece = board.pieceAt(to) ?: continue
        for (start in highlighted) {
            if (start == to || !cameFrom(board, start, to)) continue
            movers.add(piece.colour)
            val step = if (piece.colour == WHITE) 8 else -8
            if (piece.pieceType == PAWN && start == to - 2 * step) ep = to - step
        }
    }
    if (movers.size != 1) return null
    return LastMoveSeen(movers.first(), ep)
}

private fun turnValid(board: Board, turn: Boolean): Boolean {
    board.turn = turn
    return board.isValid()
}

/**
 * A bare placement made into a position: castling rights assumed conservatively from the
 * king and rook home squares, and the turn from legality. Returns the board and whether
 * the turn is still unknown (both colours legal to move), or null if neither is. When
 * both could move, `prefer` is the turn the board shows meanwhile: the side at the
 * bottom, which on a puzzle is the solver and to move.
 */
internal fun adoptPlacement(
    placement: Board,
    prefer: Boolean = WHITE,
    highlighted: Set<Int> = emptySet(),
): Pair<Board, Boolean>? {
    val board = placement.copy(withStack = false)
    var rights: Bitboard = 0uL
    for ((colour, rank) in listOf(WHITE to 0, BLACK to 7)) {
        val king = Piece(KING, colour)
        val rook = Piece(ROOK, colour)
        if (board.pieceAt(square(4, rank)) == king) {
            if (board.pieceAt(square(7, rank)) == rook) rights = rights or BB_SQUARES[square(7, rank)]
            if (board.pieceAt(square(0, rank)) == rook) rights = rights or BB_SQUARES[square(0, rank)]
        }
    }
    board.castlingRights = rights
    val validTurns = listOf(WHITE, BLACK).filter { turnValid(board, it) }
    if (validTurns.isEmpty()) return null
    // The site's last-move highlight: the other side is to move, and may take a pawn that has
    // just advanced two squares in passing (`lastMoveSeen`).
    val seen = lastMoveSeen(board, highlighted)
    if (seen != null && !seen.mover in validTurns) {
        board.turn = !seen.mover
        board.epSquare = seen.epSquare
        return board to false
    }
    board.turn = if (prefer in validTurns) prefer else validTurns[0]
    return board to (validTurns.size > 1)
}

/**
 * How long a premove queue, once its red tint is seen, is taken to be about: a queue executes one
 * step for each of the opponent's replies, within a second or two.
 */
const val QUEUE_MEMORY_S = 3.0

/**
 * `reconcile.in_flight`: whether a story has a piece stop partway along a straight line and then
 * go on the same way: a to x, and later x to y by that piece, x strictly between a and y. Seen on
 * one frame, that is one move in flight, never two. Every site slides a moved piece in a straight
 * line from its square to the next, and a frame that catches it near the centre of a square on
 * its way reads as a piece standing there; the next frame, with the piece further on (or
 * landed), then explains as the same piece moving on. Chess.com's phone app slides every move and
 * cannot be told not to, speeding up as it goes, with a trail (Tai, 2026-09-25): "I played Rxc1+
 * but it misunderstood as Rc7". Geometry alone, for every piece, both sides, any speed and any
 * capture rate; the highlights are not read (they can be turned off, and section 5.3 of DESIGN.md
 * keeps the reconciler free of them). A piece that returns to a square it left, or another piece
 * moving next, is not in flight. The caller decides when the shape means flight
 * ([Reconciler.flight]): a pawn pushed twice is two moves when each stood still to be seen.
 */
fun inFlight(moves: List<Move>): Boolean {
    val landed = HashMap<Int, Move>()
    for (move in moves) {
        val came = landed.remove(move.from)
        if (came != null && between(came.from, move.to) and BB_SQUARES[move.from] != BB_EMPTY) {
            return true
        }
        landed[move.to] = move
    }
    return false
}

fun touchedSquares(board: Board, move: Move): MutableSet<Int> {
    val touched = mutableSetOf(move.from, move.to)
    if (board.isCastling(move)) {
        val rank = squareRank(move.from)
        if (squareFile(move.to) > squareFile(move.from)) {
            touched.add(square(7, rank)); touched.add(square(5, rank))
        } else {
            touched.add(square(0, rank)); touched.add(square(3, rank))
        }
    }
    if (board.isEnPassant(move)) {
        touched.add(square(squareFile(move.to), squareRank(move.from)))
    }
    return touched
}

/**
 * Whether a reading is consistent with a square's content. A piece reading without a symbol
 * only asserts that the square is occupied; its colour is not trusted, since a capture ring
 * during a drag can flip the measured luminance and a wrong colour would let the move search
 * fabricate captures.
 */
fun cellAgrees(reading: CellReading, piece: Piece?): Boolean {
    if (reading.kind == Kind.UNKNOWN) return true
    if (reading.kind == Kind.EMPTY) return piece == null
    if (piece == null) return false
    if (reading.symbol != null) return piece.symbol() == reading.symbol
    return true
}

/** A reading a badge or a ring leaves: an unsettled piece, or a piece of no known kind. */
fun isJunk(reading: CellReading): Boolean =
    reading.kind == Kind.PIECE && (!reading.settled || reading.symbol == null)

fun exact(reading: CellReading): Boolean =
    reading.kind == Kind.EMPTY || (reading.kind == Kind.PIECE && reading.symbol != null)

fun matches(board: Board, obs: Observation, squares: Set<Int>): Boolean =
    squares.all { sq -> sq !in obs || cellAgrees(obs.getValue(sq), board.pieceAt(sq)) }

/** Legal moves whose touched squares cover `diff` and agree with the readings. */
fun singleExplanations(board: Board, obs: Observation, diff: Set<Int>): List<List<Move>> {
    val found = ArrayList<List<Move>>()
    for (move in board.legalMoves()) {
        val touched = touchedSquares(board, move)
        if (!touched.containsAll(diff)) continue
        val after = board.copy(withStack = false)
        after.push(move)
        if (matches(after, obs, touched)) found.add(listOf(move))
    }
    return found
}

class Reconciler(
    var bottom: Side,
    var userSide: Side = Side.WHITE,
    // The tunables are the desktop dataclass's fields, in its order and with its defaults, so
    // its tests (which build one with `settle_frames=1`, or set `stale_after = 2`) port as they
    // stand (`tools/make_reconcile_fixtures.py`).
    var settleFrames: Int = 2,
    var maxPlies: Int = 4,
    /**
     * Positions the missed-move search may expand per depth. Only moves that touch an
     * unexplained square are expanded, so a four-ply gap in a middlegame costs a few hundred;
     * counting every legal move looked at, as before, spent the budget on the thirty rejected
     * for each one expanded and left four-ply gaps unbridged.
     */
    var searchBudget: Int = 4000,
    var historyDepth: Int = 40,
    var staleAfter: Int = 5,
    var resyncCells: Int = 3,
    var resyncAfter: Int = 150,
    /**
     * Position resync: adopt a fully readable, stable board that no move story explains --
     * held this many frames; or, when at least [resyncJumpCells] cells disagree (a new
     * position, not a move), at least [resyncMinFrames] spanning [resyncHoldS]: on a phone
     * reading seven frames a second, ten frames were a second and a half, and more when the
     * frames were slow ones (Tai's puzzles on the Pixel, 2026-09-23). A move is a handful of
     * cells, and one whose squares wear the site's badges waits for the stale retry to explain
     * it: adopted early, the mating move under the result badges went unrecorded (the desktop's
     * stress seed 839).
     */
    var resyncHoldFrames: Int = 10,
    var resyncMinFrames: Int = 3,
    var resyncHoldS: Double = 0.4,
    var resyncJumpCells: Int = 12,
    var maxStaleSubsetCells: Int = 4,
    /**
     * A pending story survives frames a drag blocks, up to this many, and may be extended by
     * the moves that follow it, up to this many plies in all.
     */
    var pendingMaxAge: Int = 40,
    var maxChainPlies: Int = 8,
    var maxQueueSteps: Int = 6,
    var resyncMaxSoft: Int = 2,
    /**
     * Last stable reading of every cell and the frame it was seen in, remembered for this
     * many frames under a cover.
     */
    var memoryFrames: Int = 30,
    /**
     * The board unseen for this many ticks (lost, covered, minimised: the pipeline reports each
     * such tick) means anything remembered about covered cells may have changed.
     */
    var memoryGapTicks: Int = 5,
    /**
     * Fewer readable cells than this is no evidence of anything: a theme the recogniser was not
     * built for, or a page-wide overlay.
     */
    var minReadableCells: Int = 32,
) {

    var board: Board = Board()
    var gameId: String = UUID.randomUUID().toString().replace("-", "").take(8)
    var viewing: Int? = null

    /** What this game is (`GameStarted.kind`): a game from the start, or a position adopted. */
    var kind: String = "game"

    private var hoverSeen = false
    private var pending: Pair<PendingKey, Int>? = null
    private val stale = HashMap<Int, Pair<StaleKey, Int>>()
    private var unexplainedFrames = 0

    // `internal` for the desktop's tests, which set `_started`, `_finished` and `_base_fen`
    // on a reconciler as they stand (`tools/make_reconcile_fixtures.py`).
    internal var startedFlag = false
    internal var finishedFlag = false
    private var resultValue: String? = null
    private var pendingAge = 0
    internal var baseFenValue: String = STARTING_FEN
    private var turnUnknownFlag = false
    /** The placement the resync would adopt, how many frames have shown it, and since when. */
    private var resyncCandidate: Triple<String, Int, Double>? = null
    private var unseen = 0

    /**
     * Last stable reading of every cell and the frame it was seen in. `internal` for the
     * desktop's tests of the memory, which look at `_last_seen`.
     */
    internal val lastSeen = HashMap<Int, Pair<CellReading, Int>>()
    private val previous = HashMap<Int, StaleKey>()
    private var drawn = HashSet<Int>()
    private var denied: Observation = Observation()
    private var frameIndex = 0
    private var blockedFlag = false
    private var lastObs: Observation = Observation()
    private var tsValue = 0.0
    private var lastCommitTs: Double? = null

    /** When a premove queue's red tint was last seen ([flight]). */
    private var redTs: Double? = null

    val started: Boolean get() = startedFlag
    val finished: Boolean get() = finishedFlag

    /** The finished game's result ("1-0", "0-1", "1/2-1/2", "*"), else null. */
    val result: String? get() = if (finishedFlag) resultValue else null

    /** The position the move stack starts from: the start, or the last resync. */
    val baseFen: String get() = baseFenValue

    val turnUnknown: Boolean get() = turnUnknownFlag

    /** Whether a game is under way: moves were committed, or a position was adopted. */
    val inProgress: Boolean get() = board.moveStack.isNotEmpty() || baseFenValue != STARTING_FEN

    // Observation helpers

    /**
     * Readings as evidence, with covered cells remembered for a short while.
     *
     * Nothing changes under a dialog, so a cell a confirmed overlay has covered keeps its
     * last stable reading for [memoryFrames]; the explanation found while the board was
     * still uncovered then keeps settling under the dialog. Only an exact, settled reading
     * repeated in two consecutive frames is remembered, and a cell seen clearly with
     * something else on it forgets its memory at once.
     *
     * `internal` for the desktop's tests, which call `_observe` on a frame of their own.
     */
    internal fun observe(readings: Readings): Observation {
        frameIndex += 1
        val out = Observation()
        drawn = HashSet()
        // python's `if k` drops square 0; ported as it stands, a king on a1 included.
        val kings = listOfNotNull(board.king(WHITE), board.king(BLACK)).filter { it != 0 }
        for (r in 0 until 8) {
            for (c in 0 until 8) {
                var reading = readings[r * 8 + c]
                val sq = squareAt(r, c, bottom == Side.WHITE)
                if (reading.drawn) drawn.add(sq)
                if (reading.kind == Kind.UNKNOWN && reading.drawn && !reading.covered &&
                    kings.any { squareDistance(sq, it) <= 1 }
                ) {
                    // The check glow is saturated colour the set never draws with, so on
                    // some themes it reads as a drawing over the king's square. A drawing on
                    // or beside a king is the king under its glow: a piece of no known kind,
                    // which the decorated-king rules take as the king, and which blocks
                    // nothing.
                    reading = reading.copy(
                        kind = Kind.PIECE, symbol = null, colour = null, settled = false,
                    )
                }
                if (reading.kind == Kind.UNKNOWN) {
                    previous.remove(sq)
                    if (reading.covered) {
                        val remembered = lastSeen[sq]
                        if (remembered != null && frameIndex - remembered.second <= memoryFrames) {
                            out[sq] = remembered.first
                        }
                    }
                    continue
                }
                out[sq] = reading
                if (!(exact(reading) && reading.settled && !reading.hover)) {
                    previous.remove(sq)
                    continue
                }
                val key = StaleKey(reading.kind, reading.symbol, reading.colour, false)
                val remembered = lastSeen[sq]
                if (remembered != null) {
                    val was = remembered.first
                    if (StaleKey(was.kind, was.symbol, was.colour, false) != key) lastSeen.remove(sq)
                }
                if (previous[sq] == key) lastSeen[sq] = reading to frameIndex
                previous[sq] = key
            }
        }
        return out
    }

    fun disagreements(obs: Observation): MutableSet<Int> {
        val out = HashSet<Int>()
        for ((sq, reading) in obs) if (!cellAgrees(reading, board.pieceAt(sq))) out.add(sq)
        return out
    }

    // Lifecycle

    fun start(): MutableList<Event> {
        board = Board()
        gameId = UUID.randomUUID().toString().replace("-", "").take(8)
        pending = null
        stale.clear()
        unexplainedFrames = 0
        startedFlag = true
        finishedFlag = false
        resultValue = null
        viewing = null
        baseFenValue = STARTING_FEN
        turnUnknownFlag = false
        resyncCandidate = null
        lastSeen.clear()
        previous.clear()
        pendingAge = 0
        lastCommitTs = tsValue
        kind = "game"
        return mutableListOf(
            GameStarted(gameId = gameId, userSide = userSide, startFen = board.fen()),
            PositionChanged(gameId = gameId, fen = board.fen(), ply = 0),
        )
    }

    /**
     * The board was flipped or re-themed before the first move: the same game, seen anew.
     * Everything remembered per square is dropped, since the squares were mapped with the
     * old orientation.
     */
    fun reorient(bottom: Side): MutableList<Event> {
        this.bottom = bottom
        userSide = bottom
        pending = null
        pendingAge = 0
        stale.clear()
        lastSeen.clear()
        previous.clear()
        viewing = null
        unexplainedFrames = 0
        if (!startedFlag) return mutableListOf()
        return mutableListOf(
            GameStarted(gameId = gameId, userSide = userSide, startFen = board.fen(), kind = kind),
        )
    }

    // Explanations

    /**
     * Sequences of up to [maxPlies] legal moves explaining `diff` from `base`.
     *
     * `wild` cells are stale junk a badge left, ignored by the caller: a move may touch them
     * instead of an unexplained cell, and they need no exact reading, so that the mating move
     * still settles when the checkmate badge has made the king's square unreadable. One move
     * of a story may live on wild cells alone; a second is a move nobody saw invented to fill
     * a turn, and a piece that goes and comes straight back is the same invention in one
     * piece. An ignored cell that reads exactly ([denied]) may stay as wrong as it was, but a
     * story that changes it must agree with it.
     *
     * This is the catch-up for missed moves. Every move must touch a still unexplained
     * square, every touched square must be read exactly, and the final position must agree
     * with every readable cell. The shortest paths are kept; transpositions among them lead
     * to one position and are accepted, while different final positions are a genuine
     * ambiguity and nothing is committed.
     */
    private fun pathExplanations(
        obs: Observation,
        diff: Set<Int>,
        base: Board? = null,
        wild: Set<Int> = emptySet(),
    ): List<List<Move>> {
        // A move touches at most four squares, so more disagreements than that times the
        // depth cannot be a missed-move story.
        if (diff.size + wild.size < 3 || diff.size > 4 * maxPlies) return emptyList()

        val start = base ?: board

        // Iterative deepening: the shortest story is searched exhaustively before any longer
        // one is attempted, so a budget spent on deep branches can never hide a simple
        // two-ply explanation.
        for (depth in 2..maxPlies) {
            val solutions = ArrayList<List<Move>>()
            var budget = searchBudget

            fun search(
                board: Board,
                remaining: Set<Int>,
                path: ArrayList<Move>,
                unseenMoves: Int,
                decorated: Set<Int>,
            ) {
                val pliesLeft = depth - path.size
                if (pliesLeft <= 0 || solutions.size >= 64 || budget <= 0) return
                if (remaining.size > 4 * pliesLeft) return
                for (move in board.legalMoves()) {
                    if (path.size >= 2 &&
                        move.from == path[path.size - 2].to && move.to == path[path.size - 2].from
                    ) {
                        continue  // there and straight back: no trace, no story
                    }
                    val touched = touchedSquares(board, move)
                    val seen = touched.any { it in remaining }
                    if (!seen && (unseenMoves >= 1 || touched.none { it in wild })) {
                        continue  // a second move told through wild cells alone
                    }
                    val lenient = touched.filter { sq ->
                        sq !in wild && !(sq in obs && exact(obs.getValue(sq))) &&
                            kingLands(board, move, sq, obs)
                    }.toSet()
                    if (!touched.all { sq ->
                            sq in wild || (sq in obs && exact(obs.getValue(sq))) || sq in lenient
                        }
                    ) {
                        continue
                    }
                    budget -= 1
                    if (budget < 0) return
                    val after = board.copy(withStack = false)
                    after.push(move)
                    path.add(move)
                    val still = remaining - touched
                    val used = decorated + lenient
                    if (still.isEmpty()) {
                        if (matches(after, obs, obs.keys - wild) &&
                            notDenied(after, start) && checkedKings(after, used)
                        ) {
                            solutions.add(ArrayList(path))
                        }
                    } else {
                        search(after, still, path, unseenMoves + (if (seen) 0 else 1), used)
                    }
                    path.removeAt(path.size - 1)
                }
            }

            search(start, HashSet(diff), ArrayList(), 0, emptySet())
            if (solutions.isEmpty()) continue
            val finals = HashSet<String>()
            for (path in solutions) {
                val board = start.copy(withStack = false)
                for (move in path) board.push(move)
                finals.add(board.boardFen() + " " + (if (board.turn) "w" else "b"))
            }
            if (finals.size > 1) return solutions
            return listOf(solutions[0])
        }
        return emptyList()
    }

    /**
     * Whether an unsettled reading of a piece of no known kind is a king under the site's
     * decoration: on the king's own square, or one step from it. The colour, when read, must
     * be the king's.
     */
    private fun decoratedKing(square: Int, reading: CellReading): Boolean {
        if (reading.symbol != null) return false
        for (colour in booleanArrayOf(WHITE, BLACK)) {
            if (reading.colour != null &&
                reading.colour != (if (colour == WHITE) Side.WHITE else Side.BLACK)
            ) {
                continue
            }
            val king = board.king(colour)
            if (king != null && squareDistance(king, square) <= 1) return true
        }
        return false
    }

    /**
     * Whether `move` is a king's move onto `square`, where the reading is a piece of the
     * king's colour whose kind could not be read: the king under its decoration. A site
     * decorates a king's square when the king is in check, so a story that uses this must end
     * with that king in check ([checkedKings]).
     */
    private fun kingLands(board: Board, move: Move, square: Int, obs: Observation): Boolean {
        if (move.to != square || square !in obs) return false
        val piece = board.pieceAt(move.from) ?: return false
        if (piece.pieceType != KING) return false
        val reading = obs.getValue(square)
        if (reading.kind != Kind.PIECE || reading.symbol != null) return false
        return reading.colour == null ||
            reading.colour == (if (piece.colour == WHITE) Side.WHITE else Side.BLACK)
    }

    /**
     * Whether every decorated landing square of a story holds a king in check at its end.
     */
    private fun checkedKings(after: Board, decorated: Set<Int>): Boolean {
        if (decorated.isEmpty()) return true
        return after.isCheck() && after.king(after.turn) in decorated
    }

    /** Whether `after` agrees with every ignored exact reading on a cell it changed. */
    private fun notDenied(after: Board, start: Board): Boolean =
        denied.all { (sq, reading) ->
            after.pieceAt(sq) == start.pieceAt(sq) || cellAgrees(reading, after.pieceAt(sq))
        }

    /**
     * Index of the earlier position of this game the frame shows, if any. The base position
     * is checked as well however long the game is: a click on the first move of the list
     * shows it, and it must never pass for a new game while this one is running.
     */
    private fun historyIndex(obs: Observation, diff: Set<Int>): Int? {
        val earlier = board.copy()
        val undone = HashSet<Int>()
        val plies = board.moveStack.size
        for (back in 1..minOf(plies, historyDepth)) {
            val move = earlier.pop()
            undone.addAll(touchedSquares(earlier, move))
            if (undone.containsAll(diff) && matches(earlier, obs, obs.keys)) return plies - back
        }
        if (plies > historyDepth) {
            val base = Board(baseFenValue)
            if (matches(base, obs, obs.keys)) return 0
        }
        return null
    }

    private fun boardAt(index: Int): Board {
        val earlier = board.copy()
        while (earlier.moveStack.size > index) earlier.pop()
        return earlier
    }

    /**
     * Geometry evidence that `moves` are complete, independent of any colour.
     *
     * A settled piece is drawn exactly where its exemplar from the start position sits in the
     * square; a dragged piece follows the cursor and an animating piece is between squares.
     * So every square the moves touch must hold a settled piece or be empty, the square a
     * move lands on must be read at all, and no square on the board may carry a hover
     * outline. A square with an arrow over it is no witness either, while the arrow, which is
     * no drag, holds nothing else back. Highlights are not consulted.
     */
    private fun highlightOk(
        obs: Observation,
        moves: List<Move>,
        base: Board? = null,
        wild: Set<Int> = emptySet(),
    ): Boolean {
        if (hoverSeen || obs.values.any { it.hover }) return false
        val touched = HashSet<Int>()
        val decorated = HashSet<Int>()
        val board = (base ?: this.board).copy(withStack = false)
        for (move in moves) {
            if (!board.isLegal(move)) return false
            if (move.to !in wild && move.to !in obs) {
                // The piece must be seen where it landed.
                return false
            }
            if (kingLands(board, move, move.to, obs)) decorated.add(move.to)
            touched.addAll(touchedSquares(board, move))
            board.push(move)
        }
        if (!checkedKings(board, decorated)) return false
        for (sq in touched) {
            if (sq in wild || sq in decorated) continue
            if (sq in drawn) return false
            val reading = obs[sq]
            if (reading != null && reading.kind == Kind.PIECE && !reading.settled) return false
        }
        return true
    }

    // Main entry

    /** A tick on which the board could not be read at all; see [memoryGapTicks]. */
    fun boardUnseen() {
        unseen += 1
    }

    fun update(readings: Readings, ts: Double): MutableList<Event> {
        tsValue = ts
        val events = if (!startedFlag) start() else mutableListOf()
        if (unseen >= memoryGapTicks) {
            lastSeen.clear()
            previous.clear()
        }
        unseen = 0
        // A hover outline anywhere means a drag is in progress, even on a cell whose content
        // could not be read; it must be seen before unknown cells are dropped.
        hoverSeen = readings.any { it.hover }
        // A drag in progress blocks every explanation without contradicting any. A piece of
        // no known kind, off its centre, on a king's square or the one next to it is not a
        // drag but the king under what the site draws on it, and blocks nothing.
        blockedFlag = hoverSeen || (0 until 64).any { i ->
            val reading = readings[i]
            reading.kind == Kind.PIECE && !reading.settled &&
                !decoratedKing(squareAt(i / 8, i % 8, bottom == Side.WHITE), reading)
        }
        val seen = observe(readings)
        // A queue leaves its piece's real square red and empty; a checked king's red glow stands
        // on a piece, and says nothing of a queue.
        if (seen.values.any { it.tint == "red" && it.kind == Kind.EMPTY }) redTs = ts
        var obs = unqueue(seen)  // premove ghosts put back
        if (obs.size < minReadableCells) {
            holdPending(true)
            return events
        }
        var diff: Set<Int> = disagreements(obs)
        advanceStale(diff, obs)
        val ghosts = ghosts(diff, obs)
        if (ghosts.isNotEmpty()) {
            // An arrow drawn over an empty square reads as a piece of no known kind, off
            // centre, for as long as it is drawn. Once such a reading has gone stale it is
            // taken as unknown.
            val trimmed = Observation()
            for ((sq, r) in obs) if (sq !in ghosts) trimmed[sq] = r
            obs = trimmed
            diff = diff - ghosts
        }
        lastObs = obs

        if (diff.isEmpty()) {
            // The frame shows the latest position: live again if the user was browsing.
            pending = null
            viewing = null
            unexplainedFrames = 0
            return events
        }

        val blocked = blockedFlag
        // Shortest story first: one move, then the pending story confirmed or extended, then
        // a premove queue, then the opponent's move on its own, and only then a path of
        // missed moves, which could otherwise explain a queue display or a drag with a longer
        // story than what happened.
        explain(obs, diff, obs, allowPaths = false)?.let { events.addAll(it); return events }
        extendPending(obs, diff)?.let { events.addAll(it); return events }
        explainPremoveQueue(obs, diff)?.let { events.addAll(it); return events }
        explainOpponentMove(obs)?.let { events.addAll(it); return events }
        if (hoverSeen) {
            // A hover outline refuses every story below; the searches would only burn time.
            holdPending(blocked)
            return events
        }
        explainPaths(obs, diff, obs)?.let { events.addAll(it); return events }

        val staleCells = if (viewing == null) {
            diff.filter { stale.getValue(it).second > staleAfter }.toSet()
        } else {
            emptySet()
        }
        if (staleCells.isNotEmpty()) {
            // A badge or a ring on one or two cells must not stall the game, but a move whose
            // commit it blocked has cells just as stale as the badge itself. So the stale
            // cells are ignored one at a time, junk-looking readings first, then in pairs;
            // the rest of the disagreement must still be explained by legal moves. Cells a
            // badge has spoiled without contradicting the position cannot be ignored as
            // stale, yet they can be neither read exactly nor trusted as settled; in this
            // retry they are wildcards alongside the ignored cells.
            val soft = obs.filter { (_, r) ->
                r.kind == Kind.PIECE && (r.symbol == null || !r.settled)
            }.keys.toSet()
            val subsets = staleSubsets(staleCells, obs)
            // Soft wildcards only go with junk cells: ignoring a clean reading as well would
            // let the search invent a story through the cells nobody can read.
            val attempts = ArrayList<Pair<Set<Int>, Set<Int>>>()
            for (sub in subsets) if (sub.all { isJunk(obs.getValue(it)) }) attempts.add(sub to (sub + soft))
            for (sub in subsets) attempts.add(sub to sub)
            val tried = HashSet<Pair<Set<Int>, Set<Int>>>()
            for ((ignored, wild) in attempts) {
                if (!tried.add(ignored to wild)) continue
                val fresh = diff - ignored
                if ((fresh - wild).isEmpty()) {
                    // A story must rest on a disagreement read cleanly -- a square read empty
                    // or a settled piece of a known kind. When every cell left to explain is
                    // a wildcard, nothing on the board says a move was played: a board locked
                    // 0.28 of a cell off (it had slid on after the lock was taken) read every
                    // piece unsettled, the top of the king on d2 read as a piece on d3, and
                    // the king's move there was told, d2 and d3 both wildcards (the desktop's
                    // geometry sweep, 2026-09-27, seed 97).
                    continue
                }
                val liveObs = Observation()
                for ((sq, r) in obs) if (sq !in ignored) liveObs[sq] = r
                denied = Observation()
                for (sq in ignored) {
                    val r = obs.getValue(sq)
                    if (exact(r) && !isJunk(r)) denied[sq] = r
                }
                val explained = try {
                    explain(liveObs, fresh, obs, allowPaths = true, cautious = true, wild = wild)
                } finally {
                    denied = Observation()
                }
                if (explained != null) { events.addAll(explained); return events }
            }
            if (staleCells == diff && staleCells.size < resyncCells) {
                // Only stale garbage differs from the latest position, unless a drag is what
                // kept a real move from committing while its cells turned stale.
                holdPending(blocked)
                viewing = null
                unexplainedFrames += 1
                return events
            }
        }

        holdPending(blocked)

        // New game: the start position is back on the board.
        if (inProgress && obs.size >= 48 && matches(Board(), obs, obs.keys)) {
            if (!finishedFlag) events.add(ended("*", "new game started"))
            events.addAll(start())
            return events
        }

        val adopted = maybePositionResync(obs, diff)
        if (adopted.isNotEmpty()) { events.addAll(adopted); return events }
        events.addAll(maybeResync(diff))
        return events
    }

    /**
     * The picture with queued premoves undone: each of the user's pieces drawn at the end of
     * a red queue is put back on the empty square it will leave from.
     *
     * Chess.com draws a premoved piece at its destination and tints the queue's squares red,
     * so the screen is not a position. Undoing the queue gives the true position, which the
     * ordinary explanations then read -- the real move that slipped by, the opponent's reply,
     * the queued step the moment it executes.
     *
     * Where a ghost's piece really stands is read against the position the pending story
     * leads to when there is one: on the square that position has the piece, when that square
     * is red and empty, else one move of the piece away from it, on a tinted, empty square.
     * From its square the piece must reach the ghost in queued steps, each landing on a red
     * square; the chains of all the ghosts together must account for every red, empty square,
     * each once; chains built square by square are preferred to ones that pass a red square
     * over, and red origins to ones in the last-move colour; and there must then be exactly
     * one reading. Nothing is undone when the picture cannot be read for certain.
     */
    private fun unqueue(obs: Observation): Observation {
        val redEmpty = obs.filter { (_, r) -> r.tint == "red" && r.kind == Kind.EMPTY }.keys.toSet()
        if (redEmpty.isEmpty()) return obs
        val story = pendingStory()
        val anchor = story?.second ?: board
        val ghosts = ArrayList<Pair<Int, CellReading>>()
        for ((sq, reading) in obs) {
            if (reading.tint != "red" || reading.kind != Kind.PIECE || reading.symbol == null) continue
            if (reading.colour != userSide) continue
            val here = anchor.pieceAt(sq)
            if (here != null && here.symbol() == reading.symbol) continue  // really there
            ghosts.add(sq to reading)
        }
        if (ghosts.isEmpty()) return obs
        val options = ArrayList<List<QueueWay>>()
        for ((target, reading) in ghosts) {
            val ways = queueOrigins(reading.symbol!!, target, redEmpty, obs, anchor)
            if (ways.isEmpty()) return obs
            options.add(ways)
        }
        // Every red square is one queue's origin or one queue's landing, never two.
        val ranked = HashMap<Pair<Int, Int>, MutableSet<List<Int>>>()
        for (choice in cartesian(options)) {
            val origins = choice.map { it.origin }
            if (origins.toSet().size != origins.size) continue
            val covered = ArrayList<Int>()
            for (o in origins) if (o in redEmpty) covered.add(o)
            for (way in choice) covered.addAll(way.visited)
            if (covered.size != covered.toSet().size || covered.toSet() != redEmpty) continue
            val rank = choice.count { it.passesOver } to origins.count { it !in redEmpty }
            ranked.getOrPut(rank) { HashSet() }.add(origins)
        }
        if (ranked.isEmpty()) return obs
        val bestRank = ranked.keys.minWithOrNull(
            compareBy({ it.first }, { it.second }),
        )!!
        val best = ranked.getValue(bestRank)
        if (best.size != 1) return obs
        val chosen = best.first()
        val out = Observation(obs)
        for (i in ghosts.indices) {
            val (target, reading) = ghosts[i]
            out[chosen[i]] = reading.copy(highlighted = false, tint = null, settled = true)
            out.remove(target)
        }
        for (sq in redEmpty) if (sq !in chosen) out.remove(sq)  // an intermediate square: unread
        return out
    }

    private fun <T> cartesian(options: List<List<T>>): List<List<T>> {
        var out = listOf<List<T>>(emptyList())
        for (choices in options) {
            val next = ArrayList<List<T>>(out.size * choices.size)
            for (prefix in out) for (choice in choices) next.add(prefix + choice)
            out = next
        }
        return out
    }

    /**
     * Every way a ghost of `symbol` at `target` could have been queued: from a square
     * `anchor` has such a piece on, or one move of it away.
     */
    private fun queueOrigins(
        symbol: String,
        target: Int,
        red: Set<Int>,
        obs: Observation,
        anchor: Board,
    ): List<QueueWay> {
        val piece = Piece.fromSymbol(symbol)
        val out = ArrayList<QueueWay>()
        val scratch = anchor.copy(withStack = false)
        scratch.turn = piece.colour
        for ((stand, here) in anchor.pieceMap()) {
            if (here != piece) continue
            val reading = obs[stand]
            if (reading == null || reading.kind != Kind.EMPTY) continue  // still shown there
            if (stand in red) {
                out.addAll(queueWays(symbol, stand, target, red, obs))
                continue
            }
            // The first queued step has executed: the piece is one move on, on a tinted,
            // empty square the queue continues from.
            val moves = ArrayList<Move>()
            scratch.generatePseudoLegalMoves(BB_SQUARES[stand]) { moves.add(it) }
            for (move in moves) {
                val to = move.to
                if (to == target) continue
                val landed = obs[to]
                if (landed == null || landed.kind != Kind.EMPTY || landed.tint == null) continue
                out.addAll(queueWays(symbol, to, target, red, obs))
            }
        }
        return out
    }

    /**
     * The ways a piece could be queued from `origin` to `target`: every step lands on a red
     * square, past the pieces the picture shows; a slider either stops at the first red square
     * on its ray, as a queue built square by square does, or passes one over.
     */
    private fun queueWays(
        symbol: String,
        origin: Int,
        target: Int,
        red: Set<Int>,
        obs: Observation,
    ): List<QueueWay> {
        val piece = Piece.fromSymbol(symbol)
        val scratch = Board(null)
        for ((sq, reading) in obs) {
            if (reading.kind == Kind.PIECE && reading.symbol != null && reading.tint != "red") {
                scratch.setPieceAt(sq, Piece.fromSymbol(reading.symbol))
            }
        }
        scratch.turn = piece.colour
        val landings = red + target
        val slider = piece.pieceType == QUEEN || piece.pieceType == ROOK || piece.pieceType == BISHOP
        val ways = ArrayList<QueueWay>()
        for (stopAtRed in if (slider) booleanArrayOf(true, false) else booleanArrayOf(false)) {
            var frontier: Set<Int> = setOf(origin)
            val parent = HashMap<Int, Int?>()
            parent[origin] = null
            var found: Set<Int>? = null
            for (step in 0 until maxQueueSteps) {
                val nextFrontier = HashSet<Int>()
                // In square order: a set's order decided which of two equal routes was kept,
                // and it differs between Python and the phone (the parity audit).
                for (sq in frontier.sorted()) {
                    scratch.setPieceAt(sq, piece)
                    val moves = ArrayList<Move>()
                    scratch.generatePseudoLegalMoves(BB_SQUARES[sq]) { moves.add(it) }
                    for (move in moves) {
                        val to = move.to
                        if (to !in landings || to in parent) continue
                        if (stopAtRed && anyRedBetween(sq, to, red)) continue
                        parent[to] = sq
                        if (to == target) {
                            val visited = HashSet<Int>()
                            var back = parent[to]
                            while (back != null && back != origin) {
                                visited.add(back)
                                back = parent[back]
                            }
                            found = visited
                            break
                        }
                        nextFrontier.add(to)
                    }
                    scratch.removePieceAt(sq)
                    if (found != null) break
                }
                if (found != null || nextFrontier.isEmpty()) break
                frontier = nextFrontier
            }
            val visited = found ?: continue
            val way = QueueWay(origin, visited, passesOver = !stopAtRed && slider)
            if (ways.none { it.visited == visited }) ways.add(way)
        }
        return ways
    }

    private fun anyRedBetween(a: Int, b: Int, red: Set<Int>): Boolean {
        var hit = false
        scanReversed(between(a, b)) { sq -> if (sq in red) hit = true }
        return hit
    }

    /** Whether a move story seen once is waiting for its confirming frame. */
    val pendingStory: Boolean get() = pendingStory() != null

    /**
     * Whether the last frame held a drag in progress -- a hover outline, or a piece not
     * settled on its square -- which blocks every story without contradicting any.
     */
    val blocked: Boolean get() = blockedFlag

    /** Whether the last frame disagreed with the position and no story explained it. */
    val unexplained: Boolean get() = unexplainedFrames > 0

    /** For how many frames in a row the board has disagreed and no story explained it. */
    val unexplainedFor: Int get() = unexplainedFrames

    /**
     * Whether the last readable frame shows the latest position, ghosts aside: the record is
     * up to date with the screen.
     */
    val agrees: Boolean get() = disagreements(lastObs).isEmpty()

    /**
     * The story seen once and still pending, when it ends the game: the mate the site's
     * dialog then covered before a second clean frame could confirm it.
     */
    fun pendingTerminal(): Pair<List<Move>, Board>? {
        val story = pendingStory() ?: return null
        if (!story.second.isGameOver()) return null
        return story
    }

    /** Commit the pending story that ends the game (see [pendingTerminal]). */
    fun commitPendingTerminal(): MutableList<Event> {
        val story = pendingTerminal() ?: return mutableListOf()
        val events = commitMoves(story.first)()
        pending = null
        stale.clear()
        unexplainedFrames = 0
        return events
    }

    /** Stale readings of something unnameable on squares the position has empty. */
    private fun ghosts(diff: Set<Int>, obs: Observation): Set<Int> =
        diff.filter { sq ->
            stale.getValue(sq).second > staleAfter &&
                obs.getValue(sq).kind == Kind.PIECE &&
                obs.getValue(sq).symbol == null &&
                board.pieceAt(sq) == null
        }.toSet()

    // Position resync

    /**
     * Adopt the observed position when it is complete, still, and unexplained. This is what
     * recovers from a gap no move search can bridge: the window minimised for a minute, a
     * dialog over the board, a restart mid-game.
     */
    private fun maybePositionResync(obs: Observation, diff: Set<Int>): MutableList<Event> {
        if (diff.size < resyncCells || obs.size < 64) {
            resyncCandidate = null
            return mutableListOf()
        }
        if (obs.values.any { it.hover }) {
            resyncCandidate = null
            return mutableListOf()
        }
        var soft = 0
        for (reading in obs.values) {
            if (reading.kind != Kind.PIECE) continue
            if (reading.symbol == null) {
                // A badge or ring hides the type; the best guess stands in, for a few cells.
                if (reading.guess == null) { resyncCandidate = null; return mutableListOf() }
                soft += 1
            } else if (!reading.settled) {
                resyncCandidate = null
                return mutableListOf()
            }
        }
        if (soft > resyncMaxSoft) {
            resyncCandidate = null
            return mutableListOf()
        }
        // Held as a view: the pieces and the squares the site highlights, which a move
        // starting changes first (`GamePipeline.view`).
        val placement = placement(obs).boardFen() + "|" +
            highlightedSquares(obs).sorted().joinToString(",") { squareName(it) }
        val candidate = resyncCandidate
        resyncCandidate = if (candidate != null && candidate.first == placement) {
            Triple(placement, candidate.second + 1, candidate.third)
        } else {
            Triple(placement, 1, tsValue)
        }
        val (_, frames, since) = resyncCandidate!!
        val held = frames >= resyncHoldFrames ||
            (diff.size >= resyncJumpCells && frames >= resyncMinFrames && tsValue - since >= resyncHoldS)
        if (!held) return mutableListOf()

        val adopted = adoptPlacement(placement(obs), bottomColour(), highlightedSquares(obs))
        if (adopted == null) {
            resyncCandidate = null
            return mutableListOf()
        }
        val (adoptedBoard, unknownTurn) = adopted
        turnUnknownFlag = unknownTurn
        val events = mutableListOf<Event>()
        if (finishedFlag) {
            // The finished game is over for good; a position that appears after it is another
            // game, seen from the middle -- or a puzzle, or an analysis board.
            gameId = UUID.randomUUID().toString().replace("-", "").take(8)
            kind = "position"
            events.add(
                GameStarted(
                    gameId = gameId, userSide = userSide, startFen = adoptedBoard.fen(), kind = kind,
                ),
            )
        }
        board = adoptedBoard
        baseFenValue = adoptedBoard.fen()
        finishedFlag = false
        resultValue = null
        lastCommitTs = tsValue
        pending = null
        stale.clear()
        viewing = null
        unexplainedFrames = 0
        resyncCandidate = null
        var reason = "position adopted after ${diff.size} unexplained cells"
        if (turnUnknownFlag) reason += "; side to move pending"
        events.add(Resync(gameId = gameId, fen = adoptedBoard.fen(), reason = reason))
        events.add(PositionChanged(gameId = gameId, fen = adoptedBoard.fen(), ply = 0))
        if (!turnUnknownFlag && adoptedBoard.isGameOver()) {
            finishedFlag = true
            events.add(ended(adoptedBoard.result(), "game over on board"))
        }
        return events
    }

    private fun bottomColour(): Boolean = if (bottom == Side.WHITE) WHITE else BLACK

    /**
     * The side to move settled by the clock that is running. A resync or a cold start leaves
     * the turn open when both sides could legally move, and the first story then decides it --
     * which a takeback answers wrongly. The running clock is evidence of the turn that no
     * story can contradict.
     */
    fun turnFromClock(colour: Boolean): MutableList<Event> {
        if (!turnUnknownFlag || !startedFlag || finishedFlag) return mutableListOf()
        val trial = board.copy(withStack = false)
        if (!turnValid(trial, colour)) return mutableListOf()
        turnUnknownFlag = false
        if (board.turn == colour) return mutableListOf()
        // The turn stays open only until the first commit, so nothing has been played from
        // the position yet: the base is the position with the turn corrected.
        check(board.moveStack.isEmpty())
        board.turn = colour
        baseFenValue = board.fen()
        val name = if (colour == WHITE) "white" else "black"
        val events = mutableListOf<Event>(
            Resync(
                gameId = gameId, fen = board.fen(),
                reason = "side to move settled by the clock: $name",
            ),
            PositionChanged(gameId = gameId, fen = board.fen(), ply = 0),
        )
        if (board.isGameOver()) {
            finishedFlag = true
            events.add(ended(board.result(), "game over on board"))
        }
        return events
    }

    /**
     * Begin a game from an arbitrary observed position -- a puzzle or a mid-game board read
     * cold from the theme cache -- rather than from the start. `ts` is when the position was
     * read, so the first move's think time counts from it.
     */
    fun startFrom(
        placement: Board,
        ts: Double? = null,
        highlighted: Set<Int> = emptySet(),
    ): MutableList<Event> {
        val adopted = adoptPlacement(placement, bottomColour(), highlighted) ?: return mutableListOf()
        if (ts != null) tsValue = ts
        val (adoptedBoard, unknownTurn) = adopted
        turnUnknownFlag = unknownTurn
        board = adoptedBoard
        baseFenValue = adoptedBoard.fen()
        startedFlag = true
        finishedFlag = false
        resultValue = null
        lastCommitTs = tsValue
        gameId = UUID.randomUUID().toString().replace("-", "").take(8)
        kind = "position"
        return mutableListOf(
            GameStarted(
                gameId = gameId, userSide = userSide, startFen = adoptedBoard.fen(), kind = kind,
            ),
        )
    }

    private fun placement(obs: Observation): Board {
        val board = Board(null)
        for ((sq, reading) in obs) {
            val symbol = reading.symbol ?: reading.guess
            if (reading.kind == Kind.PIECE && symbol != null) {
                board.setPieceAt(sq, Piece.fromSymbol(symbol))
            }
        }
        return board
    }

    /**
     * After a resync that left the side to move open, the first story decides it. Stories are
     * searched for both sides; a unique one settles like any other and fixes the turn only
     * when it commits, so a drag or a hypothesis seen once can never flip the turn on its own.
     */
    private fun explainTurnUnknown(
        obs: Observation,
        diff: Set<Int>,
        fullObs: Observation,
        paths: Boolean,
        cautious: Boolean = false,
        wild: Set<Int> = emptySet(),
    ): MutableList<Event>? {
        val found = ArrayList<Pair<Boolean, List<Move>>>()
        for (turn in listOf(WHITE, BLACK)) {
            val trial = board.copy(withStack = false)
            trial.turn = turn
            val stories = if (paths) {
                pathExplanations(obs, diff, trial, wild)
            } else {
                singleExplanations(trial, obs, diff)
            }
            for (story in stories) found.add(turn to story)
        }
        if (found.size != 1) return null
        val (turn, moves) = found[0]
        val trial = board.copy(withStack = false)
        trial.turn = turn
        if (!highlightOk(fullObs, moves, trial, wild)) return null
        val key = PendingKey(if (cautious) "turn-cautious" else "turn", turn, moves)
        return settle(key, fullObs, commitWithTurn(turn, moves))
    }

    /**
     * Events for one explanation of `diff`, or null when nothing fits.
     *
     * Order: one move from the latest position; an earlier position of this game (the user is
     * browsing); a different move, or a short path, from the viewed position (a takeback); a
     * short path of missed moves. `fullObs` is the whole frame, for the settled and hover
     * checks, while `obs` may leave ignored cells out. `cautious` marks the retry that ignores
     * stale cells; its paths need an extra frame to settle.
     *
     * After the game has ended the board is a review board: browsing is followed, and moves
     * played on it are analysis, not the game.
     */
    private fun explain(
        obs: Observation,
        diff: Set<Int>,
        fullObs: Observation,
        allowPaths: Boolean,
        cautious: Boolean = false,
        wild: Set<Int> = emptySet(),
    ): MutableList<Event>? {
        if (turnUnknownFlag) {
            var explained = explainTurnUnknown(obs, diff, fullObs, paths = false)
            if (explained == null && allowPaths) {
                explained = explainTurnUnknown(obs, diff, fullObs, paths = true, cautious = cautious, wild = wild)
            }
            return explained
        }

        if (!finishedFlag) {
            val singles = singleExplanations(board, obs, diff)
            if (singles.size == 1 && highlightOk(fullObs, singles[0], wild = wild)) {
                return settle(PendingKey("moves", null, singles[0]), fullObs, commitMoves(singles[0]))
            }
        }

        val index = historyIndex(obs, diff)
        if (index != null) {
            if (index == 0 && finishedFlag && baseFenValue == STARTING_FEN) {
                // The start position after a finished game is the next game, not browsing.
                return start()
            }
            if (blockedFlag) {
                // A piece dragged back over its own square shows an earlier position for a
                // moment; browsing is only ever entered from a quiet frame.
                return null
            }
            viewing = index
            pending = null
            stale.clear()
            unexplainedFrames = 0
            return mutableListOf()
        }

        if (finishedFlag) return null

        val viewed = viewing
        if (viewed != null) {
            val earlier = boardAt(viewed)
            val diffEarlier = obs.filter { (sq, r) -> !cellAgrees(r, earlier.pieceAt(sq)) }.keys.toSet()
            var branches = singleExplanations(earlier, obs, diffEarlier)
            if (branches.size != 1) {
                // The user played on from the viewed position and the opponent replied before
                // the board was quiet.
                branches = pathExplanations(obs, diffEarlier, earlier)
            }
            if (branches.size == 1 && highlightOk(fullObs, branches[0], earlier)) {
                val story = branches[0]
                if (allowPaths) {
                    // The same frame told from the latest position: with fewer plies it is
                    // the live game moving on while the user browsed; with more it retells the
                    // takeback as retreats, which nobody played.
                    val latest = pathExplanations(obs, diff, wild = wild)
                    if (latest.size == 1 && latest[0].size < story.size &&
                        highlightOk(fullObs, latest[0], wild = wild)
                    ) {
                        return explainPaths(obs, diff, fullObs, cautious = cautious, wild = wild)
                    }
                }
                if (viewed == 0 && baseFenValue == STARTING_FEN) {
                    // A move from the start position while the game had moves: a new game
                    // began (resigned, aborted, or rematch), not a takeback of everything.
                    return settle(PendingKey("newgame", null, story), fullObs, commitNewGame(story))
                }
                val plies = board.moveStack.size - viewed
                return settle(
                    PendingKey("takeback", viewed, story), fullObs, commitTakeback(plies, story),
                )
            }
        }

        if (allowPaths) return explainPaths(obs, diff, fullObs, cautious = cautious, wild = wild)
        return null
    }

    /** A short path of missed moves explaining `diff`; see [pathExplanations]. */
    private fun explainPaths(
        obs: Observation,
        diff: Set<Int>,
        fullObs: Observation,
        cautious: Boolean = false,
        wild: Set<Int> = emptySet(),
    ): MutableList<Event>? {
        if (finishedFlag) return null
        if (turnUnknownFlag) {
            return explainTurnUnknown(obs, diff, fullObs, paths = true, cautious = cautious, wild = wild)
        }
        val paths = pathExplanations(obs, diff, wild = wild)
        if (paths.size == 1 && highlightOk(fullObs, paths[0], wild = wild)) {
            val key = PendingKey(if (cautious) "moves-cautious" else "moves", null, paths[0])
            return settle(key, fullObs, commitMoves(paths[0]))
        }
        return null
    }

    val userColour: Boolean get() = if (userSide == Side.WHITE) WHITE else BLACK

    /**
     * Whether [moves], the pending story or a story built on it, catch a piece in flight: the
     * pending story seen on one frame only -- a piece that stood still on a square was seen there
     * twice, which a slide never is -- no premove queue about, whose steps land one for each reply
     * faster than any frame can witness them, and a piece stopping partway along its line and
     * going on the same way ([inFlight]).
     */
    private fun flight(moves: List<Move>): Boolean {
        val current = pending
        val seenOnce = current == null || current.second < 2
        val red = redTs
        val queue = red != null && tsValue - red <= QUEUE_MEMORY_S
        return seenOnce && !queue && inFlight(moves)
    }

    /**
     * The pending story from the latest position and the position it leads to. A story with a
     * piece in flight ([inFlight]) is nothing to build on: it settles, if it ever does, on frames
     * of its own.
     */
    private fun pendingStory(): Pair<List<Move>, Board>? {
        val current = pending ?: return null
        if (current.first.kind !in STORY_KEYS) return null
        val moves = current.first.moves
        if (flight(moves)) return null
        val base = board.copy(withStack = false)
        for (move in moves) {
            if (!base.isLegal(move)) return null
            base.push(move)
        }
        return moves to base
    }

    /** Every square is read, holds a settled piece or nothing, and is not under the cursor. */
    private fun cellsClean(obs: Observation, squares: Set<Int>): Boolean {
        for (sq in squares) {
            val reading = obs[sq] ?: return false
            if (reading.hover) return false
            if (reading.kind == Kind.PIECE && !reading.settled) return false
        }
        return true
    }

    /** Keep the pending story through a blocked frame, drop it on a quiet one or when old. */
    private fun holdPending(blocked: Boolean) {
        pendingAge += 1
        if (!blocked || pendingAge > pendingMaxAge) pending = null
    }

    /**
     * Explain the frame from the position the pending story leads to.
     *
     * With premoves, two plies land in one frame and the next drag starts a fraction of a
     * second later, before the story has settled; by the time the board is quiet again it is
     * several plies further on, and a piece may have returned to a square it left, which the
     * missed-move search cannot express from the committed position. The story seen just
     * before the drag is a hypothesis; if the moves that follow it explain the frame, the
     * whole chain is what settles.
     */
    private fun extendPending(obs: Observation, diff: Set<Int>): MutableList<Event>? {
        if (viewing != null) return null
        val storyPair = pendingStory() ?: return null
        val current = pending ?: return null
        val (moves, base) = storyPair
        val diffBase = obs.filter { (sq, r) -> !cellAgrees(r, base.pieceAt(sq)) }.keys.toSet()
        if (diffBase.isEmpty()) {
            // The frame shows the position the story leads to: one more frame of evidence.
            if (!highlightOk(obs, moves)) return null
            return settle(current.first, obs, commitMoves(moves))
        }
        val singles = singleExplanations(base, obs, diffBase)
        val story: List<Move>
        if (singles.size == 1) {
            story = singles[0]
        } else {
            val paths = pathExplanations(obs, diffBase, base)
            if (paths.size != 1) return null
            story = paths[0]
        }
        val combined = moves + story
        if (flight(combined)) {
            // The frame shows a piece of the pending story further along its own line: the story
            // caught it in flight, and this frame is no second witness of it.
            return null
        }
        if (!highlightOk(obs, combined)) return null
        // A bot that answers within a frame or two and a queue that executes on every answer
        // make a chain that grows by two plies a picture and never shows the same picture long
        // enough to settle. This frame is explained from the position the pending plies lead
        // to: the second witness those plies were waiting for. So they are committed now, and
        // only the plies this frame added stay pending.
        val events = commitMoves(moves)()
        pending = PendingKey("moves", null, story) to 1
        pendingAge = 0
        stale.clear()
        unexplainedFrames = 0
        return events
    }

    /**
     * The opponent's move, committed whether or not the user is dragging.
     *
     * A drag blocks the user's own moves, since a dragged piece is not a move, but it cannot
     * move the opponent's pieces. So the opponent's move commits from the cells it touches
     * alone, as long as they are settled and not under the cursor. Without this, a bot that
     * replies while the user is already dragging the next premove is seen only as a net
     * displacement once the board is quiet.
     */
    private fun explainOpponentMove(obs: Observation): MutableList<Event>? {
        if (viewing != null || turnUnknownFlag) return null
        var moves: List<Move> = emptyList()
        var base = board.copy(withStack = false)
        val story = pendingStory()
        if (story != null) {
            moves = story.first
            base = story.second
        } else if (pending != null && pending!!.first.kind in STORY_KEYS) {
            return null  // a pending story that is no longer legal explains nothing
        }
        val userColour = userColour
        val botDiff = HashSet<Int>()
        for ((sq, reading) in obs) {
            if (cellAgrees(reading, base.pieceAt(sq))) continue
            val piece = base.pieceAt(sq)
            if ((piece != null && piece.colour != userColour) ||
                (reading.kind == Kind.PIECE && reading.colour != null && reading.colour != userSide)
            ) {
                botDiff.add(sq)
            }
        }
        if (base.turn == userColour) {
            // The pending story ends with the opponent's move. The user's part of it was seen
            // settled before the drag; the opponent's part is confirmed by its own cells,
            // which the drag cannot touch.
            if (botDiff.isNotEmpty() || moves.isEmpty() || pending == null) return null
            val before = board.copy(withStack = false)
            for (move in moves.dropLast(1)) before.push(move)
            if (!cellsClean(obs, touchedSquares(before, moves.last()))) return null
            return settle(pending!!.first, obs, commitMoves(moves))
        }
        if (botDiff.isEmpty()) return null
        val replies = singleExplanations(base, obs, botDiff)
        if (replies.size != 1) return null
        val reply = replies[0][0]
        if (!cellsClean(obs, touchedSquares(base, reply))) return null
        val combined = moves + reply
        if (combined.size > maxChainPlies || flight(combined)) return null
        return settle(PendingKey("moves", null, combined), obs, commitMoves(combined))
    }

    /**
     * Frames during a premove queue, which chess.com does not draw as a position.
     *
     * While the opponent is to move, the user may queue several moves of one piece. The site
     * then draws that piece at the end of the queue and leaves its real square empty, tints
     * the queued squares, and executes the first queued move the moment the opponent moves,
     * without changing the picture. So during a queue the frame is the true position with
     * exactly one of the user's pieces displaced, and the queue itself is the unique shortest
     * way for that piece to get there.
     */
    private fun explainPremoveQueue(obs: Observation, diff: Set<Int>): MutableList<Event>? {
        if (viewing != null || turnUnknownFlag || hoverSeen) return null
        val userColour = userColour
        val shown = ArrayList<Int>()
        val gone = ArrayList<Int>()
        for (sq in diff) {
            val reading = obs.getValue(sq)
            val piece = board.pieceAt(sq)
            val users = piece != null && piece.colour == userColour
            if (reading.kind == Kind.PIECE && reading.symbol != null && reading.settled &&
                reading.colour == userSide && !users
            ) {
                shown.add(sq)
            } else if (reading.kind == Kind.EMPTY && users) {
                gone.add(sq)
            }
        }
        if (shown.size != 1 || gone.size != 1) return null
        val target = shown[0]
        val origin = gone[0]
        val symbol = obs.getValue(target).symbol!!
        // Chess.com tints every square of the queue red; the route the user actually queued
        // runs through exactly those squares, which tells a4 from a3 then a4.
        val tinted = obs.filter { (sq, reading) ->
            reading.tint == "red" && (reading.kind == Kind.EMPTY || sq == origin || sq == target)
        }.keys.toSet()
        var path = queuePath(origin, target, symbol, tinted) ?: return null
        val rest = diff - setOf(origin, target)
        val restObs = Observation()
        for ((sq, r) in obs) if (sq != origin && sq != target) restObs[sq] = r
        if (board.turn == userColour) {
            // The opponent's move is already committed, so the queue's first step has
            // executed; only the displaced piece may differ from the position.
            if (rest.isNotEmpty() || !board.isLegal(path[0])) return null
            if (!highlightOk(obs, listOf(path[0]))) return null
            return settle(PendingKey("moves", null, listOf(path[0])), obs, commitMoves(listOf(path[0])))
        }
        if (rest.isEmpty()) {
            // The queue formed or changed; nothing has happened on the board yet.
            unexplainedFrames = 0
            return mutableListOf()
        }
        val replies = singleExplanations(board, restObs, rest)
        if (replies.size != 1) return null
        val reply = replies[0][0]
        val after = board.copy(withStack = false)
        after.push(reply)
        // The queue's way is the way on the board the reply leaves: a knight that just left
        // the queen's diagonal made her one step of two.
        path = queuePath(origin, target, symbol, tinted, after) ?: return null
        if (!after.isLegal(path[0])) return null
        val story = listOf(reply, path[0])
        if (!highlightOk(obs, story)) return null
        return settle(PendingKey("moves", null, story), obs, commitMoves(story))
    }

    /**
     * The sequence of moves taking the user's piece from origin to target. When the site tints
     * the queued squares, the route through exactly those squares is the queue, however long;
     * otherwise the unique shortest way is assumed. Several equally short ways are an
     * ambiguity and nothing is assumed. `internal` for the desktop's tests of `_queue_path`.
     */
    internal fun queuePath(
        origin: Int,
        target: Int,
        symbol: String,
        tinted: Set<Int> = emptySet(),
        board: Board? = null,
    ): List<Move>? {
        val from = board ?: this.board
        val piece = from.pieceAt(origin) ?: return null
        if (Piece.fromSymbol(symbol).colour != piece.colour) return null
        var promotion: Int? = null
        if (piece.pieceType != PAWN && symbol.uppercase() != piece.symbol().uppercase()) return null
        if (piece.pieceType == PAWN && symbol.uppercase() != "P") {
            if (squareRank(target) != 0 && squareRank(target) != 7) return null
            promotion = Piece.fromSymbol(symbol).pieceType
        } else if (piece.pieceType == PAWN && (squareRank(target) == 0 || squareRank(target) == 7)) {
            return null
        }
        val scratch = from.copy(withStack = false)
        scratch.turn = piece.colour
        scratch.epSquare = null  // a queue never passes through the en-passant square
        scratch.removePieceAt(origin)
        if (target in tinted) {
            routeThrough(scratch, piece, origin, target, tinted, promotion)?.let { return it }
        }
        // Breadth-first over the piece's own moves, counting the shortest ways to each square.
        var frontier = listOf(origin)
        val ways = HashMap<Int, Int>()
        ways[origin] = 1
        val parent = HashMap<Int, Move>()
        var depth = 0
        while (frontier.isNotEmpty() && target !in ways && depth < maxQueueSteps) {
            depth += 1
            val nextFrontier = ArrayList<Int>()
            for (sq in frontier) {
                scratch.setPieceAt(sq, piece)
                val moves = ArrayList<Move>()
                scratch.generatePseudoLegalMoves(BB_SQUARES[sq]) { moves.add(it) }
                for (move in moves) {
                    val to = move.to
                    if (to != target && scratch.pieceAt(to) != null) continue
                    if (to == target && move.promotion != 0 && move.promotion != promotion) continue
                    if (to in ways && to !in nextFrontier) continue
                    if (to !in ways) {
                        ways[to] = 0
                        nextFrontier.add(to)
                        parent[to] = move
                    }
                    ways[to] = ways.getValue(to) + ways.getValue(sq)
                }
                scratch.removePieceAt(sq)
            }
            frontier = nextFrontier
        }
        if ((ways[target] ?: 0) != 1) return null
        val path = ArrayList<Move>()
        var sq = target
        while (sq != origin) {
            val move = parent.getValue(sq)
            path.add(move)
            sq = move.from
        }
        path.reverse()
        return path
    }

    /** The unique route from origin to target visiting every tinted square once. */
    private fun routeThrough(
        scratch: Board,
        piece: Piece,
        origin: Int,
        target: Int,
        tinted: Set<Int>,
        promotion: Int?,
    ): List<Move>? {
        val stops = tinted - origin
        val routes = ArrayList<List<Move>>()

        fun walk(sq: Int, visited: Set<Int>, path: ArrayList<Move>) {
            if (routes.size > 1 || path.size > maxQueueSteps) return
            if (sq == target) {
                if (visited == stops) routes.add(ArrayList(path))
                return
            }
            scratch.setPieceAt(sq, piece)
            val moves = ArrayList<Move>()
            scratch.generatePseudoLegalMoves(BB_SQUARES[sq]) { moves.add(it) }
            for (move in moves) {
                val to = move.to
                if (to !in stops || to in visited) continue
                if (to != target && (scratch.pieceAt(to) != null || move.promotion != 0)) continue
                if (to == target && move.promotion != 0 && move.promotion != promotion) continue
                path.add(move)
                walk(to, visited + to, path)
                path.removeAt(path.size - 1)
            }
            scratch.removePieceAt(sq)
        }

        walk(origin, emptySet(), ArrayList())
        return if (routes.size == 1) routes[0] else null
    }

    // Settling and committing

    /** Count frames an explanation has held; commit it once it has settled. */
    private fun settle(
        key: PendingKey,
        obs: Observation,
        commit: () -> MutableList<Event>,
    ): MutableList<Event> {
        val current = pending
        pending = if (current != null && current.first == key) key to current.second + 1 else key to 1
        pendingAge = 0
        val needed = settleFrames +
            (if (key.moves.size > 1) 1 else 0) +
            (if (key.kind == "moves-cautious") 1 else 0)
        if (pending!!.second < needed) return mutableListOf()
        val events = commit()
        pending = null
        stale.clear()
        unexplainedFrames = 0
        viewing = null
        return events
    }

    private fun commitMoves(moves: List<Move>): () -> MutableList<Event> = {
        val events = mutableListOf<Event>()
        for (move in moves) events.addAll(commit(move))
        events
    }

    private fun commitWithTurn(turn: Boolean, moves: List<Move>): () -> MutableList<Event> = {
        board.turn = turn
        baseFenValue = board.fen()
        turnUnknownFlag = false
        commitMoves(moves)()
    }

    private fun commitTakeback(plies: Int, moves: List<Move>): () -> MutableList<Event> = {
        repeat(plies) { board.pop() }
        val events = mutableListOf<Event>(
            Takeback(gameId = gameId, plies = plies, fen = board.fen()),
            PositionChanged(gameId = gameId, fen = board.fen(), ply = board.moveStack.size),
        )
        for (move in moves) events.addAll(commit(move))
        events
    }

    private fun commitNewGame(moves: List<Move>): () -> MutableList<Event> = {
        val events = mutableListOf<Event>()
        if (!finishedFlag) events.add(ended("*", "new game started"))
        events.addAll(start())
        for (move in moves) events.addAll(commit(move))
        events
    }

    /**
     * Stale cells to ignore, singles before pairs, junk-looking readings first. At most
     * `resyncCells - 1` cells are ever ignored together: more than that is no longer a badge
     * or two, and only a position resync may act.
     */
    private fun staleSubsets(staleCells: Set<Int>, obs: Observation): List<Set<Int>> {
        val junk = staleCells.filter { isJunk(obs.getValue(it)) }.sorted()
        val clean = staleCells.filter { !isJunk(obs.getValue(it)) }.sorted()
        val pool = if (staleCells.size > maxStaleSubsetCells) junk else junk + clean
        val subsets = ArrayList<Set<Int>>()
        for (sq in pool) subsets.add(setOf(sq))
        if (resyncCells > 2) {
            for (i in pool.indices) for (j in i + 1 until pool.size) subsets.add(setOf(pool[i], pool[j]))
        }
        return subsets
    }

    /** Resync only when several cells have stayed unexplained for a long time. */
    private fun maybeResync(diff: Set<Int>): MutableList<Event> {
        if (diff.isEmpty()) {
            unexplainedFrames = 0
            return mutableListOf()
        }
        unexplainedFrames += 1
        val persistent = diff.filter { stale.getValue(it).second > staleAfter }
        if (persistent.size < resyncCells || unexplainedFrames < resyncAfter) return mutableListOf()
        unexplainedFrames = 0
        val names = persistent.sorted().joinToString(", ") { squareName(it) }
        return mutableListOf(Unexplained(gameId = gameId, fen = board.fen(), reason = "unexplained: $names"))
    }

    /**
     * Count consecutive frames each cell has disagreed with an unchanged reading. A changed
     * reading resets the count, so a cell that was stale during a long drag becomes fresh
     * again the moment the piece lands and the highlight appears.
     */
    private fun advanceStale(diff: Set<Int>, obs: Observation) {
        for (sq in stale.keys.toList()) if (sq !in diff) stale.remove(sq)
        for (sq in diff) {
            val reading = obs.getValue(sq)
            val key = StaleKey(reading.kind, reading.symbol, reading.colour, reading.highlighted)
            val was = stale[sq]
            stale[sq] = if (was != null && was.first == key) key to was.second + 1 else key to 1
        }
    }

    /** One move played on the game. `internal` for the desktop's test that calls `_commit`. */
    internal fun commit(move: Move): MutableList<Event> {
        if (finishedFlag) {
            // Defensive: nothing is played on a finished game. A new game starts it over.
            pending = null
            return mutableListOf()
        }
        if (!board.isLegal(move)) {
            // Defensive: an explanation computed against another position must never corrupt
            // the game. Report it and let the next frames explain the board again.
            pending = null
            return mutableListOf(
                Unexplained(
                    gameId = gameId, fen = board.fen(),
                    reason = "refused illegal commit ${move.uci()}",
                ),
            )
        }
        val fenBefore = board.fen()
        val san = board.san(move)
        val byUser = (board.turn == WHITE) == (userSide == Side.WHITE)
        // Time since the previous commit, as seen on screen: a chain committed at once gives
        // it all to its first move and leaves the others open.
        var thinkMs: Int? = null
        val since = lastCommitTs
        // `int(round(...))`: Python rounds half to even.
        if (since != null && tsValue > since) thinkMs = pyRound((tsValue - since) * 1000.0)
        lastCommitTs = tsValue
        board.push(move)
        val ply = board.moveStack.size
        val events = mutableListOf<Event>(
            MovePlayed(
                gameId = gameId,
                ply = ply,
                san = san,
                uci = move.uci(),
                fenBefore = fenBefore,
                fenAfter = board.fen(),
                byUser = byUser,
                thinkMs = thinkMs,
            ),
            PositionChanged(gameId = gameId, fen = board.fen(), ply = ply),
        )
        if (board.isGameOver()) {
            finishedFlag = true
            resultValue = board.result()
            events.add(ended(resultValue!!, "game over on board"))
        }
        return events
    }

    /**
     * End the game: on the user's word (a resignation, a draw, an abort) or on the site's
     * dialog. A game that ended with the result unknown ("*") can be ended again with the
     * result the user supplies.
     */
    fun end(result: String, reason: String): MutableList<Event> {
        if (!startedFlag || (finishedFlag && resultValue != "*")) return mutableListOf()
        finishedFlag = true
        pending = null
        resultValue = result
        return mutableListOf(ended(result, reason))
    }

    private fun ended(result: String, reason: String): GameEnded = GameEnded(
        gameId = gameId,
        result = result,
        reason = reason,
        pgn = pgn(),
        finalFen = board.fen(),
    )

    /** Moves since the base position; after a resync the numbering restarts at 1. */
    fun pgn(): String {
        val replay = Board(baseFenValue)
        val parts = ArrayList<String>()
        for ((i, move) in board.moveStack.withIndex()) {
            if (!replay.isLegal(move)) {
                parts.add("[${move.uci()}]")
                continue
            }
            if (replay.turn == WHITE) {
                parts.add("${replay.fullmoveNumber}.")
            } else if (i == 0) {
                parts.add("${replay.fullmoveNumber}...")
            }
            parts.add(replay.san(move))
            replay.push(move)
        }
        return parts.joinToString(" ")
    }
}
