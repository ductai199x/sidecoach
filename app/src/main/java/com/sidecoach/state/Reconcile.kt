package com.sidecoach.state

import com.sidecoach.chess.BB_EMPTY
import com.sidecoach.chess.BB_SQUARES
import com.sidecoach.chess.BISHOP
import com.sidecoach.chess.BLACK
import com.sidecoach.chess.Bitboard
import com.sidecoach.chess.Board
import com.sidecoach.chess.KING
import com.sidecoach.chess.KNIGHT
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.Piece
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.between
import com.sidecoach.chess.square
import com.sidecoach.chess.squareDistance
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareName
import com.sidecoach.chess.squareRank
import com.sidecoach.chess.squaresOf
import com.sidecoach.events.Event
import com.sidecoach.events.GameEnded
import com.sidecoach.events.GameStarted
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.PositionChanged
import com.sidecoach.events.Resync
import com.sidecoach.events.Takeback
import com.sidecoach.events.TurnPending
import com.sidecoach.events.Unexplained
import com.sidecoach.vision.Pieces.CellReading
import com.sidecoach.vision.Pieces.Kind
import com.sidecoach.vision.Pieces.Side
import com.sidecoach.vision.Squares.squareAt
import com.sidecoach.vision.pyRound
import java.util.UUID
import kotlin.math.sign

/**
 * Per-frame cell readings reconciled into a legal game: a port of
 * `sidecoach/state/reconcile.py`, line for line.
 *
 * The desktop's docs/reconciler-model.md is the design. Every way the game may have gone on
 * past the record that fits the frames is kept as a story ([Story], `Stories.kt`), costed by
 * the pictures it needs that no frame showed; the record takes only what all of them agree
 * on. Premove queues are part of the state, drawn the way the site draws them, and no colour
 * is read.
 *
 * A move is committed only after it has held `settleFrames` frames in which every square it
 * touches is read settled, none of them is under something drawn, and no square carries the
 * hover outline of a drag. Highlight colours are never consulted.
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

/** `chess.piece_name`, by piece type, for the misread correction's reason. */
private val MISREAD_NAMES = arrayOf("", "pawn", "knight", "bishop", "rook", "queen", "king")

/** The fields a stale count is keyed by; a changed reading resets it. */
private data class StaleKey(
    val kind: Kind,
    val symbol: String?,
    val colour: Side?,
    val highlighted: Boolean,
)

/** One op every story agrees on, the frames it has held, and the frames it needs (`_agreed`). */
private data class Agreed(val op: Any, val count: Int, val need: Int)

/**
 * `reconcile.highlighted_squares`: the squares the site marks as its last move (or a
 * selection); a red tint is never a highlight ([com.sidecoach.vision.ThemeModel.classify]).
 */
fun highlightedSquares(obs: Observation): Set<Int> = obs.filter { (_, r) -> r.highlighted }.keys

/** `reconcile.by_square`: a frame's readings keyed by the squares they show, [bottom] at the bottom. */
fun bySquare(readings: Readings, bottom: Side): Observation {
    val out = Observation()
    for (r in 0 until 8) for (c in 0 until 8) out[squareAt(r, c, bottom.color)] = readings[r * 8 + c]
    return out
}

/**
 * `reconcile.loose_placement`: the pieces a read names, a held or a ringed one included (its
 * kind, or its best guess), an unreadable cell taken as empty: what the board shows, not a
 * position.
 */
fun loosePlacement(obs: Observation): Board {
    val board = Board(null)
    for ((sq, reading) in obs) {
        val symbol = reading.symbol ?: reading.guess
        if (reading.kind == Kind.PIECE && symbol != null) board.setPieceAt(sq, Piece.fromSymbol(symbol))
    }
    return board
}

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

/**
 * The result a game's final position proves and how, in the words the review reads; null when it
 * proves none (the game was resigned, flagged, agreed or left). A draw by repetition or by fifty
 * moves counts only when it was played -- the position on the board for the third time, or the
 * hundredth half-move without a capture or a pawn move -- as the sites draw it, never one the next
 * move could reach (the library's `stat_facts.ended_by`). `reconcile.result_on_board`.
 */
fun resultOnBoard(board: Board): Pair<String, String>? = when {
    board.isCheckmate() -> (if (board.turn == WHITE) "0-1" else "1-0") to "checkmate"
    board.isStalemate() -> "1/2-1/2" to "stalemate"
    board.isInsufficientMaterial() -> "1/2-1/2" to "insufficient material"
    board.isRepetition(3) -> "1/2-1/2" to "repetition"
    board.halfmoveClock >= 100 -> "1/2-1/2" to "fifty-move rule"
    else -> null
}

/**
 * The squares en passant could be open on, with `board.turn` to move: behind each pawn of the
 * other side on its fourth rank whose two squares behind are empty (it may have just come from
 * there). `_en_passant_squares`.
 */
private fun enPassantSquares(board: Board): List<Int> {
    val other = !board.turn
    val rank = if (other == WHITE) 3 else 4 // its fourth rank
    val step = if (other == WHITE) -8 else 8 // towards where it came from
    val out = mutableListOf<Int>()
    for (sq in board.pieces(PAWN, other)) {
        if (squareRank(sq) != rank) continue
        val behind = sq + step
        val start = sq + 2 * step
        if (board.pieceAt(behind) == null && board.pieceAt(start) == null) out.add(behind)
    }
    return out
}

/**
 * Whether [move] is the one the two [lit] squares show: its own two squares, or, for a castling,
 * the king's or the rook's. `_lights`.
 */
private fun lights(board: Board, move: Move, lit: Set<Int>): Boolean {
    if (setOf(move.from, move.to) == lit) return true
    if (board.isCastling(move)) {
        val file = if (squareFile(move.to) == 6) 7 else 0
        val rookFrom = square(file, squareRank(move.from))
        return move.from in lit && (move.to in lit || rookFrom in lit)
    }
    return false
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
 * How long after the last frame a story with a premove queue fitted a position resync is held
 * off ([Reconciler.queueTs]): a picture with a queue drawn is not a position. While the queue
 * fits, nothing is re-read anyway (a player may queue a whole opening, some twenty premoves
 * over ten seconds); once nothing has fitted for this long, the game is lost, and the board is
 * read whole rather than left stuck (the premove sweep, 2026-10-05, seed 70).
 */
const val QUEUE_RESYNC_S = 1.0

/** A frame later than this after the last is a gap: moves may have come and gone unseen. */
const val GAP_S = 0.5

/**
 * What a shape off its square on a square a story draws empty adds to its prior
 * ([Reconciler.weighed]): less than a premove edit's.
 */
const val SHAPE_PRIOR = 0.05

/**
 * How much less likely a story with the other side to move may be and still be kept: three
 * premove assumptions more than the likeliest ([Reconciler.fitting]).
 */
const val TURN_PRIOR_GAP = 0.29

/** How long the square a dragged piece was lifted from says which piece the hand moved. */
const val LIFT_MEMORY_S = 2.0

/** Stories kept at most, cheapest first ([Story]). */
const val MAX_STORIES = 12

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

fun matches(board: Board, obs: Observation, squares: Set<Int>): Boolean =
    squares.all { sq -> sq !in obs || cellAgrees(obs.getValue(sq), board.pieceAt(sq)) }

/**
 * `reconcile.in_flight`: whether a story has a piece stop partway along a straight line and then
 * go on the same way: a to x, and later x to y by that piece, x strictly between a and y. Every
 * site slides a moved piece in a straight line from its square to the next, and a frame that
 * catches it near the centre of a square on its way reads as a piece standing there; the next
 * frame, the piece further on, then reads as the same piece moving on -- a move, or a premove
 * queued (Tai, 2026-09-25: "I played Rxc1+ but it misunderstood as Rc7"; a pawn sliding d7-d5
 * caught on d6, the stress sweep, 2026-10-05, seed 2). A piece that turns (Kd4-c3, then c3-c4
 * premoved) or comes back is no slide. Geometry alone: no highlight is read. Returns the stop,
 * the move to x, or null.
 */
fun inFlight(moves: List<Move>): Move? {
    val landed = HashMap<Int, Move>()
    for (move in moves) {
        val came = landed.remove(move.from)
        if (came != null && between(came.from, move.to) and BB_SQUARES[move.from] != BB_EMPTY) {
            return came
        }
        landed[move.to] = move
    }
    return null
}

/**
 * `reconcile._landed`: [story], two pictures dearer when it catches a piece in flight
 * ([inFlight]) at a stop seen on fewer frames than a move needs to settle: the stop was a frame
 * of the slide, and the move straight from the record is what was played. A stop seen longer was
 * a move: no hand plays a move and premoves on from it inside a tenth of a second. Only a move
 * played can be a stop: a premove is dropped by hand and drawn at once, never slid (the premove
 * sweep, 2026-10-05, seed 57).
 */
private fun landed(story: Story, agreed: List<Agreed>, queued: Boolean = false): Story {
    if (story.slide) return story  // charged once, when it was told so
    // A queue about before this frame: its steps land one for each reply, the bot's replies
    // faster than frames can witness them, and a pawn the bot pushes twice in two frames is no
    // slide. A queue this frame's story makes up to explain a piece further on along its line is
    // no such queue (the stress sweep, seed 2).
    if (queued) return story
    val stop = inFlight(story.moves + story.queue)
    if (stop == null || stop !in story.moves || stop in story.premoved) return story
    if (agreed.any { it.op == stop && it.count >= 2 }) return story
    return story.copy(cost = story.cost + 2.0, slide = true)
}

/**
 * `reconcile._story_order`: stories cheapest first; of equal cost, a takeback or a new game
 * before moves that retell it (a takeback and the bot's quick reply, never four retreats), then
 * fewer moves. The sort is stable: what is left in the order the search found them, so the
 * story followed does not change from frame to frame.
 */
private val STORY_ORDER: Comparator<Story> = compareBy<Story>(
    { it.cost },
    { it.prior },
    { s -> if (s.head.any { it is HeadOp.Back || it is HeadOp.NewGame }) 0 else 1 },
    { it.ops.size },
)

/** `reconcile._common_start`: the ops every list begins with. */
private fun commonStart(lists: List<List<Any>>): List<Any> {
    val out = ArrayList<Any>()
    val shortest = lists.minOf { it.size }
    for (i in 0 until shortest) {
        val op = lists[0][i]
        if (lists.any { it[i] != op }) break
        out.add(op)
    }
    return out
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
     * unexplained square are expanded, so a four-ply gap in a middlegame costs a few hundred.
     */
    var searchBudget: Int = 4000,
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
    /** Premove steps a frame may add to a queue ([Search]). */
    var maxQueueSteps: Int = 3,
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

    /**
     * The stories that fit the frames so far ([Story], docs/reconciler-model.md), and what the
     * record does next: each op every story agrees on, with the frames it has held and the
     * frames it needs before it is played ([decide]).
     */
    private var stories: List<Story> = listOf(Story())
    private var agreed: List<Agreed> = emptyList()

    /** Whether the last frame showed the record as it stands (a queue drawn on it aside). */
    private var agreesFlag = true

    /** Whether the side to move was last told pending ([TurnPending]). */
    private var turnPendingTold = false
    private val stale = HashMap<Int, Pair<StaleKey, Int>>()
    private var unexplainedFrames = 0

    /** Frames a board stays unexplained before the log is told what each square reads as. */
    var unexplainedNoteFrames: Int = 20
    private var unexplainedNoted = false

    // `internal` for the desktop's tests, which set `_started`, `_finished` and `_base_fen`
    // on a reconciler as they stand (`tools/make_reconcile_fixtures.py`).
    internal var startedFlag = false
    internal var finishedFlag = false
    private var resultValue: String? = null
    internal var baseFenValue: String = STARTING_FEN
    private var turnUnknownFlag = false

    /** The placement the resync would adopt, how many frames have shown it, and since when. */
    private var resyncCandidate: Triple<String, Int, Double>? = null

    /**
     * The last [briefKept] distinct placements a resync held for [briefFrames] clean frames:
     * the placement, its highlighted squares, its view key, and when it was last seen. A new
     * puzzle's first board stands for well under a second before the site plays the opening
     * move, too briefly to be adopted; the board after the move was, and the move was lost
     * (Tai's puzzles, 2026-10-03: "the coach misses the first move ... 75-80% of the time").
     * Adopted, the board after is played from the brief one when one legal move joins them.
     * `_resync_seen`.
     */
    private var resyncSeen: List<Seen> = emptyList()

    private data class Seen(val placed: Board, val highlighted: Set<Int>, val key: String, val ts: Double)

    /**
     * One clean read: the phone, its voice and engine busy, read the next puzzle's first board,
     * up a third of a second, once (Tai's dump 20261003-171356). What keeps a stray read from
     * standing in is that exactly one legal move must join it to the board adopted.
     */
    var briefFrames: Int = 1

    /** How many distinct boards held that long are kept: the first board, and frames past it. */
    var briefKept: Int = 4

    /** How long before the adopted board the brief one may have been seen and still be its start. */
    var briefRecentS: Double = 2.0

    /**
     * The squares of a position read from the screen whose pieces have not moved since: their
     * kinds rest on that one read. A knight on a highlighted square read as a bishop made its
     * first move illegal, and the game's next three moves went unread until a resync (Tai's
     * puzzle on the Pixel, 2026-10-03); [misreadRoots] corrects such a piece by its move.
     * `_read_squares`.
     */
    private var readSquares: Set<Int> = emptySet()
    var misreadMaxCells: Int = 8
    private var unseen = 0

    /**
     * Last stable reading of every cell and the frame it was seen in. `internal` for the
     * desktop's tests of the memory, which look at `_last_seen`.
     */
    internal val lastSeen = HashMap<Int, Pair<CellReading, Int>>()
    private val previous = HashMap<Int, StaleKey>()

    /**
     * Squares something is drawn over in the current frame: unreadable, and no move commits
     * through them, though the drawing blocks nothing else.
     */
    private var drawn = HashSet<Int>()
    private var frameIndex = 0
    private var blockedFlag = false

    /**
     * Whether a piece is in the user's hand this frame ([update]), and the square the last piece
     * seen in the hand was lifted from, and when ([otherHand]).
     */
    private var dragging = false
    private var lifted: Pair<Int, Double>? = null

    /** When the last frame came, and how many unseen pictures a story may need this frame. */
    private var frameTs: Double? = null
    private var maxUnseen = 1
    private var lastObs: Observation = Observation()
    private var tsValue = 0.0
    private var lastCommitTs: Double? = null

    /** When a frame last fitted a story with a premove queue drawn ([maybePositionResync]). */
    private var queueTs: Double? = null

    /** For how many frames in a row a story with a queue has fitted. */
    private var queueFrames = 0

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
                val sq = squareAt(r, c, bottom.color)
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
                if (!(reading.exact && reading.settled && !reading.hover)) {
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

    /** The squares whose reading disagrees with [against] (the latest position). */
    fun disagreements(obs: Observation, against: Board = board): MutableSet<Int> {
        val out = LinkedHashSet<Int>()
        for ((sq, reading) in obs) if (!cellAgrees(reading, against.pieceAt(sq))) out.add(sq)
        return out
    }

    // Lifecycle

    fun start(): MutableList<Event> {
        clearStory()
        viewing = null
        turnUnknownFlag = false
        resyncCandidate = null
        lastSeen.clear()
        previous.clear()
        val started = begin(Board(), "game")
        return mutableListOf(started, PositionChanged(gameId = gameId, fen = board.fen(), ply = 0))
    }

    /** A game of [kind] begun on [board], under a new id (`_begin`). */
    private fun begin(board: Board, kind: String): GameStarted {
        gameId = UUID.randomUUID().toString().replace("-", "").take(8)
        this.kind = kind
        startedFlag = true
        setPosition(board)
        return gameStarted()
    }

    /**
     * [board] adopted as the game's position, the base its moves are counted from; the first
     * move's think time counts from now (`_set_position`).
     */
    private fun setPosition(board: Board) {
        this.board = board
        baseFenValue = board.fen()
        clearStory()
        // The start position is known, not read; any other is the screen's word for it.
        val read = board.boardFen() != Board().boardFen()
        readSquares = if (read) board.pieceMap().keys.toSet() else emptySet()
        finishedFlag = false
        resultValue = null
        lastCommitTs = tsValue
    }

    private fun gameStarted(): GameStarted =
        GameStarted(gameId = gameId, userSide = userSide, startFen = board.fen(), kind = kind)

    /**
     * No story pending, no cell remembered as stale or unexplained: the position they were
     * read against is gone (`_clear_story`).
     */
    private fun clearStory() {
        stories = listOf(Story())
        agreed = emptyList()
        stale.clear()
        unexplainedFrames = 0
        unexplainedNoted = false
    }

    /** The game under way ended, unless it had, and a new one begun from the start (`_restart`). */
    private fun restart(): MutableList<Event> {
        val events = if (finishedFlag) mutableListOf<Event>() else mutableListOf<Event>(replacedEvent("new game started"))
        events.addAll(start())
        return events
    }

    /**
     * The board was flipped or re-themed before the first move: the same game, seen anew.
     * Everything remembered per square is dropped, since the squares were mapped with the
     * old orientation.
     */
    fun reorient(bottom: Side): MutableList<Event> {
        this.bottom = bottom
        userSide = bottom
        clearStory()
        lastSeen.clear()
        previous.clear()
        viewing = null
        return if (startedFlag) mutableListOf(gameStarted()) else mutableListOf()
    }

    // Explanations

    /**
     * Whether an unsettled reading of a piece of no known kind is a king under the site's
     * decoration: on the king's own square, or one step from it. The colour, when read, must
     * be the king's.
     */
    private fun decoratedKing(square: Int, reading: CellReading): Boolean {
        if (reading.symbol != null) return false
        for (colour in booleanArrayOf(WHITE, BLACK)) {
            if (reading.colour != null && reading.colour != Side.of(colour)) continue
            val king = board.king(colour)
            if (king != null && squareDistance(king, square) <= 1) return true
        }
        return false
    }

    /**
     * Whether `move` is a king's move onto `square`, where the reading is a piece of the
     * king's colour whose kind could not be read, or one read off its square (a badge over the
     * king reads as some piece): the king under its decoration. A site decorates a king's
     * square when the king is in check, so a story that uses this must end with that king in
     * check ([checkedKings]).
     */
    private fun kingLands(board: Board, move: Move, square: Int, obs: Observation): Boolean {
        if (move.to != square || square !in obs) return false
        val piece = board.pieceAt(move.from) ?: return false
        if (piece.pieceType != KING) return false
        val reading = obs.getValue(square)
        if (reading.kind != Kind.PIECE || (reading.symbol != null && reading.settled)) return false
        return reading.colour == null || reading.colour == Side.of(piece.colour)
    }

    /**
     * Whether `move` is a king's move from `square` onto a decorated square ([kingLands]), and
     * `square` reads as the decoration's spill: a shape off its square, of the king's colour or
     * none. The badge of a mate spills over the square the king came from.
     */
    private fun kingLeaves(board: Board, move: Move, square: Int, obs: Observation): Boolean {
        if (move.from != square || square !in obs) return false
        if (!kingLands(board, move, move.to, obs)) return false
        val reading = obs.getValue(square)
        val piece = board.pieceAt(square)!!
        return reading.kind == Kind.PIECE && !reading.settled &&
            (reading.colour == null || reading.colour == Side.of(piece.colour))
    }

    /**
     * Whether every decorated landing square of a story holds a king in check at its end.
     */
    private fun checkedKings(after: Board, decorated: Set<Int>): Boolean {
        if (decorated.isEmpty()) return true
        return after.isCheck() && after.king(after.turn) in decorated
    }

    /**
     * Index of the earlier position of this game the frame shows, if any. Every position back
     * to the base is checked, one by one, however long the game: a click anywhere in the move
     * list shows one. Only the last forty were, and the base; a click further back read as a
     * position no move explains and was adopted by a resync -- the game numbered again from
     * it, or, after the end, a new game begun -- and stepping forward through the list then
     * recorded the game's own moves a second time (the state review, 2026-10-01).
     */
    private fun historyIndex(obs: Observation, diff: Set<Int>): Int? {
        val earlier = board.copy()
        val undone = HashSet<Int>()
        val plies = board.moveStack.size
        for (back in 1..plies) {
            val move = earlier.pop()
            undone.addAll(touchedSquares(earlier, move))
            if (undone.containsAll(diff) && matches(earlier, obs, obs.keys)) return plies - back
        }
        return null
    }

    private fun boardAt(index: Int): Board {
        val earlier = board.copy()
        while (earlier.moveStack.size > index) earlier.pop()
        return earlier
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
        // Pictures a story may need that no frame showed: one between frames that follow each
        // other (a bot answering inside a frame), as many as a catch-up may span after a gap --
        // the board unseen, or a frame long in coming. Three events in a tenth of a second is a
        // story about cells nobody could read, not a game.
        val last = frameTs
        val gap = unseen > 0 || (last != null && ts - last > GAP_S)
        maxUnseen = if (gap) maxPlies else 1
        frameTs = ts
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
                !decoratedKing(squareAt(i / 8, i % 8, bottom.color), reading)
        }
        var obs = observe(readings)
        if (obs.size < minReadableCells) return events
        var diff: Set<Int> = disagreements(obs)
        // What stays wrong is counted against the picture the likeliest story draws, not the
        // record: a premove queue drawn for ten seconds is the story, not stale junk.
        var lead = leadPicture()
        var off: Set<Int> = disagreements(obs, lead)
        advanceStale(off, obs)
        val ghosts = ghosts(off, obs, lead)
        if (ghosts.isNotEmpty()) {
            // An arrow drawn over an empty square reads as a piece of no known kind, off
            // centre, for as long as it is drawn. Once such a reading has gone stale it is
            // taken as unknown, so a few arrows neither stall the game nor multiply the stale
            // retries below.
            val trimmed = Observation()
            for ((sq, r) in obs) if (sq !in ghosts) trimmed[sq] = r
            obs = trimmed
            diff = diff - ghosts
            off = off - ghosts
            // Nor is an arrow a piece in someone's hand.
            blockedFlag = hoverSeen || obs.any { (sq, r) ->
                r.kind == Kind.PIECE && !r.settled && !decoratedKing(sq, r)
            }
        }
        lastObs = obs
        // A piece in the user's hand: the hover outline a site draws under it, or one of the
        // user's own pieces, known and off its square. A shape of no known kind off a square is
        // a decoration -- a badge, an arrow, a glow -- and no hand.
        dragging = hoverSeen || obs.values.any { r ->
            r.kind == Kind.PIECE && !r.settled && r.symbol != null && r.colour == userSide
        }
        if (dragging) {
            // Which piece the hand holds: the one of the user's read gone from its square.
            lead = leadPicture()
            val gone = obs.filter { (sq, r) ->
                r.kind == Kind.EMPTY && lead.pieceAt(sq)?.colour == userSide.color
            }.keys.toList()
            if (gone.size == 1) lifted = gone[0] to ts
        }

        if (finishedFlag) {
            events.addAll(afterTheEnd(obs, diff))
            return events
        }

        // A cell that has read the same wrong thing for a while (a badge, a ring) is left out of
        // the matching: it no longer stalls every story, as it witnesses no move. A few only:
        // more than a badge or two is a board that changed, for a resync.
        var quiet = off.filter { stale.getValue(it).second > staleAfter }.toSet()
        if (quiet.size >= resyncCells) quiet = emptySet()
        var fitted = fitting(obs, diff, quiet = quiet)
        if (fitted.isNotEmpty()) {
            events.addAll(decide(fitted))
            return events
        }
        // Nothing fits the frame: the record is not what the screen shows. A drag hides what is
        // under the hand; the stories wait for the drop, as they are, but the frame is one no
        // story explains. Left uncounted, a new piece set read as pieces off their squares --
        // every frame a "drag" -- kept the record agreeing with a screen it could not read, and
        // the look change was never told (the look-change fixture, 2026-10-05).
        agreesFlag = false
        if (blockedFlag) {
            events.addAll(maybeResync(diff))
            return events
        }
        val staleCells = if (viewing == null) {
            off.filter { stale.getValue(it).second > staleAfter }.toSet()
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
            val soft = obs.filter { (_, r) -> r.junk }.keys.toSet()
            val subsets = staleSubsets(staleCells, obs)
            // Soft wildcards only go with junk cells: ignoring a clean reading as well would
            // let the search invent a story through the cells nobody can read.
            val attempts = ArrayList<Pair<Set<Int>, Set<Int>>>()
            for (sub in subsets) if (sub.all { obs.getValue(it).junk }) attempts.add(sub to (sub + soft))
            for (sub in subsets) attempts.add(sub to sub)
            val tried = HashSet<Pair<Set<Int>, Set<Int>>>()
            for ((ignored, wild) in attempts) {
                if (!tried.add(ignored to wild)) continue
                if (((off - ignored) - wild).isEmpty()) {
                    // A story must rest on a disagreement read cleanly -- a square read empty
                    // or a settled piece of a known kind. When every cell left to explain is
                    // a wildcard, nothing on the board says a move was played: a board locked
                    // 0.28 of a cell off (it had slid on after the lock was taken) read every
                    // piece unsettled, the top of the king on d2 read as a piece on d3, and
                    // the king's move there was told, d2 and d3 both wildcards (the desktop's
                    // geometry sweep, 2026-09-27, seed 97).
                    continue
                }
                val denied = Observation()
                for (sq in ignored) {
                    val r = obs.getValue(sq)
                    if (r.exact && !r.junk) denied[sq] = r
                }
                fitted = fitting(obs, diff, wild = wild, denied = denied)
                if (fitted.isNotEmpty()) {
                    events.addAll(decide(fitted, cautious = true))
                    return events
                }
            }
            if (staleCells == off && staleCells.size < resyncCells) {
                // Only stale garbage differs from the latest position, unless a drag is what
                // kept a real move from committing while its cells turned stale.
                viewing = null
                unexplainedFrames += 1
                return events
            }
        }
        // A piece of a position read from the screen was of another kind; or, last of all, the
        // side to move was told wrong and the other side has moved.
        fitted = fitting(obs, diff, roots = misreadRoots(obs, diff))
        if (fitted.isEmpty()) {
            fitted = fitting(obs, diff, roots = listOf(Story(head = listOf(HeadOp.Flip), cost = 3.0)))
        }
        if (fitted.isNotEmpty()) {
            events.addAll(decide(fitted, cautious = true))
            return events
        }

        // New game: the start position is back on the board.
        if (inProgress && obs.size >= 48 && matches(Board(), obs, obs.keys)) {
            events.addAll(restart())
            return events
        }

        val adopted = maybePositionResync(obs, diff)
        if (adopted.isNotEmpty()) { events.addAll(adopted); return events }
        events.addAll(maybeResync(diff))
        return events
    }

    /**
     * Whether a move is seen played (`_witness`): every square it changes is read -- the square
     * it lands on above all -- nothing on them is in motion or under a drawing, and no hover
     * outline is anywhere. A king may land on a square read as a piece of no known kind: the
     * king under what the site draws on it ([kingLands]). Wild cells (a stale badge) witness
     * anything. While the user drags a piece, only the opponent moves: a hand holding a piece
     * plays nothing else, and the opponent's move is seen from its own cells, away from the
     * cursor -- a bot answering while the user drags the next premove.
     */
    private fun witness(obs: Observation, wild: Set<Int>, quiet: Set<Int> = emptySet()): (Board, Move) -> Boolean =
        fun(board: Board, move: Move): Boolean {
            if (dragging && board.turn == userSide.color) return false
            for (sq in touchedSquares(board, move)) {
                if (sq in wild) continue
                val reading = obs[sq]
                if (reading == null) {
                    if (sq == move.to) return false  // seen where it landed, or not seen at all
                    continue  // a square left, unread (an arrow over it): no contradiction
                }
                if (sq in quiet || reading.hover) return false
                if (kingLands(board, move, sq, obs) || kingLeaves(board, move, sq, obs)) {
                    continue  // the king under what the site draws on it: glow, badge, label
                }
                if (sq in drawn || reading.unsettled) return false
            }
            return true
        }

    /**
     * The stories that draw this frame (docs/reconciler-model.md, section 5; `_fitting`): each
     * story kept, and the record itself, extended by the cheapest events that fit; the record
     * shown at an earlier ply (browsing); a move played from the ply shown (a takeback).
     * [quiet] cells are left out of the matching but witness no move; [wild] cells carry no
     * evidence at all, and a move may pass through them (stale badges); [denied] readings among
     * them must not be contradicted by a square a story changes; [roots], when given, are the
     * only stories extended.
     */
    private fun fitting(
        obs: Observation,
        diff: Set<Int>,
        wild: Set<Int> = emptySet(),
        quiet: Set<Int> = emptySet(),
        denied: Observation? = null,
        roots: List<Story>? = null,
    ): List<Story> {
        val ignore = { sq: Int, reading: CellReading ->
            sq in wild || sq in quiet || sq in drawn || reading.hover ||
                (reading.kind == Kind.PIECE && !reading.settled)
        }
        val search = Search(
            record = board,
            user = userSide.color,
            obs = obs,
            ignore = ignore,
            witness = witness(obs, wild, quiet),
            lifted = dragging,
            liftable = liftable(),
            maxPlies = maxPlies,
            maxSteps = maxQueueSteps,
            budget = searchBudget,
            maxUnseen = maxUnseen,
        )
        var from = roots
        if (from == null) {
            // The record itself is a root too: a story built on a frame that caught a piece
            // mid-slide is told again from it. Told from the record, the frames the stories ahead
            // of it explained -- their moves, the queue they drew -- go unexplained, and that
            // costs a picture.
            val fresh = fresh()
            val all = ArrayList(stories)
            all.add(Story(cost = fresh))
            if (turnUnknownFlag) {
                val flipped = board.copy(withStack = false)
                if (turnValid(flipped, !board.turn)) all.add(Story(head = listOf(HeadOp.Flip), cost = fresh))
            }
            from = all
        }
        var found = ArrayList<Story>()
        val seenRoots = HashSet<Pair<List<Any>, List<Move>>>()
        // The cheapest stories first; each found sets a ceiling on what the rest may cost (the
        // margin kept by), so a story that could only end up dearer is not searched through --
        // with five premove stories alive, frames took a third of a second.
        var ceiling = Double.POSITIVE_INFINITY
        val queued = stories.any { it.queue.isNotEmpty() || it.premoved.isNotEmpty() }
        for (story in from.sortedWith(compareBy({ it.cost }, { it.prior }))) {
            if (story.view != null || story.cost > ceiling) continue
            if (!seenRoots.add(story.ops to story.queue)) continue
            val more = search.extend(story, ceiling)
            found.addAll(more)
            if (more.isNotEmpty()) {
                // As charged: a slide's stop costs its two pictures here too.
                val cheapest = found.minOf { landed(it, agreed, queued).cost }
                ceiling = minOf(ceiling, cheapest + (if (cheapest > 0) 1.0 else 0.0))
            }
        }

        fun weigh(list: List<Story>): List<Story> =
            list.filter { decorationsHold(it, obs) }.map { weighed(landed(it, agreed, queued), obs) }

        // Weighed before browsing is looked for: a takeback's pieces caught sliding back read as
        // moves from the ply shown, and only the slide's charge lets the browsing in (the
        // takebacks pressed quickly, 2026-10-05).
        found = ArrayList(weigh(found))
        val plain = found.any { it.cost == 0.0 && it.prior == 0.0 && it.queue.isEmpty() }
        if (diff.isNotEmpty() && !plain && !blockedFlag) {
            // The move list browsed: an earlier ply of the record, as it stands.
            val index = if (board.moveStack.isNotEmpty()) historyIndex(obs, diff) else null
            if (index != null) found.add(Story(view = index, cost = fresh()))
            // A move played from the ply shown: a takeback, or, from the start position of a
            // game begun there, a new game. An earlier position of the game is browsing: a
            // different move from the one shown is the only thing a takeback shows.
            for (story in stories) {
                val view = story.view
                if (view == null || index != null) continue
                val head = if (view == 0 && baseFenValue == STARTING_FEN) {
                    listOf<HeadOp>(HeadOp.NewGame)
                } else {
                    listOf<HeadOp>(HeadOp.Back(board.moveStack.size - view))
                }
                found.addAll(weigh(search.extend(Story(head = head)).filter { it.moves.isNotEmpty() }))
            }
        }
        if (!denied.isNullOrEmpty()) found = ArrayList(found.filter { spares(it, denied) })
        // A piece corrected is one the story moves: its move is what says what it was.
        found = ArrayList(
            found.filter { s ->
                s.head.all { op -> op !is HeadOp.Fix || s.moves.any { it.from == op.square } }
            },
        )
        // A side to move read the other way is told by a move played, never by a queue alone: a
        // rook sliding h1-h4 to take the checking queen, caught on h2 and h3, was read as the turn
        // wrong and the rook premoved, and the side to move was flipped (the desktop's stress
        // sweep, 2026-10-05, seed 116).
        found = ArrayList(found.filter { s -> s.moves.isNotEmpty() || HeadOp.Flip !in s.head })
        if (found.isEmpty()) return emptyList()
        // The same story found from two roots is one; within a margin of the cheapest only.
        val best = LinkedHashMap<Triple<List<Any>, List<Move>, Int?>, Story>()
        for (story in found) {
            val ident = Triple(story.ops, story.queue, story.view)
            val had = best[ident]
            if (had == null || story.cost < had.cost) best[ident] = story
        }
        // When the cheapest story needs no picture the frames did not show, the frames showed
        // everything, and a story that needs one did not happen: after e2-e4, "e2-e3, then e4
        // premoved" draws the same pieces from a picture nobody saw. When even the cheapest needs
        // one, frames were missed, and a story needing one more is as possible: kept, so that
        // nothing is told that only one of them says (the side to move above all).
        val top = best.values.minWith(compareBy({ it.cost }, { it.prior }))
        // Likelihood never picks a side to move (Tai, 2026-10-05: "the turn order ... should
        // never fail"): an equally seen story that leaves the other side to move is kept,
        // however much less likely, until a move or a cancel tells them apart. Not for ever: a
        // story that needs `TURN_PRIOR_GAP` more premove assumptions than the likeliest is let
        // go, or the side to move would stay unknown for good.
        val turn = top.board(board).turn
        val kept = best.values.filter { s ->
            (s.cost == top.cost && s.prior == top.prior) ||
                (s.cost == top.cost && s.board(board).turn != turn && s.prior < top.prior + TURN_PRIOR_GAP) ||
                (top.cost > 0 && s.cost <= top.cost + 1.0 && s.prior <= top.prior)
        }.sortedWith(STORY_ORDER)
        return kept.take(MAX_STORIES)
    }

    /**
     * What a story told afresh from the record costs (`_fresh`): a picture when the stories kept
     * had moves or a queue the frames before showed. A premove queue walked e5-d6-e6 drew the
     * board as it was a ply before, and browsing, told afresh, took the next step for a takeback
     * (the premove sweep, 2026-10-05, seed 16).
     */
    private fun fresh(): Double = if (stories.any { it.ops.isNotEmpty() || it.queue.isNotEmpty() }) 1.0 else 0.0

    /**
     * [story] with what the cells read off their squares says of it (`_weighed`): a piece of a
     * known kind off its square is a piece moving, or one under something drawn -- no evidence
     * of what stands there, but some that something does. A story drawing the square empty is the
     * less likely of two that fit otherwise alike ([Story.prior]).
     */
    private fun weighed(story: Story, obs: Observation): Story {
        val view = story.view
        val picture = if (view != null) boardAt(view) else premovePicture(story.board(board), story.queue)
        val hints = obs.count { (sq, r) ->
            r.kind == Kind.PIECE && !r.settled && r.symbol != null && picture.pieceAt(sq) == null
        }
        val more = SHAPE_PRIOR * hints + (if (otherHand(story)) EDIT_PRIOR else 0.0)
        return if (more != 0.0) story.copy(prior = story.prior + more) else story
    }

    /**
     * Whether the user's move the story plays first is not the piece the frames just showed in
     * the user's hand (`_other_hand`). With both queens moved, one played and one premoved, the
     * pictures after are one and the same; the drag before them showed the queen on g5 lifted
     * while the one on h4 stood, so g5 is the one played (Tai's third game, 2026-09-14).
     */
    private fun otherHand(story: Story): Boolean {
        val hand = lifted ?: return false
        if (tsValue - hand.second > LIFT_MEMORY_S) return false
        val replay = Story(head = story.head).board(board).copy(withStack = false)
        for (move in story.moves) {
            if (replay.turn == userSide.color && move !in story.premoved) return move.from != hand.first
            if (!replay.isLegal(move)) return false
            replay.push(move)
        }
        return false
    }

    /**
     * Whether every king the story lands on a square read as a piece of no known kind
     * ([kingLands]) is in check when the story ends (`_decorations_hold`): a site decorates a
     * king in check, and a king moves into no check of its own, so the decoration is the reply's.
     */
    private fun decorationsHold(story: Story, obs: Observation): Boolean {
        val end = story.board(board)
        val decorated = HashSet<Int>()
        val replay = end.copy()
        for (move in story.moves.asReversed()) {
            if (replay.moveStack.isEmpty()) break
            replay.pop()
            if (kingLands(replay, move, move.to, obs)) decorated.add(move.to)
        }
        return checkedKings(end, decorated)
    }

    /**
     * Whether a story leaves alone, or agrees with, every exact reading among the cells ignored
     * as stale (`_spares`): a stale empty square is no place to land a queen.
     */
    private fun spares(story: Story, denied: Observation): Boolean {
        val after = premovePicture(story.board(board), story.queue)
        return denied.all { (sq, reading) ->
            after.pieceAt(sq) == board.pieceAt(sq) || cellAgrees(reading, after.pieceAt(sq))
        }
    }

    /**
     * What the record does with the stories that fit (docs/reconciler-model.md, section 5,
     * steps 3 and 4; `_decide`). When every story leads to one key -- one position, one side to
     * move, one queue, one ply shown -- they differ only in how they got there, and the cheapest
     * is the record's to follow. Otherwise the record follows only the ops every story begins
     * with. An op is played once it has held [settleFrames] frames, one more when it came with
     * others in one frame or from the stale retry. A wrong side to move is never told.
     */
    private fun decide(found: List<Story>, cautious: Boolean = false): MutableList<Event> {
        var fitted = found
        if (fitted.size == 1 && (fitted[0].cost != 0.0 || fitted[0].prior != 0.0)) {
            fitted = listOf(fitted[0].copy(cost = 0.0, prior = 0.0))  // alone, its past is the past
        }
        stories = fitted
        val told = tellTurn()
        unexplainedFrames = 0
        unexplainedNoted = false
        agreesFlag = fitted.any { it.ops.isEmpty() && it.view == null }
        queueTs = if (fitted.any { it.queue.isNotEmpty() }) tsValue else null
        queueFrames = if (queueTs != null) queueFrames + 1 else 0
        val keys = fitted.map { it.key(board) }.toSet()
        // One key is one future: the position, the side to move, the queue and the ply shown are
        // certain, and the record follows the simplest story to them (Tai, 2026-10-05: the side
        // to move first, then the position current, then the moves themselves).
        val ops: List<Any>
        if (keys.size == 1) {
            val chosen = fitted[0]
            ops = chosen.ops
            if (ops.isEmpty()) {
                agreed = emptyList()
                if (chosen.view == null || !blockedFlag) viewing = chosen.view
                return told
            }
        } else {
            ops = commonStart(fitted.map { it.ops })
        }
        viewing = null
        // A frame confirms a move only when the squares it changed are seen settled in it: a
        // frame with a piece in motion keeps the stories, and confirms nothing it cannot see
        // (the bishop caught on c5 sliding d6-b4 was told as Bc5 on the slide's next frame: the
        // stress sweep, 2026-10-05, seed 1).
        val seen = seenOps(ops)
        val next = ArrayList<Agreed>()
        for ((i, op) in ops.withIndex()) {
            if (i >= agreed.size || agreed[i].op != op) break
            next.add(Agreed(op, agreed[i].count + seen[i], agreed[i].need))
        }
        val fresh = ops.drop(next.size)
        val plies = fresh.count { it is Move }
        val need = settleFrames + (if (plies > 1) 1 else 0) + (if (cautious) 1 else 0)
        val done = next.size
        for ((i, op) in fresh.withIndex()) next.add(Agreed(op, seen[done + i], need))
        agreed = next
        var ready = 0
        while (ready < next.size && next[ready].count >= next[ready].need) ready += 1
        if (ready == 0) return told
        val events = play(ops.take(ready))
        told.addAll(events)
        told.addAll(tellTurn())
        return told
    }

    /**
     * For each op, 1 when this frame shows it settled (`_seen_ops`): a move whose squares are
     * all read, none in motion, under the cursor or under a drawing (a king under its decoration
     * aside); what is done to the record before the moves always. Nor does a frame that shows
     * the moved piece off its square further along the move's line ([slidingOn]).
     */
    private fun seenOps(ops: List<Any>): List<Int> {
        val obs = lastObs
        val head = ops.filterIsInstance<HeadOp>()
        val replay = Story(head = head).board(board).copy(withStack = false)
        val out = ArrayList<Int>()
        for (op in ops) {
            if (op !is Move) {
                out.add(1)
                continue
            }
            val clean = touchedSquares(replay, op).all { settledOn(replay, op, it, obs) } &&
                !slidingOn(replay, op, obs)
            out.add(if (clean) 1 else 0)
            if (replay.isLegal(op)) replay.push(op)
        }
        return out
    }

    /**
     * Whether the frame reads the moved piece off its square further along the move's line
     * (`_sliding_on`): still sliding on, the square it seems to have stopped on a frame of its
     * way. A pawn sliding a7-a5 was caught on a6, told as a6 on the next frame with the pawn
     * still on its way, and a5 then read as a premove nobody played (the desktop's stress sweep,
     * 2026-10-05, seed 69).
     */
    private fun slidingOn(board: Board, move: Move, obs: Observation): Boolean {
        val piece = board.pieceAt(move.from) ?: return false
        val files = squareFile(move.to) - squareFile(move.from)
        val ranks = squareRank(move.to) - squareRank(move.from)
        if (files != 0 && ranks != 0 && kotlin.math.abs(files) != kotlin.math.abs(ranks)) {
            return false  // a knight's jump: no line to go on along
        }
        val stepFile = files.sign
        val stepRank = ranks.sign
        var file = squareFile(move.to) + stepFile
        var rank = squareRank(move.to) + stepRank
        while (file in 0..7 && rank in 0..7) {
            val reading = obs[square(file, rank)]
            if (reading != null && reading.kind == Kind.PIECE && !reading.settled &&
                reading.symbol == piece.symbol()
            ) {
                return true
            }
            file += stepFile
            rank += stepRank
        }
        return false
    }

    /**
     * Whether [sq], a square [move] changes, is read and nothing on it is moving, under the
     * cursor or under a drawing -- a king under its decoration aside (`_settled_on`).
     */
    private fun settledOn(board: Board, move: Move, sq: Int, obs: Observation): Boolean {
        val reading = obs[sq] ?: return sq != move.to  // a square left may go unread; the landing may not
        if (reading.hover) return false
        if (kingLands(board, move, sq, obs) || kingLeaves(board, move, sq, obs)) return true
        return sq !in drawn && !reading.unsettled
    }

    /** [TurnPending] when whether the stories agree on the side to move has changed (`_tell_turn`). */
    private fun tellTurn(): MutableList<Event> {
        val pending = turnPending
        if (pending == turnPendingTold) return mutableListOf()
        turnPendingTold = pending
        return mutableListOf(TurnPending(gameId = gameId, pending = pending))
    }

    /** The record does [ops], and every story is told from it (`_play`). */
    private fun play(ops: List<Any>): MutableList<Event> {
        val before = stories
        val held = agreed
        val events = mutableListOf<Event>()
        for (op in ops) {
            when (op) {
                is Move -> {
                    // The first move played settles a side to move left open: who played it.
                    turnUnknownFlag = false
                    events.addAll(commit(op))
                }
                is HeadOp.NewGame -> events.addAll(restart())
                is HeadOp.Back -> {
                    repeat(op.plies) { board.pop() }
                    events.add(Takeback(gameId = gameId, plies = op.plies, fen = board.fen()))
                    events.add(PositionChanged(gameId = gameId, fen = board.fen(), ply = board.moveStack.size))
                }
                is HeadOp.Flip -> events.addAll(flip())
                is HeadOp.Fix -> events.addAll(fix(op.square, op.kind))
            }
        }
        val done = ops.size
        stories = before.filter { it.ops.take(done) == ops }.map { it.rebased(done) }.ifEmpty { listOf(Story()) }
        agreed = held.drop(done)
        stale.clear()
        viewing = null
        return events
    }

    /**
     * The record's side to move read the other way (`_flip`): the first move of a position
     * adopted with the turn open settles it, silently; a turn told wrong is told again.
     */
    private fun flip(): MutableList<Event> {
        val flipped = board.copy(withStack = false)
        flipped.turn = !flipped.turn
        flipped.epSquare = null
        if (turnUnknownFlag) {
            board.turn = flipped.turn
            board.epSquare = null
            baseFenValue = board.fen()
            turnUnknownFlag = false
            return mutableListOf()
        }
        val read = readSquares
        val since = lastCommitTs
        setPosition(flipped)
        readSquares = read
        lastCommitTs = since
        return resynced("side to move corrected: ${Side.of(flipped.turn).name.lowercase()} to move")
    }

    /**
     * A piece of a position read from the screen, never moved since, was of another kind
     * (`_fix`): the position is adopted again with it corrected. Still the same game: the pieces
     * that had moved stay known, and the next move's think time counts from the last move.
     */
    private fun fix(square: Int, kind: Int): MutableList<Event> {
        val was = board.pieceAt(square)!!
        val fixed = board.copy(withStack = false)
        fixed.setPieceAt(square, Piece(kind, was.colour))
        val read = readSquares - square
        val since = lastCommitTs
        setPosition(fixed)
        readSquares = read
        lastCommitTs = since
        return resynced(
            "the ${MISREAD_NAMES[was.pieceType]} read on ${squareName(square)} was a ${MISREAD_NAMES[kind]}",
        )
    }

    /**
     * After the game has ended the board is a review board (`_after_the_end`): browsing is
     * followed, and moves played on it are analysis, not the game. Only the start position
     * begins the next game, or a position far enough away to be adopted by a resync.
     */
    private fun afterTheEnd(obs: Observation, diff: Set<Int>): MutableList<Event> {
        if (diff.isEmpty()) {
            viewing = null
            return mutableListOf()
        }
        val index = if (board.moveStack.isNotEmpty()) historyIndex(obs, diff) else null
        if (index != null) {
            // The start position after a finished game is the next game, not browsing.
            if (index == 0 && baseFenValue == STARTING_FEN) return start()
            if (!blockedFlag) viewing = index
            return mutableListOf()
        }
        if (inProgress && obs.size >= 48 && matches(Board(), obs, obs.keys)) return restart()
        return maybePositionResync(obs, diff)
    }

    /** Whether the screen is ahead of the record: a story with moves still to be told. */
    val pendingStory: Boolean get() = stories.any { it.ops.isNotEmpty() }

    /**
     * Whether the last readable frame shows the record as it stands, a premove queue drawn on it
     * aside: the record is up to date with the screen.
     */
    val agrees: Boolean get() = agreesFlag

    /**
     * Whether the stories disagree on the side to move: the record is behind the screen and
     * cannot yet say whose move it is.
     */
    val turnPending: Boolean get() = stories.map { it.board(board).turn }.toSet().size > 1

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
     * The story every story agrees with, when it ends the game: the mate the site's dialog then
     * covered before a second clean frame could confirm it.
     */
    fun pendingTerminal(): Pair<List<Move>, Board>? {
        if (stories.isEmpty()) return null
        val keys = stories.map { it.key(board) }.toSet()
        val story = stories[0]
        if (keys.size != 1 || story.head.isNotEmpty() || story.moves.isEmpty()) return null
        val end = story.board(board)
        if (!end.isGameOver()) return null
        return story.moves to end
    }

    /** Commit the story that ends the game (see [pendingTerminal]). */
    fun commitPendingTerminal(): MutableList<Event> {
        val story = pendingTerminal() ?: return mutableListOf()
        return play(story.first)
    }

    /**
     * Stories in which one piece of a position read from the screen, never moved since, is of
     * another kind (`_misread_roots`): the piece has left its square, so the frame cannot say
     * what it was, and the move it made says. Without this the game stood on a piece that could
     * not make the move it made, and every move after it went unread until the board was adopted
     * afresh (Tai's puzzle, 2026-10-03: a knight on h4 read as a bishop; Nxg6, hxg6 and Bf3 lost).
     */
    private fun misreadRoots(obs: Observation, diff: Set<Int>): List<Story> {
        if (viewing != null || turnUnknownFlag || diff.size > misreadMaxCells) return emptyList()
        val roots = ArrayList<Story>()
        for (sq in diff.filter { it in readSquares }.sorted()) {
            val piece = board.pieceAt(sq) ?: continue
            if (piece.pieceType == KING) continue
            val reading = obs[sq]
            if (reading == null || reading.kind != Kind.EMPTY) continue
            for (kind in intArrayOf(PAWN, KNIGHT, BISHOP, ROOK, QUEEN)) {
                if (kind == piece.pieceType) continue
                if (kind == PAWN && (squareRank(sq) == 0 || squareRank(sq) == 7)) continue
                val trial = board.copy(withStack = false)
                trial.setPieceAt(sq, Piece(kind, piece.colour))
                if (turnValid(trial, trial.turn)) roots.add(Story(head = listOf(HeadOp.Fix(sq, kind)), cost = 2.0))
            }
        }
        return roots
    }

    /** Stale readings of something unnameable on squares the picture has empty (`_ghosts`). */
    private fun ghosts(off: Set<Int>, obs: Observation, picture: Board): Set<Int> =
        off.filter { sq ->
            stale.getValue(sq).second > staleAfter &&
                obs.getValue(sq).kind == Kind.PIECE &&
                obs.getValue(sq).symbol == null &&
                picture.pieceAt(sq) == null
        }.toSet()

    /**
     * Where the user's pieces stood before this frame, as the likeliest story draws them
     * (`_liftable`): the squares a piece in the hand can have come from.
     */
    private fun liftable(): Set<Int> =
        squaresOf(leadPicture().occupiedCo[if (userSide.color) 1 else 0]).toSet()

    /**
     * What the likeliest story draws (`_lead_picture`): its position with its queue played out,
     * or the ply of the record it shows.
     */
    private fun leadPicture(): Board {
        val story = stories.firstOrNull() ?: Story()
        val view = story.view
        if (view != null) return boardAt(view)
        return premovePicture(story.board(board), story.queue)
    }

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
        val queued = queueTs
        if (queued != null && tsValue - queued < QUEUE_RESYNC_S) {
            // A premove queue drawn: the picture is the position with the queued pieces at the
            // ends of their queues, not a position. Adopted, it was the queue's end taken for
            // the game (Tai's game, 2026-09-28: b6-b1=Q premoved; the moves till then lost).
            // Once the queue has resolved it is a position again.
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
        val placement = loosePlacement(obs).boardFen() + "|" +
            highlightedSquares(obs).sorted().joinToString(",") { squareName(it) }
        val candidate = resyncCandidate
        resyncCandidate = if (candidate != null && candidate.first == placement) {
            Triple(placement, candidate.second + 1, candidate.third)
        } else {
            Triple(placement, 1, tsValue)
        }
        val (_, frames, since) = resyncCandidate!!
        if (frames >= briefFrames) {
            // Remembered as a board shown: the start a move may have been played from.
            val seen = Seen(loosePlacement(obs), highlightedSquares(obs), placement, tsValue)
            resyncSeen = if (resyncSeen.isNotEmpty() && resyncSeen.last().key == placement) {
                resyncSeen.dropLast(1) + seen
            } else {
                resyncSeen.takeLast(briefKept - 1) + seen
            }
        }
        val held = frames >= resyncHoldFrames ||
            (diff.size >= resyncJumpCells && frames >= resyncMinFrames && tsValue - since >= resyncHoldS)
        if (!held) return mutableListOf()

        val adopted = adoptPlacement(loosePlacement(obs), bottom.color, highlightedSquares(obs))
        if (adopted == null) {
            resyncCandidate = null
            return mutableListOf()
        }
        val (adoptedBoard, unknownTurn) = adopted
        turnUnknownFlag = unknownTurn
        val first = openingMove(adoptedBoard, placement, since, highlightedSquares(obs))
        resyncSeen = emptyList()
        clearStory()
        viewing = null
        resyncCandidate = null
        if (first != null) {
            // The brief board before it is the start, and the move between them is played.
            val (start, move) = first
            turnUnknownFlag = false
            val events: MutableList<Event> = if (finishedFlag) {
                (mutableListOf<Event>(begin(start, "position")) + positioned()).toMutableList()
            } else {
                setPosition(start)
                resynced("position adopted after ${diff.size} unexplained cells; its first move seen")
            }
            events.addAll(commit(move))
            return events
        }
        if (finishedFlag) {
            // The finished game is over for good; a position that appears after it is another
            // game, seen from the middle -- or a puzzle, or an analysis board. Its start is all
            // there is to say: a Resync after it as well had the engine measure the position
            // twice and the coach note a re-read board that was never read before (the state
            // review, 2026-10-01).
            val events = mutableListOf<Event>(begin(adoptedBoard, "position"))
            events.addAll(positioned())
            return events
        }
        setPosition(adoptedBoard)
        var reason = "position adopted after ${diff.size} unexplained cells"
        if (turnUnknownFlag) reason += "; side to move pending"
        return resynced(reason)
    }

    /**
     * The board shown just before the one being adopted, and the one legal move from it to this
     * one -- a new puzzle's first board and the site's opening move on it -- or null. The
     * earlier board must have held [briefFrames] clean frames, ended at most [briefRecentS]
     * before this one began, and be one move away. `_opening_move`.
     */
    private fun openingMove(after: Board, placement: String, since: Double, lit: Set<Int>): Pair<Board, Move>? {
        // The newest of the boards kept that one move joins: a frame caught mid-slide, repeated by
        // the still ticks, was kept, and as the only earlier board it stood in for the first one.
        for (prior in resyncSeen.asReversed()) {
            if (prior.key == placement || since - prior.ts > briefRecentS) continue
            oneMove(prior.placed, prior.highlighted, after.boardFen(), lit)?.let { return it }
        }
        return null
    }

    /**
     * The position [placed] shows and the one legal move from it to the placement [target],
     * when exactly one move by either side gets there. `_one_move`.
     */
    private fun oneMove(placed: Board, highlighted: Set<Int>, target: String, lit: Set<Int> = emptySet()): Pair<Board, Move>? =
        oneMoveBetween(placed, highlighted, target, bottom.color, lit)

    companion object {
        /**
         * [oneMove] for any caller (`Reconciler.one_move`): the pipeline asks it which of the boards
         * it read before is the one a move joins to the board it adopts. [lit]: the squares the site
         * lights on the board after the move, its last move's -- when two are lit, the move must be
         * theirs. Halfway through a pawn's double step the pawn stands centred on the square between,
         * a board of its own, and e6-e5 joined it to the board after e7-e5 (the puzzle sweep,
         * 2026-10-03): e7 and e5 lit say it is not the move.
         */
        fun oneMoveBetween(
            placed: Board,
            highlighted: Set<Int>,
            target: String,
            prefer: Boolean,
            lit: Set<Int> = emptySet(),
        ): Pair<Board, Move>? {
            val adopted = adoptPlacement(placed, prefer, highlighted) ?: return null
            val found = LinkedHashMap<Pair<Boolean, Move>, Board>()
            for (turn in listOf(WHITE, BLACK)) {
                val base = adopted.first.copy(withStack = false)
                if (!turnValid(base, turn)) continue
                // A board read from the screen cannot say en passant is open; each square it could
                // be open on is tried as well (d5xe6 e.p. as a puzzle's opening move was no move at
                // all: the puzzle sweep, 2026-10-03).
                for (ep in listOf<Int?>(null) + enPassantSquares(base)) {
                    val start = base.copy(withStack = false)
                    start.epSquare = ep
                    for (move in start.legalMoves()) {
                        if (lit.size == 2 && !lights(start, move, lit)) continue
                        start.push(move)
                        val hit = start.boardFen() == target
                        start.pop()
                        if (hit) found.getOrPut(turn to move) { start.copy(withStack = false) }
                    }
                }
            }
            // Exactly one move, or the jump is not read as one.
            if (found.size != 1) return null
            val (key, start) = found.entries.first()
            return start to key.second
        }
    }

    /**
     * A new position for the game: Resync and PositionChanged, and the end when the position
     * is over and the side to move known (`_resynced`).
     */
    private fun resynced(reason: String): MutableList<Event> {
        val events = mutableListOf<Event>(Resync(gameId = gameId, fen = board.fen(), reason = reason))
        events.addAll(positioned())
        return events
    }

    /**
     * The position the game now stands on, numbered from it: PositionChanged, and the end
     * when the position is over and the side to move known (`_positioned`).
     */
    private fun positioned(): MutableList<Event> {
        val events = mutableListOf<Event>(PositionChanged(gameId = gameId, fen = board.fen(), ply = 0))
        if (!turnUnknownFlag && board.isGameOver()) {
            finishedFlag = true
            events.add(endedEvent(board.result(), "game over on board"))
        }
        return events
    }

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
        return resynced("side to move settled by the clock: ${Side.of(colour).name.lowercase()}")
    }

    /**
     * Begin a game from an arbitrary observed position -- a puzzle or a mid-game board read
     * cold from the theme cache -- rather than from the start. `ts` is when the position was
     * read, so the first move's think time counts from it.
     *
     * [before]: a placement read moments earlier, and its highlighted squares. When exactly one
     * legal move takes it to this one -- a new puzzle's first board and the site's opening move,
     * played before the read could confirm the first -- the game begins from it and the move is
     * played ([oneMove]).
     */
    fun startFrom(
        placement: Board,
        ts: Double? = null,
        highlighted: Set<Int> = emptySet(),
        before: Pair<Board, Set<Int>>? = null,
    ): MutableList<Event> {
        val adopted = adoptPlacement(placement, bottom.color, highlighted) ?: return mutableListOf()
        if (ts != null) tsValue = ts
        if (before != null) {
            val first = oneMove(before.first, before.second, placement.boardFen(), highlighted)
            if (first != null) {
                val (start, move) = first
                turnUnknownFlag = false
                val events = mutableListOf<Event>(begin(start, "position"))
                events.addAll(commit(move))
                return events
            }
        }
        val (adoptedBoard, unknownTurn) = adopted
        turnUnknownFlag = unknownTurn
        return mutableListOf(begin(adoptedBoard, "position"))
    }

    // Settling and committing

    /**
     * Stale cells to ignore, singles before pairs, junk-looking readings first. At most
     * `resyncCells - 1` cells are ever ignored together: more than that is no longer a badge
     * or two, and only a position resync may act.
     */
    private fun staleSubsets(staleCells: Set<Int>, obs: Observation): List<Set<Int>> {
        val junk = staleCells.filter { obs.getValue(it).junk }.sorted()
        val clean = staleCells.filter { !obs.getValue(it).junk }.sorted()
        val pool = if (staleCells.size > maxStaleSubsetCells) junk else junk + clean
        val subsets = ArrayList<Set<Int>>()
        for (sq in pool) subsets.add(setOf(sq))
        if (resyncCells > 2) {
            for (i in pool.indices) for (j in i + 1 until pool.size) subsets.add(setOf(pool[i], pool[j]))
        }
        return subsets
    }

    /**
     * Say so when several cells have stayed unexplained for a long time. The game is kept as it
     * stands -- a position resync is what adopts a new one -- so this is a note ([Unexplained]),
     * not a Resync, which everything downstream reads as the numbering starting again.
     */
    private fun maybeResync(diff: Set<Int>): MutableList<Event> {
        if (diff.isEmpty()) {
            unexplainedFrames = 0
            unexplainedNoted = false
            return mutableListOf()
        }
        unexplainedFrames += 1
        if (unexplainedFrames >= unexplainedNoteFrames && !unexplainedNoted) {
            // Early, for the log: what the reader sees on each square the game cannot explain,
            // while it is still fresh -- a dump's log said only the squares' names (Tai's puzzle
            // on the Pixel, 2026-10-03: 34 squares, and why was nowhere).
            unexplainedNoted = true
            return mutableListOf(
                Unexplained(gameId = gameId, fen = board.fen(), reason = "unexplained for now: ${describe(diff.sorted())}"),
            )
        }
        val persistent = diff.filter { val s = stale[it]; s != null && s.second > staleAfter }
        if (persistent.size < resyncCells || unexplainedFrames < resyncAfter) return mutableListOf()
        unexplainedFrames = 0
        return mutableListOf(
            Unexplained(
                gameId = gameId, fen = board.fen(),
                reason = "unexplained: ${describe(persistent.sorted())}", told = true,
            ),
        )
    }

    /**
     * Each square as read against the game: "b1 N off its square, game N", "e4 empty, game P",
     * "d5 piece of no kind, game -" (`_describe`).
     */
    private fun describe(squares: List<Int>): String = squares.joinToString("; ") { sq ->
        val reading = lastObs[sq]
        val read = when {
            reading == null -> "unread"
            reading.kind == Kind.EMPTY -> "empty"
            reading.kind != Kind.PIECE -> "unknown"
            else -> (reading.symbol ?: "piece of no kind") + if (!reading.settled) " off its square" else ""
        }
        "${squareName(sq)} $read, game ${board.pieceAt(sq)?.symbol() ?: "-"}"
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
        // Defensive: nothing is played on a finished game. A new game starts it over.
        if (finishedFlag) return mutableListOf()
        if (!board.isLegal(move)) {
            // Defensive: an explanation computed against another position must never corrupt
            // the game. Report it and let the next frames explain the board again.
            clearStory()
            return mutableListOf(
                Unexplained(
                    gameId = gameId, fen = board.fen(),
                    reason = "refused illegal commit ${move.uci()}",
                ),
            )
        }
        val fenBefore = board.fen()
        val san = board.san(move)
        val byUser = board.turn == userSide.color
        // Time since the previous commit, as seen on screen: a chain committed at once gives
        // it all to its first move and leaves the others open.
        var thinkMs: Int? = null
        val since = lastCommitTs
        // `int(round(...))`: Python rounds half to even.
        if (since != null && tsValue > since) thinkMs = pyRound((tsValue - since) * 1000.0)
        lastCommitTs = tsValue
        readSquares = readSquares - setOf(move.from, move.to)
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
            events.add(endedEvent(resultValue!!, "game over on board"))
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
        clearStory()
        resultValue = result
        return mutableListOf(endedEvent(result, reason))
    }

    /**
     * The game under way ended because the next one replaced it, before its result was seen
     * (`replaced_event`): its result is what its final position proves ([resultOnBoard]), and "*"
     * only when the position proves nothing (Tai, 2026-10-02).
     */
    fun replacedEvent(reason: String): GameEnded {
        val proved = resultOnBoard(board) ?: return endedEvent("*", reason)
        return endedEvent(proved.first, proved.second)
    }

    /**
     * The session ends with this game under way and its end never seen (`end_unseen`): it ends
     * with the result its final position proves, by the rule a replaced game's is set with
     * ([replacedEvent]). Nothing when the position proves none: the record keeps the game
     * unfinished ("*"), as it did (Tai, 2026-10-02).
     */
    fun endUnseen(): MutableList<Event> {
        if (!inProgress || finishedFlag) return mutableListOf()
        val proved = resultOnBoard(board) ?: return mutableListOf()
        return end(proved.first, proved.second)
    }

    /** The GameEnded of this game as it stands (`ended_event`): also the pipeline's, for a game it drops. */
    fun endedEvent(result: String, reason: String): GameEnded = GameEnded(
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
