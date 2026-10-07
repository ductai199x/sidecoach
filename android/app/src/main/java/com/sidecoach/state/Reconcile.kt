package com.sidecoach.state

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
 * past the record that the frames allow is kept as a story ([Story], `Stories.kt`): moves, the
 * user's premove queue as the site draws it, a ply of the move list shown, a piece on its way. A
 * story costs the pictures it needs that no frame showed; of the stories of the fewest, the ones
 * kept assume the fewest events (`_events`), and nothing is weighed. A piece caught on its way
 * and an opponent's move the drawn queue hides assume nothing the pictures deny, and stay beside
 * the rest. The record takes only what every story agrees on, and the side to move is told
 * pending while they disagree on it ([TurnPending]). When they differ only in what no picture
 * tells apart, the record follows the one the board shows as a position and keeps the others,
 * told again from it.
 *
 * Cell readings can be exact (a symbol), occupied (a piece whose type is hidden by a badge),
 * empty, or unknown; unknown cells carry no evidence either way. A move is played only after
 * `settleFrames` frames have shown every square it touches settled, none under something drawn
 * (an arrow), and no hover outline of a drag. No colour is read, highlight or tint, save the
 * site's last-move mark, and only to recover a record that every story has led astray
 * ([Reconciler.marked]).
 *
 * Browsing the move list is a story too: a frame that shows an earlier ply of a story's game is
 * that story with the ply shown, and nothing is played while the user browses. A takeback is a
 * different move played from the ply shown, the only way the two can be told apart.
 *
 * Cells that keep disagreeing without any move explaining them are treated as stale after a few
 * frames and ignored, so one unreadable cell cannot stall the game. Resync fires only when
 * several cells stay unexplained for a long time, which a long drag never triggers.
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
 * `reconcile._story_order`: stories cheapest first; of equal cost, a takeback or a new game
 * before moves that retell it (a takeback and the bot's quick reply, never four retreats), then
 * fewer moves. The sort is stable: what is left in the order the search found them, the order
 * the moves are generated in, so the story followed does not change from frame to frame.
 */
private val STORY_ORDER: Comparator<Story> = compareBy<Story>(
    { it.cost },
    { s -> if (s.head.any { it is HeadOp.Back || it is HeadOp.NewGame }) 0 else 1 },
    { it.ops.size },
    { it.queue.size },
)

/**
 * `_retold`: [story] told from a record that has played [played], which it does not begin with:
 * the moves the record played past their common start taken back, then the story's own. Null
 * when the record's part past the common start is no run of moves to take back, or the story's
 * own head lies past it.
 */
private fun retold(story: Story, played: List<Any>): Story? {
    val common = commonStart(listOf(story.ops, played)).size
    if (common < story.head.size) return null
    val tail = played.drop(common)
    if (tail.isEmpty() || !tail.all { it is Move }) return null
    val moves = story.ops.drop(common)
    check(moves.all { it is Move })
    return story.copy(head = listOf(HeadOp.Back(tail.size)), moves = moves.map { it as Move })
}

/**
 * `_events`: the events [story] assumes: what it does to the record, its moves, and a premove
 * queue as one -- where a queue ends is drawn, the way it went there never is (a king queued
 * h3-g3-h4 lost to h3-h4 by a step, and the first step fired put the king on the wrong square:
 * the flow matrix, 2026-10-06) -- and every step through the move list, a click as a move is:
 * free, a ply shown read the knights shuffled back to a position as the user browsing to it (the
 * repetition tests), and a takeback and a move from the ply as the user browsing, coming back and
 * premoving the move (the flow matrix, 2026-10-06, seed 8 puzzle).
 */
private fun events(story: Story): Int =
    story.head.size + story.moves.size + (if (story.queue.isNotEmpty()) 1 else 0) + story.steps

/** `_draws_nothing`: whether [story]'s premove queue, played out, draws the position itself. */
private fun drawsNothing(record: Board, story: Story): Boolean {
    val board = story.board(record)
    return premovePicture(board, story.queue).boardFen() == board.boardFen()
}

/**
 * `_on_way`: whether a piece sliding in a straight line from [a] to [b] passes over [sq] (or
 * lands on it): its centre within half a square of the line, past [a]. A knight's slide g1-f3
 * passes over the corners of g2 and f2.
 */
private fun onWay(a: Int, b: Int, sq: Int): Boolean {
    val ax = squareFile(a)
    val ay = squareRank(a)
    val dx = squareFile(b) - ax
    val dy = squareRank(b) - ay
    val px = squareFile(sq) - ax
    val py = squareRank(sq) - ay
    val length = dx * dx + dy * dy
    if (sq == a || length == 0) return false
    val t = (px * dx + py * dy).toDouble() / length
    if (!(0 < t && t <= 1)) return false
    val cross = px * dy - py * dx
    return (cross * cross).toDouble() <= 0.25 * length + 1e-9
}

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
     * The stories that fit the frames so far ([Story], docs/reconciler-model.md), and for each
     * run of ops a story begins with, the frames that have shown it settled ([decide]): counted
     * for every story, whether the stories agree yet or not.
     */
    // `internal` for `ReconcileRecordTest` (the desktop's tests/test_reconcile_record.py), which
    // sets the stories no frame can stage on cue.
    internal var stories: List<Story> = listOf(Story())
    internal var seen: Map<List<Any>, Int> = emptyMap()

    /** Whether the last frame showed the record as it stands (a queue drawn on it aside). */
    private var agreesFlag = true

    /**
     * Whether the listener (the coach) holds the side to move pending: the last [TurnPending]
     * said so, and no event since has told a position whole ([toldWhole]).
     */
    internal var turnPendingTold = false
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
    internal var resyncCandidate: Triple<String, Int, Double>? = null

    /**
     * The last [briefKept] distinct placements a resync held for [briefFrames] clean frames:
     * the placement, its highlighted squares, its view key, and when it was last seen. A new
     * puzzle's first board stands for well under a second before the site plays the opening
     * move, too briefly to be adopted; the board after the move was, and the move was lost
     * (Tai's puzzles, 2026-10-03: "the coach misses the first move ... 75-80% of the time").
     * Adopted, the board after is played from the brief one when one legal move joins them.
     * `_resync_seen`.
     */
    internal var resyncSeen: List<Seen> = emptyList()

    internal data class Seen(val placed: Board, val highlighted: Set<Int>, val key: String, val ts: Double)

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

    /** Whether a piece is in the user's hand this frame ([update]). */
    private var dragging = false

    /** When the last frame came, and how many unseen pictures a story may need this frame. */
    private var frameTs: Double? = null
    private var maxUnseen = 1
    private var lastObs: Observation = Observation()

    /** The frame before it ([inFlight]: a piece read on one square twice running has stopped). */
    internal var prevObs: Observation = Observation()

    // The site's last-move mark agreed with no story, nor with one told the other way, this
    // frame ([marked]), and for how many frames running ([markedResync]).
    internal var markDenied = false
    internal var unmarked = 0
    internal var unmarkedView: Pair<String, List<Int>>? = null
    private var tsValue = 0.0
    private var lastCommitTs: Double? = null

    /** When a frame last fitted a story with a premove queue drawn ([maybePositionResync]). */
    internal var queueTs: Double? = null

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
        viewing = null
        lastSeen.clear()
        previous.clear()
        val started = begin(Board(), "game")
        return mutableListOf(started, PositionChanged(gameId = gameId, fen = board.fen(), ply = 0))
    }

    /** A game of [kind] begun on [board], under a new id (`_begin`). */
    private fun begin(board: Board, kind: String, turnUnknown: Boolean = false): GameStarted {
        gameId = UUID.randomUUID().toString().replace("-", "").take(8)
        this.kind = kind
        startedFlag = true
        setPosition(board, turnUnknown)
        return gameStarted()
    }

    /**
     * [board] adopted as the game's position, the base its moves are counted from; the first
     * move's think time counts from now (`_set_position`). [turnUnknown]: both sides could be to
     * move on it and nothing has said which; every other position set has its side to move
     * settled (a mark recovery left it open: the audit of 2026-10-07). What was remembered of
     * the position before goes with it ([resetPositionMemory]).
     */
    internal fun setPosition(board: Board, turnUnknown: Boolean = false) {
        this.board = board
        baseFenValue = board.fen()
        turnUnknownFlag = turnUnknown
        clearStory()
        resetPositionMemory()
        // The start position is known, not read; any other is the screen's word for it.
        val read = board.boardFen() != Board().boardFen()
        readSquares = if (read) board.pieceMap().keys.toSet() else emptySet()
        finishedFlag = false
        resultValue = null
        lastCommitTs = tsValue
    }

    /**
     * `POSITION_MEMORY`: what the reconciler remembers of the position the game stands on, beside
     * the stories, each set back to its default whenever a position is set ([setPosition]: a new
     * game, a position adopted or re-read, a correction that starts the record again), as the
     * stories are. Kept over a new game, a premove queue drawn on the last one held a resync off
     * ([queueTs]), the mark's count of a still board went on counting ([unmarked]), and "already
     * told pending" stood for a listener that a Resync or a GameStarted had told otherwise
     * ([turnPendingTold]; the audit of 2026-10-07). The desktop resets the dataclass fields it
     * names; the same fields, in its order, here.
     */
    private fun resetPositionMemory() {
        turnPendingTold = false
        queueTs = null
        resyncCandidate = null
        resyncSeen = emptyList()
        unmarked = 0
        unmarkedView = null
        markDenied = false
        prevObs = Observation()
    }

    /**
     * [event], which tells the position whole with its side to move (GameStarted, Takeback,
     * Resync, GameEnded): its listener takes the turn as known from it (the coach's `onEvent`),
     * so the turn is no longer told pending, and [tellTurn] tells it again if the stories still
     * disagree. Kept over such an event, "already told" let a Takeback played in the same frame
     * as the stories parting leave the coach sure of a side to move nobody knew (the audit of
     * 2026-10-07). `_told_whole`.
     */
    private fun <E : Event> toldWhole(event: E): E {
        turnPendingTold = false
        return event
    }

    private fun gameStarted(): GameStarted =
        toldWhole(GameStarted(gameId = gameId, userSide = userSide, startFen = board.fen(), kind = kind))

    /**
     * No story pending, no cell remembered as stale or unexplained: the position they were
     * read against is gone (`_clear_story`).
     */
    private fun clearStory() {
        stories = listOf(Story())
        seen = emptyMap()
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
     * Index of the earlier position of this game (or of [board]'s) the frame shows, if any.
     *
     * Every position back to the base is checked, one by one, however long the game: a click
     * anywhere in the move list shows one. Only the last forty were, and the base; a click
     * further back read as a position no move explains and was adopted by a resync -- the game
     * numbered again from it, or, after the end, a new game begun -- and stepping forward
     * through the list then recorded the game's own moves a second time (the state review,
     * 2026-10-01).
     */
    private fun historyIndex(obs: Observation, diff: Set<Int>, board: Board? = null, shown: Int? = null): Int? {
        val earlier = (board ?: this.board).copy()
        val undone = HashSet<Int>()
        val plies = earlier.moveStack.size
        for (back in 1..plies) {
            val move = earlier.pop()
            undone.addAll(touchedSquares(earlier, move))
            if (shown != null && plies - back > shown) continue  // a ply the screen never showed
            if (undone.containsAll(diff) && matches(earlier, obs, obs.keys)) return plies - back
        }
        return null
    }


    /**
     * What [story] draws (`_shown`): the ply of its game it shows, or its position with its
     * premove queue played out.
     */
    private fun shown(story: Story): Board {
        val view = story.view ?: return premovePicture(story.board(board), story.queue)
        val earlier = story.copy(view = null, queue = emptyList()).board(board).copy()
        while (earlier.moveStack.size > view) earlier.pop()
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
        // What stays wrong is what no story kept draws: a premove queue drawn for ten seconds is
        // a story, not stale junk, and so is a square every story but the first explains (a
        // bishop's square left empty, read stale while the record waited on the stories, and
        // ignored: the stress sweep, 2026-10-06, seed 60).
        val pictures = stories.map { shown(it) }.ifEmpty { listOf(board) }
        var off: Set<Int> = disagreements(obs, pictures[0])
        for (picture in pictures.drop(1)) {
            val other = disagreements(obs, picture)
            off = off.filterTo(LinkedHashSet()) { it in other }
        }
        advanceStale(off, obs)
        val ghosts = ghosts(off, obs, pictures)
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
        prevObs = lastObs
        lastObs = obs
        // A piece in the user's hand: the hover outline a site draws under it, or one of the
        // user's own pieces, known and off its square. A shape of no known kind off a square is
        // a decoration -- a badge, an arrow, a glow -- and no hand.
        dragging = hoverSeen || obs.values.any { r ->
            r.kind == Kind.PIECE && !r.settled && r.symbol != null && r.colour == userSide
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
            val adopted = markedResync(obs)
            if (adopted.isNotEmpty()) {
                events.addAll(adopted)
                return events
            }
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
            fitted = fitting(obs, diff, roots = listOf(Story(head = listOf(HeadOp.Flip))))
        }
        if (fitted.isNotEmpty()) {
            events.addAll(decide(fitted, cautious = true))
            return events
        }

        // New game: the start position is back on the board.
        if (inProgress && obs.size >= 48 && matches(Board(), obs, obs.keys)) {
            // The game replaced: a story whose position proves its end is the game's end.
            val concluded = if (finishedFlag) mutableListOf() else commitPendingTerminal()
            events.addAll(concluded)
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
     * story kept extended by the cheapest events that fit, or the record told again when none
     * fits; a story's game shown at an earlier ply (browsing); a move played from the ply shown
     * (a takeback).
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
        var found: List<Story>
        if (roots == null) {
            // The stories kept are what every frame since the record allows: each carries their
            // evidence, and only they are extended. A side to move left open is a story of its
            // own while the record still stands as it was adopted.
            val from = ArrayList(stories)
            val record = from.firstOrNull { it.ops.isEmpty() && it.view == null }
            if (turnUnknownFlag && record != null) {
                val flipped = board.copy(withStack = false)
                if (turnValid(flipped, !board.turn)) {
                    // The same history as the record's story: the steps it took through the move
                    // list are the flipped one's too.
                    from.add(Story(head = listOf(HeadOp.Flip), steps = record.steps))
                }
            }
            found = this.found(from, search, obs)
            if (found.isEmpty() && !stories.all { it.ops.isEmpty() && it.queue.isEmpty() }) {
                // No story kept fits: the frames before were misread. The game is told again from
                // the record, with a picture the frames since it do not show.
                found = this.found(listOf(Story(cost = 1.0)), search, obs)
            }
            found = marked(found, search, obs, record)
        } else {
            found = this.found(roots, search, obs)
        }
        val browsed = ArrayList<Story>()
        if (diff.isNotEmpty() && !blockedFlag) {
            // The move list browsed: an earlier ply of a story's game, as it stands -- its moves
            // not yet told included: a move the user takes back before the record has told it is
            // in the list all the same (the flow matrix, 2026-10-06, a puzzle with takebacks: a
            // king's move taken back read as the king premoved back). Browsing changes nothing of
            // the game: the story goes on underneath, the ply shown over it.
            for (story in stories) {
                // The premove queue stays under the ply shown: a frame read as a ply browsed may
                // be none, and one that dropped the queue had the queue's moves told as a walk of
                // their own (the premove sweep, 2026-10-06, seed 16). Where browsing did clear it
                // (chess.com), the live board shown again is the queue cancelled
                // ([Search.extend]).
                val base = story.copy(view = null, flight = null)
                if (base.head.any { it !is HeadOp.Back }) {
                    continue  // a story's own game is its moves on the record, taken back or not
                }
                val game = base.board(board)
                // The plies the screen has shown: the record's, and the story's own moves as far
                // as each has been seen settled. A premove fired and never shown at rest is no ply
                // to browse back to (a premove drawn and cancelled left a story that read a still
                // board as its own ply browsed, for good: the stress sweep, 2026-10-06, seed 28).
                var shown = game.moveStack.size - base.moves.size
                for (i in base.moves.indices) {
                    if ((this.seen[base.ops.take(base.head.size + i + 1)] ?: 0) < settleFrames) break
                    shown += 1
                }
                var index = historyIndex(obs, disagreements(obs, game), game, shown)
                var flight: Int? = null
                if (index == null) {
                    val ply = browsedInFlight(base, game, shown, search, obs)
                    index = ply.first
                    flight = ply.second
                }
                val viewed = base.copy(
                    view = index,
                    flight = flight,
                    steps = story.steps + (if (story.view != index) 1 else 0),
                )
                if (index != null && viewed !in browsed) browsed.add(viewed)
            }
            // A move played from the ply shown: a takeback, or, from the start position of a game
            // begun there, a new game. An earlier position of the game is browsing: a different
            // move from the one shown is the only thing a takeback shows.
            if (browsed.isEmpty()) {
                for (story in stories) {
                    val cut = cut(story) ?: continue
                    // The move the game itself played from the ply is no takeback: it is the live
                    // board shown again ([found]).
                    val game = story.copy(view = null, queue = emptyList(), flight = null).board(board)
                    val view = story.view!!
                    val stack = game.moveStack
                    val taken = if (view < stack.size) stack[view] else null
                    for (s in search.extend(cut)) {
                        if (s.ops.size > cut.ops.size && s.ops[cut.ops.size] != taken &&
                            decorationsHold(s, obs)
                        ) {
                            browsed.add(s)
                        }
                    }
                }
            }
        }
        return kept(found, browsed, denied)
    }

    /**
     * The ply of [base]'s game whose move on from it the frame shows on its way, either way --
     * played forward through the list, or slid back by a takeback or a step back -- and where
     * ([inFlight]): the ply and the flight, or (null, null). A takeback slides each move back in
     * turn, and a frame catches the plies between at their move's slide only: a queen caught on
     * c2 sliding b3-d1 back was read as two moves played from the ply before (the quick
     * takebacks, 2026-10-06). `_browsed_in_flight`.
     */
    private fun browsedInFlight(
        base: Story,
        game: Board,
        shown: Int,
        search: Search,
        obs: Observation,
    ): Pair<Int?, Int?> {
        for (ply in minOf(shown, game.moveStack.size - 1) downTo 0) {
            val flight = inFlight(base.copy(view = ply), search, obs) ?: continue
            // The piece on its way is the one the move from the ply moves, off its square and
            // short of the other end: read on either, it is the ply or the next one at rest.
            val move = game.moveStack[ply]
            // Read on its way, never only missing: a queen taken on h8, gone from the board, was
            // the ply before it browsed with the queen "unread on her way", every frame for good
            // (the premove sweep, 2026-10-06, seed 356).
            if (obs.getValue(move.from).kind == Kind.EMPTY &&
                flight >= 0 &&
                flight != move.to &&
                onWay(move.from, move.to, flight)
            ) {
                return ply to flight
            }
        }
        return null to null
    }

    /**
     * The recovery from a side to move read the other way (`_marked`; Tai, 2026-10-06: "a good
     * and reliable recovery mechanism"). On a site that draws premoves, the user's moves and the
     * same moves premoved, each fired by the opponent's next, draw every frame alike: once the
     * record is a ply out, the pieces alone never put it back (the premove sweep, 2026-10-06,
     * seeds 360, 481, 673, ...). What tells them apart is the site's own mark of the move just
     * played, which a premove is not. When the frame shows the mark and no story that fits has
     * its last move marked, the stories are told again with the side to move the other way, and
     * kept when one of them has. A frame with nothing marked, or a site that marks nothing,
     * changes nothing.
     */
    private fun marked(
        found: List<Story>,
        search: Search,
        obs: Observation,
        record: Story?,
    ): List<Story> {
        markDenied = false
        val lit = highlightedSquares(obs)
        if (lit.isEmpty() || found.isEmpty() || blockedFlag) return found
        if (found.any { marksHold(it, lit) }) return found
        var flipped: List<Story> = emptyList()
        if (turnValid(board.copy(withStack = false), !board.turn)) {
            val root = Story(head = listOf(HeadOp.Flip), steps = record?.steps ?: 0)
            flipped = this.found(listOf(root), search, obs).filter {
                it.moves.isNotEmpty() && it.view == null && marksHold(it, lit)
            }
        }
        // Nothing told the other way has the mark either: the record has gone wrong further back
        // than the side to move ([markedResync]).
        markDenied = flipped.isEmpty()
        return flipped.ifEmpty { found }
    }

    /**
     * The board read whole and adopted, when the site's last-move mark has agreed with no story
     * for as many frames as a move needs to be played ([marked]; `_marked_resync`): the record
     * went wrong further back than a side to move, and every story since explains the screen
     * with a premove queue drawn over the true position (the premove sweep, 2026-10-06, seed
     * 956: a queued king's step read as a takeback). The mark names the side that moved; a
     * picture it does not fit -- a queue still drawn, a piece unread -- is not adopted, and the
     * stories go on.
     */
    internal fun markedResync(obs: Observation): List<Event> {
        if (!markDenied || obs.size < 64 || obs.values.any {
                it.hover || (it.kind == Kind.PIECE && (it.symbol == null || !it.settled))
            }
        ) {
            unmarked = 0
            return emptyList()
        }
        // Counted over frames of one still board -- the same pieces, the same marks: a move under
        // way is marked by the site before its pieces settle, and the stories wait for it (Tai's
        // puzzle, 2026-10-06: Nxg5 and hxg5 caught under the "Brilliant!" badge were adopted as a
        // position a second after, the two moves lost from the list).
        val view = loosePlacement(obs).boardFen() to highlightedSquares(obs).sorted()
        if (view != unmarkedView) {
            unmarkedView = view
            unmarked = 0
        }
        unmarked += 1
        if (unmarked < settleFrames + 1 || finishedFlag) return emptyList()
        val adopted = adoptPlacement(
            loosePlacement(obs),
            prefer = bottom.color,
            highlighted = highlightedSquares(obs),
        )
        if (adopted == null || adopted.second) {
            return emptyList()  // the mark does not settle the side to move: nothing proven
        }
        val adoptedBoard = adopted.first
        if (adoptedBoard.boardFen() == board.boardFen() && adoptedBoard.turn == board.turn) {
            return emptyList()
        }
        // A record a move or two out, never another position: a board that jumped is a new
        // puzzle or game, which the position's own resync begins afresh (Tai's puzzles,
        // 2026-10-06: the next puzzle was adopted into the last one's record).
        val changed = (0 until 64).count { adoptedBoard.pieceAt(it) != board.pieceAt(it) }
        if (changed >= resyncJumpCells) return emptyList()
        // A position the game has stood in is a ply shown or a takeback under way, marked with
        // that ply's own move (the flow matrix, 2026-10-06, seed 5 puzzle: two takebacks pressed
        // quickly were adopted as a position, the game's moves lost).
        val lit = highlightedSquares(obs)
        val earlier = board.copy()
        while (earlier.moveStack.isNotEmpty()) {
            earlier.pop()
            if (earlier.boardFen() != adoptedBoard.boardFen() || earlier.moveStack.isEmpty()) {
                continue
            }
            val into = earlier.moveStack.last()
            if (into.from in lit && into.to in lit) return emptyList()
        }
        // The record went wrong further back than its moves can say: the game is told again
        // from this board, the mark's side to move settled ([setPosition]). What was played
        // before it is not this game's any more; the record keeps a game from the board
        // adopted ([pgn]).
        viewing = null
        setPosition(adoptedBoard)
        return resynced("position adopted: the site's last-move mark fits no story")
    }

    /**
     * Whether the site's last-move mark, [lit], agrees with [story] (`_marks_hold`): both
     * squares of the last move of its game are marked. A ply browsed and a piece on its way carry
     * their own marks, and a game with no move, or a castling (marked by each site its own way),
     * says nothing.
     */
    private fun marksHold(story: Story, lit: Set<Int>): Boolean {
        if (story.view != null || story.flight != null || story.queue.isNotEmpty()) {
            return true  // a queue drawn covers marks with its tint and its pieces
        }
        val game = story.board(board)
        if (game.moveStack.isEmpty()) return true
        val last = game.moveStack.last()
        val moved = game.pieceAt(last.to)
        val files = kotlin.math.abs(squareFile(last.from) - squareFile(last.to))
        if (moved != null && moved.pieceType == KING && files > 1) return true
        return last.from in lit && last.to in lit
    }

    /**
     * The story a move played from the ply [story] shows goes on from (`_cut`): its own moves
     * past the ply cut off, or, past them, the record's taken back; a new game from the start
     * position when that is the ply and the game began there. Null when [story] shows no ply.
     */
    private fun cut(story: Story): Story? {
        val view = story.view ?: return null
        // The ply shown was the takeback, not a step through the list: the step it was counted
        // as is the takeback's own event.
        val base = story.copy(view = null, queue = emptyList(), flight = null, steps = maxOf(story.steps - 1, 0))
        val back = base.board(board).moveStack.size - view
        if (back <= base.moves.size) return base.copy(moves = base.moves.take(base.moves.size - back))
        if (view == 0 && baseFenValue == STARTING_FEN) {
            return Story(head = listOf(HeadOp.NewGame), steps = base.steps)
        }
        val taken = back - base.moves.size + base.head.sumOf { op ->
            when (op) {
                is HeadOp.Back -> op.plies
                is HeadOp.Fix -> op.square
                else -> error("a head of $op has no count")
            }
        }
        return Story(head = listOf(HeadOp.Back(taken)), steps = base.steps)
    }

    /** Every way the [roots] go on to draw the frame, the cheapest first (`_found`). */
    private fun found(roots: List<Story>, search: Search, obs: Observation): List<Story> {
        val found = ArrayList<Story>()
        val seenRoots = HashSet<Pair<List<Any>, List<Move>>>()
        // The cheapest stories first; each found sets a ceiling on what the rest may cost, so a
        // story that could only end up dearer than the ones kept is not searched through.
        var ceiling = Double.POSITIVE_INFINITY
        for (root in roots.sortedWith(compareBy({ it.cost }, { it.view != null }))) {
            // Browsing goes on as the story underneath it: the live board shown again, or a move
            // played on it (an opponent's while the user browsed).
            val story = if (root.view == null) root else root.copy(view = null)
            if (story.cost > ceiling) continue
            if (!seenRoots.add(story.ops to story.queue)) continue
            var more = search.extend(story, ceiling).map {
                it.copy(flight = null, steps = root.steps + (if (root.view != null) 1 else 0))
            }
            val passed = root.flight
            if (root.view == null && passed != null && passed >= 0) {
                // The piece was on its way over the square it was read on: its move goes there or
                // past it (a king read sliding g3-h2 was told as g3-g2, a move that never passes
                // h2, and the game fell a move behind: the premove sweep, 2026-10-06, seed 48).
                val mover = root.board(board).turn
                more = more.filter { s ->
                    // Put back: a drag of the user's let go. An opponent's piece is never put back
                    // (a king read on f3 sliding from f2, then on f2, was read as no move at all,
                    // and the game fell two moves behind: the premove sweep, seed 673).
                    (s.moves.size == root.moves.size && mover == userSide.color) ||
                        s.moves.drop(root.moves.size).any { m -> onWay(m.from, m.to, passed) }
                }
            }
            found.addAll(more)
            if (more.isNotEmpty()) {
                val cheapest = found.minOf { it.cost }
                ceiling = minOf(ceiling, cheapest)
            }
            val flight = inFlight(root, search, obs)
            if (flight != null && root.cost <= ceiling) found.add(root.copy(flight = flight))
        }
        return found.filter { decorationsHold(it, obs) }
    }

    /**
     * The square of the frame's one piece of [story] on its way (`_in_flight`): off its square,
     * which reads empty, and read on a square in a straight line from it that the story has empty
     * -- every site slides a moved piece in a straight line, and a frame can catch it centred on
     * a square it passes; a negative mark when it is read on none, its way over squares the frame
     * cannot read. Null when the frame shows anything else. A slide never shows its piece centred
     * on one square in two frames running: read so in the frame before as well, it has stopped
     * there, and is on its way no longer. Off centre it is moving, however many frames catch it
     * near one square (a rook sliding a7-c7, caught near b7 twice, was told as Rb7: the stress
     * sweep, 2026-10-06, seed 16).
     */
    private fun inFlight(story: Story, search: Search, obs: Observation): Int? {
        val picture = shown(story)
        val wrong = search.wrong(picture)
        val left = wrong.filter { picture.pieceAt(it) != null && obs.getValue(it).kind == Kind.EMPTY }
        if (left.size != 1 || wrong.size > 2) return null
        val start = left[0]
        val piece = picture.pieceAt(start)!!
        if (piece.colour != picture.turn) {
            // Only a move slides a piece, and only the side to move moves; a premove is not drawn
            // sliding until it fires, on its side's move (a white pawn read sliding c4-c5 with
            // black to move let a later frame tell black's reply before it: the flow matrix,
            // 2026-10-06, seed 7 puzzle).
            return null
        }
        var ends: Set<Int> = wrong - start
        if (ends.isEmpty()) {
            // The piece read off its square carries no evidence; it is somewhere on its way.
            ends = obs.filter { (_, r) ->
                r.kind == Kind.PIECE && !r.settled && r.symbol == piece.symbol()
            }.keys
        }
        // Where a move of the piece could take it: a pawn read four squares up its file is not on
        // its way (a pawn's a3, a4, a5 one frame each read as one slide a2-a7, and a3-a4 as one
        // push a2-a4: the six-step ladder).
        val ways = picture.legalMoves().filter { it.from == start }.map { it.to }
        val flown = story.flight
        for (sq in ends.sorted()) {
            val reading = obs.getValue(sq)
            val before = prevObs[sq]
            if (picture.pieceAt(sq) == null &&
                reading.symbol == piece.symbol() &&
                ways.any { onWay(start, it, sq) } &&
                // Read on its way before, it goes on along that line, past where it was read
                // (seed 48: a king read on h2 sliding from g3, then on g2, was "on its way" g3-g2
                // as well).
                (flown == null || flown < 0 || onWay(start, sq, flown)) &&
                !(reading.settled && before != null && before.settled && before.symbol == piece.symbol())
            ) {
                return sq
            }
        }
        if (wrong == setOf(start)) {
            // Its square empty and the piece read on no other: it has not vanished, it is
            // between squares, or over ones the frame cannot read (a pawn sliding d7-d5 under an
            // arrow was taken as stopped on d6: the stress sweep, 2026-10-06, seed 2; a bishop
            // over the pawn it was taking: seed 71). Seen on no square, it is seen stopped on
            // none: the mark is this frame's own, never a square another frame can repeat (a
            // bishop unread on its way f8-c5, then read on e7, was taken as read on e7 twice and
            // stopped: seed 60).
            return -1 - frameIndex
        }
        return null
    }

    /**
     * Of the stories that draw the frame, the ones kept (`_kept`): every one the pictures allow,
     * one to each state.
     */
    private fun kept(found: List<Story>, browsed: List<Story>, denied: Observation?): List<Story> {
        var all = found + browsed
        // No story holds a premove queue that draws nothing: played out over the position it
        // leaves every piece where it stands, so no picture shows it, and it is an assumption with
        // no trace. A king premoved f6-e7-e6 and then cancelled was read as the queue walked on
        // back to f6, and that queue shadowed every move after it (the premove sweep, 2026-10-06,
        // seed 7). Were one real, it fires as the user's instant reply.
        all = all.filter { it.queue.isEmpty() || !drawsNothing(board, it) }
        if (!denied.isNullOrEmpty()) all = all.filter { spares(it, denied) }
        // A piece corrected is one the story moves: its move is what says what it was.
        all = all.filter { s ->
            s.head.all { op -> op !is HeadOp.Fix || s.moves.any { it.from == op.square } }
        }
        // A side to move read the other way is told by a move played, never by a queue alone: a
        // rook sliding h1-h4 to take the checking queen, caught on h2 and h3, was read as the turn
        // wrong and the rook premoved, and the side to move was flipped (the stress sweep,
        // 2026-10-05, seed 116).
        // With the side to move still open, a piece on its way is that move being played, and
        // says which side it is (a puzzle's setup move caught sliding, its story dropped, was read
        // as the queen taken and premoved back: the fast puzzles, 2026-10-06).
        all = all.filter { s ->
            s.moves.isNotEmpty() || (s.flight != null && turnUnknownFlag) || HeadOp.Flip !in s.head
        }
        if (all.isEmpty()) return emptyList()
        // No story assumes more pictures nobody saw than the frames require: after e2-e4, "e2-e3,
        // then e4 premoved" draws the same pieces from a picture nobody saw.
        val least = all.minOf { it.cost }
        var allowed = all.filter { it.cost == least }
        // Of what the pictures allow alike, the story that assumes the fewest events: moves,
        // queued steps, what is done to the record. A premove drawn and taken back was read as
        // moves invented around it (the stress sweep, 2026-10-06, seed 23). A story of a piece on
        // its way assumes nothing yet, and stays. Nor does a story of an opponent's move the drawn
        // queue covers whole ([Search.unseenReplies]): no picture could show it, and none says it
        // did not happen; the queue's end tells (a queen's move under a queued pawn's path,
        // dropped as an event too many, left the game a move behind for good: the premove sweep,
        // 2026-10-06, seed 24).
        // A story whose position ends the game -- a mate, a draw by repetition -- is kept all the
        // same: the end shown, or the next game, is the evidence that decides it
        // ([pendingTerminal]).
        val atRest = allowed.filter { it.flight == null && !it.hidden }
        if (atRest.isNotEmpty()) {
            val fewest = atRest.minOf { events(it) }
            allowed = allowed.filter { s ->
                s.flight != null ||
                    s.hidden ||
                    events(s) == fewest ||
                    resultOnBoard(s.board(board)) != null
            }
        }
        // Visible before hidden (Tai, 2026-10-06): when a story with no premove queue draws the
        // frame, a story that needs a queue no frame has shown it needs is dropped. On a site that
        // draws premoves, the user's moves and the same moves premoved a step behind draw every
        // frame alike, and kept side by side they never part: the record stalled. Read wrong, a
        // real queue fires with the opponent's next move, and the record is put right by the move
        // after it. It covers browsing and takebacks against a premove drawing the same pieces as
        // well (Tai, 2026-10-05: "takeback and browse back has to work correctly always; premove
        // can be a bit wrong").
        val queued = allowed.filter { it.queue.isNotEmpty() && it.view == null }
        if (queued.isNotEmpty() && allowed.any { it !in queued && it.flight == null }) {
            allowed = allowed.filter { it !in queued }
        }
        // Likelihood never drops a story (Tai, 2026-10-05: the side to move must never go wrong,
        // and the position must be right once premoves resolve): every story the pictures allow
        // is kept, whatever it assumes, and the record takes only what all of them agree on.
        // Stories that end in one state -- position, side to move, queue, ply shown -- have one
        // future, and are kept as one: the simplest of them, which is the one told when the record
        // follows it.
        allowed = allowed.sortedWith(STORY_ORDER)
        val byState = LinkedHashMap<List<Any?>, Story>()
        for (story in allowed) byState.putIfAbsent(story.key(board), story)
        var kept = byState.values.toList()
        if (kept.size > MAX_STORIES) {
            // More states than are searched each frame: the cap may never cost a side to move --
            // the first story of each comes before the rest.
            val firsts = LinkedHashMap<Boolean, Story>()
            for (story in kept) firsts.putIfAbsent(story.board(board).turn, story)
            val heads = firsts.values.toList()
            kept = heads + kept.filter { it !in heads }.take(MAX_STORIES - heads.size)
        }
        return kept
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
     * The story whose position the screen shows, when the stories differ only in what no picture
     * can tell apart (`_shown_alike`): each leaves the same side to move and draws the same board,
     * and the first with no premove queue (and no ply browsed) has the board shown as its
     * position. Null otherwise.
     */
    private fun shownAlike(stories: List<Story>): Story? {
        if (stories.any { it.view != null }) return null
        if (stories.map { it.board(board).turn }.toSet().size != 1) return null
        val drawn = stories.map { premovePicture(it.board(board), it.queue).boardFen() }.toSet()
        if (drawn.size != 1) return null
        val lead = stories.firstOrNull { it.queue.isEmpty() } ?: return null
        // Followed only when every other story can be kept, told again from it: one that cannot
        // (its side to move read the other way before its moves) would be lost, and with it the
        // side to move it may be right about.
        for (story in stories) {
            if (story.ops.take(lead.ops.size) != lead.ops && retold(story, lead.ops) == null) return null
        }
        return lead
    }

    /**
     * What the record does with the stories that fit (docs/reconciler-model.md, section 5,
     * steps 3 and 4; `_decide`). When every story leads to one key -- one position, one side to
     * move, one queue, one ply shown -- they differ only in how they got there, and the cheapest
     * is the record's to follow. Otherwise the record follows only the ops every story begins
     * with. An op is played once it has held [settleFrames] frames, one more when it came with
     * others in one frame or from the stale retry. A wrong side to move is never told: while
     * stories disagree on it, nothing past their common start is played.
     */
    internal fun decide(found: List<Story>, cautious: Boolean = false): MutableList<Event> {
        var fitted = found
        if (fitted.size == 1 && fitted[0].cost != 0.0) {
            fitted = listOf(fitted[0].copy(cost = 0.0))  // alone, its past is the past
        }
        var concluded: MutableList<Event> = mutableListOf()
        if (!finishedFlag && fitted.any { it.head.take(1) == listOf(HeadOp.NewGame) }) {
            // A new game on the board: the game it replaces ends first, with a story the stories
            // before it prove the end of (a mate, a draw by repetition).
            concluded = commitPendingTerminal()
            if (concluded.isNotEmpty()) {
                // The record has moved on to the old game's end: what is told from it now is the
                // new game alone, which owes nothing to the record.
                fitted = fitted.filter { it.head.take(1) == listOf(HeadOp.NewGame) }
            }
        }
        stories = fitted
        val told = concluded
        told.addAll(tellTurn())
        unexplainedFrames = 0
        unexplainedNoted = false
        agreesFlag = fitted.any { it.ops.isEmpty() && it.view == null && it.flight == null }
        queueTs = if (fitted.any { it.queue.isNotEmpty() }) tsValue else null
        // A frame confirms a move only when the squares it changed are seen settled in it: a
        // frame with a piece in motion keeps the stories, and confirms nothing it cannot see (the
        // bishop caught on c5 sliding d6-b4 was told as Bc5 on the slide's next frame: the stress
        // sweep, 2026-10-05, seed 1).
        val seen = HashMap<List<Any>, Int>()
        for (story in fitted) {
            if (story.view != null) {
                // A ply browsed shows nothing of the story underneath, and takes nothing from it:
                // what earlier frames showed of it stands (a takeback's story lost the sightings
                // of the move it went back to, so that ply was no longer one the screen had shown,
                // and a story with the wrong side to move took over: the flow matrix, 2026-10-06,
                // seed 6 puzzle).
                for (i in story.ops.indices) {
                    val path = story.ops.take(i + 1)
                    val had = this.seen[path]
                    if (had != null) seen[path] = maxOf(seen[path] ?: 0, had)
                }
                continue
            }
            for ((i, now) in seenOps(story.ops).withIndex()) {
                val path = story.ops.take(i + 1)
                seen[path] = maxOf(seen[path] ?: 0, (this.seen[path] ?: 0) + now)
            }
        }
        this.seen = seen
        if (fitted.any { it.view != null }) {
            // The board may show a ply browsed: a frame of an earlier ply confirms nothing of the
            // stories underneath, and nothing is played until the live board is shown again. A
            // takeback's piece caught mid-slide read as a move from the ply browsed, and the
            // browsing frames after it, counted for that move, told it (the takebacks pressed
            // quickly, 2026-10-06).
            if (!blockedFlag) viewing = fitted.firstOrNull { it.view != null }?.view
            return told
        }
        val keys = fitted.map { it.key(board) }.toSet()
        // One key is one future: the position, the side to move, the queue and the ply shown are
        // certain, and the record follows the simplest story to them (Tai, 2026-10-05: the side
        // to move first, then the position current, then the moves themselves).
        var ops: List<Any>
        if (keys.size == 1) {
            val chosen = fitted[0]
            ops = chosen.ops
            if (ops.isEmpty()) {
                if (chosen.view == null || !blockedFlag) viewing = chosen.view
                return told
            }
        } else {
            ops = commonStart(fitted.map { it.ops })
            val lead = shownAlike(fitted)
            if (lead != null) {
                // Every story has the side to move of the others and draws the same board, and
                // one of them has no premove queue: the board shown is its position. The record
                // follows it at once (Tai, 2026-10-05: the position current, the side to move
                // exact); the others stay, retold from it ([play]), and the moves are what may be
                // put right later.
                ops = lead.ops
            }
            if (ops.isEmpty()) {
                // Nothing all stories agree on yet. What is shown meanwhile is the simplest
                // story's ply: browsing the move list before a premove that draws the same
                // picture (Tai, 2026-10-05: a takeback and browsing are always read right). It is
                // shown only, never played: the record waits for agreement.
                if (fitted[0].view == null || !blockedFlag) viewing = fitted[0].view
                return told
            }
        }
        viewing = null
        // An op is played once the frames have shown it settled [settleFrames] times, one more
        // when it came from the stale retry.
        val need = settleFrames + (if (cautious) 1 else 0)
        var ready = 0
        while (ready < ops.size && (seen[ops.take(ready + 1)] ?: 0) >= need) ready += 1
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

    /**
     * [TurnPending] when whether the stories agree on the side to move differs from what the
     * listener holds: the last told, or known since an event told a position whole ([toldWhole];
     * `_tell_turn`).
     */
    internal fun tellTurn(): MutableList<Event> {
        val pending = turnPending
        if (pending == turnPendingTold) return mutableListOf()
        turnPendingTold = pending
        return mutableListOf(TurnPending(gameId = gameId, pending = pending))
    }

    /** The record does [ops], and every story is told from it (`_play`). */
    private fun play(ops: List<Any>): MutableList<Event> {
        val before = stories
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
                    events.add(toldWhole(Takeback(gameId = gameId, plies = op.plies, fen = board.fen())))
                    events.add(PositionChanged(gameId = gameId, fen = board.fen(), ply = board.moveStack.size))
                }
                is HeadOp.Flip -> events.addAll(flip())
                is HeadOp.Fix -> events.addAll(fix(op.square, op.kind))
            }
        }
        val done = ops.size
        val kept = ArrayList<Story>()
        for (story in before) {
            if (story.ops.take(done) == ops) {
                kept.add(story.rebased(done))
                continue
            }
            val retold = retold(story, ops)
            if (retold != null) kept.add(retold)
        }
        stories = kept.ifEmpty { listOf(Story()) }
        val seen = LinkedHashMap<List<Any>, Int>()
        for ((path, count) in this.seen) {
            if (path.take(done) == ops && path.size > done) seen[path.drop(done)] = count
        }
        this.seen = seen
        stale.clear()
        viewing = null
        return events
    }

    /**
     * The record's side to move read the other way (`_flip`): the first move of a position
     * adopted with the turn open settles it, silently; a turn told wrong is told again.
     *
     * Told again from the position, not from the base: the sides alternate, so the moves played
     * since the base, each its side's, never play out from the base with the other side to move.
     * The record keeps a game from the position corrected ([pgn]).
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

    /** The game's moves played again from [base], or null when one of them is not legal there (`_replayed`). */
    private fun replayed(base: Board): Board? {
        val out = base.copy(withStack = false)
        for (move in board.moveStack) {
            if (!out.isLegal(move)) return null
            out.push(move)
        }
        return out
    }

    /**
     * A piece of a position read from the screen, never moved since, was of another kind
     * (`_fix`). Still the same game: the piece has stood on its square since the base, so the
     * base is corrected and the moves played since are played again from it -- the record keeps
     * every one, and their numbering (the audit of 2026-10-07: the game was adopted afresh from
     * the correction, and the moves before it left the record and the review). The position is
     * told as it now stands; the move that showed the piece's kind follows it.
     *
     * When there is no move to keep, or one is not legal from the corrected base, the position
     * is adopted again with the piece corrected, as a re-read. The pieces that had moved stay
     * known either way, and the next move's think time counts from the last move.
     */
    private fun fix(square: Int, kind: Int): MutableList<Event> {
        val was = board.pieceAt(square)!!
        val read = readSquares - square
        val since = lastCommitTs
        val base = Board(baseFenValue)
        var again: Board? = null
        if (board.moveStack.isNotEmpty() && base.pieceAt(square) == was) {
            base.setPieceAt(square, Piece(kind, was.colour))
            again = replayed(base)
        }
        if (again != null) {
            board = again
            baseFenValue = base.fen()
            readSquares = read
            val events = mutableListOf<Event>(
                PositionChanged(gameId = gameId, fen = board.fen(), ply = board.moveStack.size),
            )
            if (board.isGameOver()) {
                finishedFlag = true
                resultValue = board.result()
                events.add(endedEvent(resultValue!!, "game over on board"))
            }
            return events
        }
        val fixed = board.copy(withStack = false)
        fixed.setPieceAt(square, Piece(kind, was.colour))
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
     * The story that ends the game, when the end is all the stories leave (`pending_terminal`):
     * the mate the site's dialog or badges then covered before a clean frame could confirm it.
     * The end shown is evidence as good as a frame: a story that does not end the game -- with
     * [result], when the site shows one -- is ruled out by it (a premove queue still drawn is not
     * a game over; the site clears it). A game over is what its final position proves
     * ([resultOnBoard]): a mate or a stalemate, dead material, the third repetition or the
     * fiftieth move played. Every story left must end in one state.
     */
    fun pendingTerminal(result: String? = null): Pair<List<Any>, Board>? {
        val ended = ArrayList<Pair<Story, Board>>()
        for (story in stories) {
            val takesBack = story.head.all { it is HeadOp.Back }
            if (!takesBack || story.moves.isEmpty() || story.queue.isNotEmpty()) {
                continue  // a ply browsed over a story is the story still: its view is no end
            }
            val end = story.board(board)
            val proved = resultOnBoard(end)
            if (proved == null || (result != null && proved.first != result)) continue
            ended.add(story to end)
        }
        if (ended.isEmpty() || ended.map { it.first.copy(view = null).key(board) }.toSet().size != 1) return null
        val (story, end) = ended[0]
        return story.ops to end
    }

    /** Commit the story that ends the game (see [pendingTerminal]). */
    fun commitPendingTerminal(result: String? = null): MutableList<Event> {
        val story = pendingTerminal(result) ?: return mutableListOf()
        stories = stories.filter { it.ops == story.first }.take(1)
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
                if (turnValid(trial, trial.turn)) roots.add(Story(head = listOf(HeadOp.Fix(sq, kind))))
            }
        }
        return roots
    }

    /** Stale readings of something unnameable on squares every story has empty (`_ghosts`). */
    private fun ghosts(off: Set<Int>, obs: Observation, pictures: List<Board>): Set<Int> =
        off.filter { sq ->
            stale.getValue(sq).second > staleAfter &&
                obs.getValue(sq).kind == Kind.PIECE &&
                obs.getValue(sq).symbol == null &&
                pictures.all { it.pieceAt(sq) == null }
        }.toSet()

    /**
     * Where the user's pieces stood before this frame, in any story kept (`_liftable`): the
     * squares a piece in the hand can have come from. One story's squares only, and the king
     * lifted off b6 in the story that had it there was no king in hand (the drag of the fourth
     * game's king walk, 2026-10-06).
     */
    private fun liftable(): Set<Int> {
        val squares = HashSet<Int>()
        for (story in stories.ifEmpty { listOf(Story()) }) {
            val picture = shown(story)
            squares.addAll(squaresOf(picture.occupiedCo[if (userSide.color) 1 else 0]))
        }
        return squares
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
        val first = openingMove(adoptedBoard, placement, since, highlightedSquares(obs))
        viewing = null
        if (first != null) {
            // The brief board before it is the start, and the move between them is played.
            val (start, move) = first
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
            val events = mutableListOf<Event>(begin(adoptedBoard, "position", unknownTurn))
            events.addAll(positioned())
            return events
        }
        setPosition(adoptedBoard, unknownTurn)
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
    internal fun resynced(reason: String): MutableList<Event> {
        val events = mutableListOf<Event>(toldWhole(Resync(gameId = gameId, fen = board.fen(), reason = reason)))
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
        // The stories stay, and told again pending when they still disagree on the turn.
        val events = resynced("side to move settled by the clock: ${Side.of(colour).name.lowercase()}")
        events.addAll(tellTurn())
        return events
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
                val events = mutableListOf<Event>(begin(start, "position"))
                events.addAll(commit(move))
                return events
            }
        }
        val (adoptedBoard, unknownTurn) = adopted
        return mutableListOf(begin(adoptedBoard, "position", unknownTurn))
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
        val concluded = commitPendingTerminal()
        if (finishedFlag) return concluded
        val proved = resultOnBoard(board) ?: return concluded
        concluded.addAll(end(proved.first, proved.second))
        return concluded
    }

    /** The GameEnded of this game as it stands (`ended_event`): also the pipeline's, for a game it drops. */
    fun endedEvent(result: String, reason: String): GameEnded = toldWhole(
        GameEnded(
            gameId = gameId,
            result = result,
            reason = reason,
            pgn = pgn(),
            finalFen = board.fen(),
        ),
    )

    /**
     * The game as PGN: the moves since the base position, numbered from it. A base that is not
     * the start position (a position adopted, joined or re-read) is given in the SetUp and FEN
     * headers, so what is kept is a game that plays out from where it began; the moves alone,
     * numbered from the middle, were no game anyone could read back (the audit of 2026-10-07).
     * A game from the start is its moves alone.
     */
    fun pgn(): String {
        val head = if (baseFenValue == STARTING_FEN) "" else "[SetUp \"1\"]\n[FEN \"$baseFenValue\"]\n\n"
        return head + movetext()
    }

    private fun movetext(): String {
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
