package com.sidecoach.analysis

import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.events.Analysis
import com.sidecoach.events.Event
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import com.sidecoach.events.MoveAssessment
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * From moves to assessments: what a move cost, what the position holds, what is threatened.
 * A port of `sidecoach/analysis/annotator.py`.
 *
 * [GameAnnotator] does the work for one game and is synchronous, which is what a post-game
 * review wants. [Analyser] wraps it in a worker thread, which is what a live session needs:
 * the engine takes a third of a second per position and the capture loop must not wait.
 *
 * Every position of a game is measured once, to the same depth, and cached, so the position
 * after one move is the position before the next and a game costs one analysis per ply. The
 * matching depth is what makes the loss of a move meaningful: two searches of different
 * lengths would differ by more than most mistakes do.
 *
 * Its collaborators are the desktop's: the motif tagger (what a move did, what a position
 * holds -- a sacrifice is what makes a move brilliant) and the opening book (whether a move is
 * theory, which opening the game is in). Both are optional, so a bare engine still works;
 * `Assessing.fromSettings` passes both, as `_make_analyser` does.
 */

/** What the motif tagger provides: `annotator.Tagger`, which `Motifs` is. */
interface Tagger {
    fun tagsForMove(before: Board, move: Move, pv: List<Move> = emptyList(), mateIn: Int? = null): List<MotifTag>
    fun tagsForPosition(board: Board, pv: List<Move> = emptyList(), mateIn: Int? = null): List<MotifTag>
}

/** A whole measurement while the analyser is behind a fast game: capped, and no threats. */
const val HURRY_S = 1.0
const val BEHIND_S = 10.0

/** One position, measured once: its lines and what they mean. */
class Measured(
    val fen: String,
    val lines: List<Line>,
    val score: Double,
    val decisive: Boolean,
    val onlyMove: Boolean,
    val threats: List<String> = emptyList(),
    val traps: List<Line> = emptyList(),
) {
    /** How deep the search actually got, which the time cap can cut short. */
    val depth: Int get() = if (lines.isEmpty()) 0 else lines[0].depth
}

/** Assessments for the moves of one game. Not thread-safe; [Analyser] owns one. */
class GameAnnotator(
    val engine: Engine,
    var depth: Int = 18,
    var multipv: Int = 6,
    var thresholds: Thresholds = Thresholds(),
    var tagger: Tagger? = null,
    /** Set late when the tables arrive after the engine was built, read by the worker. */
    @Volatile var book: Book? = null,
    var withThreats: Boolean = true,
) {

    private val cache = HashMap<String, Measured>()

    /**
     * Positions measured for a verdict only: the best line at the full depth, no candidates,
     * no threats. Kept apart from [cache], so a later whole measurement of the same position
     * is not answered with the thin one.
     */
    private val verdictCache = HashMap<String, Measured>()

    /**
     * The last move assessed: game, ply, loss, where it landed and whether it captured, so
     * the next move knows whether the opponent just erred and whether it is merely taking
     * back.
     */
    private class Previous(
        val gameId: String,
        val ply: Int,
        val loss: Double,
        val to: Int,
        val captured: Boolean,
    )

    private var previous: Previous? = null

    /** Forget the cache; a new game shares no positions with the last one. */
    fun reset() {
        cache.clear()
        verdictCache.clear()
        previous = null
    }

    /**
     * The position measured for a verdict: whole if it already is, else its best line alone
     * at the full depth -- what [evalLoss] needs, at a fraction of the cost. When the engine
     * is behind a fast game, this is all a stale ply gets.
     */
    fun measureForVerdict(board: Board): Measured {
        val key = board.fen()
        cache[key]?.let { return it }
        verdictCache[key]?.let { return it }
        val lines = engine.analyse(board, depth = depth, multipv = 1)
        val thin = Measured(
            fen = key,
            lines = lines,
            score = positionScore(board, lines),
            decisive = false,
            onlyMove = false,
        )
        verdictCache[key] = thin
        return thin
    }

    /**
     * The position whole: the candidates, the threats, the decisive-moment and only-move
     * calls. `hurry` caps the search at [HURRY_S] and skips the threats: the analyser is (or
     * was just) behind a fast game, and the player is not waiting ten seconds for the
     * candidates of a position they will have left.
     */
    fun measure(board: Board, hurry: Boolean = false): Measured {
        val key = board.fen()
        cache[key]?.let { return it }
        val lines = engine.analyse(
            board,
            depth = depth,
            multipv = multipv,
            timeS = if (hurry) HURRY_S else null,
            carryBest = !hurry,
        )
        val measured = Measured(
            fen = key,
            lines = lines,
            score = positionScore(board, lines),
            decisive = isDecisive(board, lines, thresholds),
            onlyMove = isOnlyMove(board, lines, thresholds),
            threats = if (withThreats && !hurry) threatMoves(engine, board, depth) else emptyList(),
            // The captures that look free and are not, beside the threats: what the first hint
            // warns of ("the knight on g4 is bait"), searched only when not hurried.
            traps = if (withThreats && !hurry) Traps.trapLines(engine, board, lines) else emptyList(),
        )
        cache[key] = measured
        return measured
    }

    /**
     * What the position holds, for the side to move. `thin`: only what the verdict already
     * measured -- the analysis of a position the game has left, so the coach still has the
     * facts of the position each move was played from and the review has its score.
     */
    fun analysisEvent(
        gameId: String,
        board: Board,
        ply: Int,
        hurry: Boolean = false,
        thin: Boolean = false,
    ): Analysis {
        val measured = if (thin) measureForVerdict(board) else measure(board, hurry)
        return Analysis(
            gameId = gameId,
            fen = measured.fen,
            ply = ply,
            lines = measured.lines,
            decisive = measured.decisive,
            onlyMove = measured.onlyMove,
            threats = measured.threats,
            traps = measured.traps,
            tags = positionTags(board, measured),
            opening = opening(board),
            inBook = inBook(board),
        )
    }

    /** What the move cost its mover, and what the position it led to holds. */
    fun assess(gameId: String, before: Board, move: Move, ply: Int): Pair<MoveAssessment, Analysis> {
        val assessment = assessMove(gameId, before, move, ply)
        val after = before.copy()
        after.push(move)
        return assessment to analysisEvent(gameId, after, ply)
    }

    /**
     * What the move cost its mover -- from the best line of the position it led to, at the
     * full depth but that one line: a sixth of the time of the six the coach's candidates
     * take, so the verdict, and the word on the move, come that much sooner.
     */
    fun assessMove(gameId: String, before: Board, move: Move, ply: Int): MoveAssessment {
        if (!before.isLegal(move)) {
            throw IllegalArgumentException("${move.uci()} is not legal in ${before.fen()}")
        }
        val measuredBefore = measureForVerdict(before)
        // The history comes along: the opening a game is in is a fact about its moves, and
        // the position after this move is the next move's history.
        val after = before.copy()
        after.push(move)
        val measuredAfter = measureForVerdict(after)
        val loss = evalLoss(measuredBefore.lines, move.uci(), measuredAfter.score)
        val tags = moveTags(before, move, measuredAfter)
        val best = if (measuredBefore.lines.isEmpty()) emptyList() else measuredBefore.lines[0].moves
        var last = previous
        if (last != null && (last.gameId != gameId || last.ply != ply - 1)) {
            last = null  // another game, or a gap: the move before this one is unknown
        }
        val opponentErred = last != null && last.loss >= thresholds.mistake
        // Taking back on the square the opponent just captured on. Taking a piece that merely
        // moved there is not obvious: that is how a blunder gets punished.
        val captures = before.isCapture(move)
        val recapture = last != null && last.captured && captures && move.to == last.to
        // A capture nobody can take back is a grab, not a find; a king's capture always is.
        // Taking back is read by the rules: a piece guarded only by one pinned to its own king
        // is taken for nothing.
        val freeGrab = captures && !Exchange.canRetake(before, move)
        val obvious = recapture || freeGrab || before.legalMoves().size == 1
        val label = labelMove(
            loss,
            thresholds,
            isBest = best.isNotEmpty() && best[0] == move.uci(),
            inBook = inBook(after),
            onlyMove = measuredBefore.onlyMove,
            sacrifice = tags.any { it.name == "sacrifice" },
            beforeScore = measuredBefore.score,
            opponentErred = opponentErred,
            obvious = obvious,
        )
        previous = Previous(gameId, ply, loss, move.to, captures)
        return MoveAssessment(
            gameId = gameId,
            ply = ply,
            uci = move.uci(),
            expectedLoss = loss,
            severity = severityOf(loss, thresholds),
            best = best,
            tags = tags,
            depth = minOf(
                measuredBefore.depth,
                if (measuredAfter.depth == 0) measuredBefore.depth else measuredAfter.depth,
            ),
            label = label,
            accuracy = moveAccuracy(loss),
        )
    }

    // Tagging and openings are optional collaborators, so a bare engine still works.

    /**
     * What the position holds for the side to move; their mate distance counts their own
     * next move, which is the same convention a move's tags use.
     */
    private fun positionTags(board: Board, measured: Measured): List<MotifTag> {
        val tagger = tagger ?: return emptyList()
        if (measured.lines.isEmpty()) return emptyList()
        return tagger.tagsForPosition(board, pv = lineMoves(measured.lines[0]), mateIn = measured.lines[0].mate)
    }

    private fun moveTags(before: Board, move: Move, after: Measured): List<MotifTag> {
        val tagger = tagger ?: return emptyList()
        val pv = if (after.lines.isNotEmpty()) lineMoves(after.lines[0]) else emptyList()
        val tags = tagger.tagsForMove(before, move, pv = pv, mateIn = mateIn(before, move, after.lines))
        // A tactic the move makes is only its point when the line after it cashes it in: "the
        // queen on e7 is loose" was said of a queen that simply steps away, 98 times in the
        // master games at moves whose note names no tactic (2026-09-24).
        var line = listOf(move) + pv
        val boards = arrayListOf(before.copy(withStack = false))
        for (m in line) {
            if (!boards.last().isLegal(m)) break
            val nxt = boards.last().copy(withStack = false)
            nxt.push(m)
            boards.add(nxt)
        }
        line = line.subList(0, boards.size - 1)
        return tags.filter { it.name !in Motifs.TARGETED || Motifs.paysOff(boards, line, 0, it) }
    }

    /** Whether the game is still in theory after this move; needs the moves that got here. */
    private fun inBook(board: Board): Boolean {
        val book = book ?: return false
        if (board.moveStack.isEmpty()) return false
        if (board.root().fen() != STARTING_FEN) return false
        return book.inBook(ArrayList(board.moveStack))
    }

    /**
     * The opening this game is in, when the board carries the moves that got here. A position
     * read off the screen, or one adopted by a resync, has no history to look up, and a
     * fragment of a history would name the wrong opening.
     */
    private fun opening(board: Board): String? {
        val book = book ?: return null
        if (board.moveStack.isEmpty()) return null
        if (board.root().fen() != STARTING_FEN) return null
        val found = book.identify(ArrayList(board.moveStack)) ?: return null
        return if (found.eco.isNotEmpty()) "${found.eco} ${found.name}" else found.name
    }
}

/** A line's moves as moves, up to the first that does not parse: `annotator._moves`. */
private fun lineMoves(line: Line): List<Move> {
    val out = ArrayList<Move>()
    for (uci in line.moves) out.add(Move.fromUciOrNull(uci) ?: break)
    return out
}

/**
 * Moves the mover needs to mate, counting this one; negative when the mover gets mated.
 * `annotator._mate_in`: the engine reports the mate distance in the position after the move,
 * where the opponent is to move and counts their own moves -- "mated in two" there means two
 * more moves by the mover, three counting the one just played.
 */
internal fun mateIn(before: Board, move: Move, afterLines: List<Line>): Int? {
    val after = before.copy(withStack = false)
    after.push(move)
    if (after.isCheckmate()) return 1
    if (afterLines.isEmpty()) return null
    val mate = afterLines[0].mate ?: return null
    return if (mate > 0) -mate else Math.abs(mate) + 1
}

/**
 * A [GameAnnotator] on a worker thread: submit moves, poll events, never block.
 *
 * The engine is slow by design and the capture loop runs many times a second, so nothing
 * here waits for a result. Assessments arrive a moment after the move, which is when a coach
 * would speak anyway.
 */
class Analyser(val annotator: GameAnnotator, private val maxPending: Int = 64) : AutoCloseable {

    var dropped = 0
        private set
    var failures = 0
        private set

    /** Whole analyses not run for plies the game had already left. */
    var skipped = 0
        private set

    /** Requests let go because the game took their move back. */
    var withdrawn = 0
        private set
    var lastError: String? = null
        private set

    private var behindUntil = 0L
    private val requests = LinkedBlockingQueue<Request>()
    private val results = ConcurrentLinkedQueue<Event>()
    private var thread: Thread? = null

    /**
     * Requests submitted and not yet answered (or dropped): counted, not read off the queue,
     * so `pending` is never 0 in the instant between a request leaving the queue and the
     * worker starting on it -- a review taken then would miss the last verdict.
     */
    private var outstanding = 0
    private val countLock = Any()

    private class Request(
        val gameId: String,
        val ply: Int,
        val board: Board,
        val move: Move?,
    ) {
        /**
         * Set when the game takes the move back while it is being measured ([withdraw]):
         * nothing about it is delivered, and the position after it is not measured.
         */
        @Volatile var withdrawn = false
    }

    /** The request the worker is on, so a takeback can let go of it. */
    @Volatile private var current: Request? = null

    /** The sentinel that ends the worker. */
    private val stop = Request("", 0, Board(null), null)

    val pending: Int get() = synchronized(countLock) { outstanding }

    fun start() {
        if (thread != null) return
        thread = Thread({ run() }, "sidecoach-analyser").also {
            it.isDaemon = true
            it.start()
        }
    }

    override fun close() {
        val worker = thread
        thread = null
        if (worker != null) {
            requests.put(stop)
            worker.join(5000)
        }
        annotator.engine.close()
    }

    fun submitMove(gameId: String, before: Board, move: Move, ply: Int) {
        // With its moves: the book is asked with the moves that led here (`inBook`,
        // `opening`), and a copy without them was in the book on the first move alone.
        submit(Request(gameId, ply, before.copy(), move))
    }

    fun submitPosition(gameId: String, board: Board, ply: Int) {
        submit(Request(gameId, ply, board.copy(withStack = false), null))
    }

    /**
     * A new game: drop what is queued and forget the measured positions, since none of it is
     * about this game. What the worker is busy with finishes and is delivered; the pipeline
     * ignores an event for a game that is over.
     */
    fun reset() {
        while (true) {
            val taken = requests.poll() ?: break
            if (taken === stop) { requests.put(stop); break }
            settle()
        }
        annotator.reset()
    }

    /**
     * The game's moves after [ply] taken back: what is queued for them is dropped, and one
     * being measured is let go once its search ends -- nothing about it is delivered, and the
     * position after it is not measured. The engine gets to the replacement move at once,
     * where it used to measure every withdrawn move first (Tai, 2026-09-23: "takebacks make
     * the program lag behind by a lot"; seconds on the phone). `Analyser.withdraw`.
     */
    fun withdraw(gameId: String, ply: Int) {
        val kept = ArrayList<Request>()
        while (true) {
            val request = requests.poll() ?: break
            if (request !== stop && request.gameId == gameId && request.ply > ply) {
                withdrawn += 1
                settle()
            } else {
                kept.add(request)
            }
        }
        for (request in kept) requests.put(request)
        val now = current
        if (now != null && now.gameId == gameId && now.ply > ply) {
            now.withdrawn = true
            withdrawn += 1
        }
    }

    fun poll(): List<Event> {
        val out = ArrayList<Event>()
        while (true) out.add(results.poll() ?: break)
        return out
    }

    private fun submit(request: Request) {
        if (requests.size >= maxPending) {
            // The engine is further behind than a game can get; the newest position still
            // matters more than an old one, so the oldest goes.
            val taken = requests.poll()
            if (taken != null && taken !== stop) {
                dropped += 1
                settle()
            } else if (taken === stop) {
                requests.put(stop)
            }
        }
        synchronized(countLock) { outstanding += 1 }
        requests.put(request)
    }

    private fun settle() {
        synchronized(countLock) { outstanding = maxOf(0, outstanding - 1) }
    }

    private fun run() {
        while (true) {
            val request = requests.take()
            if (request === stop) return
            current = request
            try {
                if (request.move == null) {
                    val event = annotator.analysisEvent(request.gameId, request.board, request.ply)
                    if (!request.withdrawn) results.add(event)
                } else {
                    // The verdict first, from the best line alone; the position whole after --
                    // unless newer moves are waiting: then this ply is history, its verdict is
                    // all the coach and the review need, and the whole analysis goes to the
                    // position the player is actually looking at.
                    val verdict = annotator.assessMove(request.gameId, request.board, request.move, request.ply)
                    if (request.withdrawn) continue
                    results.add(verdict)
                    val after = request.board.copy()
                    after.push(request.move)
                    val behind = requests.isNotEmpty()
                    if (behind) {
                        skipped += 1
                        behindUntil = System.nanoTime() + TimeUnit.SECONDS.toNanos(BEHIND_S.toLong())
                    }
                    results.add(
                        annotator.analysisEvent(
                            request.gameId,
                            after,
                            request.ply,
                            thin = behind,
                            hurry = System.nanoTime() < behindUntil,
                        ),
                    )
                }
            } catch (exc: Exception) {
                // A failed analysis must never end a session; the game goes on unassessed.
                failures += 1
                lastError = "${exc::class.java.simpleName}: ${exc.message}"
            } finally {
                current = null
                settle()
            }
        }
    }
}
