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

/**
 * Within BEHIND_S of the analyser being behind a fast game, the position the player faces is
 * measured at EARLY_DEPTH after the verdict rather than before it, the side searches with it --
 * a tenth of the full depth's time, where the full depth capped at a second went without them
 * (the live panel, 2026-09-28: a hint asked in a fast stretch waited a third of a second for
 * that second) -- and every depth follows ([Analyser]'s `serve`).
 */
const val BEHIND_S = 10.0

/**
 * While no move waits -- the player thinking, or the opponent -- the engine looks ahead: the
 * positions after the LOOK_AHEAD best moves from the position on the board, each then after the
 * best reply to it, measured whole and cached. A game that goes there has its verdict and its
 * facts at once (Tai, 2026-09-27: "cache ... the possible legal moves from both sides"); a move
 * of its own stops the look ahead within the millisecond.
 */
const val LOOK_AHEAD = 2

/**
 * The depth a position the player faces is measured to first, the full depth following: its
 * best move is the full depth's in every one of 77 puzzle positions and 81 % of 232 game
 * positions, found in a twentieth of the time (0.05-0.08 s at the phone's two threads, on the
 * desktop), so a hint, the move or the line asked for at once is answered at once (Tai,
 * 2026-09-28: "first usable answer is depth 12 ... the computation continues until max depth
 * of 18").
 */
const val EARLY_DEPTH = 12

/** One position, measured once: its lines and what they mean. */
class Measured(
    val fen: String,
    val lines: List<Line>,
    val score: Double,
    val decisive: Boolean,
    val onlyMove: Boolean,
    val threats: List<String> = emptyList(),
    val traps: List<Line> = emptyList(),
    val defences: List<Line> = emptyList(),
    val threatens: List<Line> = emptyList(),
    /**
     * What it is cached by ([cacheKey]): the FEN, and the moves since the last capture or
     * pawn move when there are any. Empty in a record kept before it had one.
     */
    val key: String = "",
) {
    /** How deep the search actually got, which the time cap can cut short. */
    val depth: Int get() = if (lines.isEmpty()) 0 else lines[0].depth
}

/**
 * What a position is cached by: its FEN, and the moves since the last capture or pawn move,
 * the ones a repetition can reach back over. The engine is told the moves that led to a
 * position and scores a repetition as the draw it is, so the same placement after different
 * quiet moves is a different search; after a capture or a pawn move nothing before can repeat
 * and the FEN is the whole key. `annotator.cache_key`.
 */
fun cacheKey(board: Board): String {
    val fen = board.fen()
    val count = minOf(board.halfmoveClock, board.moveStack.size)
    if (count == 0) return fen
    val tail = board.moveStack.subList(board.moveStack.size - count, board.moveStack.size)
    return fen + " " + tail.joinToString(" ") { it.uci() }
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

    /**
     * The positions measured whole. Concurrent, as the maps below: the worker writes them while
     * [reset] clears them from the capture thread -- a Python dict's operations are atomic
     * under the GIL, a HashMap's are not, and with the look ahead the worker is at them far
     * more often.
     */
    private val cache = java.util.concurrent.ConcurrentHashMap<String, Measured>()

    /**
     * Positions measured for a verdict only: the best line at the full depth, no candidates,
     * no threats. Kept apart from [cache], so a later whole measurement of the same position
     * is not answered with the thin one.
     */
    private val verdictCache = java.util.concurrent.ConcurrentHashMap<String, Measured>()

    /**
     * Each move assessed, by its ply: the game, the position it led to, its loss, where it
     * landed and whether it captured, so the move after it knows whether the opponent just
     * erred and whether it is merely taking back. By ply, not the last one assessed: after a
     * takeback the move before the replacement is the one the game kept, and the last one
     * assessed was the withdrawn move, a ply too far -- "the opponent erred" and "a recapture"
     * were lost on every replacement (the state review, 2026-10-01).
     */
    private class Assessed(
        val gameId: String,
        val epdAfter: String,
        val loss: Double,
        val to: Int,
        val captured: Boolean,
    )

    private val assessed = java.util.concurrent.ConcurrentHashMap<Int, Assessed>()

    /**
     * Positions measured ahead of the game ([lookAhead]) and not yet reached, and how many the
     * game has reached: what looking ahead saved.
     */
    private val ahead: MutableSet<String> = java.util.concurrent.ConcurrentHashMap.newKeySet()
    @Volatile var aheadHits = 0
        private set

    /**
     * Every search's lines as it finishes each usable depth, for any reader at any time (the
     * coach's engine reads it before it searches).
     */
    val computed = Computed()

    /**
     * How many times [forgetMeasured] has run: a search that began before it (another thread
     * changed the settings mid-search) was made the old way, and is not kept. `_era`.
     */
    @Volatile private var era = 0

    /** Forget the cache; a new game shares no positions with the last one. */
    fun reset() {
        cache.clear()
        verdictCache.clear()
        ahead.clear()
        computed.clear()
        assessed.clear()
    }

    /**
     * Forget the positions measured, keeping the game's moves: the depth, the lines, the time
     * cap, the curve or the thresholds changed, and a loss compares two positions measured the
     * same way, so none measured before is set against one measured after. Called from another
     * thread (the settings' save): a search under way now finishes, and is handed to its
     * caller, but is not kept.
     */
    fun forgetMeasured() {
        era += 1
        cache.clear()
        verdictCache.clear()
        ahead.clear()
        computed.clear()
    }

    /**
     * A position the game may reach, measured whole before it does, as the game's own positions
     * are, and cached with them. Inside [Engine.interruptible] a search stopped throws
     * [Interrupted], and nothing is cached.
     */
    fun lookAhead(board: Board): Measured {
        cache[cacheKey(board)]?.let { return it }
        val measured = measure(board)
        if (cache.containsKey(measured.key)) ahead.add(measured.key)
        return measured
    }

    /**
     * The position measured whole, counting a position measured ahead reached. A position
     * adopted without a key ([adopt]) is found by its FEN.
     */
    private fun cached(board: Board): Measured? {
        val key = cacheKey(board)
        val measured = cache[key] ?: cache[board.fen()]
        if (measured != null && key in ahead) {
            ahead.remove(key)
            aheadHits += 1
        }
        return measured
    }

    /**
     * The position measured for a verdict: whole if it already is, else its best line alone
     * at the full depth -- what [evalLoss] needs, at a fraction of the cost. When the engine
     * is behind a fast game, this is all a stale ply gets.
     */
    fun measureForVerdict(board: Board): Measured {
        val key = cacheKey(board)
        cached(board)?.let { return it }
        verdictCache[key]?.let { return it }
        val began = era
        val lines = engine.analyse(board, depth = depth, multipv = 1)
        val thin = Measured(
            fen = board.fen(),
            lines = lines,
            score = positionScore(board, lines),
            decisive = false,
            onlyMove = false,
            key = key,
        )
        if (began == era) {  // not searched the old way ([forgetMeasured])
            computed.add(board, lines, thin = true)
            verdictCache[key] = thin
        }
        return thin
    }

    /**
     * The position whole, at the full depth: the candidates, the threats, the decisive-moment
     * and only-move calls. `onDepth`: handed the candidates of each depth the search finishes
     * on its way. A position the player is not to wait for is measured early first
     * ([measureEarly]), never in place of this.
     */
    fun measure(board: Board, onDepth: ((List<Line>) -> Unit)? = null): Measured {
        cached(board)?.let { return it }
        val began = era
        val measured = measuredAt(board, depth, onDepth = onDepth)
        if (began == era) cache[measured.key] = measured  // not searched the old way
        return measured
    }

    /**
     * The position whole at [EARLY_DEPTH], its side searches no deeper: what the player can be
     * told while the full depth is searched. Not cached: the full depth follows.
     */
    fun measureEarly(board: Board): Measured = measuredAt(board, minOf(EARLY_DEPTH, depth), early = true)

    /** `_measured`: the position whole at [depth], its side searches no deeper than it. */
    private fun measuredAt(
        board: Board,
        depth: Int,
        early: Boolean = false,
        onDepth: ((List<Line>) -> Unit)? = null,
    ): Measured {
        val usable = minOf(EARLY_DEPTH, depth)
        val began = era
        // A depth done: kept from the first usable one, and handed on.
        val finished: (List<Line>) -> Unit = { found ->
            if (found.isNotEmpty() && found[0].depth >= usable && began == era) computed.add(board, found)
            onDepth?.invoke(found)
        }
        val lines = engine.analyse(
            board,
            depth = depth,
            multipv = multipv,
            // The best line carried on to its point costs seconds; the full depth does it.
            carryBest = !early,
            onDepth = finished,
        )
        // Before the side searches: the lines are the news, and a reader has them now.
        if (began == era) computed.add(board, lines)
        val searched = withThreats
        return Measured(
            fen = board.fen(),
            lines = lines,
            score = positionScore(board, lines),
            decisive = isDecisive(board, lines, thresholds),
            onlyMove = isOnlyMove(board, lines, thresholds),
            threats = if (searched) threatMoves(engine, board, depth) else emptyList(),
            // The captures that look free and are not, beside the threats: what the first hint
            // warns of ("the knight on g4 is bait"), searched only when not hurried.
            traps = if (searched) Traps.trapLines(engine, board, lines, depth = minOf(Traps.TRAP_DEPTH, depth)) else emptyList(),
            // And the recaptures the best line declines, with what refutes them: why the obvious
            // defence fails is the lesson ("the pawn on f2 cannot take back").
            defences = if (searched) {
                Defences.defenceLines(engine, board, lines, depth = minOf(Defences.DEFENCE_DEPTH, depth))
            } else {
                emptyList()
            },
            // And what a quiet best move threatens: how it wins ("it threatens mate").
            threatens = if (searched) aimLines(engine, board, lines, depth = minOf(AIM_DEPTH, depth)) else emptyList(),
            key = cacheKey(board),
        )
    }

    /**
     * What the position holds, for the side to move. `thin`: only what the verdict already
     * measured -- the analysis of a position the game has left, so the coach still has the
     * facts of the position each move was played from and the review has its score. `early`:
     * at [EARLY_DEPTH] ([measureEarly]), a provisional analysis the full depth follows.
     */
    fun analysisEvent(
        gameId: String,
        board: Board,
        ply: Int,
        thin: Boolean = false,
        early: Boolean = false,
        onDepth: ((List<Line>) -> Unit)? = null,
    ): Analysis {
        val measured = when {
            early -> measureEarly(board)
            thin -> measureForVerdict(board)
            else -> measure(board, onDepth)
        }
        return Analysis(
            gameId = gameId,
            fen = measured.fen,
            ply = ply,
            lines = measured.lines,
            decisive = measured.decisive,
            onlyMove = measured.onlyMove,
            threats = measured.threats,
            traps = measured.traps,
            defences = measured.defences,
            threatens = measured.threatens,
            tags = positionTags(board, measured),
            opening = opening(board),
            inBook = inBook(board),
            provisional = early,
        )
    }

    /**
     * The early analysis of [board] with the lines of a deeper depth: its candidates, the
     * decisive-moment and only-move calls and the pattern read from them again; its threats,
     * traps and defences the early depth's, until the full depth brings its own.
     */
    fun deepened(early: Analysis, board: Board, lines: List<Line>): Analysis {
        val measured = Measured(
            fen = early.fen,
            lines = lines,
            score = positionScore(board, lines),
            decisive = isDecisive(board, lines, thresholds),
            onlyMove = isOnlyMove(board, lines, thresholds),
        )
        // `early.model_copy(update=...)`: every other field the early analysis's.
        return Analysis(
            gameId = early.gameId,
            fen = early.fen,
            ply = early.ply,
            lines = lines,
            decisive = measured.decisive,
            onlyMove = measured.onlyMove,
            threats = early.threats,
            traps = early.traps,
            defences = early.defences,
            threatens = early.threatens,
            tags = positionTags(board, measured),
            opening = early.opening,
            inBook = early.inBook,
            provisional = early.provisional,
        ).also {
            it.ts = early.ts
            it.sourceFrameId = early.sourceFrameId
        }
    }

    /** Whether the position is measured whole already (the game's own, or ahead of it). */
    fun measured(board: Board): Boolean =
        cache.containsKey(cacheKey(board)) || cache.containsKey(board.fen())

    /**
     * Positions measured whole by another annotator -- another engine, on another thread, for a
     * review measuring a game's positions side by side (`GameReview`) -- kept as if measured
     * here: the walk through the game's moves then searches nothing.
     */
    fun adopt(measured: Iterable<Measured>) {
        for (m in measured) cache[m.key.ifEmpty { m.fen }] = m
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
        var last = assessed[ply - 1]
        if (last != null && (last.gameId != gameId || last.epdAfter != before.epd())) {
            // Another game, or a move that did not lead here: the move before is unknown.
            last = null
        }
        val erred = last?.loss ?: 0.0
        // Taking back on the square the opponent just captured on. Taking a piece that merely
        // moved there is not obvious: that is how a blunder gets punished.
        val captures = before.isCapture(move)
        val recapture = last != null && last.captured && captures && move.to == last.to
        // A capture nobody can take back is a grab, not a find; a king's capture always is.
        // Taking back is read by the rules: a piece guarded only by one pinned to its own king
        // is taken for nothing.
        val freeGrab = captures && !Exchange.canRetake(before, move)
        val obvious = recapture || freeGrab || before.legalMoves().size == 1
        val top = measuredBefore.lines.firstOrNull()
        val afterTop = measuredAfter.lines.firstOrNull()
        // A forced mate the best move had, and the move played let go; one the move walks into.
        // The mates are the lines' own: the curve puts a thousand centipawns a fortieth of a
        // game below a mate, and a mate in four let go read as an inaccuracy.
        val mating = top?.mate != null && top.mate > 0
        val mateMissed = mating && !after.isCheckmate() && !(afterTop?.mate != null && afterTop.mate < 0)
        val mateAllowed = afterTop?.mate != null && afterTop.mate > 0 && !(top?.mate != null && top.mate < 0)
        // What the best move had to find: a mate, material won, or -- after the opponent's
        // error -- any material at all (chess.com's miss: "a tactical opportunity or a chance
        // to punish the opponent").
        val won = if (top != null) materialWon(before, top.moves) else 0
        val chance = mating || won >= TACTIC_GAIN || (erred >= thresholds.mistake && won >= 1)
        val label = labelMove(
            loss,
            thresholds,
            isBest = best.isNotEmpty() && best[0] == move.uci(),
            inBook = inBook(after),
            decisive = measuredBefore.decisive,
            sacrifice = tags.any { it.name == "sacrifice" },
            beforeScore = measuredBefore.score,
            opponentBlundered = erred >= thresholds.blunder,
            secondScore = measuredBefore.lines.getOrNull(1)?.expectedScore,
            chance = chance,
            mateMissed = mateMissed,
            mateAllowed = mateAllowed,
            obvious = obvious,
        )
        assessed[ply] = Assessed(gameId, after.epd(), loss, move.to, captures)
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
class Analyser(
    val annotator: GameAnnotator,
    private val maxPending: Int = 64,
    /**
     * How many of the best moves from the position on the board are looked ahead at
     * ([LOOK_AHEAD] in a live session; none unless asked).
     */
    private val lookAhead: Int = 0,
) : AutoCloseable {

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
    /** What the worker has finished; a test puts a late word here, as the desktop's put one in `_results`. */
    internal val results = ConcurrentLinkedQueue<Event>()
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
        /**
         * The player's own move: judged first, its word the news; the opponent's is not, and
         * the position after it, which the player faces, is measured early first.
         */
        val byUser: Boolean = true,
    ) {
        /**
         * Set when the game takes the move back while it is being measured ([withdraw]):
         * nothing about it is delivered, and the position after it is not measured.
         */
        @Volatile var withdrawn = false
    }

    /** The request the worker is on, so a takeback can let go of it. */
    @Volatile private var current: Request? = null

    /**
     * The positions to look ahead at, with how many moves ahead each is, and whether the worker
     * is looking ahead now (a request then interrupts it). The worker alone changes the list;
     * a new game or a takeback puts a new one in its place.
     */
    // Internal, as the desktop test reaches its _ahead: a test drives the look ahead a step at a time.
    @Volatile internal var ahead: MutableList<Pair<Board, Int>> = ArrayList()

    /**
     * The game's positions whose full measure gave way to a newer request ([yielding]), measured
     * again once the worker is free; changed with the count of work owed, under [countLock].
     */
    private val deferred = ArrayList<Pair<Request, Board>>()

    /**
     * Whether the worker is on a search a request interrupts: a look ahead, or a full measure of
     * one of the game's positions.
     */
    @Volatile internal var looking = false
        private set

    /** Positions measured ahead. */
    @Volatile var looked = 0
        private set

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
            if (looking) annotator.engine.interrupt()
            worker.join(5000)
        }
        annotator.engine.close()
    }

    fun submitMove(gameId: String, before: Board, move: Move, ply: Int, byUser: Boolean = true) {
        // With its moves: the book is asked with the moves that led here (`inBook`,
        // `opening`), and a copy without them was in the book on the first move alone.
        submit(Request(gameId, ply, before.copy(), move, byUser))
    }

    fun submitPosition(gameId: String, board: Board, ply: Int) {
        submit(Request(gameId, ply, board.copy(withStack = false), null))
    }

    /**
     * A new game, or the board re-read: drop what is queued and forget the measured positions,
     * since none of it is about the game as it now stands. What the worker is busy with is let
     * go as a takeback lets go of it ([withdraw]): nothing about it is delivered, and it is not
     * put off to be measured later. It used to finish and be delivered -- or, its full measure
     * interrupted by the next request, be put off after the reset and measured again after the
     * new position, its analysis then the last word; on a resync of the same game (the side to
     * move settled by the clock) the profile filed it as the new position's (the state review,
     * 2026-10-01).
     */
    fun reset() {
        // Everything queued goes, the stop that [close] puts there included, as the desktop's
        // `get_nowait` loop takes it.
        while (true) {
            requests.poll() ?: break
            settle()
        }
        ahead = ArrayList()
        val now = synchronized(countLock) {
            outstanding = maxOf(0, outstanding - deferred.size)
            deferred.clear()
            current?.also { it.withdrawn = true }
        }
        if (now != null) annotator.engine.interrupt()
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
        ahead = ArrayList()  // ahead of moves no longer played
        // Under the lock the worker takes from the list with ([next]): an ArrayList walked while
        // another thread takes from it throws.
        synchronized(countLock) {
            for ((request, _) in deferred) {
                if (request.gameId == gameId && request.ply > ply) request.withdrawn = true
            }
        }
        val now = current
        if (now != null && now.gameId == gameId && now.ply > ply) {
            now.withdrawn = true
            withdrawn += 1
            // Its search under way stops now, and none of its searches starts after it: all of a
            // request's work runs interruptible ([run]).
            annotator.engine.interrupt()
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
            // matters more than an old one, so the oldest goes -- whatever it is, as the
            // desktop's `get_nowait` takes it.
            if (requests.poll() != null) {
                dropped += 1
                settle()
            }
        }
        synchronized(countLock) { outstanding += 1 }
        requests.put(request)
        // After the put: a worker that starts looking ahead after this sees the request.
        if (looking) annotator.engine.interrupt()
    }

    private fun settle() {
        synchronized(countLock) { outstanding = maxOf(0, outstanding - 1) }
    }

    private fun run() {
        while (true) {
            val request = next()
            if (request === stop) return
            current = request
            try {
                // A takeback stops any of the request's searches ([withdraw]), a newer request only
                // its full measure ([yielding], [looking]).
                annotator.engine.interruptible { serve(request) }
            } catch (exc: Interrupted) {
                // Withdrawn by a takeback: nothing about it is delivered.
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

    /**
     * One request: a position measured early and then whole; a move judged, and the position
     * after it measured -- early first when the player faces it.
     */
    private fun serve(request: Request) {
        if (request.move == null) {
            val early = early(request, request.board)
            if (request.withdrawn) return
            val deepening = deepening(request, request.board, early)
            val event = yielding(request, request.board, deepening)
            if (event != null && !request.withdrawn) {
                results.add(event)
                planAhead(request.board, event.lines)
            }
        } else {
            // The verdict first, from the best line alone; the position whole after -- unless
            // newer moves are waiting: then this ply is history, its verdict is all the coach and
            // the review need, and the whole analysis (candidates, threats, the decisive-moment
            // call) goes to the position the player is actually looking at. The position after
            // the opponent's move is the one the player faces: measured early first.
            val after = request.board.copy()
            after.push(request.move)
            var early: Analysis? = null
            if (!request.byUser && requests.isEmpty()) early = this.early(request, after)
            if (request.withdrawn) return
            val verdict = annotator.assessMove(request.gameId, request.board, request.move, request.ply)
            if (request.withdrawn) return
            results.add(verdict)
            if (requests.isNotEmpty()) {
                skipped += 1
                behindUntil = System.nanoTime() + TimeUnit.SECONDS.toNanos(BEHIND_S.toLong())
                val event = annotator.analysisEvent(request.gameId, after, request.ply, thin = true)
                if (!request.withdrawn) results.add(event)
                return
            }
            if (early == null && !request.byUser && System.nanoTime() < behindUntil) {
                // Behind a moment ago: the verdict came first, and now the first usable depth of
                // the position the player faces. It is never the last word: every depth follows,
                // then the look ahead. Measured at EARLY_DEPTH and kept as the whole, it was never
                // deepened and nothing was looked ahead at, and the next move's verdict set depth
                // 12 against depth 18 -- every position of the panel's puzzle room after the
                // first, its move and reply submitted together (the state review, 2026-10-01;
                // Tai, 2026-09-28: "first usable answer is depth 12 ... the computation continues
                // until max depth of 18").
                early = this.early(request, after, hurried = true)
                if (request.withdrawn) return
            }
            val event = yielding(request, after, deepening(request, after, early)) ?: return
            if (request.withdrawn) return
            results.add(event)
            planAhead(after, event.lines)
        }
    }

    /**
     * The position measured at [EARLY_DEPTH] and delivered, provisional, unless it is measured
     * whole already (ahead of the game). Before the verdict only when the game was not behind a
     * moment ago; [hurried], after it ([serve]).
     */
    private fun early(request: Request, board: Board, hurried: Boolean = false): Analysis? {
        if (annotator.measured(board) || (!hurried && System.nanoTime() < behindUntil)) return null
        val event = annotator.analysisEvent(request.gameId, board, request.ply, early = true)
        if (!request.withdrawn) results.add(event)
        return event
    }

    /**
     * Each depth between the early one and the full delivered as the search finishes it,
     * provisional ([GameAnnotator.deepened]): whoever reads gets the deepest search done so far
     * (Tai, 2026-09-28: "if the consumers read after stockfish reaches depth > 12 then they will
     * get whatever lines that is available at the time"). The results queue is the hand-over:
     * each depth goes in whole, the reader takes the newest.
     */
    private fun deepening(request: Request, board: Board, early: Analysis?): ((List<Line>) -> Unit)? {
        if (early == null) return null
        return { lines ->
            val depth = if (lines.isNotEmpty()) lines[0].depth else 0
            if (!request.withdrawn && EARLY_DEPTH < depth && depth < annotator.depth) {
                results.add(annotator.deepened(early, board, lines))
            }
        }
    }

    /**
     * The next request, and while none waits, the positions ahead measured one by one
     * ([lookAhead]), a request interrupting the one being measured.
     */
    private fun next(): Request {
        while (true) {
            requests.poll()?.let { return it }
            val deferred = synchronized(countLock) { if (this.deferred.isNotEmpty()) this.deferred.removeAt(0) else null }
            if (deferred != null) {
                resume(deferred.first, deferred.second)
                continue
            }
            val ahead = ahead
            if (ahead.isEmpty()) return requests.take()
            val (board, moves) = ahead.removeAt(0)
            lookAhead(board, moves, ahead)
        }
    }

    /**
     * The game's position measured whole, giving way to any newer request: the game has moved
     * on, and its new position comes first. Interrupted, it waits in [deferred] for the worker
     * to be free again, still owed ([pending]), and null is returned (Tai's puzzles, 2026-09-28:
     * the site replies 0.6 s after the player's move and a puzzle's setup move comes 0.6 s
     * after it appears, and the position the player then faced waited behind the full measure
     * of the one before it -- up to five seconds to its first answer).
     */
    private fun yielding(request: Request, board: Board, onDepth: ((List<Line>) -> Unit)? = null): Analysis? {
        looking = true
        try {
            return annotator.engine.interruptible {
                if (requests.isNotEmpty()) throw Interrupted()
                annotator.analysisEvent(request.gameId, board, request.ply, onDepth = onDepth)
            }
        } catch (exc: Interrupted) {
            synchronized(countLock) {
                outstanding += 1
                deferred.add(request to board)
            }
            return null
        } finally {
            looking = false
        }
    }

    /**
     * A position whose full measure gave way, measured again now the worker is free -- the
     * request current meanwhile, so a takeback lets go of it as of any other.
     */
    private fun resume(request: Request, board: Board) {
        current = request
        try {
            if (request.withdrawn) return
            val event = yielding(request, board)
            if (event != null && !request.withdrawn) results.add(event)
        } catch (exc: Exception) {
            failures += 1
            lastError = "${exc::class.java.simpleName}: ${exc.message}"
        } finally {
            current = null
            settle()
        }
    }

    /**
     * The positions to look ahead at from [board], the position the game is at: after each of
     * its [lookAhead] best moves first; after the best reply to each follows as each is
     * measured ([lookAhead]).
     */
    internal fun planAhead(board: Board, lines: List<Line>) {
        val ahead = ArrayList<Pair<Board, Int>>()
        for (line in lines.take(lookAhead)) {
            val move = lineMoves(line).take(1)
            if (move.isNotEmpty() && board.isLegal(move[0])) {
                val after = board.copy()  // with its moves: the engine reads repetitions in them
                after.push(move[0])
                ahead.add(after to 1)
            }
        }
        this.ahead = ahead
    }

    /**
     * One position ahead measured, unless a request is waiting or comes meanwhile; one move
     * ahead, the best reply to it goes next in [ahead], the plan it came from -- unless the game
     * has planned anew meanwhile (a new game, a takeback).
     */
    internal fun lookAhead(board: Board, moves: Int, ahead: MutableList<Pair<Board, Int>>) {
        val engine = annotator.engine
        looking = true
        val measured = try {
            // Inside the interruptible searches, so a request put before this is seen here and
            // one put after it interrupts ([submit]).
            engine.interruptible {
                if (requests.isNotEmpty()) null else annotator.lookAhead(board)
            } ?: return
        } catch (exc: Interrupted) {
            return  // the game moved: its own position comes first, and plans anew
        } catch (exc: Exception) {
            lastError = "${exc::class.java.simpleName}: ${exc.message}"
            return
        } finally {
            looking = false
        }
        looked += 1
        val reply = if (measured.lines.isNotEmpty()) lineMoves(measured.lines[0]).take(1) else emptyList()
        if (moves == 1 && reply.isNotEmpty() && board.isLegal(reply[0]) && ahead === this.ahead) {
            val after = board.copy()
            after.push(reply[0])
            ahead.add(0, after to 2)
        }
    }
}
