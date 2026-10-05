package com.sidecoach.analysis

import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.events.Analysis
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import com.sidecoach.events.MoveAssessment

/**
 * From moves to assessments: what a move cost, what the position holds, what is threatened.
 * A port of `sidecoach/analysis/annotator.py`.
 *
 * [GameAnnotator] does the work for one game and is synchronous, which is what a post-game
 * review wants. [Analyser] (Analyser.kt) runs its searches on a pool of workers, which is what
 * a live session needs: the capture loop must not wait.
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
 * The first depth of a position's search told to its readers, each deeper one following as the
 * search finishes it: its best move is the full depth's in every one of 77 puzzle positions and
 * 81 % of 232 game positions, found in a twentieth of the time, so a hint, the move or the line
 * asked for at once is answered at once (Tai, 2026-09-28: "first usable answer is depth 12 ...
 * the computation continues until max depth of 18"). Depth 12 of the position's own search: a
 * search of its own to depth 12 first cost 27% more than the full depth alone (the engine
 * audit, 2026-10-04).
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
    /** How deep the search got: the depth asked for, or less for a search stopped. */
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

    /** A move taken back: the move before the next one played at [ply] is not it. */
    internal fun forgetAssessed(ply: Int) {
        assessed.remove(ply)
    }

    /**
     * Positions measured ahead of the game ([lookAhead]) and not yet reached, and how many the
     * game has reached: what looking ahead saved.
     */
    private val ahead: MutableSet<String> = java.util.concurrent.ConcurrentHashMap.newKeySet()
    @Volatile var aheadHits = 0
        internal set

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
        val measured = search(board, depth = depth, early = early, onDepth = onDepth)
        return withSide(board, measured, depth = depth)
    }

    /**
     * The position's own search, nothing beside it: its lines at [depth] (the full depth unless
     * told), the decisive-moment and only-move calls read from them. [engine]: the one to search
     * with (a worker's own; the annotator's unless told). Each depth from the first usable one is
     * kept for any reader as the search finishes it ([computed]), and handed to [onDepth].
     */
    fun search(
        board: Board,
        engine: Engine = this.engine,
        depth: Int = this.depth,
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
        return measuredLines(board, lines)
    }

    /**
     * [measured] with its side searches: what the coach's words on the position are made of
     * (the threats, the traps, the declined recaptures, what a quiet best move threatens). The
     * position's own lines are what they search from; none of them is a search of the position
     * again. Nothing when the annotator searches no threats.
     */
    fun withSide(board: Board, measured: Measured, engine: Engine = this.engine, depth: Int = this.depth): Measured {
        if (!withThreats) return measured
        val lines = measured.lines
        return Measured(
            fen = measured.fen,
            lines = lines,
            score = measured.score,
            decisive = measured.decisive,
            onlyMove = measured.onlyMove,
            threats = threatMoves(engine, board, depth),
            // The captures that look free and are not, beside the threats: what the first hint
            // warns of ("the knight on g4 is bait").
            traps = Traps.trapLines(engine, board, lines, depth = minOf(Traps.TRAP_DEPTH, depth)),
            // And the recaptures the best line declines, with what refutes them: why the obvious
            // defence fails is the lesson ("the pawn on f2 cannot take back").
            defences = Defences.defenceLines(engine, board, lines, depth = minOf(Defences.DEFENCE_DEPTH, depth)),
            // And what a quiet best move threatens: how it wins ("it threatens mate").
            threatens = aimLines(engine, board, lines, depth = minOf(AIM_DEPTH, depth)),
            key = measured.key,
        )
    }

    /** [lines] of [board] as a measurement: the calls they make (a depth on the way). */
    fun measuredLines(board: Board, lines: List<Line>): Measured = Measured(
        fen = board.fen(),
        lines = lines,
        score = positionScore(board, lines),
        decisive = isDecisive(board, lines, thresholds),
        onlyMove = isOnlyMove(board, lines, thresholds),
        key = cacheKey(board),
    )

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
        return analysisFrom(gameId, board, ply, measured, provisional = early)
    }

    /**
     * What [measured] says of [board], for the side to move: the event the readers get.
     * [provisional]: a depth on the way to the full one, or the full one before its side
     * searches -- what follows replaces it.
     */
    fun analysisFrom(gameId: String, board: Board, ply: Int, measured: Measured, provisional: Boolean = false): Analysis {
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
            provisional = provisional,
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
     * What the move cost its mover ([assessFrom]), the positions measured here when they are
     * not measured yet: the position before, and the one after only when the move is not among
     * the lines of the one before.
     */
    fun assessMove(gameId: String, before: Board, move: Move, ply: Int): MoveAssessment {
        if (!before.isLegal(move)) {
            throw IllegalArgumentException("${move.uci()} is not legal in ${before.fen()}")
        }
        val measuredBefore = measureForVerdict(before)
        var measuredAfter: Measured? = null
        if (ownLine(measuredBefore.lines, move) == null) {
            val after = before.copy()
            after.push(move)
            measuredAfter = measureForVerdict(after)
        }
        return assessFrom(gameId, before, move, ply, measuredBefore, measuredAfter)
    }

    /** Whether the position before alone judges [move]: the move is among its lines. */
    fun judgeable(measuredBefore: Measured, move: Move): Boolean = ownLine(measuredBefore.lines, move) != null

    /**
     * What the move cost its mover, from searches already made. A move among the lines of the
     * position before is judged from its own line there: the same search and the same depth as
     * the best move it is set against, and the position after read as the rest of that line --
     * no search of its own (85% of a session's moves, the same label in 319 of 320: the engine
     * audit, 2026-10-04). Any other move is judged from the position after's own measure
     * ([givenAfter], needed then), as before.
     */
    fun assessFrom(
        gameId: String,
        before: Board,
        move: Move,
        ply: Int,
        measuredBefore: Measured,
        givenAfter: Measured? = null,
    ): MoveAssessment {
        // The history comes along: the opening a game is in is a fact about its moves, and
        // the position after this move is the next move's history.
        val after = before.copy()
        after.push(move)
        val own = ownLine(measuredBefore.lines, move)
        val loss: Double
        val measuredAfter: Measured
        if (own != null) {
            val bestLine = measuredBefore.lines[0]
            loss = if (own === bestLine) 0.0 else maxOf(0.0, bestLine.expectedScore - own.expectedScore)
            val rest = lineAfter(own, 1)
            // The rest of the line is the position after's line; one that stops at the move
            // still carries the position's score.
            measuredAfter = Measured(
                fen = after.fen(),
                lines = if (rest.moves.isNotEmpty()) listOf(rest) else emptyList(),
                score = positionScore(after, listOf(rest)),
                decisive = false,
                onlyMove = false,
                key = cacheKey(after),
            )
        } else {
            measuredAfter = givenAfter
                ?: throw IllegalArgumentException("${move.uci()} is not among the lines: the position after is needed")
            loss = evalLoss(measuredBefore.lines, move.uci(), measuredAfter.score)
        }
        val tags = moveTags(before, move, measuredAfter)
        val best = if (measuredBefore.lines.isEmpty()) emptyList() else measuredBefore.lines[0].moves
        var last = assessed[ply - 1]
        if (last != null && (last.gameId != gameId || last.epdAfter != before.epd())) {
            // Another game, or a move that did not lead here: the move before is unknown.
            last = null
        }
        val label = verdict(
            before,
            move,
            measuredBefore.lines,
            measuredAfter.lines,
            loss = loss,
            beforeScore = measuredBefore.score,
            decisive = measuredBefore.decisive,
            inBook = inBook(after),
            sacrifice = tags.any { it.name == "sacrifice" },
            previous = last?.let { Previous(it.loss, it.to, it.captured) },
            thresholds = thresholds,
        )
        assessed[ply] = Assessed(gameId, after.epd(), loss, move.to, before.isCapture(move))
        return MoveAssessment(
            gameId = gameId,
            ply = ply,
            uci = move.uci(),
            expectedLoss = loss,
            severity = severityOf(loss, thresholds),
            best = best,
            tags = tags,
            depth = if (own != null) {
                measuredBefore.depth
            } else {
                minOf(measuredBefore.depth, if (measuredAfter.depth == 0) measuredBefore.depth else measuredAfter.depth)
            },
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
        return openingName(book, board.moveStack)
    }
}

/**
 * The opening [moves] from the start position are in, as the analysis carries it ("C50 Italian
 * Game"); null when the book names none. `annotator.opening_name`.
 */
fun openingName(book: Book, moves: List<Move>): String? {
    val found = book.identify(ArrayList(moves)) ?: return null
    return if (found.eco.isNotEmpty()) "${found.eco} ${found.name}" else found.name
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

/** The line among [lines] that begins with [move], or null: `annotator.own_line`. */
fun ownLine(lines: List<Line>, move: Move): Line? {
    val uci = move.uci()
    return lines.firstOrNull { it.moves.isNotEmpty() && it.moves[0] == uci }
}
