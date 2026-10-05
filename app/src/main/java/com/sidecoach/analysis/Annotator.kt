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
 * review wants. [Analyser] (Analyser.kt) runs the same searches on a pool of workers, which is
 * what a live session needs: the capture loop must not wait.
 *
 * Every search is one line to the full depth (the engine audit, 2026-10-04/05). Asking
 * Stockfish for the exact value of several moves at once -- a search of several lines --
 * explodes on decided and forcing positions: two lines of one real position took 53 s, one line
 * 0.18 s; 84 of 8,886 positions of real sessions took over 5 s with six lines at depth 18, the
 * worst 47 s. So a position is measured whole as:
 *
 * - its best line: one line to the full depth (the verdict, the best move);
 * - its candidates: the best moves of a six-line search to [EARLY_DEPTH] (never over 1.9 s),
 *   each then searched as its own position, one line to the full depth -- the position's lines
 *   at the full depth, built from those, and exactly the positions the game may go to next;
 * - its side searches (threats, traps, declined recaptures, what a quiet best move threatens).
 *
 * Every position is measured once, to the same depth, and cached, so the position after one
 * move is the position before the next, and a candidate searched is the next position measured.
 * The matching depth is what makes the loss of a move meaningful: every value a verdict compares
 * is one position's own one-line search to the full depth.
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
 * The depth of a position's first answer: its six candidates at once, the full depth to follow.
 * Its best move is the full depth's in every one of 77 puzzle positions and 81 % of 232 game
 * positions (Tai, 2026-09-28: "first usable answer is depth 12 ... the computation continues
 * until max depth of 18"), and six lines to it never took more than 1.9 s on any of 8,886
 * positions of real sessions (p99 0.29 s; the engine audit, 2026-10-05).
 */
const val EARLY_DEPTH = 12

/**
 * Lines a reader may set against a full-depth search: none of them the first answer's hint, at
 * [EARLY_DEPTH] (the audits of 2026-10-05: the hint's 12 set against a reply searched to 18 read
 * a move as good as the best as worse, or better, by what the depths differ by). `at_full_depth`.
 */
fun atFullDepth(lines: List<Line>): Boolean = lines.isNotEmpty() && lines.minOf { it.depth } > EARLY_DEPTH

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
    // Saved from the settings' thread, read by the analyser's workers: volatile, as the desktop's
    // are under the GIL.
    @Volatile var depth: Int = 18,
    @Volatile var multipv: Int = 6,
    @Volatile var thresholds: Thresholds = Thresholds(),
    @Volatile var tagger: Tagger? = null,
    /** Set late when the tables arrive after the engine was built, read by the worker. */
    @Volatile var book: Book? = null,
    @Volatile var withThreats: Boolean = true,
) {

    /**
     * The positions measured whole. Concurrent, as the maps below: the workers write them while
     * [reset] clears them from the capture thread -- a Python dict's operations are atomic
     * under the GIL, a HashMap's are not.
     */
    private val cache = java.util.concurrent.ConcurrentHashMap<String, Measured>()

    /**
     * Each position's best line alone ([bestLine]): what a verdict reads of the position a move
     * led to, and a candidate's value for the position before it. Kept apart from [cache], so a
     * later whole measurement of the same position is not answered with it.
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
     * Every search's lines as it finishes, for any reader at any time (the coach's engine reads
     * it before it searches).
     */
    val computed = Computed()

    /**
     * How many times [forgetMeasured] has run: a search that began before it (another thread
     * changed the settings mid-search) was made the old way, and is not kept. `_era`.
     */
    @Volatile var era = 0
        private set

    /** What the caches are written under, the era checked there ([keep]). */
    private val lock = Object()

    /** Forget the cache; a new game shares no positions with the last one. */
    fun reset() {
        synchronized(lock) {
            cache.clear()
            verdictCache.clear()
            computed.clear()
            assessed.clear()
        }
    }

    /**
     * [measured] kept for readers and for the next asking -- unless the settings changed since its
     * search began ([began]): checked and written under the lock, so a change in between cannot
     * let an old search's answer in. `_keep`.
     */
    private fun keep(began: Int, board: Board, measured: Measured, whole: Boolean) {
        synchronized(lock) {
            if (began != era) return
            computed.add(board, measured.lines, thin = !whole)
            (if (whole) cache else verdictCache)[measured.key.ifEmpty { cacheKey(board) }] = measured
        }
    }

    /**
     * Forget the positions measured, keeping the game's moves: the depth, the lines, the curve
     * or the thresholds changed, and a loss compares two positions measured the same way, so
     * none measured before is set against one measured after. Called from another thread (the
     * settings' save): a search under way now finishes, and is handed to its caller, but is not
     * kept.
     */
    fun forgetMeasured() {
        synchronized(lock) {
            era += 1
            cache.clear()
            verdictCache.clear()
            computed.clear()
        }
    }

    // ---- the searches ----------------------------------------------------------------------

    /**
     * The position's best line: one line to the full depth, carried on to its point (the coach
     * reads the line for it). What a verdict reads of a position, and a candidate's value for
     * the position before it. A finished position has no line; a position with one legal move
     * has that move's line, from its own position's search -- the move needs no search to be
     * chosen (the slowest one-line search of 8,886 was such a position: 10.6 s). A position
     * measured whole answers with its whole, whose first line is this one. Cached; [onDepth] is
     * handed each depth the search finishes on its way.
     */
    fun bestLine(board: Board, engine: Engine = this.engine, onDepth: ((List<Line>) -> Unit)? = null): Measured {
        cached(board)?.let { return it }
        verdictCache[cacheKey(board)]?.let { return it }
        val began = era
        val legal = board.legalMoves()
        val measured = when {
            terminalExpectedScore(board) != null || legal.isEmpty() -> oneLine(board, emptyList())
            legal.size == 1 -> {
                val child = board.copy()
                child.push(legal[0])
                fromOnlyMove(board, legal[0], bestLine(child, engine, onDepth))
            }
            else -> oneLine(board, engine.analyse(board, depth = depth, multipv = 1, carryBest = true, onDepth = onDepth))
        }
        keep(began, board, measured, whole = false)
        return measured
    }

    /**
     * The best line of a position whose one legal move is [move]: that move's, from the position
     * it leads to ([child], its best line). `from_only_move`.
     */
    fun fromOnlyMove(board: Board, move: Move, child: Measured): Measured =
        oneLine(board, listOf(candidateLine(board, move, child)))

    private fun oneLine(board: Board, lines: List<Line>): Measured = Measured(
        fen = board.fen(),
        lines = lines,
        score = positionScore(board, lines),
        decisive = false,
        onlyMove = false,
        key = cacheKey(board),
    )

    /** What a verdict reads of a position: whole if it already is, else its best line. */
    fun measureForVerdict(board: Board): Measured = bestLine(board)

    /**
     * The position's candidates at [EARLY_DEPTH]: [multipv] lines, the first answer to a hint
     * and the moves whose positions the full depth then searches ([candidates]). None for a
     * position with one legal move or none: its best line says all there is. Not kept for readers
     * ([computed]): a reader takes the position whole, or its best line.
     */
    fun hintLines(board: Board, engine: Engine = this.engine): List<Line> {
        if (terminalExpectedScore(board) != null || board.legalMoves().size < 2) return emptyList()
        return engine.analyse(board, depth = minOf(EARLY_DEPTH, depth), multipv = multipv)
    }

    /**
     * The moves the position is measured whole by: its best line's move first, then the hint's
     * in its order, [multipv] at most, each legal and once.
     */
    fun candidates(board: Board, best: Measured, hint: List<Line>): List<Move> {
        val out = ArrayList<Move>()
        for (line in best.lines.take(1) + hint) {
            val move = firstMove(line)
            if (move != null && board.isLegal(move) && move !in out) out.add(move)
            if (out.size == multipv) break
        }
        return out
    }

    /**
     * [move]'s line for the side to move in [board], from its own position's search ([child],
     * measured for the side to move there): the move, then the child's best line; every score
     * turned to this side -- the expected score and the win-draw-loss mirrored, the centipawns
     * negated, a mate the child's side gives a mate this side takes, and the reverse a move
     * longer. A finished child is scored as it stands.
     */
    fun candidateLine(board: Board, move: Move, child: Measured): Line {
        val after = board.copy()
        after.push(move)
        val finished = terminalExpectedScore(after)
        val line = child.lines.firstOrNull()
        if (finished != null || line == null) {
            val score = 1.0 - (finished ?: child.score)
            return Line(
                moves = listOf(move.uci()),
                depth = depth,
                expectedScore = score,
                mate = if (after.isCheckmate()) 1 else null,
            )
        }
        val mate = line.mate?.let { if (it > 0) -it else Math.abs(it) + 1 }
        return Line(
            moves = listOf(move.uci()) + line.moves,
            depth = line.depth,
            expectedScore = 1.0 - line.expectedScore,
            scoreCp = line.scoreCp?.let { -it },
            mate = mate,
            wdl = line.wdl?.let { Triple(it.third, it.second, it.first) },
        )
    }

    /**
     * The position whole: its best line first -- its own search, which picks the best move more
     * often than its candidates' searches do (231 against 222 of 300 positions against a depth-24
     * truth, the engine audit 2026-10-05), so the engine's own move is never charged a loss --
     * then its other candidates from their own positions' searches, best first, a shorter mate
     * ahead of a longer one; the decisive-moment and only-move calls read from them. No side
     * searches ([withSide]).
     */
    fun whole(board: Board, best: Measured, children: List<Pair<Move, Measured>>): Measured {
        val first = best.lines.take(1)
        val shown = first.mapNotNull { it.moves.firstOrNull() }.toSet()
        // Stable: the hint's order on ties.
        val rest = children.filter { (move, _) -> move.uci() !in shown }
            .map { (move, child) -> candidateLine(board, move, child) }
            .sortedWith(LINE_ORDER)
        return measuredLines(board, first + rest)
    }

    /**
     * The position whole, at the full depth: its best line, its candidates' own searches and its
     * side searches (the file's docstring). Cached, as every search under it.
     */
    fun measure(board: Board): Measured {
        cached(board)?.let { return it }
        val began = era
        val best = bestLine(board)
        val hint = hintLines(board)
        val children = candidates(board, best, hint).map { move ->
            val child = board.copy()
            child.push(move)
            move to bestLine(child)
        }
        val measured = withSide(board, whole(board, best, children))
        keep(began, board, measured, whole = true)
        return measured
    }

    /**
     * [measured] with its side searches: what the coach's words on the position are made of
     * (the threats, the traps, the declined recaptures, what a quiet best move threatens). The
     * position's own lines are what they search from; none of them is a search of the position
     * again. Nothing when the annotator searches no threats.
     */
    fun withSide(board: Board, measured: Measured, engine: Engine = this.engine): Measured {
        if (!withThreats || terminalExpectedScore(board) != null) return measured
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

    // ---- what the readers get --------------------------------------------------------------

    /**
     * What the position holds, for the side to move. `thin`: only its best line (no candidates,
     * no threats, no decisive call) -- what a verdict measured.
     */
    fun analysisEvent(gameId: String, board: Board, ply: Int, thin: Boolean = false): Analysis {
        val measured = if (thin) measureForVerdict(board) else measure(board)
        return analysisFrom(gameId, board, ply, measured)
    }

    /**
     * What [measured] says of [board], for the side to move: the event the readers get.
     * [provisional]: the position not whole yet (its first answer, or whole but for its side
     * searches), what follows replacing it.
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
     * The position's first answer, before its candidates are searched: the hint's lines at
     * [EARLY_DEPTH] when they are in, else the best line -- one search's lines, never two
     * searches' set against each other (the calls, a probe's distance from the best, the lines
     * best first all compare them); the calls read from them.
     */
    fun firstAnswer(board: Board, best: Measured?, hint: List<Line>): Measured =
        measuredLines(board, hint.ifEmpty { best?.lines ?: emptyList() })

    /** [lines] of [board] as a measurement: the calls they make. */
    fun measuredLines(board: Board, lines: List<Line>): Measured = Measured(
        fen = board.fen(),
        lines = lines,
        score = positionScore(board, lines),
        decisive = isDecisive(board, lines, thresholds),
        onlyMove = isOnlyMove(board, lines, thresholds),
        key = cacheKey(board),
    )

    /** The position measured whole, by its key or (adopted without one) its FEN. */
    private fun cached(board: Board): Measured? = cache[cacheKey(board)] ?: cache[board.fen()]

    /** Whether the position is measured whole already. */
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

/**
 * A line's place among a position's lines: the better first; of equal score, a mate given sooner,
 * then no mate, then a mate taken later (every mate scores alike). `_rank`.
 */
private val LINE_ORDER: Comparator<Line> = compareBy<Line>({ -it.expectedScore }, { line ->
    val mate = line.mate
    when {
        mate == null -> 1_000_000
        mate > 0 -> mate
        else -> 2_000_000 + mate
    }
})

/**
 * [lines] best lines of a position on one engine of its own, measured whole the one-line way
 * ([GameAnnotator.measure], no side searches) -- never several lines to the full depth at once,
 * which explode on decided and forcing positions. For what searches with no analyser beside it: a
 * review's line view, a coach made on demand. Each depth asked for keeps its own measurements.
 * `annotator.WholeSearcher`.
 */
class WholeSearcher(val engine: Engine, private val lines: Int, private val depth: Int) {
    private val annotators = HashMap<Int, GameAnnotator>()

    /** [board]'s lines whole; with [bestOnly], its best line alone (or its whole, if measured). */
    fun analyse(board: Board, depth: Int? = null, bestOnly: Boolean = false): List<Line> {
        val at = depth ?: this.depth
        val annotator = synchronized(annotators) {
            annotators.getOrPut(at) { GameAnnotator(engine, depth = at, multipv = lines, withThreats = false) }
        }
        if (bestOnly) return annotator.bestLine(board).lines
        return annotator.measure(board).lines
    }

    fun close() = engine.close()
}

/** A line's first move, or null for a line with none or one that does not parse. */
private fun firstMove(line: Line): Move? = line.moves.firstOrNull()?.let { Move.fromUciOrNull(it) }

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
