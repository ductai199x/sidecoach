package com.sidecoach.analysis

import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.PgnGame
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.chess.WHITE
import com.sidecoach.events.Analysis
import com.sidecoach.events.Label
import com.sidecoach.events.MotifTag
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.ReviewedGame
import com.sidecoach.events.ReviewedMove
import com.sidecoach.events.Severity
import com.sidecoach.events.Side
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean

/**
 * A finished game reviewed whole (`analysis/game_review.py`): every position of it measured by
 * several engines at once, then every move judged in order from what they found -- the numbers
 * behind the review's summary, its step-through, and the coach's words on every move.
 *
 * The live analyser judges a move as the game goes, one engine, one position at a time. A
 * review has the whole game at once, and a game's positions are independent searches, so a few
 * engines measure them side by side -- whole, with the threats and traps the live coach's facts
 * are made of. One annotator then walks the moves in order from those measurements, which costs
 * no search.
 */

/** How hard the engines look at every position: chess.com's Fast, Standard and Deep. */
data class Strength(val name: String, val depth: Int, val multipv: Int)

object GameReview {
    val STRENGTHS: Map<String, Strength> = linkedMapOf(
        "fast" to Strength("fast", 14, 3),
        "standard" to Strength("standard", 18, 6),
        "deep" to Strength("deep", 22, 6),
    )

    // Each engine's threads: one. At a fixed depth more threads widen the search instead of
    // speeding it -- four took 7.7 times the processor time of one, and twice as long, for an
    // answer no better (the engine audit, 2026-10-04). Engines side by side take up to two
    // thirds of the cores between them, the rest left to everything else the device does.
    const val THREADS = 1
    // Each engine loads its own copy of the 98 MB network: on a phone, beside the game's own
    // workers, more than three side by side is low-memory-killer territory (the audit of
    // 2026-10-05; the desktop's eight).
    const val MAX_WORKERS = 3

    /** How often a review running with a stop looks at it: the searches under way are cut at most this long after the stop. */
    const val STOP_LOOK_MS = 200L

    /** How many engines measure side by side on a device with [cores] cores. `workers_for`. */
    fun workersFor(cores: Int = Runtime.getRuntime().availableProcessors()): Int =
        maxOf(1, minOf(MAX_WORKERS, cores * 2 / 3 / THREADS))

    /**
     * Every position of the game's main line, the first before any move and the last after the
     * final one, each carrying the moves that led to it (the opening book reads them).
     */
    fun positions(game: PgnGame): List<Board> {
        val board = game.board()
        val boards = arrayListOf(board.copy())
        for (move in game.mainlineMoves()) {
            board.push(move)
            boards.add(board.copy())
        }
        return boards
    }

    /**
     * Every position measured whole (Annotator.kt's docstring), each engine taking the game's
     * next position left and finishing it -- its best line, its candidates at [EARLY_DEPTH],
     * each candidate's own position, its side searches -- in the game's order, so a caller's
     * progress reads from the first move and a review stopped part way keeps every position it
     * finished. A best line is searched once for the review: the game's next position is mostly a
     * candidate of the one before, and an engine wanting one another is searching waits for it.
     * A search that fails ends its engine's share; the others finish the rest, and the walk
     * searches whatever is left. [known]: positions measured whole before (by FEN: a review
     * stopped part way, or the live game's own), searched no more and counted done; [onMeasured]
     * is handed each position as it is measured whole, to keep it. `measure_all`.
     */
    fun measureAll(
        boards: List<Board>,
        engines: List<Engine>,
        depth: Int,
        multipv: Int,
        thresholds: Thresholds,
        onProgress: ((Int, Int) -> Unit)? = null,
        stop: AtomicBoolean? = null,
        known: Map<String, Measured> = emptyMap(),
        onMeasured: ((Measured) -> Unit)? = null,
        failed: MutableList<Engine>? = null,
    ): List<Measured> {
        val todo = ConcurrentLinkedQueue<Board>()
        val found = ArrayList<Measured>()
        for (board in boards) {
            val fen = board.fen()
            val was = known[fen]
            if (was != null) found.add(was) else todo.add(board)
        }
        if (found.isNotEmpty()) onProgress?.invoke(found.size, boards.size)
        val lock = Object()
        val best = HashMap<String, Measured>()
        val searching = HashSet<String>()  // the best lines an engine is on

        /** [board]'s best line: known, found, waited for, or searched here. */
        fun bestOf(annotator: GameAnnotator, board: Board): Measured {
            known[board.fen()]?.let { return it }
            val key = cacheKey(board)
            synchronized(lock) {
                while (key in searching) lock.wait()
                best[key]?.let { return it }
                searching.add(key)
            }
            val line = try {
                annotator.bestLine(board)
            } catch (exc: Throwable) {
                synchronized(lock) {  // a waiter searches it itself
                    searching.remove(key)
                    lock.notifyAll()
                }
                throw exc
            }
            synchronized(lock) {
                best[key] = line
                searching.remove(key)
                lock.notifyAll()
            }
            return line
        }

        fun measure(annotator: GameAnnotator, board: Board): Measured {
            val first = bestOf(annotator, board)
            val hint = annotator.hintLines(board)
            val children = annotator.candidates(board, first, hint).map { move ->
                val child = board.copy()
                child.push(move)
                move to bestOf(annotator, child)
            }
            return annotator.withSide(board, annotator.whole(board, first, children))
        }

        fun work(engine: Engine) {
            val annotator = GameAnnotator(engine, depth = depth, multipv = multipv, thresholds = thresholds)
            while (stop == null || !stop.get()) {
                val board = todo.poll() ?: return
                val measured = try {
                    engine.interruptible { measure(annotator, board) } // a stop reaches the search under way
                } catch (_: Interrupted) {
                    return // stopped: the position is not done, and nothing failed
                } catch (_: Exception) { // the walk measures what is missing itself
                    if (failed != null) synchronized(lock) { failed.add(engine) } // named: the walk searches on another
                    return
                }
                // Not kept is only a resume the slower.
                if (onMeasured != null) runCatching { onMeasured(measured) }
                synchronized(lock) {
                    found.add(measured)
                    onProgress?.invoke(found.size, boards.size) // in order: under the lock
                }
            }
        }

        val threads = engines.mapIndexed { i, engine ->
            Thread({ work(engine) }, "sidecoach-review-$i").apply { isDaemon = true }
        }
        threads.forEach { it.start() }
        if (stop != null) {
            // A stop (the room's Pause) cuts the searches under way, not only the positions after
            // them: each engine's position whole is its best line, its hint, up to six candidates
            // and its side searches. Told again while any runs: a stop landing between two
            // positions is cleared as the next search begins ([Engine.interruptible]).
            while (threads.any { it.isAlive }) {
                if (stop.get()) for (engine in engines) engine.interrupt()
                threads.firstOrNull { it.isAlive }?.join(STOP_LOOK_MS)
            }
        }
        threads.forEach { it.join() }
        return synchronized(lock) { ArrayList(found) }
    }

    /**
     * The game reviewed whole with [engines] (open, at least one; the first whose share did not
     * fail also walks the moves). [onProgress] (done, total) counts the positions measured; [known] and [onMeasured]
     * carry a review over a stop ([measureAll]). `review_game`.
     */
    fun reviewGame(
        game: PgnGame,
        engines: List<Engine>,
        depth: Int,
        multipv: Int,
        thresholds: Thresholds,
        tagger: Tagger? = null,
        book: Book? = null,
        strength: String = "standard",
        gameId: String = "review",
        onProgress: ((Int, Int) -> Unit)? = null,
        stop: AtomicBoolean? = null,
        known: Map<String, Measured> = emptyMap(),
        onMeasured: ((Measured) -> Unit)? = null,
    ): ReviewedGame {
        val started = System.nanoTime()
        val boards = positions(game)
        val failed = ArrayList<Engine>()
        val measured = measureAll(boards, engines, depth, multipv, thresholds, onProgress, stop, known, onMeasured, failed)
        // The walk measures what is missing on an engine whose share did not end in a failure.
        val walker = engines.firstOrNull { e -> failed.none { it === e } } ?: engines[0]
        val annotator = GameAnnotator(
            walker,
            depth = depth,
            multipv = multipv,
            thresholds = thresholds,
            tagger = tagger,
            book = book,
        )
        annotator.adopt(measured)
        if (stop != null && stop.get()) {
            // Stopped: what is kept is what [onMeasured] was handed; nothing more is searched.
            return ReviewedGame(
                headers = game.headers.items().toMap(LinkedHashMap()),
                start = annotator.analysisFrom(gameId, boards[0], 0, annotator.measuredLines(boards[0], emptyList())),
                moves = emptyList(),
                strength = strength,
                depth = depth,
                seconds = Math.round((System.nanoTime() - started) / 1e9 * 100) / 100.0,
            )
        }
        val start = annotator.analysisEvent(gameId, boards[0], 0)
        val clocks = clocks(game)
        val moves = ArrayList<ReviewedMove>()
        for ((index, move) in game.mainlineMoves().withIndex()) {
            if (stop != null && stop.get()) break
            val ply = index + 1
            val before = boards[ply - 1]
            val (assessment, after) = annotator.assess(gameId, before, move, ply)
            val (clockMs, spentMs) = clocks[ply - 1]
            moves.add(
                ReviewedMove(
                    ply = ply,
                    san = before.san(move),
                    side = if (before.turn == WHITE) Side.WHITE else Side.BLACK,
                    fenBefore = before.fen(),
                    assessment = assessment,
                    after = after,
                    clockMs = clockMs,
                    spentMs = spentMs,
                ),
            )
        }
        val seconds = (System.nanoTime() - started) / 1e9
        return ReviewedGame(
            headers = game.headers.items().toMap(LinkedHashMap()),
            start = start,
            moves = moves,
            strength = strength,
            depth = depth,
            seconds = Math.round(seconds * 100) / 100.0,
        )
    }

    private val CONTROL = Regex("^(\\d+)(?:\\+(\\d+))?$")

    /**
     * A saved review made again from what it kept -- each position's lines and whether it was in
     * book, each move's loss -- as a review made today with [thresholds] makes it, no engine: each
     * position's decisive and only-move calls, each move's severity and label ([verdict], read off
     * the lines [verdictLines] gives, as the annotator reads them), and with [tagger] each move's
     * and position's motifs read again (a sacrifice is what makes a move brilliant: the rule of
     * 2026-10-06 reaches reviews saved before it); without, the saved tags stand. With [book],
     * each position's opening is named again. A review saved before the rules changed keeps its
     * measurements and is labelled as one made today (Tai, 2026-10-02). `relabel`.
     */
    fun relabel(reviewed: ReviewedGame, thresholds: Thresholds, book: Book? = null, tagger: Tagger? = null): ReviewedGame {
        val start = Board(reviewed.start.fen)
        val named = book != null && start.fen() == STARTING_FEN
        val first = recalled(start, reviewed.start, thresholds, tagger, reviewed.start.opening)
        var beforeLines = first.lines
        var beforeDecisive = first.decisive
        var previous: Previous? = null
        val played = ArrayList<Move>()
        val moves = ArrayList<ReviewedMove>()
        for (item in reviewed.moves) {
            val board = Board(item.fenBefore)
            val move = Move.fromUci(item.assessment.uci)
            val loss = item.assessment.expectedLoss
            val afterLines = verdictLines(ownLine(beforeLines, move), item.after.lines)
            val tags = if (tagger != null) moveTags(tagger, board, move, afterLines) else item.assessment.tags
            val label = verdict(
                board,
                move,
                beforeLines,
                afterLines,
                loss = loss,
                beforeScore = positionScore(board, beforeLines),
                decisive = beforeDecisive,
                inBook = item.after.inBook,
                sacrifice = tags.any { it.name == "sacrifice" },
                previous = previous,
                thresholds = thresholds,
            )
            val afterBoard = board.copy(withStack = false)
            afterBoard.push(move)
            played.add(move)
            val opening = if (named) openingName(book!!, played) else item.after.opening
            val after = recalled(afterBoard, item.after, thresholds, tagger, opening)
            moves.add(
                ReviewedMove(
                    ply = item.ply,
                    san = item.san,
                    side = item.side,
                    fenBefore = item.fenBefore,
                    assessment = withLabel(item.assessment, label, severityOf(loss, thresholds), tags),
                    after = after,
                    clockMs = item.clockMs,
                    spentMs = item.spentMs,
                ),
            )
            previous = Previous(loss, move.to, board.isCapture(move))
            beforeLines = after.lines
            beforeDecisive = after.decisive
        }
        return ReviewedGame(
            headers = reviewed.headers,
            start = first,
            moves = moves,
            strength = reviewed.strength,
            depth = reviewed.depth,
            seconds = reviewed.seconds,
        )
    }

    /** [a] with another label, severity and tags, its time and frame kept. */
    private fun withLabel(a: MoveAssessment, label: Label, severity: Severity, tags: List<MotifTag>): MoveAssessment =
        MoveAssessment(
            gameId = a.gameId,
            ply = a.ply,
            uci = a.uci,
            expectedLoss = a.expectedLoss,
            severity = severity,
            best = a.best,
            tags = tags,
            depth = a.depth,
            label = label,
            accuracy = a.accuracy,
        ).also {
            it.ts = a.ts
            it.sourceFrameId = a.sourceFrameId
        }

    /**
     * A saved position's calls made again from its saved lines, as `GameAnnotator.whole` makes
     * them with [thresholds], with [tagger] its motifs ([positionTags]), and [opening] its
     * opening; its time and frame kept. `_recalled`.
     */
    private fun recalled(board: Board, a: Analysis, thresholds: Thresholds, tagger: Tagger?, opening: String?): Analysis =
        Analysis(
            gameId = a.gameId,
            fen = a.fen,
            ply = a.ply,
            lines = a.lines,
            decisive = isDecisive(board, a.lines, thresholds),
            onlyMove = isOnlyMove(board, a.lines, thresholds),
            threats = a.threats,
            traps = a.traps,
            defences = a.defences,
            threatens = a.threatens,
            tags = if (tagger != null) positionTags(tagger, board, a.lines) else a.tags,
            opening = opening,
            inBook = a.inBook,
            provisional = a.provisional,
        ).also {
            it.ts = a.ts
            it.sourceFrameId = a.sourceFrameId
        }

    /**
     * Each move's clock after it and the time it took, in ms, from the PGN's `[%clk]` comments
     * and its `TimeControl` ("600+5": 600 s a side, 5 s added a move). A correspondence game's
     * clock ("1/86400") is days, not a move's thinking: none there. `clocks`.
     */
    fun clocks(game: PgnGame): List<Pair<Int?, Int?>> {
        val control = CONTROL.matchEntire(game.headers["TimeControl"] ?: "")
        val initial = control?.let { it.groupValues[1].toInt() * 1000 }
        val increment = control?.let { it.groupValues[2].ifEmpty { "0" }.toInt() * 1000 } ?: 0
        val last = hashMapOf<Boolean, Int?>(true to initial, false to initial)
        val out = ArrayList<Pair<Int?, Int?>>()
        val board = game.board()
        for (node in game.mainline()) {
            val mover = board.turn
            board.push(node.move)
            val seconds = node.clock()
            if (seconds == null) {
                out.add(null to null)
                continue
            }
            val clock = Math.rint(seconds * 1000).toInt() // Python's round: halves to even
            val before = last[mover]
            val spent = if (before != null && control != null) before - clock + increment else null
            out.add(clock to spent?.let { maxOf(0, it) })
            last[mover] = clock
        }
        return out
    }
}
