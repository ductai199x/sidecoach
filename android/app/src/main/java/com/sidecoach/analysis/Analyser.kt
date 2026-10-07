package com.sidecoach.analysis

import android.util.Log
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Event
import com.sidecoach.events.Line
import com.sidecoach.events.VerdictFailed
import java.util.Timer
import java.util.TreeMap
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.concurrent.schedule

/**
 * The engine service for a game followed live: every position measured once, on a pool of
 * single-thread Stockfish workers, by one-line searches only (Annotator.kt's docstring). A port
 * of `sidecoach/analysis/analyser.py`.
 *
 * What the measurements decided (tools/engine/, the engine audit of 2026-10-04/05):
 * - one thread a search: more threads never answered a fixed depth sooner, at up to 17 times
 *   the processor time;
 * - one line a search: several lines to depth 18 explode on decided and forcing positions (up
 *   to 53 s), one line does not (p50 0.19 s, p99 1.1 s);
 * - no look-ahead beyond a position's candidates: with searches this cheap, searching further
 *   ahead bought hundredths of a second and threw away a third of its work. The candidates --
 *   each of the position's best moves searched as its own position -- are the positions the
 *   game goes to next.
 *
 * A game position is measured in stages, each told to the readers as it comes, the next
 * replacing it: its first answer (its candidates at [EARLY_DEPTH], or its best line while they
 * are not in yet), at once; the position whole -- its best line first, its other candidates from
 * their own positions' searches -- provisional; then with its side searches, final.
 *
 * The workers take, first to last: what the next verdict needs (the position a move was played
 * from, whole; the position it led to, its best line); what the coach asks for ([request]); the
 * position the game is on; the game's other positions (their review is then a search of none of
 * them). A move is judged once the position it was played from is whole -- its candidates were
 * searched while its mover thought -- and the position it led to has its best line, or for a
 * search running past [DECIDED_AFTER_MS] of its engine's own running time (the voice's pauses not
 * counted), a decided score at the deepest depth it finished (Tai, 2026-10-05: the severities
 * this flips are as few as two depth-18 searches' own disagreement). Verdicts go out in move
 * order.
 *
 * A position with one legal move is never searched: its best line is its move's, read from the
 * position that move leads to ([GameAnnotator.fromOnlyMove]).
 *
 * A search fails alone: what it held back is judged or told without it, never waited on; a move
 * it leaves no way to judge is told as not judged ([com.sidecoach.events.VerdictFailed]), in its
 * place among the verdicts, never skipped in silence; and nothing a worker or the decided rule's timer does after a search can end its thread -- an
 * exception escaping a thread ends the app.
 */

/**
 * How many workers search side by side, one thread each: the desktop's three, which answer as
 * fast as four or six (tools/engine/workers_compare.py); the phone's two ([PHONE_WORKERS]), the
 * two cores its one engine of two threads took before.
 */
const val WORKERS = 3
const val PHONE_WORKERS = 2

/**
 * A search still running this long (its engine's own running time) whose deepest finished depth
 * -- [EARLY_DEPTH] at least -- shows the position decided (an expected score at most
 * [DECIDED_LOW] or at least [DECIDED_HIGH]) has that depth's score stand in for the full depth's
 * in a verdict (the file's docstring).
 */
const val DECIDED_AFTER_MS = 1000L

/** The soonest the decided rule looks again when the engine's own clock is short of its time. */
const val DECIDED_RELOOK_MS = 50L
const val DECIDED_LOW = 0.15
const val DECIDED_HIGH = 0.85

/**
 * The positions kept, searched, for the game and any game after it (a puzzle tried again is
 * answered at once); the oldest go past this. The desktop keeps 20000; a phone's heap is shared
 * with the vision and the voice, and each position holds a board.
 */
const val KEEP_NODES = 5000

/**
 * How long the coach waits on the pool for a position ([PoolSearcher]): a whole position takes
 * about a second, two on a phone; past this the coach answers from what is known.
 */
const val REQUEST_TIMEOUT_MS = 8_000L

/** What a ply has been told, in order; a stage told is not told again for that position. */
private const val FIRST = 1
private const val FIRST_BOTH = 2
private const val WHOLE = 3
private const val FINAL = 4

/**
 * An analyser ready to take moves, its workers' engines started side by side (each loads its
 * network: one after another, the first move waited for all of them), one thread each ([make] is
 * handed the threads). An engine that will not start is said here, the others closed, rather
 * than failing every search after. `open_analyser`.
 */
fun openAnalyser(
    depth: Int,
    multipv: Int,
    workers: Int = WORKERS,
    thresholds: Thresholds = Thresholds(),
    tagger: Tagger? = null,
    book: Book? = null,
    make: (threads: Int) -> Engine,
): Analyser {
    val engines = List(maxOf(1, workers)) { make(1) }
    val errors = java.util.Collections.synchronizedList(ArrayList<Throwable>())
    val openers = engines.map { engine ->
        Thread({
            try {
                engine.open()
            } catch (exc: Throwable) {
                errors.add(exc) // said once all have tried, below
            }
        }, "sidecoach-engine-open").apply { isDaemon = true }
    }
    openers.forEach { it.start() }
    try {
        openers.forEach { it.join() }
    } catch (exc: InterruptedException) {
        // The openers finished first (an `open` re-opens an engine closed before it), then every
        // engine closed: none is left running with no one to close it.
        for (opener in openers) {
            while (opener.isAlive) runCatching { opener.join() }
        }
        for (engine in engines) runCatching { engine.close() }
        Thread.currentThread().interrupt()
        throw exc
    }
    if (errors.isNotEmpty()) {
        for (engine in engines) runCatching { engine.close() }
        throw errors[0]
    }
    val annotator = GameAnnotator(engines[0], depth = depth, multipv = multipv, thresholds = thresholds, tagger = tagger, book = book)
    return Analyser(annotator, engines = engines).also { it.start() }
}

/**
 * A game's positions measured on a pool of workers: submit positions and moves, poll events,
 * never block ([request] alone waits, for the coach's questions). [ranMs] reads how long a
 * worker's search has run, its engine's own clock; [schedule] runs the decided rule's check
 * later (a test hands in its own).
 */
class Analyser(
    val annotator: GameAnnotator,
    /** One per worker (open or not); the annotator's own engine alone when not given. */
    engines: List<Engine>? = null,
    private val ranMs: ((Int) -> Long)? = null,
    private val schedule: ((Long, () -> Unit) -> () -> Unit)? = null,
) : AutoCloseable {

    val engines: List<Engine> = if (engines.isNullOrEmpty()) listOf(annotator.engine) else engines.toList()

    /**
     * One position: its board (with the moves that led there, which the engine reads for
     * repetitions), and what has been found of it.
     */
    internal class Node(val key: String, val board: Board) {
        /** Its one legal move, when it has only one: its best line is that move's (never searched). */
        var only: Move? = null
        var best: Measured? = null
        var hint: List<Line>? = null
        /** Its candidates, once its best line and its hint are in, and their positions' keys. */
        var candidates: List<Pair<Move, String>>? = null
        val parents = HashSet<String>()  // the positions it is a candidate of
        var whole: Measured? = null  // its candidates' searches, no side searches
        var final: Measured? = null  // with its side searches
        var standIn: Measured? = null  // a decided score while its best line is searched
        var deepest: Line? = null  // its best line's search: the deepest depth finished so far
        val failed = HashSet<String>()

        /** What a verdict may read of it: its best line, or the decided stand-in. */
        val value: Measured? get() = best ?: standIn

        /** Its best line and its hint are in (a failed one counts as in, with nothing). */
        val firstDone: Boolean
            get() = (best != null || BEST in failed) && (hint != null || HINT in failed)

        /** Its best line and its hint both failed: nothing will be known of it. */
        val lost: Boolean get() = BEST in failed && HINT in failed
    }

    internal class Played(val ply: Int, val before: String, val after: String, val move: Move)

    internal data class Job(val kind: String, val key: String, val era: Int)

    /** What a job's search found. */
    private sealed interface Found {
        class Best(val measured: Measured) : Found
        class Hint(val lines: List<Line>) : Found
        class Side(val measured: Measured) : Found
    }

    /** A position the coach asked for ([request]), the caller waiting on it, and the answer it was given. */
    internal class Request(val key: String, val whole: Boolean) {
        val done = CountDownLatch(1)
        var answer: Measured? = null
    }

    /** What [told] holds of a ply: its position, and the stage told. */
    private data class Told(val key: String, val stage: Int)

    internal val lock = Object()
    internal var nodes = LinkedHashMap<String, Node>()
    private var gameId: String? = null
    internal val path = TreeMap<Int, String>()  // ply -> the game's position at it
    private val boards = HashMap<Int, Board>()  // ply -> the game's board, its moves with it
    private val plyOf = HashMap<String, Int>()
    private val moves = HashMap<Int, Played>()
    private var current = 0
    internal var nextVerdict: Int? = null
    private var told = HashMap<Int, Told>()
    internal var requests = ArrayList<Request>()
    private val running = HashMap<Int, Job>()  // by worker: the job a stop can reach
    private val stopping = HashSet<Int>()  // workers told to stop, not yet back
    private val deciding = HashMap<Int, () -> Unit>()  // by worker: the decided rule's next look, cancelled when its search ends
    internal val claimed = HashMap<Int, Job>()  // by worker: the job it took, running or about to
    internal var era = 0
        private set

    /** What the workers have finished; a test reads it as the desktop's `_results`. */
    internal val results = ConcurrentLinkedQueue<Event>()
    internal val threads = ArrayList<Thread>()
    private var closed = false

    /** The positions kept before the oldest go ([trim]); a test trims sooner. */
    internal var keepNodes = KEEP_NODES

    /** The decided rule's clock, when no [schedule] is handed in: one thread for the analyser's life. */
    private val timer: Timer? = if (schedule == null) Timer("sidecoach-analyser-decided", true) else null

    // What it did, for the logs and the measures.
    @Volatile var searches = 0  // one-line searches to the full depth
        private set
    @Volatile var hints = 0
        private set
    @Volatile var sideSearches = 0
        private set
    @Volatile var stopped = 0  // searches let go because nothing would read them
        private set
    @Volatile var withdrawn = 0
        private set
    @Volatile var failures = 0
        private set
    /**
     * The engine's own searches that failed (Stockfish stopped answering, a fault in it): what
     * the player is told of, once (the panel's `engineTrouble`).
     */
    @Volatile var searchFailures = 0
        private set
    @Volatile var lastError: String? = null
        private set

    /**
     * What a test watches the workers by, as the desktop's tests wrap `_cond.notify_all` and
     * `_results.put`: told after each word put in [results] and after each change of state the
     * workers are woken for. Null in the app, where nothing is told and nothing else changes.
     */
    internal interface Watch {
        fun said(event: Event) {}
        fun changed() {}

        /** At the end of [withdraw], what was under way for the moves taken back let go. */
        fun withdrew(gameId: String, ply: Int) {}
    }

    @Volatile internal var watch: Watch? = null

    // ---- the caller's side -------------------------------------------------------------

    fun start() {
        if (threads.isNotEmpty()) return
        for ((index, engine) in engines.withIndex()) {
            threads.add(
                Thread({ work(index, engine) }, "sidecoach-analyser-$index").also {
                    it.isDaemon = true
                    it.start()
                },
            )
        }
    }

    override fun close() {
        synchronized(lock) {
            closed = true
            for (index in running.keys.toList()) engines[index].interrupt()
            for (request in requests) request.done.countDown()  // the coach waits no more: it has what is known
            requests.clear()
            wake()
        }
        try {
            for (thread in threads) thread.join(5000)
        } finally {
            // However the waiting ended (an interrupt on the closing thread): the engines and the
            // decided rule's thread are let go all the same.
            threads.clear()
            timer?.cancel()
            for (engine in uniqueEngines()) runCatching { engine.close() }
        }
    }

    /**
     * What is still owed to the game: its positions not told all they will be, and its moves
     * not judged.
     */
    val pending: Int
        get() = synchronized(lock) {
            var owed = path.entries.count { (ply, key) -> !settled(ply, key) }
            val next = nextVerdict
            if (next != null) owed += moves.keys.count { it >= next }
            owed
        }

    fun poll(): List<Event> {
        val out = ArrayList<Event>()
        while (true) out.add(results.poll() ?: break)
        return out
    }

    /** A position the game stands on with no move to it: its start, or a resync. */
    fun submitPosition(gameId: String, board: Board, ply: Int) {
        synchronized(lock) {
            this.gameId = gameId
            dropAfter(ply)
            val key = node(board).key
            val move = moves[ply]
            if (move != null && move.after != key) {
                // The move to this ply led elsewhere: it is not this game's any more.
                moves.remove(ply)
                withdrawn += 1
                if (moves.isEmpty()) nextVerdict = null
            }
            setPath(ply, key, board)
            current = ply
            // Told again: a position submitted is one the player faces anew (a takeback to the
            // start, a re-read), whatever was told of it before.
            told.remove(ply)
            tell(ply)
            settle()
        }
    }

    fun submitMove(gameId: String, before: Board, move: Move, ply: Int) {
        synchronized(lock) {
            this.gameId = gameId
            if (!before.isLegal(move)) {
                failures += 1
                lastError = "${move.uci()} is not legal in ${before.fen()}"
                return
            }
            // What stood at this ply and after it is the game's no longer: a move played here
            // again replaces them (the glue withdraws them first; this holds without it).
            dropAfter(ply - 1)
            val source = node(before.copy())
            if (path[ply - 1] != source.key) {
                // The game is on another position than the one the move before led to: that
                // move is not this game's, and is not judged.
                if (moves.remove(ply - 1) != null) withdrawn += 1
                setPath(ply - 1, source.key, before)
                tell(ply - 1)
            }
            val after = before.copy()
            after.push(move)
            val target = node(after)
            setPath(ply, target.key, after)
            moves[ply] = Played(ply, source.key, target.key, move)
            val next = nextVerdict
            if (next == null || ply < next) nextVerdict = ply
            current = ply
            // The verdict first: the word on the move is the news, the position after it next.
            judge()
            tell(ply)
            stopUnwanted()
            settle()
        }
    }

    /**
     * The game's moves after [ply] taken back: nothing more is said of them, and a search made
     * only for them stops.
     */
    fun withdraw(gameId: String, ply: Int) {
        synchronized(lock) {
            if (gameId != this.gameId) return
            withdrawn += moves.keys.count { it > ply }
            dropAfter(ply)
            current = ply
            stopUnwanted()
            settle()
        }
        watch?.withdrew(gameId, ply)
    }

    /**
     * A new game, or the board re-read: the game's path let go, and what is under way for it
     * stops; what was measured is kept, for a position seen again.
     */
    fun reset() {
        synchronized(lock) {
            gameId = null
            path.clear()
            boards.clear()
            plyOf.clear()
            moves.clear()
            told.clear()
            nextVerdict = null
            current = 0
            for (node in nodes.values) node.failed.clear()  // a search that failed is tried again for a new game
            stopUnwanted()
            annotator.reset()
            settle()
        }
    }

    /**
     * The coach's question on a position (a move it or the player asks about, a position its
     * words are about): the position whole -- its candidates at the full depth -- or its best
     * line alone, waited for. Taken after the verdicts and before the game's own work. What is
     * known by [timeoutMs], or when the analyser closes, is the answer then (null for nothing);
     * asked again, what is in comes at once. The asking thread interrupted, its question is
     * dropped all the same.
     */
    fun request(board: Board, whole: Boolean = true, timeoutMs: Long? = null): Measured? {
        val key = cacheKey(board)
        val request: Request
        synchronized(lock) {
            if (closed) return answer(key, whole)
            val node = node(board)
            if (answered(node, whole)) return answer(key, whole)
            request = Request(key, whole)
            requests.add(request)
            settle()
        }
        try {
            if (timeoutMs == null) request.done.await() else waitRunning(request, timeoutMs)
        } finally {
            synchronized(lock) {
                if (requests.remove(request)) stopUnwanted()  // given up: nothing more is searched for it
            }
        }
        synchronized(lock) {
            return request.answer ?: answer(key, whole)
        }
    }

    /**
     * [request] waited for up to [timeoutMs] of the pool's running time: the phone's own -- the
     * voice holds the engines paused while it makes a line ([Engine.pause]), and that time is no
     * search's, so the question's wait is longer by it.
     */
    private fun waitRunning(request: Request, timeoutMs: Long) {
        val engine = engines.first() // the pool is paused and resumed together
        val began = System.nanoTime()
        val pausedAtStart = engine.pausedMs()
        while (true) {
            val waited = (System.nanoTime() - began) / 1_000_000 - (engine.pausedMs() - pausedAtStart)
            val left = timeoutMs - waited
            if (left <= 0 || request.done.await(left, TimeUnit.MILLISECONDS)) return
        }
    }

    /**
     * The game's positions measured whole with their side searches, in the order of the game:
     * what its review need not search again ([Assessing.keepForReview]). Nothing for a game this
     * is not on.
     */
    fun measuredPath(gameId: String): List<Measured> = synchronized(lock) {
        if (gameId != this.gameId) return emptyList()
        path.values.mapNotNull { nodes[it]?.final }
    }

    /**
     * The settings a search depends on changed: nothing measured before counts, and a search
     * under way is not kept ([GameAnnotator.forgetMeasured]). The game's positions are measured
     * again; what was told of the ones it has left stands -- they are past, and telling them
     * again would only repeat them to the readers.
     */
    fun forgetMeasured() {
        synchronized(lock) {
            era += 1
            // The game's positions and the coach's are kept, unmeasured; the rest go.
            val kept = HashSet(plyOf.keys) + requests.map { it.key }
            val fresh = LinkedHashMap<String, Node>()
            for ((k, n) in nodes) if (k in kept) fresh[k] = fresh(k, n.board)
            nodes = fresh
            told = HashMap(told.filterKeys { it != current }.mapValues { (_, v) -> Told(v.key, FINAL) })
            for (index in running.keys.toList()) engines[index].interrupt()
            annotator.forgetMeasured()
            settle()
        }
    }

    /** Hash, the lines, their length or the scoring saved mid-session: on every worker's engine. */
    fun reconfigure(hashMb: Int? = null, multipv: Int? = null, pvPlies: Int? = null, scoreModel: String? = null) {
        for (engine in uniqueEngines()) {
            if (hashMb != null) engine.reconfigure(hashMb = hashMb)
            if (multipv != null) engine.multipv = multipv
            if (pvPlies != null) engine.pvPlies = pvPlies
            if (scoreModel != null) engine.scoreModel = scoreModel
        }
    }

    // ---- the positions -----------------------------------------------------------------

    /**
     * A node for [board], nothing found of it yet: a finished position or one with more than one
     * legal move as it is; one with a single legal move has that move, and nothing to hint (the
     * move is the answer).
     */
    private fun fresh(key: String, board: Board): Node {
        val node = Node(key, board.copy())
        if (terminalExpectedScore(board) == null) {
            val legal = board.legalMoves()
            if (legal.size == 1) {
                node.only = legal[0]
                node.hint = emptyList()
            }
        }
        return node
    }

    internal fun node(board: Board): Node {
        val key = cacheKey(board)
        nodes[key]?.let { return it }
        return fresh(key, board).also { nodes[key] = it }
    }

    /** The candidate [move]'s position, made again if it was let go ([trim]). */
    private fun child(node: Node, move: Move, key: String): Node {
        val child = nodes[key] ?: run {
            val board = node.board.copy()
            board.push(move)
            node(board)
        }
        child.parents.add(node.key)
        return child
    }

    /**
     * Past [keepNodes], the oldest tenth of the positions let go -- but for the game's, its
     * moves', the coach's, those a worker is on, their candidates and the forced moves' positions
     * they take their best lines from. Run between the steps that
     * choose candidates, never inside one; a candidate let go is made again when it is wanted.
     */
    private fun trim() {
        if (nodes.size < keepNodes) return
        val roots = path.values + moves.values.flatMap { listOf(it.before, it.after) } +
            requests.map { it.key } + claimed.values.map { it.key }
        val keep = HashSet(roots)
        for (key in roots) {
            val node = nodes[key] ?: continue
            for (wanted in listOf(key) + (node.candidates ?: emptyList()).map { it.second }) {
                keep.addAll(forced(wanted, make = false))
            }
        }
        for (key in nodes.keys.take(keepNodes / 10)) {
            if (key in keep) continue
            val node = nodes.remove(key) ?: continue
            for ((_, child) in node.candidates ?: emptyList()) nodes[child]?.parents?.remove(key)
        }
    }

    /**
     * The game's position at [ply] is [key], [board] as the game reached it: the one it held
     * before, if another, is no longer the game's -- nothing more is told of it here.
     */
    private fun setPath(ply: Int, key: String, board: Board) {
        val old = path[ply]
        if (old != null && old != key) {
            if (plyOf[old] == ply) plyOf.remove(old)
            told.remove(ply)
        }
        path[ply] = key
        boards[ply] = board.copy()
        plyOf[key] = ply
    }

    /** The game's plies after [ply] let go: their positions, their moves, what was told. */
    private fun dropAfter(ply: Int) {
        for (p in path.keys.filter { it > ply }) {
            val key = path.remove(p)!!
            boards.remove(p)
            if (plyOf[key] == p) plyOf.remove(key)
            told.remove(p)
        }
        for (p in moves.keys.filter { it > ply }) {
            moves.remove(p)
            annotator.forgetAssessed(p)
        }
        val next = nextVerdict
        if (next != null && next > ply + 1) nextVerdict = ply + 1
        if (moves.isEmpty()) nextVerdict = null
    }

    /**
     * The position's candidates and their positions' keys: chosen once -- its one legal move at
     * once, else when its best line and its hint are in -- each candidate's position made a node
     * that knows it.
     */
    private fun candidatesOf(node: Node): List<Pair<Move, String>> {
        if (node.candidates == null && (node.only != null || node.firstDone)) {
            val only = node.only
            val chosen = if (only != null) {
                listOf(only)
            } else {
                annotator.candidates(node.board, node.best ?: empty(node), node.hint ?: emptyList())
            }
            node.candidates = chosen.map { move ->
                val child = node.board.copy()
                child.push(move)
                val childNode = node(child)
                childNode.parents.add(node.key)
                move to childNode.key
            }
        }
        return node.candidates ?: emptyList()
    }

    private fun empty(node: Node): Measured =
        Measured(node.board.fen(), emptyList(), 0.5, decisive = false, onlyMove = false, key = node.key)

    /**
     * [key], and the positions a chain of single legal moves from it leads to: a position with
     * one legal move takes its best line from the next ([jobsFor]), so the searches it waits on
     * are theirs, however many forced moves down. [make]: the chain's positions made as
     * [jobsFor] makes them; without it ([trim]), only those already made.
     */
    private fun forced(key: String, make: Boolean = true): List<String> {
        val out = arrayListOf(key)
        var node = nodes[key]
        while (node?.only != null) {
            val candidates = if (make) candidatesOf(node) else node.candidates ?: emptyList()
            val next = candidates.firstOrNull()?.second ?: break
            if (next in out) break
            out.add(next)
            node = nodes[next]
        }
        return out
    }

    /**
     * Every position a search is wanted for: the game's, what the coach asked for, their
     * candidates, and the forced moves' positions each of those takes its best line from.
     */
    private fun wanted(): Set<String> {
        val out = HashSet<String>()
        for (key in path.values + requests.map { it.key }) {
            val node = nodes[key] ?: continue
            for (wanted in listOf(key) + candidatesOf(node).map { it.second }) out.addAll(forced(wanted))
        }
        return out
    }

    // ---- what the workers take -------------------------------------------------------------

    /**
     * The searches [node] still needs, in order: its best line (a single legal move's, from its
     * position's); for [whole], its hint and its candidates' best lines; for [side], its side
     * searches once it is whole.
     */
    private fun jobsFor(node: Node, whole: Boolean, side: Boolean): List<Job> {
        val jobs = ArrayList<Job>()
        if (node.only != null) {
            // Its best line is its move's position's: that search is its own.
            for ((move, key) in candidatesOf(node)) {
                val child = child(node, move, key)
                if (node.best == null) jobs += jobsFor(child, whole = false, side = false)
            }
        } else if (node.best == null && BEST !in node.failed) {
            jobs.add(Job(BEST, node.key, era))
        }
        if (!whole) return jobs
        if (node.hint == null && HINT !in node.failed) jobs.add(Job(HINT, node.key, era))
        for ((move, key) in candidatesOf(node)) {
            val child = child(node, move, key)
            if (child.best == null && BEST !in child.failed) jobs += jobsFor(child, whole = false, side = false)
        }
        if (side && node.whole != null && node.final == null && SIDE !in node.failed) {
            jobs.add(Job(SIDE, node.key, era))
        }
        return jobs
    }

    /**
     * Every search wanted, first to last (the file's docstring): the verdicts' needs, the coach's
     * requests, the position the game is on, the game's other positions.
     */
    internal fun agenda(): List<Job> {
        val out = ArrayList<Job>()
        val next = nextVerdict
        if (next != null) {
            for (ply in moves.keys.filter { it >= next }.sorted()) {
                val move = moves.getValue(ply)
                out += jobsFor(nodes.getValue(move.before), whole = true, side = false)
                out += jobsFor(nodes.getValue(move.after), whole = false, side = false)
            }
        }
        for (request in requests) out += jobsFor(nodes.getValue(request.key), whole = request.whole, side = false)
        path[current]?.let { out += jobsFor(nodes.getValue(it), whole = true, side = true) }
        for (ply in path.descendingKeySet()) out += jobsFor(nodes.getValue(path.getValue(ply)), whole = true, side = true)
        return out
    }

    internal fun pick(): Job? {
        val busy = claimed.values.map { it.kind to it.key }.toSet()
        return agenda().firstOrNull { (it.kind to it.key) !in busy }
    }

    private fun work(index: Int, engine: Engine) {
        while (true) {
            val job: Job
            val board: Board
            val whole: Measured?
            synchronized(lock) {
                var picked: Job? = null
                while (!closed) {
                    picked = try {
                        pick()
                    } catch (exc: Throwable) {
                        failed("choosing a search", exc) // a bug here must not end the worker
                        lock.wait(1000) // not a hot loop on a fault that stays
                        continue
                    }
                    if (picked != null) break
                    lock.wait()
                }
                if (closed || picked == null) return
                job = picked
                // Taken the moment it is picked: no other worker picks it meanwhile.
                claimed[index] = job
                val node = nodes.getValue(job.key)
                board = node.board.copy()
                whole = node.whole
            }
            val found = try {
                run(index, engine, job, board, whole)
            } catch (exc: Interrupted) {
                null
            } catch (exc: Throwable) {
                // A failed search must never end a session.
                synchronized(lock) {
                    failed("the ${job.kind} search of ${board.fen()}", exc)
                    searchFailures += 1
                    val failedNode = nodes[job.key]
                    if (failedNode != null && job.era == era) {
                        failedNode.failed.add(job.kind)
                        guarded("advance") { advance(failedNode) }
                    }
                }
                null
            }
            synchronized(lock) {
                running.remove(index)
                stopping.remove(index)
                claimed.remove(index)
                if (found != null && job.era == era && job.key in nodes) guarded("found") { found(job, found) }
                guarded("trim") { trim() }
                wake()
            }
        }
    }

    /** [step] run under the lock; a fault in it counted and logged, never its thread's end. */
    private inline fun guarded(what: String, step: () -> Unit) {
        try {
            step()
        } catch (exc: Throwable) {
            failed(what, exc)
        }
    }

    private fun failed(what: String, exc: Throwable) {
        // Never a throw from here (the host's android.jar stub throws on every call).
        runCatching { Log.w(TAG, "analyser: $what failed: $exc") }
        failures += 1
        lastError = "${exc::class.java.simpleName}: ${exc.message}"
    }

    /**
     * One job's search, stoppable from the moment it is running ([stopUnwanted]); null when it
     * was no longer wanted by then.
     */
    private fun run(index: Int, engine: Engine, job: Job, board: Board, whole: Measured?): Found? =
        engine.interruptible {
            synchronized(lock) {
                if (closed || job.era != era || !stillWanted(job)) return@interruptible null
                running[index] = job  // inside the stoppable block: a stop reaches it
            }
            when (job.kind) {
                HINT -> Found.Hint(annotator.hintLines(board, engine = engine))
                SIDE -> Found.Side(annotator.withSide(board, whole!!, engine = engine))
                else -> {
                    synchronized(lock) { deciding[index] = later(DECIDED_AFTER_MS) { decide(job, index) } }
                    try {
                        Found.Best(annotator.bestLine(board, engine = engine, onDepth = { lines -> deepened(job, index, lines) }))
                    } finally {
                        synchronized(lock) { deciding.remove(index) }?.invoke()
                    }
                }
            }
        }

    /** [task] run after [ms], on the decided rule's clock; the call that cancels it. */
    private fun later(ms: Long, task: () -> Unit): () -> Unit {
        schedule?.let { return it(ms, task) }
        val scheduled = try {
            timer!!.schedule(ms) { task() }
        } catch (exc: IllegalStateException) {
            return {} // the timer let go at the close: nothing more to decide
        }
        return { scheduled.cancel() }
    }

    /** How long worker [index]'s search has run, its engine's own clock (the voice's pauses not counted). */
    private fun searchedMs(index: Int): Long = ranMs?.invoke(index) ?: engines[index].searchRanMs()

    private fun stillWanted(job: Job): Boolean {
        val node = nodes[job.key] ?: return false
        val done = when (job.kind) {
            BEST -> node.best
            HINT -> node.hint
            else -> node.final
        }
        return done == null && job.key in wanted()
    }

    private fun stopUnwanted() {
        val wanted = wanted()
        for ((index, job) in running.entries.toList()) {
            if (job.key !in wanted && stopping.add(index)) {
                stopped += 1
                engines[index].interrupt()
            }
        }
    }

    /** After a change from outside: the positions trimmed, the workers woken. */
    private fun settle() {
        guarded("trim") { trim() }
        wake()
    }

    // ---- what comes back ---------------------------------------------------------------

    /**
     * A depth of a best line's search finished, the search going on: kept, and past
     * [DECIDED_AFTER_MS] read for a decided score.
     */
    private fun deepened(job: Job, index: Int, lines: List<Line>) {
        synchronized(lock) {
            val node = nodes[job.key] ?: return
            if (job.era != era || lines.isEmpty()) return
            node.deepest = lines[0]
            if (running[index] == job && searchedMs(index) >= DECIDED_AFTER_MS) guarded("decide") { standIn(node) }
        }
    }

    /**
     * [DECIDED_AFTER_MS] into a best line's search (the timer): its deepest finished depth read
     * for a decided score, as each depth after it is ([deepened]) -- once the search has run
     * that long on its engine's own clock. Never throws: an exception here would end the timer.
     */
    internal fun decide(job: Job, index: Int) {
        try {
            synchronized(lock) {
                val node = nodes[job.key] ?: return
                if (job.era != era || running[index] != job) return
                val ran = searchedMs(index)
                if (ran >= DECIDED_AFTER_MS) {
                    standIn(node)
                } else if (index in deciding) {
                    // The engine's own clock starts at its `go`, after the worker's: it has run
                    // less than the rule's time yet (or the voice paused it). Looked at again
                    // once it has, as the desktop's clock would.
                    // Never sooner than DECIDED_RELOOK_MS: paused by the voice, the engine's
                    // clock stands still, and a near-second left would look again every
                    // millisecond for the whole pause.
                    deciding[index] = later(maxOf(DECIDED_RELOOK_MS, DECIDED_AFTER_MS - ran)) { decide(job, index) }
                }
            }
        } catch (exc: Throwable) {
            synchronized(lock) { failed("deciding", exc) }
        }
    }

    /**
     * A decided score at the deepest depth finished stands in for the full depth's in a verdict
     * (the file's docstring); nothing for a position still open, or too shallow.
     */
    private fun standIn(node: Node) {
        val line = node.deepest ?: return
        if (node.best != null || line.depth < EARLY_DEPTH) return
        if (line.expectedScore > DECIDED_LOW && line.expectedScore < DECIDED_HIGH) return
        node.standIn = Measured(
            fen = node.board.fen(),
            lines = listOf(line),
            score = positionScore(node.board, listOf(line)),
            decisive = false,
            onlyMove = false,
            key = node.key,
        )
        carryStandIn(node)
        judge()
        wake()
    }

    private fun found(job: Job, found: Found) {
        val node = nodes.getValue(job.key)
        when (found) {
            is Found.Best -> {
                node.best = found.measured
                searches += 1
            }
            is Found.Hint -> {
                node.hint = found.lines
                hints += 1
            }
            is Found.Side -> {
                node.final = found.measured
                sideSearches += 1
            }
        }
        advance(node)
    }

    /**
     * A search of [node] done (or failed): a single legal move's position it is the candidate of
     * given its best line, it and every position it is a candidate of made whole if they can be,
     * and what waited on them told, judged and answered.
     */
    private fun advance(node: Node) {
        // Every position given a best line here -- this one, and those up a forced sequence --
        // and the positions each is a candidate of.
        for (changed in listOf(node) + carryOnly(node)) {
            for (key in listOf(changed.key) + changed.parents) {
                val parent = nodes[key]
                if (parent != null && parent.whole == null) complete(parent)
            }
        }
        if (SIDE in node.failed && node.whole != null && node.final == null) node.final = node.whole
        // The positions moves are judged from told before the verdicts (a verdict's reader finds
        // the position it was judged from told), the position a move led to after its verdict
        // (the word on the move is the news).
        val waiting = nextVerdict
        for (ply in path.keys.toList()) if (waiting == null || ply < waiting) tell(ply)
        judge()
        for (ply in path.keys.toList()) tell(ply)
        answerRequests()
    }

    /**
     * [node]'s best line in (or failed): each position whose one legal move leads here takes its
     * best line from it, and so on back up a forced sequence. The positions it reached.
     */
    private fun carryOnly(node: Node): List<Node> {
        val reached = ArrayList<Node>()
        for (key in node.parents.toList()) {
            val parent = nodes[key] ?: continue
            val only = parent.only ?: continue
            if (parent.best != null) continue
            val best = node.best
            when {
                best != null -> parent.best = annotator.fromOnlyMove(parent.board, only, best)
                BEST in node.failed -> parent.failed.add(BEST)
                else -> continue
            }
            reached.add(parent)
            reached.addAll(carryOnly(parent))
        }
        return reached
    }

    /**
     * [node]'s decided stand-in, carried as its best line is ([carryOnly]): a position whose one
     * legal move leads here, and so on back up a forced sequence, stands in with it -- a verdict
     * on the move that led into the forced sequence reads that position.
     */
    private fun carryStandIn(node: Node) {
        val standIn = node.standIn ?: return
        for (key in node.parents.toList()) {
            val parent = nodes[key] ?: continue
            val only = parent.only ?: continue
            if (parent.best != null) continue
            parent.standIn = annotator.fromOnlyMove(parent.board, only, standIn)
            carryStandIn(parent)
        }
    }

    /**
     * [node] made whole once its candidates are all in (a failed one left out) -- but for its
     * best line's own move: that line is its own search's ([GameAnnotator.whole]), and its move's
     * position, searched all the same as the game's likeliest next, is not waited for.
     */
    private fun complete(node: Node) {
        val candidates = candidatesOf(node)
        if (!node.firstDone || node.lost) return
        val own = node.best?.lines?.firstOrNull()?.moves?.take(1) ?: emptyList()
        val children = candidates.filter { (move, _) -> listOf(move.uci()) != own }
            .map { (move, key) -> move to child(node, move, key) }
        if (children.any { (_, child) -> child.best == null && BEST !in child.failed }) return
        val made = annotator.whole(
            node.board,
            node.best ?: empty(node),
            children.mapNotNull { (move, child) -> child.best?.let { move to it } },
        )
        node.whole = made
        annotator.computed.add(node.board, made.lines)
        if (!annotator.withThreats || terminalExpectedScore(node.board) != null) node.final = made
    }

    /**
     * Each move judged as soon as it can be, in the order of the moves: the move before is what
     * a verdict reads ("the opponent just erred", "a recapture").
     */
    private fun judge() {
        while (true) {
            var at = nextVerdict ?: return
            if (at !in moves) {
                // A gap in the numbering: the next move's.
                at = moves.keys.filter { it > at }.minOrNull() ?: return
                nextVerdict = at
            }
            val move = moves.getValue(at)
            val before = nodes.getValue(move.before)
            val after = nodes.getValue(move.after)
            val whole = before.whole
            if (whole == null) {
                if (before.lost) {
                    // Nothing to judge it from: told as not judged, never skipped in silence.
                    unjudged(move, "the position it was played from could not be searched")
                    continue
                }
                return
            }
            val own = ownLine(whole.lines, move.move) != null
            if (!own && after.value == null) {
                if (BEST in after.failed) {
                    unjudged(move, "the position it led to could not be searched")
                    continue
                }
                return
            }
            val board = boards[move.ply - 1] ?: before.board
            val verdict = try {
                annotator.assessFrom(gameId ?: "", board, move.move, move.ply, whole, if (own) null else after.value)
            } catch (exc: Exception) {
                // A move that cannot be judged is told so.
                failed("judging ${move.move.uci()} at ply ${move.ply}", exc)
                unjudged(move, "judging it failed: ${exc::class.java.simpleName}: ${exc.message}")
                continue
            }
            put(verdict)
            nextVerdict = at + 1
        }
    }

    /**
     * [move] told as not judged ([VerdictFailed]), in its place among the verdicts, and the
     * verdicts carried on from the next. `_unjudged`.
     */
    private fun unjudged(move: Played, why: String) {
        val reason = why + (lastError?.let { " ($it)" } ?: "")
        put(VerdictFailed(gameId = gameId ?: "", ply = move.ply, uci = move.move.uci(), reason = reason))
        nextVerdict = move.ply + 1
    }

    private fun answered(node: Node, whole: Boolean): Boolean =
        if (whole) node.whole != null || node.lost else node.best != null || BEST in node.failed

    private fun answer(key: String, whole: Boolean): Measured? {
        val node = nodes[key] ?: return null
        if (whole && node.whole != null) return node.final ?: node.whole
        return node.best
    }

    private fun answerRequests() {
        val left = ArrayList<Request>()
        for (request in requests) {
            val node = nodes[request.key]
            if (node == null || answered(node, request.whole)) {
                request.answer = answer(request.key, request.whole)
                request.done.countDown()
            } else {
                left.add(request)
            }
        }
        requests = left
    }

    /** Whether the game's position at [ply] has been told all it will be. */
    private fun settled(ply: Int, key: String): Boolean {
        val was = told[ply] ?: return false
        return was.key == key && was.stage == FINAL
    }

    /**
     * The game's position at [ply] told as far as it has come (the file's docstring), each stage
     * once. The event is made before the stage is marked told: a fault making it leaves the stage
     * to be told again, never marked and never said.
     */
    private fun tell(ply: Int) {
        val key = path[ply] ?: return
        val game = gameId ?: return
        val node = nodes.getValue(key)
        val stage = stage(node)
        val was = told[ply]
        if (stage == 0 || (was != null && was.key == key && was.stage >= stage)) return
        val measured = when (stage) {
            FINAL -> node.final
            WHOLE -> node.whole
            else -> annotator.firstAnswer(node.board, node.best, node.hint ?: emptyList())
        }
        val event = measured?.let { annotator.analysisFrom(game, boards[ply] ?: node.board, ply, it, provisional = stage < FINAL) }
        told[ply] = Told(key, stage)
        if (event != null) put(event)
    }

    /**
     * How far [node] has come (a position whose searches all failed is settled, with nothing to
     * tell).
     */
    private fun stage(node: Node): Int = when {
        node.final != null || node.lost -> FINAL
        node.whole != null -> WHOLE
        node.best != null && !node.hint.isNullOrEmpty() -> FIRST_BOTH
        node.best != null || !node.hint.isNullOrEmpty() -> FIRST
        else -> 0
    }

    private fun put(event: Event) {
        results.add(event)
        watch?.said(event)
    }

    /** The workers woken (under [lock]), and a test's watch told. */
    private fun wake() {
        lock.notifyAll()
        watch?.changed()
    }

    private fun uniqueEngines(): List<Engine> {
        val seen = ArrayList<Engine>()
        for (engine in engines) if (seen.none { it === engine }) seen.add(engine)
        return seen
    }

    companion object {
        private const val TAG = "sidecoach.analysis"
        internal const val BEST = "best"  // one line to the full depth
        internal const val HINT = "hint"  // the candidates at EARLY_DEPTH
        internal const val SIDE = "side"  // the side searches of a whole position
    }
}

/**
 * The coach's engine ([ComputedFirst]): its questions answered by the analyser's pool -- the
 * position whole, its candidates at the full depth -- never by a Stockfish of its own (the
 * engine audit, 2026-10-05: the coach's second engine searched three lines, which explode as six
 * do). The depth is the analyser's; the wait is bounded by [timeoutMs]. What it answers the
 * analyser has kept already ([ComputedFirst]'s `fillsComputed`). `analyser.PoolSearcher`.
 */
class PoolSearcher(private val analyser: Analyser, private val timeoutMs: Long? = REQUEST_TIMEOUT_MS) {
    /** [board]'s lines, whole -- or with [bestOnly] its best line, not waiting for the whole -- at the analyser's depth. */
    fun analyse(board: Board, bestOnly: Boolean = false): List<Line> =
        analyser.request(board, whole = !bestOnly, timeoutMs = timeoutMs)?.lines ?: emptyList()
}
