package com.sidecoach.analysis

import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Event
import com.sidecoach.events.Line
import java.util.TreeMap
import java.util.Timer
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
 * replacing it: its best line and its candidates at [EARLY_DEPTH] (the first answer, at once);
 * its candidates at the full depth (the whole position, provisional); its side searches (final).
 *
 * The workers take, first to last: what the next verdict needs (the position a move was played
 * from, whole; the position it led to, its best line); what the coach asks for ([request]); the
 * position the game is on; the game's other positions (their review is then a search of none of
 * them). A move is judged once the position it was played from is whole -- its candidates were
 * searched while its mover thought -- and the position it led to has its best line, or for a
 * search running past [DECIDED_AFTER_MS], a decided score at the deepest depth it finished (Tai,
 * 2026-10-05: the severities this flips are as few as two depth-18 searches' own disagreement).
 * Verdicts go out in move order.
 *
 * A search fails alone: what it held back is judged or told without it, never waited on.
 */

/**
 * How many workers search side by side, one thread each: the desktop's three, which answer as
 * fast as four or six (tools/engine/workers_compare.py); the phone's two ([PHONE_WORKERS]), the
 * two cores its one engine of two threads took before.
 */
const val WORKERS = 3
const val PHONE_WORKERS = 2

/**
 * A search still running this long whose deepest finished depth shows the position decided (an
 * expected score at most [DECIDED_LOW] or at least [DECIDED_HIGH]) has that depth's score stand
 * in for the full depth's in a verdict (the file's docstring).
 */
const val DECIDED_AFTER_MS = 1000L
const val DECIDED_LOW = 0.15
const val DECIDED_HIGH = 0.85

/**
 * The positions kept, searched, for the game and any game after it (a puzzle tried again is
 * answered at once); the oldest go past this.
 */
const val KEEP_NODES = 20000

/** What a ply has been told, in order; a stage told is not told again for that position. */
private const val FIRST = 1
private const val FIRST_BOTH = 2
private const val WHOLE = 3
private const val FINAL = 4

/**
 * An analyser ready to take moves, its workers' engines started, one thread each ([make] is handed
 * the threads). An engine that will not start is said here, the others closed, rather than
 * failing every search after. `open_analyser`.
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
    try {
        for (engine in engines) engine.open()
    } catch (exc: Exception) {
        for (engine in engines) runCatching { engine.close() }
        throw exc
    }
    val annotator = GameAnnotator(engines[0], depth = depth, multipv = multipv, thresholds = thresholds, tagger = tagger, book = book)
    return Analyser(annotator, engines = engines).also { it.start() }
}

/**
 * A game's positions measured on a pool of workers: submit positions and moves, poll events,
 * never block ([request] alone waits, for the coach's questions).
 */
class Analyser(
    val annotator: GameAnnotator,
    /** One per worker (open or not); the annotator's own engine alone when not given. */
    engines: List<Engine>? = null,
) : AutoCloseable {

    val engines: List<Engine> = if (engines.isNullOrEmpty()) listOf(annotator.engine) else engines.toList()

    /**
     * One position: its board (with the moves that led there, which the engine reads for
     * repetitions), and what has been found of it.
     */
    internal class Node(val key: String, val board: Board) {
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

        val allFailed: Boolean get() = BEST in failed && HINT in failed
    }

    internal class Played(val ply: Int, val before: String, val after: String, val move: Move)

    internal data class Job(val kind: String, val key: String, val era: Int)

    /** What a job's search found. */
    private sealed interface Found {
        class Best(val measured: Measured) : Found
        class Hint(val lines: List<Line>) : Found
        class Side(val measured: Measured) : Found
    }

    /** A position the coach asked for ([request]), and the caller waiting on it. */
    private class Request(val key: String, val whole: Boolean) {
        val done = CountDownLatch(1)
    }

    /** What [told] holds of a ply: its position, and the stage told. */
    private data class Told(val key: String, val stage: Int)

    internal val lock = Object()
    internal var nodes = LinkedHashMap<String, Node>()
    private var gameId: String? = null
    internal val path = TreeMap<Int, String>()  // ply -> the game's position at it
    private val plyOf = HashMap<String, Int>()
    private val moves = HashMap<Int, Played>()
    private var current = 0
    internal var nextVerdict: Int? = null
    private var told = HashMap<Int, Told>()
    private var requests = ArrayList<Request>()
    private val running = HashMap<Int, Job>()  // by worker: the job a stop can reach
    internal val claimed = HashMap<Int, Job>()  // by worker: the job it took, running or about to
    internal var era = 0
        private set

    /** What the workers have finished; a test reads it as the desktop's `_results`. */
    internal val results = ConcurrentLinkedQueue<Event>()
    private val threads = ArrayList<Thread>()
    private var closed = false

    /** The decided rule's clock: one thread for the analyser's whole life. */
    private val timer = Timer("sidecoach-analyser-decided", true)

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
        for (thread in threads) thread.join(5000)
        threads.clear()
        timer.cancel()  // after the workers: none schedules on it any more
        for (engine in uniqueEngines()) engine.close()
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
            val key = node(board).key
            setPath(ply, key)
            current = ply
            // Told again: a position submitted is one the player faces anew (a takeback to the
            // start, a re-read), whatever was told of it before.
            told.remove(ply)
            tell(ply)
            wake()
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
            val source = node(before.copy())
            if (path[ply - 1] != source.key) {
                setPath(ply - 1, source.key)
                tell(ply - 1)
            }
            val after = before.copy()
            after.push(move)
            val target = node(after)
            setPath(ply, target.key)
            moves[ply] = Played(ply, source.key, target.key, move)
            val next = nextVerdict
            if (next == null || ply < next) nextVerdict = ply
            current = ply
            // The verdict first: the word on the move is the news, the position after it next.
            judge()
            tell(ply)
            stopUnwanted()
            wake()
        }
    }

    /**
     * The game's moves after [ply] taken back: nothing more is said of them, and a search made
     * only for them stops.
     */
    fun withdraw(gameId: String, ply: Int) {
        synchronized(lock) {
            if (gameId != this.gameId) return
            val gone = path.keys.filter { it > ply }
            withdrawn += gone.count { it in moves }
            for (p in gone) {
                val key = path.remove(p)!!
                if (plyOf[key] == p) plyOf.remove(key)
                moves.remove(p)
                told.remove(p)
                annotator.forgetAssessed(p)
            }
            val next = nextVerdict
            if (next != null && next > ply + 1) nextVerdict = ply + 1
            if (moves.isEmpty()) nextVerdict = null
            current = ply
            stopUnwanted()
            wake()
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
            plyOf.clear()
            moves.clear()
            told.clear()
            nextVerdict = null
            current = 0
            for (node in nodes.values) node.failed.clear()  // a search that failed is tried again for a new game
            stopUnwanted()
            annotator.reset()
            wake()
        }
    }

    /**
     * The coach's question on a position (a move it or the player asks about, a position its
     * words are about): the position whole -- its candidates at the full depth -- or its best
     * line alone, waited for. Taken after the verdicts and before the game's own work. What is
     * known by [timeoutMs], or when the analyser closes, is the answer then (null for nothing);
     * asked again, what is in comes at once.
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
            wake()
        }
        if (timeoutMs == null) request.done.await() else request.done.await(timeoutMs, TimeUnit.MILLISECONDS)
        synchronized(lock) {
            if (requests.remove(request)) stopUnwanted()  // given up: nothing more is searched for it
            return answer(key, whole)
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
     * under way is not kept ([GameAnnotator.forgetMeasured]).
     */
    fun forgetMeasured() {
        synchronized(lock) {
            era += 1
            // The game's positions and the coach's are kept, unmeasured; the rest go.
            val kept = HashSet(plyOf.keys) + requests.map { it.key }
            val fresh = LinkedHashMap<String, Node>()
            for ((k, n) in nodes) if (k in kept) fresh[k] = Node(k, n.board)
            nodes = fresh
            told = HashMap()
            for (index in running.keys.toList()) engines[index].interrupt()
            annotator.forgetMeasured()
            wake()
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

    internal fun node(board: Board): Node {
        val key = cacheKey(board)
        nodes[key]?.let { return it }
        if (nodes.size >= KEEP_NODES) forgetOldest()
        return Node(key, board.copy()).also { nodes[key] = it }
    }

    /**
     * The oldest tenth of the positions let go, but for the game's, the coach's, those a worker
     * is on, and their candidates (read as they are: choosing them makes nodes).
     */
    private fun forgetOldest() {
        val roots = path.values + requests.map { it.key } + claimed.values.map { it.key }
        val keep = HashSet(roots)
        for (key in roots) nodes[key]?.candidates?.let { found -> keep.addAll(found.map { it.second }) }
        for (key in nodes.keys.take(KEEP_NODES / 10)) {
            if (key in keep) continue
            val node = nodes.remove(key) ?: continue
            for ((_, child) in node.candidates ?: emptyList()) nodes[child]?.parents?.remove(key)
        }
    }

    /**
     * The game's position at [ply] is [key]: the one it held before, if another, is no longer the
     * game's -- nothing more is told of it at this ply.
     */
    private fun setPath(ply: Int, key: String) {
        val old = path[ply]
        if (old != null && old != key) {
            if (plyOf[old] == ply) plyOf.remove(old)
            told.remove(ply)
        }
        path[ply] = key
        plyOf[key] = ply
    }

    /**
     * The position's candidates and their positions' keys: chosen once, when its best line and
     * its hint are in, each candidate's position made a node that knows it.
     */
    private fun candidatesOf(node: Node): List<Pair<Move, String>> {
        if (node.candidates == null && node.firstDone) {
            val best = node.best ?: empty(node)
            node.candidates = annotator.candidates(node.board, best, node.hint ?: emptyList()).map { move ->
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
     * Every position a search is wanted for: the game's, what the coach asked for, and their
     * candidates.
     */
    private fun wanted(): Set<String> {
        val out = HashSet<String>()
        for (key in path.values + requests.map { it.key }) {
            val node = nodes[key] ?: continue
            out.add(key)
            candidatesOf(node).mapTo(out) { it.second }
        }
        return out
    }

    // ---- what the workers take -------------------------------------------------------------

    /**
     * The searches [node] still needs, in order: its best line; for [whole], its hint and its
     * candidates' best lines; for [side], its side searches once it is whole.
     */
    private fun jobsFor(node: Node, whole: Boolean, side: Boolean): List<Job> {
        val jobs = ArrayList<Job>()
        if (node.best == null && BEST !in node.failed) jobs.add(Job(BEST, node.key, era))
        if (!whole) return jobs
        if (node.hint == null && HINT !in node.failed) jobs.add(Job(HINT, node.key, era))
        for ((_, key) in candidatesOf(node)) {
            val child = nodes.getValue(key)
            if (child.best == null && BEST !in child.failed) jobs.add(Job(BEST, key, era))
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
                    picked = pick()
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
            } catch (exc: Exception) {
                // A failed search must never end a session.
                synchronized(lock) {
                    failures += 1
                    lastError = "${exc::class.java.simpleName}: ${exc.message}"
                    val failed = nodes[job.key]
                    if (failed != null && job.era == era) {
                        failed.failed.add(job.kind)
                        advance(failed)
                    }
                }
                null
            }
            synchronized(lock) {
                running.remove(index)
                claimed.remove(index)
                if (found != null && job.era == era && job.key in nodes) found(job, found)
                wake()
            }
        }
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
                    val decide = timer.schedule(DECIDED_AFTER_MS) { decide(job) }
                    try {
                        Found.Best(annotator.bestLine(board, engine = engine, onDepth = { lines -> deepened(job, lines) }))
                    } finally {
                        decide.cancel()
                    }
                }
            }
        }

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
            if (job.key !in wanted) {
                stopped += 1
                engines[index].interrupt()
            }
        }
    }

    // ---- what comes back ---------------------------------------------------------------

    /** A depth of a best line's search finished, the search going on. */
    private fun deepened(job: Job, lines: List<Line>) {
        synchronized(lock) {
            val node = nodes[job.key] ?: return
            if (job.era == era && lines.isNotEmpty()) node.deepest = lines[0]
        }
    }

    /**
     * [DECIDED_AFTER_MS] into a best line's search: a decided score at its deepest finished depth
     * stands in for the full depth's in a verdict (the file's docstring).
     */
    internal fun decide(job: Job) {
        synchronized(lock) {
            val node = nodes[job.key] ?: return
            if (job.era != era || node.best != null) return
            val line = node.deepest ?: return
            if (line.expectedScore > DECIDED_LOW && line.expectedScore < DECIDED_HIGH) return
            node.standIn = Measured(
                fen = node.board.fen(),
                lines = listOf(line),
                score = positionScore(node.board, listOf(line)),
                decisive = false,
                onlyMove = false,
                key = node.key,
            )
            judge()
            wake()
        }
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
     * A search of [node] done (or failed): it and every position it is a candidate of made whole
     * if they can be, and what waited on them judged, answered and told.
     */
    private fun advance(node: Node) {
        for (key in listOf(node.key) + node.parents) {
            val parent = nodes[key]
            if (parent != null && parent.whole == null) complete(parent)
        }
        if (SIDE in node.failed && node.whole != null && node.final == null) node.final = node.whole
        judge()
        answerRequests()
        for (ply in path.keys.toList()) tell(ply)
    }

    /** [node] made whole once its candidates are all in (a failed one left out). */
    private fun complete(node: Node) {
        val candidates = candidatesOf(node)
        if (!node.firstDone || node.allFailed) return
        val children = candidates.map { (move, key) -> move to nodes.getValue(key) }
        if (children.any { (_, child) -> child.best == null && BEST !in child.failed }) return
        val made = annotator.whole(
            node.board,
            node.best ?: empty(node),
            children.mapNotNull { (move, child) -> child.best?.let { move to it } },
        )
        node.whole = made
        annotator.computed.add(node.board, made.lines)
        if (!annotator.withThreats) node.final = made
    }

    /**
     * Each move judged as soon as it can be, in the order of the moves: the move before is what
     * a verdict reads ("the opponent just erred", "a recapture").
     */
    private fun judge() {
        while (true) {
            val at = nextVerdict ?: return
            val move = moves[at] ?: return
            val before = nodes.getValue(move.before)
            val after = nodes.getValue(move.after)
            val whole = before.whole
            if (whole == null) {
                if (before.allFailed) {
                    nextVerdict = at + 1  // nothing to judge it from: it goes without
                    continue
                }
                return
            }
            val own = ownLine(whole.lines, move.move) != null
            if (!own && after.value == null) {
                if (BEST in after.failed) {
                    nextVerdict = at + 1
                    continue
                }
                return
            }
            try {
                val verdict = annotator.assessFrom(gameId ?: "", before.board, move.move, move.ply, whole, if (own) null else after.value)
                put(verdict)
            } catch (exc: Exception) {
                // A move that cannot be judged is skipped.
                failures += 1
                lastError = "${exc::class.java.simpleName}: ${exc.message}"
            }
            nextVerdict = at + 1
        }
    }

    private fun answered(node: Node, whole: Boolean): Boolean =
        if (whole) node.whole != null || node.allFailed else node.best != null || BEST in node.failed

    private fun answer(key: String, whole: Boolean): Measured? {
        val node = nodes[key] ?: return null
        if (whole && node.whole != null) return node.final ?: node.whole
        return node.best
    }

    private fun answerRequests() {
        val left = ArrayList<Request>()
        for (request in requests) {
            if (answered(nodes.getValue(request.key), request.whole)) request.done.countDown() else left.add(request)
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
     * once.
     */
    private fun tell(ply: Int) {
        val key = path[ply] ?: return
        val game = gameId ?: return
        val node = nodes.getValue(key)
        val stage = stage(node)
        val was = told[ply]
        if (stage == 0 || (was != null && was.key == key && was.stage >= stage)) return
        told[ply] = Told(key, stage)
        val measured = when (stage) {
            FINAL -> node.final
            WHOLE -> node.whole
            else -> annotator.firstAnswer(node.board, node.best, node.hint ?: emptyList())
        } ?: return
        put(annotator.analysisFrom(game, node.board, ply, measured, provisional = stage < FINAL))
    }

    /**
     * How far [node] has come (a position whose searches all failed is settled, with nothing to
     * tell).
     */
    private fun stage(node: Node): Int = when {
        node.final != null || node.allFailed -> FINAL
        node.whole != null -> WHOLE
        node.best != null && node.hint != null -> FIRST_BOTH
        node.best != null || node.hint != null -> FIRST
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
        internal const val BEST = "best"  // one line to the full depth
        internal const val HINT = "hint"  // the candidates at EARLY_DEPTH
        internal const val SIDE = "side"  // the side searches of a whole position
    }
}

/**
 * The coach's engine ([ComputedFirst]): its questions answered by the analyser's pool -- the
 * position whole, its candidates at the full depth -- never by a Stockfish of its own (the
 * engine audit, 2026-10-05: the coach's second engine searched three lines, which explode as six
 * do). The depth is the analyser's; the wait is bounded by [timeoutMs]. `analyser.PoolSearcher`.
 */
class PoolSearcher(private val analyser: Analyser, private val timeoutMs: Long = 30_000) {
    /** [board]'s lines, whole, at the analyser's depth. */
    fun analyse(board: Board): List<Line> =
        analyser.request(board, whole = true, timeoutMs = timeoutMs)?.lines ?: emptyList()
}
