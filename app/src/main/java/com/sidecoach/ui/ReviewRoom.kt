package com.sidecoach.ui

import android.content.Context
import android.os.SystemClock
import com.sidecoach.analysis.Engine
import com.sidecoach.analysis.EngineAssets
import com.sidecoach.analysis.GameReview
import com.sidecoach.analysis.Measured
import com.sidecoach.analysis.Motifs
import com.sidecoach.analysis.OpeningBook
import com.sidecoach.analysis.Strength
import com.sidecoach.coach.ReviewLinks
import com.sidecoach.coach.ReviewWords
import com.sidecoach.events.GameSummary
import com.sidecoach.events.ReviewJson
import com.sidecoach.events.ReviewedGame
import com.sidecoach.events.Side
import com.sidecoach.llm.KeyStore
import com.sidecoach.llm.LLMProvider
import com.sidecoach.llm.OpenAICompatProvider
import com.sidecoach.llm.ThinkingControl
import com.sidecoach.llm.Usage
import com.sidecoach.llm.makeProvider
import com.sidecoach.llm.modelOf
import com.sidecoach.llm.pricesFor
import com.sidecoach.llm.usageOf
import com.sidecoach.profile.GameSummaries
import com.sidecoach.profile.Library
import com.sidecoach.profile.LibraryRow
import com.sidecoach.profile.Puzzles
import com.sidecoach.profile.StatFacts
import com.sidecoach.profile.Store
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * A game reviewed in a page of its own (`/review/<key>`; `ui/review_room.py`), apart from the
 * screen the panel watches, as a puzzle is (PuzzleRoom.kt).
 *
 * A room opens a game of the library: the review saved for it at this strength when there is
 * one of the current format, else the game reviewed whole ([GameReview]) on engines of the
 * room's own, side by side, its progress shown as it goes, then summed up ([GameSummaries]) and
 * saved beside the profile. On the phone a game is reviewed only when the player opens it (Tai,
 * 2026-09-30: no background review, as on chess.com), and a review stopped part way -- the page
 * left, the app closed -- goes on from where it was the next time. The page polls the room as
 * the panel polls the engine: the state and the room's log lines. The AI coach's words on the
 * key moves are written only when the player asks ([writeWords]) and kept with the review.
 */
class ReviewRoom(
    private val files: File,
    private val settings: SettingsStore,
    val row: LibraryRow,
    storePath: File?,
    strength: String = "standard",
    private val makeEngine: EngineMaker,
    private val mirror: ((String, String, String, JSONObject?) -> Unit)? = null,
    private val model: (() -> Pair<LLMProvider?, String?>)? = null,
) {
    val key = row.key
    val strength: Strength = GameReview.STRENGTHS[strength] ?: GameReview.STRENGTHS.getValue("standard")
    private val store = Store(storePath)
    private val log = DebugLog()
    private val lock = Any()
    private val stop = AtomicBoolean(false)
    private val stopped = CountDownLatch(1)

    @Volatile var status = "starting" // starting | analysing | ready | failed
        private set
    private var done = 0
    private var total = 0
    private var error: String? = null
    private var reviewed: ReviewedGame? = null
    private var summary: GameSummary? = null
    private var moves = JSONArray()
    private var practice = JSONArray()

    @Volatile var retries: Retries? = null
        private set

    // The AI coach's words on the key moves, as saved: none yet | writing | ready | failed.
    private var words: JSONObject? = null
    private var wordsStatus = "none"
    private var wordsError: String? = null

    // The answers paid for by ply, and what they cost, while words are being written or after
    // the app was closed in the middle of them: kept, and never asked again.
    private var answered = LinkedHashMap<Int, String>()
    private var spent = 0.0

    // The words' cost as the button says it, by model name and the moves answered already.
    private var estimated: Pair<Pair<String?, Set<Int>>, Double?>? = null

    private val playback = playback(row.pgn)

    @Volatile var touched = SystemClock.elapsedRealtime()
        private set

    @Volatile var closed = false
        private set

    private val thread = Thread({ run() }, "review-$key").apply { isDaemon = true }

    init {
        thread.start()
    }

    // ---- the page's side --------------------------------------------------------------------

    fun poll(since: Int = 0): JSONObject =
        JSONObject().put("state", state()).put("lines", log.since(since)).put("cursor", log.cursor)

    fun state(): JSONObject {
        touched = SystemClock.elapsedRealtime()
        val (provider, unavailable) = model?.invoke() ?: (null to NO_AI)
        val estimate = estimate(provider)
        synchronized(lock) {
            val reviewed = reviewed
            return JSONObject()
                .put("key", key)
                .put("status", status)
                .put("progress", JSONObject().put("done", done).put("total", total))
                .put("error", error ?: JSONObject.NULL)
                .put("strength", strength.name)
                .put(
                    "game",
                    JSONObject()
                        .put("source", row.source)
                        .put("white", row.white ?: JSONObject.NULL)
                        .put("black", row.black ?: JSONObject.NULL)
                        .put("white_elo", row.whiteElo ?: JSONObject.NULL)
                        .put("black_elo", row.blackElo ?: JSONObject.NULL)
                        .put("result", row.result ?: JSONObject.NULL)
                        .put("time_class", row.timeClass ?: JSONObject.NULL)
                        .put("played_at", row.playedAt ?: JSONObject.NULL)
                        .put("user_side", row.userSide ?: JSONObject.NULL),
                )
                .put("start_fen", reviewed?.start?.fen ?: JSONObject.NULL)
                // While the game is analysed, its positions: the page's board plays the game out
                // as the progress goes, as chess.com's review does (Tai, 2026-09-30).
                .put("playback", if (status == "starting" || status == "analysing") playback else JSONObject.NULL)
                .put("summary", summary?.let { ReviewJson.summary(it) } ?: JSONObject.NULL)
                .put("moves", moves)
                .put("practice", practice)
                .put(
                    "ai",
                    JSONObject()
                        .put("status", wordsStatus)
                        .put("error", wordsError ?: JSONObject.NULL)
                        .put("words", words ?: JSONObject.NULL)
                        .put("unavailable", unavailable ?: JSONObject.NULL)
                        .put("moves", if (reviewed != null) ReviewWords.keyPlies(reviewed).size else 0)
                        .put("answered", if (wordsStatus != "ready") answered.size else 0)
                        .put("estimate", estimate ?: JSONObject.NULL),
                )
                .put("closed", closed)
        }
    }

    /**
     * What the AI coach's words would cost with [provider] (dollars), made once a model; null
     * when its prices are not known (a model on the player's own computer). `_estimate`.
     */
    private fun estimate(provider: LLMProvider?): Double? {
        val name = modelOf(provider)
        val reviewed = synchronized(lock) { reviewed }
        if (provider == null || reviewed == null) return null
        val done = synchronized(lock) { answered.keys.toSet() }
        val cached = estimated
        if (cached == null || cached.first != (name to done)) {
            val prices = pricesFor(settings.llm(), name)
            val cost = ReviewWords.estimate(reviewed, row.userSide, ReviewViews.opening(reviewed), prices, done)
            estimated = (name to done) to cost
        }
        return estimated?.second
    }

    /**
     * Set the AI coach writing on the key moves, in the background; why it cannot, else null
     * (already writing is no reason). `write_words`.
     */
    fun writeWords(): String? {
        val (provider, unavailable) = model?.invoke() ?: (null to NO_AI)
        val reviewed: ReviewedGame
        synchronized(lock) {
            val r = this.reviewed
            if (status != "ready" || r == null) return "the review is not ready"
            if (wordsStatus == "writing") return null
            if (provider == null) return unavailable
            wordsStatus = "writing"
            wordsError = null
            reviewed = r
        }
        Thread({ writeWords(reviewed, provider!!) }, "words-$key").apply { isDaemon = true }.start()
        return null
    }

    private fun writeWords(reviewed: ReviewedGame, given: LLMProvider) {
        // A model that reasons by its effort (luna) reasons at REVIEW_EFFORT here, whatever the
        // live coach is set to; a local model does not reason: it reasons worse about chess, not
        // better (Tai, 2026-09-30).
        val reasons = given is OpenAICompatProvider && given.thinkingControl == ThinkingControl.EFFORT
        val provider = if (reasons) (given as OpenAICompatProvider).withEffort(REVIEW_EFFORT) else given
        val prices = pricesFor(settings.llm(), modelOf(provider))
        val (before, spentBefore) = synchronized(lock) { LinkedHashMap(answered) to spent }

        // Each answer kept the moment it is paid for: the app closed now loses none of them.
        fun keep(ply: Int, text: String, stats: Map<String, Any?>) {
            val one = Usage()
            one.record(stats)
            val partial = synchronized(lock) {
                answered[ply] = text
                spent += one.cost(prices) ?: 0.0
                JSONObject()
                    .put("partial", true)
                    .put("model", modelOf(provider) ?: "")
                    .put("cost", spent)
                    .put("answers", JSONObject().apply { for ((p, t) in answered) put(p.toString(), t) })
            }
            store.setReviewWords(key, strength.name, partial.toString())
        }

        val saved: JSONObject
        val shown: JSONObject
        val written: ReviewWords.Words
        try {
            written = ReviewWords.write(
                reviewed,
                provider,
                user = row.userSide,
                opening = ReviewViews.opening(reviewed),
                prices = prices,
                thinking = reasons,
                answered = before,
                onAnswer = ::keep,
            )
            val cost = written.cost?.let { it + spentBefore }
            saved = JSONObject()
                .put("model", written.model)
                .put("cost", cost ?: JSONObject.NULL)
                .put("texts", JSONObject().apply { for ((ply, parts) in written.texts) put(ply.toString(), JSONObject(parts as Map<*, *>)) })
                .put("reveals", JSONArray(written.reveals.sorted()))
            shown = shown(saved, reviewed)
        } catch (exc: Exception) { // a failure is said, never left "writing"
            synchronized(lock) {
                wordsStatus = "failed"
                wordsError = exc.message ?: exc.toString()
            }
            emit("sys", "error", "the AI coach's words failed: $exc")
            return
        }
        runCatching { store.setReviewWords(key, strength.name, saved.toString()) } // a room closed meanwhile keeps them unsaved
        synchronized(lock) {
            words = shown
            wordsStatus = "ready"
            answered = LinkedHashMap()
            spent = 0.0
        }
        val cost = if (saved.isNull("cost")) "" else ", $" + String.format(Locale.ROOT, "%.4f", saved.getDouble("cost"))
        emit("sys", "info", "AI words on ${written.texts.size} of ${ReviewWords.keyPlies(reviewed).size} key moves (${written.model}$cost)")
        for ((ply, why) in written.dropped.toSortedMap()) emit("sys", "warn", "AI words on ply $ply dropped: $why")
    }

    /**
     * The saved words as the page draws them: each paragraph also as its links to the board
     * (coach/ReviewLinks.kt), made on showing so they follow the linking's rules. `_shown`.
     */
    private fun shown(saved: JSONObject, reviewed: ReviewedGame): JSONObject {
        val links = ReviewLinks.linked(saved.optJSONObject("texts") ?: JSONObject(), reviewed, row.userSide)
        return JSONObject(saved.toString()).put("links", links)
    }

    fun close() {
        stop.set(true)
        stopped.countDown()
    }

    fun join(timeoutMs: Long = 5000) = thread.join(timeoutMs)

    // ---- the review ---------------------------------------------------------------------------

    private fun run() {
        try {
            try {
                if (!load()) review()
            } catch (exc: Exception) { // a failed review is said, not raised
                // The room stays: the page reads the failure at its next poll.
                synchronized(lock) {
                    status = "failed"
                    error = exc.message ?: exc.toString()
                }
                emit("sys", "error", "review failed: $exc")
            }
            while (!stop.get()) {
                if (SystemClock.elapsedRealtime() - touched > IDLE_CLOSE_MS) {
                    emit("sys", "info", "room idle; closing")
                    break
                }
                stopped.await(TICK_MS, TimeUnit.MILLISECONDS)
            }
        } finally {
            closed = true
            runCatching { store.close() }
            retries?.close()
        }
    }

    /** The review saved at this strength, when it is of the current format. `_load`. */
    private fun load(): Boolean {
        val saved = store.review(key, strength.name) ?: return false
        if (saved.version != REVIEW_VERSION) return false
        val reviewed = ReviewJson.reviewed(saved.report)
        runCatching { store.clearProgress(key) } // a kill between the save and the clearing
        // Its words ready with it: the page stops asking once the review is. Words the app was
        // closed in the middle of writing are the answers paid for so far, kept to go on from.
        val kept = saved.words?.takeIf { it.isNotEmpty() }?.let { JSONObject(it) }
        var words: JSONObject? = null
        if (kept != null && kept.optBoolean("partial", false)) {
            val answers = kept.optJSONObject("answers") ?: JSONObject()
            synchronized(lock) {
                answered = LinkedHashMap<Int, String>().apply { for (p in answers.keys()) put(p.toInt(), answers.get(p).toString()) }
                spent = if (kept.isNull("cost")) 0.0 else kept.optDouble("cost", 0.0)
            }
        } else if (kept != null && kept.length() > 0) {
            words = shown(kept, reviewed)
        }
        ready(reviewed, ReviewJson.summary(saved.summary ?: "{}"), words)
        emit("sys", "info", "review of $key loaded (${strength.name})")
        return true
    }

    private fun review() {
        val workers = GameReview.workersFor()
        synchronized(lock) { status = "analysing" }
        emit("sys", "info", "reviewing $key: depth ${strength.depth}, $workers engines")
        val done = reviewRow(
            settings,
            files,
            row,
            store,
            strength,
            workers = workers,
            makeEngine = makeEngine,
            onProgress = { d, t -> synchronized(lock) { this.done = d; total = t } },
            stop = stop,
        ) ?: return
        val (reviewed, summary) = done
        ready(reviewed, summary)
        emit("sys", "info", "reviewed in ${String.format(Locale.ROOT, "%.1f", reviewed.seconds)}s")
    }

    /**
     * Puzzles for what the game passed up (the last card's training): the motifs the player's
     * mistakes missed -- both sides' when the library does not know which was the player's --
     * each with puzzles of it at the player's puzzle level. `_practice`.
     */
    private fun practice(reviewed: ReviewedGame): JSONArray {
        val user = row.userSide
        val sides = if (user != null) setOf(Side.of(user)) else setOf(Side.WHITE, Side.BLACK)
        val motifs = GameSummaries.missedMotifs(reviewed, sides).take(PRACTICE_MOTIFS)
        val puzzles = (if (motifs.isNotEmpty()) runCatching { Puzzles.openStore(files) }.getOrNull() else null)
            ?: return JSONArray()
        try {
            val level = Puzzles.puzzleLevel(store)
            return JSONArray(
                motifs.map { (motif, missed) ->
                    JSONObject()
                        .put("motif", motif)
                        .put("missed", missed)
                        .put(
                            "puzzles",
                            JSONArray(puzzles.pick(motif, level, count = PRACTICE_PUZZLES).map { JSONObject().put("id", it.id).put("rating", it.rating) }),
                        )
                },
            )
        } finally {
            puzzles.close()
        }
    }

    private fun ready(reviewed: ReviewedGame, summary: GameSummary, words: JSONObject? = null) {
        val moves = ReviewViews.moveViews(reviewed)
        val practice = practice(reviewed)
        val retries = Retries(
            reviewed,
            strength,
            gameId = key,
            thresholds = settings.thresholds(),
            makeEngine = makeEngine,
            tagger = Motifs,
            book = { OpeningBook.load(File(files, "openings")) },
        )
        synchronized(lock) {
            this.reviewed = reviewed
            this.summary = summary
            this.moves = moves
            this.practice = practice
            this.retries = retries
            if (words != null) {
                this.words = words
                wordsStatus = "ready"
            }
            status = "ready"
            total = reviewed.moves.size + 1
            done = total
        }
    }

    private fun emit(chan: String, level: String, text: String, data: JSONObject? = null) {
        log.emit(chan, level, text, data)
        mirror?.let { runCatching { it("review", level, "$key $chan: $text", data) } }
    }

    companion object {
        // The review's format: a review saved in an older one is done again.
        const val REVIEW_VERSION = 1

        // The loop's pace, and how long an untouched room lives.
        private const val TICK_MS = 250L
        private const val IDLE_CLOSE_MS = 20 * 60 * 1000L

        // The last card's training: the motifs the game missed most, and puzzles of each.
        const val PRACTICE_MOTIFS = 2
        const val PRACTICE_PUZZLES = 3

        // A model call on one key move may take this long (a model that reasons first is slow).
        const val AI_TIMEOUT_S = 300.0

        // The review's words reason at this effort (Tai 2026-09-30: "Reasoning none is usually
        // not very accurate"), though the live coach may be set to none for speed.
        const val REVIEW_EFFORT = "medium"
        const val NO_AI = "No AI coach is set up: choose one in Settings."

        /**
         * The game's main line as the board plays it while the game is analysed: the position
         * before any move, then each move (UCI) and the position it leaves; none for a PGN with
         * no game in it. `playback`.
         */
        fun playback(pgn: String): JSONArray {
            val game = Library.gamesIn(pgn).firstOrNull() ?: return JSONArray()
            val board = game.board()
            val out = JSONArray().put(JSONObject().put("fen", board.fen()).put("uci", JSONObject.NULL))
            for (move in game.mainlineMoves()) {
                board.push(move)
                out.put(JSONObject().put("fen", board.fen()).put("uci", move.uci()))
            }
            return out
        }

        /**
         * A game of the library reviewed whole on [workers] engines of its own, summed up, and
         * saved with its facts for Stats; null when stopped part way. Every position is kept as
         * it is measured (Store.addProgress), so a review stopped part way -- the page left, the
         * app quit or killed -- goes on from there next time instead of starting over (Tai,
         * 2026-09-30); what was kept goes once the review is saved. `review_row`.
         */
        fun reviewRow(
            settings: SettingsStore,
            files: File,
            row: LibraryRow,
            store: Store,
            strength: Strength,
            workers: Int,
            makeEngine: EngineMaker,
            onProgress: ((Int, Int) -> Unit)? = null,
            stop: AtomicBoolean? = null,
        ): Pair<ReviewedGame, GameSummary>? {
            val games = Library.gamesIn(row.pgn)
            if (games.isEmpty()) throw IllegalArgumentException("the game's PGN has no moves")
            val known = HashMap<String, Measured>()
            for (text in store.progress(row.key, strength.name, strength.depth, strength.multipv)) {
                // A record that no longer reads is measured again.
                runCatching { ReviewJson.measured(JSONObject(text)) }.getOrNull()?.let { known[it.fen] = it }
            }

            fun keep(measured: Measured) {
                store.addProgress(row.key, strength.name, strength.depth, strength.multipv, measured.fen, ReviewJson.measured(measured).toString())
            }

            val engines = ArrayList<Engine>()
            val reviewed = try {
                repeat(workers) { engines.add(makeEngine.make(threads = GameReview.THREADS, multipv = strength.multipv).also { it.open() }) }
                GameReview.reviewGame(
                    games[0],
                    engines,
                    depth = strength.depth,
                    multipv = strength.multipv,
                    thresholds = settings.thresholds(),
                    tagger = Motifs,
                    book = OpeningBook.load(File(files, "openings")),
                    strength = strength.name,
                    gameId = row.key,
                    onProgress = onProgress,
                    stop = stop,
                    known = known,
                    onMeasured = ::keep,
                )
            } finally {
                for (engine in engines) runCatching { engine.close() }
            }
            if (stop != null && stop.get()) return null
            val summary = GameSummaries.summarize(reviewed)
            store.saveReview(
                row.key,
                strength.name,
                REVIEW_VERSION,
                System.currentTimeMillis() / 1000.0,
                ReviewJson.reviewed(reviewed).toString(),
                ReviewJson.summary(summary).toString(),
            )
            store.clearProgress(row.key)
            StatFacts.keepStats(store, strength.name, reviewed, row)
            return reviewed to summary
        }

        /**
         * The engines a room makes, as the desktop's `Engine(path=find_stockfish(settings),
         * ...)`: the bundled Stockfish with the network Setup fetched, the player's engine
         * settings, at most 128 MB each.
         */
        fun engineMaker(context: Context, settings: SettingsStore): EngineMaker = EngineMaker { threads, multipv ->
            val binary = Engine.binaryIn(context.applicationInfo.nativeLibraryDir)
            val net = EngineAssets.found(context)
            if (!binary.exists() || net == null) throw IllegalStateException("no Stockfish to review with")
            val config = settings.engine()
            Engine(
                binary,
                net,
                threads = threads,
                hashMb = minOf(config.hashMb, 128),
                multipv = multipv,
                pvPlies = config.pvPlies,
                scoreModel = config.scoreModel,
                maxSeconds = config.maxSeconds,
            )
        }
    }
}

/** The open review rooms by library key, and the model their words are asked of. `ReviewRooms`. */
class ReviewRooms(
    val settings: SettingsStore,
    val storePath: File?,
    private val files: File,
    private val makeEngine: EngineMaker,
    private val mirror: ((String, String, String, JSONObject?) -> Unit)? = null,
) {
    private val rooms = LinkedHashMap<String, ReviewRoom>()
    private val lock = Any()

    // The model as the settings name it, made once for them.
    private var model: LLMProvider? = null
    private var modelFor: List<Any?>? = null

    /** The model the reviews' words are asked of, and why there is none. `model`. */
    fun model(): Pair<LLMProvider?, String?> {
        val llm = settings.llm()
        if (llm.provider == "none") return null to ReviewRoom.NO_AI
        val named = listOf(llm.provider, llm.model, llm.baseUrl, llm.reasoningEffort)
        synchronized(lock) {
            if (named != modelFor) {
                try {
                    model = makeProvider(llm, timeoutS = ReviewRoom.AI_TIMEOUT_S, keys = KeyStore { name -> settings.secret(name) })
                } catch (exc: Exception) { // a key not given, a name not known
                    return null to (exc.message ?: exc.toString())
                }
                modelFor = named
            }
            return model to null
        }
    }

    /** The model calls of the reviews' words, for the panel's strip. `usages`. */
    fun usages(): List<Usage> = synchronized(lock) { listOfNotNull(usageOf(model)) }

    /** A game of the library by its key: imported, or one Sidecoach watched. `row`. */
    fun row(key: String): LibraryRow? {
        val store = Store(storePath)
        try {
            return Library.find(store, key)
        } finally {
            store.close()
        }
    }

    fun open(key: String, strength: String = "standard"): ReviewRoom? = synchronized(lock) {
        val room = rooms[key]
        if (room != null && !room.closed && room.strength.name == strength) return room
        val row = row(key) ?: return null
        room?.close()
        ReviewRoom(
            files,
            settings,
            row,
            storePath = storePath,
            strength = strength,
            makeEngine = makeEngine,
            mirror = mirror,
            model = ::model,
        ).also { rooms[key] = it }
    }

    fun get(key: String): ReviewRoom? {
        val room = synchronized(lock) { rooms[key] }
        return if (room != null && !room.closed) room else null
    }

    companion object {
        @Volatile private var shared: ReviewRooms? = null

        /** The app's rooms if any were made, without making them. */
        fun sharedOrNull(): ReviewRooms? = shared

        /**
         * The app's rooms: the profile store in the app's files, engines of their own, every
         * line mirrored into the panel's log. One per process, so a review under way outlives
         * the window being rebuilt (a turn of the phone).
         */
        fun shared(context: Context): ReviewRooms = shared ?: synchronized(this) {
            shared ?: run {
                val app = context.applicationContext
                val settings = SettingsStore(app)
                ReviewRooms(
                    settings,
                    storePath = File(app.filesDir, "profile.sqlite"),
                    files = app.filesDir,
                    makeEngine = ReviewRoom.engineMaker(app, settings),
                    mirror = { chan, level, text, data -> PanelEngine.emit(chan, level, text, data) },
                ).also { shared = it }
            }
        }
    }
}
