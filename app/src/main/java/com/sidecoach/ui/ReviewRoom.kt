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
import com.sidecoach.llm.apiKeyOf
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
    /** Keeps a PGN in the phone's Downloads (its name, the text) and says where it went. */
    private val saver: ((String, String) -> String)? = null,
    /** Puts a PGN on the phone's clipboard: the web view's own clipboard refuses every write. */
    private val copier: ((String) -> Unit)? = null,
) {
    val key = row.key
    val strength: Strength = GameReview.STRENGTHS[strength] ?: GameReview.STRENGTHS.getValue("standard")
    private val store = Store(storePath)
    private val log = DebugLog()
    private val lock = Any()
    private val stop = AtomicBoolean(false)

    // Counted down by [close]; a room opened again ([reopen]) waits on a new one.
    @Volatile private var stopped = CountDownLatch(1)

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

    // Analysis: the engine made on the first position asked of it, the explorer on the first
    // explore, the opening book on the first opening named.
    private val lines = ReviewAnalysis.Lines(this.strength, makeEngine)
    private var explorer: ReviewAnalysis.Explorer? = null
    private var book: com.sidecoach.analysis.Book? = null
    private var bookRead = false

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
            // Removed meanwhile: the words would be paid for and saved on nothing (the review of
            // 2026-10-01).
            if (!store.inRecord(key)) return GONE
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
                stop = { !store.inRecord(key) },
            )
            if (!store.inRecord(key)) throw IllegalStateException(GONE)
            if (written.failed.isNotEmpty()) {
                // Some calls failed: the answers paid for are kept ([keep]), and Try again asks
                // the rest. They were saved as done, the failed moves without words for good and
                // nothing offered to finish them (the review of 2026-10-01).
                val total = ReviewWords.keyPlies(reviewed).size
                val first = written.dropped[written.failed.sorted().first()]
                val error = "Written on ${total - written.failed.size} of $total key moves; the rest failed ($first)."
                synchronized(lock) {
                    wordsStatus = "failed"
                    wordsError = error
                }
                emit("sys", "error", "the AI coach's words: $error")
                return
            }
            val cost = written.cost?.let { it + spentBefore }
            saved = JSONObject()
                .put("model", written.model)
                .put("cost", cost ?: JSONObject.NULL)
                .put("texts", JSONObject().apply { for ((ply, parts) in written.texts) put(ply.toString(), JSONObject(parts as Map<*, *>)) })
                .put("reveals", JSONArray(written.reveals.sorted()))
            shown = shown(saved, reviewed)
        } catch (exc: Exception) { // a failure is said, never left "writing"
            val gone = runCatching { !store.inRecord(key) }.getOrDefault(false)
            synchronized(lock) {
                wordsStatus = "failed"
                wordsError = if (gone) GONE else exc.message ?: exc.toString()
            }
            emit("sys", "error", "the AI coach's words failed: $exc")
            return
        }
        // The room stays open while they are written ([run]), so its store is open to keep them.
        try {
            store.setReviewWords(key, strength.name, saved.toString())
        } catch (exc: Exception) { // shown even so; said, never left "writing"
            emit("sys", "error", "the AI coach's words could not be saved: $exc")
        }
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

    // ---- analysis (ReviewAnalysis.kt): the engine's lines for the board, the explorer, the game
    // as a PGN ---------------------------------------------------------------------------------

    /** The engine's best lines for a position of the analysis board. `lines`. */
    fun lines(fen: String): JSONObject {
        touched = SystemClock.elapsedRealtime()
        return lines.lines(fen)
    }

    /**
     * What the player's games went on with from [fen], and the opening [moves] (UCI, from the
     * game's start) are in. `explore`.
     */
    fun explore(fen: String, moves: List<String>): JSONObject {
        touched = SystemClock.elapsedRealtime()
        val explorer = synchronized(lock) {
            explorer ?: run {
                val twins = store.twins()
                val watched = Library.fromWatched(store).filter { it.key !in twins }
                ReviewAnalysis.Explorer(store.library() + watched).also { explorer = it }
            }
        }
        val opening = openingOf(moves)
        return explorer.explore(fen).put("opening", opening ?: JSONObject.NULL).put("book", synchronized(lock) { book != null })
    }

    /**
     * The game as a PGN with the analysis board's lines ([tree]), annotated from the review when
     * asked; kept in the phone's Downloads when [save], put on its clipboard when [copy] (the
     * phone's own: the desktop's page copies it itself). `pgn`.
     */
    fun pgn(tree: JSONArray, annotated: Boolean, save: Boolean, copy: Boolean = false): JSONObject {
        touched = SystemClock.elapsedRealtime()
        val (moves, words) = synchronized(lock) { (if (status == "ready") this.moves else null) to this.words }
        val text = ReviewAnalysis.pgn(row, moves, tree, annotated = annotated, words = words)
        val name = ReviewAnalysis.pgnName(row)
        val out = JSONObject().put("pgn", text).put("name", name)
        if (save) {
            val keep = saver ?: throw IllegalStateException("nowhere to keep the PGN")
            out.put("saved", keep(name, text))
        }
        if (copy) {
            val put = copier ?: throw IllegalStateException("no clipboard to copy the PGN to")
            put(text)
            out.put("copied", true)
        }
        return out
    }

    /** The opening the moves are in, by the book, from the game's start position. `_opening_of`. */
    private fun openingOf(moves: List<String>): String? {
        val game = Library.gamesIn(row.pgn).firstOrNull() ?: return null
        if (game.board().fen() != com.sidecoach.chess.STARTING_FEN) return null
        val book = synchronized(lock) {
            if (!bookRead) {
                book = OpeningBook.load(File(files, "openings"))
                bookRead = true
            }
            book
        } ?: return null
        val board = com.sidecoach.chess.Board()
        val played = ArrayList<com.sidecoach.chess.Move>()
        for (uci in moves) {
            val m = com.sidecoach.chess.Move.fromUciOrNull(uci) ?: break
            if (!board.isLegal(m)) break
            board.push(m)
            played.add(m)
        }
        val found = (if (played.isNotEmpty()) book.identify(played) else null) ?: return null
        return if (found.eco.isNotEmpty()) "${found.eco} ${found.name}" else found.name
    }

    /**
     * The page left. The room closes at its next tick -- once the AI coach's words being written
     * are saved, if they are ([run]). `close`.
     */
    fun close() {
        synchronized(lock) {
            stop.set(true)
            stopped.countDown()
        }
    }

    /**
     * Opened again: whether this room goes on. One closed but still open -- its words being
     * written, or its next tick not come -- stays open; one closed, or whose analysis is
     * stopping, does not (a new room goes on from what it kept). `reopen`.
     */
    fun reopen(): Boolean = synchronized(lock) {
        when {
            closed -> false
            !stop.get() -> true
            status != "ready" -> false
            else -> {
                stop.set(false)
                stopped = CountDownLatch(1)
                true
            }
        }
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
            var halted = false
            while (true) {
                val waitOn = synchronized(lock) {
                    // Closed, or idle, while the AI coach's words are written: open until they are
                    // saved. Its store closed under them lost every answer that came after, each
                    // one paid for (the review of 2026-10-01).
                    val writing = wordsStatus == "writing"
                    halted = stop.get()
                    val idle = SystemClock.elapsedRealtime() - touched > IDLE_CLOSE_MS
                    if (!writing && (halted || idle)) {
                        closed = true
                        null
                    } else {
                        stopped
                    }
                } ?: break
                if (halted) Thread.sleep(TICK_MS) else waitOn.await(TICK_MS, TimeUnit.MILLISECONDS)
            }
            if (!halted) emit("sys", "info", "room idle; closing")
        } finally {
            closed = true
            runCatching { store.close() }
            retries?.close()
            lines.close()
        }
    }

    /** The review saved at this strength, when it is of the current format. `_load`. */
    private fun load(): Boolean {
        val saved = store.review(key, strength.name) ?: return false
        if (saved.version != REVIEW_VERSION) return false
        val reviewed = ReviewJson.reviewed(saved.report)
        runCatching { store.clearProgress(key, strength.name) } // a kill between the save and the clearing
        runCatching { keepMissingStats(store, key) } // and between the save and its Stats facts
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
        ) ?: run {
            if (!stop.get()) { // not stopped: the game went from the record meanwhile
                synchronized(lock) {
                    status = "failed"
                    error = GONE
                }
                emit("sys", "warn", "$key left the record while it was reviewed")
            }
            return
        }
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

        /** Why a room has nothing more to do: its game was removed (or the profile reset) meanwhile. */
        const val GONE = "This game is no longer in your record."

        // What a review's summary and its Stats facts are made of from its report ([rederive]):
        // raised when that changes, the reviews kept. 2: the phases are Lichess's Divider's.
        const val DERIVED = 2
        const val DERIVED_KEY = "reviews.derived"

        /**
         * Every saved review's summary, and its facts for Stats, made again from its report when
         * what they are made of has changed since ([DERIVED]): no engine. The phases became
         * Lichess's with the reviews saved as they were: 98 of 106 sides' phase grades and 745 of
         * 3,522 moves' phases in Stats were the old rule's (the review of 2026-10-01). Returns the
         * reviews made again. `rederive`.
         */
        fun rederive(store: Store): Int {
            if (store.setting(DERIVED_KEY) == DERIVED.toString()) return 0
            val counted = store.statStrengths()
            var done = 0
            for ((key, strength) in store.savedReviews()) {
                val saved = store.review(key, strength) ?: continue
                if (saved.version != REVIEW_VERSION) continue
                val ok = runCatching {
                    val reviewed = ReviewJson.reviewed(saved.report)
                    store.setReviewSummary(key, strength, ReviewJson.summary(GameSummaries.summarize(reviewed)).toString())
                    val row = Library.find(store, key)
                    if (counted[key] == strength && row != null) StatFacts.keepStats(store, strength, reviewed, row)
                }.isSuccess
                if (ok) done += 1
            }
            store.setSetting(DERIVED_KEY, DERIVED.toString())
            return done
        }

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
            // The watched game this import stands for, reviewed at this strength: what its
            // review measured is not searched again (its record differs from the import's, or
            // carryTwins would have taken the review whole).
            for ((watched, imported) in store.twins()) {
                val twin = if (imported == row.key) store.review(watched, strength.name) else null
                if (twin == null || twin.version != REVIEW_VERSION) continue
                // A review that no longer reads gives nothing.
                val was = runCatching { ReviewJson.reviewed(twin.report) }.getOrNull() ?: continue
                if (was.depth != strength.depth) continue
                for (analysis in listOf(was.start) + was.moves.map { it.after }) {
                    known.getOrPut(analysis.fen) { Retries.measured(analysis) }
                }
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
            if (!store.inRecord(row.key)) {
                // Taken out of the player's record while it was reviewed: removed, or the profile
                // reset (the review of 2026-10-01: a review saved after a reset left Stats a game
                // that nothing listed and nothing could remove).
                store.clearProgress(row.key)
                return null
            }
            val summary = GameSummaries.summarize(reviewed)
            val saved = store.saveReview(
                row.key,
                strength.name,
                REVIEW_VERSION,
                System.currentTimeMillis() / 1000.0,
                ReviewJson.reviewed(reviewed).toString(),
                ReviewJson.summary(summary).toString(),
            )
            if (!saved) {
                // Another review of the game at this strength finished first and stays, with any
                // words the player paid for on it (the review of 2026-10-01): this one is handed
                // that one. Null if the game was removed.
                val kept = store.review(row.key, strength.name)
                if (kept == null) {
                    store.clearProgress(row.key)
                    return null
                }
                store.clearProgress(row.key, strength.name)
                return ReviewJson.reviewed(kept.report) to ReviewJson.summary(kept.summary ?: "{}")
            }
            // What this strength's review kept part way goes; another strength's stays, to go on from.
            store.clearProgress(row.key, strength.name)
            StatFacts.keepStats(store, strength.name, reviewed, row)
            return reviewed to summary
        }

        /**
         * Each saved review whose Stats facts are missing -- a kill between saving it and keeping
         * them -- has them kept now. The desktop's background review catches these up
         * (`backlog._keep_saved`); the phone has none, so such a game never reached Stats (the
         * review of 2026-10-01): Stats and a review opened catch it up instead. A review that
         * no longer reads is left as it is.
         */
        fun keepMissingStats(store: Store, key: String? = null) {
            for (review in store.reviewsWithoutStats()) {
                if (key != null && review.key != key) continue
                val row = Library.find(store, review.key) ?: continue
                runCatching { StatFacts.keepStats(store, review.strength, ReviewJson.reviewed(review.report), row) }
            }
        }

        /**
         * Each watched game's review carried to its import when the import has none and the two
         * are the same game position for position (Sidecoach read every move): the watched
         * review's measurements with the import's own headers and clocks, summed up and saved
         * with its facts for Stats at once, no engine -- pairing left the game with no accuracy
         * until it was reviewed again (Tai, 2026-09-30). A record that differs is reviewed as
         * the import, from the positions the watched review measured ([reviewRow]). Returns the
         * imports given a review. `carry_twins`.
         */
        fun carryTwins(store: Store, twins: Map<String, String>): List<String> {
            val carried = ArrayList<String>()
            for ((watched, imported) in twins) {
                if (store.bestReview(imported) != null) continue
                val old = store.bestReview(watched)
                val row = Library.find(store, imported)
                if (old == null || old.version != REVIEW_VERSION || row == null) continue
                val games = Library.gamesIn(row.pgn)
                // The report as saved (the desktop's JSON): the import's key, headers and clocks
                // put in, then read back.
                val was = runCatching { JSONObject(old.report) }.getOrNull() ?: continue
                val moves = was.optJSONArray("moves")
                if (games.isEmpty() || moves == null || moves.length() == 0) continue
                val game = games[0]
                val seen = (0 until moves.length()).map { moves.getJSONObject(it).getString("fen_before") } +
                    moves.getJSONObject(moves.length() - 1).getJSONObject("after").getString("fen")
                if (seen != GameReview.positions(game).map { it.fen() }) continue
                val times = GameReview.clocks(game)
                for (i in 0 until moves.length()) {
                    val move = moves.getJSONObject(i)
                    move.getJSONObject("assessment").put("game_id", imported)
                    move.getJSONObject("after").put("game_id", imported)
                    move.put("clock_ms", times[i].first ?: JSONObject.NULL)
                    move.put("spent_ms", times[i].second ?: JSONObject.NULL)
                }
                was.getJSONObject("start").put("game_id", imported)
                was.put("headers", JSONObject().apply { for ((k, v) in game.headers.items()) put(k, v) })
                val reviewed = runCatching { ReviewJson.reviewed(was) }.getOrNull() ?: continue
                val saved = store.saveReview(
                    imported,
                    old.strength,
                    REVIEW_VERSION,
                    System.currentTimeMillis() / 1000.0,
                    ReviewJson.reviewed(reviewed).toString(),
                    ReviewJson.summary(GameSummaries.summarize(reviewed)).toString(),
                )
                if (!saved) continue // reviewed meanwhile (that review stays, with its words), or removed
                old.words?.takeIf { it.isNotEmpty() }?.let { store.setReviewWords(imported, old.strength, it) }
                store.clearProgress(imported, old.strength)
                StatFacts.keepStats(store, old.strength, reviewed, row)
                carried.add(imported)
            }
            return carried
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
    private val saver: ((String, String) -> String)? = null,
    private val copier: ((String) -> Unit)? = null,
) {
    private val rooms = LinkedHashMap<String, ReviewRoom>()
    private val lock = Any()

    // The model as the settings name it, made once for them.
    private var model: LLMProvider? = null
    private var modelFor: List<Any?>? = null

    /** The ledgers of the models replaced. */
    private val spentLedgers = ArrayList<Usage>()

    /** The model the reviews' words are asked of, and why there is none. `model`. */
    fun model(): Pair<LLMProvider?, String?> {
        val llm = settings.llm()
        if (llm.provider == "none") return null to ReviewRoom.NO_AI
        val keys = KeyStore { name -> settings.secret(name) }
        // The key too: a key fixed or forgotten in Settings left the model made with the old one
        // until a restart (the settings review, 2026-10-01).
        val named = listOf(llm.provider, llm.model, llm.baseUrl, llm.reasoningEffort, apiKeyOf(llm, keys))
        synchronized(lock) {
            if (named != modelFor) {
                try {
                    val old = usageOf(model)
                    model = makeProvider(llm, timeoutS = ReviewRoom.AI_TIMEOUT_S, keys = keys)
                    if (old != null) spentLedgers.add(old) // its calls still count
                } catch (exc: Exception) { // a key not given, a name not known
                    return null to (exc.message ?: exc.toString())
                }
                modelFor = named
            }
            return model to null
        }
    }

    /** The model calls of the reviews' words, for the panel's strip. `usages`. */
    fun usages(): List<Usage> = synchronized(lock) { spentLedgers + listOfNotNull(usageOf(model)) }

    /** A game of the library by its key: imported, or one Sidecoach watched. `row`. */
    fun row(key: String): LibraryRow? {
        val store = Store(storePath)
        try {
            return Library.find(store, key)
        } finally {
            store.close()
        }
    }

    /**
     * The game's room at [strength]; with none, at the strongest review saved, else [fallback]
     * (the phone's own, Fast). It opened at the one strength always: a game reviewed Deep was
     * reviewed again, its Stats overwritten (the review of 2026-10-01).
     */
    fun open(key: String, strength: String? = null, fallback: String = "fast"): ReviewRoom? = synchronized(lock) {
        val strength = strength ?: run {
            val store = Store(storePath)
            try {
                store.bestReview(key)?.strength?.takeIf { it in GameReview.STRENGTHS }
            } finally {
                store.close()
            }
        } ?: fallback
        val room = rooms[key]
        if (room != null && room.strength.name == strength && room.reopen()) return room
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
            saver = saver,
            copier = copier,
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
         * A PGN kept in the phone's Downloads, where the Files app and any chess app find it --
         * the desktop's `review_analysis.save` keeps it in the Downloads folder too. Android
         * names a name taken anew ("x (1).pgn"); the name it kept is said.
         */
        fun saveToDownloads(context: Context, name: String, text: String): String {
            val resolver = context.contentResolver
            val values = android.content.ContentValues().apply {
                put(android.provider.MediaStore.Downloads.DISPLAY_NAME, name)
                put(android.provider.MediaStore.Downloads.MIME_TYPE, "application/x-chess-pgn")
                put(android.provider.MediaStore.Downloads.RELATIVE_PATH, android.os.Environment.DIRECTORY_DOWNLOADS)
            }
            val uri = resolver.insert(android.provider.MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                ?: throw java.io.IOException("the phone would not make a file in Downloads")
            resolver.openOutputStream(uri)?.use { it.write(text.toByteArray(Charsets.UTF_8)) }
                ?: throw java.io.IOException("the phone would not write the file")
            val kept = resolver.query(uri, arrayOf(android.provider.MediaStore.Downloads.DISPLAY_NAME), null, null, null)?.use {
                if (it.moveToFirst()) it.getString(0) else null
            } ?: name
            return "Downloads/$kept"
        }

        /**
         * A PGN on the phone's clipboard. The web view's `navigator.clipboard` refuses every
         * write ("Write permission denied", a tap or not), so the app puts it there; Android
         * shows its own note that it was copied.
         */
        fun copyToClipboard(context: Context, text: String) {
            val clipboard = context.getSystemService(android.content.ClipboardManager::class.java)
                ?: throw java.io.IOException("the phone has no clipboard to copy to")
            clipboard.setPrimaryClip(android.content.ClipData.newPlainText("PGN", text))
        }

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
                    saver = { name, text -> saveToDownloads(app, name, text) },
                    copier = { text -> copyToClipboard(app, text) },
                ).also { shared = it }
            }
        }
    }
}
