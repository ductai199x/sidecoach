package com.sidecoach.ui

import android.content.Context
import android.os.SystemClock
import com.sidecoach.analysis.Assessing
import com.sidecoach.coach.CoachRunner
import com.sidecoach.events.CoachMessage
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.Trigger
import com.sidecoach.llm.Usage
import com.sidecoach.llm.modelOf
import com.sidecoach.llm.total
import com.sidecoach.profile.Puzzle
import com.sidecoach.profile.PuzzleStore
import com.sidecoach.profile.Puzzles
import com.sidecoach.profile.Store
import java.io.File
import java.math.BigDecimal
import java.math.RoundingMode
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import org.json.JSONArray
import org.json.JSONObject

/**
 * A puzzle on a page of its own (`/puzzle/<id>`), apart from the screen the panel watches.
 * `sidecoach/ui/puzzle_room.py`.
 *
 * Tai, 2026-09-20: puzzle mode "should open in a new tab with a new UI and not interfere with
 * the current window where the game capture is going on". So a puzzle room has its own coach
 * (quiet unless asked), its own engine, its own log and its own little loop thread; the
 * session goes on following the screen. The room drives a [PuzzleSession] -- the board, the
 * line, the outcome -- and the page polls it the way the panel polls the engine; a move, a
 * take-back and a give-up are played on the page's request and answered with the state they
 * leave. The outcome
 * goes to the profile store beside the puzzle's motif (`Puzzles.recordAttempt`). The coach
 * speaks through the voice the panel was given (`PanelVoice`), when the page has it on.
 *
 * Its engine is a second Stockfish, as on the desktop. It searches only while a puzzle is
 * worked, and the app reads no screen while its own page is in front, so the game's engine is
 * idle meanwhile: the two take turns rather than compete.
 */
class PuzzleRoom(
    context: Context,
    val settings: SettingsStore,
    puzzle: Puzzle,
    motif: String?,
    storePath: File?,
    private val speak: ((String) -> Unit)? = null,
    private val stopSpeaking: (() -> Unit)? = null,
    analysis: Boolean = true,
    coach: Boolean = true,
    /**
     * The parent panel's log, when there is one: every line of this room goes there too, under
     * a "puzzle" channel, so the puzzle page itself shows no terminal (Tai, 2026-09-21: the
     * debugging stays in the parent window).
     */
    private val mirror: ((String, String, String, JSONObject?) -> Unit)? = null,
) {
    val session = PuzzleSession(puzzle, motif)
    val store = Store(storePath)
    private val log = DebugLog()
    private val id = puzzle.id
    private val lock = Any()
    private val commands = ArrayList<Pair<String, Any?>>()
    private val stop = CountDownLatch(1)

    @Volatile var voice: Boolean = speak != null
        private set

    /** The room's clock, in ms: a test moves it. */
    @Volatile var clockMs: () -> Long = { SystemClock.elapsedRealtime() }

    /** When the page was last heard from, and whether it last said it was out of view. */
    @Volatile var seen: Long = clockMs()
        private set

    @Volatile var hidden: Boolean = false
        private set

    @Volatile var closed = false
        private set

    /** Told to close; the loop thread finishes it within a tick. */
    @Volatile var closing = false
        private set

    // The engine and the coach of this room alone; the coach is told the position as a game
    // of kind "puzzle" and kept quiet: it speaks when asked.
    private val assessing = Assessing.fromSettings(context, settings, analysis)
    private val coaching = CoachRunner(
        settings,
        coach && assessing.on,
        if (coach && assessing.on) Assessing.coachEngine(context, settings) else null,
        assessing.computed,
    )
    private val thread: Thread

    init {
        assessing.sink = { line -> emit("engine", "debug", line) }
        coaching.sink = { message -> onMessage(message) }
        coaching.coach?.quiet = true
        emit("sys", "info", "engine: ${assessing.detail}")
        emit("sys", "info", "coach: ${coaching.detail}")
        emit(
            "sys",
            "info",
            "puzzle ${puzzle.id}: ${motif ?: "practice"}, rated ${puzzle.rating}, " +
                "${session.side.name.lowercase()} to move; level ${Puzzles.puzzleLevel(store)}",
        )
        val started = session.started()
        assessing.submit(started)
        coaching.events(listOf(started), now())
        thread = Thread({ run() }, "puzzle-${puzzle.id}").apply {
            isDaemon = true
            start()
        }
    }

    // ---- the page's side -----------------------------------------------------------------

    /** The page's poll; [hidden]: the page is out of view (it says so as it goes). */
    fun poll(since: Int = 0, hidden: Boolean = false): JSONObject {
        this.hidden = hidden
        return JSONObject().apply {
            put("state", state())
            put("lines", log.since(since))
            put("cursor", log.cursor)
        }
    }

    /**
     * The page said it was in view and has not been heard from since: gone without a word
     * ([VANISHED_MS]). `puzzle_room.PuzzleRoom.vanished`.
     */
    fun vanished(): Boolean {
        // The clock read on every check, hidden or not: the reading has no effect, and each read
        // is a check the tests can wait on (PuzzleModeTest), as the desktop's wait on `vanished`.
        val now = clockMs()
        return !hidden && now - seen > VANISHED_MS
    }

    /** The session's state as the page reads it, with the room's voice, its closing and its model calls. */
    fun state(): JSONObject {
        seen = clockMs()
        val state = synchronized(lock) { session.state() }
        state["voice"] = voice
        state["voice_ready"] = speak != null
        state["closed"] = closed
        val ledgers = usages()
        if (ledgers.isNotEmpty()) {
            state["llm"] = total(ledgers, settings.llm(), modelOf(coaching.coach?.speaker?.provider))
        }
        return jsonOf(state)
    }

    // A move, a take-back and a give-up are played on the page's request and answered with the
    // state they leave: they waited for the loop's next tick, and the page for its next poll
    // after that, half a second the board stood still after a move (Tai, 2026-09-27). What the
    // engine and the coach are told goes through the loop as before.

    fun move(uci: String): JSONObject {
        synchronized(lock) {
            val events = session.move(uci)
            commands.add("played" to (uci to events))
        }
        return state()
    }

    fun retract(): JSONObject {
        synchronized(lock) { session.retract() }
        return state()
    }

    fun giveUp(): JSONObject {
        synchronized(lock) {
            val events = session.giveUp()
            commands.add("gave_up" to events)
        }
        return state()
    }

    /** A command for the loop thread: ask <text> | voice <bool> | close. */
    fun command(name: String, payload: Any? = null) {
        seen = clockMs()
        if (name == "close") {
            // Said at once, so a page reloading (its old page's close, then its open) gets a
            // fresh room rather than this one on its way out.
            closing = true
        }
        synchronized(lock) { commands.add(name to payload) }
    }

    // ---- the loop ------------------------------------------------------------------------

    private fun run() {
        try {
            while (stop.count > 0) {
                val ts = now()
                drainCommands(ts)
                val engineEvents = assessing.drain()
                coaching.events(engineEvents, ts)
                coaching.tick(ts)
                coaching.drain()
                if (vanished()) {
                    emit("sys", "info", "the page is gone without closing; closing")
                    break
                }
                stop.await(TICK_MS, TimeUnit.MILLISECONDS)
            }
        } finally {
            finish()
        }
    }

    private fun drainCommands(ts: Double) {
        val taken = synchronized(lock) { ArrayList(commands).also { commands.clear() } }
        for ((name, payload) in taken) {
            when (name) {
                "played" -> {
                    @Suppress("UNCHECKED_CAST")
                    val played = payload as Pair<String, List<MovePlayed>>
                    val (uci, events) = played
                    dispatch(events, ts)
                    if (events.isEmpty()) {
                        if (session.wrong == uci) emit("game", "info", "$uci: not the move")
                    } else if (session.done && session.status == "solved") {
                        record(solved = !session.failed)
                    }
                }
                "gave_up" -> {
                    @Suppress("UNCHECKED_CAST")
                    dispatch(payload as List<MovePlayed>, ts)
                    record(solved = false)
                }
                "ask" -> {
                    val text = payload.toString()
                    if (coaching.on) {
                        emit("user", "info", "you: $text")
                        coaching.ask(text, ts)
                    } else {
                        // No coach here (no Stockfish yet): said, as the panel says it, so the
                        // page's typing dots end; asked in silence, they stayed for good.
                        emit("user", "warn", "no coach to ask: $text")
                    }
                }
                "voice" -> {
                    voice = payload == true && speak != null
                    if (!voice) stopSpeaking?.invoke()
                }
                "close" -> {
                    if (!session.done) record(solved = false)
                    stop.countDown()
                }
            }
        }
    }

    private fun dispatch(events: List<MovePlayed>, ts: Double) {
        for (event in events) {
            val who = if (event.byUser) "you" else "the line"
            emit(
                "game",
                "info",
                "${event.san}  ($who)",
                JSONObject().put("uci", event.uci).put("ply", event.ply),
            )
            assessing.submit(event)
        }
        coaching.events(events, ts)
    }

    private fun record(solved: Boolean) {
        if (session.recorded) return
        session.recorded = true
        val outcome = try {
            Puzzles.recordAttempt(
                store,
                session.puzzle.id,
                session.motif,
                solved,
                session.sans.toList(),
                System.currentTimeMillis() / 1000.0,
            )
        } catch (exc: Exception) { // the record must not break the puzzle
            emit("sys", "error", "profile error: ${PanelEngine.pyRepr(exc)}")
            return
        }
        emit(
            "game",
            "info",
            "puzzle ${outcome.puzzleId} ${if (outcome.solved) "solved" else "failed"}" +
                (if (outcome.motif != null) " (${outcome.motif})" else "") +
                "; puzzle level ${outcome.level}",
            JSONObject().put("solved", outcome.solved).put("puzzle_level", outcome.level),
        )
    }

    /**
     * A coach message: logged for the page, spoken when the voice is on. Only what the player
     * asked for comes (the coach is quiet), so nothing is stale.
     */
    private fun onMessage(message: CoachMessage) {
        val trigger = message.trigger.slug
        emit(
            "coach",
            "info",
            "$trigger — ${PanelEngine.elide(message.text, 80)}",
            JSONObject().apply {
                put("full", message.text)
                put("trigger", trigger)
                put("annotations", PanelEngine.marks(message.annotations))
                put("fen", message.fen ?: JSONObject.NULL)
            },
        )
        val speak = speak
        if (voice && speak != null && message.trigger == Trigger.USER_ASK) {
            try {
                speak(message.text)
            } catch (exc: Exception) {
                emit("voice", "warn", "voice failed: ${exc.message ?: ""}")
            }
        }
    }

    private fun emit(chan: String, level: String, text: String, data: JSONObject? = null) {
        log.emit(chan, level, text, data)
        val mirror = mirror ?: return
        runCatching { mirror("puzzle", level, "$id $chan: $text", data) }
    }

    /** The coach's ledgers, kept past its close. */
    @Volatile
    private var ledgers: List<Usage> = emptyList()

    /**
     * The room's model-call ledgers, for the parent panel's strip: kept from the start, as
     * the coach is closed with the room. `PuzzleRoom.usages`.
     */
    fun usages(): List<Usage> {
        coaching.coach?.speaker?.let { ledgers = it.usages() }
        return ledgers
    }

    private fun finish() {
        closed = true
        runCatching { assessing.close() }
        usages() // the ledgers kept before the coach goes
        runCatching { coaching.close() }
        store.close()
    }

    fun close() = command("close")

    fun join(timeoutMs: Long = 5000) = thread.join(timeoutMs)

    /** The session's state as the page reads it: `json.dumps` of the dict. */
    private fun jsonOf(map: Map<String, Any?>): JSONObject {
        val out = JSONObject()
        for ((key, value) in map) out.put(key, wrap(value))
        return out
    }

    private fun wrap(value: Any?): Any = when (value) {
        null -> JSONObject.NULL
        is Map<*, *> -> JSONObject().apply { for ((k, v) in value) put(k.toString(), wrap(v)) }
        is List<*> -> JSONArray().apply { for (one in value) put(wrap(one)) }
        else -> value
    }

    private fun now(): Double = SystemClock.elapsedRealtime() / 1000.0

    companion object {
        /** The loop's pace. */
        const val TICK_MS = 250L

        /**
         * A room closes when its page says it goes (its close, its beacon as it unloads or
         * leaves) and with the app; never for being left alone (Tai, 2026-10-02: a puzzle left
         * open over lunch was gone after 20 idle minutes). The one safety net is for a page that
         * vanished without a word (its process died): a page in view polls four times a second,
         * so one that last said it was in view and has not polled for this long is gone. A page
         * out of view (its timers slowed or stopped) is never closed by it. `VANISHED_S`.
         */
        const val VANISHED_MS = 60_000L
    }
}

/**
 * The open rooms by puzzle id, made on demand from the puzzle store. `puzzle_room.PuzzleRooms`.
 */
class PuzzleRooms(
    private val context: Context,
    val settings: SettingsStore,
    val storePath: File?,
    @Volatile var puzzles: PuzzleStore?,
    /**
     * Where the puzzle store is opened when there was none: puzzles Setup fetches after the
     * rooms were made are then found at the next open, as the session finds them at its next
     * Start.
     */
    val dataDir: File? = null,
    private val speak: ((String) -> Unit)? = null,
    private val stopSpeaking: (() -> Unit)? = null,
    val analysis: Boolean = true,
    val coach: Boolean = true,
    private val mirror: ((String, String, String, JSONObject?) -> Unit)? = null,
) {
    private val rooms = LinkedHashMap<String, PuzzleRoom>()

    /** The ledgers of rooms replaced by a reopened puzzle. */
    private val spent = ArrayList<Usage>()
    private val lock = Any()

    /** The motif each puzzle was recommended for, told by the session's profiler. */
    var motifs: Map<String, String> = emptyMap()

    /** The rooms being made, by puzzle: a second open of one waits for the first's. */
    private val opening = HashMap<String, CountDownLatch>()

    /**
     * The puzzle's room, made if it has none open. It is made outside the lock: making one asks
     * the model whether it answers, and the session loop reads the rooms' spend under this lock
     * every tick -- the live game froze for the length of that check (the API review,
     * 2026-10-01). Two opens of one puzzle at once make one room. `PuzzleRooms.open`.
     */
    fun open(puzzleId: String): PuzzleRoom? {
        var found: Puzzle? = null
        var mine: CountDownLatch? = null
        while (true) {
            var waitFor: CountDownLatch? = null
            synchronized(lock) {
                val room = rooms[puzzleId]
                if (room != null && !room.closed && !room.closing) return room
                waitFor = opening[puzzleId]
                if (waitFor == null) {
                    val files = dataDir
                    if (puzzles == null && files != null) {
                        puzzles = runCatching { Puzzles.openStore(files) }.getOrNull()
                    }
                    found = puzzles?.get(puzzleId) ?: return null
                    mine = CountDownLatch(1).also { opening[puzzleId] = it }
                }
            }
            val other = waitFor ?: break
            other.await() // another open is making it: take that one
        }
        val puzzle = found!!
        val making = mine!!
        try {
            val room = PuzzleRoom(
                context,
                settings,
                puzzle,
                motifs[puzzleId],
                storePath = storePath,
                speak = speak,
                stopSpeaking = stopSpeaking,
                analysis = analysis,
                coach = coach,
                mirror = mirror,
            )
            synchronized(lock) {
                // A puzzle opened again: the closed room's calls still count.
                rooms[puzzleId]?.let { spent.addAll(it.usages()) }
                rooms[puzzleId] = room
            }
            return room
        } finally {
            synchronized(lock) { opening.remove(puzzleId) }
            making.countDown()
        }
    }

    /** The model-call ledgers of the rooms, open or closed, for the panel's strip. */
    fun usages(): List<Usage> {
        val (all, before) = synchronized(lock) { rooms.values.toList() to spent.toList() }
        return before + all.flatMap { it.usages() }
    }

    fun get(puzzleId: String): PuzzleRoom? {
        val room = synchronized(lock) { rooms[puzzleId] }
        return if (room != null && !room.closed && !room.closing) room else null
    }

    companion object {
        @Volatile private var shared: PuzzleRooms? = null

        /** The app's rooms if any were made, without making them. */
        fun sharedOrNull(): PuzzleRooms? = shared

        /**
         * The app's rooms (`server._rooms_for`): the puzzle store in the app's files, the
         * profile store beside it for the outcomes, the panel's voice for the coach's answers,
         * the motifs the session's recommendations name, and every line mirrored into the
         * panel's log. One per process, so a puzzle worked on outlives the window being rebuilt
         * (a turn of the phone); made on the first call, which is the local server's thread,
         * never the main one.
         */
        fun shared(context: Context): PuzzleRooms = shared ?: synchronized(this) {
            shared ?: run {
                val app = context.applicationContext
                val files = app.filesDir
                PuzzleRooms(
                    app,
                    SettingsStore(app),
                    storePath = File(files, "profile.sqlite"),
                    puzzles = runCatching { Puzzles.openStore(files) }.getOrNull(),
                    dataDir = files,
                    // The panel's voice (`tts.say`, `tts.stop`), whichever the warm-up built;
                    // the words captioned as the panel's are.
                    // Muted, the words are captioned and not said; a room stops only its own.
                    speak = { text -> if (PanelVoice.tts != null) PanelVoice.roomSay(text) },
                    stopSpeaking = { if (PanelVoice.tts != null) PanelVoice.roomStop() },
                    mirror = { chan, level, text, data -> PanelEngine.emit(chan, level, text, data) },
                ).also {
                    it.motifs = PanelEngine.puzzleMotifs
                    shared = it
                }
            }
        }
    }
}

/**
 * A log of records for a page that polls it: `ui/log.py`'s `DebugLog` without the file (a
 * room's log is never written down). Thread-safe; the oldest records fall off past [maxlen].
 */
class DebugLog(private val maxlen: Int = 3000) {
    private val records = ArrayDeque<PanelEngine.Record>()
    private var seq = 0
    private val t0 = SystemClock.elapsedRealtime()
    private val lock = Any()

    val cursor: Int get() = synchronized(lock) { seq }

    fun emit(chan: String, level: String, text: String, data: JSONObject? = null): PanelEngine.Record =
        synchronized(lock) {
            seq += 1
            val t = BigDecimal((SystemClock.elapsedRealtime() - t0) / 1000.0)
                .setScale(3, RoundingMode.HALF_EVEN)
                .toDouble()
            PanelEngine.Record(seq, t, chan, level, text, data ?: JSONObject()).also {
                records.addLast(it)
                while (records.size > maxlen) records.removeFirst()
            }
        }

    fun since(from: Int): JSONArray = synchronized(lock) {
        JSONArray().apply { for (record in records) if (record.seq > from) put(record.json()) }
    }
}
