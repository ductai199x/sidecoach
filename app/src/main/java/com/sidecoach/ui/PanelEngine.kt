package com.sidecoach.ui

import android.os.SystemClock
import com.sidecoach.events.ProfileEvidence
import com.sidecoach.events.ProfileSnapshot
import com.sidecoach.events.PuzzleOutcome
import com.sidecoach.events.Recommendation
import com.sidecoach.events.GameReview
import com.sidecoach.profile.Causes
import com.sidecoach.profile.Profiler
import java.io.File
import com.sidecoach.Maintenance
import com.sidecoach.chess.Board
import com.sidecoach.coach.CoachRunner
import com.sidecoach.coach.Spoken
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.events.Analysis
import com.sidecoach.events.Annotation
import com.sidecoach.events.BoardUnread
import com.sidecoach.events.ClockObservation
import com.sidecoach.events.CoachMessage
import com.sidecoach.events.Event
import com.sidecoach.events.GameEnded
import com.sidecoach.events.GameStarted
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.Resync
import com.sidecoach.events.ScreenState
import com.sidecoach.events.ScreenStateChanged
import com.sidecoach.events.Takeback
import com.sidecoach.events.Unexplained
import com.sidecoach.analysis.sanLine
import com.sidecoach.state.GamePipeline
import com.sidecoach.state.Readings
import com.sidecoach.vision.Pieces
import com.sidecoach.voice.Py
import com.sidecoach.voice.VoiceLog
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfInt
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.math.BigDecimal
import java.math.RoundingMode
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.LinkedBlockingQueue

/**
 * What the panel's page polls: the state of the session and the log since its cursor.
 * `sidecoach/ui/panel.py`'s `PanelEngine`, as much of it as the phone has.
 *
 * The page is the desktop's own, so the shape of what it is given is the desktop's:
 * `{"state": {...}, "lines": [...], "cursor": n}`, the state's keys as `PanelState` in
 * `desktop/src/lib/api.ts` names them, and one log record per thing that happened, on the
 * channel the terminal filters by. What the phone does not have -- a recording, a GPU -- is
 * reported as absent rather than faked, and the page's own code shows it as off.
 *
 * One engine for the app, since the capture is a foreground service and the page is a
 * window onto it. The session's loop itself is [com.sidecoach.vision.GameWatcher], on the
 * vision's thread; what it learns is written down here.
 */
/** What the person is told when a board stays unread ([BoardUnread]), by why: `panel.BOARD_UNREAD`. */
private val BOARD_UNREAD = mapOf(
    "unlearned" to "I can't read this board yet: I haven't learned these pieces on it. Show me the " +
        "starting position with this board and these pieces once -- a new game, or an " +
        "analysis board at the start -- and I'll know them from then on, puzzles included.",
    "orientation" to "I can see the pieces, but not which side is at the bottom: the board's " +
        "coordinates seem to be off. Turn them on in the site's board settings, or press " +
        "Flip to tell me.",
)

object PanelEngine {

    /**
     * How a game ends when the board cannot say so (a resignation, a flag, an agreed draw):
     * outcome -> (result from the user's side, reason). "forget" is not an outcome but a
     * reset. `panel.OUTCOMES`.
     */
    val OUTCOMES = listOf("resigned", "won", "draw", "forget")

    /**
     * Unreadable cells (unknown or covered) beyond which a frame's reading is called
     * degraded, and how many frames in a row it must stay that way before it is worth a line.
     */
    const val DEGRADED_CELLS = 6
    const val DEGRADED_FRAMES = 5

    /** How often the status strip is brought up to date: `panel.HEALTH_EVERY_S`. */
    const val HEALTH_EVERY_S = 1.0

    /**
     * A tick of the loop that takes longer than this is logged with where the time went: the
     * capture, the recorder, the vision, or the rest (engine, profile, coach). A live game
     * froze for 323 s once (2026-09-19) with nothing in the log to say which.
     */
    const val STALL_WARN_S = 2.0

    /** One line of the panel's log, as `ui/log.py` records it. */
    class Record(
        val seq: Int,
        val t: Double,
        val chan: String,
        val level: String,
        val text: String,
        val data: JSONObject,
    ) {
        fun json(): JSONObject = JSONObject().apply {
            put("seq", seq)
            put("t", t)
            put("chan", chan)
            put("level", level)
            put("text", text)
            put("data", data)
        }
    }

    /** The desktop keeps three thousand records and a dump writes them all. */
    private const val KEPT = 3000

    private val lock = Any()

    /**
     * When the session began. `ui/log.py` keeps a `time.monotonic()` of its own and stamps
     * each record with the seconds since; the page prints that as mm:ss.s, so an epoch here
     * would be both wrong and too wide for the column.
     */
    private val began = SystemClock.elapsedRealtime()

    private val records = ArrayDeque<Record>()
    private var cursor = 0
    private val state = LinkedHashMap<String, Any?>()

    /** The positions before each ply, for the verdict lines; the labels for the move list. */
    private val fenBefore = HashMap<Int, String>()
    private val labels = LinkedHashMap<Int, String>()

    /** The game's last ply as the moves came, so a takeback knows which plies it withdrew. */
    private var lastPly = 0

    /** Consecutive frames read as degraded (`panel._bad_run`). */
    private var badRun = 0

    /** What Start and Stop reach: the activity owns the projection consent. */
    var onStart: (() -> Unit)? = null
    var onStop: (() -> Unit)? = null

    /**
     * The control actions for the session's loop (`panel._commands`): ask, flip, analyse,
     * end. Queued here and drained by the loop where `panel._drain_commands` drains them, so
     * nothing touches the pipeline or the coach from two threads at once. With no session
     * running a flip, a re-read or an end waits for the next Start, as on the desktop.
     */
    val commands = LinkedBlockingQueue<Pair<String, Any?>>()

    /**
     * The last frames the pipeline saw, in memory only (`panel.frames`): what a Dump writes
     * out. Nothing of a session reaches the disk otherwise.
     */
    val frames = FrameRing()

    init {
        reset()
        // `_VoiceLogHandler`: the voice's warnings (a failed load, a failed line) to the log.
        VoiceLog.handler = { level, message -> emit("voice", level, message) }
    }

    private fun reset() {
        synchronized(lock) {
            state.clear()
            state["running"] = false
            state["fen"] = null
            state["moves"] = emptyList<String>()
            state["fens"] = emptyList<String>()
            state["ucis"] = emptyList<String>()
            state["labels"] = emptyMap<String, String>()
            state["last_move"] = null
            state["view_flip"] = false
            state["game_id"] = null
            state["user_side"] = null
            state["bottom"] = null
            state["locked"] = false
            state["following"] = false
            state["ply"] = 0
            state["status"] = "idle"
            // The voice: filled in by `PanelVoice.build` when the warm-up makes it.
            state["voice"] = ""
            state["voices"] = emptyList<String>()
            state["muted"] = false
            state["recording_dir"] = null
            state["record_wanted"] = false
            // What the engine is asked for (`self.coach`, `self.analysis`): both, on the phone.
            state["coach_on"] = true
            state["analysis_on"] = true
            state["voice_detail"] = "the voice is not built yet"
            state["stt_detail"] = "parakeet-tdt_ctc-110m, cpu (not loaded yet)"
            // Whether the page's listening button has anything behind it. True while the
            // models are still being built, so it does not appear partway through a run; the
            // warm-up replaces it with the recogniser's own answer.
            state["stt_ready"] = true
            state["listen"] = "off"
            state["ptt"] = false
            state["ptt_key"] = ""
            state["heard"] = null
            state["dictation"] = mapOf("text" to "", "live" to false, "sends_at" to null)
            state["clock"] = null
            state["health"] = emptyMap<String, Any?>()
            state["session_dir"] = null
            state["review"] = null
            state["profile"] = null
        }
    }

    // ---- what the page asks for --------------------------------------------------------

    fun poll(since: Int): JSONObject {
        val lines = JSONArray()
        val snapshot: JSONObject
        val at: Int
        synchronized(lock) {
            for (record in records) if (record.seq > since) lines.put(record.json())
            snapshot = jsonOf(state)
            at = cursor
        }
        return JSONObject().apply {
            put("state", snapshot)
            put("lines", lines)
            put("cursor", at)
        }
    }

    /** Every record the engine holds, for `POST /api/dump`. */
    fun all(): List<Record> = synchronized(lock) { records.toList() }

    /** Whether a session is running (`panel.running`). */
    val running: Boolean get() = synchronized(lock) { state["running"] == true }

    /**
     * Queue a control action for the loop (ask, flip, analyse, end). `panel.command`. With no
     * session running there is no loop to take an ask: it is answered on the spot rather than
     * kept for a Start minutes later, about another game.
     */
    fun command(name: String, payload: Any? = null) {
        if (!running && (name == "ask" || name == "ask_spoken")) {
            heard(payload.toString(), spoken = name == "ask_spoken")
            emit(
                "coach",
                "info",
                "user_ask · not watching — Press Start first: I can only answer about a " +
                    "game I am watching.",
                JSONObject().apply {
                    put("full", "Press Start first: I can only answer about a game I am watching.")
                    put("trigger", "user_ask")
                    put("ply", JSONObject.NULL)
                    put("voice", "text")
                },
            )
            return
        }
        commands.put(name to payload)
    }

    /** An ask, typed or [spoken]: `command("ask" | "ask_spoken")`. */
    fun ask(text: String, spoken: Boolean = false) = command(if (spoken) "ask_spoken" else "ask", text)

    /**
     * End the game on the user's word: they resigned or flagged, the opponent did, a draw was
     * agreed -- or forget the game altogether and wait for the next start. `panel.end_game`.
     */
    fun endGame(outcome: String) {
        if (outcome !in OUTCOMES) {
            throw IllegalArgumentException(
                "outcome must be one of " + OUTCOMES.joinToString(", ", "(", ")") { "'$it'" },
            )
        }
        command("end", outcome)
    }

    /** The PGN result and the reason for an outcome, from the user's side. `outcome_result`. */
    fun outcomeResult(outcome: String, userSide: String?): Pair<String, String> {
        val userWhite = userSide != "black"
        return when (outcome) {
            "resigned" -> (if (userWhite) "0-1" else "1-0") to "user resigned or lost on time"
            "won" -> (if (userWhite) "1-0" else "0-1") to "opponent resigned or lost on time"
            "draw" -> "1/2-1/2" to "draw agreed"
            else -> "*" to "abandoned by the user"
        }
    }

    // ---- what the session tells it -----------------------------------------------------

    /** Seconds since the session began, rounded as Python's `round(x, 3)` rounds. */
    private fun sinceStart(): Double =
        BigDecimal((SystemClock.elapsedRealtime() - began) / 1000.0)
            .setScale(3, RoundingMode.HALF_EVEN)
            .toDouble()

    fun emit(chan: String, level: String, text: String, data: JSONObject? = null) {
        synchronized(lock) {
            cursor += 1
            // A record with nothing underneath carries an empty object, as `LogRecord.data`.
            records.addLast(
                Record(cursor, sinceStart(), chan, level, text, data ?: JSONObject())
            )
            while (records.size > KEPT) records.removeFirst()
        }
    }

    /** The position on the screen as the page shows it (`self._state.get("fen")`), or null. */
    fun fen(): String? = synchronized(lock) { state["fen"] as? String }

    /** The side the page is showing as the player's, as `panel._state["user_side"]`. */
    fun userSide(): String? = synchronized(lock) { state["user_side"] as? String }

    /**
     * The game is forgotten: `_end`'s "forget" branch, which drops the positions it held for
     * the verdict lines and waits for the next start position.
     */
    fun forgotten() {
        synchronized(lock) { fenBefore.clear() }
        set("game_id" to null, "user_side" to null, "following" to false)
    }

    fun set(vararg fields: Pair<String, Any?>) {
        synchronized(lock) { for ((key, value) in fields) state[key] = value }
    }

    /** The game's moves (SAN) and the engine's label by ply, as the move list shows them. */
    @Suppress("UNCHECKED_CAST")
    fun moveList(): Pair<List<String>, Map<String, String>> = synchronized(lock) {
        ((state["moves"] as? List<String>).orEmpty()) to ((state["labels"] as? Map<String, String>).orEmpty())
    }

    /**
     * A session begins or is over: `panel._run`'s first and last lines. The phone saves no
     * session folder, as the desktop app's (`--no-persist`) does not.
     */
    fun running(on: Boolean) {
        if (on) {
            set("running" to true, "status" to "starting", "session_dir" to null)
            emit("sys", "info", "engine started (nothing saved)")
        } else {
            set("running" to false, "status" to "stopped", "recording_dir" to null)
            emit("sys", "info", "engine stopped")
        }
    }

    /**
     * The status strip: `panel._health`, what `watch` prints every few seconds, kept current
     * instead -- brought up to date once a second ([HEALTH_EVERY_S]) from the loop's ticks.
     * [ticks] ran in the last [elapsed] seconds, each [pipelineMs] long; [ts] is the last
     * frame's time on the loop's clock. The phone keeps no recording.
     */
    fun health(
        pipe: GamePipeline,
        coaching: CoachRunner?,
        ticks: Int,
        pipelineMs: List<Double>,
        elapsed: Double,
        ts: Double,
    ) {
        val timings = pipelineMs.sorted().ifEmpty { listOf(0.0) }
        val p50 = timings[timings.size / 2]
        val p95 = timings[minOf(timings.size - 1, (timings.size * 0.95).toInt())]
        val board = pipe.board
        val lock = when {
            board == null -> "no board"
            pipe.tracker.verified -> "locked, fit " + format2(board.fit)
            else -> "lock in doubt"
        }
        val pending = coaching?.coach?.speaker?.pending ?: 0
        set(
            "health" to mapOf(
                // The session's model calls: tokens, and dollars when the prices are known.
                "llm" to PanelWarm.usage(coaching),
                "latency" to PanelVoice.latency(),
                "rss_mb" to residentMb(),
                "tts_device" to if (PanelVoice.tts?.loaded == true) "cpu" else null,
                "stt_device" to PanelListen.sttDevice(),
                "mic_dropped" to PanelListen.micDropped(),
                "fps" to if (elapsed != 0.0) round1(ticks / elapsed) else 0.0,
                "lock" to lock,
                "stage" to if (pipe.model != null) "following" else "waiting for a start",
                "pipeline_ms" to listOf(roundInt(p50), roundInt(p95)),
                // A live capture's lag: from the frame to now, on the loop's clock.
                "lag_ms" to roundInt((SystemClock.elapsedRealtime() / 1000.0 - ts) * 1000.0),
                "recording" to "off",
                "coach_pending" to pending,
            ),
        )
    }

    /** This process's resident memory in MB, as psutil gives the desktop's: /proc/self/statm. */
    private fun residentMb(): Long? = runCatching {
        val pages = File("/proc/self/statm").readText().trim().split(" ")[1].toLong()
        Math.round(pages * 4096 / 1e6)
    }.getOrNull()

    /**
     * Warn once when many cells cannot be read for a while (an overlay, a dialog), and say
     * once when the reading is clean again. A single frame's blip -- a piece being dragged
     * across the board -- is not worth a line, hence the run of frames required.
     *
     * A picked-up rook or queen puts a legal-move dot on a dozen squares, each too small to be
     * a piece and read as unknown; that is the site showing where it may go, not a degraded
     * reading, so those cells are not counted. `panel._check_degraded`.
     */
    fun checkDegraded(readings: Readings?, degraded: Boolean): Boolean {
        if (readings == null) return degraded
        val bad = readings.count { reading ->
            reading.covered || (
                reading.kind == Pieces.Kind.UNKNOWN &&
                    (reading.hover || reading.fgFraction >= Pieces.PIECE_MIN_FRACTION)
                )
        }
        badRun = if (bad >= DEGRADED_CELLS) badRun + 1 else 0
        if (badRun >= DEGRADED_FRAMES && !degraded) {
            emit(
                "vision",
                "warn",
                "reading degraded: $bad cells unreadable (an overlay or a dialog?)",
                JSONObject().put("unreadable", bad),
            )
            return true
        }
        if (badRun == 0 && degraded) {
            emit("vision", "info", "reading clean again")
            return false
        }
        return degraded
    }

    /** The game's own events, as `panel._dispatch_game` writes them down. */
    fun onGameEvent(event: Event) {
        when (event) {
            is GameStarted -> {
                synchronized(lock) {
                    fenBefore.clear()
                    labels.clear()
                }
                lastPly = 0
                val what = when (event.kind) {
                    "game" -> "new game"
                    "puzzle" -> "puzzle on the panel's board"
                    else -> "position joined (a puzzle, an analysis board, or a game joined late)"
                }
                val side = event.userSide.name.lowercase()
                emit(
                    "game",
                    "info",
                    "$what ${event.gameId}, you play $side",
                    JSONObject().apply {
                        put("game_id", event.gameId)
                        put("start_fen", event.startFen)
                        put("kind", event.kind)
                    },
                )
                set("game_id" to event.gameId, "user_side" to side, "kind" to event.kind)
            }

            is MovePlayed -> {
                val before = Board(event.fenBefore)
                val number = before.fullmoveNumber
                val dots = if (before.turn == WHITE) "." else "..."
                val who = if (event.byUser) "you" else "opponent"
                val think = event.thinkMs?.let { ", " + format1(it / 1000.0) + "s" } ?: ""
                synchronized(lock) { fenBefore[event.ply] = event.fenBefore }
                lastPly = event.ply
                PanelVoice.noteMove(event.ply)
                // The move list itself comes from the game's moves each tick
                // (`GameWatcher.refreshState`, as `panel._refresh_state`), so a takeback or an
                // adopted position leaves nothing of the old moves behind.
                emit(
                    "game",
                    "info",
                    "$number$dots ${event.san}  ($who$think)",
                    JSONObject().apply {
                        put("ply", event.ply)
                        put("uci", event.uci)
                        put("by_user", event.byUser)
                        put("think_ms", event.thinkMs ?: JSONObject.NULL)
                        put("fen_after", event.fenAfter)
                    },
                )
            }

            is Takeback -> {
                emit(
                    "game",
                    "warn",
                    "takeback ${event.plies} ply",
                    JSONObject().put("fen", event.fen),
                )
                // The engine lets go of the withdrawn moves at once, and their badges leave the
                // move list: a replacement played at the same ply showed the withdrawn move's.
                lastPly = maxOf(0, lastPly - event.plies)
                synchronized(lock) {
                    for (gone in labels.keys.filter { it > lastPly }) labels.remove(gone)
                    for (gone in fenBefore.keys.filter { it > lastPly }) fenBefore.remove(gone)
                }
                set("labels" to labelState())
            }

            is Resync -> {
                emit(
                    "game",
                    "warn",
                    "resync: ${event.reason}",
                    JSONObject().put("fen", event.fen),
                )
                // The numbering starts again from the adopted position: the old plies' badges
                // would sit on the new moves.
                synchronized(lock) {
                    labels.clear()
                    fenBefore.clear()
                }
                lastPly = 0
                set("labels" to emptyMap<String, String>())
            }

            // A note: cells the game has not explained for a while. The game is kept as it
            // stands; nothing downstream starts again.
            is Unexplained -> emit("vision", "warn", "board unexplained (${event.reason}); the game is kept")

            // A board up and nothing read from it: said in the coach's pane, where the person
            // looks, as "Press Start first" is -- and, the phone's own, in the caption box,
            // over the app the board is in, where the person is.
            is BoardUnread -> {
                val text = BOARD_UNREAD.getValue(event.reason)
                emit(
                    "coach",
                    "info",
                    "board · ${event.reason} — $text",
                    JSONObject().apply {
                        put("full", text)
                        put("trigger", "board")
                        put("ply", JSONObject.NULL)
                        put("voice", "text")
                    },
                )
                Captions.show(text)
            }

            is GameEnded -> {
                val text = if (event.result == "*") {
                    "game over (${event.reason}); the result was not read -- End game… can set it"
                } else {
                    "game over ${event.result} (${event.reason})"
                }
                emit(
                    "game",
                    "info",
                    text,
                    JSONObject().apply {
                        put("result", event.result)
                        put("pgn", event.pgn ?: JSONObject.NULL)
                    },
                )
            }

            is ClockObservation -> set(
                "clock" to mapOf(
                    "white_ms" to event.whiteMs,
                    "black_ms" to event.blackMs,
                    "running" to event.running?.name?.lowercase(),
                )
            )

            is ScreenStateChanged -> {
                if (event.state == ScreenState.COVERED) {
                    emit(
                        "game",
                        "info",
                        "the board is covered by a dialog: game over? The game stays open " +
                            "until a new one starts or End game… says how it ended; the coach waits",
                    )
                }
                emit("vision", "debug", "screen: ${event.state.name.lowercase()}")
            }

            else -> Unit
        }
    }

    /** The move list's badges as the page reads them: ply (as text) -> label. */
    private fun labelState(): Map<String, String> =
        synchronized(lock) { labels.entries.associate { e -> e.key.toString() to e.value } }

    /**
     * The engine's word, as `panel._dispatch_engine` writes it down: each move's assessment,
     * and the position signals that drive the coach -- the decisive moment, the only move, the
     * opening when it changes. [opening] is the last one named; the one named now comes back.
     */
    fun onEngineEvents(events: List<Event>, opening: String?): String? {
        var named = opening
        for (event in events) {
            when (event) {
                is MoveAssessment -> {
                    val fen = synchronized(lock) { fenBefore.remove(event.ply) }
                    val board = fen?.let { Board(it) }
                    val san = if (board != null) board.san(Move.fromUci(event.uci)) else event.uci
                    val severity = event.severity.name.lowercase()
                    val parts = ArrayList<String>()
                    parts.add("ply ${event.ply} $san: loss " + format2(event.expectedLoss))
                    if (severity != "none") {
                        parts.add(severity)
                        val better = if (board != null) sanLine(board, event.best, limit = 4) else ""
                        if (better.isNotEmpty()) parts.add("better $better")
                    }
                    val tags = event.tags.map { it.name }.distinct()
                    event.label?.let {
                        synchronized(lock) { labels[event.ply] = it.name.lowercase() }
                        set("labels" to labelState())
                    }
                    // "quietMove" is the tagger's "nothing special"; it stays in the data only.
                    val shown = tags.filter { it != "quietMove" }
                    if (shown.isNotEmpty()) {
                        parts.add(shown.take(3).joinToString(", ") + if (shown.size > 3) " …" else "")
                    }
                    emit(
                        "engine",
                        "info",
                        parts.joinToString(" · "),
                        JSONObject().apply {
                            put("ply", event.ply)
                            put("loss", round3(event.expectedLoss))
                            put("severity", severity)
                            put("label", event.label?.name?.lowercase() ?: JSONObject.NULL)
                            put("best", JSONArray(if (fen != null) sansOf(fen, event.best) else event.best))
                            put("tags", JSONArray(tags))
                            put("depth", event.depth)
                        },
                    )
                }

                is Analysis -> {
                    val top = event.lines.firstOrNull()
                    val name = event.opening
                    if (!name.isNullOrEmpty() && name != named) {
                        named = name
                        emit("engine", "info", "opening: $name", JSONObject().put("ply", event.ply))
                    }
                    val score = if (top != null) format2(top.expectedScore) else "?"
                    val depth = top?.depth ?: 0
                    val threats = sansOf(event.fen, event.threats, byOpponent = true)
                    val pv = if (top != null) sansOf(event.fen, top.moves) else emptyList()
                    val data = JSONObject().apply {
                        put("ply", event.ply)
                        put("decisive", event.decisive)
                        put("only_move", event.onlyMove)
                        put("threats", JSONArray(threats))
                        put("pv", JSONArray(pv))
                        put("score", top?.expectedScore ?: JSONObject.NULL)
                        put("depth", depth)
                    }
                    // Only the moments that drive the coach are worth a line at info; almost
                    // every position carries some threat, so plain threats stay at debug.
                    val signals = ArrayList<String>()
                    if (event.decisive) signals.add("decisive moment")
                    if (event.onlyMove) signals.add("only move")
                    val tail = "  (score $score, d$depth)"
                    val threatText = if (threats.isNotEmpty()) "; threats " + threats.take(3).joinToString(", ") else ""
                    if (event.provisional) {
                        // Measured early, the full depth to follow: its line is the news.
                        emit("engine", "debug", "early ply ${event.ply}" + threatText + tail, data)
                    } else if (signals.isNotEmpty()) {
                        emit("engine", "info", "ply ${event.ply}: " + signals.joinToString(", ") + threatText + tail, data)
                    } else {
                        emit("engine", "debug", "analysed ply ${event.ply}" + threatText + tail, data)
                    }
                }

                else -> Unit
            }
        }
        return named
    }

    /**
     * UCI moves from [fen] as SAN, stopping at the first that does not parse. `panel._sans_of`.
     *
     * A threat is what the side NOT to move would play if it could, so it is read after a
     * null move; naming it on the wrong side's turn makes python-chess call a quiet pawn push
     * a capture ("dxd4"). The moves are read one after another, as the desktop reads them.
     */
    fun sansOf(fen: String, ucis: List<String>, byOpponent: Boolean = false): List<String> {
        val board = try {
            Board(fen)
        } catch (_: Exception) {
            return ucis.toList()
        }
        if (byOpponent) board.push(Move.NULL)
        val out = ArrayList<String>()
        for (uci in ucis) {
            val move = Move.fromUciOrNull(uci) ?: break
            // python-chess's `san` asserts a piece on the square the move leaves.
            if (!move.isNull && board.pieceTypeAt(move.from) == 0) break
            try {
                out.add(board.san(move))
                board.push(move)
            } catch (_: Exception) {
                break
            }
        }
        return out
    }

    /** What the player typed or said, and what the coach answered. */
    fun heard(text: String, spoken: Boolean = false) {
        emit("user", "info", "you: $text", JSONObject().put("spoken", spoken))
    }

    /** A message's marks as the page reads them: `Annotation.model_dump` each. */
    fun marks(annotations: List<Annotation>): JSONArray {
        val marks = JSONArray()
        for (mark in annotations) {
            marks.put(
                JSONObject().apply {
                    put("kind", mark.kind)
                    put("squares", JSONArray(mark.squares))
                    put("color", mark.color)
                    put("opacity", mark.opacity)
                    put("label", mark.label ?: JSONObject.NULL)
                }
            )
        }
        return marks
    }

    /**
     * A coach message on the page, as `_coach_meta` logs it: [meta] (the trigger, the
     * disclosure, the ply, the attempts, what the checks and the voice did), the text, and
     * underneath everything the page draws it from.
     */
    fun said(message: CoachMessage, spoken: Spoken, meta: String, action: String) {
        emit(
            "coach",
            if (spoken.problems.isEmpty()) "info" else "warn",
            "$meta — " + elide(message.text, 80),
            JSONObject().apply {
                put("full", message.text)
                put("trigger", message.trigger.slug)
                put("disclosure", spoken.level.value)
                put("ply", message.ply ?: JSONObject.NULL)
                put("attempts", spoken.attempts)
                put("problems", JSONArray(spoken.problems))
                put("leaks", JSONArray(spoken.leaks))
                put("fallback", spoken.problems.isNotEmpty())
                put("voice", action)
                put("fen", message.fen ?: JSONObject.NULL)
                put("annotations", marks(message.annotations))
            },
        )
    }

    /** The ply the board is on, as the voice reads it (`self._state.get("ply")`). */
    fun plyNow(): Int = synchronized(lock) { (state["ply"] as? Number)?.toInt() ?: 0 }

    // ---- the dump ----------------------------------------------------------------------------

    /**
     * Write the frames held in memory and the log to `<dumps>/<stamp>/`: the only way a
     * session reaches the disk. Returns the folder. `panel.dump`; the phone's folder is in the
     * app's own files on shared storage, where `adb pull` reaches it.
     */
    fun dump(dumps: File): String {
        val folder = File(dumps, SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date()))
        folder.mkdirs()
        val n = frames.write(folder)
        File(folder, "debug.jsonl").bufferedWriter(Charsets.UTF_8).use { out ->
            for (record in all()) out.write(record.json().toString() + "\n")
        }
        emit("sys", "info", "dumped $n frames and the log to ${folder.path}")
        return folder.path
    }

    // ---- the profile ---------------------------------------------------------------------

    /** The running session's player model (`panel._profiler`); null between sessions. */
    @Volatile var profiler: Profiler? = null

    /** The puzzles recommended last, and the motif each was for. */
    @Volatile var practiceIds: List<String> = emptyList()
    val puzzleMotifs: MutableMap<String, String> = java.util.concurrent.ConcurrentHashMap()

    /** `panel._profile_state`: what the page shows of the player model. */
    fun profileState(snapshot: ProfileSnapshot): Map<String, Any?> = mapOf(
        "games" to snapshot.games,
        "insights" to snapshot.insights,
        "last_session" to snapshot.lastSession,
    )

    /**
     * Feed the profile stage (`panel._profile`); its own events go on to the coach, and the
     * review and the snapshot to the page (the review as a card in the coach's stream, with
     * the insights of the moment). [tick] lets a finished game's review time out waiting for
     * the engine; [flush] reviews it now.
     */
    fun profile(profiler: Profiler, events: List<Event>, tick: Boolean = false, flush: Boolean = false): List<Event> {
        val out = ArrayList<Event>()
        val now = System.currentTimeMillis() / 1000.0
        try {
            for (event in events) out += profiler.onEvent(event, now)
            if (flush && profiler.gameOverUnreviewed) {
                out += profiler.flush(now)
            } else if (tick) {
                out += profiler.tick(now)
            }
        } catch (exc: Exception) {  // a profile bug must not end the session
            emit("sys", "error", "profile error: ${pyRepr(exc)}")
        }
        val snapshot = out.firstOrNull { it is ProfileSnapshot } as ProfileSnapshot?
        for (event in out) {
            when (event) {
                is GameReview -> {
                    val review = reviewJson(event)
                    set("review" to review)
                    val insights = snapshot?.insights ?: emptyList()
                    emit(
                        "coach",
                        "info",
                        "review: " + elide(event.lines.joinToString(" "), 80),
                        JSONObject().put("review", review).put("insights", JSONArray(insights)),
                    )
                }
                is Recommendation -> {
                    val practice = practiceJson(event)
                    practiceIds = event.puzzles.map { it.id }
                    for (puzzle in event.puzzles) puzzleMotifs[puzzle.id] = event.motif
                    set("practice" to practice)
                    emit(
                        "coach",
                        "info",
                        "practice: ${event.motifWords} -- " + event.puzzles.joinToString(", ") { it.id } +
                            " (level ${event.level}); ${event.reason}",
                        JSONObject().put("practice", practice),
                    )
                }
                is PuzzleOutcome -> logOutcome(event)
                is ProfileSnapshot -> set("profile" to profileState(event))
                is ProfileEvidence -> {
                    val motif = if (event.motif != null) " (${event.motif})" else ""
                    emit(
                        "game",
                        "debug",
                        "evidence at ply ${event.ply}: ${event.cause}$motif, ${format2(event.expectedLoss)} in the ${event.phase}",
                    )
                }
                else -> Unit
            }
        }
        return out
    }

    /** What was said, written down in the profile: `panel._record_dialogue`. */
    fun recordDialogue(event: Event) {
        val profiler = profiler ?: return
        try {
            profiler.onEvent(event, System.currentTimeMillis() / 1000.0)
        } catch (exc: Exception) {
            emit("sys", "error", "profile error: ${pyRepr(exc)}")
        }
    }

    /** `panel._log_outcome`. */
    fun logOutcome(event: PuzzleOutcome) {
        emit(
            "game",
            "info",
            "puzzle ${event.puzzleId} ${if (event.solved) "solved" else "failed"}" +
                (if (event.motif != null) " (${event.motif})" else "") + "; puzzle level ${event.level}",
            JSONObject().put("puzzle_id", event.puzzleId).put("solved", event.solved).put("puzzle_level", event.level),
        )
    }

    /**
     * The user's word on why a mistake happened, on the review's card: the profile moves the
     * evidence and the review is redrawn. `panel.correct`.
     */
    fun correct(gameId: String, segment: Int, ply: Int, cause: String?): Boolean {
        if (cause != null && cause !in Causes.CAUSES) throw IllegalArgumentException("cause must be one of ${Causes.CAUSES}")
        val profiler = profiler ?: return false
        val ok = profiler.correct(gameId, segment, ply, cause)
        if (ok) {
            val review = synchronized(lock) { state["review"] as? JSONObject }
            if (review != null && review.optString("game_id") == gameId) {
                // The state's review is replaced, not edited: a poll may be serialising it.
                val fresh = JSONObject(review.toString())
                val points = fresh.optJSONArray("turning_points") ?: JSONArray()
                for (i in 0 until points.length()) {
                    val tp = points.getJSONObject(i)
                    if (tp.optInt("segment", 0) == segment && tp.opt("ply") == ply) tp.put("cause", cause ?: JSONObject.NULL)
                }
                set("review" to fresh, "profile" to profileState(profiler.snapshot()))
            }
            // Python writes a cleared cause as None.
            emit("user", "info", "you corrected the cause of the mistake at ply $ply: ${cause ?: "None"}")
        }
        return ok
    }

    /**
     * Empty the player model (`maintenance.reset_profile`): every table of the file emptied in
     * place, so the running session's store stays open on it and simply finds nothing. The
     * running profiler, if any, drops its game in progress, and the page shows a fresh, empty
     * profile. Returns the path.
     */
    fun resetProfile(files: File): String {
        val path = Maintenance.resetProfile(files)
        val profiler = profiler
        if (profiler != null) {
            profiler.forget()
            set("profile" to profileState(profiler.snapshot()), "review" to null, "practice" to null)
        }
        practiceIds = emptyList()
        puzzleMotifs.clear()
        emit("user", "warn", "profile reset: $path is empty")
        return path.path
    }

    /**
     * Remove the caches (`maintenance.clear_caches`): the theme files, and the puzzle store
     * rebuilt from its CSVs when the running session holds it open.
     */
    fun clearCaches(files: File): List<String> {
        val done = Maintenance.clearCaches(files, profiler?.puzzles)
        for (line in done) emit("user", "warn", "cache: $line")
        if (done.isEmpty()) emit("user", "info", "cache: nothing to clear")
        return done
    }

    /** A review as the page reads it: `GameReview.model_dump`. */
    private fun reviewJson(r: GameReview): JSONObject = JSONObject().apply {
        put("game_id", r.gameId)
        put("user_side", r.userSide.name.lowercase())
        put("result", r.result)
        put("reason", r.reason ?: JSONObject.NULL)
        put("opening", r.opening ?: JSONObject.NULL)
        put("moves", r.moves)
        put("accuracy", r.accuracy ?: JSONObject.NULL)
        put("tallies", JSONObject().apply { for ((k, v) in r.tallies) put(k, v) })
        put("turning_points", JSONArray(r.turningPoints.map { tp ->
            JSONObject().apply {
                put("ply", tp.ply)
                put("san", tp.san)
                put("label", tp.label)
                put("expected_loss", tp.expectedLoss)
                put("segment", tp.segment)
                put("best_san", tp.bestSan ?: JSONObject.NULL)
                put("cause", tp.cause ?: JSONObject.NULL)
                put("motif", tp.motif ?: JSONObject.NULL)
                put("phase", tp.phase)
                put("fen_before", tp.fenBefore ?: JSONObject.NULL)
            }
        }))
        put("phases", JSONArray(r.phases.map {
            JSONObject().put("phase", it.phase).put("moves", it.moves).put("expected_loss", it.expectedLoss).put("mistakes", it.mistakes)
        }))
        put("causes", JSONObject().apply { for ((k, v) in r.causes) put(k, v) })
        put("lines", JSONArray(r.lines))
    }

    /** A recommendation as the page reads it: `Recommendation.model_dump`. */
    private fun practiceJson(r: Recommendation): JSONObject = JSONObject().apply {
        put("game_id", r.gameId)
        put("motif", r.motif)
        put("motif_words", r.motifWords)
        put("reason", r.reason)
        put("level", r.level)
        put("puzzles", JSONArray(r.puzzles.map {
            JSONObject().put("id", it.id).put("url", it.url).put("rating", it.rating).put("themes", JSONArray(it.themes))
        }))
    }

    // ---- the odds and ends ---------------------------------------------------------------

    /**
     * The first [limit] characters of a line, with an ellipsis when it was cut: `log.elide`,
     * whitespace run together first -- Python's whitespace, counted in code points as Python
     * counts a string.
     */
    fun elide(text: String, limit: Int = 80): String {
        val flat = Py.split(text).joinToString(" ")
        if (flat.codePointCount(0, flat.length) <= limit) return flat
        val cut = flat.substring(0, flat.offsetByCodePoints(0, limit - 1))
        return cut.trimEnd { Py.isSpace(it) } + "…"
    }

    /**
     * An exception as Python's `{exc!r}` writes one in the log: its class and its message,
     * `RuntimeError('boom')`. The class is the phone's own.
     */
    fun pyRepr(exc: Throwable): String {
        val message = exc.message ?: return "${exc::class.java.simpleName}()"
        return "${exc::class.java.simpleName}(${pyStr(message)})"
    }

    /** A string as Python's `repr` writes it. */
    fun pyStr(text: String): String {
        val quote = if ('\'' in text && '"' !in text) '"' else '\''
        val out = StringBuilder().append(quote)
        for (ch in text) {
            when {
                ch == '\\' -> out.append("\\\\")
                ch == quote -> out.append('\\').append(ch)
                ch == '\n' -> out.append("\\n")
                ch == '\r' -> out.append("\\r")
                ch == '\t' -> out.append("\\t")
                ch.code < 0x20 || ch.code == 0x7f -> out.append("\\x%02x".format(ch.code))
                else -> out.append(ch)
            }
        }
        return out.append(quote).toString()
    }

    /** `f"{x:.2f}"`: the exact binary value, rounded half to even. */
    fun format2(value: Double): String =
        BigDecimal(value).setScale(2, RoundingMode.HALF_EVEN).toPlainString()

    /** `f"{x:.1f}"`. */
    fun format1(value: Double): String =
        BigDecimal(value).setScale(1, RoundingMode.HALF_EVEN).toPlainString()

    private fun round1(value: Double): Double =
        BigDecimal(value).setScale(1, RoundingMode.HALF_EVEN).toDouble()

    private fun round3(value: Double): Double =
        BigDecimal(value).setScale(3, RoundingMode.HALF_EVEN).toDouble()

    /** `round(x)`: an int, half to even. */
    private fun roundInt(value: Double): Long =
        BigDecimal(value).setScale(0, RoundingMode.HALF_EVEN).toLong()

    /** A map of plain Kotlin values as the page's JSON. */
    private fun jsonOf(map: Map<String, Any?>): JSONObject {
        val out = JSONObject()
        for ((key, value) in map) out.put(key, wrap(value))
        return out
    }

    private fun wrap(value: Any?): Any = when (value) {
        null -> JSONObject.NULL
        is Map<*, *> -> JSONObject().apply {
            for ((k, v) in value) put(k.toString(), wrap(v))
        }
        is List<*> -> JSONArray().apply { for (one in value) put(wrap(one)) }
        is Pair<*, *> -> JSONArray().apply { put(wrap(value.first)); put(wrap(value.second)) }
        else -> value
    }
}

/**
 * The last frames the pipeline saw, in memory only, as JPEGs at half scale: `panel.FrameRing`.
 * [maxlen] frames (a minute at a changing board) stay a few megabytes on a phone's smaller
 * frames. Encoding costs a few milliseconds per changed frame and nothing on a still one.
 */
class FrameRing(private val maxlen: Int = 150, private val quality: Int = 85) {
    private val frames = ArrayDeque<Pair<Double, ByteArray>>()
    private val lock = Any()

    fun push(ts: Double, frame: Mat) {
        val bytes = try {
            val small = Mat()
            val buf = MatOfByte()
            try {
                Imgproc.resize(frame, small, Size((frame.cols() / 2).toDouble(), (frame.rows() / 2).toDouble()))
                if (!Imgcodecs.imencode(".jpg", small, buf, MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, quality))) return
                buf.toArray()
            } finally {
                small.release()
                buf.release()
            }
        } catch (_: Throwable) {  // a frame the ring cannot keep is not an error
            return
        }
        synchronized(lock) {
            frames.addLast(ts to bytes)
            while (frames.size > maxlen) frames.removeFirst()
        }
    }

    val size: Int get() = synchronized(lock) { frames.size }

    /** The frames as files under [folder], named by their timestamp; returns the count. */
    fun write(folder: File): Int {
        val kept = synchronized(lock) { frames.toList() }
        for ((ts, data) in kept) {
            // `f"{ts:014.3f}.jpg"`: zero-padded to fourteen characters.
            val stamp = BigDecimal(ts).setScale(3, RoundingMode.HALF_EVEN).toPlainString().padStart(14, '0')
            File(folder, "$stamp.jpg").writeBytes(data)
        }
        return kept.size
    }
}
