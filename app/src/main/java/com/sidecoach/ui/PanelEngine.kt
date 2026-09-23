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
import com.sidecoach.coach.Spoken
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.events.Analysis
import com.sidecoach.events.Annotation
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
import org.json.JSONArray
import org.json.JSONObject
import java.math.BigDecimal
import java.math.RoundingMode
import java.util.Locale

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
 * window onto it.
 */
object PanelEngine {

    /** One line of the panel's log, as `ui/log.py` records it. */
    class Record(
        val seq: Int,
        val t: Double,
        val chan: String,
        val level: String,
        val text: String,
        val data: JSONObject?,
    ) {
        fun json(): JSONObject = JSONObject().apply {
            put("seq", seq)
            put("t", t)
            put("chan", chan)
            put("level", level)
            put("text", text)
            put("data", data ?: JSONObject.NULL)
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

    /** What Start and Stop reach: the activity owns the projection consent. */
    var onStart: (() -> Unit)? = null
    var onStop: (() -> Unit)? = null

    /** What an ask reaches: the session's loop, when one is running; whether it was spoken. */
    var onAsk: ((String, Boolean) -> Unit)? = null

    /** Flip, End game and Re-read, when the session can take them. */
    var onFlip: (() -> Unit)? = null
    var onEnd: ((String) -> Unit)? = null
    var onAnalyze: (() -> Unit)? = null

    init {
        reset()
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
            state["kind"] = "game"
            state["status"] = "idle"
            // The voice: filled in by `PanelVoice.build` when the warm-up makes it.
            state["voice"] = ""
            state["voices"] = emptyList<String>()
            state["muted"] = false
            state["listen"] = "off"
            state["ptt"] = false
            state["ptt_key"] = ""
            state["heard"] = null
            state["dictation"] = mapOf("text" to "", "live" to false, "sends_at" to null)
            state["voice_detail"] = "the voice is not built yet"
            state["stt_detail"] = "parakeet-tdt_ctc-110m, cpu (not loaded yet)"
            // Whether the page's listening button has anything behind it. True while the
            // models are still being built, so it does not appear partway through a run; the
            // warm-up replaces it with the recogniser's own answer.
            state["stt_ready"] = true
            state["recording_dir"] = null
            state["record_wanted"] = false
            state["coach_on"] = false
            state["analysis_on"] = false
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

    // ---- what the session tells it -----------------------------------------------------

    /** Seconds since the session began, rounded as Python's `round(x, 3)` rounds. */
    private fun sinceStart(): Double =
        BigDecimal((SystemClock.elapsedRealtime() - began) / 1000.0)
            .setScale(3, RoundingMode.HALF_EVEN)
            .toDouble()

    fun emit(chan: String, level: String, text: String, data: JSONObject? = null) {
        synchronized(lock) {
            cursor += 1
            records.addLast(
                Record(cursor, sinceStart(), chan, level, text, data)
            )
            while (records.size > KEPT) records.removeFirst()
        }
    }

    /** The position on the screen as the page shows it (`self._state.get("fen")`), or null. */
    fun fen(): String? = synchronized(lock) { state["fen"] as? String }

    /** The side the page is showing as the player's, as `panel._state["user_side"]`. */
    fun userSide(): String? = synchronized(lock) { state["user_side"] as? String }

    /**
     * The game is forgotten: `_end`'s "forget" branch, which drops what the page held about
     * it and waits for the next start position.
     */
    fun forgotten() {
        synchronized(lock) {
            fenBefore.clear()
            labels.clear()
        }
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
            set("running" to false, "status" to "stopped")
            emit("sys", "info", "engine stopped")
        }
    }

    /** What the vision made of this frame: the lock, the pace, the position. */
    fun watched(
        boardFound: Boolean,
        verified: Boolean,
        fit: Double,
        fen: String?,
        ply: Int,
        userSide: String?,
        bottom: String?,
        screen: ScreenState,
        following: Boolean,
        engineOn: Boolean,
        coachOn: Boolean,
        fps: Double,
        pipelineMs: Pair<Long, Long>,
        coachPending: Int,
        rssMb: Long?,
    ) {
        val lock = when {
            !boardFound -> "no board"
            verified -> "locked, fit " + format2(fit)
            else -> "lock in doubt"
        }
        set(
            "fen" to fen,
            "ply" to ply,
            "user_side" to userSide,
            "bottom" to bottom,
            "locked" to (boardFound && verified),
            "following" to following,
            "analysis_on" to engineOn,
            "coach_on" to coachOn,
            "health" to mapOf(
                // The session's model calls: tokens, and dollars when the prices are known.
                "llm" to PanelWarm.usage(),
                "latency" to PanelVoice.latency(),
                "rss_mb" to (rssMb ?: residentMb()),
                "tts_device" to if (PanelVoice.tts?.loaded == true) "cpu" else null,
                "stt_device" to PanelListen.sttDevice(),
                "mic_dropped" to PanelListen.micDropped(),
                "fps" to round1(fps),
                "lock" to lock,
                "stage" to if (following) "following" else "waiting for a start",
                "pipeline_ms" to listOf(pipelineMs.first, pipelineMs.second),
                "lag_ms" to null,
                "recording" to "off",
                "coach_pending" to coachPending,
            ),
        )
    }

    private var rssAt = 0L
    private var rss: Long? = null

    /**
     * This process's resident memory in MB, as psutil gives the desktop's: /proc/self/statm's
     * resident pages, read at most once a second.
     */
    private fun residentMb(): Long? {
        val now = SystemClock.elapsedRealtime()
        if (now - rssAt < 1000) return rss
        rssAt = now
        rss = runCatching {
            val pages = java.io.File("/proc/self/statm").readText().trim().split(" ")[1].toLong()
            Math.round(pages * 4096 / 1e6)
        }.getOrNull()
        return rss
    }

    /** The game's own events, as `panel._dispatch_game` writes them down. */
    fun onGameEvent(event: Event) {
        when (event) {
            is GameStarted -> {
                fenBefore.clear()
                labels.clear()
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
                set(
                    "game_id" to event.gameId,
                    "user_side" to side,
                    "bottom" to side,
                    "kind" to event.kind,
                    "moves" to emptyList<String>(),
                    "fens" to emptyList<String>(),
                    "ucis" to emptyList<String>(),
                    "labels" to emptyMap<String, String>(),
                    "last_move" to null,
                    "review" to null,
                )
            }

            is MovePlayed -> {
                val before = Board(event.fenBefore)
                val number = before.fullmoveNumber
                val dots = if (before.turn == WHITE) "." else "..."
                val who = if (event.byUser) "you" else "opponent"
                val think = event.thinkMs?.let { ", " + round1(it / 1000.0) + "s" } ?: ""
                fenBefore[event.ply] = event.fenBefore
                lastPly = event.ply
                PanelVoice.noteMove(event.ply)
                synchronized(lock) {
                    @Suppress("UNCHECKED_CAST")
                    state["moves"] = (state["moves"] as List<String>) + event.san
                    @Suppress("UNCHECKED_CAST")
                    state["fens"] = (state["fens"] as List<String>) + event.fenAfter
                    @Suppress("UNCHECKED_CAST")
                    state["ucis"] = (state["ucis"] as List<String>) + event.uci
                    state["last_move"] = event.uci
                }
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
                // The withdrawn moves' badges leave the move list: a replacement played at the
                // same ply showed the withdrawn move's.
                lastPly = maxOf(0, lastPly - event.plies)
                synchronized(lock) {
                    for (gone in labels.keys.filter { it > lastPly }) labels.remove(gone)
                    for (gone in fenBefore.keys.filter { it > lastPly }) fenBefore.remove(gone)
                }
                set("labels" to labels.entries.associate { e -> e.key.toString() to e.value })
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

    /** The engine's word on a move, as `panel._dispatch_engine` writes it down. */
    fun onEngineEvent(event: Event) {
        when (event) {
            is MoveAssessment -> {
                val fen = fenBefore.remove(event.ply)
                val board = fen?.let { Board(it) }
                val move = Move.fromUciOrNull(event.uci)
                val san = if (board != null && move != null) board.san(move) else event.uci
                val parts = ArrayList<String>()
                parts.add("ply ${event.ply} $san: loss " + format2(event.expectedLoss))
                if (event.severity.name.lowercase() != "none") {
                    parts.add(event.severity.name.lowercase())
                    val better = if (board != null) sanLine(board, event.best, limit = 4) else ""
                    if (better.isNotEmpty()) parts.add("better $better")
                }
                val tags = event.tags.map { it.name }.distinct()
                event.label?.let {
                    labels[event.ply] = it.name.lowercase()
                    set("labels" to labels.entries.associate { e -> e.key.toString() to e.value })
                }
                // "quietMove" is the tagger's "nothing special"; it stays in the data only.
                val shown = tags.filter { it != "quietMove" }
                if (shown.isNotEmpty()) parts.add(shown.joinToString(" "))
                emit(
                    "engine",
                    if (event.severity.name.lowercase() == "none") "info" else "warn",
                    parts.joinToString(", "),
                    JSONObject().apply {
                        put("ply", event.ply)
                        put("uci", event.uci)
                        put("loss", event.expectedLoss)
                        put("severity", event.severity.name.lowercase())
                        put("label", event.label?.name?.lowercase() ?: JSONObject.NULL)
                        put("tags", JSONArray(tags))
                    },
                )
            }

            is Analysis -> {
                val top = event.lines.firstOrNull() ?: return
                emit(
                    "engine",
                    "debug",
                    "ply ${event.ply}: " + format2(top.expectedScore) +
                        (if (event.decisive) ", decisive" else "") +
                        (if (event.onlyMove) ", only move" else ""),
                    JSONObject().apply {
                        put("ply", event.ply)
                        put("expected", top.expectedScore)
                        put("decisive", event.decisive)
                        put("only_move", event.onlyMove)
                    },
                )
            }

            else -> Unit
        }
    }

    /** What the player typed or said, and what the coach answered. */
    fun heard(text: String, spoken: Boolean = false) {
        emit("user", "info", "you: $text", JSONObject().put("spoken", spoken))
    }

    /**
     * An ask, typed or [spoken], for the session's loop (`panel.command("ask" | "ask_spoken")`).
     * With no session running there is no loop to take it: it is answered on the spot rather
     * than kept for a Start minutes later, about another game.
     */
    fun ask(text: String, spoken: Boolean = false) {
        val session = onAsk
        if (session != null) {
            session(text, spoken)
            return
        }
        heard(text, spoken)
        emit(
            "coach",
            "info",
            "user_ask · not watching — Press Start first: I can only answer about a game I am watching.",
            JSONObject().apply {
                put("full", "Press Start first: I can only answer about a game I am watching.")
                put("trigger", "user_ask")
                put("ply", JSONObject.NULL)
                put("voice", "text")
            },
        )
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
                put("fallback", spoken.problems.isNotEmpty())
                put("voice", action)
                put("fen", message.fen ?: JSONObject.NULL)
                put("annotations", marks(message.annotations))
            },
        )
    }

    /** The ply the board is on, as the voice reads it (`self._state.get("ply")`). */
    fun plyNow(): Int = synchronized(lock) { (state["ply"] as? Number)?.toInt() ?: 0 }

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
            emit("sys", "error", "profile error: $exc")
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
            emit("sys", "error", "profile error: $exc")
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
            emit("user", "info", "you corrected the cause of the mistake at ply $ply: $cause")
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
     * whitespace run together first.
     */
    fun elide(text: String, limit: Int = 80): String {
        val flat = text.split(Regex("\\s+")).filter { it.isNotEmpty() }.joinToString(" ")
        return if (flat.length <= limit) flat else flat.take(limit - 1).trimEnd() + "…"
    }

    private fun format2(value: Double): String =
        BigDecimal(value).setScale(2, RoundingMode.HALF_EVEN).toPlainString()

    private fun round1(value: Double): Double =
        BigDecimal(value).setScale(1, RoundingMode.HALF_EVEN).toDouble()

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

    @Suppress("unused")
    private fun formatted(value: Double, places: Int): String =
        String.format(Locale.ROOT, "%.${places}f", value)
}
