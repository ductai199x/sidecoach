package com.sidecoach.ui

import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.events.Analysis
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
 * channel the terminal filters by. Anything the phone does not have yet -- the voice, the
 * microphone, the profile, the puzzle room -- is reported as absent rather than faked, and
 * the page's own code shows it as off.
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
    private val records = ArrayDeque<Record>()
    private var cursor = 0
    private val state = LinkedHashMap<String, Any?>()

    /** The positions before each ply, for the verdict lines; the labels for the move list. */
    private val fenBefore = HashMap<Int, String>()
    private val labels = LinkedHashMap<Int, String>()

    /** What Start and Stop reach: the activity owns the projection consent. */
    var onStart: (() -> Unit)? = null
    var onStop: (() -> Unit)? = null

    /** What an ask reaches: the coach, when one is running. */
    var onAsk: ((String) -> Unit)? = null

    /** Flip and End game, when the session can take them. */
    var onFlip: (() -> Unit)? = null
    var onEnd: ((String) -> Unit)? = null

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
            // The voice is not on the phone yet; the page reads these and shows it off.
            state["voice"] = ""
            state["voices"] = emptyList<String>()
            state["muted"] = true
            state["listen"] = "off"
            state["ptt"] = false
            state["ptt_key"] = ""
            state["heard"] = null
            state["dictation"] = mapOf("text" to "", "live" to false, "sends_at" to null)
            state["voice_detail"] = "no voice on the phone yet"
            state["stt_detail"] = "no speech recognition on the phone yet"
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

    fun emit(chan: String, level: String, text: String, data: JSONObject? = null) {
        synchronized(lock) {
            cursor += 1
            records.addLast(
                Record(cursor, System.currentTimeMillis() / 1000.0, chan, level, text, data)
            )
            while (records.size > KEPT) records.removeFirst()
        }
    }

    fun set(vararg fields: Pair<String, Any?>) {
        synchronized(lock) { for ((key, value) in fields) state[key] = value }
    }

    fun running(on: Boolean) {
        set("running" to on, "status" to if (on) "watching" else "idle")
        emit("sys", "info", if (on) "watching the screen" else "stopped watching")
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
                "fps" to round1(fps),
                "lock" to lock,
                "stage" to screen.name.lowercase(),
                "pipeline_ms" to listOf(pipelineMs.first, pipelineMs.second),
                "rss_mb" to rssMb,
                "coach_pending" to coachPending,
            ),
        )
    }

    /** The game's own events, as `panel._dispatch_game` writes them down. */
    fun onGameEvent(event: Event) {
        when (event) {
            is GameStarted -> {
                fenBefore.clear()
                labels.clear()
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

            is Takeback -> emit(
                "game",
                "warn",
                "takeback ${event.plies} ply",
                JSONObject().put("fen", event.fen),
            )

            is Resync -> emit(
                "game",
                "warn",
                "resync: ${event.reason}",
                JSONObject().put("fen", event.fen),
            )

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

    /** A line from the coach that needed no model and no game: it is said as itself. */
    fun answered(text: String, trigger: String = "user_ask") {
        emit(
            "coach",
            "info",
            "$trigger — $text",
            JSONObject().apply {
                put("full", text)
                put("trigger", trigger)
                put("ply", JSONObject.NULL)
                put("voice", "text")
                put("annotations", JSONArray())
            },
        )
    }

    fun said(message: CoachMessage, problems: List<String>, attempts: Int) {
        val marks = JSONArray()
        for (mark in message.annotations) {
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
        val trigger = message.trigger.slug
        emit(
            "coach",
            if (problems.isEmpty()) "info" else "warn",
            "$trigger — " + elide(message.text, 80),
            JSONObject().apply {
                put("full", message.text)
                put("trigger", trigger)
                put("disclosure", message.level.value)
                put("ply", message.ply ?: JSONObject.NULL)
                put("attempts", attempts)
                put("problems", JSONArray(problems))
                put("fallback", problems.isNotEmpty())
                put("voice", "text")
                put("fen", message.fen ?: JSONObject.NULL)
                put("annotations", marks)
            },
        )
    }

    // ---- the odds and ends ---------------------------------------------------------------

    private fun elide(text: String, limit: Int): String =
        if (text.length <= limit) text else text.take(limit - 1) + "…"

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
