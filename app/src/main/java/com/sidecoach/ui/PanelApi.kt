package com.sidecoach.ui

import android.net.Uri
import android.util.Log
import com.sidecoach.profile.Puzzles
import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/**
 * The control surface the page calls, route for route with `sidecoach/ui/server.py`.
 *
 * Thin over [PanelEngine]. What the phone does not have -- a recording, a GPU pack -- answers
 * honestly rather than pretending, so the page shows it off instead of hanging on a promise.
 */
class PanelApi(
    private val version: String,
    private val setup: SetupApi,
    private val settings: SettingsApi,
    /** The app's files: the profile `Reset profile` empties, the caches `Clear caches` removes. */
    private val files: File,
    /** The puzzle rooms (`server._rooms_for`), made on the first call that needs them. */
    private val rooms: () -> PuzzleRooms,
    /**
     * Where Debug > Dump writes (`data_dir/dumps` on the desktop): the app's own folder on
     * shared storage, where `adb pull` reaches it.
     */
    private val dumps: File = File(files, "dumps"),
    /** Your games: the library, imports, taking a game out. */
    private val games: GamesApi? = null,
    /** Stats: the player's reviewed games added up. */
    private val stats: StatsApi? = null,
    /** The review rooms (`server.py`'s `reviews`), made on the first call that needs them. */
    private val reviews: (() -> ReviewRooms)? = null,
) : LocalServer.Api {

    override fun handle(
        method: String,
        path: String,
        query: Map<String, String>,
        body: String,
        raw: ByteArray,
    ): LocalServer.Reply? {
        // A route the server does not serve, or a method the route does not take: the local
        // server answers as the desktop's does (404 or 405).
        if (!serves(method, path)) return null
        val json = if (body.isBlank()) JSONObject() else runCatching { JSONObject(body) }
            .getOrElse { JSONObject() }
        return when {
            path == "/api/health" -> ok(
                JSONObject().apply {
                    put("ok", true)
                    put("version", version)
                    put("pid", android.os.Process.myPid())
                    put("running", PanelEngine.running)
                    put("warm", PanelWarm.status())
                }
            )

            path == "/api/poll" -> ok(PanelEngine.poll(query["since"]?.toIntOrNull() ?: 0, files))

            path == "/api/start" -> {
                PanelEngine.onStart?.invoke()
                ok(done())
            }

            path == "/api/stop" -> {
                PanelEngine.onStop?.invoke()
                ok(done())
            }

            path == "/api/hint" -> ask(QUICK.getValue("hint"))
            path == "/api/move" -> ask(QUICK.getValue("move"))
            path == "/api/line" -> ask(QUICK.getValue("line"))
            path == "/api/ask" -> if (json.has("text")) ask(json.optString("text")) else unprocessable()

            // With no session running these wait for the next Start (`panel.command`).
            path == "/api/flip" -> {
                PanelEngine.command("flip")
                ok(done())
            }

            // An outcome the panel does not know is refused (`end_game`'s ValueError).
            path == "/api/end" -> {
                if (!json.has("outcome")) return unprocessable()
                PanelEngine.endGame(json.optString("outcome"))
                ok(done())
            }

            path == "/api/analyze" -> {
                PanelEngine.command("analyze")
                ok(done())
            }

            path == "/api/client_log" -> {
                val where = json.optString("page").ifEmpty { "" }
                val level = json.optString("level").ifEmpty { "error" }
                PanelEngine.emit(
                    "sys",
                    if (level in listOf("info", "warn", "error")) level else "error",
                    "page" + (if (where.isEmpty()) "" else " [$where]") + ": " +
                        json.optString("text").take(2000),
                )
                ok(done())
            }

            // The page's own frame meter. The phone keeps no perf files.
            path == "/api/perf" -> ok(done())

            // The frames the engine keeps in memory, and its log, written out: the one way
            // anything about a live session reaches the disk in the app.
            path == "/api/dump" -> ok(JSONObject().put("ok", true).put("path", PanelEngine.dump(dumps)))

            // The panel's own board cannot be on the phone's screen behind the page; the
            // app simply does not read while its own window is in front.
            path == "/api/exclude" -> ok(done())

            path == "/api/muted" -> {
                if (!json.has("muted")) return unprocessable()
                PanelVoice.mute(settings, json.getBoolean("muted"))  // the next session too
                ok(done())
            }

            path == "/api/voice" -> {
                if (!json.has("voice")) return unprocessable()
                PanelVoice.setVoice(json.getString("voice"))
                ok(done())
            }

            // One line in a voice, for the settings page's Listen button.
            path == "/api/say" -> {
                val voice = if (json.isNull("voice")) null else json.optString("voice")
                val text = if (json.isNull("text")) "" else json.optString("text")
                ok(JSONObject().put("ok", PanelVoice.saySample(voice, text.ifEmpty { PanelVoice.VOICE_SAMPLE })))
            }
            // The settings page's live test of the microphone and speech recognition.
            path == "/api/hear_test" && method == "GET" -> ok(PanelListen.hearStatus())
            path == "/api/hear_test" -> ok(PanelListen.setHearTest(json.optBoolean("on", true)))

            path == "/api/listen" -> {
                if (!json.has("mode")) return unprocessable()
                val wanted = json.getString("mode")
                // Set and kept in one step, as muted is (`PanelVoice.mute`), under the settings'
                // own lock (the settings review, 2026-10-01).
                val mode = synchronized(SettingsApi.SAVING) {
                    PanelListen.setListen(wanted).also { // refuses a mode there is not
                        settings.remember(JSONObject().put("voice", JSONObject().put("listen", wanted))) // the next session too
                    }
                }
                ok(JSONObject().put("ok", true).put("mode", mode))
            }

            path == "/api/ptt" -> {
                if (!json.has("down")) return unprocessable()
                PanelListen.ptt(json.getBoolean("down"))
                ok(done())
            }

            path == "/api/hear" -> hear(raw)

            // The person typed over the dictation: it stays in the box, nothing sends itself.
            path == "/api/dictation/cancel" -> ok(PanelListen.cancelDictation())

            path == "/api/record" -> ok(
                JSONObject().apply {
                    put("ok", false)
                    put("on", false)
                    put("running", false)
                    put("folder", JSONObject.NULL)
                }
            )

            // The desktop shell asks its sidecar to end its session and exit before a quit
            // (desktop 18a4715); the phone's app is ended by Android, never by the page.
            path == "/api/shutdown" -> ok(JSONObject().put("ok", false))

            path == "/api/clear_caches" -> {
                // The rooms let go of the puzzle store first (a file held open cannot go), and
                // open it again after -- rebuilt from the CSVs by the session, or at the open.
                val rooms = rooms()
                rooms.puzzles?.let { runCatching { it.close() } }
                rooms.puzzles = null
                val done = PanelEngine.clearCaches(files)
                rooms.puzzles = runCatching { Puzzles.openStore(files) }.getOrNull()
                ok(JSONObject().put("ok", true).put("done", JSONArray(done)))
            }

            path == "/api/reset_profile" -> ok(
                JSONObject().put("ok", true).put("path", PanelEngine.resetProfile(files))
            )

            path == "/api/correct" -> {
                // `_Correct`: game_id and ply required, segment 0 and cause null by default.
                if (!json.has("game_id") || !json.has("ply")) return unprocessable()
                val cause = if (json.isNull("cause")) null else json.optString("cause")
                ok(
                    JSONObject().put(
                        "ok",
                        PanelEngine.correct(json.getString("game_id"), json.optInt("segment", 0), json.getInt("ply"), cause),
                    )
                )
            }

            path.startsWith("/api/puzzle/") -> puzzle(path.removePrefix("/api/puzzle/"), query, json)

            path == "/api/setup/status" -> ok(setup.status())
            path == "/api/setup/start" -> ok(setup.start(json))
            path == "/api/setup/cancel" -> ok(setup.cancel())
            path == "/api/setup/remove_pack" -> ok(JSONObject().put("ok", false))

            path == "/api/settings" && method == "GET" -> ok(settings.describe())
            path == "/api/settings" -> ok(settings.update(json))

            path == "/api/games" && games != null -> ok(games.list())
            path == "/api/games/import" && games != null ->
                if (json.has("source")) ok(games.import(json)) else unprocessable()
            path == "/api/games/remove" && games != null -> ok(games.remove(json))
            path == "/api/games/restore" && games != null -> ok(games.restore(json))

            path.startsWith("/api/review/") && reviews != null -> review(path.removePrefix("/api/review/"), query, json)

            // Stats: the player's games added up (profile/Stats.kt).
            path == "/api/stats" && stats != null -> ok(stats.stats(query))
            path == "/api/stats/positions" && stats != null -> ok(stats.positions(query))

            // How the boards look (Looks.kt): every page reads it; the pieces a player chose.
            path == "/api/look" -> settings.store.let { store ->
                val (pieces, theme) = store.board()
                ok(Looks.look(pieces, theme, store.files))
            }
            // A player's own sets: a new one from the images chosen, more images for one, a new
            // name, deleted (put aside for its Undo) and back (`server.py`'s /api/pieces/own).
            path == "/api/pieces/own" ->
                json.optJSONArray("files")?.let { ok(touched(Looks.createOwn(settings.store.files, json.optString("name", ""), it))) }
                    ?: unprocessable()
            path.startsWith("/api/pieces/own/") -> ownSet(method, path.removePrefix("/api/pieces/own/"), json)

            // The long jobs the panel's bell shows: the setup's fetch; no background review here.
            path == "/api/jobs" -> ok(setup.jobs())
            path == "/api/jobs/reviews" ->
                if (json.has("paused")) ok(JSONObject().put("ok", false).put("error", "no background review here")) else unprocessable()

            else -> {
                Log.i(TAG, "no route for $method $path")
                null
            }
        }
    }

    /** `PanelApi.ask`: the ask for the session's loop, whatever it says (an empty one too). */
    private fun ask(text: String): LocalServer.Reply {
        PanelEngine.command("ask", text)
        return ok(done())
    }

    /**
     * A 16 kHz mono WAV in the body is treated as something the user said -- the way to
     * exercise the whole hear -> correct -> ask path without a microphone. `server.hear`, with
     * Python's `wave` reading the file: PCM only, and anything but 16 kHz mono 16-bit refused.
     */
    private fun hear(raw: ByteArray): LocalServer.Reply {
        val wav = java.nio.ByteBuffer.wrap(raw).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        fun tag(at: Int) = String(raw, at, 4, Charsets.US_ASCII)
        if (raw.size < 12 || tag(0) != "RIFF" || tag(8) != "WAVE") throw IllegalArgumentException("file does not start with RIFF id")
        var at = 12
        var format = -1
        var channels = 0
        var rate = 0
        var width = 0
        var data: Pair<Int, Int>? = null
        while (at + 8 <= raw.size) {
            val id = tag(at)
            val size = wav.getInt(at + 4)
            val start = at + 8
            // A negative or overlong size walked backwards (-8 looped forever) or off the end.
            if (size < 0) throw IllegalArgumentException("bad chunk size: $size")
            when (id) {
                "fmt " -> {
                    if (size < 16 || start + 16 > raw.size) throw IllegalArgumentException("fmt chunk too short")
                    format = wav.getShort(start).toInt() and 0xffff
                    channels = wav.getShort(start + 2).toInt()
                    rate = wav.getInt(start + 4)
                    width = wav.getShort(start + 14).toInt() / 8
                }
                "data" -> data = start to minOf(size, raw.size - start)
            }
            if (data != null) break
            val next = start.toLong() + size + (size and 1)
            if (next > raw.size) break
            at = next.toInt()
        }
        if (format != 1) throw IllegalArgumentException("unknown format: $format")
        val (start, length) = data ?: throw IllegalArgumentException("data chunk missing")
        if (rate != 16000 || channels != 1 || width != 2) {
            return ok(JSONObject().put("ok", false).put("error", "need 16 kHz mono 16-bit PCM WAV"))
        }
        val audio = FloatArray(length / 2)
        for (i in audio.indices) audio[i] = wav.getShort(start + 2 * i) / 32768.0f
        PanelListen.hear(audio)
        val seconds = java.math.BigDecimal(audio.size / 16000.0).setScale(2, java.math.RoundingMode.HALF_EVEN).toDouble()
        return ok(JSONObject().put("ok", true).put("seconds", seconds))
    }

    private fun done(): JSONObject = JSONObject().put("ok", true)

    /** A change to the player's sets that took raises the settings' counter (`touched`). */
    private fun touched(result: JSONObject): JSONObject {
        if (result.optBoolean("ok")) PanelEngine.settingsTouched()
        return result
    }

    /** `/api/pieces/own/{set_id}` (PUT, DELETE) and its `/images` and `/restore` (POST). */
    private fun ownSet(method: String, rest: String, json: JSONObject): LocalServer.Reply? {
        val files = settings.store.files
        val parts = rest.split('/')
        val id = parts[0]
        return when {
            parts.size == 1 && method == "PUT" ->
                if (json.has("name")) ok(touched(Looks.renameOwn(files, id, json.optString("name")))) else unprocessable()
            parts.size == 1 && method == "DELETE" -> ok(touched(Looks.deleteOwn(files, id)))
            parts.size == 2 && parts[1] == "images" ->
                json.optJSONArray("files")?.let { ok(touched(Looks.addImages(files, id, it))) } ?: unprocessable()
            parts.size == 2 && parts[1] == "restore" -> ok(touched(Looks.restoreOwn(files, id)))
            else -> null
        }
    }

    /**
     * A puzzle on a page of its own (ui/puzzle_room.py): `/api/puzzle/<id>/<action>`, as
     * `server.py` routes them.
     */
    private fun puzzle(rest: String, query: Map<String, String>, json: JSONObject): LocalServer.Reply? {
        val slash = rest.lastIndexOf('/')
        if (slash <= 0) return null
        val id = Uri.decode(rest.substring(0, slash))
        val rooms = rooms()
        return when (rest.substring(slash + 1)) {
            "open" -> {
                val room = rooms.open(id)
                    ?: return ok(JSONObject().put("ok", false).put("error", "no puzzle $id in the store"))
                val following = PanelEngine.practiceIds.filter { it != id }
                ok(
                    JSONObject()
                        .put("ok", true)
                        .put("next", JSONArray(following))
                        .put("state", room.poll(0).getJSONObject("state"))
                )
            }
            "poll" -> {
                val room = rooms.get(id) ?: return ok(JSONObject().put("ok", false).put("closed", true))
                val polled = room.poll(query["since"]?.toIntOrNull() ?: 0)
                val out = JSONObject().put("ok", true)
                for (key in polled.keys()) out.put(key, polled.get(key))
                ok(out)
            }
            // A move, a take-back, a give-up: played at once, answered with the state they leave.
            "move" -> {
                if (!json.has("uci")) return unprocessable()
                val uci = json.getString("uci")
                played(rooms, id) { it.move(uci) }
            }
            "retract" -> played(rooms, id) { it.retract() }
            "giveup" -> played(rooms, id) { it.giveUp() }
            "ask" -> if (json.has("text")) roomCommand(rooms, id, "ask", json.getString("text")) else unprocessable()
            "voice" -> if (json.has("on")) roomCommand(rooms, id, "voice", json.getBoolean("on")) else unprocessable()
            "close" -> roomCommand(rooms, id, "close")
            else -> null
        }
    }

    /**
     * A game reviewed on a page of its own (ui/ReviewRoom.kt): `/api/review/<key>/<action>`, as
     * `server.py` routes them. The card on the move shown is said in the panel's voice and
     * captioned, a new card cutting off the one before; closing the review silences it.
     */
    private fun review(rest: String, query: Map<String, String>, json: JSONObject): LocalServer.Reply? {
        val slash = rest.lastIndexOf('/')
        if (slash <= 0) return null
        val key = rest.substring(0, slash)
        val rooms = reviews?.invoke() ?: return null
        val voice = PanelVoice.tts
        fun notOpen() = ok(JSONObject().put("ok", false).put("error", "the review is not open"))
        fun polled(room: ReviewRoom, since: Int, extra: JSONObject): LocalServer.Reply {
            val polled = room.poll(since)
            for (k in polled.keys()) extra.put(k, polled.get(k))
            return ok(extra)
        }
        return when (rest.substring(slash + 1)) {
            "open" -> {
                // No strength named: the strongest review saved, else the phone's own, Fast --
                // its engine is slower (the plan's depth 14); the desktop's is Standard.
                val room = rooms.open(key, json.optString("strength", "").ifEmpty { null })
                    ?: return ok(JSONObject().put("ok", false).put("error", "no game $key in the library"))
                polled(room, 0, JSONObject().put("ok", true).put("voice", voice != null))
            }
            "poll" -> {
                val room = rooms.get(key) ?: return ok(JSONObject().put("ok", false).put("closed", true))
                polled(room, query["since"]?.toIntOrNull() ?: 0, JSONObject().put("ok", true))
            }
            // The AI coach's words on the key moves: only when the player asks, a call a move.
            "ai" -> {
                val room = rooms.get(key) ?: return notOpen()
                val refused = room.writeWords()
                ok(JSONObject().put("ok", refused == null).put("error", refused ?: JSONObject.NULL))
            }
            "say" -> {
                if (!json.has("text")) return unprocessable()
                if (voice == null) return ok(JSONObject().put("ok", false))
                PanelVoice.roomStop()
                PanelVoice.roomSay(json.getString("text"))
                ok(done())
            }
            // Retry (ui/ReviewRetry.kt): the position a move was played from, a new move graded
            // as the review graded the game's, and the coach's hints there.
            "retry" -> {
                val ply = query["ply"]?.toIntOrNull() ?: return unprocessable()
                val retries = rooms.get(key)?.retries ?: return notOpen()
                try {
                    ok(JSONObject().put("ok", true).apply { retries.position(ply).let { p -> for (k in p.keys()) put(k, p.get(k)) } })
                } catch (exc: IllegalArgumentException) {
                    ok(JSONObject().put("ok", false).put("error", exc.message))
                }
            }
            "try" -> {
                if (!json.has("ply") || !json.has("uci")) return unprocessable()
                val retries = rooms.get(key)?.retries ?: return notOpen()
                try {
                    ok(JSONObject().put("ok", true).apply { retries.attempt(json.getInt("ply"), json.getString("uci")).let { p -> for (k in p.keys()) put(k, p.get(k)) } })
                } catch (exc: RuntimeException) { // `(ValueError, RuntimeError)`: a move not legal, no Stockfish
                    ok(JSONObject().put("ok", false).put("error", exc.message ?: exc.toString()))
                }
            }
            "hint" -> {
                if (!json.has("ply") || !json.has("level")) return unprocessable()
                val retries = rooms.get(key)?.retries ?: return notOpen()
                try {
                    ok(JSONObject().put("ok", true).apply { retries.hint(json.getInt("ply"), json.getInt("level")).let { p -> for (k in p.keys()) put(k, p.get(k)) } })
                } catch (exc: IllegalArgumentException) {
                    ok(JSONObject().put("ok", false).put("error", exc.message))
                }
            }
            "hush" -> {
                if (voice != null) PanelVoice.roomStop()
                ok(JSONObject().put("ok", voice != null))
            }
            // Analysis (ReviewAnalysis.kt): any move from any position, the engine's lines for the
            // board, the explorer of the player's games, the game as a PGN.
            "position" -> {
                if (!json.has("fen")) return unprocessable()
                try {
                    ok(JSONObject().put("ok", true).apply { ReviewAnalysis.position(json.getString("fen")).let { p -> for (k in p.keys()) put(k, p.get(k)) } })
                } catch (exc: IllegalArgumentException) {
                    ok(JSONObject().put("ok", false).put("error", exc.message))
                }
            }
            "move" -> {
                if (!json.has("fen") || !json.has("uci")) return unprocessable()
                try {
                    ok(JSONObject().put("ok", true).apply { ReviewAnalysis.move(json.getString("fen"), json.getString("uci")).let { p -> for (k in p.keys()) put(k, p.get(k)) } })
                } catch (exc: IllegalArgumentException) {
                    ok(JSONObject().put("ok", false).put("error", exc.message))
                }
            }
            "lines" -> {
                if (!json.has("fen")) return unprocessable()
                val room = rooms.get(key) ?: return notOpen()
                try {
                    ok(JSONObject().put("ok", true).apply { room.lines(json.getString("fen")).let { p -> for (k in p.keys()) put(k, p.get(k)) } })
                } catch (exc: RuntimeException) { // `(ValueError, RuntimeError)`: a position not read, no Stockfish
                    ok(JSONObject().put("ok", false).put("error", exc.message ?: exc.toString()))
                }
            }
            "explore" -> {
                if (!json.has("fen")) return unprocessable()
                val room = rooms.get(key) ?: return notOpen()
                val moves = json.optJSONArray("moves")?.let { m -> (0 until m.length()).map { m.getString(it) } } ?: emptyList()
                try {
                    ok(JSONObject().put("ok", true).apply { room.explore(json.getString("fen"), moves).let { p -> for (k in p.keys()) put(k, p.get(k)) } })
                } catch (exc: IllegalArgumentException) {
                    ok(JSONObject().put("ok", false).put("error", exc.message))
                }
            }
            "pgn" -> {
                val room = rooms.get(key) ?: return notOpen()
                try {
                    val got = room.pgn(json.optJSONArray("tree") ?: JSONArray(), json.optBoolean("annotated", true), json.optBoolean("save", false), json.optBoolean("copy", false))
                    ok(JSONObject().put("ok", true).apply { for (k in got.keys()) put(k, got.get(k)) })
                } catch (exc: Exception) { // `(ValueError, OSError)`: no game in the PGN, the file not kept
                    ok(JSONObject().put("ok", false).put("error", exc.message ?: exc.toString()))
                }
            }
            "close" -> {
                val room = rooms.get(key)
                room?.close()
                if (voice != null) PanelVoice.roomStop()
                ok(JSONObject().put("ok", room != null))
            }
            else -> null
        }
    }

    /** `{"ok": True, "state": ...}` of what [play] leaves the room in, or `{"ok": False}` without one. */
    private fun played(rooms: PuzzleRooms, id: String, play: (PuzzleRoom) -> JSONObject): LocalServer.Reply {
        val room = rooms.get(id) ?: return ok(JSONObject().put("ok", false))
        return ok(JSONObject().put("ok", true).put("state", play(room)))
    }

    /** `server._room_command`. */
    private fun roomCommand(rooms: PuzzleRooms, id: String, name: String, payload: Any? = null): LocalServer.Reply {
        val room = rooms.get(id) ?: return ok(JSONObject().put("ok", false))
        room.command(name, payload)
        return ok(JSONObject().put("ok", true))
    }

    /** What FastAPI answers a body that does not fit the route's model. */
    private fun unprocessable() = LocalServer.Reply(
        422,
        "application/json; charset=utf-8",
        JSONObject().put("detail", "the body does not fit the request").toString().toByteArray(),
    )

    private fun ok(json: JSONObject) = LocalServer.Reply(json.toString())

    companion object {
        private const val TAG = "sidecoach.panel"

        /**
         * Every route `server.build_app` serves, by method: what the page may call. Anything
         * else is answered as the desktop's server answers it (see `LocalServer`).
         */
        val ROUTES: Map<String, Set<String>> = mapOf(
            "GET" to setOf(
                "/api/health", "/api/poll", "/api/hear_test", "/api/settings", "/api/setup/status",
                "/api/games", "/api/look", "/api/jobs", "/api/stats", "/api/stats/positions",
            ),
            "POST" to setOf(
                "/api/perf", "/api/client_log", "/api/dump", "/api/start", "/api/stop", "/api/hint",
                "/api/move", "/api/line", "/api/flip", "/api/analyze", "/api/ask", "/api/hear_test",
                "/api/say", "/api/voice", "/api/muted", "/api/dictation/cancel", "/api/record", "/api/shutdown",
                "/api/exclude", "/api/end", "/api/setup/start", "/api/setup/cancel",
                "/api/setup/remove_pack", "/api/reset_profile", "/api/clear_caches", "/api/correct",
                "/api/listen", "/api/ptt", "/api/hear", "/api/games/import", "/api/pieces/own",
                "/api/jobs/reviews", "/api/games/remove", "/api/games/restore",
            ),
            "PUT" to setOf("/api/settings"),
        )

        /** An own set's routes, `/api/pieces/own/{set_id}[/<action>]`, by method. */
        val OWN_SET_ROUTES: Map<String, Set<String>> = mapOf(
            "PUT" to setOf(""),
            "DELETE" to setOf(""),
            "POST" to setOf("images", "restore"),
        )

        /** A puzzle room's routes, `/api/puzzle/{puzzle_id}/<action>`, by method. */
        val PUZZLE_ROUTES: Map<String, Set<String>> = mapOf(
            "GET" to setOf("poll"),
            "POST" to setOf("open", "move", "retract", "giveup", "ask", "voice", "close"),
        )

        /** A review room's routes, `/api/review/{key}/<action>`, by method. */
        val REVIEW_ROUTES: Map<String, Set<String>> = mapOf(
            "GET" to setOf("poll", "retry"),
            "POST" to setOf("open", "ai", "say", "try", "hint", "hush", "close", "position", "move", "lines", "explore", "pgn"),
        )

        /** Whether the server serves [method] [path]. */
        fun serves(method: String, path: String): Boolean {
            if (path in ROUTES[method].orEmpty()) return true
            if (path.startsWith("/api/pieces/own/")) {
                // `{set_id}` is one segment of the path, as Starlette matches it.
                val parts = path.removePrefix("/api/pieces/own/").split('/')
                if (parts[0].isEmpty() || parts.size > 2) return false
                return parts.getOrElse(1) { "" } in OWN_SET_ROUTES[method].orEmpty()
            }
            if (path.startsWith("/api/review/")) {
                val parts = path.removePrefix("/api/review/").split('/')
                return parts.size == 2 && parts[0].isNotEmpty() && parts[1] in REVIEW_ROUTES[method].orEmpty()
            }
            if (!path.startsWith("/api/puzzle/")) return false
            // `{puzzle_id}` is one segment of the path, as Starlette matches it.
            val parts = path.removePrefix("/api/puzzle/").split('/')
            return parts.size == 2 && parts[0].isNotEmpty() && parts[1] in PUZZLE_ROUTES[method].orEmpty()
        }

        /**
         * The panel's three quick pills and what each asks the coach (`panel.hint`, `.move`,
         * `.line`); the caption box has them too.
         */
        val QUICK = linkedMapOf(
            "hint" to "hint",
            "move" to "what should I play",
            "line" to "show me the line",
        )
    }
}
