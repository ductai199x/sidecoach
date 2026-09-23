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
 * Thin over [PanelEngine]. What the phone does not have yet answers honestly -- the voice
 * says it has no voices, the microphone says it is not listening -- rather than pretending,
 * so the page shows them off instead of hanging on a promise.
 */
class PanelApi(
    private val version: String,
    private val setup: SetupApi,
    private val settings: SettingsApi,
    /** The app's files: the profile `Reset profile` empties, the caches `Clear caches` removes. */
    private val files: File,
    /** The puzzle rooms (`server._rooms_for`), made on the first call that needs them. */
    private val rooms: () -> PuzzleRooms,
) : LocalServer.Api {

    override fun handle(
        method: String,
        path: String,
        query: Map<String, String>,
        body: String,
    ): LocalServer.Reply? {
        val json = if (body.isBlank()) JSONObject() else runCatching { JSONObject(body) }
            .getOrElse { JSONObject() }
        return when {
            path == "/api/health" -> ok(
                JSONObject().apply {
                    put("ok", true)
                    put("version", version)
                    put("pid", android.os.Process.myPid())
                    put("running", PanelEngine.poll(Int.MAX_VALUE).getJSONObject("state").optBoolean("running"))
                    put("warm", PanelWarm.status())
                }
            )

            path == "/api/poll" -> ok(PanelEngine.poll(query["since"]?.toIntOrNull() ?: 0))

            path == "/api/start" -> {
                PanelEngine.onStart?.invoke()
                ok(done())
            }

            path == "/api/stop" -> {
                PanelEngine.onStop?.invoke()
                ok(done())
            }

            path == "/api/hint" -> ask("hint")
            path == "/api/move" -> ask("what should I play")
            path == "/api/line" -> ask("show me the line")
            path == "/api/ask" -> ask(json.optString("text"))

            path == "/api/flip" -> {
                PanelEngine.onFlip?.invoke()
                ok(done())
            }

            path == "/api/end" -> {
                PanelEngine.onEnd?.invoke(json.optString("outcome"))
                ok(done())
            }

            path == "/api/analyze" -> {
                PanelEngine.onAnalyze?.invoke()
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

            path == "/api/dump" -> {
                val lines = PanelEngine.all()
                PanelEngine.emit("sys", "info", "${lines.size} records in the log")
                ok(JSONObject().apply { put("ok", true); put("path", JSONObject.NULL) })
            }

            // The panel's own board cannot be on the phone's screen behind the page; the
            // app simply does not read while its own window is in front.
            path == "/api/exclude" -> ok(done())

            path == "/api/muted" -> {
                PanelEngine.set("muted" to json.optBoolean("muted", true))
                ok(done())
            }

            path == "/api/voice" -> ok(JSONObject().put("ok", false))
            path == "/api/say" -> ok(JSONObject().put("ok", false))
            path == "/api/hear_test" -> ok(
                JSONObject().apply { put("ok", false); put("on", false) }
            )

            path == "/api/listen" -> {
                // No speech recognition on the phone yet: the mode stays off, and the page
                // reads that back from the state.
                PanelEngine.set("listen" to "off")
                ok(done())
            }

            path == "/api/ptt" -> ok(done())

            path == "/api/dictation/cancel" ->
                ok(JSONObject().apply { put("ok", true); put("text", "") })

            path == "/api/record" -> ok(
                JSONObject().apply {
                    put("ok", false)
                    put("on", false)
                    put("running", false)
                    put("folder", JSONObject.NULL)
                }
            )

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

            else -> {
                Log.i(TAG, "no route for $method $path")
                null
            }
        }
    }

    private fun ask(text: String): LocalServer.Reply {
        if (text.isBlank()) return ok(done())
        val coach = PanelEngine.onAsk
        if (coach == null) {
            // Nothing is watching, so there is no game to answer about: said now rather
            // than kept for a Start minutes later, about another game.
            PanelEngine.heard(text)
            PanelEngine.answered(
                "Press Start first: I can only answer about a game I am watching.",
            )
            return ok(done())
        }
        coach.invoke(text)
        return ok(done())
    }

    private fun done(): JSONObject = JSONObject().put("ok", true)

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
            "targets" -> {
                if (!json.has("square")) return unprocessable()
                val room = rooms.get(id)
                ok(
                    JSONObject()
                        .put("ok", room != null)
                        .put("targets", JSONArray(room?.targets(json.getString("square")) ?: emptyList<String>()))
                )
            }
            "move" -> if (json.has("uci")) roomCommand(rooms, id, "move", json.getString("uci")) else unprocessable()
            "retract" -> roomCommand(rooms, id, "retract")
            "giveup" -> roomCommand(rooms, id, "giveup")
            "ask" -> if (json.has("text")) roomCommand(rooms, id, "ask", json.getString("text")) else unprocessable()
            "voice" -> if (json.has("on")) roomCommand(rooms, id, "voice", json.getBoolean("on")) else unprocessable()
            "close" -> roomCommand(rooms, id, "close")
            else -> null
        }
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
    }
}
