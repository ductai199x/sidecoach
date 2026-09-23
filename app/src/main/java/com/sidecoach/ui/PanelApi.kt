package com.sidecoach.ui

import android.util.Log
import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/**
 * The control surface the page calls, route for route with `sidecoach/ui/server.py`.
 *
 * Thin over [PanelEngine]. What the phone does not have yet answers honestly -- the voice
 * says it has no voices, the microphone says it is not listening, the profile and the puzzle
 * room say they are not here -- rather than pretending, so the page shows them off instead of
 * hanging on a promise.
 */
class PanelApi(
    private val version: String,
    private val setup: SetupApi,
    private val settings: SettingsApi,
    /** Where the vision keeps what it has learned; `Clear caches` empties it. */
    private val themes: File,
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

            path == "/api/clear_caches" -> ok(clearCaches())

            path == "/api/reset_profile" -> {
                // The player model is not on the phone yet; saying so beats a silent no.
                PanelEngine.emit("user", "info", "profile: there is no player model on the phone yet")
                ok(JSONObject().put("ok", false))
            }

            path == "/api/correct" -> ok(JSONObject().put("ok", false))

            path.startsWith("/api/puzzle/") -> ok(
                JSONObject().apply { put("ok", false); put("error", "no puzzle room on the phone yet") }
            )

            path == "/api/setup/status" -> ok(setup.status())
            path == "/api/setup/start" -> ok(setup.start())
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
     * `maintenance.clear_caches` for the phone: the theme files go, and what was done is
     * said in words on the `user` channel, as the desktop says it. There is no puzzle store
     * here to rebuild.
     */
    private fun clearCaches(): JSONObject {
        val done = JSONArray()
        val files = themes.listFiles { f: File -> f.name.startsWith("theme-") && f.name.endsWith(".bin") }
            ?.sortedBy { it.name }
            .orEmpty()
        for (file in files) {
            if (runCatching { file.delete() }.getOrDefault(false)) {
                val line = "removed ${file.name}"
                done.put(line)
                PanelEngine.emit("user", "warn", "cache: $line")
            }
        }
        if (done.length() == 0) PanelEngine.emit("user", "info", "cache: nothing to clear")
        return JSONObject().apply {
            put("ok", true)
            put("done", done)
        }
    }

    private fun ok(json: JSONObject) = LocalServer.Reply(json.toString())

    companion object {
        private const val TAG = "sidecoach.panel"
    }
}
