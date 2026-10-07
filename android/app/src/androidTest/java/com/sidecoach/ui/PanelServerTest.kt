package com.sidecoach.ui

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * The panel's backend over real HTTP, as `tests/test_panel_server.py` holds the desktop's: the
 * controls reach the engine and the voice, the page and its assets are served -- marked as the
 * phone's -- and every route the shipped page calls is one the server serves. (The poll over
 * HTTP after a game is `PanelEngineTest.theLoopFollowsAGameAndReportsIt`.)
 */
@RunWith(AndroidJUnit4::class)
class PanelServerTest {

    private val app = InstrumentationRegistry.getInstrumentation().targetContext

    private fun ok(reply: Pair<Int, String>): Boolean = reply.first == 200 && JSONObject(reply.second).getBoolean("ok")

    /** `test_controls_and_static_page`. Start and Stop reach what the activity hangs on them. */
    @Test
    fun theControlsAndThePageAreServed() {
        val pressed = ArrayList<String>()
        val hooks = PanelEngine.onStart to PanelEngine.onStop
        PanelEngine.onStart = { pressed.add("start") }
        PanelEngine.onStop = { pressed.add("stop") }
        // The voice the app builds when it opens, as the desktop's engine does when it is made.
        if (PanelVoice.tts == null) PanelVoice.build(app, SettingsStore(app))
        val voice = PanelVoice.voice
        val server = serve(app)
        val port = server.start()
        PanelEngine.commands.clear()
        try {
            val poll = JSONObject(request(port, "GET", "/api/poll").second)
            assertEquals(setOf("state", "lines", "cursor"), poll.keys().asSequence().toSet())
            assertEquals(voice, poll.getJSONObject("state").getString("voice"))
            assertEquals(200 to "{\"ok\":true}", request(port, "POST", "/api/voice", JSONObject().put("voice", "am_adam")))
            assertEquals("am_adam", PanelVoice.voice)
            assertTrue(ok(request(port, "POST", "/api/muted", JSONObject().put("muted", true))))
            assertTrue(PanelVoice.muted)
            assertTrue(ok(request(port, "POST", "/api/ask", JSONObject().put("text", "why not d5"))))
            for (path in listOf("/api/flip", "/api/analyze", "/api/hint", "/api/start", "/api/stop")) {
                assertTrue(path, ok(request(port, "POST", path)))
            }
            assertEquals(listOf("start", "stop"), pressed)

            val (status, index) = request(port, "GET", "/")
            assertEquals(200, status)
            assertTrue(index, "<title>Sidecoach</title>" in index && "_app/" in index)
            // The page is told it is the phone's: its drawer offers Debug > Dump here.
            assertTrue(index, "<html lang=\"en\" data-app=\"phone\"" in index)
            // The single-page frontend: its routes all serve the page, its assets come with it.
            for (path in listOf("/puzzle/abc", "/settings", "/pieces/wK.svg")) {
                assertEquals(path, 200, request(port, "GET", path).first)
            }
            assertTrue("data-app=\"phone\"" in request(port, "GET", "/settings").second)
        } finally {
            server.stop()
            PanelEngine.onStart = hooks.first
            PanelEngine.onStop = hooks.second
            PanelVoice.setMuted(false)
            PanelVoice.setVoice(voice)
            PanelEngine.commands.clear()
        }
    }

    /**
     * Every `/api/...` the shipped page calls -- read out of its scripts, a puzzle room's
     * actions with them -- is a route the server serves (the page is synced from the desktop's
     * build, and a call the phone does not answer shows nothing but a 404 in the log).
     */
    @Test
    fun everyRouteThePageCallsIsServed() {
        val scripts = StringBuilder()
        fun walk(dir: String) {
            for (name in app.assets.list(dir).orEmpty()) {
                val path = "$dir/$name"
                if (name.endsWith(".js")) {
                    scripts.append(app.assets.open(path).bufferedReader().use { it.readText() }).append('\n')
                } else if ('.' !in name) {
                    walk(path)
                }
            }
        }
        walk("web/_app")
        val page = scripts.toString()
        val calls = Regex("(?<=[`\"'}])/api/[a-z_]+(?:/[a-z_]+)*").findAll(page).map { it.value }.toSortedSet()
        assertTrue("the page's calls were not found: $calls", calls.size >= 25 && "/api/dump" in calls)
        // A room's routes are built on `/api/puzzle/<id>` or `/api/review/<key>`: its actions follow
        // the base, up to the next route the page builds (the bundle names both bases alike).
        fun served(path: String) = listOf("GET", "POST", "PUT", "DELETE").any { PanelApi.serves(it, path) }
        val missing = ArrayList<String>()
        for (room in listOf("puzzle", "review")) {
            val base = Regex("(\\w+)=`/api/$room/\\$\\{[^`]*\\}`").find(page)
            assertTrue("the page's $room routes were not found", base != null && "/api/$room" in calls)
            val name = Regex.escape(base!!.groupValues[1])
            val next = page.indexOf("`/api/", base.range.last).let { if (it < 0) page.length else it }
            val actions = Regex("`\\$\\{$name\\}/([a-z_]+)").findAll(page.substring(base.range.last, minOf(next, base.range.last + 2000)))
                .map { it.groupValues[1] }.toSortedSet()
            assertTrue("the $room rooms' actions were not found: $actions", actions.size >= 5)
            missing += actions.map { "/api/$room/abc/$it" }.filterNot { served(it) }
        }
        missing += (calls - "/api/puzzle" - "/api/review").filterNot { served(it) }
        assertEquals("the page calls what the phone does not serve", emptyList<String>(), missing)
    }

    companion object {
        /** The local server the page talks to, with the app's own API behind it. */
        fun serve(app: Context): LocalServer = LocalServer(
            app.assets,
            api = PanelApi(
                "test",
                SetupApi.shared(app),
                SettingsApi(SettingsStore(app)),
                files = app.filesDir,
                rooms = { PuzzleRooms.shared(app) },
                dumps = File(app.cacheDir, "dumps"),
            ),
        )

        /** One request: its status and its body as text. */
        fun request(port: Int, method: String, path: String, json: JSONObject? = null): Pair<Int, String> {
            val connection = URL("http://127.0.0.1:$port$path").openConnection() as HttpURLConnection
            connection.requestMethod = method
            if (json != null) {
                connection.doOutput = true
                connection.setRequestProperty("Content-Type", "application/json")
                connection.outputStream.use { it.write(json.toString().toByteArray()) }
            }
            val status = connection.responseCode
            val text = (if (status < 400) connection.inputStream else connection.errorStream)
                ?.bufferedReader()?.use { it.readText() } ?: ""
            connection.disconnect()
            return status to text
        }
    }
}
