package com.sidecoach.ui

import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.analysis.Engine
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.Completion
import com.sidecoach.llm.LLMProvider
import com.sidecoach.profile.Library
import com.sidecoach.profile.Store
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/**
 * A review room on the phone end to end (ui/ReviewRoom.kt), as the page drives it: a game of the
 * library reviewed on real engines with its progress, summed up and saved; opened again, the
 * saved review read at once; the AI coach's words written by a scripted model and kept with it;
 * words the app was closed in the middle of carried over, the answers already paid for never
 * asked again. Skipped without the engine's network (tools/push_net.ps1).
 */
@RunWith(AndroidJUnit4::class)
class ReviewRoomTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    private val pgn = "[White \"nosegazm\"]\n[Black \"Someone\"]\n[Result \"1-0\"]\n[TimeControl \"180\"]\n\n" +
        "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0\n"

    /** A model answering every key move alike, recording which it was asked about. */
    private class Scripted : LLMProvider {
        override val name = "scripted"
        val asked: MutableSet<Int> = ConcurrentHashMap.newKeySet()

        override fun healthcheck(): Pair<Boolean, String> = true to "scripted"

        override fun chat(system: String, messages: List<ChatMessage>, maxTokens: Int, temperature: Double, thinking: Boolean): Completion {
            val ply = PLY.find(messages[0].content)?.groupValues?.get(1)?.toInt() ?: -1
            asked.add(ply)
            return Completion(
                text = "### why\nThis gives the opponent time.\n### lesson\nBefore you move, look at what your opponent can do next, and answer it first.",
                stats = mapOf("prompt_tokens" to 1000.0, "completion_tokens" to 300.0),
            )
        }
    }

    private fun maker(): EngineMaker? {
        val binary = Engine.binaryIn(context.applicationInfo.nativeLibraryDir)
        val net = context.filesDir.listFiles { f: File -> f.name.endsWith(".nnue") }?.firstOrNull()
        if (!binary.exists() || net == null) return null
        return EngineMaker { threads, multipv -> Engine(binary, net, threads = threads, hashMb = 64, multipv = multipv) }
    }

    private fun until(room: ReviewRoom, seconds: Int, what: String, done: (JSONObject) -> Boolean): JSONObject {
        val end = SystemClock.elapsedRealtime() + seconds * 1000L
        while (true) {
            val state = room.state()
            if (done(state)) return state
            assertTrue("$what: ${state.toString().take(400)}", SystemClock.elapsedRealtime() < end && state.getString("status") != "failed")
            Thread.sleep(200)
        }
    }

    @Test
    fun aGameIsReviewedSavedOpenedAgainAndItsWordsKept() {
        val maker = maker()
        assumeTrue("the engine or its network is missing", maker != null)
        val files = File(context.cacheDir, "review-room").apply {
            deleteRecursively()
            mkdirs()
        }
        val storePath = File(files, "profile.sqlite")
        val row = Library.fromPgn(pgn, listOf("nosegazm")).single()
        Store(storePath).apply { addLibraryGame(row) }.close()
        val settings = SettingsStore(context)
        val model = Scripted()
        fun room() = ReviewRoom(files, settings, row, storePath, "fast", maker!!, model = { model to null })

        // While it is analysed, the game's positions for the board to play out as the progress goes.
        val playing = ReviewRoom.playback(row.pgn)
        assertEquals(8, playing.length())
        assertTrue(playing.getJSONObject(0).isNull("uci") && playing.getJSONObject(1).getString("uci") == "e2e4")

        // Reviewed: its progress to the whole game, then the summary and every move.
        val first = room()
        val opening = first.state()
        if (opening.getString("status") != "ready") assertEquals(playing.toString(), opening.getJSONArray("playback").toString())
        var state = until(first, 180, "reviewed") { it.getString("status") == "ready" }
        assertTrue("the review shown, the playback gone", state.isNull("playback"))
        assertEquals(8, state.getJSONObject("progress").getInt("done"))
        assertEquals(8, state.getJSONObject("progress").getInt("total"))
        assertEquals(7, state.getJSONArray("moves").length())
        assertFalse(state.isNull("summary"))
        assertEquals("fast", state.getString("strength"))
        val keyMoves = state.getJSONObject("ai").getInt("moves")
        assertTrue("an estimate for the scripted model is null: its prices are not known", state.getJSONObject("ai").isNull("estimate"))

        // The words, written in the background and kept with the review.
        assertEquals(null, first.writeWords())
        state = until(first, 60, "words") { it.getJSONObject("ai").getString("status") == "ready" }
        assertEquals(keyMoves, model.asked.size)
        val saved = Store(storePath).let { s -> s.review(row.key, "fast").also { s.close() } }!!
        val words = JSONObject(saved.words!!)
        assertFalse("saved whole, not partial", words.optBoolean("partial", false))
        assertTrue(words.has("texts") && words.has("model"))
        first.close()
        first.join()

        // Opened again: the saved review and its words at once, nothing searched.
        val second = room()
        state = until(second, 10, "loaded") { it.getString("status") == "ready" }
        assertEquals("ready", state.getJSONObject("ai").getString("status"))
        assertTrue(state.getJSONObject("ai").getJSONObject("words").has("links"))
        second.close()
        second.join()

        // The app closed in the middle of the words: the answer paid for is kept, never asked again.
        val plies = state.getJSONArray("moves").let { m -> (0 until m.length()).map { m.getJSONObject(it).getInt("ply") } }
        val paid = model.asked.minOrNull() ?: plies.first()
        Store(storePath).apply {
            setReviewWords(
                row.key,
                "fast",
                JSONObject().put("partial", true).put("model", "scripted").put("cost", 0.001)
                    .put("answers", JSONObject().put("$paid", "### why\nKept.\n### lesson\nLook first.")).toString(),
            )
        }.close()
        model.asked.clear()
        val third = room()
        state = until(third, 10, "loaded part way") { it.getString("status") == "ready" }
        assertEquals("none", state.getJSONObject("ai").getString("status"))
        assertEquals(1, state.getJSONObject("ai").getInt("answered"))
        assertEquals(null, third.writeWords())
        until(third, 60, "the rest of the words") { it.getJSONObject("ai").getString("status") == "ready" }
        assertFalse("the paid answer was asked again", paid in model.asked)
        assertEquals(keyMoves - 1, model.asked.size)
        third.close()
        third.join()
    }

    companion object {
        private val PLY = Regex("\"key move\": \\{\"ply\": (\\d+)")
    }
}
