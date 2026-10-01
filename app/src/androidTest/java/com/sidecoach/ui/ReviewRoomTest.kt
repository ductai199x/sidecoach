package com.sidecoach.ui

import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.analysis.Engine
import com.sidecoach.analysis.GameReview
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.Completion
import com.sidecoach.llm.LLMProvider
import com.sidecoach.profile.Library
import com.sidecoach.profile.LibraryRow
import com.sidecoach.profile.Store
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

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
        fun room(copier: ((String) -> Unit)? = null) = ReviewRoom(files, settings, row, storePath, "fast", maker!!, model = { model to null }, copier = copier)

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
        val copied = mutableListOf<String>()
        val second = room(copier = { copied.add(it) })
        state = until(second, 10, "loaded") { it.getString("status") == "ready" }
        assertEquals("ready", state.getJSONObject("ai").getString("status"))
        assertTrue(state.getJSONObject("ai").getJSONObject("words").has("links"))
        // Copy PGN: the app puts the annotated game on the clipboard (the web view's refuses) and says so.
        val pgnOut = second.pgn(JSONArray(), annotated = true, save = false, copy = true)
        assertTrue(pgnOut.getBoolean("copied"))
        assertEquals(listOf(pgnOut.getString("pgn")), copied)
        assertTrue(copied.single(), "[Annotator \"Sidecoach\"]" in copied.single() && "4. Qxf7#" in copied.single())
        assertFalse("only a copy asked for, nothing kept", pgnOut.has("saved"))
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

    /**
     * Tai, 2026-09-30: a watched game reviewed, then brought in, lost its accuracy -- the import
     * stood for it unreviewed. Sidecoach read every move: the import takes the review whole on
     * pairing, no engine, with its own headers and clocks, and Stats counts it once. A move
     * misread: the import is reviewed as itself, the positions before the misread move not
     * searched again. The desktop's twin tests in tests/test_review_server.py.
     */
    @Test
    fun aWatchedGamesReviewGoesWithTheImportThatStandsForIt() {
        val maker = maker()
        assumeTrue("the engine or its network is missing", maker != null)
        val files = File(context.cacheDir, "review-twins").apply {
            deleteRecursively()
            mkdirs()
        }
        val settings = SettingsStore(context)
        val fast = GameReview.STRENGTHS.getValue("fast")

        fun watchedAndImported(name: String, sans: List<String>): Pair<Store, LibraryRow> {
            val store = Store(File(files, "$name.sqlite"))
            val row = Library.fromPgn(LONGER, listOf("nosegazm")).single()
            store.addLibraryGame(row)
            val started = row.playedAt!! + 3600
            store.startGame("g1", started, "black", STARTING_FEN)
            sans.forEachIndexed { i, san ->
                val ply = i + 1
                store.addMove("g1", 0, ply, (ply + 1) / 2, san, "0000", ply % 2 == 0, "-", "-", null, null, null, false, false, null, started + ply)
            }
            store.endGame("g1", started + 900, "0-1", "resignation")
            val watched = Library.fromWatched(store).single { it.key == "watched:g1" }
            assertTrue(ReviewRoom.reviewRow(settings, files, watched, store, fast, 1, maker!!) != null)
            return store to row
        }

        val (store, row) = watchedAndImported("same", SANS)
        val twins = Library.reconcile(store)
        assertEquals(mapOf("watched:g1" to row.key), twins)
        assertEquals(listOf(row.key), ReviewRoom.carryTwins(store, twins))
        val report = JSONObject(store.review(row.key, "fast")!!.report)
        val first = report.getJSONArray("moves").getJSONObject(0)
        assertEquals("the import's headers", "1018", report.getJSONObject("headers").getString("BlackElo"))
        assertEquals("and clocks", 598_000, first.getInt("clock_ms"))
        assertEquals(row.key, first.getJSONObject("assessment").getString("game_id"))
        assertEquals("Stats has it once, as the import", listOf(row.key), store.statGames().map { it.key })
        assertEquals("carried once", emptyList<String>(), ReviewRoom.carryTwins(store, twins))
        store.close()

        val (other, otherRow) = watchedAndImported("misread", SANS.take(6) + "d4" + SANS.drop(7))
        assertEquals("nothing to carry whole", emptyList<String>(), ReviewRoom.carryTwins(other, Library.reconcile(other)))
        var known = -1
        val reviewed = ReviewRoom.reviewRow(settings, files, otherRow, other, fast, 1, maker!!, onProgress = { done, _ -> if (known < 0) known = done })
        assertTrue(reviewed != null)
        assertEquals("the seven positions before the misread move, not searched again", 7, known)
        other.close()
    }

    /**
     * The review of 2026-10-01: the desktop's background review and a review window on one game
     * at once -- the window saved first and the player paid for the AI coach's words; the
     * background then saved its own review over it, the words with it. The review saved first
     * stays, words and all; a newer format still replaces it; a removed game is not saved. The
     * desktop's test_the_review_saved_first_stays_with_the_words_paid_for, test_library.py.
     */
    @Test
    fun theReviewSavedFirstStaysWithTheWordsPaidFor() {
        val store = Store(null)
        val row = Library.fromPgn(pgn, listOf("nosegazm")).single()
        store.addLibraryGame(row)
        assertTrue(store.saveReview(row.key, "standard", 1, 100.0, "{\"moves\": []}", null))
        store.setReviewWords(row.key, "standard", "{\"texts\": {\"4\": {\"why\": \"Paid for.\"}}}")
        val first = store.review(row.key, "standard")
        assertFalse("a review of this format saved already stays", store.saveReview(row.key, "standard", 1, 150.0, "{}", null))
        assertEquals("with its words", first, store.review(row.key, "standard"))
        assertTrue("a newer format replaces it", store.saveReview(row.key, "standard", 2, 200.0, "{}", null))
        assertNull("its words with it", store.review(row.key, "standard")!!.words)
        store.removeGame(row.key)
        assertFalse("removed: not saved", store.saveReview(row.key, "deep", 2, 300.0, "{}", null))
        assertNull(store.review(row.key, "deep"))
        store.close()

        // A review finishing second is handed the one saved first.
        val maker = maker()
        assumeTrue("the engine or its network is missing", maker != null)
        val files = File(context.cacheDir, "review-first").apply {
            deleteRecursively()
            mkdirs()
        }
        val settings = SettingsStore(context)
        val fast = GameReview.STRENGTHS.getValue("fast")
        val own = Store(File(files, "profile.sqlite"))
        own.addLibraryGame(row)
        val once = ReviewRoom.reviewRow(settings, files, row, own, fast, 1, maker!!)!!
        own.setReviewWords(row.key, "fast", "{\"texts\": {}}")
        val saved = own.review(row.key, "fast")
        val twice = ReviewRoom.reviewRow(settings, files, row, own, fast, 1, maker)!!
        assertEquals("the review saved first, and its words", saved, own.review(row.key, "fast"))
        assertEquals("handed the saved one", once.first.seconds, twice.first.seconds, 0.0)
        own.close()
    }

    /** A model whose every answer waits for [go], recording which key moves it was asked about. */
    private class Held(private val go: CountDownLatch) : LLMProvider {
        override val name = "held"
        val asked: MutableSet<Int> = ConcurrentHashMap.newKeySet()

        override fun healthcheck(): Pair<Boolean, String> = true to "held"

        override fun chat(system: String, messages: List<ChatMessage>, maxTokens: Int, temperature: Double, thinking: Boolean): Completion {
            asked.add(PLY.find(messages[0].content)?.groupValues?.get(1)?.toInt() ?: -1)
            go.await(20, TimeUnit.SECONDS)
            return Completion(
                text = "### why\nThis gives the opponent time.\n### lesson\nLook at what your opponent can do next.",
                stats = mapOf("prompt_tokens" to 1000.0, "completion_tokens" to 300.0),
            )
        }
    }

    /**
     * The review of 2026-10-01: the window closed (the page sends /close as it unloads) while
     * the AI coach's words were written closed the room's store, and every answer that came
     * after -- each one paid for -- was thrown away. The room stays open until the words are
     * saved; opened again meanwhile it goes on, still writing (nothing asked twice), and stays.
     * No engine: the review is the desktop's, saved (profile/reviews.json). The desktop's
     * test_the_ai_words_are_kept_when_the_review_closes_while_they_are_written.
     */
    @Test
    fun theAiWordsAreKeptWhenTheReviewClosesWhileTheyAreWritten() {
        val fixture = JSONObject(InstrumentationRegistry.getInstrumentation().context.assets.open("profile/reviews.json").bufferedReader().use { it.readText() })
        val games = fixture.getJSONArray("games")
        val game = (0 until games.length()).map { games.getJSONObject(it) }
            .first { it.getString("strength") == "fast" && it.getJSONArray("key_plies").length() > 0 }
        val files = File(context.cacheDir, "review-close-writing").apply {
            deleteRecursively()
            mkdirs()
        }
        val storePath = File(files, "profile.sqlite")
        val row = Library.fromPgn(game.getString("pgn"), listOf("nosegazm")).single()
        Store(storePath).apply {
            addLibraryGame(row)
            assertTrue(saveReview(row.key, "fast", ReviewRoom.REVIEW_VERSION, 1.0, game.getJSONObject("reviewed").toString(), game.getJSONObject("summary").toString()))
        }.close()
        val settings = SettingsStore(context)
        val noEngine = EngineMaker { _, _ -> throw AssertionError("a saved review needs no engine") }

        fun saved(): JSONObject? = Store(storePath).let { s -> s.review(row.key, "fast").also { s.close() } }
            ?.words?.takeIf { it.isNotEmpty() }?.let { JSONObject(it) }

        var go = CountDownLatch(1)
        var model = Held(go)
        val room = ReviewRoom(files, settings, row, storePath, "fast", noEngine, model = { model to null })
        until(room, 10, "loaded") { it.getString("status") == "ready" }
        assertNull(room.writeWords())
        room.close()
        Thread.sleep(1000)
        assertFalse("open while its words are written", room.closed)
        assertTrue("opened again: the same room goes on", room.reopen())
        assertEquals("writing", room.state().getJSONObject("ai").getString("status"))
        go.countDown()
        until(room, 20, "words") { it.getJSONObject("ai").getString("status") == "ready" }
        assertFalse("saved whole", saved()!!.optBoolean("partial", false))
        val asked = model.asked.size
        assertEquals("asked once each", game.getJSONArray("key_plies").length(), asked)
        Thread.sleep(1000)
        assertFalse("opened again, it stays open", room.closed)
        room.close()
        val end = SystemClock.elapsedRealtime() + 5000
        while (!room.closed && SystemClock.elapsedRealtime() < end) Thread.sleep(50)
        assertTrue("nothing being written: closed at once", room.closed)

        // Closed while writing, not opened again: open until the words are saved, then closed.
        Store(storePath).apply { setReviewWords(row.key, "fast", "") }.close()
        go = CountDownLatch(1)
        model = Held(go)
        val again = ReviewRoom(files, settings, row, storePath, "fast", noEngine, model = { model to null })
        until(again, 10, "loaded again") { it.getString("status") == "ready" }
        assertNull(again.writeWords())
        again.close()
        Thread.sleep(1000)
        assertFalse(again.closed)
        go.countDown()
        val done = SystemClock.elapsedRealtime() + 20_000
        while (!again.closed && SystemClock.elapsedRealtime() < done) Thread.sleep(50)
        assertTrue("closed once its words are saved", again.closed)
        assertFalse("saved whole", saved()!!.optBoolean("partial", false))
    }

    companion object {
        private val PLY = Regex("\"key move\": \\{\"ply\": (\\d+)")

        // tests/test_review_server.py's LONGER: one move misread still leaves twelve to pair on.
        private const val LONGER = "[Event \"Live Chess\"]\n[Site \"Chess.com\"]\n[Date \"2026.09.29\"]\n" +
            "[White \"Anakin_737\"]\n[Black \"nosegazm\"]\n[Result \"0-1\"]\n[WhiteElo \"880\"]\n" +
            "[BlackElo \"1018\"]\n[TimeControl \"600\"]\n\n" +
            "1. e4 {[%clk 0:09:58]} e5 {[%clk 0:09:55]} 2. Nf3 {[%clk 0:09:50]} Nc6 {[%clk 0:09:40]}\n" +
            "3. Bc4 {[%clk 0:09:45]} Nf6 {[%clk 0:09:30]} 4. d3 Bc5 5. Nc3 d6 6. h3 Na5 7. Bb3 Nxb3 8. axb3 c6 0-1\n"
        private val SANS = "e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Bc5 Nc3 d6 h3 Na5 Bb3 Nxb3 axb3 c6".split(" ")
    }
}
