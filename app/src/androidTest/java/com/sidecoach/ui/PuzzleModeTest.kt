package com.sidecoach.ui

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.events.GameStarted
import com.sidecoach.profile.Puzzle
import com.sidecoach.profile.Puzzles
import com.sidecoach.profile.Store
import com.sidecoach.vision.Pieces
import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * A puzzle on a page of its own (PuzzleSession, PuzzleRoom): the position from the solver's
 * side, the solver's legal moves with the state, a version that grows with every change, the
 * positions played one by one, the line's replies, a wrong move shown and taken back, any mate
 * where the line mates, the solution on give-up; and the room around it, over the API -- each
 * move answered with the state it leaves, its own coach and log, the outcome to the profile
 * once. `tests/test_puzzle_mode.py`, test for test.
 */
@RunWith(AndroidJUnit4::class)
class PuzzleModeTest {

    private val fork = Puzzle(
        id = "000Pw",
        fen = "6k1/5p1p/4p3/4q3/3nN3/2Q3P1/PP3P1P/6K1 w - - 2 37",
        moves = listOf("e4d2", "d4e2", "g1f1", "e2c3"),
        rating = 1547,
        popularity = 76,
        plays = 629,
        themes = listOf("fork", "endgame"),
    )

    // Black's king steps to the corner; White mates on the back rank.
    private val mate = Puzzle(
        id = "mate1",
        fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1",
        moves = listOf("g8h8", "a1a8"),
        rating = 900,
        popularity = 70,
        plays = 95,
        themes = listOf("mateIn1", "backRankMate"),
    )

    @Test
    fun theSessionShowsTheSolversPositionAndWalksTheLine() {
        val s = PuzzleSession(fork, "fork")
        assertTrue(s.side == Pieces.Side.BLACK && s.state()["bottom"] == "black")
        assertEquals("after the opponent's e4d2, Black to move", "b", s.startFen.split(" ")[1])
        val started = s.started()
        assertTrue(started is GameStarted && started.kind == "puzzle")
        assertTrue(started.gameId == "puzzle-000Pw" && started.userSide == Pieces.Side.BLACK)
        val legal = legal(s.state())
        assertEquals(listOf("b3", "b5", "c2", "c6", "e2", "f3", "f5"), legal["d4"])
        assertTrue("the solver's pieces only", "e5" in legal && "g8" in legal && "d2" !in legal)
        assertTrue(s.state()["your_move"] == true && s.state()["remaining"] == 2)
        val before = s.state()
        assertEquals("the start alone", listOf<String?>(null), positions(before).map { it["uci"] })
        var events = s.move("d4e2")
        assertEquals("the move, then the line's reply", listOf("Ne2+", "Kf1"), events.map { it.san })
        assertTrue(events[0].byUser && !events[1].byUser)
        assertTrue(events[0].gameId == "puzzle-000Pw" && events[0].ply == 1 && events[1].ply == 2)
        val after = s.state()
        assertTrue(after["your_move"] == true && after["remaining"] == 1)
        assertTrue("a newer answer tells itself apart", (after["version"] as Int) > (before["version"] as Int))
        // One position a ply, for the page to step through: its move, the check, who played it.
        val plies = positions(after).drop(1)
        assertEquals(
            listOf(Triple("d4e2", "g1", true), Triple("g1f1", null, false)),
            plies.map { Triple(it["uci"], it["check"], it["by_user"]) },
        )
        assertEquals(s.board.fen(), plies.last()["fen"])
        events = s.move("e2c3")
        assertTrue(events.map { it.san } == listOf("Nxc3") && s.status == "solved" && s.done)
        assertTrue(s.state()["message"] == "Solved." && !s.failed)
        assertTrue("over", s.move("g8f8").isEmpty())
    }

    @Test
    fun aWrongMoveIsShownAndTakenBackAndCountsAsAMiss() {
        val s = PuzzleSession(fork, "fork")
        assertTrue(s.move("d4c2").isEmpty())
        val st = s.state()
        assertTrue(st["wrong"] == "d4c2" && st["your_move"] == false && st["misses"] == 1)
        assertTrue((st["message"] as String).contains("Not that one"))
        assertTrue("nothing moves while the wrong move is shown", s.move("d4e2").isEmpty())
        assertEquals(emptyMap<String, List<String>>(), st["legal"])
        s.retract()
        assertTrue(s.state()["wrong"] == null && s.state()["your_move"] == true)
        assertTrue("the take-back is a newer state", (s.state()["version"] as Int) > (st["version"] as Int))
        assertEquals("the position as it was", s.startFen, s.board.fen())
        assertTrue(s.move("d4e2").isNotEmpty() && s.move("e2c3").isNotEmpty() && s.status == "solved" && s.failed)
        assertEquals("Solved, after a miss.", s.state()["message"])
        // Not legal, not their piece, not their turn: ignored, no miss.
        val t = PuzzleSession(fork, "fork")
        assertTrue(t.move("d4d8").isEmpty() && t.move("d2e4").isEmpty() && t.move("zz").isEmpty())
        assertTrue(t.misses == 0 && t.wrong == null)
    }

    @Test
    fun aMatePuzzleAndGivingUpOnIt() {
        val s = PuzzleSession(mate, "mate")
        assertTrue(s.side == Pieces.Side.WHITE && s.board.turn == WHITE)
        val events = s.move("a1a8")
        assertTrue(events.map { it.san } == listOf("Ra8#") && s.status == "solved" && !s.failed)
        // Any mate counts where the line's move mates; a non-mating move does not.
        val t = PuzzleSession(mate, "mate")
        assertTrue(t.mates(Move.fromUci("a1a8")))
        assertFalse(t.mates(Move.fromUci("a1a7")))
        assertTrue(t.move("a1a7").isEmpty() && t.wrong == "a1a7")
        t.retract()
        val given = t.giveUp()
        assertTrue(given.map { it.uci } == listOf("a1a8") && t.status == "given_up" && t.failed)
        assertEquals("The solution, played out.", t.state()["message"])
        assertTrue("once", t.giveUp().isEmpty())
    }

    @Test
    fun aRoomRunsAPuzzleOverTheApiAndRecordsTheOutcomeOnce() {
        val tmp = tempDir()
        try {
            csv(tmp)
            val (api, _) = api(tmp, Puzzles.openStore(tmp))
            PanelEngine.puzzleMotifs[fork.id] = "fork"
            PanelEngine.practiceIds = listOf(fork.id, mate.id)
            assertEquals(false, call(api, "POST", "/api/puzzle/nope/open").getBoolean("ok"))

            val opened = call(api, "POST", "/api/puzzle/${fork.id}/open")
            assertTrue(opened.getBoolean("ok") && strings(opened, "next") == listOf(mate.id))
            val state = opened.getJSONObject("state")
            assertTrue(state.getString("id") == fork.id && state.getString("motif") == "fork" && state.getString("bottom") == "black")
            assertTrue(state.getBoolean("your_move") && !state.getBoolean("closed"))
            assertEquals(listOf("b3", "b5", "c2", "c6", "e2", "f3", "f5"), strings(state.getJSONObject("legal"), "d4"))
            // Each move is answered with the state it leaves, at once. A wrong move shows, then
            // is taken back.
            val wrong = call(api, "POST", "/api/puzzle/${fork.id}/move", """{"uci": "d4c2"}""")
            assertTrue(wrong.getBoolean("ok") && wrong.getJSONObject("state").getString("wrong") == "d4c2")
            val back = call(api, "POST", "/api/puzzle/${fork.id}/retract").getJSONObject("state")
            assertTrue(back.isNull("wrong") && back.getBoolean("your_move"))
            assertTrue(back.getInt("version") > wrong.getJSONObject("state").getInt("version"))
            // The line: the reply comes with the right move.
            val right = call(api, "POST", "/api/puzzle/${fork.id}/move", """{"uci": "d4e2"}""").getJSONObject("state")
            assertTrue(strings(right, "moves") == listOf("Ne2+", "Kf1") && right.getBoolean("your_move"))
            val done = call(api, "POST", "/api/puzzle/${fork.id}/move", """{"uci": "e2c3"}""").getJSONObject("state")
            assertTrue(done.getString("status") == "solved" && done.getString("message") == "Solved, after a miss.")
            logged(api, fork.id, "puzzle ${fork.id} failed (fork)")
            var store = Store(File(tmp, "profile.sqlite"))
            assertTrue(store.attempts().map { it.first } == listOf(fork.id) && !store.attempts()[0].third)
            assertEquals("1075", store.setting("puzzle_level"))
            // The user's question reaches the room. Its coach is off here: the room says so, as
            // the panel does, or the page's typing dots waited for an answer for good.
            assertTrue(call(api, "POST", "/api/puzzle/${fork.id}/ask", """{"text": "hint"}""").getBoolean("ok"))
            logged(api, fork.id, "no coach to ask: hint")
            val said = call(api, "GET", "/api/puzzle/${fork.id}/poll").getJSONArray("lines")
            assertTrue(
                "no turn of a conversation nobody answers",
                (0 until said.length()).none { said.getJSONObject(it).getString("text") == "you: hint" },
            )
            assertTrue(call(api, "POST", "/api/puzzle/${fork.id}/voice", """{"on": false}""").getBoolean("ok"))
            // Closing ends the room; the outcome is not recorded twice.
            assertTrue(call(api, "POST", "/api/puzzle/${fork.id}/close").getBoolean("ok"))
            for (i in 0 until 50) {
                if (call(api, "GET", "/api/puzzle/${fork.id}/poll").optBoolean("closed")) break
                Thread.sleep(50)
            }
            assertTrue(call(api, "GET", "/api/puzzle/${fork.id}/poll").optBoolean("closed"))
            assertEquals(1, store.attempts().size)
            store.close()
            // Giving up on the next one: a fail, the solution played out, recorded once.
            assertTrue(call(api, "POST", "/api/puzzle/${mate.id}/open").getBoolean("ok"))
            val gave = call(api, "POST", "/api/puzzle/${mate.id}/giveup").getJSONObject("state")
            assertTrue(gave.getString("status") == "given_up" && strings(gave, "moves") == listOf("Ra8#"))
            logged(api, mate.id, "puzzle ${mate.id} failed")
            store = Store(File(tmp, "profile.sqlite"))
            assertEquals(listOf(mate.id, fork.id), store.attempts().map { it.first })
            assertEquals("1050", store.setting("puzzle_level"))
            store.close()
            call(api, "POST", "/api/puzzle/${mate.id}/close")
        } finally {
            PanelEngine.puzzleMotifs.clear()
            PanelEngine.practiceIds = emptyList()
            tmp.deleteRecursively()
        }
    }

    @Test
    fun puzzlesFetchedAfterTheRoomsWereMadeOpenInARoom() {
        // Setup fetches the puzzles while the app runs: the session finds them at its next
        // Start and recommends one, and the room must find it too, without a restart.
        val tmp = tempDir()
        try {
            val (api, _) = api(tmp, null)
            assertEquals(false, call(api, "POST", "/api/puzzle/${fork.id}/open").getBoolean("ok"))
            csv(tmp)
            val opened = call(api, "POST", "/api/puzzle/${fork.id}/open")
            assertTrue(opened.getBoolean("ok") && opened.getJSONObject("state").getString("id") == fork.id)
            call(api, "POST", "/api/puzzle/${fork.id}/close")
        } finally {
            tmp.deleteRecursively()
        }
    }

    /**
     * Tai, 2026-10-02: a puzzle left open over lunch was gone after 20 idle minutes. A room
     * closes when its page says it goes; the one safety net is a page that said it was in view
     * and then went quiet (its process died). The room's clock is the test's; the loop's reads
     * of it are the signals the test waits on. `test_a_room_closes_with_its_page_never_for_being_left_alone`.
     */
    @Test
    fun aRoomClosesWithItsPageNeverForBeingLeftAlone() {
        val tmp = tempDir()
        try {
            csv(tmp)
            val (api, rooms) = api(tmp, null)
            assertTrue(call(api, "POST", "/api/puzzle/${fork.id}/open").getBoolean("ok"))
            val room = rooms.get(fork.id)!!
            val now = java.util.concurrent.atomic.AtomicLong(1_000_000L)
            // The loop thread's reads of the clock (its check for a vanished page), in order.
            val loopReads = java.util.concurrent.LinkedBlockingQueue<Long>()
            room.clockMs = {
                val t = now.get()
                if (Thread.currentThread().name == "puzzle-${fork.id}") loopReads.put(t)
                t
            }
            /** Two loop checks at [t]: the first of them ran whole, so its verdict stood. */
            fun checkedTwiceAt(t: Long) {
                var seenAt = 0
                while (seenAt < 2) {
                    val read = loopReads.poll(60, java.util.concurrent.TimeUnit.SECONDS)
                        ?: throw AssertionError("the room's loop never checked at $t")
                    if (read == t) seenAt++
                }
            }
            // Out of view, as the page says when it is hidden: hours of quiet close nothing.
            assertTrue(call(api, "GET", "/api/puzzle/${fork.id}/poll?hidden=1").getBoolean("ok"))
            now.addAndGet(3 * 3600 * 1000L)
            checkedTwiceAt(now.get())
            assertFalse(room.closed)
            // Back in view: polled, then quiet with the page in view.
            val polled = call(api, "GET", "/api/puzzle/${fork.id}/poll")
            assertTrue(polled.getBoolean("ok") && !polled.getJSONObject("state").getBoolean("closed"))
            now.addAndGet(PuzzleRoom.VANISHED_MS)
            checkedTwiceAt(now.get())
            assertFalse("not past it", room.closed)
            now.addAndGet(1000L)
            room.join(60_000)
            assertTrue("the vanished page's room never finished", room.closed)
            val gone = call(api, "GET", "/api/puzzle/${fork.id}/poll")
            assertTrue(gone.getBoolean("closed") && !gone.getBoolean("ok"))
            // A page that says it goes closes its room at once (its close, or its beacon).
            assertTrue(call(api, "POST", "/api/puzzle/${fork.id}/open").getBoolean("ok"))
            assertTrue(call(api, "POST", "/api/puzzle/${fork.id}/close").getBoolean("ok"))
            assertTrue(call(api, "GET", "/api/puzzle/${fork.id}/poll").getBoolean("closed"))
        } finally {
            tmp.deleteRecursively()
        }
    }

    // ---- helpers -------------------------------------------------------------------------

    private fun tempDir(): File =
        File(InstrumentationRegistry.getInstrumentation().targetContext.cacheDir, "puzzle-mode-${System.nanoTime()}")
            .apply { mkdirs() }

    private fun csv(dir: File) {
        val header = "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags"
        val rows = listOf(
            "${fork.id},${fork.fen},${fork.moves.joinToString(" ")},${fork.rating},76,92,629,fork endgame,u,",
            "${mate.id},${mate.fen},${mate.moves.joinToString(" ")},${mate.rating},70,95,900,mateIn1,u,",
        )
        File(dir, "puzzles").mkdirs()
        File(dir, "puzzles/sample.csv").writeText((listOf(header) + rows + "").joinToString("\n"))
    }

    /** The API over rooms of their own, the engine and the coach off (as the desktop's test). */
    private fun api(dir: File, puzzles: com.sidecoach.profile.PuzzleStore?): Pair<PanelApi, PuzzleRooms> {
        val app = InstrumentationRegistry.getInstrumentation().targetContext
        val settings = SettingsStore(app)
        val rooms = PuzzleRooms(
            app,
            settings,
            storePath = File(dir, "profile.sqlite"),
            puzzles = puzzles,
            dataDir = dir,
            analysis = false,
            coach = false,
        )
        rooms.motifs = PanelEngine.puzzleMotifs
        return PanelApi("test", SetupApi(app), SettingsApi(settings), dir, rooms = { rooms }) to rooms
    }

    private fun call(api: PanelApi, method: String, path: String, body: String = ""): JSONObject {
        val (bare, query) = if ('?' in path) {
            path.substringBefore('?') to path.substringAfter('?').split('&').associate {
                it.substringBefore('=') to it.substringAfter('=')
            }
        } else {
            path to emptyMap()
        }
        val reply = api.handle(method, bare, query, body) ?: error("no route for $method $path")
        return JSONObject(String(reply.body))
    }

    /**
     * Wait for the room's log line that starts with [prefix]: the outcome is recorded by the
     * room's loop, a tick after the answer that ended the puzzle.
     */
    private fun logged(api: PanelApi, id: String, prefix: String) {
        var lines = emptyList<String>()
        repeat(100) {
            val all = call(api, "GET", "/api/puzzle/$id/poll").getJSONArray("lines")
            lines = (0 until all.length()).map { all.getJSONObject(it).getString("text") }
            if (lines.any { it.startsWith(prefix) }) return
            Thread.sleep(50)
        }
        throw AssertionError("no line '$prefix' in $lines")
    }

    @Suppress("UNCHECKED_CAST")
    private fun legal(state: Map<String, Any?>) = state["legal"] as Map<String, List<String>>

    @Suppress("UNCHECKED_CAST")
    private fun positions(state: Map<String, Any?>) = state["positions"] as List<Map<String, Any?>>

    private fun strings(json: JSONObject, key: String): List<String> {
        val all = json.getJSONArray(key)
        return (0 until all.length()).map { all.getString(it) }
    }
}
