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
 * side, the line's replies, a wrong move shown and taken back, any mate where the line mates,
 * the solution on give-up; and the room around it, over the API -- its own coach and log, the
 * outcome to the profile once. `tests/test_puzzle_mode.py`, test for test.
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
        assertEquals(listOf("b3", "b5", "c2", "c6", "e2", "f3", "f5"), s.targets("d4"))
        // Python's `a and b and c == []`: the queen and the king may move, White's knight is not theirs.
        assertTrue("not theirs", s.targets("e5").isNotEmpty() && s.targets("g8").isNotEmpty() && s.targets("d2").isEmpty())
        assertTrue(s.state()["your_move"] == true && s.state()["remaining"] == 2)
        var events = s.move("d4e2")
        assertEquals("the move, then the line's reply", listOf("Ne2+", "Kf1"), events.map { it.san })
        assertTrue(events[0].byUser && !events[1].byUser)
        assertTrue(events[0].gameId == "puzzle-000Pw" && events[0].ply == 1 && events[1].ply == 2)
        assertTrue(s.state()["your_move"] == true && s.state()["remaining"] == 1)
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
        assertTrue(s.targets("d4").isEmpty())
        s.retract()
        assertTrue(s.state()["wrong"] == null && s.state()["your_move"] == true)
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
            val targets = call(api, "POST", "/api/puzzle/${fork.id}/targets", """{"square": "d4"}""")
            assertEquals(listOf("b3", "b5", "c2", "c6", "e2", "f3", "f5"), strings(targets, "targets"))
            // A wrong move shows, then is taken back.
            assertTrue(call(api, "POST", "/api/puzzle/${fork.id}/move", """{"uci": "d4c2"}""").getBoolean("ok"))
            waitFor(api, fork.id) { it.optString("wrong") == "d4c2" }
            call(api, "POST", "/api/puzzle/${fork.id}/retract")
            waitFor(api, fork.id) { it.isNull("wrong") && it.getBoolean("your_move") }
            // The line: the reply comes with the right move.
            call(api, "POST", "/api/puzzle/${fork.id}/move", """{"uci": "d4e2"}""")
            var res = waitFor(api, fork.id) { strings(it, "moves") == listOf("Ne2+", "Kf1") }
            assertTrue(res.getJSONObject("state").getBoolean("your_move"))
            call(api, "POST", "/api/puzzle/${fork.id}/move", """{"uci": "e2c3"}""")
            res = waitFor(api, fork.id) { it.getString("status") == "solved" }
            assertEquals("Solved, after a miss.", res.getJSONObject("state").getString("message"))
            // The outcome's line follows the state by a moment; wait for it.
            var lines = emptyList<String>()
            for (i in 0 until 50) {
                val all = call(api, "GET", "/api/puzzle/${fork.id}/poll").getJSONArray("lines")
                lines = (0 until all.length()).map { all.getJSONObject(it).getString("text") }
                if (lines.any { it.startsWith("puzzle ${fork.id} failed (fork)") }) break
                Thread.sleep(50)
            }
            assertTrue(lines.toString(), lines.any { it.startsWith("puzzle ${fork.id} failed (fork)") })
            var store = Store(File(tmp, "profile.sqlite"))
            assertTrue(store.attempts().map { it.first } == listOf(fork.id) && !store.attempts()[0].third)
            assertEquals("1075", store.setting("puzzle_level"))
            // The user's question reaches the room's coach (off here: nothing answers, no error).
            assertTrue(call(api, "POST", "/api/puzzle/${fork.id}/ask", """{"text": "hint"}""").getBoolean("ok"))
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
            call(api, "POST", "/api/puzzle/${mate.id}/giveup")
            res = waitFor(api, mate.id) { it.getString("status") == "given_up" }
            assertEquals(listOf("Ra8#"), strings(res.getJSONObject("state"), "moves"))
            // The attempt follows the state by a moment, as the outcome's line does.
            store = Store(File(tmp, "profile.sqlite"))
            for (i in 0 until 50) {
                if (store.attempts().size == 2) break
                Thread.sleep(50)
            }
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
        return PanelApi("test", SetupApi(app), SettingsApi(settings), dir) { rooms } to rooms
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

    /** Poll the room until [want] holds of its state; the loop runs a quarter second apart. */
    private fun waitFor(api: PanelApi, id: String, want: (JSONObject) -> Boolean): JSONObject {
        val deadline = System.nanoTime() + 5_000_000_000L
        var res = JSONObject()
        while (System.nanoTime() < deadline) {
            res = call(api, "GET", "/api/puzzle/$id/poll")
            if (res.optBoolean("ok") && want(res.getJSONObject("state"))) return res
            Thread.sleep(50)
        }
        throw AssertionError("the room never reached the state wanted: $res")
    }

    private fun strings(json: JSONObject, key: String): List<String> {
        val all = json.getJSONArray(key)
        return (0 until all.length()).map { all.getString(it) }
    }
}
