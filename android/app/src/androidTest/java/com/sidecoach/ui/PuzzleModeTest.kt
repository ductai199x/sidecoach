package com.sidecoach.ui

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.events.GameStarted
import com.sidecoach.profile.Glicko
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

    private companion object {
        // A safety net for a room's signal, never the expected path.
        const val HANG_MS = 60_000L

        // A room's outcome line as mirrored: "<id> game: puzzle <id> solved|failed ...".
        val OUTCOME = Regex("^\\S+ game: puzzle ")
    }

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
        assertTrue((st["message"] as String).contains("Not quite"))
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
        // The result line's numbers come once the room has recorded the outcome.
        assertEquals(null, s.state()["result"])
        s.recordedAs(true, 1500, 1712, true)
        assertEquals(mapOf("solved" to true, "level" to 1712, "change" to 212, "provisional" to true), s.state()["result"])
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
            val (api, rooms) = api(tmp, Puzzles.openStore(tmp))
            PanelEngine.puzzleMotifs[fork.id] = "fork"
            PanelEngine.practiceIds = listOf(fork.id, mate.id)
            assertEquals(false, call(api, "POST", "/api/puzzle/nope/open").getBoolean("ok"))

            val opened = call(api, "POST", "/api/puzzle/${fork.id}/open")
            assertTrue(opened.getBoolean("ok") && strings(opened, "next") == listOf(mate.id))
            val state = opened.getJSONObject("state")
            assertTrue(state.getString("id") == fork.id && state.getString("motif") == "fork" && state.getString("bottom") == "black")
            assertTrue(state.getBoolean("your_move") && !state.getBoolean("closed"))
            assertTrue("no result line while the puzzle is played", state.isNull("result"))
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
            outcome(api, fork.id, "puzzle ${fork.id} failed (fork)")
            var store = Store(File(tmp, "profile.sqlite"))
            assertTrue(store.attempts().map { it.first } == listOf(fork.id) && !store.attempts()[0].third)
            // The puzzle rating as Lichess moves it: a new player (1500) loses to the puzzle's 1547 / 76,
            // opened as a fork puzzle (the page says "Puzzle · fork"): Lichess's themed training keeps
            // 0.7 of a loss on a theme that hints at the solution.
            val first = Glicko.finish(Glicko.Perf(), Glicko.Rating(1547.0, 76.0, 0.09), false, 0L, "fork").glicko.intRating
            assertTrue(Puzzles.puzzleLevel(store) == first && first < 1500)
            assertEquals("the old level is neither read nor written", null, store.setting("puzzle_level"))
            // The page's result line reads the outcome as the profile took it: a miss is not a solve.
            assertEquals(result(false, first, first - 1500), polledResult(api, fork.id))
            // The user's question reaches the room. Its coach is off here: the room says so, as
            // the panel does, or the page's typing dots waited for an answer for good.
            assertTrue(call(api, "POST", "/api/puzzle/${fork.id}/ask", """{"text": "hint"}""").getBoolean("ok"))
            logged(api, fork.id, "user", "no coach to ask: hint")
            val said = call(api, "GET", "/api/puzzle/${fork.id}/poll").getJSONArray("lines")
            assertTrue(
                "no turn of a conversation nobody answers",
                (0 until said.length()).none { said.getJSONObject(it).getString("text") == "you: hint" },
            )
            assertTrue(call(api, "POST", "/api/puzzle/${fork.id}/voice", """{"on": false}""").getBoolean("ok"))
            // Closing ends the room; the outcome is not recorded twice, counted once the room's
            // loop has drained the close and finished (its thread joined), not merely once the
            // close was asked for.
            val forkRoom = rooms.get(fork.id)!!
            assertTrue(call(api, "POST", "/api/puzzle/${fork.id}/close").getBoolean("ok"))
            forkRoom.join(HANG_MS)
            assertTrue("the room never finished", forkRoom.closed)
            assertTrue(call(api, "GET", "/api/puzzle/${fork.id}/poll").optBoolean("closed"))
            assertEquals(1, store.attempts().size)
            assertTrue("an outcome logged twice", outcomes.isEmpty())
            store.close()
            // Giving up on the next one: a fail, the solution played out, recorded once.
            assertTrue(call(api, "POST", "/api/puzzle/${mate.id}/open").getBoolean("ok"))
            val gave = call(api, "POST", "/api/puzzle/${mate.id}/giveup").getJSONObject("state")
            assertTrue(gave.getString("status") == "given_up" && strings(gave, "moves") == listOf("Ra8#"))
            outcome(api, mate.id, "puzzle ${mate.id} failed")
            store = Store(File(tmp, "profile.sqlite"))
            assertEquals(listOf(mate.id, fork.id), store.attempts().map { it.first })
            val second = Puzzles.puzzleLevel(store)
            assertTrue("a give-up is a loss", second < first)
            store.close()
            assertEquals(result(false, second, second - first), polledResult(api, mate.id))
            val mateRoom = rooms.get(mate.id)!!
            call(api, "POST", "/api/puzzle/${mate.id}/close")
            mateRoom.join(HANG_MS)
            assertTrue("the room never finished", mateRoom.closed)
            assertTrue("an outcome logged twice", outcomes.isEmpty())
        } finally {
            PanelEngine.puzzleMotifs.clear()
            PanelEngine.practiceIds = emptyList()
            tmp.deleteRecursively()
        }
    }

    /**
     * The result line after a solve: the puzzle rating after it and the change Lichess shows (the
     * difference of the two ratings as shown, truncated).
     * `test_a_solved_puzzle_shows_the_rating_it_moved_to`.
     */
    @Test
    fun aSolvedPuzzleShowsTheRatingItMovedTo() {
        val tmp = tempDir()
        try {
            csv(tmp)
            val won = Glicko.finish(Glicko.Perf(), Glicko.Rating(900.0, 70.0, 0.09), true, 0L).glicko.intRating
            assertTrue("a solve against a much weaker puzzle: a small gain", won in 1501..1599)
            val (api, rooms) = api(tmp, Puzzles.openStore(tmp))
            val opened = call(api, "POST", "/api/puzzle/${mate.id}/open")
            assertTrue(opened.getJSONObject("state").isNull("result"))
            val done = call(api, "POST", "/api/puzzle/${mate.id}/move", """{"uci": "a1a8"}""").getJSONObject("state")
            assertEquals("solved", done.getString("status"))
            outcome(api, mate.id, "puzzle ${mate.id} solved")
            assertEquals(result(true, won, won - 1500), polledResult(api, mate.id))
            val mateRoom = rooms.get(mate.id)!!
            call(api, "POST", "/api/puzzle/${mate.id}/close")
            mateRoom.join(HANG_MS)
            assertTrue("the room never finished", mateRoom.closed)
        } finally {
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

    /**
     * Lichess records a puzzle's result at its first wrong move or at its end; one left before
     * either is not played. One closed after a wrong move is the fail it already was.
     * `test_a_puzzle_closed_before_any_wrong_move_records_nothing`.
     */
    @Test
    fun aPuzzleClosedBeforeAnyWrongMoveRecordsNothing() {
        val tmp = tempDir()
        try {
            csv(tmp)
            val (api, rooms) = api(tmp, Puzzles.openStore(tmp))
            assertTrue(call(api, "POST", "/api/puzzle/${fork.id}/open").getBoolean("ok"))
            val right = call(api, "POST", "/api/puzzle/${fork.id}/move", """{"uci": "d4e2"}""").getJSONObject("state")
            assertEquals("one right move, then left", listOf("Ne2+", "Kf1"), strings(right, "moves"))
            val forkRoom = rooms.get(fork.id)!!
            assertTrue(call(api, "POST", "/api/puzzle/${fork.id}/close").getBoolean("ok"))
            forkRoom.join(HANG_MS)
            assertTrue("the room never finished", forkRoom.closed)
            var store = Store(File(tmp, "profile.sqlite"))
            assertTrue(store.attempts().isEmpty() && Puzzles.puzzleLevel(store) == 1500)
            store.close()
            // A wrong move, then closed: the fail is recorded.
            assertTrue(call(api, "POST", "/api/puzzle/${mate.id}/open").getBoolean("ok"))
            val wrong = call(api, "POST", "/api/puzzle/${mate.id}/move", """{"uci": "a1a7"}""").getJSONObject("state")
            assertEquals("a1a7", wrong.getString("wrong"))
            val mateRoom = rooms.get(mate.id)!!
            assertTrue(call(api, "POST", "/api/puzzle/${mate.id}/close").getBoolean("ok"))
            mateRoom.join(HANG_MS)
            assertTrue("the room never finished", mateRoom.closed)
            store = Store(File(tmp, "profile.sqlite"))
            assertEquals(listOf(mate.id to false), store.attempts().map { it.first to it.third })
            assertTrue(Puzzles.puzzleLevel(store) < 1500)
            store.close()
        } finally {
            tmp.deleteRecursively()
        }
    }

    /**
     * Lichess writes a rating with a "?" while its deviation is 110 or more; a player with a settled
     * rating (here 1600 / 60) gets the number alone, and a small change.
     * `test_an_established_rating_shows_without_the_question_mark`.
     */
    @Test
    fun anEstablishedRatingShowsWithoutTheQuestionMark() {
        val tmp = tempDir()
        try {
            csv(tmp)
            val store = Store(File(tmp, "profile.sqlite"))
            store.setSetting(Puzzles.PERF_KEY, """{"r": 1600.0, "d": 60.0, "v": 0.06, "nb": 500, "la": null}""")
            store.close()
            val settled = Glicko.Perf(Glicko.Rating(1600.0, 60.0, 0.06), nb = 500)
            val won = Glicko.finish(settled, Glicko.Rating(900.0, 70.0, 0.09), true, 0L).glicko
            assertFalse(won.provisional)
            val (api, rooms) = api(tmp, Puzzles.openStore(tmp))
            assertTrue(call(api, "POST", "/api/puzzle/${mate.id}/open").getBoolean("ok"))
            val done = call(api, "POST", "/api/puzzle/${mate.id}/move", """{"uci": "a1a8"}""").getJSONObject("state")
            assertEquals("solved", done.getString("status"))
            outcome(api, mate.id, "puzzle ${mate.id} solved")
            assertEquals(result(true, won.intRating, won.intRating - 1600, false), polledResult(api, mate.id))
            assertTrue("a settled rating barely moves for a far weaker puzzle", won.intRating - 1600 in 0..4)
            val mateRoom = rooms.get(mate.id)!!
            call(api, "POST", "/api/puzzle/${mate.id}/close")
            mateRoom.join(HANG_MS)
            assertTrue("the room never finished", mateRoom.closed)
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

    /**
     * What the rooms' loop threads said, as they said it: every line a room logs comes through
     * the rooms' own mirror (as the app's panel gets them), so the test waits on the line itself
     * rather than polling the page against a clock (the desktop's `_RoomSignals`, 0bf0f27). An
     * outcome ("<id> game: puzzle ...") goes to [outcomes], every other line to [lines].
     */
    private val outcomes = java.util.concurrent.LinkedBlockingQueue<String>()
    private val lines = java.util.concurrent.LinkedBlockingQueue<String>()

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
            mirror = { _, _, text, _ -> if (OUTCOME.containsMatchIn(text)) outcomes.put(text) else lines.put(text) },
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
     * The room logs its outcome, the next one it logs, starting with [prefix] -- the outcome is
     * recorded by the room's loop, a tick after the answer that ended the puzzle -- and the
     * page's poll shows it. [HANG_MS] is only a safety net: the line comes at the loop's next tick.
     */
    private fun outcome(api: PanelApi, id: String, prefix: String) {
        val said = outcomes.poll(HANG_MS, java.util.concurrent.TimeUnit.MILLISECONDS)
        assertTrue("no outcome '$prefix'", said != null && said.startsWith("$id game: $prefix"))
        assertTrue(polled(api, id).toString(), polled(api, id).any { it.startsWith(prefix) })
    }

    /** The room logs a line on [chan] starting with [prefix] (others it logs before are passed by), and the poll shows it. */
    private fun logged(api: PanelApi, id: String, chan: String, prefix: String) {
        while (true) {
            val said = lines.poll(HANG_MS, java.util.concurrent.TimeUnit.MILLISECONDS)
            assertTrue("no line '$prefix'", said != null)
            if (said!!.startsWith("$id $chan: $prefix")) break
        }
        assertTrue(polled(api, id).toString(), polled(api, id).any { it.startsWith(prefix) })
    }

    private fun polled(api: PanelApi, id: String): List<String> {
        val all = call(api, "GET", "/api/puzzle/$id/poll").getJSONArray("lines")
        return (0 until all.length()).map { all.getJSONObject(it).getString("text") }
    }

    /** The result the page's poll serves, as plain values. */
    private fun polledResult(api: PanelApi, id: String): Map<String, Any?> {
        val r = call(api, "GET", "/api/puzzle/$id/poll").getJSONObject("state").getJSONObject("result")
        return result(r.getBoolean("solved"), r.getInt("level"), r.getInt("change"), r.getBoolean("provisional"))
    }

    /** A result line's numbers; a new player's rating is provisional after a puzzle or two. */
    private fun result(solved: Boolean, level: Int, change: Int, provisional: Boolean = true): Map<String, Any?> =
        mapOf("solved" to solved, "level" to level, "change" to change, "provisional" to provisional)

    @Suppress("UNCHECKED_CAST")
    private fun legal(state: Map<String, Any?>) = state["legal"] as Map<String, List<String>>

    @Suppress("UNCHECKED_CAST")
    private fun positions(state: Map<String, Any?>) = state["positions"] as List<Map<String, Any?>>

    private fun strings(json: JSONObject, key: String): List<String> {
        val all = json.getJSONArray(key)
        return (0 until all.length()).map { all.getString(it) }
    }
}
