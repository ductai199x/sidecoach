package com.sidecoach.profile

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.Lane
import com.sidecoach.chess.Board
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.chess.WHITE
import com.sidecoach.events.Analysis
import com.sidecoach.events.ClockObservation
import com.sidecoach.events.CoachMessage
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.Event
import com.sidecoach.events.GameEnded
import com.sidecoach.events.GameStarted
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.ProfileEvidence
import com.sidecoach.events.Severity
import com.sidecoach.events.Takeback
import com.sidecoach.events.Trigger
import com.sidecoach.vision.Pieces
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The record the profiler keeps (profile/Profiler.kt), fed games as the pipeline and the
 * annotator emit them -- `tests/profiling.py`'s Feed -- for the fixes of the review of
 * 2026-10-01. Their desktop twins are in tests/test_profile.py.
 */
@Lane(Lane.Kind.FREE)
@RunWith(AndroidJUnit4::class)
class ProfilerRecordTest {

    /** Events of a game the way the pipeline and the annotator emit them; the player White. */
    private class Feed(val store: Store = Store(null)) {
        val profiler = Profiler(store, sessionId = "s", ts = 0.0)
        var gameId = "g"
        var board = Board()
        var ply = 0
        var ts = 0.0

        init {
            send(GameStarted("g", Pieces.Side.WHITE, STARTING_FEN))
        }

        fun send(event: Event): List<Event> {
            ts += 1.0
            return profiler.onEvent(event, ts)
        }

        /** The next game begun, from [fen] ([kind] "position": a game joined from the middle). */
        fun nextGame(id: String, kind: String = "game", fen: String = STARTING_FEN): List<Event> {
            gameId = id
            board = Board(fen)
            ply = 0
            return send(GameStarted(id, Pieces.Side.WHITE, fen, kind))
        }

        fun unjudged(san: String): List<Event> {
            val move = board.parseSan(san)!!
            val before = board.fen()
            val byUser = board.turn == WHITE
            board.push(move)
            ply += 1
            return send(MovePlayed(gameId, ply, san, move.uci(), before, board.fen(), byUser = byUser, thinkMs = 3000))
        }

        /** The engine's verdict on [ply] and its analysis of the position it left. */
        fun judge(ply: Int, uci: String, fenAfter: String, loss: Double = 0.0, label: Label = Label.GOOD, afterTags: List<MotifTag> = emptyList()): List<Event> {
            val severity = when {
                loss >= 0.28 -> Severity.BLUNDER
                loss >= 0.10 -> Severity.MISTAKE
                loss >= 0.04 -> Severity.INACCURACY
                else -> Severity.NONE
            }
            val got = send(
                MoveAssessment(
                    gameId, ply, uci, loss, severity,
                    best = if (loss > 0) listOf("g1f3") else listOf(uci),
                    depth = 14, label = label, accuracy = maxOf(0.0, 100.0 - loss * 200),
                ),
            )
            return got + send(
                Analysis(
                    gameId, fenAfter, ply,
                    lines = listOf(Line(moves = emptyList(), depth = 14, expectedScore = 0.5)),
                    decisive = false, onlyMove = false, tags = afterTags,
                ),
            )
        }

        fun play(san: String, loss: Double = 0.0, label: Label = Label.GOOD, afterTags: List<MotifTag> = emptyList()): List<Event> {
            val got = unjudged(san)
            return got + judge(ply, board.moveStack.last().uci(), board.fen(), loss, label, afterTags)
        }

        fun end(result: String = "1-0"): List<Event> = send(GameEnded(gameId, result, "resignation", "1. e4", board.fen()))
    }

    /**
     * The last clock reading of a game carried into the next, so a mistake in an untimed game
     * after a blitz game finished on 4 seconds was blamed on the clock. A reading belongs to the
     * game it was taken in. `test_a_clock_read_in_one_game_never_reaches_the_next`.
     */
    @Test
    fun aClockReadInOneGameNeverReachesTheNext() {
        val f = Feed()
        f.send(ClockObservation(180_000, 180_000, Pieces.Side.WHITE))
        for (san in listOf("e4", "e5", "Nf3", "Nc6")) f.play(san)
        f.send(ClockObservation(4_000, 30_000, Pieces.Side.WHITE))
        f.end("0-1")
        // An untimed bot game next: no clock on the screen, so no reading.
        f.nextGame("bot")
        f.play("e4")
        f.play("e5")
        val got = f.play("Qh5", 0.12, Label.MISTAKE, listOf(FORK))
        val evidence = got.filterIsInstance<ProfileEvidence>()
        assertTrue(evidence.isNotEmpty())
        assertNotEquals(Causes.TIME_PRESSURE, evidence[0].cause)
        assertEquals(setOf<Int?>(null), f.store.moves("bot").map { it.clockMs }.toSet())
    }

    /**
     * A reading with one side's box unreadable is no reading of that side: its clock stays as
     * last read. `test_a_clock_box_not_read_keeps_its_last_reading`.
     */
    @Test
    fun aClockBoxNotReadKeepsItsLastReading() {
        val f = Feed()
        f.send(ClockObservation(6_000, 200_000, Pieces.Side.WHITE))
        f.play("e4")
        f.send(ClockObservation(null, 190_000, Pieces.Side.BLACK))
        f.play("e5")
        val got = f.play("Qh5", 0.12, Label.MISTAKE, listOf(FORK))
        assertEquals(listOf<Int?>(6_000, 190_000, 6_000), f.store.moves("g").map { it.clockMs })
        assertEquals(Causes.TIME_PRESSURE, got.filterIsInstance<ProfileEvidence>()[0].cause)
    }

    /**
     * A game removed from the player's record right after it ended still took rows -- the
     * engine's late verdict as evidence, the coach's words, the review -- and the profile's cells
     * counted the late mistake. Once removed, nothing more is kept.
     * `test_a_game_removed_right_after_it_ends_keeps_nothing_more`.
     */
    @Test
    fun aGameRemovedRightAfterItEndsKeepsNothingMore() {
        val f = Feed()
        f.play("e4")
        f.play("e5")
        f.unjudged("Qh5") // the engine trails: its verdict comes after the end
        val move = f.board.moveStack.last().uci()
        val after = f.board.fen()
        f.end("1-0")
        assertEquals(listOf("watched:g"), f.store.removeGame("watched:g"))
        val got = f.judge(3, move, after, 0.12, Label.MISTAKE, listOf(FORK)) +
            f.send(CoachMessage("You missed a chance there.", DisclosureLevel.IDEA, Trigger.POST_MISTAKE, ply = 2)) +
            f.profiler.tick(f.ts + 1000.0)
        assertEquals("no review, no evidence for a game that is gone", emptyList<Event>(), got)
        for (table in listOf("games", "moves", "evidence", "dialogue", "recommendations")) {
            assertEquals(table, 0, f.store.query("SELECT * FROM $table WHERE game_id = 'g'").second.size)
        }
        assertEquals(0, f.store.cells().sumOf { it.errors })
        assertNull("nor in the session's summary", f.profiler.endSession(f.ts + 2000.0))
    }

    /**
     * A watched game's time control was its first clock reading as read -- 599000/597500, read
     * a second or two in, was "9.98333 min": a ten-minute game counted as blitz -- and a game
     * joined from the middle took the time left for its control. The control is the reading to
     * the whole minute; a joined game's clocks give none.
     * `test_the_time_control_is_the_clock_as_the_game_began`.
     */
    @Test
    fun theTimeControlIsTheClockAsTheGameBegan() {
        val f = Feed()
        f.send(ClockObservation(599_000, 597_500, Pieces.Side.WHITE))
        assertEquals("10 min", f.store.game("g")!!.timeControl)
        assertEquals("rapid", Library.timeClassOf("10 min"))
        assertEquals("as records kept before say it", "rapid", Library.timeClassOf("9.98333 min"))
        assertEquals("bullet", Library.timeClassOf("2.5 min"))
        assertEquals("2.5 min", Profiler.timeControlWords(150_000, null))
        f.end()
        // A game joined from the middle: its clocks say how much time was left, not the control.
        f.nextGame("j", kind = "position", fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
        f.send(ClockObservation(271_000, 244_000, Pieces.Side.WHITE))
        assertNull(f.store.game("j")!!.timeControl)
    }

    /**
     * The analyser tells a past position's final word after later verdicts (the audit of
     * 2026-10-05): the game's opening is the latest ply's, whatever lands last, and the tactics a
     * position offered reach the move played from it when its analysis comes after that move's
     * verdict. `test_the_engines_words_may_come_in_any_order`.
     */
    @Test
    fun theEnginesWordsMayComeInAnyOrder() {
        val f = Feed()
        fun verdict(san: String): String {
            f.unjudged(san)
            val uci = f.board.moveStack.last().uci()
            f.send(MoveAssessment(f.gameId, f.ply, uci, 0.0, Severity.NONE, best = listOf(uci), depth = 14, label = Label.GOOD, accuracy = 100.0))
            return f.board.fen()
        }
        fun told(ply: Int, fen: String, opening: String, tags: List<MotifTag>) {
            f.send(Analysis(f.gameId, fen, ply, lines = emptyList(), decisive = false, onlyMove = false, tags = tags, opening = opening))
        }
        val afterE4 = verdict("e4")
        val afterE5 = verdict("e5") // judged before the position it was played from is told
        told(2, afterE5, "C20 King's Pawn Game", emptyList())
        told(1, afterE4, "B00 King's Pawn", listOf(FORK)) // the earlier ply's, last
        f.end()
        val moves = f.store.moves("g")
        assertEquals(listOf("e4", "e5"), moves.map { it.san })
        assertEquals("the tactic e5 was played past", listOf("fork"), moves[1].availableTags)
        assertEquals("C20 King's Pawn Game", f.store.game("g")!!.opening)
    }

    /**
     * The opening is the latest position's name -- of the moves that stand: a line taken back and
     * played otherwise is not named by the moves withdrawn (the audit of 2026-10-05, round 2: the
     * withdrawn line's later name stood over the new line's).
     * `test_a_takeback_takes_the_withdrawn_lines_opening_with_it`.
     */
    @Test
    fun aTakebackTakesTheWithdrawnLinesOpeningWithIt() {
        val f = Feed()
        fun told(opening: String) {
            f.send(Analysis(f.gameId, f.board.fen(), f.board.moveStack.size, lines = emptyList(), decisive = false, onlyMove = false, opening = opening))
        }
        f.unjudged("e4")
        told("B00 King's Pawn")
        f.unjudged("e5")
        told("C20 King's Pawn Game")
        f.unjudged("Nf3")
        told("C40 King's Knight Opening")
        f.board.pop()
        f.send(Takeback(f.gameId, 1, f.board.fen()))
        f.end()
        assertEquals("C20 King's Pawn Game", f.store.game("g")!!.opening)
    }

    /**
     * A record kept by an older build (`user_version` 0): its reviews under way measured their
     * positions the old way, and go -- a review goes on from nothing rather than mix the two --
     * and nothing else does; opened again, what this build keeps stays.
     * `test_a_record_from_before_the_one_line_engine_drops_only_its_reviews_under_way`.
     */
    @Test
    fun aRecordFromBeforeTheOneLineEngineDropsOnlyItsReviewsUnderWay() {
        val context = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().targetContext
        val file = java.io.File(context.cacheDir, "upgrade-test.sqlite").apply { delete() }
        try {
            var store = Store(file)
            store.startGame("g", 1.0, "white", STARTING_FEN)
            store.addProgress("k", "standard", 18, 6, STARTING_FEN, "{}")
            store.close()
            android.database.sqlite.SQLiteDatabase.openDatabase(file.path, null, 0).use { it.version = 0 }
            store = Store(file)
            assertEquals(emptyList<String>(), store.progress("k", "standard", 18, 6))
            assertTrue(store.game("g") != null)
            store.addProgress("k", "standard", 18, 6, STARTING_FEN, "{}")
            store.close()
            store = Store(file)
            assertEquals(listOf("{}"), store.progress("k", "standard", 18, 6))
            store.close()
        } finally {
            file.delete()
        }
    }

    /**
     * `test_a_board_replaced_after_a_few_moves_was_no_game`: Tai, 2026-10-05: chess.com's
     * puzzles page showed a start position, the board was followed as a game, and the puzzle that
     * replaced it ended "a game" of one move -- reviewed. Replaced before GAME_MIN_PLIES with no
     * result seen, it was no game: no record, no review. A short game whose result was read (a
     * fool's mate) and one that reached GAME_MIN_PLIES are games.
     */
    @Test
    fun aBoardReplacedAfterAFewMovesWasNoGame() {
        val f = Feed()
        val out = ArrayList<Event>()
        out += f.play("e4")
        out += f.send(GameEnded(f.gameId, "*", "board reoriented", "1. e4", f.board.fen()))
        assertNull(f.store.game("g"))
        assertTrue(out.none { it is com.sidecoach.events.GameReview })

        f.nextGame("fools")
        for (san in listOf("f3", "e5", "g4", "Qh4#")) f.play(san)
        f.send(GameEnded(f.gameId, "0-1", "checkmate", "1. f3 e5 2. g4 Qh4#", f.board.fen()))
        assertTrue("a mate read is a game, however short", f.store.game("fools") != null)

        f.nextGame("six")
        for (san in listOf("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5").take(Profiler.GAME_MIN_PLIES)) f.play(san)
        f.send(GameEnded(f.gameId, "*", "new game started", "", f.board.fen()))
        assertTrue("GAME_MIN_PLIES half-moves is a game", f.store.game("six") != null)
    }

    private companion object {
        val FORK = MotifTag("fork", listOf("e5", "c6", "g6"))
    }
}
