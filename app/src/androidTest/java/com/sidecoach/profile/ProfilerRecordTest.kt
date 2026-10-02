package com.sidecoach.profile

import androidx.test.ext.junit.runners.AndroidJUnit4
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

    private companion object {
        val FORK = MotifTag("fork", listOf("e5", "c6", "g6"))
    }
}
