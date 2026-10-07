package com.sidecoach.state

import com.sidecoach.chess.BLACK
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.PgnReader
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.parseSquare
import com.sidecoach.events.Event
import com.sidecoach.events.GameEnded
import com.sidecoach.events.GameStarted
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.Resync
import com.sidecoach.events.Takeback
import com.sidecoach.events.TurnPending
import com.sidecoach.vision.Pieces.Side
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * `tests/test_reconcile_record.py`, case for case: the reconciler's record and what its listener
 * is told, at the level of its stories, which no frame can stage on cue. The rendered cases
 * (`test_reconcile.py`) are `ReconcileOpsTest`'s; the game replaced with its mate pending is
 * `PipelinePartsTest.aGameReplacedEndsWithTheMateItsStoriesLeave`, the pipeline needing the net.
 */
class ReconcileRecordTest {

    /**
     * What the coach holds of the side to move (`Coach.on_event`): pending as the last
     * TurnPending said, and known again from any event that tells a position whole.
     */
    private class Listener {
        var pending = false

        fun hear(events: List<Event>) {
            for (event in events) {
                if (event is GameStarted || event is Takeback || event is Resync || event is GameEnded) pending = false
                if (event is TurnPending) pending = event.pending
            }
        }
    }

    private fun played(vararg sans: String): Board {
        val board = Board()
        for (san in sans) board.push(board.parseSan(san)!!)
        return board
    }

    private fun reconciler(vararg sans: String): Reconciler {
        val rec = Reconciler(bottom = Side.WHITE, userSide = Side.WHITE)
        rec.start()
        rec.board = played(*sans)
        return rec
    }

    /** `test_a_takeback_played_as_the_stories_part_tells_the_turn_pending_again`. */
    @Test
    fun aTakebackPlayedAsTheStoriesPartTellsTheTurnPendingAgain() {
        val rec = reconciler("e4", "e5")
        val listener = Listener()
        val back = listOf<HeadOp>(HeadOp.Back(1))
        val d5 = Move.fromUci("d7d5")
        val nf3 = Move.fromUci("g1f3")
        rec.stories = listOf(Story(head = back, moves = listOf(d5)), Story(head = back, moves = listOf(d5, nf3)))
        listener.hear(rec.tellTurn())
        assertTrue(listener.pending && rec.turnPending)
        val stories = rec.stories.toList()
        val seen = LinkedHashMap<List<Any>, Int>()
        for (s in stories) for (i in s.ops.indices) seen[s.ops.take(i + 1)] = 5
        rec.seen = seen
        val events = rec.decide(stories)
        listener.hear(events)
        assertTrue(events.any { it is Takeback })
        assertEquals(listOf("d5"), events.filterIsInstance<MovePlayed>().map { it.san })
        assertTrue("the stories still disagree after the common moves", rec.turnPending)
        assertEquals(events.map { it::class.simpleName }.toString(), rec.turnPending, listener.pending)
    }

    /** `test_the_clock_settling_a_turn_tells_it_pending_again_while_the_stories_disagree`. */
    @Test
    fun theClockSettlingATurnTellsItPendingAgainWhileTheStoriesDisagree() {
        val rec = Reconciler(bottom = Side.WHITE, userSide = Side.WHITE)
        rec.startFrom(Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"), ts = 0.0)
        assertTrue(rec.turnUnknown && rec.board.turn == WHITE)
        val listener = Listener()
        val king = Move.fromUci("e8d8")
        val pawn = Move.fromUci("e2e4")
        rec.stories = listOf(Story(moves = listOf(pawn)), Story(moves = listOf(pawn, king)))
        listener.hear(rec.tellTurn())
        assertTrue(listener.pending)
        val events = rec.turnFromClock(BLACK)
        listener.hear(events)
        assertTrue(events.any { it is Resync })
        assertEquals(rec.turnPending, listener.pending)
    }

    /** `test_a_position_set_forgets_what_was_remembered_of_the_last`, both cases. */
    @Test
    fun aPositionSetForgetsWhatWasRememberedOfTheLast() {
        for (how in listOf("new game", "position set")) {
            val rec = Reconciler(bottom = Side.WHITE, userSide = Side.WHITE)
            rec.startFrom(Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"), ts = 0.0)
            assertTrue(rec.turnUnknown)
            rec.turnPendingTold = true
            rec.queueTs = 1.0
            rec.resyncCandidate = Triple("x", 2, 0.5)
            rec.resyncSeen = listOf(Reconciler.Seen(Board(), emptySet(), "x", 0.5))
            rec.unmarked = 2
            rec.unmarkedView = "x" to emptyList()
            rec.markDenied = true
            rec.prevObs = Observation().also { it[4] = blank() }
            if (how == "new game") rec.start() else rec.setPosition(Board("4k3/8/8/8/8/4P3/8/4K3 b - - 0 1"))
            val fresh = Reconciler(bottom = Side.WHITE, userSide = Side.WHITE)
            assertEquals(how, fresh.turnPendingTold, rec.turnPendingTold)
            assertEquals(how, fresh.queueTs, rec.queueTs)
            assertEquals(how, fresh.resyncCandidate, rec.resyncCandidate)
            assertEquals(how, fresh.resyncSeen, rec.resyncSeen)
            assertEquals(how, fresh.unmarked, rec.unmarked)
            assertEquals(how, fresh.unmarkedView, rec.unmarkedView)
            assertEquals(how, fresh.markDenied, rec.markDenied)
            assertEquals(how, fresh.prevObs, rec.prevObs)
            assertFalse(how, rec.turnUnknown)
        }
    }

    private fun blank() = com.sidecoach.vision.Pieces.CellReading(kind = com.sidecoach.vision.Pieces.Kind.EMPTY)

    /** `test_a_board_read_whole_is_the_game_when_it_shows_its_position`. */
    @Test
    fun aBoardReadWholeIsTheGameWhenItShowsItsPosition() {
        val rec = reconciler("e4", "e5", "Nf3")
        val shown = rec.board.copy(withStack = false)
        val marked = setOf(parseSquare("g1"), parseSquare("f3"))
        assertTrue(rec.showsTheGame(shown, marked))
        assertTrue("no mark: the turn the stories have", rec.showsTheGame(shown))
        assertFalse("Black just moved", rec.showsTheGame(shown, setOf(parseSquare("e7"), parseSquare("e5"))))
        val other = rec.board.copy()
        other.push(other.parseSan("Nc6")!!)
        assertFalse(rec.showsTheGame(other.copy(withStack = false)))
        rec.end("1-0", "resignation")
        assertFalse("a game over is followed no more", rec.showsTheGame(shown, marked))
    }

    /** `test_the_record_of_a_position_is_a_game_from_it`. */
    @Test
    fun theRecordOfAPositionIsAGameFromIt() {
        val rec = Reconciler(bottom = Side.WHITE, userSide = Side.WHITE)
        val start = Board("4k3/8/8/8/8/8/4P3/r3K3 w - - 0 1") // in check: White to move
        rec.startFrom(start, ts = 0.0)
        assertFalse(rec.turnUnknown)
        rec.commit(Move.fromUci("e1d2"))
        rec.commit(Move.fromUci("e8d8"))
        val game = PgnReader(rec.pgn()).readGame()
        assertNotNull(game)
        assertTrue(game!!.errors.toString(), game.errors.isEmpty())
        assertEquals(start.fen(), game.board().fen())
        val end = game.board()
        for (move in game.mainlineMoves()) end.push(move)
        assertEquals(rec.board.fen(), end.fen())
        assertEquals("a game from the start is its moves alone", "", reconciler().pgn())
    }
}
