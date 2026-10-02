package com.sidecoach.state

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.chess.Board
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import org.junit.runner.RunWith

/**
 * A game the next one replaced gets the result its final position proves ([resultOnBoard];
 * Tai, 2026-10-02): `test_the_final_position_proves_a_result_only_when_the_game_is_over_on_it`.
 * The rendered case, a repetition played and then a new game, is the desktop's
 * `test_a_game_replaced_by_the_next_has_the_result_its_final_position_proves`, which
 * ReconcileOpsTest plays from reconcile_ops.json.
 */
@RunWith(AndroidJUnit4::class)
class ReplacedGameResultTest {

    private fun played(vararg sans: String): Board {
        val board = Board()
        for (san in sans) board.push(board.parseSan(san)!!)
        return board
    }

    @Test
    fun theFinalPositionProvesAResultOnlyWhenTheGameIsOverOnIt() {
        assertEquals("0-1" to "checkmate", resultOnBoard(played("f3", "e5", "g4", "Qh4#")))
        assertEquals("1/2-1/2" to "stalemate", resultOnBoard(Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")))
        assertEquals("1/2-1/2" to "insufficient material", resultOnBoard(Board("8/8/4k3/8/8/2K5/8/8 w - - 0 60")))
        assertEquals("1/2-1/2" to "fifty-move rule", resultOnBoard(Board("8/8/4k3/8/r7/2K5/8/R7 w - - 100 80")))
        assertNull("one half-move short", resultOnBoard(Board("8/8/4k3/8/r7/2K5/8/R7 w - - 99 80")))
        assertNull("the start position twice: a third is only possible", resultOnBoard(played("Nf3", "Nf6", "Ng1", "Ng8")))
        assertEquals(
            "1/2-1/2" to "repetition",
            resultOnBoard(played("e4", "e5", "Nf3", "Nc6", "Ng1", "Nb8", "Nf3", "Nc6", "Ng1", "Nb8")),
        )
        assertNull(resultOnBoard(Board()))
    }

    @Test
    fun aReplacedGameEndsWithTheProvedResultOrUnknown() {
        val drawn = Reconciler(bottom = com.sidecoach.vision.Pieces.Side.WHITE, userSide = com.sidecoach.vision.Pieces.Side.WHITE)
        drawn.board = played("e4", "e5", "Nf3", "Nc6", "Ng1", "Nb8", "Nf3", "Nc6", "Ng1", "Nb8")
        val ended = drawn.replacedEvent("new game started")
        assertEquals("1/2-1/2" to "repetition", ended.result to ended.reason)
        val open = Reconciler(bottom = com.sidecoach.vision.Pieces.Side.WHITE, userSide = com.sidecoach.vision.Pieces.Side.WHITE)
        open.board = played("e4", "e5", "Nf3")
        val unknown = open.replacedEvent("new game started")
        assertEquals("*" to "new game started", unknown.result to unknown.reason)
    }

    /**
     * `test_a_game_whose_end_was_never_seen_ends_at_the_session_by_its_final_position`: a
     * position that proves nothing ends nothing (the record keeps "*"); the third repetition
     * played is a draw, once.
     */
    @Test
    fun aGameWhoseEndWasNeverSeenEndsAtTheSessionByItsFinalPosition() {
        val rec = Reconciler(bottom = com.sidecoach.vision.Pieces.Side.WHITE, userSide = com.sidecoach.vision.Pieces.Side.WHITE)
        rec.start()
        rec.board = played("e4", "e5", "Nf3", "Nc6", "Ng1", "Nb8")
        assertEquals(emptyList<com.sidecoach.events.Event>(), rec.endUnseen())
        assertEquals(false, rec.finished)
        rec.board = played("e4", "e5", "Nf3", "Nc6", "Ng1", "Nb8", "Nf3", "Nc6", "Ng1", "Nb8")
        val ended = rec.endUnseen().filterIsInstance<com.sidecoach.events.GameEnded>()
        assertEquals(listOf("1/2-1/2" to "repetition"), ended.map { it.result to it.reason })
        assertEquals("1/2-1/2", rec.result)
        assertEquals(emptyList<com.sidecoach.events.Event>(), rec.endUnseen())
    }
}
