package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.chess.Board
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.events.Analysis
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * `tests/test_idea.py`'s cases of desktop c08a1bc: a tactic whose own move takes a piece the line
 * nets, worth more than anything it aims at, is not the blow ([IdeaReader.captureIsThePoint]).
 */
@RunWith(AndroidJUnit4::class)
class IdeaTest {

    private fun analysis(fen: String, sans: String, score: Double = 0.95, mate: Int? = null): Analysis {
        val board = Board(fen)
        val moves = ArrayList<String>()
        for (san in sans.split(" ")) {
            val move = board.parseSan(san)!!
            moves.add(move.uci())
            board.push(move)
        }
        return Analysis(
            gameId = "g",
            fen = fen,
            ply = 0,
            lines = listOf(Line(moves = moves, depth = 20, expectedScore = score, mate = mate)),
            decisive = true,
            onlyMove = false,
        )
    }

    private fun read(fen: String, line: String, score: Double = 0.95): Idea =
        IdeaReader.read(Board(fen), analysis(fen, line, score))

    /**
     * 35...Nxf5 in Johner-Nimzowitsch: Ng4+ pulls the pinned bishop off the diagonal and Qxc2+
     * takes the queen, forking a bishop, a pawn and the king as it does. "That's a fork" beside a
     * queen won on c2 whose targets were c1, h2 and a4 was false: the pin is it.
     */
    @Test
    fun aQueenWonByTakingItIsNotWonByTheForkTheCaptureMakes() {
        val idea = read(JOHNER, "Ng4+ Bxg4 Qxc2+ Re2 Qxc1 Ne3", score = 0.96)
        assertEquals("win", idea.goal)
        assertEquals("a queen for a knight", idea.won)
        assertEquals("pin", idea.blow?.name)
    }

    /**
     * The capture is the point only of what the line wins: a queen taken back after they took one
     * is not, and the pin that nets the rook after it stays the blow.
     */
    @Test
    fun aQueenTakenBackLeavesThePinThePoint() {
        val board = Board(JOHNER)
        for (san in listOf("Ng4+", "Bxg4")) board.push(board.parseSan(san)!!)
        val move = board.parseSan("Qxc2+")!!
        val fork = MotifTag(name = "fork", squares = listOf("c2", "c1", "h2", "a4"))
        assertTrue(IdeaReader.captureIsThePoint(board, move, fork, setOf(QUEEN)))
        assertFalse(IdeaReader.captureIsThePoint(board, move, fork, setOf(ROOK)))
        val rook = MotifTag(name = "fork", squares = listOf("c2", "e3", "h2")) // a rook aimed at: worth less
        assertTrue(IdeaReader.captureIsThePoint(board, move, rook, setOf(QUEEN)))
    }

    private companion object {
        const val JOHNER = "6rk/5prq/1p1p1n2/p1pP1B1p/P1P2P1n/2P1R1P1/2Q4K/2B2NR1 b - - 0 35"
    }
}
