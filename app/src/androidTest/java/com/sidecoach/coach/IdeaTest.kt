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

    /**
     * `test_the_target_says_when_the_pattern_comes_about`: the loose piece is loose once the
     * blow is on the board, not now; a fork on the third move comes after the second (the facts
     * audit, 2026-10-01).
     */
    @Test
    fun theTargetSaysWhenThePatternComesAbout() {
        var fen = "8/2p5/1p2B3/p7/5r2/P4p2/1PP1N2k/1K4R1 b - - 2 59"
        var idea = read(fen, "fxe2 Re1 Rf1 Kc1 Rxe1+ Kd2")
        assertEquals("their rook on g1: after your first move, nothing defends it", IdeaWords.target(idea, Board(fen)))
        fen = "2k3nr/ppp2ppp/2n5/8/8/1Q2P3/q4rPP/1R2KB1R w K - 0 16"
        idea = read(fen, "Qxb7+ Kd7 Rd1+ Ke7 Qxc7+ Kf6 Qxc6+ Kg5 h4+ Kh5 g4+ Kxg4 Qe4+ Kh5 Qe5+ f5")
        assertEquals(3, idea.blowAt)
        val said = IdeaWords.target(idea, Board(fen))
        assertTrue(said, said != null && said.endsWith("one move can hit both after your second move"))
    }

    /**
     * `test_a_piece_given_for_nothing_is_no_sacrifice`: in check and lost, Qxf6 Qxf6 gives the
     * queen for a knight and nothing comes back (Fischer-Benko).
     */
    @Test
    fun aPieceGivenForNothingIsNoSacrifice() {
        val fen = "r1b2rk1/p4p1p/4pN1Q/4qp2/1p3p2/1n6/PPP3PP/3R1R1K b - - 0 22"
        val idea = read(fen, "Qxf6 Qxf6 Nc5 Rxf4 Ne4 Rxe4 fxe4 h3 Bb7 Rf1 Ba6 Qg5+ Kh8 Qe5+ Kg8 Rf4", score = 0.07)
        assertTrue(idea.sacrifices.toString(), idea.sacrifices.isEmpty())
        assertFalse("give up" in (IdeaWords.piece(idea, Board(fen)) ?: ""))
    }
}
