package com.sidecoach.coach

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

/**
 * `tests/test_idea.py`'s cases of desktop c08a1bc: a tactic whose own move takes a piece the line
 * nets, worth more than anything it aims at, is not the blow ([IdeaReader.captureIsThePoint]).
 */
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
     * The desktop's test_a_named_defender_is_asked_of_the_target_whatever_its_wording, at the
     * words: with the defender named beside it, the loose piece is said plainly whenever the
     * blow comes -- Facts cut "nothing defends it" out with a pattern that missed "after your
     * next few moves" (the round 3 audit, 2026-10-02). Facts asks for these words.
     */
    @Test
    fun aNamedDefenderIsAskedOfTheTargetWhateverItsWording() {
        val fen = "8/2p5/1p2B3/p7/5r2/P4p2/1PP1N2k/1K4R1 b - - 2 59"
        // The line carried on quietly, long enough for a seventh move of the player's.
        val read = read(fen, "fxe2 Re1 Rf1 Kc1 Rxe1+ Kd2 a4 b3 b5 b4 c5 c3 c4 Kc2")
        assertEquals(
            "their rook on g1: after your next few moves, nothing defends it",
            IdeaWords.target(read.copy(blowAt = 6), Board(fen)),
        )
        for (blowAt in 1..7) {
            val idea = read.copy(blowAt = blowAt)
            val said = IdeaWords.target(idea, Board(fen))
            assertTrue(said, said != null && said.endsWith(", nothing defends it"))
            assertEquals("their rook on g1", IdeaWords.target(idea, Board(fen), defenderNamed = true))
        }
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

    /**
     * `test_a_sound_sacrifice_in_a_worse_position_is_one`: Bxh7+ Kxh7 Ng5+ Kg8 Qh5 a little worse
     * (0.47), nothing back in the line: giving the bishop costs less than an inaccuracy against the
     * even material the player had, a Good move's loss, so it is sacrificed (Tai, 2026-10-02).
     */
    @Test
    fun aSoundSacrificeInAWorsePositionIsOne() {
        val fen = "r1bq1rk1/pppn1ppp/4p3/3pP3/1b1P4/2NB1N2/PPP2PPP/R2QK2R w KQ - 0 8"
        val idea = read(fen, "Bxh7+ Kxh7 Ng5+ Kg8 Qh5", score = 0.47)
        assertEquals(listOf(0 to "bishop"), idea.sacrifices)
    }

    /**
     * `test_a_piece_another_move_could_defend_is_sacrificed_not_lost_anyway`: the knight on a4 is
     * attacked and every square it can go to is covered, but Ra1 or Rd4 would defend it: h3 Rxa4
     * gives it, a sacrifice, not a piece lost whatever it did.
     */
    @Test
    fun aPieceAnotherMoveCouldDefendIsSacrificedNotLostAnyway() {
        val fen = "r6k/2p3pp/3p4/4b3/N7/8/5PPP/3R2K1 w - - 0 30"
        val idea = read(fen, "h3 Rxa4", score = 0.6)
        assertEquals(listOf(0 to "knight"), idea.sacrifices)
    }

    /**
     * `test_a_bishop_given_in_a_lost_position_for_nothing_is_no_sacrifice`: six points down, ...Bg4
     * gives the bishop and the line is worth what nine points down is: the gift costs more than an
     * inaccuracy against the six-down count, below a Good move -- given up, not sacrificed.
     */
    @Test
    fun aBishopGivenInALostPositionForNothingIsNoSacrifice() {
        val fen = "r1b3k1/ppp3pp/5r2/n7/8/5B2/PPP1NbPP/R2Q1R1K b - - 1 15"
        val idea = read(fen, "Bg4 Bxg4 Kh8 Bh3 Nc6 a3 Bb6 Rxf6 gxf6 Qf1 f5 Qxf5", score = 0.0352)
        assertTrue(idea.sacrifices.toString(), idea.sacrifices.isEmpty())
    }
}
