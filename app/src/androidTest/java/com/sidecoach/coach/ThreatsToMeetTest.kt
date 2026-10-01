package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.analysis.Motifs
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Analysis
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * `tests/test_threats_to_meet.py` (desktop 6637098), case for case: tactics that win something,
 * and threats the opponent prepares (the master games' notes, 2026-09-24). A tactic's capture
 * must leave the player up ([Motifs.paysOff]), and a quiet move of theirs that would threaten
 * mate or a piece next is a threat to meet when the best move takes it away ([IdeaReader]).
 */
@RunWith(AndroidJUnit4::class)
class ThreatsToMeetTest {

    private fun boards(board: Board, line: List<Move>): List<Board> {
        val boards = arrayListOf(board.copy(withStack = false))
        for (move in line) {
            val nxt = boards.last().copy(withStack = false)
            nxt.push(move)
            boards.add(nxt)
        }
        return boards
    }

    private fun line(board: Board, sans: String): List<Move> {
        val b = board.copy(withStack = false)
        val moves = ArrayList<Move>()
        for (san in sans.split(" ")) {
            val move = b.parseSan(san)!!
            moves.add(move)
            b.push(move)
        }
        return moves
    }

    @Test
    fun aLoosePieceTradedOffWinsNothing() {
        // Smyslov-Tal: Na4 opens the long diagonal onto the bishop on g7, but it takes the bishop
        // on b2 first and the knight takes back: the dark-squared bishops traded.
        val board = Board("r1bqk2r/pp1pnpbp/2n1p1p1/2p5/2P5/1PN3P1/PB1PPPBP/R2QK1NR w KQkq - 2 7")
        val line = line(board, "Na4 Bxb2 Nxb2 d5 e3 O-O")
        val tag = MotifTag("hangingPiece", listOf("g7"), "c3a4")
        assertFalse(Motifs.paysOff(boards(board, line), line, 0, tag))
    }

    @Test
    fun aLoosePieceTakenForNothingIsWon() {
        val board = Board("4k3/8/8/3n4/8/8/8/R3K3 w - - 0 1")
        val line = line(board, "Ra5 Ke7 Rxd5")
        val tag = MotifTag("hangingPiece", listOf("d5"), "a1a5")
        assertTrue(Motifs.paysOff(boards(board, line), line, 0, tag))
    }

    @Test
    fun theExchangeOnASquareIsPlayedOutToItsEnd() {
        // Nf2+ uncovers the queen on d1: Qxd1 Rxd1 Nxd1 wins the rook on the third capture
        // (Lichess 01q4A).
        val board = Board("r7/ppk5/2n3pp/2b1pN1q/3pP1n1/1P4NP/1P4P1/R1BQ1R1K b - - 1 22")
        val line = line(board, "Nf2+ Kh2 Qxd1 Rxd1 Nxd1")
        val tag = MotifTag("discoveredAttack", listOf("g4", "h5", "d1"), "g4f2")
        assertTrue(Motifs.paysOff(boards(board, line), line, 0, tag))
    }

    @Test
    fun aDiscoveredAttackOnAPawnIsNotNamed() {
        // The knight steps off the e-file and the rook bears on the pawn on e7: a pawn, not a
        // piece, and "the rook now bears on the pawn" teaches nothing.
        val board = Board("6k1/4p3/8/8/4N3/8/8/4R1K1 w - - 0 1")
        val line = line(board, "Nc5 Kf7 Rxe7+ Kf6")
        val tag = MotifTag("discoveredAttack", listOf("e4", "e1", "e7"), "e4c5")
        assertFalse(Motifs.paysOff(boards(board, line), line, 0, tag))
    }

    private fun analysis(fen: String, sans: String, threats: List<String>): Analysis {
        val board = Board(fen)
        return Analysis(
            gameId = "g",
            fen = fen,
            ply = 0,
            lines = listOf(Line(moves = line(board, sans).map { it.uci() }, depth = 20, expectedScore = 0.39)),
            threats = threats,
            decisive = false,
            onlyMove = false,
        )
    }

    // aQuietMoveThatPreparesToWinTheQueenIsAThreatToMeet: coach.json's test_idea case SHORT_TIMMAN (Nd4): goal, prepared, threatened and the words held by CoachPrimitivesTest.

    // aBestMoveThatLetsThePreparedThreatComeIsNoDefence: coach.json's test_idea case SHORT_TIMMAN (Kh8 Rd1): idea.goal held by CoachPrimitivesTest.

    companion object {
        /**
         * Short-Timman, before 11...Nd4: were Black to pass, Rd1 would come, and then Rxd8 wins
         * the queen ("After anything else, White seizes the d-file by Rd1", Nunn).
         */
        const val SHORT_TIMMAN = "r1bq1rk1/1pp1ppbp/1nn3p1/p3P3/P7/1B3N1P/1PP1QPP1/RNB2RK1 b - - 0 11"
    }
}
