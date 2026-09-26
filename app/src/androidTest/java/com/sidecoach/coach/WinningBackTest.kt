package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.chess.Board
import com.sidecoach.chess.KNIGHT
import com.sidecoach.chess.Move
import com.sidecoach.events.Analysis
import com.sidecoach.events.Line
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * `tests/test_winning_back.py` (desktop f513fd5), case for case: taking back is not winning
 * (Tai's puzzles, 2026-09-26: "why is the coaching telling me to win a knight when I have to
 * trade equal value to it"): what the opponent's last capture took counts against the line
 * ([IdeaReader.exchangeOwed]), so a recapture is "you can win back a knight", a net win is named
 * by what it nets, and a lesson never says a move "allowed" a recapture.
 *
 * The positions are Lichess puzzles from the sample (the setup move played on the board, as the
 * coach has it), with the engine's line written out.
 */
@RunWith(AndroidJUnit4::class)
class WinningBackTest {

    private val ruy = listOf("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6")

    /** `_board`: the position with its moves played on it, so the board knows them. */
    private fun boardOf(fen: String?, moves: List<String>): Board {
        val board = if (fen != null) Board(fen) else Board()
        for (uci in moves) board.push(Move.fromUci(uci))
        return board
    }

    private fun analysisOf(board: Board, sans: String, score: Double): Analysis {
        val b = board.copy()
        val moves = ArrayList<String>()
        for (san in sans.split(" ")) {
            val move = b.parseSan(san)!!
            moves.add(move.uci())
            b.push(move)
        }
        return Analysis(
            gameId = "g",
            fen = board.fen(),
            ply = 0,
            lines = listOf(Line(moves = moves, depth = 20, expectedScore = score)),
            decisive = false,
            onlyMove = false,
        )
    }

    private fun goalOf(board: Board, sans: String, score: Double): Pair<Idea, String?> {
        val idea = IdeaReader.read(board, analysisOf(board, sans, score))
        return idea to IdeaWords.goal(idea)
    }

    @Test
    fun theExchangeInProgressIsWhatTheLastCapturesTook() {
        val board = boardOf(null, ruy + "b5c6")  // 4.Bxc6: Black is owed the knight
        assertEquals(mapOf(KNIGHT to 1), IdeaReader.exchangeOwed(board))
        board.push(Move.fromUci("d7c6"))  // 4...dxc6: the trade is done, nobody is owed
        assertEquals(emptyMap<Int, Int>(), IdeaReader.exchangeOwed(board))
        assertEquals(emptyMap<Int, Int>(), IdeaReader.exchangeOwed(boardOf(null, ruy)))  // the last move took nothing
        // A position set up from a FEN has no moves to read: nothing is assumed.
        assertEquals(emptyMap<Int, Int>(), IdeaReader.exchangeOwed(Board(boardOf(null, ruy + "b5c6").fen())))
    }

    @Test
    fun aPlainRecaptureIsWinningBack() {
        val (idea, goal) = goalOf(boardOf(null, ruy + "b5c6"), "dxc6 O-O", score = 0.5)
        assertTrue(idea.back)
        assertEquals("you can win back a bishop", goal)
    }

    @Test
    fun aPuzzleRecaptureIsWinningBack() {
        // Lichess 00tMQ: 20...Nxh5 took the bishop; Qxh5 takes the knight back.
        val knight = boardOf("r5rk/p2nq2p/2pppn1b/1p2p1PB/7P/1PN5/PBPRQ3/2K3R1 b - - 0 20", listOf("f6h5"))
        val (idea, goal) = goalOf(knight, "Qxh5 Bg7", score = 0.7)
        assertTrue(idea.back)
        assertEquals("you can win back a knight", goal)
        // Lichess 00s16: 10...Rxb8 took the bishop; the check wins the one on b4 back.
        val bishop = boardOf("rBbqk2r/1p3ppp/p3pn2/8/1b1P4/2N2N1P/PP3PP1/R2QKB1R b KQkq - 0 10", listOf("a8b8"))
        val (idea2, goal2) = goalOf(bishop, "Qa4+ b5 Qxb4", score = 0.75)
        assertTrue(idea2.back)
        assertEquals("you can win back a bishop", goal2)
    }

    @Test
    fun winningTheirQueenBackAfterTheyTookYours() {
        // Lichess 00ISm: 29...Bxg5 took the queen; Nxg5+ Kh6 Nxe6 takes the bishop, then theirs.
        val board = boardOf("5r2/4bp1k/2ppq1p1/4p1Q1/4N2P/3P4/1P1R1P2/4K1R1 b - - 0 29", listOf("e7g5"))
        val (idea, goal) = goalOf(board, "Nxg5+ Kh6 Nxe6 fxe6", score = 0.6)
        assertTrue(idea.back)
        assertEquals("you can win back their queen", goal)
    }

    @Test
    fun aNetWinIsNamedByWhatItNets() {
        // Lichess 00d6g: 23...Nxe6 took the rook; Bxf5+ g6 Bxe6 takes both knights.
        val board = boardOf("r4n2/1p4pk/2p1R3/p2p1nPp/7P/3P1P1B/PP6/2K4R b - - 0 23", listOf("f8e6"))
        val (idea, goal) = goalOf(board, "Bxf5+ g6 Bxe6 Rf8", score = 0.9)
        assertFalse(idea.back)
        assertEquals("you can win two knights for a rook", goal)
    }

    @Test
    fun aLessonNeverSaysACaptureAllowedTheRecapture() {
        val board = boardOf(null, ruy + "b5c6")  // the player's 4.Bxc6, Black to move
        val analysis = analysisOf(board, "dxc6 O-O", score = 0.5)
        assertNull(IdeaWords.allowed(board, analysis, "Bxc6"))
    }
}
