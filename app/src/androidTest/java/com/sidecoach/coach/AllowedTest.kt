package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.chess.Board
import com.sidecoach.events.Analysis
import com.sidecoach.events.Line
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * `tests/test_allowed.py` (desktop 25a3333), case for case: what a mistake allowed (Tai's own
 * games, 2026-09-24) -- the lesson after a mistake says what the move let the opponent do, their
 * best line in the position it left, read by the same idea reader -- and a win far down a line
 * is not read out move by move.
 */
@RunWith(AndroidJUnit4::class)
class AllowedTest {

    private fun analysis(fen: String, sans: String, score: Double): Analysis {
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
            lines = listOf(Line(moves = moves, depth = 20, expectedScore = score)),
            decisive = false,
            onlyMove = false,
        )
    }

    @Test
    fun aBlunderThatDropsARookSaysSo() {
        // g02, 13.Qxd5??: the lesson named Rxa6 and never the rook Rxa1+ took.
        val fen = "1n1qk2r/2pp1pp1/r7/3Qp2p/8/2B2P2/1PP3PP/R3K1NR b KQk - 0 13"
        val a = analysis(fen, "Rxa1+ Kf2 Qf6 Bxe5 Qf5 Qe4 Qxe4 fxe4", score = 0.92)
        assertEquals("after Qxd5, Rxa1+ wins a rook", IdeaWords.allowed(Board(fen), a, "Qxd5"))
    }

    @Test
    fun aMoveThatLetsThemTradeIntoAWonEndingSaysSo() {
        // g05, 18.Qd4: the queens come off and the ending is theirs.
        val fen = "r1b3k1/ppp2ppp/3b4/8/3Qq3/P4N2/1BP2PPP/R5K1 b - - 1 18"
        val a = analysis(fen, "Qxd4 Bxd4 Bf5 c3 b6 Be5 Rd8 Rd1", score = 0.82)
        val text = IdeaWords.allowed(Board(fen), a, "Qd4")
        assertNotNull(text)
        assertTrue(text, text!!.startsWith("after Qd4, Qxd4 trades them into a won"))
        assertFalse("mixed" in text)
    }

    @Test
    fun aQuietLineAllowsNothingToSay() {
        val fen = "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1"
        assertNull(IdeaWords.allowed(Board(fen), analysis(fen, "Kd7 Kd2 Kd6 Kd3", score = 0.5), "e2"))
    }

    @Test
    fun theLessonSaysWhatTheMoveAllowed() {
        val fen = "r2qk2r/p1p1bppp/2p5/4p3/2b1P3/2N4P/PPP2PP1/R1BQR1K1 w kq - 0 12"
        val board = Board(fen)
        val facts = factsFromAnalysis(board, analysis(fen, "Qg4 g6 b3 Be6", score = 0.6))
        val text = lessonTemplate(
            facts,
            "Qg4",
            "That one cost you: Qg4 was better",
            allowed = "after Be3, Qxd1 wins your queen",
        )
        assertTrue(text, "After Be3, Qxd1 wins your queen." in text)
    }

    @Test
    fun aWinFarDownTheLineIsNotReadOutMoveByMove() {
        // g01, 7.Be3 for 7.h4: the knight falls on the seventh move of the line.
        val fen = "rnbqkb1r/pppp4/4p1p1/5n2/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 7"
        val board = Board(fen)
        val line = "h4 d5 g4 Nxh4 Bg5 Be7 Bxh4 Bxh4 g5 Nc6 Rxh4 Rxh4 Nxh4 Qxg5"
        val idea = IdeaReader.read(board, analysis(fen, line, score = 0.9))
        assertEquals("win", idea.goal)
        assertTrue(idea.gainPly > IdeaWords.LATER_READ)
        assertEquals(
            "h4: you win ${idea.won}, a few moves on",
            IdeaWords.why(idea, board, IdeaWords.Describe(::describeTactic)),
        )
    }

    @Test
    fun theFocusRungNamesNoStrayFeature() {
        // g06, before 9.O-O: "your king is still in the centre and can still castle" when a3 was
        // the move and castling the inaccuracy.
        val fen = "rnbqkb1r/pppp4/4p1p1/5n2/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 7"
        val facts = factsFromAnalysis(Board(fen), analysis(fen, "a3 d5", score = 0.5))
            .copy(target = null, plan = null, ideas = listOf("a stray feature elsewhere"))
        assertFalse("stray" in focusTemplate(facts))
    }
}
