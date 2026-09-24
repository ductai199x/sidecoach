package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.analysis.bestMoveIdea
import com.sidecoach.analysis.isCentralBreak
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.parseSquare
import com.sidecoach.events.Analysis
import com.sidecoach.events.Line
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * `tests/test_rungs.py` (desktop 268990e), case for case: the rungs say only what is so (the
 * master games' notes read beside the three rungs, and Tai's own games read move by move,
 * 2026-09-24): two knights are "your knights", a threat to meet keeps its own reason, a pawn
 * attacked gives way to the move's own idea, the focus rung points at the plan, a win that
 * comes a move later is read as one, the right capture with the other piece is taught as that,
 * and a long mate is read only as far as an ear can follow.
 */
@RunWith(AndroidJUnit4::class)
class RungsTest {

    private fun analysis(
        fen: String,
        sans: String,
        threats: List<String> = emptyList(),
        score: Double = 0.5,
        mate: Int? = null,
    ): Analysis {
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
            threats = threats,
            decisive = false,
            onlyMove = false,
        )
    }

    private val describe = IdeaWords.Describe(::describeTactic)

    @Test
    fun twoKnightsAreYourKnights() {
        val plan = LinePlan(
            "attack",
            square = parseSquare("g8"),
            pieces = listOf(parseSquare("c3"), parseSquare("f3")),
            sans = listOf("Ne4", "Ng5"),
        )
        val board = Board("6k1/5ppp/8/8/8/2N2N2/5PPP/6K1 w - - 0 1")
        assertEquals("Ne4 and Ng5 bring your knights against their king", Plans.why(plan, board))
    }

    @Test
    fun aThreatToMeetKeepsItsOwnReason() {
        // Short-Timman: Nd4 meets Rd1-Rxd8; that it also puts a knight in the centre is not why.
        val fen = "r1bq1rk1/1pp1ppbp/1nn3p1/p3P3/P7/1B3N1P/1PP1QPP1/RNB2RK1 b - - 0 11"
        val facts = factsFromAnalysis(Board(fen), analysis(fen, "Nd4 Nxd4 Qxd4 Re1", threats = listOf("f1d1")))
        assertNotNull(facts.goal)
        assertTrue(facts.goal!!.startsWith("your opponent is preparing"))
        assertNull(facts.plan)
    }

    @Test
    fun aPawnAttackedGivesWayToTheMovesOwnIdea() {
        // d4 strikes at the pawn on e5; the pawn ending's point is the passed pawn it makes.
        val fen = "4k3/8/8/4p3/8/8/3P4/4K3 w - - 0 1"
        val facts = factsFromAnalysis(Board(fen), analysis(fen, "d4 Kd7"))
        assertNotEquals(Plans.THREAT_NAME.getValue("win"), facts.plan)
        assertNotNull(facts.plan)
    }

    @Test
    fun theFocusRungPointsAtThePlanNotAtAStrayFeature() {
        val fen = "4k3/8/8/4p3/8/8/3P4/4K3 w - - 0 1"
        val facts = factsFromAnalysis(Board(fen), analysis(fen, "d4 Kd7"))
            .copy(target = null, ideas = listOf("a stray feature elsewhere"))
        assertFalse("stray" in focusTemplate(facts))
    }

    @Test
    fun aCaptureAwayFromTheCentreIsNotTheFightForIt() {
        var board = Board("rnbqkbnr/ppppp1pp/8/5p2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        assertFalse(isCentralBreak(board, Move.fromUci("e4f5")))
        board = Board("4k3/8/8/4p3/5P2/8/8/4K3 w - - 0 1")
        assertTrue(isCentralBreak(board, Move.fromUci("f4e5")))
    }

    @Test
    fun aPromotionIsPraisedPlainly() {
        // c8=Q was "prophylaxis: stop their plan before pushing your own" (Tai's game g02).
        val board = Board("8/2P5/8/8/8/k7/8/K7 w - - 0 1")
        assertNull(bestMoveIdea(board, Move.fromUci("c7c8q"), WHITE, phase = "endgame"))
    }

    @Test
    fun aWinThatComesAMoveLaterIsReadAsOne() {
        // Tai's game g04, 12.Nxd4 for 12.O-O: castling first, and the pawn falls after it.
        val fen = "r1bq1rk1/2pnbppp/p2p4/1p1P4/3pP3/1B3P2/PPP1N1PP/R1BQK2R w KQ - 1 12"
        val board = Board(fen)
        val idea = IdeaReader.read(board, analysis(fen, "O-O Nc5 Nxd4 Bd7 c3 Nxb3 axb3 a5", score = 0.67))
        assertEquals("win", idea.goal)
        assertEquals(2, idea.gainPly)
        assertEquals("O-O, and after Nc5, Nxd4; you win a pawn", IdeaWords.why(idea, board, describe))
        assertEquals("What has to happen first, before you can take it?", IdeaWords.nudge(idea))
    }

    @Test
    fun theRightCaptureWithTheOtherPieceIsTaughtAsThat() {
        // Tai's game g03, 13.Raxd1 for 13.Rexd1: he took the queen back, with the other rook.
        val fen = "r3k2r/p1p1bppp/2p5/4p3/2b1P3/2N1B2P/PPP2PP1/R2qR1K1 w kq - 0 13"
        val board = Board(fen)
        val facts = factsFromAnalysis(board, analysis(fen, "Rexd1 Bb4 Na4 Be2", score = 0.54))
        val text = lessonTemplate(
            facts, "Rexd1", "That one cost you: Rexd1 was better", board = board, played = "a1d1",
        )
        assertEquals(
            "That one cost you: Rexd1 was better -- take with the rook on e1, not the one on a1. " +
                "Keep this one: when two pieces can take, ask which you want to keep where it is.",
            text,
        )
    }

    @Test
    fun aLessonSaysItOfTheBetterMove() {
        // Tai's game g03, 12.Be3 for 12.Qg4: "Qg4 was better -- goes after..." lacked its "it".
        val fen = "r2qk2r/p1p1bppp/2p5/4p3/2b1P3/2N4P/PPP2PP1/R1BQR1K1 w kq - 0 12"
        val board = Board(fen)
        val facts = factsFromAnalysis(board, analysis(fen, "Qg4 g6 b3 Be6", score = 0.6))
        val text = lessonTemplate(facts, "Qg4", "That one cost you: Qg4 was better")
        assertTrue(text, text.startsWith("That one cost you: Qg4 was better -- it goes after their pawn on g7"))
    }

    @Test
    fun aLongMateIsReadAsFarAsAnEarCanFollow() {
        // Tai's game g02, 53.Kc5 for 53.b5: a mate in twelve, sixteen plies of it.
        val fen = "8/k6n/2K3NP/8/1P4P1/2P5/8/8 w - - 1 53"
        val board = Board(fen)
        val line = "b5 Ka8 Ne5 Nf8 b6 Ne6 b7+ Ka7 b8=Q+ Kxb8 h7 Nd8+ Kb6 Nf7 Nxf7 Kc8"
        val idea = IdeaReader.read(board, analysis(fen, line, score = 1.0, mate = 12))
        assertEquals("b5 forces mate: b5 Ka8 Ne5 Nf8 b6, and mate follows", IdeaWords.why(idea, board, describe))
    }

    @Test
    fun aForcingCapturePointsAtWhatItTakes() {
        // Tai's game g04, 8.d5 for 8.dxe5: the focus was "your king is still in the centre".
        val fen = "r1bqk2r/2p1bppp/p1np1n2/1p2p3/3PP3/1BN2N2/PPP2PPP/R1BQK2R w KQkq - 0 8"
        val board = Board(fen)
        val facts = factsFromAnalysis(board, analysis(fen, "dxe5 dxe5 Qxd8+ Bxd8", score = 0.53))
        assertEquals("their pawn on e5", facts.target)
    }
}
