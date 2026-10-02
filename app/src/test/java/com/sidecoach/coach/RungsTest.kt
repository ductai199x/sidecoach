package com.sidecoach.coach

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
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The cases of `tests/test_rungs.py` as desktop 268990e wrote them: the rungs say only what is so
 * (the master games' notes read beside the three rungs, and Tai's own games read move by move,
 * 2026-09-24): two knights are "your knights", a threat to meet keeps its own reason, a pawn
 * attacked gives way to the move's own idea, the focus rung points at the plan, a win that
 * comes a move later is read as one, the right capture with the other piece is taught as that,
 * and a long mate is read only as far as an ear can follow. The desktop's consolidations of
 * 2026-09-28/29 moved some of them to other files: each test names its twin as it stands now.
 */
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

    /** `test_two_knights_are_your_knights`. */
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

    /**
     * Now `test_a_quiet_move_that_prepares_to_win_the_queen_is_a_threat_to_meet` in
     * tests/test_threats_to_meet.py.
     */
    @Test
    fun aThreatToMeetKeepsItsOwnReason() {
        // Short-Timman: Nd4 meets Rd1-Rxd8; that it also puts a knight in the centre is not why.
        val fen = "r1bq1rk1/1pp1ppbp/1nn3p1/p3P3/P7/1B3N1P/1PP1QPP1/RNB2RK1 b - - 0 11"
        val facts = factsFromAnalysis(Board(fen), analysis(fen, "Nd4 Nxd4 Qxd4 Re1", threats = listOf("f1d1")))
        assertNotNull(facts.goal)
        assertTrue(facts.goal!!.startsWith("your opponent is preparing"))
        assertNull(facts.plan)
    }

    // The first half of `test_the_move_s_own_idea_leads_and_the_focus_rung_names_no_stray_feature`.
    // aPawnAttackedGivesWayToTheMovesOwnIdea: coach.json's test_idea case 4k3/8/8/4p3/8/8/3P4/4K3: facts.plan held by CoachPrimitivesTest.

    /** The second half of `test_the_move_s_own_idea_leads_and_the_focus_rung_names_no_stray_feature`. */
    @Test
    fun theFocusRungPointsAtThePlanNotAtAStrayFeature() {
        val fen = "4k3/8/8/4p3/8/8/3P4/4K3 w - - 0 1"
        val facts = factsFromAnalysis(Board(fen), analysis(fen, "d4 Kd7"))
            .copy(target = null, ideas = listOf("a stray feature elsewhere"))
        assertFalse("stray" in focusTemplate(facts))
    }

    /** Now `test_a_capture_away_from_the_centre_is_not_the_fight_for_it` in tests/test_positional.py. */
    @Test
    fun aCaptureAwayFromTheCentreIsNotTheFightForIt() {
        var board = Board("rnbqkbnr/ppppp1pp/8/5p2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        assertFalse(isCentralBreak(board, Move.fromUci("e4f5")))
        board = Board("4k3/8/8/4p3/5P2/8/8/4K3 w - - 0 1")
        assertTrue(isCentralBreak(board, Move.fromUci("f4e5")))
    }

    /** Now the "a promotion" case of `test_the_best_move_s_idea` in tests/test_positional.py. */
    @Test
    fun aPromotionIsPraisedPlainly() {
        // c8=Q was "prophylaxis: stop their plan before pushing your own" (Tai's game g02).
        val board = Board("8/2P5/8/8/8/k7/8/K7 w - - 0 1")
        assertNull(bestMoveIdea(board, Move.fromUci("c7c8q"), WHITE, phase = "endgame"))
    }

    // The "a win a move later" case of `test_a_line_is_read_as_far_as_an_ear_can_follow`.
    // aWinThatComesAMoveLaterIsReadAsOne: coach.json's test_idea case r1bq1rk1/2pnbppp (g04): goal, gain_ply, why and nudge held by CoachPrimitivesTest.

    // `test_the_right_capture_with_the_other_piece_is_taught_as_that`.
    // theRightCaptureWithTheOtherPieceIsTaughtAsThat: coach.json's test_idea case r3k2r/p1p1bppp (g03): its lessons_played text held by CoachPrimitivesTest.

    // Now `test_the_lesson_says_what_the_move_allowed` in tests/test_allowed.py.
    // aLessonSaysItOfTheBetterMove: coach.json's test_idea case r2qk2r/...R1BQR1K1: its lessons text held by CoachPrimitivesTest.

    // The "a long mate" case of `test_a_line_is_read_as_far_as_an_ear_can_follow`.
    // aLongMateIsReadAsFarAsAnEarCanFollow: coach.json's test_idea case 8/k6n/2K3NP (mate in 12): words.why held by CoachPrimitivesTest.

    // `test_a_forcing_capture_points_at_what_it_takes`.
    // aForcingCapturePointsAtWhatItTakes: coach.json's test_idea case r1bqk2r/2p1bppp: facts.target held by CoachPrimitivesTest.

}
