package com.sidecoach.coach

import com.sidecoach.analysis.Computed
import com.sidecoach.analysis.ComputedFirst
import com.sidecoach.chess.Board
import com.sidecoach.chess.G1
import com.sidecoach.chess.Move
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.chess.WHITE
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.Line
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * `tests/test_harness_truth.py` (desktop a050c5a and ce06c4e), case for case: the harness says only what is
 * true of the board -- each cause of a false claim the v2 audit traced to the harness (dev and
 * test reruns, 2026-09-26), on the position it was found in.
 */
class HarnessTruthTest {

    private fun facts(
        fen: String,
        onlyMove: Boolean = false,
        threats: List<String> = emptyList(),
        why: String? = null,
    ): PositionFacts = PositionFacts(
        fen = fen,
        sideToMove = "white",
        phase = "middlegame",
        onlyMove = onlyMove,
        threats = threats,
        why = why,
    )

    private fun pushSan(board: Board, san: String) {
        board.push(board.parseSan(san)!!)
    }

    /** `_NoSearch`: an engine that must not be asked. */
    private val noSearch = EngineLike { _, _ ->
        throw AssertionError("a move the position's own lines play needs no second search")
    }

    @Test
    fun aKingStepClaimsNothingEveryKingMoveHas() {
        // 27...Kh7: every legal king move lands where nothing attacks it, so "a square nothing
        // of theirs can reach" read as the reason for Kh7 over Kf7, Kh5 and Kf5.
        val board = Board("2r5/3r2p1/b1N1R1kp/3n4/5B2/5P2/PP3KPP/2R5 b - - 4 27")
        assertEquals("Kh7 steps out of check", Situation.evasionWhy(board, board.parseSan("Kh7")!!))
    }

    @Test
    fun aKingStepToTheOneSquareWithoutACheckSaysSo() {
        // Kd2 walks into Nb3+; Kc2 leaves them no check at all.
        val board = Board("8/8/6P1/8/8/8/7k/2nK3r w - - 0 1")
        assertEquals(
            "Kc2 steps out of check to the one square where no check follows",
            Situation.evasionWhy(board, board.parseSan("Kc2")!!),
        )
        assertEquals("Kd2 steps out of check", Situation.evasionWhy(board, board.parseSan("Kd2")!!))
    }

    @Test
    fun aMoveAfterAPassIsAThreatOnlyWhenItThreatens() {
        // 39...: White's best free move, Rc3, guards the third rank against the mate; it attacks
        // nothing, and was said as "They threaten Rc3".
        val board = Board("5rk1/B5pp/8/2R5/8/6KP/4q3/8 b - - 7 39")
        val passed = board.copy()
        passed.push(Move.NULL)
        assertFalse(threatens(passed, passed.parseSan("Rc3")!!))
        assertTrue(threatens(passed, passed.parseSan("Rc8")!!)) // attacks the rook on f8
        // A quiet move whose threat is a mate on an empty square still counts.
        val quiet = Board("7k/5Qpp/8/8/8/8/8/6K1 b - - 0 1")
        quiet.push(Move.NULL)
        assertTrue(threatens(quiet, quiet.parseSan("Qe7")!!))
    }

    @Test
    fun inCheckTheThreatIsTheCheckAndWhatElseItWins() {
        // 12.Qxe6+: "Nothing is threatened" with the queen also hitting the loose pawn on c6;
        // the defended bishop on c8 and pawn on d5 are no threat.
        val board = Board("r1bqk2r/p1p4p/1bp1Qpp1/3p4/1P1P1B1P/2P1P3/P4PP1/RN2KB1R b KQkq - 0 12")
        assertEquals(
            "Their queen on e6 gives check, and it also attacks your pawn on c6.",
            inCheckThreat(board),
        )
    }

    @Test
    fun aGreatMoveSaysWhichKindOfGreat() {
        val fen = STARTING_FEN
        assertTrue(greatMeaning(facts(fen, onlyMove = true)).startsWith("the only good move"))
        // Not the only move: the best reply to their error (29...Qg5 after 29.Rc1, a miss),
        // never "the one move that keeps the advantage".
        assertTrue(greatMeaning(facts(fen)).startsWith("the chance taken"))
    }

    @Test
    fun theLessonSaysWhenTheMovePlayedStoppedTheThreatToo() {
        // 19.Qxg2 took the pawn that threatened gxh1=Q+; Qc4+ "stops gxh1=Q+" was taught as
        // "the priority was to stop the promotion threat".
        val faced = Board("1r1q3r/pPp2kbp/6p1/8/1P2QP2/P3P3/6pP/1NB1K2R w K - 2 19")
        val sheet = facts(
            faced.fen(),
            threats = listOf("gxh1=Q+"),
            why = "Qc4+ gives check and stops gxh1=Q+: it is the only move that keeps the balance",
        )
        val played = faced.copy()
        pushSan(played, "Qxg2")
        assertEquals("gxh1=Q+", alsoStopped(sheet, played))
        // A move that left the threat on says nothing of the kind.
        val other = faced.copy()
        pushSan(other, "Qe5")
        assertNull(alsoStopped(sheet, other))
    }

    @Test
    fun aNamedMoveTheLinesPlayIsReadFromThem() {
        // 64.Bd2: engine line 2 had Rxh3, a second search after Bd2 had Kh6 -- and the model,
        // shown both, said Bd2 avoided Rxh3.
        val board = Board("8/7k/6p1/8/4KPP1/4B2P/8/7r w - - 5 64")
        val known = listOf(
            Line(moves = listOf("f4f5", "h1h3", "e3g5"), depth = 20, expectedScore = 0.50),
            Line(moves = listOf("e3d2", "h1h3", "f4f5", "h3g3"), depth = 20, expectedScore = 0.50),
        )
        // The look-ahead's own search after Bd2, which answers it with Kh6: the sheet's line wins.
        val after = board.copy()
        pushSan(after, "Bd2")
        val computed = Computed()
        computed.add(after, listOf(Line(moves = listOf("h7h6", "d2e3"), depth = 18, expectedScore = 0.48)))
        val engine = ComputedFirst(noSearch::analyse, computed)
        val text = examine(
            parseProbe("your Bd2", listOf("Bd2"), board),
            board,
            EngineLike(engine::analyse),
            known[0],
            userColour = WHITE,
            level = DisclosureLevel.LINE,
            known = known,
        )
        assertTrue(text, "the opponent's reply would be 64... Rxh3" in text)
        assertTrue(text, "0.50 from the player's side" in text)
    }

    @Test
    fun aSquareSaysWhetherAPieceOfTheirsCanGetThere() {
        // Tai's puzzle, 3...Rxf1+: "Why can't I block on g1?" -- told only who controls g1, Luna
        // answered that g1 was empty. No piece of White's can go there; after Kg2 two of Black's
        // can.
        val board = Board(RXF1_CHECK)
        val fact = examinePiece(Probe(label = "g1", moves = emptyList(), square = G1), board, userColour = WHITE)
        assertTrue(fact, fact.endsWith("; none of the player's pieces can move there."))
        val said = spoken(fact)
        assertTrue(said, said.endsWith("; none of your pieces can move there."))
        val tools = EngineTools(board, null, noSearch, WHITE)
        val g1 = tools.run("square", JSONObject().put("square", "g1"))
        assertTrue(g1, g1.endsWith("None of the player's pieces can move there."))
        val afterKg2 = tools.run("square", JSONObject().put("square", "g1").put("moves", JSONArray(listOf("Kg2"))))
        assertTrue(afterKg2, afterKg2.endsWith("The opponent can move the rook on f1 or the bishop on a7 there."))
    }

    @Test
    fun aPieceGotOutOfAttackNamesWhatAttackedIt() {
        // The engine's line 4. Kg2 Rg1+ 5. Kf3 g2 6. Nb2: told "gets the knight out of attack",
        // Luna said 5...g2 attacked the knight -- the rook on g1 did.
        val board = Board(RXF1_CHECK)
        val notes = lineNotes(board, listOf("h1g2", "f1g1", "g2f3", "g3g2", "d1b2"), player = WHITE)
        assertTrue(notes, notes.endsWith("the player's Nb2 (gets the knight out of attack from the rook on g1)"))
    }

    @Test
    fun aPassedPawnIsToldToClearItsWayOnlyWhenTheMoveDoes() {
        // 57...Re5: the rook stands behind the a-pawn, White's rook on a3 in front of it; "Which
        // move clears its way?" was taught as "that rook is blocking your passed pawn's route".
        val board = Board("8/8/1K4pp/r7/p5Pk/R6P/8/8 b - - 10 57")
        val line = listOf("a5e5", "b6c7", "e5b5", "a3c3", "a4a3", "c3c1", "a3a2", "c7d6", "a2a1q")
            .map { Move.fromUci(it) }
        val behind = Idea(goal = "promote", first = line[0], line = line)
        assertFalse(IdeaWords.clearsPath(behind, board))
        assertEquals(IdeaWords.PROMOTE_HELP_QUESTION, IdeaWords.question(behind, null, board))
        assertEquals(IdeaWords.PROMOTE_HELP_NUDGE, IdeaWords.nudge(behind, board))
        // A knight that steps off the pawn's square in front of it does clear the way.
        val ahead = Board("4k3/1N6/1P6/8/8/8/8/4K3 w - - 0 1")
        val move = Move.fromUci("b7d6")
        val lane = listOf(move) + listOf("e8e7", "b6b7", "e7d7", "b7b8q").map { Move.fromUci(it) }
        assertTrue(IdeaWords.clearsPath(Idea(goal = "promote", first = move, line = lane), ahead))
    }

    @Test
    fun aKnightIsFreedNeverUncovered() {
        // 25...Ke7 stepped off the d-file, and the knight on d6, pinned there by the rook on d1,
        // could take on e4 again: "uncovers an attack ... by the knight on d6" was said back.
        val board = Board("7r/1p1k2p1/1Ppnbp2/r3p3/NRP1P2p/pK3P2/P3B1PP/3R4 b - - 2 25")
        val effects = describeEffects(board, board.parseSan("Ke7")!!)
        assertTrue("$effects", "frees the knight on d6 to attack the pawn on e4" in effects)
        // A piece that stood in the line does uncover it.
        val line = Board("7r/8/8/8/3N4/8/1B6/K6k w - - 0 1")
        val uncovered = describeEffects(line, line.parseSan("Nb5")!!)
        assertTrue("$uncovered", "uncovers an attack on the rook on h8 by the bishop on b2" in uncovered)
    }

    private companion object {
        const val RXF1_CHECK = "6k1/b1p2pp1/p7/1p6/1P6/P5pP/3R4/3N1r1K w - - 0 4"
    }
}
