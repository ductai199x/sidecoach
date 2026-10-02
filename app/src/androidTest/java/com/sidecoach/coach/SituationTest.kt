package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.analysis.PositionalFact
import com.sidecoach.analysis.Traps
import com.sidecoach.analysis.atHome
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.parseSquare
import com.sidecoach.events.Line
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * `tests/test_situation.py` (desktop fc30fa7), case for case: the first hint's parts
 * ([Situation], [Traps]), held to the moments of the master games it was read against
 * (Capablanca-Marshall, Johner-Nimzowitsch) and to the Lichess puzzles that caught its first
 * slips (one of them, the material given for their attack, is `tests/test_idea.py`'s now).
 * Then the verdict changes of `tests/test_verdicts.py` (a centre pawn's lesson only with what it
 * opens: `test_the_lesson_is_what_the_board_teaches_not_what_the_move_did`) and desktop
 * 3895242's Lichess vgdbX (their threat to the square the best move takes, said by their piece:
 * `test_their_threat_to_the_square_the_best_move_takes_is_said_by_their_piece`), read from the
 * coach fixture's engine analysis.
 */
@RunWith(AndroidJUnit4::class)
class SituationTest {

    private fun line(moves: List<String>, score: Double, mate: Int? = null): Line =
        Line(moves = moves, depth = 18, expectedScore = score, mate = mate)

    private fun moves(vararg uci: String): List<Move> = uci.map { Move.fromUci(it) }

    @Test
    fun theBaitKnightIsATrap() {
        // Capablanca-Marshall, 14.Qf3: "14 hxg4? Qh4" and Black's attack comes.
        val board = Board("r1bq1rk1/2p2ppp/p2b4/1p6/6n1/1BP4P/PP1P1PP1/RNBQR1K1 w - - 1 14")
        val trap = line(listOf("h3g4", "d8h4", "d1f3", "h4h2", "g1f1", "h2h1", "f1e2", "f8e8"), 0.45)
        val said = Situation.trapWords(board, trap, null, 0.60)
        assertNotNull(said)
        assertTrue(said, said!!.startsWith("careful, though: hxg4 looks like it wins a knight, but after hxg4 Qh4"))
    }

    // theRookTheLineWinsLaterIsNotYet: coach.json's trap_tests 'on the target, into mate, the worst': held by StoryTest#trapsAreRankedAndSaidAsTheDesktopSaysThem.

    @Test
    fun aCaptureThatComesOutAheadIsNoTrap() {
        // 19...Bxf2 Qxf2 wins the exchange: only worse than 19...Qh1+, in a game lost either way.
        val board = Board("r4rk1/2p2ppp/p7/1p6/3P2Pq/1BP2Qb1/PP3RP1/RNB2K2 b - - 0 19")
        val trap = line(listOf("g3f2", "f3f2", "h4h1", "f2g1", "h1h4", "b1d2", "a8e8", "d2f3"), 0.052)
        assertNull(Situation.trapWords(board, trap, null, 0.25))
    }

    @Test
    fun aCaptureIntoMateIsATrapInALostGame() {
        // Johner-Nimzowitsch, 34.Bxe4: "34 Kxh3 allows mate after 34...Qf5+", 0.06 below the best.
        val board = Board("6rk/5prq/1p1p1n2/p1pP3p/P1P1pP1n/2P1R1Pb/2Q4K/2B2NRB w - - 0 34")
        val trap = line(listOf("h2h3", "h7f5", "g3g4", "g7g4", "g1g4", "f5g4", "h3h2", "g4g1"), 0.0, mate = -9)
        assertTrue(Traps.worse(trap, line(listOf("h1e4"), 0.06)))
        assertFalse(Traps.worse(line(listOf("h1e4"), 0.04), line(listOf("h1e4"), 0.06)))
        assertEquals(
            "careful, though: Kxh3 looks like it wins a bishop, but after Kxh3 Qf5+, they have mate",
            Situation.trapWords(board, trap, null, 0.06),
        )
    }

    // `test_material_given_for_their_attack_is_not_won`, in tests/test_idea.py since the consolidations of 2026-09-28/29.
    // materialGivenForTheirAttackIsNotWon: coach.json's test_situation case 6rk/3b1prq: idea.goal held by CoachPrimitivesTest.

    @Test
    fun oneGoodMoveIsSaidInTheSingular() {
        // "Only 1 moves keep you in the game" (30.Qb3 in Johner-Nimzowitsch).
        val stakes = Situation.Stakes("few", 1, 0.5, line(listOf("a2a3"), 0.40), 20)
        val said = Situation.stakesWords(Board(), stakes, null)
        assertNotNull(said)
        assertTrue(said, said!!.startsWith("Only one move keeps the balance here"))
    }

    @Test
    fun theirIdeaIsSaidOnlyWhenItChangesSomething() {
        val board = Board()
        assertNull(Situation.theirIdeaKey(board, Move.fromUci("g8f6")))
        assertEquals("If you let them, their best is Nf6", Situation.theirIdeaKey(board, Move.fromUci("g8f6"), sharp = true))
        assertEquals("If you let them, their best is e5", Situation.theirIdeaKey(board, Move.fromUci("e7e5")))
    }

    @Test
    fun aFeatureIsThePointOnlyOfTheMoveThatActsOnIt() {
        val board = Board()
        val home = PositionalFact("undeveloped", "your bishop on c1 is still at home, not yet in the game", listOf("c1"))
        // 15.Bd2 in Johner-Nimzowitsch: a4 was best, Bd2 only later in its line.
        assertNull(Situation.relevantFeature(board, listOf(home), moves("a2a4", "e7e5", "c1d2")))
        assertEquals(home.text, Situation.relevantFeature(board, listOf(home), moves("c1d2")))
        // 13...Ng4 in Capablanca-Marshall: the queen's file is not the point of Qd7.
        val asset = PositionalFact("halfOpenFile", "your queen on d8 has the half-open d-file", listOf("d8"))
        assertNull(Situation.relevantFeature(board, listOf(asset), moves("d8d7")))
    }

    @Test
    fun aKnightBackOnF1IsNotAtHome() {
        // 39.Ne3 in Johner-Nimzowitsch: "2 pieces on their home squares (c1, f1)".
        val board = Board("6k1/8/8/8/8/8/8/2B2NK1 w - - 0 40")
        assertEquals(listOf(parseSquare("c1")), atHome(board, WHITE))
    }

    // theKingsFaceToFaceInAPawnEnding: coach.json's case 'lichess R8gGn': facts.key held by CoachPrimitivesTest.

    @Test
    fun theirPasserNearQueeningIsThePointOfTakingIt() {
        val board = Board("8/P3R3/6p1/5p2/5P2/1P2p1k1/r7/3K4 w - - 0 41")  // Lichess UkpFL, Rxe3+
        assertEquals("Their passed pawn on e3 is two squares from queening", Situation.passerKey(board, Move.fromUci("e7e3")))
    }

    @Test
    fun aQuietMoveSaysWhatItIsFor() {
        val board = Board()
        assertEquals(
            "Which pawn move covers e5?" to "f4: the pawn now covers e5",
            Situation.centreMove(board, Move.fromUci("f2f4")),
        )
        assertEquals("Nc3: your knight goes from two squares to five", Situation.reachWhy(board, Move.fromUci("b1c3")))
    }

    @Test
    fun thePatternToKeepIsNotSaidTwice() {
        // 11.f4 in Johner-Nimzowitsch: the reason said it already.
        val text = "e4 was better -- their pawn on e5 wants to come forward: it is worth stopping before it comes."
        assertTrue(said("their idea is worth stopping before it starts", text))
        assertFalse(said("more space gives your pieces room and cramps theirs", text))
    }

    // `tests/test_verdicts.py`'s change: a pawn to the centre is a lesson only with what it opens.
    // aCentrePawnsLessonIsWhatItOpens: the same two lesson() calls and asserts as VerdictsTest#theLessonIsWhatTheBoardTeachesNotWhatTheMoveDid.

    // theirThreatToTheSquareTheBestMoveTakesIsSaidByTheirPiece: coach.json's case 'cache: vgdbX' (desktop 3895242): facts.key held by CoachPrimitivesTest on the whole case.

    @Test
    fun theFirstHintIsSaidTheWayACoachSpeaks() {
        // Tai, 2026-09-25: "make it sound a bit better and more natural". Contractions, except
        // where the verb ends its clause.
        assertEquals(
            "It's Boden's mate. Their pawn can't take back, it isn't defended, and you're " +
                "winning -- leave the knight where it is.",
            Situation.spoken(
                "It is Boden's mate. Their pawn cannot take back, it is not defended, and you are " +
                    "winning -- leave the knight where it is.",
            ),
        )
        // A trap said first has nothing to turn from; a target named already leaves the order.
        assertEquals(
            "careful: Nxe5 looks like it wins a pawn",
            Situation.firstSaid("careful, though: Nxe5 looks like it wins a pawn"),
        )
        assertEquals(
            "careful with the order, though: Qxb8 right away runs into Qd2+",
            Situation.inOrder("their rook on b8 can be won, but not at once: Qxb8 right away runs into Qd2+"),
        )
    }

    /**
     * Ten sentences when every detail stacked up (Anderssen-Kieseritzky, move 20): past the budget
     * the second pattern, the quiet start, the in-between move and the timing go, the least
     * telling first; the prize, the key fact, the pattern, a sacrifice and the question stay
     * (desktop c08a1bc).
     */
    @Test
    fun aLongFirstHintDropsTheDetailsFirst() {
        val text = "You're in check but winning. Their queen on a1 gives check. This position has a " +
            "tactic that wins a bishop. The prize is their bishop on c8. Their rook on a8 and " +
            "their king on e8: one move can hit both after your first move. That is a fork: one " +
            "piece attacks two of theirs at once. It comes on your second move, once a line has " +
            "been cleared. There is a discovered attack in it too. It starts with a quiet move: " +
            "no check and no capture, but a threat they cannot meet. Which two of their pieces " +
            "could one of yours hit at once?"
        val short = Situation.withinBudget(text, budget = 8)
        assertFalse(short, "in it too" in short || "quiet move" in short)
        assertTrue("dropped only as far as the budget", "It comes on your second move" in short)
        assertTrue(short, "The prize is their bishop on c8." in short && short.endsWith("hit at once?"))
        val shortest = Situation.withinBudget(text, budget = 5)
        assertTrue(shortest, "It comes on your" !in shortest && "The prize is" in shortest)
        val sacrifice = "That is a fork: two at once. It comes on your third move, after a sacrifice."
        assertEquals("a sacrifice is said", sacrifice, Situation.withinBudget(sacrifice, budget = 1))
        assertEquals("One. Two.", Situation.withinBudget("One. Two.", budget = 5))
    }

    /** `test_the_stakes_in_check_say_what_the_lines_bear_out` (the facts audit, 2026-10-01). */
    @Test
    fun theStakesInCheckSayWhatTheLinesBearOut() {
        // In check and lost: every way out is "within a hair" because all of them lose.
        val board = Board("4k3/8/8/8/8/8/3q4/4K3 w - - 0 1")
        var said = Situation.stakesWords(board, Situation.Stakes("check", 3, 0.02, null, 2), null)
        assertTrue(said, said != null && "fine" !in said && "losing" in said)
        // One way out best, the next only 0.09 behind: not "only one way out holds".
        said = Situation.stakesWords(board, Situation.Stakes("check", 1, 0.93, line(listOf("e1f1"), 0.84), 2), null)
        assertTrue(said, said != null && "only one" !in said)
        said = Situation.stakesWords(board, Situation.Stakes("check", 1, 0.6, line(listOf("e1f1"), 0.52), 2), null)
        assertTrue(said, said != null && "only one" !in said)
        // Winning, the next still winning: "only one move keeps it that way" is false.
        assertNull(Situation.stakesWords(board, Situation.Stakes("won", 1, 0.99, line(listOf("e1f1"), 0.82), 20), null))
    }

    /** `test_the_plan_words_hold_on_the_board_now` (the facts audit, 2026-10-01). */
    @Test
    fun thePlanWordsHoldOnTheBoardNow() {
        fun key(fen: String, uci: String, kind: String): String {
            val said = Situation.positionalKey(Board(fen), Move.fromUci(uci), kind, null, "endgame")
            assertNotNull(said)
            return said!!
        }
        // Rec2 with a rook already on c7: "none of your rooks is there yet" was false.
        var said = key("r4k2/2Rn1p2/p2r2p1/3p4/1P3P1p/3B4/P3RKPP/8 w - - 0 28", "e2c2", "takeFile")
        assertTrue(said, "none of your rooks" !in said)
        // Behind their passed pawn on g7, the rook holds it back; it pushes nothing.
        said = key("6k1/R5P1/1p6/3pK3/P2P2r1/8/8/8 b - - 0 46", "g4g1", "rookBehindPasser")
        assertTrue(said, "their passed pawn on g7" in said && "pushes" !in said)
        // A king on e1 has no wing: not "the other wing from yours".
        val fen = "r1bq1rk1/ppp1npbp/3pn1p1/4p3/4P2P/2PPNN2/PPB2PP1/R1BQK2R w KQ - 1 12"
        said = key(fen, "h4h5", "pawnStorm")
        assertTrue(said, "other wing" !in said)
        val storm = Situation.planKey(Board(fen), LinePlan("storm", square = parseSquare("g8")))
        assertTrue(storm, storm != null && "opposite wings" !in storm)
        // b6 hits c5 now: c5 is no square "no pawn of theirs can ever attack" yet.
        val outpost = Situation.planKey(Board("4k3/8/1p6/8/3P4/1N6/8/4K3 w - - 0 1"), LinePlan("outpost", square = parseSquare("c5")))
        assertTrue(outpost, outpost != null && "can ever attack" !in outpost)
    }
}
