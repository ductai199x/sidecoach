package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.analysis.PositionalFact
import com.sidecoach.analysis.Traps
import com.sidecoach.analysis.atHome
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.parseSquare
import com.sidecoach.events.Analysis
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * `tests/test_situation.py` (desktop fc30fa7), case for case: the first hint's parts
 * ([Situation], [Traps]), held to the moments of the master games it was read against
 * (Capablanca-Marshall, Johner-Nimzowitsch) and to the Lichess puzzles that caught its first
 * slips. Then the verdict changes of `tests/test_verdicts.py` (a centre pawn's lesson only with
 * what it opens) and desktop 3895242's Lichess vgdbX (their threat to the square the best move
 * takes, said by their piece), read from the coach fixture's engine analysis.
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

    @Test
    fun theRookTheLineWinsLaterIsNotYet() {
        // 16.Re2: the rook on a8 falls later; 16.Qxa8? Nxh3+ mates.
        val board = Board("r1b2rk1/2p2ppp/p2b4/1p6/3P3q/1BP2Q1P/PP3nP1/RNB1R1K1 w - - 0 16")
        val trap = line(listOf("f3a8", "f2h3", "g2h3", "h4e1", "g1g2", "e1e2", "g2g1", "e2h2"), 0.0, mate = -8)
        assertEquals(
            "their rook on a8 can be won, but not at once: Qxa8 right away runs into Nxh3+, and they " +
                "have mate",
            Situation.trapWords(board, trap, parseSquare("a8"), 0.68),
        )
    }

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

    @Test
    fun materialGivenForTheirAttackIsNotWon() {
        // 29.exf4 in a lost game: the knight Black gives with ...Ng4+ is not "won back".
        val board = Board("6rk/3b1prq/1p1p1n2/p1pP1n1p/P1P1pp2/2P1P1PP/R6K/2BQ1NRB w - - 0 29")
        val moves = listOf("e3f4", "f6g4", "h3g4", "h5h4", "h1e4", "h4g3", "h2g2", "f5h4", "g2h3", "d7g4")
        val analysis = Analysis(
            gameId = "",
            fen = board.fen(),
            ply = 0,
            lines = listOf(line(moves, 0.055)),
            decisive = false,
            onlyMove = false,
        )
        assertNotEquals("win", IdeaReader.read(board, analysis).goal)
    }

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

    @Test
    fun theKingsFaceToFaceInAPawnEnding() {
        val board = Board("8/8/5k2/8/4K3/5P2/8/8 b - - 5 51")  // Lichess R8gGn, ...Ke6
        assertEquals(
            "The kings stand close, yours on f6 and theirs on e4: whichever has to give way loses ground",
            Situation.kingsKey(board, Move.fromUci("f6e6")),
        )
    }

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
        assertEquals("Nc3: your knight goes from 2 squares to 5", Situation.reachWhy(board, Move.fromUci("b1c3")))
    }

    @Test
    fun thePatternToKeepIsNotSaidTwice() {
        // 11.f4 in Johner-Nimzowitsch: the reason said it already.
        val text = "e4 was better -- their pawn on e5 wants to come forward: it is worth stopping before it comes."
        assertTrue(said("their idea is worth stopping before it starts", text))
        assertFalse(said("more space gives your pieces room and cramps theirs", text))
    }

    /** `tests/test_verdicts.py`'s change: a pawn to the centre is a lesson only with what it opens. */
    @Test
    fun aCentrePawnsLessonIsWhatItOpens() {
        var board = Board(STARTING_FEN)
        assertEquals("centre" to listOf("that lets your bishop out"), lesson(board, Move.fromUci("e2e4"), emptyList<MotifTag>(), "opening"))
        board = Board("rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2")
        val said = lesson(board, Move.fromUci("c2c4"), emptyList<MotifTag>(), "opening")
        assertTrue(said == null || said.first != "centre")
    }

    /**
     * Desktop 3895242, Lichess vgdbX: were White to pass, ...e6 would come, and then exd5 wins the
     * queen -- and e6 is White's own answer. Said by their piece, never as the move.
     */
    @Test
    fun theirThreatToTheSquareTheBestMoveTakesIsSaidByTheirPiece() {
        val context = InstrumentationRegistry.getInstrumentation().context
        val text = context.assets.open("chess/coach.json").bufferedReader().use { it.readText() }
        val cases = JSONObject(text).getJSONArray("cases")
        var found: JSONObject? = null
        for (i in 0 until cases.length()) {
            if (cases.getJSONObject(i).getString("source") == "cache: vgdbX") found = cases.getJSONObject(i)
        }
        assertNotNull("the vgdbX case is in the coach fixture", found)
        val case = found!!
        val fen = case.getString("fen")
        assertEquals("1rbqkb1r/3npppp/2B5/p2QP3/8/5N2/PPP2nPP/R1B1K2R w KQk - 0 12", fen)
        val o = case.getJSONObject("analysis")
        val analysis = Analysis(
            gameId = "g",
            fen = fen,
            ply = 0,
            lines = lines(o.getJSONArray("lines")),
            decisive = o.getBoolean("decisive"),
            onlyMove = o.getBoolean("only_move"),
            threats = (0 until o.getJSONArray("threats").length()).map { o.getJSONArray("threats").getString(it) },
            traps = lines(o.getJSONArray("traps")),
            defences = lines(o.getJSONArray("defences")),
            opening = if (o.isNull("opening")) null else o.getString("opening"),
            inBook = o.getBoolean("in_book"),
        )
        val board = Board(fen)
        assertEquals("e5e6", analysis.lines[0].moves[0])
        val facts = factsFromAnalysis(board, analysis, lastMove = "e4", inBook = false, user = board.turn)
        assertEquals("Their pawn on e7 is one move from winning material", facts.key)
        assertEquals(case.getJSONObject("facts").getString("key"), facts.key)
        assertFalse(facts.key!!.contains("e6"))  // the key names their piece, never the answer
    }

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

    private fun lines(a: JSONArray): List<Line> = (0 until a.length()).map { k ->
        val l = a.getJSONObject(k)
        Line(
            moves = (0 until l.getJSONArray("moves").length()).map { l.getJSONArray("moves").getString(it) },
            depth = l.getInt("depth"),
            expectedScore = l.getDouble("expected_score"),
            scoreCp = if (l.isNull("score_cp")) null else l.getInt("score_cp"),
            mate = if (l.isNull("mate")) null else l.getInt("mate"),
        )
    }
}
