package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.chess.Board
import com.sidecoach.chess.KNIGHT
import com.sidecoach.chess.Move
import com.sidecoach.events.Analysis
import com.sidecoach.events.GameStarted
import com.sidecoach.events.Line
import com.sidecoach.events.MovePlayed
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.Completion
import com.sidecoach.llm.LLMProvider
import com.sidecoach.vision.Pieces
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * `tests/test_winning_back.py` (desktop f513fd5 and 2820821), case for case: taking back is not
 * winning (Tai's puzzles, 2026-09-26: "why is the coaching telling me to win a knight when I
 * have to trade equal value to it"): what the opponent's last capture took counts against the
 * line ([IdeaReader.exchangeOwed]), so a recapture is "you can win back a knight", a net win is
 * named by what it nets, and a lesson never says a move "allowed" a recapture. The live coach's
 * positions keep the moves that led to them ([boardAfter], [boardBack]), so it sees the same.
 *
 * The positions are Lichess puzzles from the sample (the setup move played on the board, as the
 * coach has it), with the engine's line written out.
 */
@RunWith(AndroidJUnit4::class)
class WinningBackTest {

    private val ruy = listOf("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6")

    /** `ScriptedProvider([])` of `tests/test_coach.py`: no replies, every prompt kept. */
    private class ScriptedProvider : LLMProvider {
        override val name = "scripted"
        val prompts = ArrayList<Pair<String, List<ChatMessage>>>()

        override fun healthcheck(): Pair<Boolean, String> = true to "scripted"

        override fun chat(
            system: String,
            messages: List<ChatMessage>,
            maxTokens: Int,
            temperature: Double,
            thinking: Boolean,
        ): Completion {
            synchronized(prompts) { prompts.add(system to messages.toList()) }
            return Completion(text = "Keep looking.")
        }
    }

    /** `_coach([])` of `tests/test_coach.py`. */
    private fun coachOf(): Coach =
        Coach(ScriptedProvider(), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))

    /** `_moves`: the moves from the start, each as the reconciler hands it over. */
    private fun movesOf(ucis: List<String>): List<MovePlayed> {
        val board = Board()
        val out = ArrayList<MovePlayed>()
        for ((i, uci) in ucis.withIndex()) {
            val ply = i + 1
            val move = Move.fromUci(uci)
            val before = board.fen()
            val san = board.san(move)
            board.push(move)
            out.add(
                MovePlayed(
                    gameId = "g",
                    ply = ply,
                    san = san,
                    uci = uci,
                    fenBefore = before,
                    fenAfter = board.fen(),
                    byUser = ply % 2 == 0,
                ),
            )
        }
        return out
    }

    /** `_board`: the position with its moves played on it, so the board knows them. */
    private fun boardOf(fen: String?, moves: List<String>): Board {
        val board = if (fen != null) Board(fen) else Board()
        for (uci in moves) board.push(Move.fromUci(uci))
        return board
    }

    /** `_analysis`; [ply] stands for the test's `model_copy(update={"ply": ...})`. */
    private fun analysisOf(board: Board, sans: String, score: Double, ply: Int = 0): Analysis {
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
            ply = ply,
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

    /**
     * The coach built each position from the move's FEN, so the exchange in progress was never
     * seen live: the fix reached only the benches (the Android port's review, 2026-09-26).
     */
    @Test
    fun theLiveCoachKeepsTheMovesThatLedToAPosition() {
        val coach = coachOf()
        val start = Board().fen()
        coach.onEvent(GameStarted(gameId = "g", userSide = Pieces.Side.BLACK, startFen = start), 0.0)
        for ((i, event) in movesOf(ruy + "b5c6").withIndex()) coach.onEvent(event, i.toDouble())
        val position = coach.positions.getValue(7)
        assertEquals(ruy + "b5c6", position.board.moveStack.map { it.uci() })
        assertEquals(mapOf(KNIGHT to 1), IdeaReader.exchangeOwed(position.board))
        val analysis = analysisOf(position.board, "dxc6 O-O", score = 0.5, ply = 7)
        coach.onEvent(analysis, 8.0)
        assertTrue(position.facts != null && position.facts!!.goal == "you can win back a bishop")
    }

    @Test
    fun aBoardKeepsItsMovesOnlyWhenTheEventAgreesWithIt() {
        val (first, second) = movesOf(listOf("e2e4", "e7e5"))
        val board = boardAfter(Board(), first)
        assertEquals(listOf("e2e4"), board.moveStack.map { it.uci() })
        // Played from somewhere else than the board: from the position the event gives.
        val elsewhere = boardAfter(Board("8/8/8/8/8/8/8/K6k w - - 0 1"), second)
        assertEquals(listOf("e7e5"), elsewhere.moveStack.map { it.uci() })
        // A FEN the board does not reach (the counters differ): the event's own, no moves.
        val odd = MovePlayed(
            gameId = second.gameId,
            ply = second.ply,
            san = second.san,
            uci = second.uci,
            fenBefore = second.fenBefore,
            fenAfter = second.fenAfter.replace(" 0 2", " 0 9"),
            byUser = second.byUser,
            thinkMs = second.thinkMs,
        )
        assertTrue(boardAfter(board, odd).moveStack.isEmpty() && boardAfter(board, odd).fen() == odd.fenAfter)
    }

    @Test
    fun aTakebackKeepsTheMovesBeforeIt() {
        val board = Board()
        for (uci in ruy) board.push(Move.fromUci(uci))
        val back = boardBack(board, movesOf(ruy)[1].fenAfter)
        assertEquals(ruy.take(2), back.moveStack.map { it.uci() })
        assertTrue(boardBack(board, "8/8/8/8/8/8/8/K6k w - - 0 1").moveStack.isEmpty())
    }
}
