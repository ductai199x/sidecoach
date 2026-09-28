package com.sidecoach.analysis

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.events.Analysis
import com.sidecoach.events.GameStarted
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.Takeback
import com.sidecoach.vision.Pieces
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * `test_a_live_game_stays_in_the_book_past_its_first_move`, on the phone: the live path hands
 * the analyser each move's board, and the book needs the moves that led to it. An Italian
 * Game submitted the way the watcher submits it -- a takeback in the middle -- is book to its
 * last move and named as the whole line names it, not as its first move does.
 */
@RunWith(AndroidJUnit4::class)
class LiveBookTest {

    /** Every position answered with the same quiet line, as the desktop test's FakeEngine. */
    private class QuietEngine : Engine(File("quiet"), null) {
        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> =
            if (board.isGameOver(claimDraw = true)) emptyList()
            else listOf(Line(moves = listOf("a2a3"), depth = 1, expectedScore = 0.5))

        override fun close() = Unit
    }

    @Test
    fun aLiveGameStaysInTheBookPastItsFirstMove() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val folder = File(instrumentation.targetContext.cacheDir, "openings-sample").apply {
            deleteRecursively()
            mkdirs()
        }
        instrumentation.context.assets.open("openings_sample/c.tsv").use { input ->
            File(folder, "c.tsv").outputStream().use { input.copyTo(it) }
        }
        val book = OpeningBook.load(folder)
        assertNotNull(book)
        val analyser = Analyser(GameAnnotator(QuietEngine(), book = book))
        analyser.start()
        val assessing = Assessing(analyser)

        fun play(board: Board, san: String, ply: Int) {
            val before = board.fen()
            val move = board.parseSan(san)!!
            board.push(move)
            assessing.submit(
                MovePlayed(
                    gameId = "g", ply = ply, san = san, uci = move.uci(), fenBefore = before,
                    fenAfter = board.fen(), byUser = ply % 2 == 1,
                ),
            )
        }

        assessing.submit(GameStarted(gameId = "g", userSide = Pieces.Side.WHITE, startFen = STARTING_FEN))
        val board = Board()
        for ((i, san) in "e4 e5 Nf3 Nc6 Bc4 Bc5 d4".split(" ").withIndex()) play(board, san, i + 1)
        // A takeback of 3.d4: the next move is played from the position before it.
        board.pop()
        assessing.submit(Takeback(gameId = "g", plies = 1, fen = board.fen()))
        play(board, "d3", 7)

        val labels = HashMap<Int, Label?>()
        val openings = HashMap<Int, String?>()
        val deadline = System.currentTimeMillis() + 10_000
        // Ply 7 comes twice, d4 then d3: waited for until d3's own name arrives.
        while (System.currentTimeMillis() < deadline && openings[7] != "C50 Italian Game: Giuoco Pianissimo") {
            for (event in assessing.drain()) {
                when (event) {
                    is MoveAssessment -> labels[event.ply] = event.label
                    is Analysis -> openings[event.ply] = event.opening
                    else -> Unit
                }
            }
            Thread.sleep(20)
        }
        analyser.close()
        assertEquals(labels.toString(), List(6) { Label.BOOK }, (1..6).map { labels[it] })
        assertEquals("the move after a takeback is still in the book", Label.BOOK, labels[7])
        assertEquals("B00 King's Pawn Game", openings[1])
        assertEquals("C50 Italian Game", openings[5])
        assertEquals("C50 Italian Game: Giuoco Pianissimo", openings[7])
    }
}
