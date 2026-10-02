package com.sidecoach.analysis

import com.sidecoach.chess.Board
import com.sidecoach.chess.WHITE
import com.sidecoach.coach.EngineLike
import com.sidecoach.coach.EngineTools
import com.sidecoach.events.Line
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * What the engine has computed, read before a search: `tests/test_computed.py`.
 *
 * Tai, 2026-09-29: "our harness must first look at what was already computed, then if that's not
 * computed only then invoke a call to stockfish to ask for analysis over that specific line" --
 * the analyser is the producer, the coach's engine one of its readers, and a move on a line the
 * model was shown is read from that line.
 */
class ComputedTest {

    private val start = Board()
    private val lines = listOf(
        Line(moves = listOf("e2e4", "e7e5", "g1f3", "b8c6"), depth = 18, expectedScore = 0.55),
        Line(moves = listOf("d2d4", "d7d5", "c2c4"), depth = 18, expectedScore = 0.54),
    )

    private fun after(vararg sans: String): Board {
        val board = Board()
        for (san in sans) board.push(board.parseSan(san)!!)
        return board
    }

    /** `_Recording`: the coach's own engine, every position it searches, a line of its own each. */
    private class Recording {
        val seen = ArrayList<String>()

        fun analyse(board: Board, depth: Int?): List<Line> {
            seen.add(board.fen())
            val move = board.legalMoves().minBy { it.uci() }
            return listOf(Line(moves = listOf(move.uci()), depth = depth ?: 14, expectedScore = 0.5))
        }
    }

    /** `FakeEngine`: the lines for a position's FEN, else a quiet line. */
    private open class FakeEngine(private val answers: Map<String, List<Line>> = emptyMap()) : Engine(File("fake"), null) {
        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
            if (com.sidecoach.analysis.gameOver(board)) return emptyList()
            return answers[board.fen()] ?: listOf(Line(moves = listOf("a2a3"), depth = 1, expectedScore = 0.5))
        }

        override fun close() = Unit
    }

    @Test
    fun theRestOfALineIsReadFromTheSideToMoveThere() {
        val line = Line(
            moves = listOf("a", "b", "c", "d"), depth = 18, expectedScore = 0.8, scoreCp = 300,
            wdl = Triple(700, 200, 100),
        )
        val once = lineAfter(line, 1)
        assertEquals(listOf("b", "c", "d"), once.moves)
        assertEquals(17, once.depth)
        assertEquals(0.2, once.expectedScore, 1e-9)
        assertEquals(-300, once.scoreCp)
        assertEquals(Triple(100, 200, 700), once.wdl)
        val twice = lineAfter(line, 2)
        assertEquals(0.8, twice.expectedScore, 0.0)
        assertEquals(300, twice.scoreCp)
        assertEquals(Triple(700, 200, 100), twice.wdl)
    }

    @Test
    fun aMateIsCountedInTheMovesStillToCome() {
        // The mover mates in 2 (A1 B1 A2#): after A1 the side to move is mated in 1; after B1 the
        // mover mates in 1.
        val mating = Line(moves = listOf("a1", "b1", "a2"), depth = 18, expectedScore = 1.0, mate = 2)
        assertEquals(-1, lineAfter(mating, 1).mate)
        assertEquals(1, lineAfter(mating, 2).mate)
        // The mover is mated in 2 (A1 B1 A2 B2#): after A1 the other side mates in 2, after B1 the
        // mover is mated in 1, after A2 the other side mates in 1.
        val mated = Line(moves = listOf("a1", "b1", "a2", "b2"), depth = 18, expectedScore = 0.0, mate = -2)
        assertEquals(listOf(2, -1, 1), listOf(1, 2, 3).map { lineAfter(mated, it).mate })
        // A line carried on past its mate (a slower one found from its end) has no count left.
        val carried = Line(moves = listOf("a1", "b1", "a2", "b2", "a3"), depth = 18, expectedScore = 1.0, mate = 1)
        assertNull(lineAfter(carried, 2).mate)
    }

    @Test
    fun aMoveOnALineIsReadFromIt() {
        val rest = restOfLine(lines, listOf("e2e4", "e7e5"))
        assertEquals(listOf("g1f3", "b8c6"), rest?.moves)
        assertEquals(0.55, rest!!.expectedScore, 0.0)
        assertEquals(lineAfter(lines[1], 1), restOfLine(lines, listOf("d2d4")))
        // Not on a line, a whole line (nothing after it), or nothing asked: none.
        assertNull(restOfLine(lines, listOf("a2a3")))
        assertNull(restOfLine(lines, listOf("d2d4", "d7d5", "c2c4")))
        assertNull(restOfLine(lines, emptyList()))
    }

    @Test
    fun aPositionIsTheSameWhateverItsClocks() {
        val board = Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
        val later = Board("4k3/8/8/8/8/8/8/4K2R w K - 37 80")
        assertEquals(positionKey(board), positionKey(later))
        val computed = Computed()
        computed.add(board, listOf(Line(moves = listOf("h1h8"), depth = 10, expectedScore = 0.9)))
        assertEquals(listOf("h1h8"), computed.lines(later)[0].moves)
    }

    @Test
    fun aPositionMeasuredWholeComesBeforeItsVerdictAndNothingElseIsKnown() {
        val computed = Computed()
        val afterE4 = after("e4")
        computed.add(start, lines)
        // A line passing through a position is no measure of it: a move refuted deeper still read
        // off its line at depth 12 (the code review, 2026-09-29).
        assertEquals(emptyList<Line>(), computed.lines(afterE4))
        val thin = listOf(Line(moves = listOf("c7c5"), depth = 18, expectedScore = 0.47))
        computed.add(afterE4, thin, thin = true)
        assertEquals(thin, computed.lines(afterE4))
        val whole = listOf(
            Line(moves = listOf("e7e5"), depth = 12, expectedScore = 0.46),
            Line(moves = listOf("c7c5"), depth = 12, expectedScore = 0.45),
        )
        computed.add(afterE4, whole)
        assertEquals(whole, computed.lines(afterE4))
        computed.clear()
        assertEquals(emptyList<Line>(), computed.lines(start))
        assertEquals(emptyList<Line>(), computed.lines(afterE4))
    }

    @Test
    fun aShallowerSearchLeavesTheDeeperWordStanding() {
        val computed = Computed()
        computed.add(start, lines)
        computed.add(start, listOf(Line(moves = listOf("g1f3"), depth = 12, expectedScore = 0.52)))
        assertEquals(lines, computed.lines(start))
        val deeper = listOf(Line(moves = listOf("c2c4"), depth = 20, expectedScore = 0.53))
        computed.add(start, deeper)
        assertEquals(deeper, computed.lines(start))
    }

    @Test
    fun theCoachsEngineSearchesOnlyAPositionNotComputedAndNeverOnceClosed() {
        val computed = Computed()
        computed.add(start, lines)
        val searcher = Recording()
        val engine = ComputedFirst(searcher::analyse, computed)
        assertEquals(lines, engine.analyse(start, 14))
        assertEquals(emptyList<String>(), searcher.seen)
        assertEquals(0, engine.searches)
        // Not computed: searched once, and known from then on.
        engine.analyse(after("a3"), 14)
        engine.analyse(after("a3"), 14)
        assertEquals(listOf(after("a3").fen()), searcher.seen)
        assertEquals(1, engine.searches)
        // Closed with a question still being answered: what is known, and no new Stockfish.
        engine.close()
        assertEquals(emptyList<Line>(), engine.analyse(after("h3"), 14))
        assertEquals(1, engine.searches)
        assertEquals(lines, engine.analyse(start, 14))
    }

    /**
     * Tai, 2026-09-29: "it only launches IF the move doesn't exist in the cached line or in the
     * line already computed". The analyser measured the board, the sheet shows its lines, and the
     * model asks the tools.
     */
    @Test
    fun theToolsReadTheSheetsLinesAndSearchOnlyAPositionNothingHas() {
        val annotator = GameAnnotator(FakeEngine(mapOf(start.fen() to lines)), depth = 18, withThreats = false)
        annotator.measure(start)
        val searcher = Recording()
        val engine = ComputedFirst(searcher::analyse, annotator.computed)
        val tools = EngineTools(start, null, EngineLike(engine::analyse), WHITE, knownNow = lines)
        fun analyse(vararg moves: String) = tools.run("analyse", JSONObject().put("moves", JSONArray(moves.toList())))
        val here = analyse()
        assertTrue(here, "The engine's lines, best first:" in here)
        val onLine = analyse("e4")
        assertTrue(onLine, "The rest of the engine's line through it, as given above:" in onLine)
        assertTrue(onLine, "1... e5 2. Nf3 Nc6" in onLine)
        assertTrue(onLine, "The first move given, 1. e4, is the engine's best move" in onLine)
        // A move the second line plays: rated by that line's score, as the sheet shows it.
        val d4 = analyse("d4")
        assertTrue(d4, "0.01 behind the engine's best from the board now, 1. e4." in d4)
        assertEquals("a move on the sheet's lines was searched", emptyList<String>(), searcher.seen)
        // A move no line plays: its position searched once.
        val out = analyse("a3")
        assertEquals(listOf(after("a3").fen()), searcher.seen)
        assertTrue(out, "The first move given, 1. a3, is" in out && "behind the engine's best" in out)
        analyse("a3")
        assertEquals(1, searcher.seen.size)
    }

    /**
     * The look-ahead measures the position after each of the two best moves on its own, and its
     * search can answer the move differently from the sheet's line through it: the model, shown
     * both, believed each in turn (Bd2 answered with Kh6 where the lines had Rxh3, the dev audit,
     * 2026-09-26). A move on the sheet's line is read from the line.
     */
    @Test
    fun theSheetsLineWinsOverASeparateSearchOfThePositionAfterIt() {
        val computed = Computed()
        computed.add(start, lines)
        computed.add(after("e4"), listOf(Line(moves = listOf("c7c5", "g1f3"), depth = 18, expectedScore = 0.44)))
        val engine = ComputedFirst(Recording()::analyse, computed)
        val tools = EngineTools(start, null, EngineLike(engine::analyse), WHITE, knownNow = lines)
        val out = tools.run("analyse", JSONObject().put("moves", JSONArray(listOf("e4"))))
        assertTrue(out, "1... e5 2. Nf3 Nc6" in out && "c5" !in out.substringBefore("On the board there"))
    }

    /**
     * `_Streaming`: an engine that hands each depth on as it finishes it, and notes what a reader
     * had of the position at each.
     */
    private class Streaming : FakeEngine() {
        lateinit var annotator: GameAnnotator
        val read = ArrayList<Int>()

        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
            for (d in listOf(EARLY_DEPTH - 2, EARLY_DEPTH, EARLY_DEPTH + 2)) {
                onDepth?.invoke(listOf(Line(moves = listOf("e2e4"), depth = d, expectedScore = 0.5)))
                read.add(annotator.computed.lines(board).firstOrNull()?.depth ?: 0)
            }
            return listOf(Line(moves = listOf("e2e4"), depth = 18, expectedScore = 0.5))
        }
    }

    /**
     * A reader gets the newest depth from the first usable one (Tai, 2026-09-28: "if the consumers
     * read after stockfish reaches depth > 12 then they will get whatever lines that is available
     * at the time"); a verdict's line counts too, and a new game forgets them.
     */
    @Test
    fun theAnalyserPutsInEachUsableDepthAsItFinishesIt() {
        val engine = Streaming()
        val annotator = GameAnnotator(engine, depth = 18, withThreats = false)
        engine.annotator = annotator
        annotator.measure(start)
        assertEquals(listOf(0, EARLY_DEPTH, EARLY_DEPTH + 2), engine.read)
        assertEquals(18, annotator.computed.lines(start)[0].depth)
        val afterE4 = after("e4")
        annotator.measureForVerdict(afterE4)
        assertEquals(18, annotator.computed.lines(afterE4)[0].depth)
        annotator.reset()
        assertEquals(emptyList<Line>(), annotator.computed.lines(start))
        assertEquals(emptyList<Line>(), annotator.computed.lines(afterE4))
    }
}
