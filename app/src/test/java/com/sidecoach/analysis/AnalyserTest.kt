package com.sidecoach.analysis

import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Analysis
import com.sidecoach.events.Event
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.MoveAssessment
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.TimeUnit

/**
 * The engine service for a game followed live ([Analyser], `tests/test_analyser.py`): every
 * search one line, every position searched once, on a pool of workers; a position told as it
 * comes (its best line and its hint, then whole); a move judged as soon as what it needs is in,
 * in the order of the moves; the coach's questions ahead of the game's other positions. Gated
 * fake engines stage every race; nothing waits on a clock. The tests that go through the engine
 * glue (`Assessing`) are the device's (LiveBookTest, TakebackTest, WatchedReviewTest).
 */
class AnalyserTest {

    private fun line(vararg moves: String, score: Double = 0.5) = Line(moves = moves.toList(), depth = 18, expectedScore = score)

    private fun board(vararg sans: String): Board {
        val board = Board()
        for (san in sans) board.push(board.parseSan(san)!!)
        return board
    }

    private fun after(vararg sans: String): String = board(*sans).fen()

    private fun analyser(engines: List<ScriptEngine>, threats: Boolean = false): Analyser {
        val annotator = GameAnnotator(engines[0], depth = 18, multipv = 6, withThreats = threats)
        return Analyser(annotator, engines = engines)
    }

    private fun entered(engine: GatedEngine, fen: String) =
        engine.entered.getValue(fen).await(HANG_MS, TimeUnit.MILLISECONDS)

    /** Every search the engines made: the position, the lines asked for and the depth. */
    private fun searches(engines: List<ScriptEngine>): List<Triple<String, Int, Int?>> =
        engines.flatMap { e -> synchronized(e.calls) { e.calls.indices.map { Triple(e.calls[it], e.multipvs[it], e.depths[it]) } } }

    /**
     * `test_every_search_is_one_line_but_the_hint_and_none_is_made_twice`: the engine audit
     * (2026-10-05): several lines to the full depth explode on decided and forcing positions, one
     * line does not. A game played, a move taken back and played again: every search to the full
     * depth is one line, the hint's alone several at EARLY_DEPTH, and no position is searched
     * twice either way.
     */
    @Test
    fun everySearchIsOneLineButTheHintAndNoneIsMadeTwice() {
        val lines = mapOf(
            START to listOf(line("e2e4"), line("d2d4")),
            after("e4") to listOf(line("e7e5"), line("c7c5")),
            after("e4", "e5") to listOf(line("g1f3"), line("b1c3")),
        )
        val engines = listOf(ScriptEngine(lines), ScriptEngine(lines))
        val analyser = analyser(engines)
        val watch = AnalyserWatch(analyser)
        analyser.start()
        try {
            val board = Board()
            analyser.submitPosition("g", board, 0)
            for ((i, san) in listOf("e4", "e5", "Nf3").withIndex()) {
                val move = board.parseSan(san)!!
                analyser.submitMove("g", board, move, i + 1)
                board.push(move)
                watch.idle()
            }
            board.pop()
            analyser.withdraw("g", 2)
            analyser.submitMove("g", board, board.parseSan("Nf3")!!, 3)
            watch.idle()
        } finally {
            analyser.close()
        }
        val searched = searches(engines)
        assertEquals(searched.sortedBy { it.toString() }.toString(), searched.toSet().size, searched.size)
        assertEquals(setOf(1 to 18, 6 to EARLY_DEPTH), searched.map { it.second to it.third }.toSet())
        assertEquals(searched.count { it.second == 1 }, analyser.searches)
    }

    /**
     * `test_a_position_is_whole_from_its_candidates_own_searches`: each of a position's best moves
     * is searched as its own position, one line to the full depth: the lines are those searches
     * turned to the side to move, best first, the best line keeping its own moves -- where six
     * lines at once to the full depth took up to 47 s.
     */
    @Test
    fun aPositionIsWholeFromItsCandidatesOwnSearches() {
        val engine = ScriptEngine(
            mapOf(
                START to listOf(line("e2e4", score = 0.6), line("d2d4", score = 0.55)),
                after("e4") to listOf(line("e7e5", score = 0.40)),
                after("d4") to listOf(line("d7d5", score = 0.48)),
            ),
        )
        val analyser = analyser(listOf(engine))
        val watch = AnalyserWatch(analyser)
        analyser.start()
        val told: List<Analysis>
        try {
            analyser.submitPosition("g", Board(), 0)
            told = watch.settled().filterIsInstance<Analysis>()
        } finally {
            analyser.close()
        }
        val whole = told.last()
        assertTrue(!whole.provisional)
        assertEquals(listOf(listOf("e2e4"), listOf("d2d4", "d7d5")), whole.lines.map { it.moves })
        assertEquals(0.6, whole.lines[0].expectedScore, 1e-9)
        assertEquals(0.52, whole.lines[1].expectedScore, 1e-9)
    }

    /**
     * `test_a_position_is_told_at_once_and_then_whole`: the first answer comes at once -- the
     * best line to the full depth, the hint's moves at EARLY_DEPTH -- provisional; then the
     * position whole, the candidates' own searches in; with the side searches, whole once they
     * are in too. Each stage once.
     */
    @Test
    fun aPositionIsToldAtOnceAndThenWhole() {
        val lines = mapOf(START to listOf(line("e2e4"), line("d2d4")))
        for (threats in listOf(false, true)) {
            val engine = GatedEngine(lines) // its lines at the depth asked for
            val analyser = analyser(listOf(engine), threats = threats)
            val watch = AnalyserWatch(analyser)
            analyser.start()
            val told: List<Analysis>
            try {
                analyser.submitPosition("g", Board(), 0)
                told = watch.settled().filterIsInstance<Analysis>()
            } finally {
                analyser.close()
            }
            assertEquals("the best line alone, first", listOf(18), told[0].lines.map { it.depth })
            assertEquals("then the hint's", listOf(18, EARLY_DEPTH), told[1].lines.map { it.depth })
            assertEquals(List(told.size - 1) { true } + false, told.map { it.provisional })
            assertEquals("${told.map { it.lines }}", if (threats) 4 else 3, told.size)
            assertTrue(told.last().lines.all { it.depth == 18 })
        }
    }

    /**
     * `test_a_move_not_among_the_lines_waits_for_the_position_after`: a move none of the
     * candidates is judged from the position after's best line -- the game's next position,
     * searched anyway -- so its verdict waits for that search; one among them is judged from its
     * own candidate's search at once.
     */
    @Test
    fun aMoveNotAmongTheLinesWaitsForThePositionAfter() {
        val engine = GatedEngine(
            mapOf(
                START to listOf(line("e2e4")),
                after("e4") to listOf(line("e7e5", score = 0.40)),
                after("a3") to listOf(line("e7e5", score = 0.55)),
            ),
            gated = listOf(after("a3")),
        )
        val analyser = analyser(listOf(engine))
        val watch = AnalyserWatch(analyser)
        analyser.start()
        val got: List<Event>
        try {
            analyser.submitPosition("g", Board(), 0)
            watch.idle()
            analyser.submitMove("g", Board(), Move.fromUci("a2a3"), 1)
            assertTrue(entered(engine, after("a3")))
            assertEquals("judged before the position after was searched", 1, synchronized(analyser.lock) { analyser.nextVerdict })
            engine.release(after("a3"))
            got = watch.settled()
        } finally {
            analyser.close()
        }
        val verdict = got.filterIsInstance<MoveAssessment>().first()
        assertEquals(0.60 - (1 - 0.55), verdict.expectedLoss, 1e-9)
        assertTrue("${verdict.label}", verdict.label != Label.BEST)
    }

    /**
     * `test_a_decided_search_running_long_stands_in_for_the_verdict`: Tai, 2026-10-05: a
     * best-line search still running after DECIDED_AFTER_MS whose deepest finished depth shows the
     * position decided has that depth's score judge the move -- it flips a severity as rarely as
     * two full searches disagree. Undecided, it is waited for.
     */
    @Test
    fun aDecidedSearchRunningLongStandsInForTheVerdict() {
        val engine = GatedEngine(
            mapOf(START to listOf(line("e2e4")), after("e4") to listOf(line("e7e5", score = 0.40))),
            gated = listOf(after("h3")),
        )
        val analyser = analyser(listOf(engine))
        val watch = AnalyserWatch(analyser)
        analyser.start()
        val verdict: MoveAssessment
        try {
            analyser.submitPosition("g", Board(), 0)
            watch.idle()
            analyser.submitMove("g", Board(), Move.fromUci("h2h3"), 1)
            assertTrue(entered(engine, after("h3")))
            val key = cacheKey(board("h3"))
            synchronized(analyser.lock) { analyser.nodes.getValue(key).deepest = line("e7e5", score = 0.5).copy(depth = 15) }
            analyser.decide(Analyser.Job(Analyser.BEST, key, analyser.era))
            assertEquals("an undecided depth judged the move", 1, synchronized(analyser.lock) { analyser.nextVerdict })
            synchronized(analyser.lock) { analyser.nodes.getValue(key).deepest = line("d8h4", score = 0.9).copy(depth = 15) }
            analyser.decide(Analyser.Job(Analyser.BEST, key, analyser.era))
            assertEquals("the decided depth did not judge it", 2, synchronized(analyser.lock) { analyser.nextVerdict })
            verdict = analyser.poll().filterIsInstance<MoveAssessment>().first()
            engine.release(after("h3"))
            watch.idle()
        } finally {
            analyser.close()
        }
        assertEquals(0.60 - 0.1, verdict.expectedLoss, 1e-9)
    }

    /**
     * `test_moves_are_judged_in_the_order_they_were_played`: a verdict reads the move before it
     * ("the opponent just erred", "a recapture"): the second move's is held until the first's is
     * out, though its own searches were done first.
     */
    @Test
    fun movesAreJudgedInTheOrderTheyWerePlayed() {
        val lines = mapOf(START to listOf(line("e2e4", score = 0.55)), after("e4") to listOf(line("e7e5", score = 0.5)))
        val engines = List(3) { GatedEngine(lines, listOf(START)) }
        val analyser = analyser(engines)
        val watch = AnalyserWatch(analyser)
        analyser.start()
        val got: List<Event>
        try {
            val board = Board()
            analyser.submitMove("g", board, board.parseSan("e4")!!, 1)
            board.push(board.parseSan("e4")!!)
            analyser.submitMove("g", board, board.parseSan("e5")!!, 2)
            val key = cacheKey(board)
            watch.until("e4 whole") { synchronized(analyser.lock) { analyser.nodes[key]?.whole != null } }
            assertEquals("e5 judged before e4", 1, synchronized(analyser.lock) { analyser.nextVerdict })
            for (engine in engines) engine.release(START)
            got = watch.settled()
        } finally {
            analyser.close()
        }
        assertEquals(listOf(1, 2), got.filterIsInstance<MoveAssessment>().map { it.ply })
    }

    /**
     * `test_a_takeback_says_nothing_more_of_the_withdrawn_moves`: Tai, 2026-09-23: "takebacks
     * make the program lag behind by a lot". The moves taken back get no verdict and no analysis,
     * and nothing is owed for them; a search made only for them stops.
     */
    @Test
    fun aTakebackSaysNothingMoreOfTheWithdrawnMoves() {
        val gated = listOf(after("e4", "e5"))
        val engine = GatedEngine(gated = gated)
        val analyser = analyser(listOf(engine))
        val watch = AnalyserWatch(analyser)
        val got: List<Event>
        try {
            // The moves in before the worker starts: the first verdict's needs, then the
            // second's, whose position after e5 holds.
            val board = Board()
            for ((i, san) in listOf("e4", "e5", "Nf3").withIndex()) {
                val move = board.parseSan(san)!!
                analyser.submitMove("g", board, move, i + 1)
                board.push(move)
            }
            analyser.start()
            assertTrue("the position after e5 never searched", entered(engine, gated[0]))
            analyser.withdraw("g", 1)
            watch.until("the withdrawn move's search stopped") { engine.halted.toList() == gated }
            got = watch.settled()
        } finally {
            analyser.close()
        }
        val late = got.filter { (it is MoveAssessment && it.ply >= 2) || (it is Analysis && it.ply >= 2) }
        assertEquals(emptyList<Any>(), late)
        assertEquals(2, analyser.withdrawn)
        assertEquals(0, analyser.pending)
    }

    /**
     * `test_a_position_seen_again_is_not_searched_again`: a puzzle tried again, a board re-read:
     * a position searched before is answered from what was found, after the game's reset too.
     */
    @Test
    fun aPositionSeenAgainIsNotSearchedAgain() {
        val engine = ScriptEngine(mapOf(START to listOf(line("e2e4"))))
        val analyser = analyser(listOf(engine))
        val watch = AnalyserWatch(analyser)
        analyser.start()
        val searched: Int
        val again: List<Event>
        try {
            analyser.submitPosition("g", Board(), 0)
            watch.settled()
            searched = engine.calls.size
            analyser.reset()
            analyser.submitPosition("h", Board(), 0)
            again = watch.settled()
        } finally {
            analyser.close()
        }
        assertEquals(searched, engine.calls.size)
        assertEquals(listOf("h"), again.filterIsInstance<Analysis>().map { it.gameId }.distinct())
    }

    /**
     * `test_a_failed_search_does_not_end_the_session_nor_hold_the_game`: an engine that fails
     * every search: each failure counted, the moves go unjudged, and nothing is owed -- the game
     * is never held waiting on a search that cannot come.
     */
    @Test
    fun aFailedSearchDoesNotEndTheSessionNorHoldTheGame() {
        val broken = object : ScriptEngine() {
            override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> =
                throw RuntimeException("engine gone")
        }
        val analyser = Analyser(GameAnnotator(broken, depth = 6, withThreats = false))
        val watch = AnalyserWatch(analyser)
        analyser.start()
        val failed: Int
        try {
            analyser.submitMove("g", Board(), Move.fromUci("e2e4"), 1)
            val got = watch.settled()
            failed = analyser.failures
            assertEquals("each position's best line and hint, of both positions", 4, failed)
            assertTrue("${analyser.lastError}", analyser.lastError?.contains("engine gone") == true)
            assertTrue(got.none { it is MoveAssessment })
            analyser.submitPosition("h", board("d4"), 0)
            watch.idle()
        } finally {
            analyser.close()
        }
        assertEquals("the next position tried, and failed alike", failed + 2, analyser.failures)
    }

    /**
     * `test_the_workers_take_the_verdicts_then_the_coach_then_the_game`: who waits on a search,
     * first to last: the next verdict (the word on the move just played), the coach's question
     * (the player waiting on it), the position the game is on, the game's other positions.
     */
    @Test
    fun theWorkersTakeTheVerdictsThenTheCoachThenTheGame() {
        val analyser = analyser(listOf(ScriptEngine(mapOf(START to listOf(line("e2e4"))))))
        val watch = AnalyserWatch(analyser)
        analyser.submitPosition("g", Board(), 0)
        val asked = board("d4")
        Thread({ analyser.request(asked) }, "the coach").apply { isDaemon = true }.start()
        watch.until("the coach asked") { synchronized(analyser.lock) { analyser.pick()?.key == cacheKey(asked) } }
        synchronized(analyser.lock) {
            assertEquals("the coach's not first", Analyser.Job(Analyser.BEST, cacheKey(asked), 0), analyser.pick())
        }
        analyser.submitMove("g", Board(), Move.fromUci("e2e4"), 1)
        val jobs = synchronized(analyser.lock) {
            assertEquals(Analyser.Job(Analyser.BEST, cacheKey(Board()), 0), analyser.pick())
            analyser.agenda()
        }
        val askedJob = Analyser.Job(Analyser.BEST, cacheKey(asked), 0)
        val gameJob = Analyser.Job(Analyser.HINT, cacheKey(board("e4")), 0)
        assertTrue("the game's position before the coach", jobs.indexOf(askedJob) < jobs.indexOf(gameJob))
        analyser.close()
    }

    /**
     * `test_the_coach_is_answered_from_the_pool_and_never_searches_twice`: the coach's engine
     * ([PoolSearcher]): a question on a position gets it whole, from the analyser's workers -- no
     * Stockfish of its own -- and asked again, at once.
     */
    @Test
    fun theCoachIsAnsweredFromThePoolAndNeverSearchesTwice() {
        val engine = ScriptEngine(mapOf(after("d4") to listOf(line("d7d5", score = 0.5), line("g8f6", score = 0.45))))
        val analyser = analyser(listOf(engine))
        val searcher = PoolSearcher(analyser)
        analyser.start()
        val first: List<Line>
        val again: List<Line>
        val searched: Int
        try {
            first = searcher.analyse(board("d4"))
            searched = engine.calls.size
            again = searcher.analyse(board("d4"))
        } finally {
            analyser.close()
        }
        assertEquals(first, again)
        assertEquals(listOf("d7d5", "g8f6"), first.map { it.moves[0] })
        assertEquals(searched, engine.calls.size)
    }

    /**
     * `test_a_question_given_up_or_asked_of_a_closed_analyser_never_hangs`: the coach's wait is
     * bounded: a question past its time gets what is known (here nothing) and is dropped;
     * closed, the analyser answers at once with what it has.
     */
    @Test
    fun aQuestionGivenUpOrAskedOfAClosedAnalyserNeverHangs() {
        val gated = after("c4")
        val engine = GatedEngine(gated = listOf(gated))
        val analyser = analyser(listOf(engine))
        analyser.start()
        assertNull(analyser.request(board("c4"), timeoutMs = 0))
        engine.release(gated)
        analyser.close()
        assertNull(analyser.request(board("e4")))
    }

    /**
     * `test_the_workers_engines_search_on_one_thread_each`: at a fixed depth more threads widen
     * the search instead of speeding it: four took 7.7 times the processor time of one, and twice
     * as long, for an answer no better (the engine audit, 2026-10-04). Each worker's engine has
     * one, and the pool its size.
     */
    @Test
    fun theWorkersEnginesSearchOnOneThreadEach() {
        val made = ArrayList<Int>()
        val analyser = openAnalyser(depth = 18, multipv = 6, workers = 4) { threads ->
            made.add(threads)
            ScriptEngine()
        }
        analyser.close()
        assertEquals(listOf(1, 1, 1, 1), made)
        assertEquals(4, analyser.engines.size)
    }

    /**
     * `test_settings_changed_mid_search_keep_nothing_measured_the_old_way`: the depth or the
     * lines saved while a search runs: what it finds is not kept, and the game's position is
     * measured again and told again.
     */
    @Test
    fun settingsChangedMidSearchKeepNothingMeasuredTheOldWay() {
        val engine = GatedEngine(mapOf(START to listOf(line("e2e4"))), gated = listOf(START))
        val analyser = analyser(listOf(engine))
        val watch = AnalyserWatch(analyser)
        analyser.start()
        val told: List<Analysis>
        try {
            analyser.submitPosition("g", Board(), 0)
            assertTrue(entered(engine, START))
            analyser.forgetMeasured()
            watch.until("the search under way let go") { engine.halted.toList() == listOf(START) }
            engine.release(START)
            told = watch.settled().filterIsInstance<Analysis>()
        } finally {
            analyser.close()
        }
        assertTrue(told.isNotEmpty() && !told.last().provisional)
        assertEquals(1, analyser.era)
    }

    private companion object {
        val START: String = Board().fen()
    }
}
