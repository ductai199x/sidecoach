package com.sidecoach.analysis

import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Analysis
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
 * position searched once, on a pool of workers; the look-ahead's searches promoted when the game
 * reaches them, never begun again; a search stopped only when nothing will read it; a move judged
 * from the lines that are there. Gated fake engines stage every race; nothing waits on a clock.
 * The tests that go through the engine glue (`Assessing`) are the device's (LiveBookTest,
 * TakebackTest, WatchedReviewTest).
 */
class AnalyserTest {

    private fun line(vararg moves: String, score: Double = 0.5) = Line(moves = moves.toList(), depth = 18, expectedScore = score)

    private fun board(vararg sans: String): Board {
        val board = Board()
        for (san in sans) board.push(board.parseSan(san)!!)
        return board
    }

    private fun after(vararg sans: String): String = board(*sans).fen()

    private fun analyser(engines: List<ScriptEngine>, lookAhead: Int = 2, threats: Boolean = false): Analyser {
        val annotator = GameAnnotator(engines[0], depth = 18, withThreats = threats)
        return Analyser(annotator, engines = engines, lookAhead = lookAhead)
    }

    private fun entered(engine: GatedEngine, fen: String) =
        engine.entered.getValue(fen).await(HANG_MS, TimeUnit.MILLISECONDS)

    /** The searches of a position (several lines: its own search, never a side search). */
    private fun searches(engines: List<ScriptEngine>): List<String> =
        engines.flatMap { e -> e.asked().filter { it.second > 1 }.map { it.first } }

    /**
     * `test_every_position_of_a_game_is_searched_once`: the engine audit (2026-10-04): 258 of 624
     * positions of a session were searched two to six times. A game played, its moves both looked
     * at ahead and not, one taken back and played again: each position, with the moves that led
     * to it, searched exactly once.
     */
    @Test
    fun everyPositionOfAGameIsSearchedOnce() {
        val lines = mapOf(
            START to listOf(line("e2e4"), line("d2d4")),
            after("e4") to listOf(line("e7e5"), line("c7c5")),
            after("d4") to listOf(line("d7d5")),
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
            watch.until("the look-ahead done") { synchronized(analyser.lock) { analyser.claimed.isEmpty() } }
        } finally {
            analyser.close()
        }
        val searched = engines.flatMap { e ->
            synchronized(e.calls) { e.calls.indices.filter { e.multipvs[it] > 1 }.map { e.calls[it] to e.histories[it] } }
        }
        assertEquals(searched.sortedBy { it.toString() }.toString(), searched.toSet().size, searched.size)
        assertEquals(searched.size, analyser.searches)
    }

    /**
     * `test_a_move_among_the_lines_is_judged_without_a_search_of_its_own`: 85% of a session's
     * moves are among the lines of the position they were played from: judged from their own line
     * there, at the same depth as the best move (the same label in 319 of 320), and at once -- the
     * search of the position after them, held here, is not waited for.
     */
    @Test
    fun aMoveAmongTheLinesIsJudgedWithoutASearchOfItsOwn() {
        val engine = GatedEngine(mapOf(START to listOf(line("e2e4", score = 0.60), line("d2d4", score = 0.55))), gated = listOf(after("d4")))
        val analyser = analyser(listOf(engine), lookAhead = 0)
        val watch = AnalyserWatch(analyser)
        analyser.start()
        val verdict: MoveAssessment
        try {
            analyser.submitPosition("g", Board(), 0)
            watch.idle()
            analyser.poll()
            analyser.submitMove("g", Board(), Move.fromUci("d2d4"), 1)
            assertTrue("the position after d4 never searched", entered(engine, after("d4")))
            watch.until("d4 judged") { synchronized(analyser.lock) { analyser.nextVerdict == 2 } }
            verdict = analyser.poll().filterIsInstance<MoveAssessment>().first()
            engine.release(after("d4"))
            watch.idle()
        } finally {
            analyser.close()
        }
        assertEquals("d2d4", verdict.uci)
        assertEquals(0.05, verdict.expectedLoss, 1e-9)
        assertTrue("${verdict.label}", verdict.label != Label.BEST)
    }

    /**
     * `test_a_move_not_among_the_lines_waits_for_the_position_after`: a move none of the lines
     * begins with is judged from the position after's search -- the game's next position,
     * searched anyway -- so its verdict waits for that search.
     */
    @Test
    fun aMoveNotAmongTheLinesWaitsForThePositionAfter() {
        val engine = GatedEngine(
            mapOf(START to listOf(line("e2e4", score = 0.60)), after("a3") to listOf(line("e7e5", score = 0.55))),
            gated = listOf(after("a3")),
        )
        val analyser = analyser(listOf(engine), lookAhead = 0)
        val watch = AnalyserWatch(analyser)
        analyser.start()
        val got: List<com.sidecoach.events.Event>
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
    }

    /**
     * `test_a_position_searched_ahead_is_promoted_when_the_game_reaches_it`: the game reaches a
     * position its look-ahead is still searching: that search becomes the game's -- told as the
     * game's position when it ends -- and is never begun again. It was stopped and searched again
     * from depth 1 (the engine audit: "look-ahead -> look-ahead -> early -> verdict -> full" on one
     * position).
     */
    @Test
    fun aPositionSearchedAheadIsPromotedWhenTheGameReachesIt() {
        val lines = mapOf(START to listOf(line("e2e4"), line("d2d4")))
        val gated = listOf(after("e4"))
        val engines = listOf(GatedEngine(lines, gated), GatedEngine(lines, gated))
        val analyser = analyser(engines, lookAhead = 1)
        val watch = AnalyserWatch(analyser)
        analyser.start()
        val got: List<com.sidecoach.events.Event>
        try {
            analyser.submitPosition("g", Board(), 0)
            assertTrue("e4 never looked at ahead", entered(engines[1], after("e4")))
            analyser.submitMove("g", Board(), Move.fromUci("e2e4"), 1)
            assertEquals(1, analyser.aheadHits)
            engines[1].release(after("e4"))
            got = watch.settled()
        } finally {
            analyser.close()
        }
        assertEquals("searched again", 1, searches(engines).count { it == after("e4") })
        assertTrue("the promoted search was stopped", engines.all { it.halted.isEmpty() })
        val told = got.filterIsInstance<Analysis>().filter { it.ply == 1 }
        assertTrue("$told", told.isNotEmpty() && told.last().fen == after("e4") && !told.last().provisional)
    }

    /**
     * `test_a_search_ahead_off_every_path_stops_and_one_still_on_a_path_goes_on`: a move played:
     * the look-ahead's search of a position now on no path is stopped -- its answer would never be
     * read -- and the one the game went to goes on, now the game's.
     */
    @Test
    fun aSearchAheadOffEveryPathStopsAndOneStillOnAPathGoesOn() {
        val lines = mapOf(START to listOf(line("e2e4"), line("d2d4")))
        val gated = listOf(after("e4"), after("d4"))
        val engines = List(3) { GatedEngine(lines, gated) }
        val analyser = analyser(engines, lookAhead = 1)
        val watch = AnalyserWatch(analyser)
        analyser.start()
        try {
            analyser.submitPosition("g", Board(), 0)
            for (fen in gated) watch.until("$fen ahead") { engines.any { it.entered.getValue(fen).count == 0L } }
            analyser.submitMove("g", Board(), Move.fromUci("e2e4"), 1)
            watch.until("the search after d4 stopped") { engines.any { it.halted.isNotEmpty() } }
            for (engine in engines) engine.release(after("e4"))
            watch.idle()
        } finally {
            analyser.close()
        }
        assertEquals(listOf(after("d4")), engines.flatMap { it.halted })
        assertEquals(1, analyser.stopped)
    }

    /**
     * `test_the_game_has_a_worker_of_its_own`: the first worker takes only the game's own
     * positions: a position the game reaches never waits behind the look-ahead. Asked while the
     * look-ahead has work waiting and its worker is held, the game's worker turns it down, the
     * other takes it; and a position of the game's is the game's worker's at once.
     */
    @Test
    fun theGameHasAWorkerOfItsOwn() {
        val lines = mapOf(START to listOf(line("e2e4"), line("d2d4")))
        val gated = listOf(after("e4"), after("d4"))
        val engines = List(2) { GatedEngine(lines, gated) }
        val analyser = analyser(engines, lookAhead = 1)
        val watch = AnalyserWatch(analyser)
        analyser.start()
        try {
            analyser.submitPosition("g", Board(), 0)
            watch.until("e4 looked at ahead") { engines[1].entered.getValue(after("e4")).count == 0L }
            watch.idle()
            synchronized(analyser.lock) {
                assertNull("the game's worker took the look-ahead", analyser.pick(0))
                val ahead = analyser.pick(1)
                assertTrue("$ahead", ahead != null && ahead.key == board("d4").fen())
                val nf3 = board("Nf3")
                analyser.path[1] = analyser.node(nf3).key
                val game = analyser.pick(0)
                assertTrue("the game's position waited", game != null && game.key == analyser.path[1])
                analyser.path.remove(1)
            }
            engines[1].release(after("e4"))
            engines[1].release(after("d4"))
        } finally {
            analyser.close()
        }
        assertTrue(gated.none { it in engines[0].calls })
    }

    /**
     * `test_a_position_is_told_at_each_depth_of_its_own_search_and_whole_last`: Tai, 2026-09-28:
     * "first usable answer is depth 12 ... the computation continues until max depth of 18".
     * Depth 12 is the position's own search's: one search, every depth from 12 told as it
     * finishes, provisional, then whole -- a search of its own to depth 12 first cost 27% more
     * than the full depth alone (the engine audit, 2026-10-04).
     */
    @Test
    fun aPositionIsToldAtEachDepthOfItsOwnSearchAndWholeLast() {
        val engine = GatedEngine(mapOf(START to listOf(line("e2e4"))))
        val analyser = analyser(listOf(engine), lookAhead = 0)
        val watch = AnalyserWatch(analyser)
        analyser.start()
        val got: List<Analysis>
        try {
            analyser.submitPosition("g", Board(), 0)
            got = watch.settled().filterIsInstance<Analysis>()
        } finally {
            analyser.close()
        }
        assertEquals("one search", listOf(START), engine.calls.toList())
        assertEquals((EARLY_DEPTH..18).toList(), got.map { it.lines[0].depth })
        assertEquals(List(got.size - 1) { true } + false, got.map { it.provisional })
    }

    /**
     * `test_the_full_depth_is_whole_once_its_side_searches_are_in`: the coach's lessons wait for
     * the whole analysis (threats, traps): the full depth is told provisional when its own search
     * ends, and whole when its side searches are in -- those searching from the position's lines,
     * never the position again.
     */
    @Test
    fun theFullDepthIsWholeOnceItsSideSearchesAreIn() {
        val engine = GatedEngine(mapOf(START to listOf(line("e2e4"))))
        val analyser = analyser(listOf(engine), lookAhead = 0, threats = true)
        val watch = AnalyserWatch(analyser)
        analyser.start()
        val got: List<Analysis>
        try {
            analyser.submitPosition("g", Board(), 0)
            got = watch.settled().filterIsInstance<Analysis>().filter { it.lines[0].depth == 18 }
        } finally {
            analyser.close()
        }
        assertEquals(listOf(true, false), got.map { it.provisional })
        assertEquals("the position searched again for its side searches", 1, engine.calls.count { it == START })
        assertEquals(1, analyser.sideSearches)
    }

    /**
     * `test_moves_are_judged_in_the_order_they_were_played`: a verdict reads the move before it
     * ("the opponent just erred", "a recapture"): the second move's is held until the first's is
     * out, though its own lines came first.
     */
    @Test
    fun movesAreJudgedInTheOrderTheyWerePlayed() {
        val lines = mapOf(START to listOf(line("e2e4", score = 0.55)), after("e4") to listOf(line("e7e5", score = 0.5)))
        val engines = listOf(GatedEngine(lines, listOf(START)), GatedEngine(lines, listOf(START)))
        val analyser = analyser(engines, lookAhead = 0)
        val watch = AnalyserWatch(analyser)
        analyser.start()
        val got: List<com.sidecoach.events.Event>
        try {
            val board = Board()
            analyser.submitMove("g", board, board.parseSan("e4")!!, 1)
            board.push(board.parseSan("e4")!!)
            analyser.submitMove("g", board, board.parseSan("e5")!!, 2)
            watch.until("e4 known") { synchronized(analyser.lock) { analyser.nodes[analyser.path[1]]?.measured != null } }
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
        val gated = listOf(after("e4", "e5", "Nf3"))
        val engine = GatedEngine(gated = gated)
        val analyser = analyser(listOf(engine), lookAhead = 0)
        val watch = AnalyserWatch(analyser)
        val got: List<com.sidecoach.events.Event>
        try {
            // The moves in before the worker starts: it takes the newest first, and holds there.
            val board = Board()
            for ((i, san) in listOf("e4", "e5", "Nf3").withIndex()) {
                val move = board.parseSan(san)!!
                analyser.submitMove("g", board, move, i + 1)
                board.push(move)
            }
            analyser.start()
            assertTrue("the position after Nf3 never searched", entered(engine, gated[0]))
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
        val engine = GatedEngine(mapOf(START to listOf(line("e2e4"))))
        val analyser = analyser(listOf(engine), lookAhead = 0)
        val watch = AnalyserWatch(analyser)
        analyser.start()
        val again: List<com.sidecoach.events.Event>
        try {
            analyser.submitPosition("g", Board(), 0)
            watch.settled()
            analyser.reset()
            analyser.submitPosition("h", Board(), 0)
            again = watch.settled()
        } finally {
            analyser.close()
        }
        assertEquals(listOf(START), engine.calls.toList())
        assertEquals(listOf("h"), again.filterIsInstance<Analysis>().map { it.gameId }.distinct())
    }

    /** `test_a_failed_search_does_not_end_the_session`. */
    @Test
    fun aFailedSearchDoesNotEndTheSession() {
        val broken = object : ScriptEngine() {
            override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> =
                throw RuntimeException("engine gone")
        }
        val analyser = Analyser(GameAnnotator(broken, depth = 6, withThreats = false))
        val watch = AnalyserWatch(analyser)
        analyser.start()
        try {
            analyser.submitMove("g", Board(), Move.fromUci("e2e4"), 1)
            watch.until("the failure counted") { analyser.failures >= 1 }
            assertTrue("${analyser.lastError}", analyser.lastError?.contains("engine gone") == true)
            analyser.submitPosition("h", board("d4"), 0)
            watch.until("the next search tried, and failed alike") { analyser.failures >= 2 }
        } finally {
            analyser.close()
        }
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
        val analyser = openAnalyser(depth = 18, multipv = 6, workers = 3) { threads ->
            made.add(threads)
            ScriptEngine()
        }
        analyser.close()
        assertEquals(listOf(1, 1, 1), made)
        assertEquals(3, analyser.engines.size)
    }

    /**
     * `test_the_look_ahead_goes_where_the_game_is_likely_to_go`: Tai, 2026-10-04: "an adaptive,
     * dynamic system that is able to maximize both comp eff and the chance of catching the
     * opponent's move". A forced line -- one move far better than the rest -- is followed deep and
     * its alternatives left alone; a quiet position, its moves about equal, has each searched one
     * ply on and nothing behind them (each too unlikely by then).
     */
    @Test
    fun theLookAheadGoesWhereTheGameIsLikelyToGo() {
        val forced = mapOf(
            START to listOf(line("e2e4", score = 0.9), line("d2d4", score = 0.3)),
            after("e4") to listOf(line("e7e5", score = 0.9), line("c7c5", score = 0.2)),
            after("e4", "e5") to listOf(line("g1f3", score = 0.9), line("b1c3", score = 0.2)),
        )
        val quietMoves = listOf("e2e4", "d2d4", "c2c4", "g1f3")
        val quiet = mapOf(
            START to quietMoves.map { line(it) },
            after("e4") to listOf(line("e7e5"), line("c7c5")),
        )

        fun lookedAt(lines: Map<String, List<Line>>): Set<String> {
            val analyser = analyser(listOf(ScriptEngine(lines)), lookAhead = 4)
            val watch = AnalyserWatch(analyser)
            analyser.start()
            try {
                analyser.submitPosition("g", Board(), 0)
                watch.idle()
                watch.until("the look-ahead done") {
                    synchronized(analyser.lock) {
                        analyser.claimed.isEmpty() && analyser.ahead.keys.all { analyser.nodes[it]?.measured != null }
                    }
                }
                return synchronized(analyser.lock) { analyser.ahead.keys.map { analyser.nodes.getValue(it).board.fen() }.toSet() }
            } finally {
                analyser.close()
            }
        }

        assertEquals(setOf(after("e4"), after("e4", "e5"), after("e4", "e5", "Nf3")), lookedAt(forced))
        assertEquals(quietMoves.map { after(Board().san(Move.fromUci(it))) }.toSet(), lookedAt(quiet))
    }

    /**
     * `test_how_a_side_plays_is_learned_from_this_game_and_forgotten_with_it`: move accuracy
     * depends on the game (Tai, 2026-10-04): nothing trained, nothing kept between games. A player
     * who keeps playing the second line has it made likelier for them as the game goes -- the
     * first moves decide it, the start counting for little -- and the next game starts over.
     */
    @Test
    fun howASidePlaysIsLearnedFromThisGameAndForgottenWithIt() {
        val analyser = analyser(listOf(ScriptEngine()))
        analyser.submitPosition("g", Board(), 0)
        val lines = listOf(line("e2e4", score = 0.60), line("d2d4", score = 0.52))
        val before = analyser.odds(true, lines)[1]
        repeat(4) { analyser.learn(Analyser.Played(1, "a", "b", Move.fromUci("d2d4"), true), lines) }
        val later = analyser.odds(true, lines)[1]
        assertTrue("$before $later", analyser.t.getValue(true) > T_START && later > 2 * before)
        assertEquals("the opponent's is their own", T_START, analyser.t.getValue(false), 0.0)
        analyser.submitPosition("h", Board(), 0) // the next game
        assertEquals(T_START, analyser.t.getValue(true), 0.0)
        assertEquals(before, analyser.odds(true, lines)[1], 1e-12)
    }

    private companion object {
        val START: String = Board().fen()
    }
}
