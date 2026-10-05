package com.sidecoach.analysis

import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Analysis
import com.sidecoach.events.Event
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.MoveAssessment
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Collections
import java.util.concurrent.TimeUnit

/**
 * The engine service for a game followed live ([Analyser], `tests/test_analyser.py`): every
 * search one line, every position searched once, on a pool of workers; a position told as it
 * comes (its first answer, then whole, then final); a move judged as soon as what it needs is in,
 * in the order of the moves; the coach's questions ahead of the game's other positions; nothing a
 * worker does after a search ends it. Gated fake engines stage every race, a hand-moved search
 * clock times the decided rule; nothing waits on a clock. The tests that go through the engine
 * glue (`Assessing`) are the device's (LiveBookTest, TakebackTest, WatchedReviewTest), and so is
 * the real engine's.
 */
class AnalyserTest {

    private fun line(vararg moves: String, score: Double = 0.5, depth: Int = 18) =
        Line(moves = moves.toList(), depth = depth, expectedScore = score)

    private fun board(vararg sans: String): Board {
        val board = Board()
        for (san in sans) board.push(board.parseSan(san)!!)
        return board
    }

    private fun after(vararg sans: String): String = board(*sans).fen()

    private fun analyser(engines: List<ScriptEngine>, threats: Boolean = false, clock: SearchClock? = null): Analyser {
        val annotator = GameAnnotator(engines[0], depth = 18, multipv = 6, withThreats = threats)
        if (clock == null) return Analyser(annotator, engines = engines)
        return Analyser(annotator, engines = engines, ranMs = { clock.searched }, schedule = clock::schedule)
    }

    private fun entered(engine: GatedEngine, fen: String) =
        engine.entered.getValue(fen).await(HANG_MS, TimeUnit.MILLISECONDS)

    /** Every search the engines made: the position, the lines asked for and the depth. */
    private fun searches(engines: List<ScriptEngine>): List<Triple<String, Int, Int?>> =
        engines.flatMap { e -> synchronized(e.calls) { e.calls.indices.map { Triple(e.calls[it], e.multipvs[it], e.depths[it]) } } }

    /** The moves submitted one after another from the start, as the glue submits them. */
    private fun play(analyser: Analyser, sans: List<String>, gameId: String = "g"): Board {
        val board = Board()
        for ((i, san) in sans.withIndex()) {
            val move = board.parseSan(san)!!
            analyser.submitMove(gameId, board, move, i + 1)
            board.push(move)
        }
        return board
    }

    /** [body] with the analyser's workers running, closed after however it ends. */
    private fun <T> running(analyser: Analyser, body: () -> T): T {
        analyser.start()
        try {
            return body()
        } finally {
            analyser.close()
        }
    }

    /**
     * How long a search has run, moved by hand (the desktop's `ManualClock` given to the
     * analyser): the decided rule's checks, scheduled, run when [advance] passes their time, on
     * the test's thread.
     */
    private class SearchClock {
        @Volatile var ran = 0L

        /** How far the engine's own clock is behind the scheduler's (it starts at the engine's `go`). */
        @Volatile var lag = 0L
        private val tasks = ArrayList<Triple<Long, () -> Unit, BooleanArray>>()

        /** How long the search has run on its engine's clock. */
        val searched: Long get() = maxOf(0L, ran - lag)

        fun schedule(ms: Long, task: () -> Unit): () -> Unit {
            val cancelled = BooleanArray(1)
            synchronized(tasks) { tasks.add(Triple(ran + ms, task, cancelled)) }
            return { synchronized(tasks) { cancelled[0] = true } }
        }

        fun advance(ms: Long) {
            ran += ms
            val due = synchronized(tasks) {
                tasks.filter { (at, _, cancelled) -> at <= ran && !cancelled[0] }.also { tasks.removeAll(it) }
            }
            for ((_, task, _) in due) task()
        }
    }

    /**
     * An engine whose searches of the positions in [broken] (FENs; every position when none is
     * given) fail as a dead engine's do.
     */
    private open class Broken(answers: Map<String, List<Line>> = emptyMap(), private val broken: Set<String> = emptySet()) : ScriptEngine(answers) {
        override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
            if (broken.isEmpty() || board.fen() in broken) throw RuntimeException("engine gone")
            return super.analyse(board, depth, timeS, multipv, pvPlies, carryBest, onDepth)
        }
    }

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
        running(analyser) {
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
        }
        val searched = searches(engines)
        assertEquals(searched.sortedBy { it.toString() }.toString(), searched.toSet().size, searched.size)
        assertEquals(setOf(1 to 18, 6 to EARLY_DEPTH), searched.map { it.second to it.third }.toSet())
        assertEquals(searched.count { it.second == 1 }, analyser.searches)
    }

    /**
     * `test_a_position_is_whole_from_its_best_line_and_its_candidates_own_searches`: the best line
     * is the position's own search, first -- it picks the best move more often than the
     * candidates' searches (231 against 222 of 300 against a depth-24 truth), and the engine's own
     * move is never charged a loss for a candidate scoring higher -- then each other candidate
     * from its own position's one-line search, turned to the side to move, best first.
     */
    @Test
    fun aPositionIsWholeFromItsBestLineAndItsCandidatesOwnSearches() {
        val engine = ScriptEngine(
            mapOf(
                START to listOf(line("e2e4", "e7e5", score = 0.55), line("d2d4", score = 0.5)),
                after("e4") to listOf(line("e7e5", score = 0.40)), // higher than the best line says: not taken
                after("d4") to listOf(line("d7d5", score = 0.48)),
            ),
        )
        val analyser = analyser(listOf(engine))
        val watch = AnalyserWatch(analyser)
        val told = running(analyser) {
            analyser.submitPosition("g", Board(), 0)
            watch.settled().filterIsInstance<Analysis>()
        }
        val whole = told.last()
        assertFalse(whole.provisional)
        assertEquals(listOf(listOf("e2e4", "e7e5"), listOf("d2d4", "d7d5")), whole.lines.map { it.moves })
        assertEquals(0.55, whole.lines[0].expectedScore, 1e-9)
        assertEquals(0.52, whole.lines[1].expectedScore, 1e-9)
    }

    /**
     * `test_a_position_is_told_at_once_and_then_whole`: the first answer comes at once, one
     * search's lines -- the best line while the hint is not in, then the hint's at EARLY_DEPTH,
     * never the two set against each other -- each provisional; then the position whole; with
     * the side searches, whole once they are in too. Each stage once.
     */
    @Test
    fun aPositionIsToldAtOnceAndThenWhole() {
        val lines = mapOf(START to listOf(line("e2e4"), line("d2d4")))
        for (threats in listOf(false, true)) {
            val engine = GatedEngine(lines) // its lines at the depth asked for
            val analyser = analyser(listOf(engine), threats = threats)
            val watch = AnalyserWatch(analyser)
            val told = running(analyser) {
                analyser.submitPosition("g", Board(), 0)
                watch.settled().filterIsInstance<Analysis>()
            }
            assertEquals("the best line alone, first", listOf(18), told[0].lines.map { it.depth })
            assertEquals("then the hint's", setOf(EARLY_DEPTH), told[1].lines.map { it.depth }.toSet())
            assertEquals(List(told.size - 1) { true } + false, told.map { it.provisional })
            assertEquals("${told.map { it.lines }}", if (threats) 4 else 3, told.size)
            assertTrue(told.last().lines.all { it.depth == 18 })
        }
    }

    /**
     * `test_a_move_not_among_the_lines_waits_for_the_position_after`: a move none of the
     * candidates is judged from the position after's best line -- the game's next position,
     * searched anyway -- so its verdict waits for that search.
     */
    @Test
    fun aMoveNotAmongTheLinesWaitsForThePositionAfter() {
        val engine = GatedEngine(
            mapOf(START to listOf(line("e2e4", score = 0.60)), after("a3") to listOf(line("e7e5", score = 0.55))),
            gated = listOf(after("a3")),
        )
        val analyser = analyser(listOf(engine))
        val watch = AnalyserWatch(analyser)
        val got = running(analyser) {
            analyser.submitPosition("g", Board(), 0)
            watch.idle()
            analyser.submitMove("g", Board(), Move.fromUci("a2a3"), 1)
            assertTrue(entered(engine, after("a3")))
            assertEquals("judged before the position after was searched", 1, synchronized(analyser.lock) { analyser.nextVerdict })
            engine.release(after("a3"))
            watch.settled()
        }
        val verdict = got.filterIsInstance<MoveAssessment>().first()
        assertEquals(0.60 - (1 - 0.55), verdict.expectedLoss, 1e-9)
        assertTrue("${verdict.label}", verdict.label != Label.BEST)
    }

    /**
     * `test_the_engines_own_move_is_judged_without_waiting_for_the_position_after`: the best
     * line's move is judged from that line, the position's own search: the search of the position
     * it leads to -- the game's likeliest next, searched all the same -- is not waited for.
     */
    @Test
    fun theEnginesOwnMoveIsJudgedWithoutWaitingForThePositionAfter() {
        val engine = GatedEngine(mapOf(START to listOf(line("e2e4", score = 0.6))), gated = listOf(after("e4")))
        val analyser = analyser(listOf(engine))
        val watch = AnalyserWatch(analyser)
        running(analyser) {
            analyser.submitPosition("g", Board(), 0)
            val key = cacheKey(Board())
            watch.until("the start whole") { synchronized(analyser.lock) { analyser.nodes.getValue(key).whole != null } }
            analyser.submitMove("g", Board(), Move.fromUci("e2e4"), 1)
            // Judged on the move, the position after it still held.
            val verdicts = analyser.poll().filterIsInstance<MoveAssessment>()
            assertEquals(listOf(Triple(1, 0.0, Label.BEST)), verdicts.map { Triple(it.ply, it.expectedLoss, it.label) })
            engine.release(after("e4"))
            watch.idle()
        }
    }

    /**
     * `test_a_decided_search_running_long_stands_in_for_the_verdict`: Tai, 2026-10-05: a
     * best-line search still running DECIDED_AFTER_MS in whose deepest finished depth
     * (EARLY_DEPTH at least) the position is decided has that depth's score judge the move, once;
     * the full depth, when it comes, judges nothing again and is what is told. Undecided, or too
     * shallow, the move waits for the full depth.
     */
    @Test
    fun aDecidedSearchRunningLongStandsInForTheVerdict() {
        for ((reported, stands) in listOf(
            line("d8h4", score = 0.9, depth = 15) to true, // decided: the stand-in judges
            line("e7e5", score = 0.5, depth = 15) to false, // undecided: waited for
            line("d8h4", score = 0.9, depth = EARLY_DEPTH - 1) to false, // too shallow: waited for
        )) {
            val clock = SearchClock()
            val engine = GatedEngine(
                mapOf(START to listOf(line("e2e4", score = 0.6)), after("h3") to listOf(line("e7e5", score = 0.45))),
                gated = listOf(after("h3")),
                reports = mapOf(after("h3") to listOf(reported)),
            )
            val analyser = analyser(listOf(engine), clock = clock)
            val watch = AnalyserWatch(analyser)
            var judged: List<MoveAssessment> = emptyList()
            val got = running(analyser) {
                analyser.submitPosition("g", Board(), 0)
                watch.idle()
                analyser.submitMove("g", Board(), Move.fromUci("h2h3"), 1)
                // Held: its depths reported, the rule's check scheduled before the search began.
                assertTrue(entered(engine, after("h3")))
                clock.advance(DECIDED_AFTER_MS)
                judged = analyser.poll().filterIsInstance<MoveAssessment>()
                assertEquals("$reported", if (stands) 1 else 0, judged.size)
                engine.release(after("h3"))
                watch.settled()
            }
            val verdicts = judged + got.filterIsInstance<MoveAssessment>()
            assertEquals("judged once", listOf(1), verdicts.map { it.ply })
            val loss = 0.60 - (1 - (if (stands) 0.9 else 0.45))
            assertEquals(loss, verdicts[0].expectedLoss, 1e-9)
            val told = got.filterIsInstance<Analysis>().filter { it.ply == 1 }
            assertEquals("the full depth told", 0.45, told.last().lines[0].expectedScore, 1e-9)
        }
    }

    /**
     * The phone's own: the engine's clock starts at its `go`, after the rule's timer (and stands
     * still while the voice pauses it), so at DECIDED_AFTER_MS the search has run a little less.
     * The rule looks again once it has run its time -- the decided move is judged then, not left
     * to the full depth (the phone's audit of 2026-10-05, round 2).
     */
    @Test
    fun theDecidedRuleWaitsForTheEnginesOwnClock() {
        val clock = SearchClock().apply { lag = 100 }
        val engine = GatedEngine(
            mapOf(START to listOf(line("e2e4", score = 0.6)), after("h3") to listOf(line("e7e5", score = 0.45))),
            gated = listOf(after("h3")),
            reports = mapOf(after("h3") to listOf(line("d8h4", score = 0.9, depth = 15))),
        )
        val analyser = analyser(listOf(engine), clock = clock)
        val watch = AnalyserWatch(analyser)
        running(analyser) {
            analyser.submitPosition("g", Board(), 0)
            watch.idle()
            analyser.submitMove("g", Board(), Move.fromUci("h2h3"), 1)
            assertTrue(entered(engine, after("h3")))
            clock.advance(DECIDED_AFTER_MS)
            assertEquals("judged before the engine's own second", 0, analyser.poll().count { it is MoveAssessment })
            clock.advance(100)
            assertEquals("never looked at again", listOf(1), analyser.poll().filterIsInstance<MoveAssessment>().map { it.ply })
            engine.release(after("h3"))
            watch.idle()
        }
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
        val got = running(analyser) {
            val board = play(analyser, listOf("e4", "e5"))
            board.pop()
            val key = cacheKey(board)
            watch.until("e4 whole") { synchronized(analyser.lock) { analyser.nodes[key]?.whole != null } }
            assertEquals("e5 judged before e4", 1, synchronized(analyser.lock) { analyser.nextVerdict })
            for (engine in engines) engine.release(START)
            watch.settled()
        }
        assertEquals(listOf(1, 2), got.filterIsInstance<MoveAssessment>().map { it.ply })
    }

    /**
     * `test_a_takeback_says_nothing_more_of_the_withdrawn_moves`: Tai, 2026-09-23: "takebacks
     * make the program lag behind by a lot". The moves taken back get no verdict and no analysis,
     * and nothing is owed for them; a search made only for them stops; the move kept is judged
     * and its positions told.
     */
    @Test
    fun aTakebackSaysNothingMoreOfTheWithdrawnMoves() {
        val gated = listOf(after("e4", "e5"))
        val engine = GatedEngine(gated = gated)
        val analyser = analyser(listOf(engine))
        val watch = AnalyserWatch(analyser)
        // The moves in before the worker starts: the first verdict's needs, then the second's,
        // whose position after e5 holds.
        play(analyser, listOf("e4", "e5", "Nf3"))
        val got = running(analyser) {
            assertTrue("the position after e5 never searched", entered(engine, gated[0]))
            analyser.withdraw("g", 1)
            watch.until("the withdrawn move's search stopped") { engine.halted.toList() == gated }
            watch.settled()
        }
        val late = got.filter { (it is MoveAssessment && it.ply >= 2) || (it is Analysis && it.ply >= 2) }
        assertEquals(emptyList<Event>(), late)
        assertEquals(listOf(1), got.filterIsInstance<MoveAssessment>().map { it.ply })
        assertEquals(setOf(0, 1), got.filterIsInstance<Analysis>().filter { !it.provisional }.map { it.ply }.toSet())
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
        var searched = 0
        val again = running(analyser) {
            analyser.submitPosition("g", Board(), 0)
            watch.quiet()
            analyser.poll()
            searched = engine.calls.size
            analyser.reset()
            analyser.submitPosition("h", Board(), 0)
            watch.quiet()
            analyser.poll()
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
        val analyser = Analyser(GameAnnotator(Broken(), depth = 6, withThreats = false))
        val watch = AnalyserWatch(analyser, failuresOk = true)
        var failed = 0
        running(analyser) {
            analyser.submitMove("g", Board(), Move.fromUci("e2e4"), 1)
            val got = watch.settled()
            failed = analyser.failures
            assertEquals("each position's best line and hint, of both positions", 4, failed)
            assertTrue("${analyser.lastError}", analyser.lastError?.contains("engine gone") == true)
            assertTrue(got.none { it is MoveAssessment })
            analyser.submitPosition("h", board("d4"), 0)
            watch.idle()
        }
        assertEquals("the next position tried, and failed alike", failed + 2, analyser.failures)
    }

    /**
     * `test_a_candidate_that_fails_is_left_out_of_the_whole`: one candidate's own search fails:
     * the position is whole without it, told final, and nothing is owed.
     */
    @Test
    fun aCandidateThatFailsIsLeftOutOfTheWhole() {
        val engine = Broken(
            mapOf(START to listOf(line("e2e4", score = 0.6), line("d2d4", score = 0.5))),
            broken = setOf(after("d4")),
        )
        val analyser = analyser(listOf(engine))
        val watch = AnalyserWatch(analyser, failuresOk = true)
        val told = running(analyser) {
            analyser.submitPosition("g", Board(), 0)
            watch.settled().filterIsInstance<Analysis>()
        }
        assertEquals(1, analyser.failures)
        assertFalse(told.last().provisional)
        assertEquals(listOf("e2e4"), told.last().lines.map { it.moves[0] })
    }

    /**
     * `test_side_searches_that_fail_leave_the_position_whole_and_final`: with threats on, the
     * side searches failing: the position is told final with its lines, kept for the review, and
     * nothing is owed. (The desktop breaks `with_side` itself; here the engine fails every
     * position but the start's and its candidates', which the side searches never are.)
     */
    @Test
    fun sideSearchesThatFailLeaveThePositionWholeAndFinal() {
        val answers = mapOf(START to listOf(line("e2e4"), line("d2d4")))
        val searched = setOf(START, after("e4"), after("d4"))
        val engine = object : ScriptEngine(answers) {
            override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
                if (board.fen() !in searched) throw RuntimeException("side searches gone")
                return super.analyse(board, depth, timeS, multipv, pvPlies, carryBest, onDepth)
            }
        }
        val analyser = analyser(listOf(engine), threats = true)
        val watch = AnalyserWatch(analyser, failuresOk = true)
        val told = running(analyser) {
            analyser.submitPosition("g", Board(), 0)
            watch.settled().filterIsInstance<Analysis>()
        }
        assertEquals(analyser.lastError, 1, analyser.failures)
        assertFalse(told.last().provisional)
        assertTrue(told.last().lines.isNotEmpty())
        assertEquals(1, analyser.measuredPath("g").size)
    }

    /**
     * `test_a_position_with_one_legal_move_is_never_searched`: in check with one way out: its
     * best line is that move's, from the position it leads to; the position itself is never
     * searched, and it is told and judged as any other.
     */
    @Test
    fun aPositionWithOneLegalMoveIsNeverSearched() {
        val one = "k7/p7/8/8/8/8/6q1/7K w - - 0 1" // the queen gives check; only Kxg2
        val engine = ScriptEngine()
        val analyser = analyser(listOf(engine))
        val watch = AnalyserWatch(analyser)
        val told = running(analyser) {
            analyser.submitPosition("g", Board(one), 0)
            watch.settled().filterIsInstance<Analysis>()
        }
        assertFalse(one in engine.calls)
        assertTrue(told.last().lines.isNotEmpty())
        assertEquals("h1g2", told.last().lines[0].moves[0])
        assertFalse(told.last().provisional)
        assertEquals(engine.multipvs.count { it == 1 }, analyser.searches)
    }

    /**
     * `test_a_candidate_giving_check_with_one_reply_is_searched_through`: a candidate check with a
     * single reply: its position takes its best line from the reply's, two moves down the game --
     * wanted as the plan wants it, never refused and asked again (the audit of 2026-10-05, round
     * 2: a worker spun on it and every verdict after it waited for good). For the game's
     * position, a verdict on a move from it, and a question.
     */
    @Test
    fun aCandidateGivingCheckWithOneReplyIsSearchedThrough() {
        val asked = QG2.replace("p7/8/8/8/6q1", "1p6/8/8/8/6q1") // the same check, the pawn on b7
        val lines = mapOf(
            QG2 to listOf(line("a7a6", score = 0.5), line("g3g2", score = 0.4)),
            asked to listOf(line("b7b6", score = 0.5), line("g3g2", score = 0.4)),
        )
        val engines = listOf(ScriptEngine(lines), ScriptEngine(lines))
        val analyser = analyser(engines)
        val watch = AnalyserWatch(analyser)
        var verdicts: List<MoveAssessment> = emptyList()
        var answer: Measured? = null
        val whole = running(analyser) {
            val board = Board(QG2)
            analyser.submitPosition("g", board, 0)
            val whole = watch.settled().filterIsInstance<Analysis>().last()
            analyser.submitMove("g", board, Move.fromUci("a7a6"), 1)
            verdicts = watch.settled().filterIsInstance<MoveAssessment>()
            answer = analyser.request(Board(asked))
            watch.quiet()
            whole
        }
        assertFalse(whole.provisional)
        assertTrue("g3g2" in whole.lines.map { it.moves[0] })
        assertEquals(listOf(1), verdicts.map { it.ply })
        assertTrue("g3g2" in answer!!.lines.map { it.moves[0] })
    }

    /**
     * `test_a_decided_stand_in_reaches_a_move_into_a_forced_reply`: a move into a check with one
     * reply, the position after the reply decided while its search runs on: the stand-in is
     * carried back up the forced move, as its best line would be, and the move is judged at
     * DECIDED_AFTER_MS rather than at the full depth.
     */
    @Test
    fun aDecidedStandInReachesAMoveIntoAForcedReply() {
        val clock = SearchClock()
        val afterReply = Board(QG2).also { it.push(Move.fromUci("g3g2")); it.push(Move.fromUci("h1g2")) }.fen()
        val engine = GatedEngine(
            mapOf(QG2 to listOf(line("a7a6", score = 0.5))), // Qg2+ not among the candidates: judged from after
            gated = listOf(afterReply),
            reports = mapOf(afterReply to listOf(line("a6a5", score = 0.1, depth = 15))), // Black lost
        )
        val analyser = analyser(listOf(engine), clock = clock)
        val watch = AnalyserWatch(analyser)
        var judged: List<MoveAssessment> = emptyList()
        running(analyser) {
            val board = Board(QG2)
            analyser.submitPosition("g", board, 0)
            watch.idle()
            analyser.submitMove("g", board, Move.fromUci("g3g2"), 1)
            assertTrue(entered(engine, afterReply))
            clock.advance(DECIDED_AFTER_MS)
            judged = analyser.poll().filterIsInstance<MoveAssessment>()
            engine.release(afterReply)
            watch.settled()
        }
        assertEquals("not judged at the stand-in", listOf(1), judged.map { it.ply })
        assertEquals(0.5 - 0.1, judged[0].expectedLoss, 1e-9)
    }

    /**
     * `test_a_mate_on_the_board_is_judged_told_and_owes_nothing`: Scholar's mate played through,
     * threats on: the mating move judged, the mated position told final with no lines, nothing
     * owed.
     */
    @Test
    fun aMateOnTheBoardIsJudgedToldAndOwesNothing() {
        val analyser = analyser(listOf(ScriptEngine()), threats = true)
        val watch = AnalyserWatch(analyser)
        var board = Board()
        val got = running(analyser) {
            board = play(analyser, listOf("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"))
            watch.settled()
        }
        assertTrue(board.isCheckmate())
        assertEquals((1..7).toList(), got.filterIsInstance<MoveAssessment>().map { it.ply })
        val mated = got.filterIsInstance<Analysis>().filter { it.ply == 7 }
        assertEquals(emptyList<Line>(), mated.last().lines)
        assertFalse(mated.last().provisional)
    }

    /** `test_an_illegal_move_is_refused_and_owes_nothing`. */
    @Test
    fun anIllegalMoveIsRefusedAndOwesNothing() {
        val analyser = analyser(listOf(ScriptEngine()))
        val watch = AnalyserWatch(analyser, failuresOk = true)
        val got = running(analyser) {
            analyser.submitMove("g", Board(), Move.fromUci("e2e5"), 1)
            watch.settled()
        }
        assertEquals(1, analyser.failures)
        assertEquals(emptyList<Event>(), got)
        assertEquals(0, analyser.pending)
    }

    /**
     * `test_the_positions_kept_are_trimmed_and_the_workers_go_on`: past the positions kept the
     * oldest go -- never the game's, its moves', a worker's or their candidates -- and a candidate
     * let go is made again when wanted: no worker ever dies on a position it no longer finds (the
     * audit of 2026-10-05).
     */
    @Test
    fun thePositionsKeptAreTrimmedAndTheWorkersGoOn() {
        val keep = 40
        val analyser = analyser(listOf(ScriptEngine(), ScriptEngine()))
        analyser.keepNodes = keep
        val watch = AnalyserWatch(analyser)
        val sans = listOf("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4", "cxd4", "Bb4+")
        val got = running(analyser) {
            val board = Board()
            for ((i, san) in sans.withIndex()) {
                val move = board.parseSan(san)!!
                analyser.submitMove("g", board, move, i + 1)
                board.push(move)
                watch.idle()
                assertTrue(synchronized(analyser.lock) { analyser.nodes.size } < 2 * keep)
            }
            board.pop()
            analyser.withdraw("g", sans.size - 1) // a takeback, and a move played again
            analyser.submitMove("g", board, board.parseSan("Bb6")!!, sans.size)
            val got = watch.settled()
            assertTrue(analyser.threads.all { it.isAlive })
            got
        }
        assertEquals(sans.size, got.filterIsInstance<MoveAssessment>().last().ply)
        assertEquals(0, analyser.pending)
    }

    /**
     * `test_the_workers_take_the_verdicts_then_the_coach_then_the_game`: who waits on a search,
     * first to last: the next verdict (the word on the move just played), the coach's question
     * (the player waiting on it), the position the game is on, the game's other positions.
     */
    @Test
    fun theWorkersTakeTheVerdictsThenTheCoachThenTheGame() {
        val analyser = analyser(listOf(ScriptEngine(mapOf(START to listOf(line("e2e4"))))))
        val asked = board("d4")
        val asking = Thread({ analyser.request(asked) }, "the coach").apply { isDaemon = true }
        try {
            analyser.submitPosition("g", Board(), 0)
            asking.start()
            synchronized(analyser.lock) {
                val deadline = System.nanoTime() + HANG_MS * 1_000_000
                while (analyser.requests.isEmpty()) {
                    val left = (deadline - System.nanoTime()) / 1_000_000
                    assertTrue("the coach never asked", left > 0)
                    analyser.lock.wait(left)
                }
                assertEquals("the coach's first", Analyser.Job(Analyser.BEST, cacheKey(asked), 0), analyser.pick())
            }
            analyser.submitMove("g", Board(), Move.fromUci("e2e4"), 1)
            val jobs = synchronized(analyser.lock) {
                assertEquals(Analyser.Job(Analyser.BEST, cacheKey(Board()), 0), analyser.pick())
                analyser.agenda()
            }
            val coachJob = Analyser.Job(Analyser.BEST, cacheKey(asked), 0)
            val gameJob = Analyser.Job(Analyser.HINT, cacheKey(board("e4")), 0)
            assertTrue("the coach's after the game's", jobs.indexOf(coachJob) < jobs.indexOf(gameJob))
        } finally {
            analyser.close()
            asking.join(HANG_MS)
        }
        assertFalse(asking.isAlive)
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
        var searched = 0
        var first: List<Line> = emptyList()
        val again = running(analyser) {
            first = searcher.analyse(board("d4"))
            searched = engine.calls.size
            searcher.analyse(board("d4"))
        }
        assertEquals(first, again)
        assertEquals(listOf("d7d5", "g8f6"), first.map { it.moves[0] })
        assertEquals(searched, engine.calls.size)
        assertTrue(analyser.requests.isEmpty())
    }

    /**
     * `test_the_coach_asking_for_one_line_does_not_wait_for_the_whole`: best only: a probe that
     * reads the first line alone gets the position's best line while its candidates are still
     * held -- the whole is not waited for.
     */
    @Test
    fun theCoachAskingForOneLineDoesNotWaitForTheWhole() {
        val held = after("d4", "g8f6")
        val engine = GatedEngine(mapOf(after("d4") to listOf(line("d7d5", score = 0.5), line("g8f6", score = 0.45))), gated = listOf(held))
        val analyser = analyser(listOf(engine))
        analyser.start()
        try {
            val got = PoolSearcher(analyser, timeoutMs = null).analyse(board("d4"), bestOnly = true)
            assertEquals(listOf("d7d5"), got.map { it.moves[0] })
            assertNull("the whole waited for", synchronized(analyser.lock) { analyser.nodes.getValue(cacheKey(board("d4"))).whole })
        } finally {
            engine.release(held)
            analyser.close()
        }
    }

    /**
     * `test_settings_saved_mid_session_reach_every_workers_engine`: hash, the lines' length and
     * the scoring saved: on each worker's engine (the audit of 2026-10-05, round 2: the route ran
     * in no test).
     */
    @Test
    fun settingsSavedMidSessionReachEveryWorkersEngine() {
        val engines = listOf(ScriptEngine(), ScriptEngine())
        val analyser = analyser(engines)
        analyser.reconfigure(hashMb = 128, multipv = 3, pvPlies = 8)
        assertEquals(listOf(Triple(128, 3, 8), Triple(128, 3, 8)), engines.map { Triple(it.hashMb, it.multipv, it.pvPlies) })
    }

    /**
     * `test_a_question_is_never_left_hanging`: the coach's wait is bounded however the search
     * goes: past its time it gets what is known (here nothing) and its question is dropped; a
     * question whose search fails is answered at once; one waiting when the analyser closes is
     * let go; one asked of a closed analyser is answered at once.
     */
    @Test
    fun aQuestionIsNeverLeftHanging() {
        val gated = after("c4")
        val engine = GatedEngine(gated = listOf(gated))
        val analyser = analyser(listOf(engine))
        val answers = Collections.synchronizedList(ArrayList<Measured?>())
        val waiting = Thread({ answers.add(analyser.request(board("c4"))) }, "the coach").apply { isDaemon = true }
        analyser.start()
        try {
            assertNull(analyser.request(board("c4"), timeoutMs = 0))
            assertTrue(synchronized(analyser.lock) { analyser.requests.isEmpty() })
            waiting.start()
            synchronized(analyser.lock) {
                val deadline = System.nanoTime() + HANG_MS * 1_000_000
                while (analyser.requests.isEmpty()) {
                    val left = (deadline - System.nanoTime()) / 1_000_000
                    assertTrue("the coach never asked", left > 0)
                    analyser.lock.wait(left)
                }
            }
        } finally {
            engine.release(gated)
            analyser.close()
        }
        waiting.join(HANG_MS)
        assertFalse("let go at the close", waiting.isAlive)
        assertEquals(1, answers.size)
        assertNull(analyser.request(board("e4")))

        val broken = Analyser(GameAnnotator(Broken(), depth = 6, withThreats = false))
        running(broken) { assertNull(broken.request(board("d4"))) }
        assertEquals(2, broken.failures)
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
     * lines saved while a search runs: what it finds is not kept -- even a search that runs on to
     * its end -- and the game's position is measured again and told from the new search.
     */
    @Test
    fun settingsChangedMidSearchKeepNothingMeasuredTheOldWay() {
        /**
         * A search no stop reaches: it runs to its end, the old way -- scoring as the settings
         * were when it began, so what it found is told apart from the new search's.
         */
        class Stubborn : GatedEngine(mapOf(START to listOf(line("e2e4", score = 0.3))), gated = listOf(START)) {
            override fun interrupt() = Unit

            override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
                val began = answers[board.fen()]?.map { it.expectedScore } ?: emptyList()
                val lines = super.analyse(board, depth, timeS, multipv, pvPlies, carryBest, onDepth)
                if (began.isEmpty()) return lines
                return lines.map { it.copy(expectedScore = began[0]) }
            }
        }
        val engine = Stubborn()
        val analyser = analyser(listOf(engine))
        val watch = AnalyserWatch(analyser)
        val told = running(analyser) {
            analyser.submitPosition("g", Board(), 0)
            assertTrue(entered(engine, START))
            analyser.forgetMeasured()
            engine.answers[START] = listOf(line("e2e4", score = 0.7)) // the new settings' answer
            engine.release(START)
            watch.settled().filterIsInstance<Analysis>()
        }
        assertEquals(2, searches(listOf(engine)).count { it.first == START && it.second == 1 })
        assertTrue(told.isNotEmpty())
        assertFalse(told.last().provisional)
        assertEquals("the old search kept", 0.7, told.last().lines[0].expectedScore, 1e-9)
        assertEquals(engine.multipvs.count { it == 1 } - 1, analyser.searches)
        assertNotNull(analyser.measuredPath("g").firstOrNull())
    }

    private companion object {
        val START: String = Board().fen()

        /** Black to move; Qg2+ leaves White one legal move, Kxg2. */
        const val QG2 = "k7/p7/8/8/8/6q1/8/7K b - - 0 1"
    }
}
