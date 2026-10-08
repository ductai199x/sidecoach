package com.sidecoach.coach

import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.BLACK
import com.sidecoach.chess.KING
import com.sidecoach.chess.Piece
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.parseSquare
import com.sidecoach.events.Analysis
import com.sidecoach.events.ClockObservation
import com.sidecoach.events.CoachMessage
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.GameEnded
import com.sidecoach.events.GameReview
import com.sidecoach.events.GameStarted
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.ProfileEvidence
import com.sidecoach.events.PuzzleOutcome
import com.sidecoach.events.PuzzleRef
import com.sidecoach.events.Recommendation
import com.sidecoach.events.Resync
import com.sidecoach.events.Severity
import com.sidecoach.events.Takeback
import com.sidecoach.events.Trigger
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.Completion
import com.sidecoach.llm.LLMProvider
import com.sidecoach.llm.OfflineProvider
import com.sidecoach.llm.StreamCallback
import com.sidecoach.llm.ToolCall
import com.sidecoach.llm.ToolRound
import com.sidecoach.llm.ToolsProvider
import org.json.JSONArray
import org.json.JSONObject
import com.sidecoach.profile.Causes
import com.sidecoach.vision.Pieces
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * When the coach speaks: the desktop's `tests/test_coach.py` for desktop 88b84a6, driven the
 * way the desktop drives it -- a scripted model, the events by hand. A question asked before
 * the engine has looked is answered when it has; nothing unasked while the player thinks until
 * they are stuck, a decisive moment included (Tai, 2026-10-01); and a hint never said twice
 * along the combination the player is carrying out.
 */
class CoachTest {

    /** Answers from a script, and remembers every prompt it was given. */
    private class ScriptedProvider(replies: List<String>) : LLMProvider {
        override val name = "scripted"

        /** What it says next; a test may set them afresh, as the desktop's `provider.replies = ...`. */
        val replies = ArrayDeque(replies)
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
            val text = synchronized(replies) { replies.removeFirstOrNull() } ?: "Keep looking."
            return Completion(text = text)
        }
    }

    private fun coach(replies: List<String>): Pair<Coach, ScriptedProvider> {
        val provider = ScriptedProvider(replies)
        val coach = Coach(provider, policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
        return coach to provider
    }

    private fun start(coach: Coach, userSide: Pieces.Side = Pieces.Side.WHITE) {
        coach.onEvent(GameStarted(gameId = "g", userSide = userSide, startFen = Board().fen()), 0.0)
    }

    private fun messages(coach: Coach): List<CoachMessage> {
        coach.speaker.runPending()
        return coach.poll()
    }

    /** Waits for the coach's background threads of that name to end: the event, not a time. */
    private fun joinThreads(name: String) {
        for (thread in Thread.getAllStackTraces().keys) {
            if (thread.name == name) thread.join()
        }
    }

    /**
     * `stuck` (desktop `tests/coaching.py`): the player stuck on the position the stuck limit
     * long: the coach's first word unasked (the only one while they think -- Tai, 2026-10-01),
     * and what it then has for them.
     */
    private fun stuck(coach: Coach): List<CoachMessage> {
        val since = coach.policy.stuckFrom
        assertNotNull(since)
        coach.tick(since!! + coach.policy.stuckThreshold())
        return messages(coach)
    }

    /**
     * The engine's look at [board]: two of its legal moves, captures first, the first the best;
     * [provisional] the early depth, the full one to follow.
     */
    private fun twoLines(board: Board, ply: Int, provisional: Boolean = false): Analysis {
        val (first, second) = board.legalMoves()
            .sortedWith(compareBy<Move>({ !board.isCapture(it) }, { it.uci() }))
            .take(2)
        val depth = if (provisional) 12 else 18
        return Analysis(
            gameId = "g", fen = board.fen(), ply = ply,
            lines = listOf(
                Line(moves = listOf(first.uci()), depth = depth, expectedScore = 0.7),
                Line(moves = listOf(second.uci()), depth = depth, expectedScore = 0.5),
            ),
            decisive = false, onlyMove = false, provisional = provisional,
        )
    }

    /**
     * Every way a game begins, every way its position changes, every way the coach is asked to
     * speak: the model is given the position the player faces -- its board, the side to move,
     * the engine's lines, the facts -- and no other as the position now. At a puzzle's start, a
     * joined game's, a re-read board's or after a takeback to the start, it was given none, or
     * the withdrawn move's (Tai, 2026-10-01). Asked before the engine has looked, the question
     * is answered from its early depth, and the position now shows that depth's lines and facts:
     * only the full depth reached the record, and the model was told "Engine lines: not in yet."
     * with no facts (the coach review, 2026-10-01). Only the position now has its legal moves:
     * the setting, the takeback and the re-read show a board as where the record starts or went
     * back to, and after a re-read the setting's board was a second "position as it stands".
     * `test_conversation.py`.
     */
    @Test
    fun thePositionNowIsTheBoardThePlayerFaces() {
        val joined = "4r1k1/p6p/6p1/2Q5/8/N1P2q1P/PP1K1P2/R6R b - - 0 1"
        val reread = mapOf(
            Pieces.Side.WHITE to "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            Pieces.Side.BLACK to "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/2N5/PPPP1PPP/R1B1K1NR b KQkq - 5 4",
        )
        val starts = mapOf(
            "game" to Triple(Board().fen(), Pieces.Side.WHITE, "game"),
            "joined" to Triple(joined, Pieces.Side.BLACK, "game"),
            "position" to Triple(joined, Pieces.Side.BLACK, "position"),
            "puzzle" to Triple(joined, Pieces.Side.BLACK, "puzzle"),
        )
        for ((name, start) in starts) for (state in listOf("start", "moved", "taken back", "re-read")) {
            for (trigger in listOf("stuck", "asked")) {
                val where = "$name, $state, $trigger"
                val (fen, side, kind) = start
                val (coach, provider) = coach(emptyList())
                coach.onEvent(GameStarted(gameId = "g", userSide = side, startFen = fen, kind = kind), 0.0)
                var board = Board(fen)
                // Asked, the position now's analysis lands after the question, at its early depth.
                val later = ArrayList<Pair<Board, Int>>()
                fun look(at: Board, ply: Int, ts: Double, now: Boolean = false) {
                    if (now && trigger == "asked") later.add(at.copy() to ply) else coach.onEvent(twoLines(at, ply), ts)
                }
                look(board, 0, 0.5, now = state == "start")
                var gone: String? = null
                if (state == "moved" || state == "taken back") {
                    for (ply in 1..2) {
                        val move = board.legalMoves().sortedBy { it.uci() }.first()
                        val san = board.san(move)
                        val before = board.fen()
                        board.push(move)
                        coach.onEvent(
                            MovePlayed(gameId = "g", ply = ply, san = san, uci = move.uci(), fenBefore = before, fenAfter = board.fen(), byUser = ply == 1),
                            ply.toDouble(),
                        )
                        look(board, ply, ply + 0.5, now = state == "moved" && ply == 2)
                    }
                    if (state == "taken back") {
                        gone = board.fen()
                        board = Board(fen)
                        coach.conversation.coachSaid("Look at the knight.") // of the withdrawn position
                        coach.onEvent(Takeback("g", 2, fen), 3.0)
                        coach.conversation.clock(60_000, 50_000)
                        assertTrue("$where: withdrawn moves", coach.conversation.records.all { it.whiteMs == null })
                    }
                } else if (state == "re-read") {
                    gone = fen
                    board = Board(reread.getValue(side))
                    coach.conversation.coachSaid("Look at the knight.") // of the board before the re-read
                    coach.onEvent(Resync("g", board.fen(), "re-read"), 3.0)
                    look(board, 0, 3.5, now = true)
                }
                assertTrue(where, coach.userToMove && coach.board.fen() == board.fen())
                if (trigger == "stuck") {
                    stuck(coach)
                } else {
                    coach.ask("what should I be thinking about here?", 10.0)
                    for ((at, ply) in later) {
                        messages(coach)
                        assertTrue("$where: the question waits for the engine's look", provider.prompts.isEmpty())
                        coach.onEvent(twoLines(at, ply, provisional = true), 10.5)
                    }
                    messages(coach)
                }
                assertTrue("$where: the model was asked", provider.prompts.isNotEmpty())
                val harness = provider.prompts.last().second.single().content
                assertEquals(where, 1, harness.split("# Position now\n").size - 1)
                val now = harness.substringAfter("# Position now\n").substringBefore("\n# Task\n")
                assertTrue(where, "FEN: ${board.fen()}" in now)
                if (gone != null) {
                    assertFalse("$where: a position no longer the game's", "FEN: $gone" in now)
                    assertFalse("$where: said of a position no longer the game's", "Look at the knight." in now)
                }
                assertTrue(where, (if (board.turn == WHITE) "White" else "Black") + " to move" in now)
                assertTrue(
                    "$where: one board to play on",
                    harness.split("Legal moves: ").size - 1 == 1 && "Legal moves: " in now,
                )
                val best = board.san(Move.fromUci(twoLines(board, 0).lines[0].moves[0]))
                val lines = now.substringAfter("## Engine lines").substringBefore("\n## ")
                assertTrue("$where: the engine's best line\n$lines", " $best  0.70" in lines.split("\n")[1])
                assertTrue("$where: the facts sheet", "\n## Facts (" in now)
                assertEquals("$where: the prompt check", emptyList<String>(), coach.harnessProblems.toList())
            }
        }
    }

    /**
     * The check run before every request: the puzzle start's prompt (no position now), a
     * withdrawn position, and the wrong colour are each named; a prompt built right passes.
     * `test_conversation.py`.
     */
    @Test
    fun thePromptCheckNamesWhatIsWrong() {
        val board = Board("4r1k1/p6p/6p1/2Q5/8/N1P2q1P/PP1K1P2/R6R b - - 0 1")
        val good = "# Setting\nYou coach Black. They see the board from Black's side.\n\n" +
            "# Position now\n## Board\nFEN: ${board.fen()}\nPieces -- ..."
        assertEquals(null, harnessProblem(good, board, Pieces.Side.BLACK))
        val settingOnly = good.substringBefore("\n\n# Position now")
        assertEquals("no position now (# Position now with its board)", harnessProblem(settingOnly, board, Pieces.Side.BLACK))
        val withdrawn = good.replace(board.fen(), Board().fen())
        assertTrue(
            "the position now is rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w" in
                (harnessProblem(withdrawn, board, Pieces.Side.BLACK) ?: "")
        )
        assertEquals("the setting does not say the player is White", harnessProblem(good, board, Pieces.Side.WHITE))
        // The engine's lines and the facts, when the coach has them for the position now: the
        // early depth was answered from and never shown (the coach review, 2026-10-01).
        val unread = good + "\n\n## Engine lines\nnot in yet.\n\n## On the board (complete lists)\n..."
        assertEquals("nothing known: nothing missing", null, harnessProblem(unread, board, Pieces.Side.BLACK))
        assertEquals(
            "the position now has no engine lines, though the coach has them",
            harnessProblem(unread, board, Pieces.Side.BLACK, lines = true),
        )
        val read = good + "\n\n## Engine lines (the player's move)\n  1. Qxf2+  0.70\n\n" +
            "## On the board (complete lists)\n...\n\n## Said here\n    EVENT: ## Facts (a quote)"
        assertEquals(null, harnessProblem(read, board, Pieces.Side.BLACK, lines = true))
        assertEquals(
            "the position now has no facts, though the coach has them",
            harnessProblem(read, board, Pieces.Side.BLACK, lines = true, facts = true),
        )
        val withFacts = read.replace("\n\n## On the board", "\n\n## Facts (the player to move)\nphase: x")
        assertEquals(null, harnessProblem(withFacts, board, Pieces.Side.BLACK, lines = true, facts = true))
    }

    /**
     * A question asked before the engine has looked at the position on the board is told to
     * wait a moment and answered when the analysis lands, never from an older position (Tai,
     * 2026-09-24: a hint pressed as a puzzle appeared got "I have not seen enough of this game
     * yet to say. Give me a move or two.").
     */
    @Test
    fun askingBeforeTheEngineHasLookedIsAnsweredWhenItHas() {
        val (coach, _) = coach(emptyList())
        start(coach)
        coach.ask("hint", 1.0)
        val out = messages(coach)
        assertEquals(listOf("One moment -- still looking at this position."), out.map { it.text })
        coach.ask("hint", 1.5) // asked again while waiting: one "moment" is enough
        assertEquals(emptyList<String>(), messages(coach).map { it.text })
        val start = Board()
        coach.onEvent(
            Analysis(
                gameId = "g",
                fen = start.fen(),
                ply = 0,
                lines = listOf(
                    Line(moves = listOf("e2e4", "e7e5", "g1f3"), depth = 18, expectedScore = 0.55),
                    Line(moves = listOf("d2d4", "d7d5"), depth = 18, expectedScore = 0.54),
                ),
                decisive = false,
                onlyMove = false,
            ),
            3.0,
        )
        val answered = messages(coach)
        assertTrue(
            answered.map { it.trigger }.toString(),
            answered.isNotEmpty() && answered.all { it.trigger == Trigger.USER_ASK },
        )
        assertTrue(answered[0].text, "One moment" !in answered[0].text)
    }

    /**
     * `test_asking_before_the_engine_has_looked_is_answered_when_it_has_even_early` (desktop
     * 316026b; the early half of it). Tai, 2026-09-28: "first usable answer is depth 12 ... the
     * computation continues until max depth of 18". A question asked before the engine has
     * looked is answered from the early analysis, the full depth still to come.
     */
    @Test
    fun anEarlyAnalysisAnswersAQuestionWaitingOnIt() {
        val (coach, _) = coach(emptyList())
        playToScholar(coach)
        coach.ask("hint", 6.5)
        assertEquals(listOf("One moment -- still looking at this position."), messages(coach).map { it.text })
        coach.onEvent(mateAnalysis(6, provisional = true), 7.0)
        val answered = messages(coach)
        assertTrue(
            answered.map { it.trigger }.toString(),
            answered.isNotEmpty() && answered.all { it.trigger == Trigger.USER_ASK },
        )
    }

    /**
     * `test_an_analysis_early_or_full_says_nothing_of_its_own` (desktop, 2026-10-01): what the
     * engine finds is no reason to speak while the player thinks: neither its early look nor the
     * full depth says a word, a decisive moment included (Tai, 2026-10-01); the stuck limit does.
     */
    @Test
    fun anAnalysisEarlyOrFullSaysNothingOfItsOwn() {
        val (coach, provider) = coach(
            listOf("You can mate right now: their king on e8 is short of squares. Which check does it?")
        )
        playToScholar(coach)
        val early = mateAnalysis(6, provisional = true)
        coach.onEvent(early, 7.0)
        assertEquals(emptyList<CoachMessage>(), messages(coach))
        coach.onEvent(mateAnalysis(6), 8.0)
        assertEquals(emptyList<CoachMessage>(), messages(coach))
        assertTrue(provider.prompts.isEmpty())
        assertEquals(listOf(Trigger.STUCK), stuck(coach).map { it.trigger })
    }

    private fun puzzle(coach: Coach, fen: String = QD1): Board {
        coach.onEvent(
            GameStarted(gameId = "g", userSide = Pieces.Side.BLACK, startFen = fen, kind = "position"),
            0.0,
        )
        return Board(fen)
    }

    private fun analysed(coach: Coach, board: Board, ply: Int, line: List<String>, ts: Double, mate: Int?) {
        coach.onEvent(
            Analysis(
                gameId = "g",
                fen = board.fen(),
                ply = ply,
                lines = listOf(
                    Line(moves = line, depth = 18, expectedScore = 1.0, mate = mate),
                    Line(moves = listOf("h6h5"), depth = 18, expectedScore = 0.1),
                ),
                decisive = true,
                onlyMove = true,
            ),
            ts,
        )
    }

    private fun played(coach: Coach, board: Board, ply: Int, uci: String, byUser: Boolean, ts: Double) {
        val move = Move.fromUci(uci)
        val before = board.fen()
        val san = board.san(move)
        board.push(move)
        coach.onEvent(
            MovePlayed(
                gameId = "g",
                ply = ply,
                san = san,
                uci = uci,
                fenBefore = before,
                fenAfter = board.fen(),
                byUser = byUser,
            ),
            ts,
        )
    }

    /**
     * The puzzle appears, the player presses Hint before the engine has looked, and the hint
     * comes when it has: no puzzle question was asked, as on Tai's desktop.
     */
    private fun hintThen(coach: Coach, board: Board) {
        coach.ask("hint", 0.5)
        analysed(coach, board, 0, QD1_LINE, 1.0, 3)
        val got = messages(coach)
        assertTrue(
            got.map { it.trigger }.toString(),
            got.isNotEmpty() && got.all { it.trigger == Trigger.USER_ASK },
        )
    }

    /**
     * After 1...Qd1+ 2.Rxd1 Rxd1+ 3.Qe1 the coach asked "You can mate in one move ... Which
     * check leaves their king no square at all?" -- a fresh question in the middle of the
     * combination the player was carrying out (Tai's desktop, 2026-09-24): nothing unasked comes
     * while the line is played.
     */
    @Test
    fun noNewQuestionWhileThePlayerPlaysOutTheLineTheyFound() {
        val (coach, _) = coach(emptyList())
        val board = puzzle(coach)
        hintThen(coach, board)
        for ((i, uci) in QD1_LINE.take(4).withIndex()) {
            val ply = i + 1
            played(coach, board, ply, uci, ply % 2 == 1, 2.0 + ply)
            if (ply % 2 == 0) {
                analysed(coach, board, ply, QD1_LINE.drop(ply), 2.5 + ply, (5 - ply) / 2)
            }
        }
        val unasked = messages(coach).filter { it.trigger != Trigger.POST_MOVE }.map { it.text }
        assertEquals(unasked.toString(), emptyList<String>(), unasked)
    }

    /**
     * A reply off the engine's line brought the idea back: the stuck hint's "a tactic that wins a
     * knight ... a pin" at a puzzle's start came again after the move was found (Tai's phone,
     * 2026-09-29: "Each level can only be said once and if the level higher was said then the
     * lower one can't be said"). The next hint unasked is the focus
     * (`test_no_new_question_while_the_line_is_played_out_nor_a_hint_said_twice`).
     */
    @Test
    fun aHintIsNeverSaidTwiceAlongTheCombination() {
        val (coach, _) = coach(emptyList())
        val board = puzzle(coach)
        analysed(coach, board, 0, QD1_LINE, 1.0, 3)
        assertEquals("a puzzle's first position says nothing of its own", emptyList<CoachMessage>(), messages(coach))
        val first = stuck(coach)
        assertEquals(listOf(Trigger.STUCK to DisclosureLevel.IDEA), first.map { it.trigger to it.level })
        played(coach, board, 1, "c2d1", true, 3.0)
        played(coach, board, 2, "a5e1", false, 4.0) // blocks instead of taking: off the line
        analysed(coach, board, 2, listOf("d1e1"), 4.5, 1)
        assertEquals(emptyList<CoachMessage>(), messages(coach).filter { it.trigger != Trigger.POST_MOVE })
        coach.tick(4.0 + coach.policy.stuckThreshold() + 1)
        val focus = messages(coach).filter { it.trigger == Trigger.STUCK }
        assertEquals(focus.toString(), listOf(DisclosureLevel.FOCUS), focus.map { it.level })
    }

    /**
     * The counts of what was said unasked were kept by ply, and every puzzle starts at ply 0:
     * after one puzzle's idea and focus there, the next puzzle's log said "gave the idea and the
     * focus here" before a word of it (Tai's phone, 2026-09-29). Each comes after its own stuck
     * spell, and nothing before the first (Tai, 2026-10-01).
     */
    @Test
    fun aNewPuzzleGetsItsHintsWhateverTheLastOneHad() {
        val (coach, _) = coach(emptyList())

        fun bothHints(): List<DisclosureLevel> {
            assertEquals("nothing before the stuck limit", emptyList<CoachMessage>(), messages(coach))
            val levels = stuck(coach).map { it.level }.toMutableList()
            levels += stuck(coach).map { it.level } // a full stuck spell from the idea
            return levels
        }

        val board = puzzle(coach)
        analysed(coach, board, 0, QD1_LINE, 1.0, 3)
        assertEquals(listOf(DisclosureLevel.IDEA, DisclosureLevel.FOCUS), bothHints())
        val next = "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1"
        coach.onEvent(Resync(gameId = "g", fen = next, reason = "adopted"), 100.0)
        analysed(coach, Board(next), 0, listOf("g8f8"), 101.0, null)
        assertEquals(listOf(DisclosureLevel.IDEA, DisclosureLevel.FOCUS), bothHints())
    }

    /** White, the user, to move with Qxf7# on the board, after black's Nf6 (`_play_to_scholar`). */
    private fun playToScholar(coach: Coach) {
        start(coach)
        val board = Board()
        for ((i, san) in listOf("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6").withIndex()) {
            val ply = i + 1
            val move = board.parseSan(san)!!
            val before = board.fen()
            board.push(move)
            coach.onEvent(
                MovePlayed(
                    gameId = "g",
                    ply = ply,
                    san = san,
                    uci = move.uci(),
                    fenBefore = before,
                    fenAfter = board.fen(),
                    byUser = ply % 2 == 1,
                ),
                ply.toDouble(),
            )
        }
        assertTrue(coach.userToMove && coach.board.fen() == SCHOLAR)
    }

    /**
     * `_mate_analysis`: Qxf7# on the scholar's board; `provisional`, as the desktop's
     * `model_copy(update={"provisional": True})` makes it.
     */
    private fun mateAnalysis(ply: Int, decisive: Boolean = true, provisional: Boolean = false) = Analysis(
        gameId = "g",
        fen = SCHOLAR,
        ply = ply,
        lines = listOf(
            Line(moves = listOf("h5f7"), depth = 12, expectedScore = 1.0, mate = 1),
            Line(moves = listOf("c4f7"), depth = 12, expectedScore = 0.55),
        ),
        decisive = decisive,
        onlyMove = true,
        tags = listOf(MotifTag(name = "mateIn1", squares = listOf("f7", "e8"), move = "h5f7")),
        provisional = provisional,
    )

    /** The directive of the last model call: the harness after its "# Task" line. */
    private fun directive(provider: ScriptedProvider): String {
        val messages = synchronized(provider.prompts) { provider.prompts.last().second }
        assertEquals(1, messages.size)
        return messages[0].content.substringAfterLast("# Task\n")
    }

    /** `_given`: the messages the model was given on the [index]-th call (the last by default). */
    private fun given(provider: ScriptedProvider, index: Int = -1): List<ChatMessage> =
        synchronized(provider.prompts) {
            provider.prompts[if (index < 0) provider.prompts.size + index else index].second
        }

    /** `_harness`: the one user message, the harness with the task at its end. */
    private fun harness(messages: List<ChatMessage>): String {
        assertTrue(messages.size == 1 && messages[0].role == "user")
        return messages[0].content
    }

    /** `_directive` of a call's messages. */
    private fun directiveOf(messages: List<ChatMessage>): String =
        harness(messages).substringAfterLast("# Task\n")

    /** `_current_move_message`: the 'Position now' section, the position the player faces. */
    private fun currentMoveMessage(messages: List<ChatMessage>): String {
        val text = harness(messages)
        assertTrue("no position in the harness", "# Position now" in text)
        return text.substringAfter("# Position now").substringBefore("# Task")
    }

    /**
     * The first hint whole is on the sheet, the least the model's says, and the idea level names
     * the key fact on the board, as the first hint does (parity pass, 2026-09-25; desktop
     * 717cc9c, now `test_stuck_at_a_decisive_moment_the_player_gets_the_idea_and_no_move`). A
     * decisive moment says nothing of its own (Tai, 2026-10-01: nothing "until the user is stuck
     * or actively request a hint or after they have made a move"); stuck, the player gets the
     * idea.
     */
    @Test
    fun stuckAtADecisiveMomentTheSheetCarriesTheFirstHintWhole() {
        val (coach, provider) = coach(
            listOf("You can mate right now: their king on e8 is short of squares. Which check does it?")
        )
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        assertEquals("nothing before the stuck limit", emptyList<CoachMessage>(), messages(coach))
        assertTrue("nothing before the stuck limit", provider.prompts.isEmpty())
        val out = stuck(coach)
        assertTrue(out.size == 1 && out[0].trigger == Trigger.STUCK)
        assertEquals(DisclosureLevel.IDEA, out[0].level)
        val messages = given(provider, 0)
        val current = currentMoveMessage(messages)
        // Everything, the answer included (Tai, 2026-09-26: the model sees engine lines at hints;
        // the reply is checked; desktop 9e2e8e2).
        assertTrue("the model sees every move", "Legal moves" in current && "Qxf7#" in current)
        assertTrue(current, "line 1's first move is the answer to the position they face" in current)
        assertTrue(current, "On the board (complete lists)" in current)
        // The rung's facts -- the goal and the pattern -- stand apart from those past it.
        val facts = current.substringAfter("## Facts (the player to move)").substringBefore("## Beyond this level")
        assertTrue(facts, "goal: the player can mate in one move" in facts)
        assertTrue(facts, "pattern: a mating attack" in facts)
        for (secret in listOf("Qxf7", "queen", "h5f7", "candidates")) assertTrue(secret, secret !in facts)
        val beyond = current.substringAfter("## Beyond this level").substringBefore("## On the board")
        assertTrue(beyond, "piece: the player's queen on h5" in beyond && "Qxf7" in beyond)
        // What the level keeps back is said once, in the Task, with why the model sees it all.
        assertTrue(current, "## Disclosure" !in current)
        val directive = directiveOf(messages)
        assertTrue(directive, "You see the answer so that the hint can be precise" in directive)
        // The first hint's facts are on the sheet to teach from (parity, 2026-09-25; the prompt
        // audit, 2026-09-28), and the idea level names the key fact on the board.
        assertTrue(current, "what a first hint rests on" in current)
        assertTrue(current, "Their king on e8 has one free square" in current)
        assertTrue(directive, "the idea: what is at stake and the key fact" in directive)
        assertTrue(directive, "no square" !in directive)
        assertTrue("named the key square: it stands", out[0].text.startsWith("You can mate right now"))
    }

    /**
     * A first hint that names nothing of the key fact is thinner than the free coach's; it was sent
     * back and the template spoken (parity pass, 2026-09-25). Its form is the prompt's to ask for,
     * and a slip of it is minor (Tai, 2026-09-28: "we don't need to throw anything out"): it is
     * said, and noted for the log (desktop 3ff672b).
     */
    @Test
    fun aReplyThinnerThanTheTemplateIsSaidAndNoted() {
        val thin = "Your king is safe for now, is it not?"
        val (coach, _) = coach(listOf(thin))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        val out = stuck(coach)
        assertEquals(thin, out[0].text)
        val spoken = coach.spoken.last()
        assertTrue(spoken.attempts == 1 && coach.speaker.fallbacks == 0 && spoken.problems.isEmpty())
        assertTrue(spoken.notes.toString(), spoken.notes.any { it.startsWith("left out") })
    }

    /**
     * The lesson is taught from the position they faced, which the record keeps as one line: its
     * sheet goes with the request, and the template's lesson as the floor, whose better move the
     * reply must name (parity pass, 2026-09-25; desktop 717cc9c,
     * `test_feedback_after_the_users_move_may_say_everything_about_that_position`).
     */
    @Test
    fun theLessonAfterAMoveCarriesThePositionTheyFaced() {
        val (coach, provider) = coach(
            listOf(
                "Their king on e8 is nearly out of squares.",
                // Never opening with the move just played (desktop f25217c).
                "Qxf7 was mate on the spot, sharper than the bishop. Do you see why?",
            )
        )
        bishopCheckInsteadOfMate(coach, stuckFirst = true)
        val messages = given(provider)
        val content = messages.last().content
        val faced = content.substringAfter("# The position they faced, before Bxf7+").substringBefore("# Task")
        assertTrue(content, "# The position they faced, before Bxf7+" in content)
        assertTrue(faced, "goal: the player can mate in one move" in faced)
        // The better move the reply must name is the directive's -- not the template's lesson as
        // the least the reply says, which came back word for word (the prompt audit, 2026-09-28).
        assertTrue("The better move was Qxf7#" in directiveOf(messages))
        assertEquals(emptyList<String>(), coach.spoken.last().problems)
        // The first hint is for the side to move: never on the opponent's sheet, where labelled
        // "to the player" it was the opponent's (parity pass 2; desktop 2219e11); on the faced
        // sheet, a position already moved from, it is what a hint rested on there, as it stood.
        val now = currentMoveMessage(messages).substringBefore("# The position they faced")
        assertTrue(now, "(to the player, as it stands)" !in now.substringAfter("## Facts"))
        assertTrue(faced, "(to the player, as it stood)" in faced && "(to the player, as it stands)" !in faced)
    }

    /**
     * "What should I be aiming for?" earns the hint's rung and the hint's directive, not the
     * generic answer (parity pass 2, 2026-09-25; desktop 2219e11); asked of the AI coach, the
     * model answers it, with nothing said before it (Tai, 2026-09-28; desktop 3ff672b).
     */
    @Test
    fun aQuestionAboutThePlanIsAnsweredWithTheFirstHint() {
        val reply = "Their king has one square left. Which check takes it away?"
        val (coach, provider) = coach(listOf(reply))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        assertEquals(emptyList<CoachMessage>(), messages(coach))
        coach.ask("what should I be aiming for here?", 8.0)
        val out = messages(coach)
        assertTrue(out.map { it.text }.toString(), out.map { it.text } == listOf(reply) && out[0].level == DisclosureLevel.IDEA)
        val directive = directiveOf(given(provider, 0))
        // No keyword picked a directive: the guidelines, and the ladder with nothing said yet, so
        // a hint now is the first rung.
        assertTrue(directive, "A hint: one rung up the ladder" in directive)
        assertTrue(directive, "nothing yet" in directive && "the idea: what is at stake" in directive)
        assertTrue(directive, "They asked what to aim for" !in directive)
    }

    /**
     * Tai, 2026-09-29: "Each level can only be said once and if the level higher was said then
     * the lower one can't be said." Once a question had the move, no stuck hint follows: the
     * focus is below it (on 2026-09-28 one did, and both replies went for "d5: your bishop on c8
     * is still shut in", twice). A stuck template stays at the focus all the same
     * (`test_no_hint_unasked_comes_after_the_move_was_given`).
     */
    @Test
    fun noHintUnaskedComesAfterTheMoveWasGiven() {
        val (coach, provider) = coach(listOf("Qxf7# is mate at once."))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        coach.ask("what should I play?", 8.0)
        assertEquals(listOf("Qxf7# is mate at once."), messages(coach).map { it.text })
        for (ts in listOf(17.0, 40.0, 90.0)) coach.tick(ts)
        assertEquals(emptyList<CoachMessage>(), messages(coach))
        assertEquals(1, provider.prompts.size)
        val position = coach.positions.getValue(6) // the position on the board (`coach._current`)
        assertNotNull(position.facts)
        val template = coach.instruction(
            Trigger.STUCK, position, DisclosureLevel.MOVE, null, emptyList(), null, null,
        ).second
        assertTrue("the template is the focus's, not the move's: $template", "Qxf7" !in template)
    }

    /**
     * The runtime audit: with the voice on, an eight-sentence homily went out whole. The stream
     * stops at the task's length and its slack (desktop af1bae2, 3ff672b).
     */
    @Test
    fun withTheVoiceOnNothingGoesOutPastTheLength() {
        val heard = ArrayList<String>()
        val reader = FirstCandidate({ heard.add(it); true }, { emptyList() }, limit = 1)
        reader.feed("One. Two. Three. Four. Five. ")
        reader.finish()
        assertEquals("the limit and its slack, no more", 3, heard.sumOf { t -> t.count { it == '.' } })
    }

    /**
     * A late verdict's lesson that gives away the answer in the position faced now is noted
     * (af1bae2) -- logged, never a reason to refuse the reply (Tai, 2026-09-26; desktop 9e2e8e2).
     */
    @Test
    fun aLateLessonDoesNotGiveAwayTheMoveTheyFaceNow() {
        val request = afterMoveRequest(SCHOLAR, "Bxf7+", "h5f7", nowAnswer = listOf("Qxf7#"))
        val speaker = Speaker(ScriptedProvider(emptyList()))
        val said = "And now Qxf7# is mate on the spot."
        assertTrue("gives away Qxf7# in the position they face now" in speaker.leaks(said, request))
        assertTrue(speaker.problemsOf(said, request).none { it.startsWith("gives away") })
    }

    /**
     * A reply that gives the answer away is logged, never refused (Tai, 2026-09-26: the leak
     * check is "a debugging tool and not something used to reject response"; desktop 9e2e8e2).
     */
    @Test
    fun aLeakedDraftIsSpokenAndTheLeakLogged() {
        val (coach, provider) = coach(
            listOf("Their king on e8 is boxed in: your queen can end it on f7. Which check is it?")
        )
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        val out = stuck(coach)
        assertTrue(out.size == 1 && out[0].text.startsWith("Their king on e8 is boxed in"))
        val spoken = coach.spoken.last()
        assertTrue(spoken.attempts == 1 && spoken.problems.isEmpty() && coach.speaker.fallbacks == 0)
        assertTrue(spoken.leaks.toString(), "withheld square f7" in spoken.leaks && "withheld piece queen" in spoken.leaks)
        assertEquals("no second call for a leak", 1, provider.prompts.size)
    }

    /**
     * Every lesson names the better move, and skipping any reply that did let "That was a
     * blunder: a3 was better" through for a good move (the runtime audit; desktop af1bae2).
     */
    @Test
    fun theVerdictWordsAreCheckedClauseByClause() {
        val ruy = "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
        val good = afterMoveRequest(
            ruy, "O-O", "b5a4", severity = Severity.NONE, label = Label.GOOD, expectedLoss = 0.04,
        )
        assertEquals(
            listOf("called O-O a slip; its verdict is good"),
            verdictProblems("That was a blunder: Ba4 was better.", good),
        )
        assertTrue(verdictProblems("O-O was the best move here, though Ba4 was close.", good).isNotEmpty())
        assertEquals(emptyList<String>(), verdictProblems("Fine, but Ba4 was the best move here.", good))
        // A clause opening with "it" or "that" is about the move only where the verdict word is
        // said of it: these true sentences were refused (the deep prompt review, 2026-10-01).
        for (said in listOf(
            "It avoids the mistake of locking your bishop in.",
            "That keeps you clear of the classic blunder of moving the f-pawn.",
            "It was not a blunder, but Ba4 was a touch better.",
            "It is best to develop a piece before you push pawns.",
            "This is a position where the best move is quiet.",
        )) {
            assertEquals(said, emptyList<String>(), verdictProblems(said, good))
        }
        assertTrue(verdictProblems("It was a clear mistake; Ba4 was better.", good).isNotEmpty())
        assertTrue(verdictProblems("That is the best move here.", good).isNotEmpty())
    }

    /**
     * The verdict comes first, from the best line alone; the analysis of the position the move
     * left about a second later, and with it what the move allowed. The app holds the lesson for
     * it (at most [LESSON_WAIT_S]); the game moving on releases it at once (desktop 98c4d40).
     */
    @Test
    fun aMistakesLessonWaitsForWhatTheMoveAllowed() {
        val coach = Coach(
            ScriptedProvider(emptyList()),
            policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)),
            lessonWaitS = LESSON_WAIT_S,
        )
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        messages(coach)
        val board = Board(SCHOLAR)
        val move = board.parseSan("Bxf7+")!!
        board.push(move)
        val played = MovePlayed(
            gameId = "g",
            ply = 7,
            san = "Bxf7+",
            uci = move.uci(),
            fenBefore = SCHOLAR,
            fenAfter = board.fen(),
            byUser = true,
        )
        val verdict = MoveAssessment(
            gameId = "g",
            ply = 7,
            uci = move.uci(),
            expectedLoss = 0.45,
            severity = Severity.BLUNDER,
            best = listOf("h5f7"),
            label = Label.MISS,
        )
        coach.onEvent(played, 10.0)
        coach.onEvent(verdict, 10.1)
        assertEquals("held for the analysis of the position it left", emptyList<CoachMessage>(), messages(coach))
        coach.tick(10.1 + LESSON_WAIT_S / 2)
        assertEquals("still inside the wait", emptyList<CoachMessage>(), messages(coach))
        coach.tick(10.1 + LESSON_WAIT_S)
        var out = messages(coach)
        assertTrue(out.size == 1 && out[0].trigger == Trigger.POST_MISTAKE)
        // The analysis landing releases it at once.
        coach.onEvent(Takeback(gameId = "g", plies = 1, fen = SCHOLAR), 12.0)
        messages(coach)
        coach.onEvent(played, 13.0)
        coach.onEvent(verdict, 13.1)
        val after = Analysis(
            gameId = "g",
            fen = board.fen(),
            ply = 7,
            lines = listOf(Line(moves = listOf("e8e7"), depth = 12, expectedScore = 0.1)),
            decisive = false,
            onlyMove = true,
        )
        coach.onEvent(after, 13.5)
        out = messages(coach)
        assertEquals(listOf(Trigger.POST_MISTAKE), out.map { it.trigger })
    }

    /**
     * The second audit: "ahead by about 1.2 pawns" was no figure to the old check; "you took on
     * d5" and "that d5 pawn" said the capture back; "which was a inaccuracy" (desktop 8438193).
     */
    @Test
    fun figuresOrdinalsCapturesAndArticles() {
        assertTrue(figureSaid("You are ahead by about 1.2 pawns."))
        assertTrue(figureSaid("It cost you 15% of the game."))
        assertFalse(figureSaid("Your rook on the 7th rank and Nf3 were strong."))
        val ruy = "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
        val trade = afterMoveRequest(ruy, "Bxc6", "e1g1")
        assertTrue(restated("You took on c6, giving up the bishop pair.", trade).isNotEmpty())
        assertTrue(restated("Winning that c6 knight gives up the bishop pair.", trade).isNotEmpty())
        assertEquals(
            listOf("an inaccuracy", "a miss", "best", "a book move"),
            listOf("inaccuracy", "miss", "best", "book").map { rated(it) },
        )
    }

    /**
     * The parity pilot: "**Rc1** was better", and a closing question wrapped in ** failed the
     * question check twice, so the template spoke (desktop de68772).
     */
    @Test
    fun theModelsMarkdownNeverReachesTheVoiceOrTheChecks() {
        assertEquals("Their threat is ...f5, on e4. Why?", plain("Their threat is **...f5**, on `e4`. **Why?**"))
        assertEquals("Plan\npush the pawn\ntrade", plain("# Plan\n- push the pawn\n* trade"))
        assertEquals(
            listOf(1.0 to "Rc1 was better. Do you see why?"),
            parseCandidates("**Rc1** was better. **Do you see why?**"),
        )
    }

    /**
     * Round 2: "Stockfish gives White about 0.8 pawns" in seven answers (desktop 47138dd; the
     * review's costliest losses are held by the conversation fixture). Noted, never refused: a
     * reply's form is the prompt's to ask for (Tai, 2026-09-28; desktop 3ff672b).
     */
    @Test
    fun theCoachNeverNamesTheEngine() {
        val request = afterMoveRequest(SCHOLAR, "Bxf7+", "h5f7")
        val speaker = Speaker(ScriptedProvider(emptyList()))
        val said = "Qxf7 was better; Stockfish liked it more."
        val notes = speaker.notes(said, request)
        assertTrue(notes.toString(), "named the engine: say it as the coach who saw the game" in notes)
        assertEquals(emptyList<String>(), speaker.problemsOf(said, request))
        val asked = afterMoveRequest(SCHOLAR, "Bxf7+", "h5f7", userText = "what does the engine say?")
        assertTrue(
            "named the engine: say it as the coach who saw the game" !in
                speaker.notes("Qxf7 was better; the engine liked it more.", asked)
        )
    }

    /**
     * Hint, Best move and Line. Asked of the AI coach, the model answers, with its marks and
     * nothing said before it: its first words come within a second or two (Tai, 2026-09-28: "it's
     * already almost instant"). Offline the template for the level goes out at once, the move and
     * the line asked for bare are the template's alone, and no model is asked (desktop 3ff672b).
     */
    @Test
    fun theButtonsAreTheModelsToAnswerAndTheTemplatesOffline() {
        val (coach, provider) = coach(listOf("Qxf7# ends it: the queen takes with check and mate."))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        coach.ask("what should I play", 8.0)
        assertEquals("nothing before the model's answer", emptyList<CoachMessage>(), coach.poll())
        val out = messages(coach)
        assertEquals(listOf("Qxf7# ends it: the queen takes with check and mate."), out.map { it.text })
        assertTrue(out[0].trigger == Trigger.USER_ASK && out[0].level == DisclosureLevel.MOVE)
        assertTrue(out[0].annotations.any { it.kind == "arrow" })
        val directive = directiveOf(given(provider, 0))
        // What they ask is the model's to read: the guidelines, whatever the words.
        assertTrue(directive, "The move -- " in directive && "give it plainly, and why it works" in directive)
        val offline = Coach(OfflineProvider(), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
        playToScholar(offline)
        offline.onEvent(mateAnalysis(6, decisive = false), 7.0)
        offline.ask("what should I play", 8.0)
        val quick = offline.poll() // at once
        assertEquals(listOf("Qxf7# is mate."), quick.map { it.text })
        assertEquals(DisclosureLevel.MOVE, quick[0].level)
        assertTrue(quick[0].annotations.any { it.kind == "arrow" })
        assertEquals(emptyList<CoachMessage>(), messages(offline))
    }

    /**
     * The material, the opening by name, who stands better. Offline the template says them
     * exactly, the fact itself, never the hint for the position ("How many pieces am I up?" was
     * once answered "Something forcing is on..."). Asked of the AI coach, the model answers (Tai,
     * 2026-09-28); and a question that only mentions the opening is not a question about its name:
     * "what should I play against this opening? give me a line" got "The opening was the A40
     * Queen's Pawn Game.", and is the line's to answer (desktop 4365dfa, 3ff672b).
     */
    @Test
    fun aQuestionOnlyAboutAFactIsTheModelsAndOfflineTheTemplates() {
        for (said in listOf("how is my opening?", "why am I losing?", "am I winning? what should I do?")) {
            assertFalse(said, factQuestion(said))
        }
        val (coach, provider) = coach(listOf("The model's words.", "Qxf7# -- it is mate at once."))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        coach.ask("what opening is this?", 8.0)
        assertEquals(listOf("The model's words."), messages(coach).map { it.text })
        // No keyword picks the question's directive: the same guidelines carry both questions, the
        // opening by its name and the line, and the model reads which is asked.
        assertTrue("the opening by the name the record gives it" in directiveOf(given(provider, 0)))
        coach.ask("what should I play against this opening? give me a line", 9.0)
        assertEquals(listOf("Qxf7# -- it is mate at once."), messages(coach).map { it.text })
        val directive = directiveOf(given(provider, 1))
        assertTrue(directive, "The line, or what happens next: the engine's line move by move" in directive)
        assertTrue(directive, "They asked to see the line" !in directive)
        val offline = Coach(OfflineProvider(), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
        playToScholar(offline)
        offline.onEvent(mateAnalysis(6, decisive = false), 7.0)
        for ((i, pair) in listOf(
            "who's up material?" to listOf("Material:"),
            "how many pieces am I up?" to listOf("Material:"),
            "am I winning?" to listOf("Right now you are winning"),
            "what opening is this?" to listOf("We left the opening book early.", "The opening was the"),
        ).withIndex()) {
            val (said, opens) = pair
            assertTrue(said, factQuestion(said))
            offline.ask(said, (8 + i).toDouble())
            val out = messages(offline)
            assertTrue("$said: $out", out.isNotEmpty() && opens.any { out.last().text.startsWith(it) })
        }
    }

    /**
     * The sheet's threats are the side NOT to move's: on the opponent's turn, the player's own.
     * "What are they threatening?" there is answered with their best move now (desktop 2ec1747).
     */
    @Test
    fun theirThreatOnTheirTurnIsTheirBestMoveNotThePlayers() {
        fun scholarAsBlack(): Pair<Coach, ScriptedProvider> {
            val (coach, provider) = coach(listOf("", ""))
            start(coach, Pieces.Side.BLACK)
            val board = Board()
            for ((i, san) in listOf("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6").withIndex()) {
                val ply = i + 1
                val move = board.parseSan(san)!!
                val before = board.fen()
                board.push(move)
                coach.onEvent(
                    MovePlayed(
                        gameId = "g",
                        ply = ply,
                        san = san,
                        uci = move.uci(),
                        fenBefore = before,
                        fenAfter = board.fen(),
                        byUser = ply % 2 == 0,
                    ),
                    ply.toDouble(),
                )
            }
            assertFalse(coach.userToMove)
            coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
            return coach to provider
        }
        val (coach, provider) = scholarAsBlack()
        coach.ask("what are they threatening?", 8.0)
        val out = messages(coach)
        assertTrue(directiveOf(given(provider, 0)), "it is the opponent's move" in directiveOf(given(provider, 0)))
        assertTrue(out.toString(), out.isNotEmpty() && out.last().text.startsWith("It's their move, and their best is Qxf7#"))
        // A hint asked on their turn is the same, never the player's hint on the opponent's
        // position: the model was asked for "the first hint" and to keep back "the player's move"
        // (the coach review, 2026-10-01).
        val (hinted, hintModel) = scholarAsBlack()
        hinted.ask("hint", 9.0)
        val hint = messages(hinted)
        val task = directiveOf(given(hintModel, 0))
        assertTrue(task, "it is the opponent's move" in task)
        assertTrue(task, "Give the first hint" !in task && "Keep back only the answer" !in task)
        assertTrue(hint.toString(), hint.isNotEmpty() && hint.last().text.startsWith("It's their move, and their best is Qxf7#"))
    }

    /** Their own move is theirs even after the reply, and a threat gets its answer (af1bae2). */
    @Test
    fun theirOwnMoveIsTheirsEvenAfterTheReplyAndAThreatGetsItsAnswer() {
        for (said in listOf("what should I have played?", "did I just blunder?", "was my move bad?")) {
            assertTrue(said, MY_MOVE.containsMatchIn(said))
        }
        assertFalse("'that' is the last move played", MY_MOVE.containsMatchIn("why was that bad?"))
        for (said in listOf("what is their threat?", "what does my opponent threaten?", "am I in danger?")) {
            assertTrue(said, ASK_THREAT.containsMatchIn(said))
        }
        // And as the pilot asked it, answered from the threats, never examined as a named line
        // (desktop 653a473, whose commit names these three).
        for (said in listOf("What is he threatening?", "what are they up to", "whats the threat")) {
            assertTrue(said, ASK_THREAT.containsMatchIn(said) && !mayCarryClaims(said))
        }
        // Asked of the AI coach, no keyword picks the directive: the model reads the question
        // (Tai, 2026-10-02); these patterns are the offline coach's.
    }

    /**
     * `_bishop_check_instead_of_mate`: the user plays Bxf7+ where Qxf7 was mate, and it is judged;
     * [stuckFirst], after the stuck hint, as the desktop's
     * `test_feedback_after_the_users_move_may_say_everything_about_that_position` plays it
     * (desktop, 2026-10-01).
     */
    private fun bishopCheckInsteadOfMate(coach: Coach, stuckFirst: Boolean = false) {
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        if (stuckFirst) stuck(coach) else messages(coach)
        val board = Board(SCHOLAR)
        val move = board.parseSan("Bxf7+")!!
        board.push(move)
        coach.onEvent(
            MovePlayed(
                gameId = "g",
                ply = 7,
                san = "Bxf7+",
                uci = move.uci(),
                fenBefore = SCHOLAR,
                fenAfter = board.fen(),
                byUser = true,
            ),
            10.0,
        )
        coach.onEvent(
            MoveAssessment(
                gameId = "g",
                ply = 7,
                uci = move.uci(),
                expectedLoss = 0.45,
                severity = Severity.BLUNDER,
                best = listOf("h5f7"),
                label = Label.MISS,
            ),
            11.0,
        )
        messages(coach)
        if (!stuckFirst) {
            // The engine has looked at the position the move left (the opponent to move): with the
            // AI on a question waits for that look, as it reads the position now.
            looked(coach, board, 7, listOf("e8f7"), 11.5)
            messages(coach)
        }
    }

    /**
     * "Why was that bad?" is about a position that is past: the directive states the verdict,
     * the move the engine preferred and the idea, and says nothing is withheld -- and what it
     * cost only when they ask it: "It cost you 22% of the game." opened every answer, and Tai
     * had said the figure is for when he asks (2026-09-20; desktop 6d9ba6f). The better move is
     * named as "Qxf7# was better", never "the engine preferred" (desktop 47138dd).
     */
    @Test
    fun aQuestionAboutTheLastMoveGetsItsVerdictWithoutTheFigure() {
        val (coach, provider) = coach(listOf("...", "..."))
        bishopCheckInsteadOfMate(coach)
        synchronized(provider.replies) {
            provider.replies.clear()
            provider.replies.addAll(listOf("", ""))
        }
        coach.ask("why was that bad?", 12.0)
        val out = messages(coach)
        // Taught from the position the move was played from, as the lesson after it was.
        val asked = given(provider).last().content
        assertTrue(asked, "# The position they faced, before Bxf7+" in asked)
        // With the AI on the guidelines say a move already played may be told in full, and the
        // record gives the verdict, the better move and the sheet of the position they faced.
        val directive = directive(provider)
        assertTrue(directive, "A move they already played: its verdict is in the record" in directive)
        assertTrue(directive, "They ask about the move they already played" !in directive)
        assertTrue(directive, "45%" !in directive)
        assertTrue(directive, "no percentages, scores or move numbers unless they ask for the number" in directive)
        assertTrue(asked, "the engine preferred 4. Qxf7#" in asked)
        // The fallback (a reply that failed its checks) is the verdict itself, not a hint -- and
        // without the figure.
        assertTrue(out.last().text, out.last().text.startsWith("Qxf7# was better"))
    }

    /**
     * The verdict and the better move came back in every reply, and "I understand: I should play
     * Nxc3" got the lesson again and "Which White piece is still at home on g1?" (the AI's
     * sessions, 2026-09-26; desktop 9e49622). With a model the words are the model's to read, with
     * the sheet of the position the move was played from beside them (Tai's puzzle, 2026-09-28).
     */
    @Test
    fun aFollowUpGetsWhatItAsksAndANodOneSentence() {
        val (coach, provider) = coach(listOf("...", "...", "...", "..."))
        bishopCheckInsteadOfMate(coach)
        synchronized(coach.lock) {
            coach.conversation.coachSaid("That one cost you: Qxf7# was better -- it is mate.")
        }
        synchronized(provider.replies) {
            provider.replies.clear()
            provider.replies.addAll(listOf("", ""))
        }
        coach.ask("I see, I should have seen the mate.", 12.0)
        messages(coach)
        var directive = directive(provider)
        // What the coach has said since the move is a fact of the conversation, not a keyword:
        // the guideline for a follow-up is there, and the model reads whether the words ask
        // anything.
        assertTrue(directive, "do not say it again in other words" in directive)
        assertTrue(directive, "do not repeat the verdict or the better move you told them" in directive)
        synchronized(provider.replies) {
            provider.replies.clear()
            provider.replies.addAll(listOf("", ""))
        }
        coach.ask("Why does the bishop check not work?", 13.0)
        messages(coach)
        val request = synchronized(provider.prompts) { provider.prompts.last().second[0].content }
        assertTrue(request, "# The position they faced, before Bxf7+" in request)
        directive = directive(provider)
        assertTrue(directive, "do not say it again" in directive)
        // A question is no claim to agree with (Luna: "You are right that g1 is controlled by your
        // King", Tai's puzzle, 2026-09-29).
        assertTrue(directive, "a question states nothing to agree with" in directive)
        assertTrue("a question may be offered on top of an answer", "then, if you like, one question" in directive)
    }

    /**
     * The follow-up's directive says not to repeat the better move, and the check that a reply
     * names it refused every reply that obeyed; after two, the template said the verdict again
     * (the prompt audit, 2026-09-28; the desktop folded its test of it into
     * `test_a_follow_up_gets_what_it_asks_and_a_nod_one_sentence`). The floor, the lesson again,
     * goes with it.
     */
    @Test
    fun aFollowUpAboutTheirMoveNeedNotNameTheBetterMoveAgain() {
        val (coach, provider) = coach(listOf("...", "..."))
        bishopCheckInsteadOfMate(coach)
        synchronized(coach.lock) {
            coach.conversation.coachSaid("That one cost you: Qxf7# was better -- it is mate.")
        }
        val reply = "A check that leads nowhere only hands their king a free move."
        synchronized(provider.replies) {
            provider.replies.clear()
            provider.replies.addAll(listOf(reply, reply))
        }
        coach.ask("Why does the bishop check not work?", 13.0)
        val out = messages(coach)
        assertTrue(out.isNotEmpty())
        assertEquals(reply, out.last().text)
    }

    /**
     * Asked what it cost, the figure is said (desktop 6d9ba6f), in the desktop's words, which reach
     * the answer about the last move through askedOwnMove (desktop f733318).
     */
    @Test
    fun whatAMoveCostIsSaidWhenTheyAskIt() {
        val (coach, provider) = coach(listOf("...", "..."))
        bishopCheckInsteadOfMate(coach)
        synchronized(provider.replies) {
            provider.replies.clear()
            provider.replies.addAll(listOf("", ""))
        }
        coach.ask("how much did that cost me?", 12.0)
        val out = messages(coach)
        // The model reads the question (no keyword directive); the fallback has the figure.
        assertTrue(out.isNotEmpty())
        assertTrue(out.last().text, out.last().text.startsWith("It cost you 45% of the game. Qxf7# was better"))
    }

    @Test
    fun inBlitzAMistakeGetsAWordThatDoesNotCallItFine() {
        // Blitz keeps only the word after the move (the policy), so a blunder comes to the
        // directive as POST_MOVE: no lesson and no question, but not "fine" either.
        val (coach, _) = coach(emptyList())
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 6.5)
        val position = coach.positions.getValue(6)
        assertNotNull(position.facts)
        val board = Board(SCHOLAR)
        val move = board.parseSan("a3")!!
        board.push(move)
        val played = MovePlayed(
            gameId = "g",
            ply = 7,
            san = "a3",
            uci = move.uci(),
            fenBefore = SCHOLAR,
            fenAfter = board.fen(),
            byUser = true,
        )
        val blunder = MoveAssessment(
            gameId = "g",
            ply = 7,
            uci = move.uci(),
            expectedLoss = 0.5,
            severity = Severity.BLUNDER,
            best = listOf("h5f7"),
            label = Label.BLUNDER,
        )
        val (instruction, fallback) = coach.instruction(
            Trigger.POST_MOVE, position, DisclosureLevel.LINE, null, emptyList(), blunder, played,
        )
        // A blunder is said as one, never "that cost you something" (the facts audit, 2026-10-01).
        assertTrue(fallback, fallback in COST_BY_SEVERITY.getValue("blunder").map { it.replace("{best}", "Qxf7#") + "." })
        assertTrue(instruction, "which was a blunder" in instruction && "fine" !in instruction)
        assertTrue(instruction, "no lesson, no question" in instruction)
        // A move a little behind the best is still fine.
        val good = MoveAssessment(
            gameId = "g",
            ply = 7,
            uci = move.uci(),
            expectedLoss = 0.03,
            severity = Severity.NONE,
            best = listOf("h5f7"),
            label = Label.GOOD,
        )
        coach.messages += RULE_SPACING // clear of any rule of thumb said before (desktop 3d9568d)
        val rulesBefore = coach.rulesSaid()
        val (_, fine) = coach.instruction(
            Trigger.POST_MOVE, position, DisclosureLevel.LINE, null, emptyList(), good, played,
        )
        // The better move and what it does -- and no rule of thumb: a fine move is not a lesson
        // (Tai, 2026-09-27: "the coach is a teacher, not a nagging annoying bot"; desktop f983cd0).
        assertTrue(fine, FINE_TIGHTER.any { fine.startsWith(it.replace("{best}", "Qxf7#")) })
        assertTrue(fine, "it is mate" in fine)
        assertTrue("${coach.rulesSaid()} $fine", coach.rulesSaid() == rulesBefore && !ANNOUNCED.containsMatchIn(fine))
        // Said once, the lesson is not said again a move later: the verdict and the better move
        // stand alone (Nova bot games, 2026-09-25: 27% of lessons came again within three moves).
        coach.conversation.coachSaid(fine)
        val (_, again) = coach.instruction(
            Trigger.POST_MOVE, position, DisclosureLevel.LINE, null, emptyList(), good, played,
        )
        assertTrue(again, FINE_TIGHTER.any { again.startsWith(it.replace("{best}", "Qxf7#")) })
        assertTrue(again, "it is mate" !in again && coach.rulesSaid() == rulesBefore)
        assertTrue(instruction, "tells them nothing: do not open with it" in instruction)
    }

    /**
     * `test_a_rule_of_thumb_cools_down_across_games_and_puzzles` (desktop 3d9568d). Tai,
     * 2026-09-27, after a run of puzzles: "it keeps on nagging me about some lessons like 'A rule
     * to take with you' ... there has to be a cooldown of some number of messages before it can
     * appear again". A rule is not said again for [RULE_COOLDOWN] of the coach's messages,
     * whatever game or puzzle they fall in, and no two rules come within [RULE_SPACING] messages;
     * the model is told to leave it out while it cools.
     */
    @Test
    fun aRuleOfThumbCoolsDownAcrossGamesAndPuzzles() {
        val (coach, _) = coach(emptyList())
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 6.5)
        stuck(coach)
        val position = coach.positions.getValue(6)
        val board = Board(SCHOLAR)
        val move = board.parseSan("a3")!!
        val played = MovePlayed(
            gameId = "g",
            ply = 7,
            san = "a3",
            uci = move.uci(),
            fenBefore = SCHOLAR,
            fenAfter = SCHOLAR,
            byUser = true,
        )
        val mistake = MoveAssessment(
            gameId = "g",
            ply = 7,
            uci = move.uci(),
            expectedLoss = 0.5,
            severity = Severity.MISTAKE,
            best = listOf("h5f7"),
            label = Label.MISTAKE,
        )

        fun lesson(): Pair<String, String> {
            coach.conversation.entries.clear() // a new puzzle: nothing said in this game yet
            return coach.instruction(
                Trigger.POST_MISTAKE, position, DisclosureLevel.LINE, null, emptyList(), mistake, played,
            )
        }

        // The rules said from here on, in the order said (the desktop spies on `_rule_said`).
        val baseline = coach.rulesSaid()
        fun saidRules(): List<String> =
            coach.rulesSaid().entries.filter { (rule, at) -> baseline[rule] != at }.sortedBy { it.value }.map { it.key }

        fun hasRule(text: String) = saidRules().any { it.lowercase().trimEnd('.') in text.lowercase() }

        // The stuck hint's idea said its principle: no rule for a few messages.
        var (instruction, soon) = lesson()
        assertTrue(soon, !hasRule(soon) && RULE_HELD in instruction)
        coach.messages += RULE_SPACING
        val (firstInstruction, first) = lesson()
        assertTrue(first, hasRule(first) && RULE_HELD !in firstInstruction)
        // Said as a sentence of its own, never announced (Tai, 2026-09-27: "nagging stuff like
        // 'Remember this' 'Keep this one'"; desktop f983cd0).
        assertTrue(first, sentenceOf(saidRules().last()) in first && !ANNOUNCED.containsMatchIn(first))
        assertTrue(firstInstruction, "pattern to remember" in firstInstruction)
        repeat(RULE_COOLDOWN - 1) {
            coach.messages += 1
            val (held, again) = lesson()
            instruction = held
            assertTrue(again, !hasRule(again))
            // Held, the directive asks for no rule at all, not for one and then for none: the
            // local model gave one anyway in 2 of 17 lessons (the free check, desktop 5b3d4bf).
            assertTrue(instruction, RULE_HELD in instruction)
            assertTrue(instruction, "pattern to remember" !in instruction && "pattern to keep" !in instruction)
        }
        coach.messages += 1
        val (_, back) = lesson()
        assertTrue("back after the cooldown: $back", hasRule(back))
        // Any two rules keep their distance, and a sound move's principle is a rule too.
        assertFalse(coach.ruleOk("a different rule"))
        assertTrue("takeFile" in coach.verdictTaught())
        coach.messages += RULE_SPACING
        assertTrue(coach.ruleOk("a different rule"))
        coach.taughtByVerdict("takeFile")
        assertTrue("takeFile" in coach.verdictTaught() && !coach.ruleOk(null))
        coach.messages += RULE_SPACING
        assertTrue("takeFile" in coach.verdictTaught() && "castle" !in coach.verdictTaught())
        coach.messages += RULE_COOLDOWN
        assertTrue(coach.ruleOk("takeFile")) // cooled; the last few lessons' memory is its own
    }

    /**
     * `test_a_hints_principle_cools_down_too` (desktop 3d9568d): the first hint's principle ("A
     * move that attacks something they must then defend keeps you a step ahead: ...") came 218
     * times in the master games' hints, 59 of them again in the same game: it is a rule of thumb
     * like the lesson's, left out while it cools.
     */
    @Test
    fun aHintsPrincipleCoolsDownToo() {
        val (coach, _) = coach(emptyList())
        val rule = "A move that attacks something they must then defend keeps you a step ahead."
        val hint = "You can take the initiative. $rule Their pawn on d3 can be hit."
        val facts = PositionFacts(
            fen = SCHOLAR, sideToMove = "white", phase = "middlegame", situation = hint, principle = rule,
        )
        var (said, note) = coach.coolHint(facts, hint)
        assertTrue(said == hint && note.isEmpty())
        coach.messages += RULE_COOLDOWN - 1
        val cooled = coach.coolHint(facts, hint)
        said = cooled.first
        note = cooled.second
        assertEquals("You can take the initiative. Their pawn on d3 can be hit.", said)
        assertTrue(note, rule in note && "heard it lately" in note)
        coach.messages += 1
        assertEquals(hint to "", coach.coolHint(facts, hint))
    }

    @Test
    fun aLessonIsSaidOfTheBoardThePlayerSeesAndNotTwice() {
        // 4.d3 had just let the bishop out: "still shut in behind the pawn on d2" was the
        // position before the move (Nova bot games, 2026-09-25).
        assertTrue(stale("your bishop on c1 is still shut in behind the pawn on d2", "d2d3"))
        assertFalse(stale("your bishop on c1 is still shut in behind the pawn on d2", "b1c3"))
        // A line is about the moves to come, and stands.
        assertFalse(stale("after Qg2, Rdg1, and the queen on g2 is trapped", "g2g3"))
        val said = "That works, though d4 was a touch better -- your bishop on c1 is still shut in behind " +
            "the pawn on d2."
        val rule = "The centre is where pieces have the most room."
        val text = "Playable -- d4 was a little more precise -- your bishop on c1 is still shut in behind " +
            "the pawn on d2. $rule"
        // The reason said lately goes; a rule of thumb is not `fresh`'s to hold -- one said lately
        // is never written in ([Coach.ruleOk]), so a rule that is here stays (desktop f983cd0).
        assertEquals("Playable -- d4 was a little more precise. $rule", fresh(text, said))
        assertEquals(text, fresh(text, "Nice."))
        var opener = "Playable -- d4 was a little more precise"
        assertEquals("$opener. $rule", fresh(text, said, opener))
        // An opener with its own " -- " and no reason after it: the better move is not a reason,
        // and stays when the same move was just said ("Not that one." -- Nova bot games).
        opener = "Not that one -- Ng3 was better"
        val just = "That one cost you: Ng3 was better. After c6, they win a pawn."
        assertEquals("$opener.", fresh("$opener.", just.lowercase(), opener))
        assertEquals("$opener.", fresh("$opener -- it wins a pawn.", "$just It wins a pawn.", opener))
    }

    /**
     * Desktop f363ea0 (which adds no test of its own): asked "why was that bad?", an answer that
     * would name nothing but the better move gives the engine's line from the position the move
     * was played from -- "g6 was better." alone answered nothing, 13 of 95 answers in the
     * desktop's offline sweep of bot games (2026-09-27). An answer with a reason, or a line that
     * does not start with the better move, stands as it is.
     */
    @Test
    fun anAnswerThatNamesOnlyTheBetterMoveGivesTheLine() {
        val facts = PositionFacts(
            fen = SCHOLAR, sideToMove = "white", phase = "middlegame",
            line = listOf("g6", "O-O-O", "Bg7", "Qf3", "Ne8", "Nf6"),
        )
        val opener = "g6 was better"
        assertEquals("g6 was better -- the line runs g6 O-O-O Bg7 Qf3 Ne8.", withLine("g6 was better.", opener, "g6", facts))
        val reasoned = "g6 was better -- it keeps the long diagonal."
        assertEquals(reasoned, withLine(reasoned, opener, "g6", facts))
        assertEquals("h6 was better.", withLine("h6 was better.", "h6 was better", "h6", facts))
        assertEquals("g6 was better.", withLine("g6 was better.", opener, "g6", null))
    }

    /**
     * Four decisive moments in one game got the same sentence four times (a master-game pass,
     * 2026-09-21): each list is a bag, every wording once before any comes again (Tai,
     * 2026-09-25), and never the last one said. The desktop's rotation test went in bb9756b:
     * `test_every_wording_once_before_any_comes_again_and_never_twice_running`
     * (tests/test_bag.py) pins the bags, and the praise's bag is
     * `test_in_blitz_a_good_move_earns_two_words_without_the_model` (tests/test_coach.py).
     */
    @Test
    fun theTemplatesRotateAndNeverRepeatTheLast() {
        val coach = Coach(OfflineProvider(), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
        for ((key, options) in listOf("cost" to COST_YOU, "fine" to FINE_TIGHTER)) {
            val said = List(options.size * 8) { coach.bags.draw(key, options) }
            assertTrue("$key: $said", said.zipWithNext().all { (a, b) -> a != b })
            for (start in said.indices step options.size) {
                assertEquals(key, options.sorted(), said.subList(start, start + options.size).sorted())
            }
        }
        // The praise on a fine move too, the same bag whether a lesson follows or not.
        val said = List(10) { coach.verdictWord("best") }
        assertEquals(OPENERS.getValue("best").map { "$it." }.sorted(), said.take(5).sorted())
        // A list of one is said again, rather than nothing.
        assertEquals("only", coach.bags.draw("one", listOf("only")))
        assertEquals("only", coach.bags.draw("one", listOf("only")))
    }

    /**
     * `test_offline_coach.py`: the first hint is written into every position's facts and said in
     * few: its lead ("There is a mate in one here") is peeked from its bag as it is written and
     * taken only when it is said, so the hints never asked for do not use up the bag.
     */
    @Test
    fun theFirstHintsLeadIsDrawnWhenTheHintIsSaid() {
        val coach = Coach(OfflineProvider(), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
        playToScholar(coach)
        for (ts in listOf(7.0, 7.5)) coach.onEvent(mateAnalysis(6, decisive = false), ts)  // twice; nothing asked
        messages(coach)
        val position = coach.positions.getValue(6)
        val lead = position.leads.getValue("lead:mate-one")
        assertEquals(IdeaWords.MATE_ONE_LEADS.sorted(), coach.bags.left("lead:mate-one").sorted())
        coach.ask("hint", 8.0)
        val out = messages(coach)
        assertEquals(out.map { it.text }.toString(), 1, out.size)
        assertTrue("$lead / ${out[0].text}", out[0].text.lowercase().startsWith(Situation.spoken(lead).lowercase()))
        val left = coach.bags.left("lead:mate-one")
        assertTrue(lead !in left && left.size == IdeaWords.MATE_ONE_LEADS.size - 1)
        assertEquals(emptyMap<String, String>(), position.leads)
    }

    /** `italian`: 1.e4 e5 2.Nf3 Nc6, White (the player) to move, the position analysed. */
    private fun italian(coach: Coach) {
        start(coach)
        val board = Board()
        for ((i, san) in listOf("e4", "e5", "Nf3", "Nc6").withIndex()) {
            val ply = i + 1
            val move = board.parseSan(san)!!
            val before = board.fen()
            board.push(move)
            coach.onEvent(
                MovePlayed(
                    gameId = "g",
                    ply = ply,
                    san = san,
                    uci = move.uci(),
                    fenBefore = before,
                    fenAfter = board.fen(),
                    byUser = ply % 2 == 1,
                ),
                ply.toDouble(),
            )
        }
        coach.onEvent(
            Analysis(
                gameId = "g",
                fen = board.fen(),
                ply = 4,
                lines = listOf(Line(moves = listOf("f1b5"), depth = 12, expectedScore = 0.55)),
                decisive = false,
                onlyMove = false,
                tags = emptyList(),
            ),
            5.0,
        )
        messages(coach)
    }

    /**
     * "I'll play Qd1, blocking the check" answered the templates' hint, and got the hint back word
     * for word, three times (the students' round 5): a move proposed is examined as a question
     * about it is, and the model answers from the examination; and a template already said since
     * the last move is not said again (desktop adc4b26, 3ff672b).
     */
    @Test
    fun aMoveProposedInAStatementIsExaminedAndNothingIsSaidTwice() {
        val engine = EngineLike { board, _ ->
            if (board.turn == WHITE) {
                listOf(Line(moves = listOf("f1b5"), depth = 14, expectedScore = 0.55))
            } else {
                listOf(Line(moves = listOf("f8c5", "d2d3"), depth = 14, expectedScore = 0.48))
            }
        }
        val (coach, provider) = coach(listOf("[p=1.0] Fine."))
        coach.engine = engine
        coach.syncClaims = true // a question with the engine runs on a worker otherwise
        italian(coach)
        coach.ask("I'll play Bc4 and aim at f7.", 6.0)
        assertEquals(listOf("Fine."), messages(coach).map { it.text })
        val directive = directiveOf(given(provider))
        assertTrue(directive, "Engine on 3. Bc4, which the player asked about" in directive)
        // Offline, a statement naming no move gets the template once, and not again.
        val offline = Coach(OfflineProvider(), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
        italian(offline)
        offline.ask("I think my position is fine.", 7.0)
        val first = messages(offline).map { it.text }
        offline.ask("I think my position is fine.", 8.0)
        val again = messages(offline).map { it.text }
        assertTrue("$first / $again", first.isNotEmpty() && again.intersect(first.toSet()).isEmpty())
    }

    /**
     * Each lesson taught its own moment and a principle of its own, and the threat-blind
     * student's notebook held five unrelated rules (the simulated students, 2026-09-26): the
     * third mistake of one cause in the session says the pattern once, with its habit. Said the
     * second time and then in every game, "this keeps happening -- 5 times now" nagged (Tai's
     * phone, 2026-09-27); the review sums it up after (desktop f983cd0).
     */
    @Test
    fun aCauseThatComesBackIsNamedOnceWithTheHabitThatFixesIt() {
        val (coach, _) = coach(emptyList())
        start(coach)

        fun placed(ply: Int, cause: String, gameId: String = "g"): List<String> {
            coach.onEvent(
                ProfileEvidence(gameId = gameId, ply = ply, phase = "middlegame", expectedLoss = 0.2, cause = cause),
                ply.toDouble(),
            )
            return messages(coach).map { it.text }
        }

        assertEquals(emptyList<String>(), placed(9, Causes.MISSED_THREAT))
        assertEquals(emptyList<String>(), placed(13, Causes.POSITIONAL))
        assertEquals(emptyList<String>(), placed(21, Causes.MISSED_THREAT, gameId = "old")) // another game's
        assertEquals("twice is not a pattern yet", emptyList<String>(), placed(25, Causes.MISSED_THREAT))
        val third = placed(31, Causes.MISSED_THREAT)
        // Said as a skill to work on, never a count (Tai, 2026-10-02: "That is three times now"
        // read as scolding).
        assertTrue(third.toString(), third.size == 1 && third[0].startsWith("Something worth working on:"))
        assertTrue(third[0], "times" !in third[0] && "again" !in third[0])
        assertTrue(third[0], "spotting what their moves threaten" in third[0])
        assertTrue(third[0], "A habit that helps: before each move, look at what their last move attacks" in third[0])
        // Once a session: not again in this game, nor in the next.
        assertEquals(emptyList<String>(), placed(37, Causes.MISSED_THREAT))
        coach.gameId = "g2"
        assertEquals(emptyList<String>(), placed(9, Causes.MISSED_THREAT, gameId = "g2"))
        assertEquals(emptyList<String>(), placed(15, Causes.MISSED_THREAT, gameId = "g2"))

        // A player of 1800 and up hears the pattern without the textbook habit.
        val (strongCoach, _) = coach(emptyList())
        start(strongCoach)
        strongCoach.strongPlayer = true
        for (ply in listOf(9, 13, 17)) {
            strongCoach.onEvent(
                ProfileEvidence(gameId = "g", ply = ply, phase = "middlegame", expectedLoss = 0.2, cause = Causes.MISSED_THREAT),
                ply.toDouble(),
            )
        }
        val strong = messages(strongCoach).map { it.text }
        assertTrue(strong.toString(), strong.size == 1 && "spotting what their moves threaten" in strong[0])
        assertTrue(strong[0], "look at what" !in strong[0])
    }

    /**
     * A 2000 player was told "you kept missing endgame technique: bring your King to the middle
     * and push your passed pawns" (the AI's sessions, 2026-09-26; desktop 216dcb1).
     */
    @Test
    fun aStrongPlayersReviewNamesThePatternWithoutTheTextbookHabit() {
        val (coach, _) = coach(emptyList())
        coach.strongPlayer = true
        start(coach)
        val pattern = "One thing to work on: endgame technique. A habit that helps: in the endgame, bring " +
            "your king to the middle and push your passed pawns."
        coach.onEvent(
            GameReview(
                gameId = "g",
                userSide = Pieces.Side.WHITE,
                result = "1-0",
                moves = 40,
                lines = listOf("You played white and won in 40 moves.", pattern),
            ),
            1.0,
        )
        val said = messages(coach).joinToString(" ") { it.text }
        assertTrue(said, "One thing to work on: endgame technique." in said && "passed pawns" !in said)
        // What the model is shown of the review says it the same way.
        assertTrue("passed pawns" !in coach.conversation.render())
    }

    /**
     * The AI said "Qxf7+ is illegal here" and "Kf7 is illegal because f7 is controlled by the King
     * on e8" of legal moves (its own sessions, checked against the boards; desktop 403e675).
     */
    @Test
    fun aLegalMoveCalledIllegalIsAFault() {
        val board = Board("2r3k1/4bpp1/p3p2p/qp1bn1NQ/8/1P1BB3/P4PPP/3R2K1 w - - 0 20")
        assertEquals(
            listOf("Qxf7: called illegal, but it is a legal move here"),
            falseIllegal("No: Qxf7+ is illegal here.", board),
        )
        // A claim inside a line is about a later position; a true one is no fault.
        assertEquals(emptyList<String>(), falseIllegal("After Bh7+ Kh8, Qxf7+ is illegal.", board))
        assertEquals(emptyList<String>(), falseIllegal("Qh4 is not legal: the pawn blocks it.", Board()))
    }

    /** `_after_move_request`: a lesson's request on [fen] after [san], the engine preferring [bestUci]. */
    private fun afterMoveRequest(
        fen: String,
        san: String,
        bestUci: String,
        nowAnswer: List<String> = emptyList(),
        severity: Severity = Severity.MISTAKE,
        label: Label = Label.MISTAKE,
        expectedLoss: Double = 0.2,
        userText: String? = null,
    ): CoachRequest {
        val board = Board(fen)
        return CoachRequest(
            trigger = Trigger.POST_MISTAKE,
            level = DisclosureLevel.LINE,
            system = "",
            messages = emptyList(),
            facts = PositionFacts(fen = fen, sideToMove = "white", phase = "opening"),
            board = board,
            fallback = "",
            ply = 7,
            gameId = "g",
            userText = userText,
            playedSan = san,
            assessment = MoveAssessment(
                gameId = "g",
                ply = 7,
                uci = board.parseSan(san)!!.uci(),
                expectedLoss = expectedLoss,
                severity = severity,
                best = listOf(bestUci),
                label = label,
            ),
            nowAnswer = nowAnswer,
        )
    }

    /**
     * The owner's rules after a move, checked in code (the eval audit, 2026-09-25): never open
     * with the move just played, never say what it plainly did unless the better move does the
     * same, and no figure in a remark nobody asked for -- the last [NO_FIGURES]' note, where the
     * directive carries it ([Speaker.notes]) (desktop f25217c).
     */
    @Test
    fun aWordAfterTheMoveNeverRestatesItNorQuotesAFigure() {
        val ruy = "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
        val trade = afterMoveRequest(ruy, "Bxc6", "e1g1")
        assertEquals(listOf("opens with Bxc6, the move just played"), restated("Bxc6 gives up the bishop pair.", trade))
        assertEquals(
            listOf("says what Bxc6 plainly did: the capture on c6"),
            restated("Taking the knight on c6 gives up the bishop pair.", trade),
        )
        // The better move castles: saying so is the lesson, not a restatement.
        assertEquals(
            emptyList<String>(),
            restated("Castling first was better: the king is safe, the rook joins in.", trade),
        )
        // The better move takes on the same square with check: the words are about it.
        val miss = afterMoveRequest(SCHOLAR, "Bxf7+", "h5f7")
        assertEquals(emptyList<String>(), restated("Taking the pawn on f7 with check was right, but with the queen.", miss))
        // The check said of the move played is a restatement; a check in the better line is not
        // (the benchmark's smoke run, 2026-09-25: "after Bg4 your knight jumps in with check";
        // desktop 1117852).
        val f5 = "rnbqkbnr/ppppp1pp/8/5p2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        val check = afterMoveRequest(f5, "Qh5+", "e4f5")
        val said = "says what Qh5 plainly did: the check"
        assertTrue(said in restated("Sure, Qh5+ gives check, but exf5 wins a pawn.", check))
        assertTrue(said in restated("That gave check, but exf5 wins a pawn.", check))
        assertEquals(emptyList<String>(), restated("exf5 wins a pawn, and after Nf6 your queen comes in with check.", check))
    }

    /**
     * The first hint on the sheet is facts to teach from, not the least a reply says, so the
     * length is the directive's own: one more for the question it offers on top ("then, if you
     * like, one question"), which it does not require (the prompt audit, 2026-09-28).
     */
    @Test
    fun theFirstHintAsksTwoOrThreeSentencesAndOffersAQuestion() {
        assertEquals(4 to null, directiveLimits(IDEA_DIRECTIVE))
    }

    /**
     * A trap the template names ("Bxf7+ looks natural, but after Kxf7 ...") is a line from the
     * root: its boards are frames, so saying it is not faulted as unplayable (desktop f25217c).
     */
    @Test
    fun theDraftCheckerSeesEveryLineTheFactsWereReadFrom() {
        val base = mateAnalysis(6)
        val analysis = Analysis(
            gameId = base.gameId,
            fen = base.fen,
            ply = base.ply,
            lines = base.lines,
            decisive = base.decisive,
            onlyMove = base.onlyMove,
            traps = listOf(Line(moves = listOf("c4f7", "e8e7"), depth = 12, expectedScore = 0.4)),
            tags = base.tags,
        )
        val position = CoachPosition(ply = 6, board = Board(SCHOLAR), analysis = analysis)
        val (coach, _) = coach(emptyList())
        val frames = coach.analysisFrames(position)
        assertTrue(frames.any { it.pieceAt(parseSquare("e7")) == Piece(KING, BLACK) })
        assertTrue(frames.any { it.pieceAt(parseSquare("f7")) == Piece(QUEEN, WHITE) })
    }

    /**
     * `test_verdicts._played`: a sound move of the player's from the scholar's-mate position,
     * [san], labelled [label], losing [loss]; what the coach said.
     */
    private fun soundMove(coach: Coach, san: String, label: Label, loss: Double = 0.01): List<String> {
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        val board = Board(SCHOLAR)
        val move = board.parseSan(san)!!
        board.push(move)
        coach.onEvent(
            MovePlayed(gameId = "g", ply = 7, san = san, uci = move.uci(), fenBefore = SCHOLAR, fenAfter = board.fen(), byUser = true),
            8.0,
        )
        coach.onEvent(
            MoveAssessment(
                gameId = "g", ply = 7, uci = move.uci(), expectedLoss = loss, severity = Severity.NONE,
                best = listOf("h5f7"), label = label,
            ),
            8.5,
        )
        return messages(coach).map { it.text }
    }

    /**
     * Who writes the word on a sound move, by its label (Tai, 2026-09-20 and 2026-09-26: "They need
     * to go through the api"): theory, the engine's own move and an excellent one get the board's
     * sentence and no call (the routine model's, with the local verdicts off); a good move goes to
     * the model, which names the better one; a great or brilliant move to the MAIN model at its
     * full effort, told to show some delight, the board's line standing in when the reply fails.
     * The routine model is a separate one here, so a delight asked of it would show.
     * `test_a_good_move_goes_to_the_model_which_names_the_better_one` and
     * `test_a_great_or_brilliant_move_gets_the_models_delight` (tests/test_verdicts.py), and the
     * routine model's word with the local verdicts off
     * (`test_a_good_move_gets_one_sentence_from_the_model_with_the_template_to_fall_back_on`).
     */
    @Test
    fun aSoundMovesWordIsRoutedByItsLabel() {
        fun routed(replies: List<String>, localVerdicts: Boolean = true): Triple<Coach, ScriptedProvider, ScriptedProvider> {
            val main = ScriptedProvider(replies)
            val routine = ScriptedProvider(replies)
            val coach = Coach(
                main,
                policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)),
                routineProvider = routine,
                localVerdicts = localVerdicts,
            )
            return Triple(coach, main, routine)
        }
        // The board's own word: no call at all.
        for ((san, label) in listOf("Qxf7#" to Label.BEST, "Nc3" to Label.EXCELLENT, "Nc3" to Label.BOOK)) {
            val (coach, main, routine) = routed(listOf("never used"))
            val texts = soundMove(coach, san, label)
            assertTrue("$label: $texts", texts.size == 1 && !texts[0].startsWith(san.trimEnd('#', '+')))
            assertTrue("$label: no model", main.prompts.isEmpty() && routine.prompts.isEmpty())
            if (label == Label.BEST) assertTrue(texts[0], texts[0] in MATE)
            if (label == Label.EXCELLENT) {
                assertTrue(texts[0], texts[0].split(" -- ")[0].trimEnd('.') in OPENERS.getValue("excellent"))
            }
        }
        // The local verdicts off, the routine model writes it.
        run {
            val (coach, main, routine) = routed(listOf("[p=1.0] Mate on the board -- well played."), localVerdicts = false)
            assertEquals(listOf("Mate on the board -- well played."), soundMove(coach, "Qxf7#", Label.BEST))
            assertTrue(main.prompts.isEmpty() && routine.prompts.size == 1)
        }
        // A good move: the main model names the better one, the label said in its sense.
        run {
            val (coach, main, routine) = routed(listOf("[p=1.0] Fine, but Qxf7# was there: the mate."))
            assertEquals(listOf("Fine, but Qxf7# was there: the mate."), soundMove(coach, "Nc3", Label.GOOD, loss = 0.03))
            assertTrue(main.prompts.size == 1 && routine.prompts.isEmpty())
            val task = directive(main)
            assertTrue(task, "which was good: sound, a little behind Qxf7#" in task)
            assertTrue(task, "tells them nothing: do not open with it" in task)
        }
        // Great and brilliant: the main model, with delight; the board's line if it fails.
        for (label in listOf(Label.GREAT, Label.BRILLIANT)) {
            val (coach, main, routine) = routed(listOf("[p=1.0] Yes -- the one move, and it ends the game."))
            assertEquals(listOf("Yes -- the one move, and it ends the game."), soundMove(coach, "Qxf7#", label, loss = 0.0))
            assertTrue("$label: the main model", main.prompts.size == 1 && routine.prompts.isEmpty())
            val task = directive(main)
            assertTrue(task, "Show some delight" in task && label.name.lowercase() in task)
            assertTrue(task, "tells them nothing: do not open with it" in task)
            val (failing, _, _) = routed(listOf("", ""))
            val fallback = soundMove(failing, "Qxf7#", label, loss = 0.0)
            assertTrue("$label: $fallback", fallback.size == 1 && fallback[0] in MATE)
        }
    }

    /** `_mate_in_one_played`: the user plays Qxf7#, the engine's own move, after the scholar's position. */
    private fun mateInOnePlayed(coach: Coach, gameId: String = "g", ply: Int = 7, start: String = SCHOLAR, ts: Double = 10.0) {
        val board = Board(start)
        val move = board.parseSan("Qxf7#")!!
        board.push(move)
        coach.onEvent(MovePlayed(gameId = gameId, ply = ply, san = "Qxf7#", uci = move.uci(), fenBefore = start, fenAfter = board.fen(), byUser = true), ts)
        coach.onEvent(
            MoveAssessment(gameId = gameId, ply = ply, uci = move.uci(), expectedLoss = 0.0, severity = Severity.NONE, best = listOf("h5f7"), label = Label.BEST),
            ts + 0.5,
        )
    }

    /** `_play` of the given moves from the start, the user White, each at its ply's second. */
    private fun playFromStart(coach: Coach, sans: List<String>, thinkMs: Int? = null): Board {
        start(coach)
        val board = Board()
        for ((i, san) in sans.withIndex()) {
            val ply = i + 1
            val move = board.parseSan(san)!!
            val before = board.fen()
            board.push(move)
            coach.onEvent(
                MovePlayed(gameId = "g", ply = ply, san = san, uci = move.uci(), fenBefore = before, fenAfter = board.fen(), byUser = ply % 2 == 1, thinkMs = thinkMs),
                ply.toDouble(),
            )
        }
        return board
    }

    /** Naming the move yourself earns the candidate level (`test_coach.py`). */
    @Test
    fun namingTheMoveYourselfEarnsTheCandidateLevel() {
        // Offline the keyword rules (`unlockFor`) read the rung off the words. With the AI on the
        // rung is what the answer said: it names the move, so the move is said.
        val (coach, _) = coach(listOf("Yes, Qxf7 is mate.", "What do you think?"))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        messages(coach)
        coach.ask("Is Qxf7 mate?", 8.0)
        assertEquals(DisclosureLevel.MOVE, messages(coach)[0].level)
        assertEquals(DisclosureLevel.MOVE, coach.positions[6]!!.disclosure.level)
        val board = Board(SCHOLAR)
        assertEquals(Unlock.USER_NAMED_MOVE, unlockFor("Is Qxf7 mate?", board, listOf("Qxf7#")))
        assertEquals("not the candidate", null, unlockFor("What about Bxf7?", board, listOf("Qxf7#")))
        assertEquals(Unlock.ASK_LINE, unlockFor("show me the line", board, emptyList()))
        assertEquals(Unlock.HINT, unlockFor("HINT please", board, emptyList()))
        assertEquals(Unlock.ASK_MOVE, unlockFor("what should I play here", board, emptyList()))
        assertEquals(null, unlockFor("how are you", board, emptyList()))
        // Words, not a word: "open lines" opened the line, and "which move" or "the answer" about
        // a move already played opened the move, and the answer was given (the coach review,
        // 2026-10-01).
        for (said in listOf(
            "are there any open lines for my rooks?",
            "which move did they just play?",
            "which move did I miss there?",
            "what was the answer to their last threat?",
            "was that the best move?",
        )) {
            assertEquals(said, null, unlockFor(said, board, listOf("Qxf7#")))
        }
        for ((said, unlock) in listOf(
            "what's the line?" to Unlock.ASK_LINE,
            "what are the lines here?" to Unlock.ASK_LINE,
            "give me a line" to Unlock.ASK_LINE,
            "show me some lines" to Unlock.ASK_LINE,
            "what's the main variation?" to Unlock.ASK_LINE,
            "ok and after that?" to Unlock.ASK_LINE,
            "how would this position continues?" to Unlock.ASK_LINE,
            "which move should I play?" to Unlock.ASK_MOVE,
            "which move?" to Unlock.ASK_MOVE, // bare, the whole question (desktop 9416b99, 0da008b)
            "ok, which move ?" to Unlock.ASK_MOVE,
            "which move is best?" to Unlock.ASK_MOVE,
            "just tell me the answer" to Unlock.ASK_MOVE,
            "what's the answer to their threat?" to Unlock.ASK_MOVE,
        )) {
            assertEquals(said, unlock, unlockFor(said, board, listOf("Qxf7#")))
        }
    }

    /**
     * The model offers a leaky reply and a clean one in one distribution: the checked one is kept
     * at the first attempt; and a stump ("Given"), clean by default, never wins -- both calls are
     * kept for the log (`test_the_clean_candidate_is_kept_at_once_and_a_stump_never_reaches_the_user`
     * in tests/test_coach_verbalized.py).
     */
    @Test
    fun theCheckedCandidateIsKeptAndAStumpNeverWins() {
        val checker = DraftCheck { text, _, _, _ -> if ("rook on f7" in text) listOf("the rook on f7 is not there") else emptyList() }
        fun checked(replies: List<String>): Pair<Coach, List<CoachMessage>> {
            val coach = Coach(ScriptedProvider(replies), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)), draftCheck = checker)
            playToScholar(coach)
            coach.onEvent(mateAnalysis(6), 7.0)
            return coach to stuck(coach) // stuck: the first word unasked
        }
        var (coach, out) = checked(listOf("[p=0.6] Your rook on f7 is strong.\n[p=0.4] Their king on e8 is short of squares, is it not?"))
        assertEquals("Their king on e8 is short of squares, is it not?", out[0].text)
        assertTrue(coach.spoken.last().attempts == 1 && coach.speaker.fallbacks == 0)
        val first = "[p=0.6] Your rook on f7 is strong. What next?\n[p=0.4] Given"
        val second = "[p=0.7] Their king on e8 is short of squares. What is their threat?"
        val stump = checked(listOf(first, second))
        coach = stump.first
        out = stump.second
        assertEquals("Their king on e8 is short of squares. What is their threat?", out[0].text)
        val spoken = coach.spoken.last()
        assertEquals(2, spoken.attempts)
        assertEquals(listOf(first, second), spoken.calls.map { it.raw })
        assertEquals(listOf(1 to 0, 1 to 1), spoken.calls.map { it.candidates to it.passed })
    }

    /**
     * "hint" carries no claim: prompted to be honest about "their claim", the model answered its
     * own previous message as though the player had said it (`test_coach_verbalized.py`).
     */
    @Test
    fun aBareHintIsAskedForAsAHintNotAsAClaim() {
        for (said in listOf("hint", "Hint", "give me a hint", "a clue please", "help", "any tips?")) {
            assertTrue(said, ASK_HINT.matchesAt(said, 0))
        }
        for (said in listOf("what should I play", "is Nf3 a good idea", "help me with the knight")) {
            assertFalse(said, ASK_HINT.matchesAt(said, 0))
        }
        val (coach, provider) = coach(listOf("Where is their king weakest?"))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        coach.hint(8.0)
        messages(coach)
        val prompt = given(provider, 0).last().content
        // No keyword picked the hint's directive: the guidelines, with the ladder as it stands.
        assertTrue(prompt, "A hint: one rung up the ladder" in prompt && "nothing yet" in prompt)
        assertTrue(prompt, "They asked for a hint" !in prompt)
    }

    /** A re-read of the board is a message and the numbering restarts (`test_coach.py`). */
    @Test
    fun aRereadOfTheBoardIsAMessageAndTheNumberingRestarts() {
        val (coach, provider) = coach(listOf("Understood."))
        playToScholar(coach)
        val fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/2N5/PPPP1PPP/R1B1K1NR b KQkq - 5 4"
        coach.onEvent(Resync(gameId = "g", fen = fen, reason = "position adopted after 3 unexplained cells"), 20.0)
        val board = Board(fen)
        val move = board.parseSan("d6")!!
        board.push(move)
        coach.onEvent(MovePlayed(gameId = "g", ply = 1, san = "d6", uci = move.uci(), fenBefore = fen, fenAfter = board.fen(), byUser = false), 21.0)
        coach.onEvent(
            Analysis(gameId = "g", fen = board.fen(), ply = 1, lines = listOf(Line(moves = listOf("c3d5"), depth = 12, expectedScore = 0.6)), decisive = false, onlyMove = false),
            22.0,
        )
        coach.ask("what now?", 23.0)
        messages(coach)
        val text = harness(given(provider))
        assertTrue(text, "    EVENT: The board was re-read" in text && "Moves may have gone unseen" in text)
        // The numbering follows the position the board was read at, not a ply count.
        assertTrue("4... d6 by the opponent" in currentMoveMessage(given(provider)))
    }

    /**
     * The claims in the player's words are checked off the caller's thread -- the caller never
     * waits -- and only when there are any: a bare hint never runs the check (`test_coach.py`).
     */
    @Test
    fun claimsAreCheckedOffTheCallersThreadAndOnlyWhenThereAreAny() {
        assertFalse(mayCarryClaims("hint"))
        assertFalse(mayCarryClaims("what should I play"))
        assertFalse(mayCarryClaims("show me the line"))
        assertTrue(mayCarryClaims("is Nf3 good here"))
        assertTrue(mayCarryClaims("my knight is hanging"))
        assertTrue(mayCarryClaims("can I castle"))
        val threads = java.util.Collections.synchronizedList(ArrayList<String>())
        val provider = ScriptedProvider(listOf("Not quite. What is your most active piece already eyeing?"))
        val coach = Coach(
            provider,
            policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)),
            claims = ClaimsCheck { _, _, _ ->
                threads.add(Thread.currentThread().name)
                listOf("'Nf3 is good': not the strongest, Qxf7# mates")
            },
        )
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        coach.ask("is Nf3 good here", 8.0)
        // The check runs on its own thread, so the caller never waits on it: wait for that
        // thread to end -- an event, not a time -- and assert on where it ran and what it said.
        joinThreads("sidecoach-claims")
        assertEquals(listOf("sidecoach-claims"), threads.toList())
        assertEquals(listOf("Not quite. What is your most active piece already eyeing?"), messages(coach).map { it.text })
        assertTrue("Their claims, checked against the board" in given(provider, 0).last().content)
        coach.hint(9.0) // a bare hint never runs the check
        messages(coach)
        assertEquals(1, threads.size)
    }

    /**
     * "How did I do overall?" was answered "you were down a queen" to a player who had won one:
     * the material is the user's side's, a finished game withholds nothing, the overall verdict
     * is drawn from the tally, and when the model's review fails the template is the record
     * (`test_coach.py`).
     */
    @Test
    fun materialIsAlwaysFromTheUsersSideAndTheReviewDrawsOnTheTally() {
        val (coach, provider) = coach(listOf("You kept it clean: nothing lost, and the finish was sharp."))
        val board = playFromStart(coach, listOf("e4", "e5", "Qh5", "Nc6", "Qxe5+"))
        coach.onEvent(
            Analysis(gameId = "g", fen = board.fen(), ply = 5, lines = listOf(Line(moves = listOf("f8e7"), depth = 12, expectedScore = 0.4)), decisive = false, onlyMove = false),
            6.0,
        )
        val facts = coach.positions[5]!!.facts!!
        assertEquals("from the user's side, not the mover's", "you are up a pawn", facts.material)
        assertTrue(facts.userSide == "white" && facts.sideToMove == "black")
        assertTrue("Material: the player is up a pawn." in coach.conversation.renderPly(coach.conversation.records.last(), DisclosureLevel.LINE))
        for ((ply, pair) in mapOf(1 to ("e2e4" to Label.BOOK), 3 to ("d1h5" to Label.INACCURACY), 5 to ("h5e5" to Label.BEST))) {
            val (uci, label) = pair
            val inaccuracy = label == Label.INACCURACY
            coach.onEvent(
                MoveAssessment(
                    gameId = "g", ply = ply, uci = uci, expectedLoss = if (inaccuracy) 0.03 else 0.0,
                    severity = if (inaccuracy) Severity.INACCURACY else Severity.NONE, best = listOf(uci), label = label,
                ),
                7.0,
            )
        }
        messages(coach)
        coach.onEvent(GameEnded(gameId = "g", result = "1-0", reason = "resigned", pgn = "", finalFen = board.fen()), 8.0)
        coach.ask("how did I do overall this game?", 9.0)
        val out = messages(coach)
        assertTrue(out.isNotEmpty() && out.last().level == DisclosureLevel.LINE)
        val directive = directiveOf(given(provider))
        assertTrue(directive, "How they did overall: from the verdicts and the tally in the record" in directive)
        assertTrue(directive, "nothing inferred from the position on the board" in directive)
        val text = harness(given(provider))
        assertTrue(text, "The player's moves so far (3 judged): 1 best, 1 book, 1 inaccuracy." in text)
        assertTrue(text, "Ground lost most at: 2. Qh5 (inaccuracy)" in text)
        assertTrue(text, "    EVENT: Game over: 1-0" in text)
        for (words in listOf("what do I keep missing?", "where did I lose it?", "how am I playing so far?")) {
            assertTrue(words, ASK_REVIEW.containsMatchIn(words))
        }
        assertFalse(ASK_REVIEW.containsMatchIn("what's the plan so far?"))
        provider.replies.clear()
        provider.replies.addAll(listOf("", ""))
        coach.ask("what do I keep missing?", 10.0)
        assertEquals(
            "1 of your 3 moves were best. You lost the most ground at 2. Qh5 (inaccuracy).",
            messages(coach).last().text,
        )
    }

    /**
     * The model gives nothing usable, twice: the template for the rung speaks in its place, once,
     * and the failed calls are in the log. The second try carries the complaint inside the
     * directive, as many messages as the first: appending the failed attempt as two more
     * messages doubled the prompt on a long game, past the model's context
     * (`test_a_bare_hint_is_asked_as_a_hint_and_one_the_model_fails_twice_is_the_template`,
     * desktop 3ff672b; the retry's shape, the parity audit, 2026-09-29).
     */
    @Test
    fun aHintTheModelFailsTwiceIsTheTemplates() {
        val (coach, provider) = coach(listOf("", ""))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        coach.ask("hint", 8.0)
        val out = messages(coach)
        // The idea template: the goal and the pattern.
        assertTrue(out.map { it.text }.toString(), out.size == 1 && "mate in one" in out[0].text.split(".")[0])
        // Nothing was said before the model was asked: their words are the last turn.
        val first = given(provider, 0)
        val said = currentMoveMessage(first).substringAfter("## Said here").trim('\n').lines()
        assertEquals("    PLAYER said: \"hint\"", said.last())
        val directive = directiveOf(first)
        // No keyword picked the hint's directive: the guidelines, with the ladder as it stands.
        assertTrue(directive, "A hint: one rung up the ladder" in directive && "nothing yet" in directive)
        assertTrue(directive, "They asked for a hint." !in directive)
        val retry = given(provider, 1)
        assertEquals("the retry adds no message", first.size, retry.size)
        assertTrue(retry.last().content, "Those replies are not allowed" in retry.last().content)
        assertTrue(retry.last().content, "Your first attempt was" in retry.last().content)
        val spoken = coach.spoken.last()
        assertTrue(spoken.attempts == 2 && spoken.problems.isNotEmpty() && coach.speaker.fallbacks == 1)
        assertEquals(listOf(out[0].text), coach.conversation.entries.filter { it.role == "assistant" }.map { it.text })
    }

    /** A question about a fine last move says it was fine and invents no fault (`test_coach.py`). */
    @Test
    fun aQuestionAboutAFineLastMoveSaysItWasFine() {
        val (coach, provider) = coach(listOf("...", ""))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        messages(coach)
        mateInOnePlayed(coach, ts = 10.0)
        messages(coach)
        coach.ask("was that a mistake?", 12.0)
        messages(coach)
        val directive = directiveOf(given(provider))
        // The verdict is in the record the model reads; the guidelines say a sound move gets no
        // invented fault.
        assertTrue(directive, "inventing no fault in a sound move" in directive)
        assertTrue(harness(given(provider)), "4. Qxf7# by the player: best" in harness(given(provider)))
    }

    /**
     * A question about the player over their games goes to the record's answerer off the coach's
     * thread; the answer is the coach's turn, the lookup noted; a question about this game stays
     * with the game (`test_coach.py`).
     */
    @Test
    fun aQuestionAboutThePlayerOverTheirGamesGoesToTheRecord() {
        val (coach, _) = coach(listOf("Keep looking."))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 6.5)
        val asked = java.util.Collections.synchronizedList(ArrayList<String>())
        coach.profileAsk = { question ->
            asked.add(question)
            ProfileAnswer(
                text = "Over 9 games you left the opening ahead 7 times and were behind by move 25.",
                exchanges = listOf("SELECT 1"),
                calls = 2,
            )
        }
        coach.ask("why am I always losing in the middlegame?", 7.0)
        // The answer lands on the lookup's own thread: wait for that thread to end, not for a
        // time, then read what was said.
        joinThreads("profile-ask")
        val said = messages(coach)
        assertEquals(listOf("why am I always losing in the middlegame?"), asked.toList())
        assertTrue(said.isNotEmpty() && said.last().text.startsWith("Over 9 games") && said.last().trigger == Trigger.USER_ASK)
        assertEquals("answer", coach.profileAnswers[0].first)
        val record = coach.conversation.render()
        assertTrue("PLAYER said: \"why am I always losing in the middlegame?\"" in record)
        assertTrue("looked the question up in the record (1 query)" in record)
        coach.ask("what should I play here?", 8.0)
        assertEquals(1, asked.size)
    }

    /**
     * Offline, "what do I usually miss?" answered "I could not read the record just now; ask me
     * again in a moment." every time: no model writes the queries, so the moment never came (the
     * audit, 2026-10-07). The offline coach answers with what the record already says of the
     * player (the profile's insights); with nothing there yet, it says so plainly
     * (`test_offline_a_question_about_the_record_gets_what_the_record_says`).
     */
    @Test
    fun offlineAQuestionAboutTheRecordGetsWhatTheRecordSays() {
        val insights = listOf(
            "The thing to work on most over 12 games: forks you missed, which came up in 9 moves.",
            "Worth practising: the fork. You found it 3 of the 8 times you had one.",
        )
        for ((given, expected) in listOf(insights to insights.joinToString(" "), emptyList<String>() to RECORD_EMPTY)) {
            val coach = Coach(OfflineProvider(), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
            playToScholar(coach)
            coach.onEvent(mateAnalysis(6), 6.5)
            coach.onEvent(com.sidecoach.events.ProfileSnapshot(games = 12, insights = given), 6.6)
            // As the app wires it, the record's answerer fails with no model to call.
            coach.profileAsk = { _ -> throw IllegalStateException("no model: llm.provider is none") }
            coach.ask("what do I usually miss?", 7.0)
            joinThreads("profile-ask")
            assertEquals(listOf(expected), messages(coach).map { it.text })
        }
    }

    /**
     * "What were you expecting that move to do?" ended every third lesson, for the profile's
     * record; after a mistake it reads as a challenge (Tai, 2026-10-02: "very rude and super
     * unwelcoming"). Neither the model's instruction nor the template asks it, however many
     * mistakes came before (`test_no_lesson_on_a_mistake_asks_what_the_player_expected`).
     */
    @Test
    fun noLessonOnAMistakeAsksWhatThePlayerExpected() {
        val (coach, _) = coach(emptyList())
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 6.5)
        val position = coach.positions[6]!!
        val played = MovePlayed(gameId = "g", ply = 7, san = "a3", uci = "a2a3", fenBefore = SCHOLAR, fenAfter = SCHOLAR, byUser = true)
        val assessment = MoveAssessment(gameId = "g", ply = 7, uci = "a2a3", expectedLoss = 0.5, severity = Severity.BLUNDER, best = listOf("h5f7"), label = Label.BLUNDER)
        val (instruction, fallback) = coach.instruction(Trigger.POST_MISTAKE, position, DisclosureLevel.LINE, null, emptyList(), assessment, played)
        assertTrue(instruction, "expect" !in instruction && "expect" !in fallback)
        // The offline coach's lessons: CoachCoreTest.noOfflineLessonAsksWhatThePlayerExpected.
    }

    /**
     * A puzzle or an analysis board: the coach hears it is a position with no history; the mating
     * move on it gets no word and the end no summary, and nothing calls the model; a move that
     * does not end it gets its word as in a game (`test_coach.py`).
     */
    @Test
    fun aPositionIsNotAGameAndAPuzzleSolvedByMateGetsNoWord() {
        val (told, _) = coach(emptyList())
        val setup = "r3k2r/pp3ppp/2n1b3/2bpp3/8/1PN1PN2/PBP2PPP/R3KB1R b KQkq - 0 1"
        told.onEvent(GameStarted(gameId = "p", userSide = Pieces.Side.BLACK, startFen = setup, kind = "position"), 0.0)
        val record = told.conversation.render()
        assertTrue(record, "You coach Black." in record && "This is a position, not a game from the start" in record)
        assertTrue(record, "joined in progress" !in record)

        val (coach, provider) = coach(emptyList())
        for (n in 0 until 2) {
            val id = "p$n"
            coach.onEvent(GameStarted(gameId = id, userSide = Pieces.Side.WHITE, startFen = SCHOLAR, kind = "position"), 10.0 * n)
            coach.onEvent(mateAnalysis(0).let { Analysis(gameId = id, fen = it.fen, ply = 0, lines = it.lines, decisive = it.decisive, onlyMove = it.onlyMove, tags = it.tags) }, 10.0 * n + 0.5)
            assertEquals(emptyList<CoachMessage>(), messages(coach)) // the position itself: nothing said unasked
            provider.prompts.clear()
            mateInOnePlayed(coach, gameId = id, ply = 1, ts = 10.0 * n + 1.0)
            assertEquals("the mating move is not commented on", emptyList<CoachMessage>(), messages(coach))
            coach.onEvent(GameEnded(gameId = id, result = "1-0", reason = "game over on board", pgn = "1. Qxf7#"), 10.0 * n + 2.0)
            assertEquals("no summary of a puzzle", emptyList<CoachMessage>(), messages(coach))
        }
        assertTrue("no model call at all", provider.prompts.isEmpty())
        coach.onEvent(GameStarted(gameId = "q", userSide = Pieces.Side.WHITE, startFen = SCHOLAR, kind = "position"), 50.0)
        coach.onEvent(mateAnalysis(0).let { Analysis(gameId = "q", fen = it.fen, ply = 0, lines = it.lines, decisive = it.decisive, onlyMove = it.onlyMove, tags = it.tags) }, 50.5)
        messages(coach)
        val board = Board(SCHOLAR)
        val move = board.parseSan("Nf3")!!
        board.push(move)
        coach.onEvent(MovePlayed(gameId = "q", ply = 1, san = "Nf3", uci = move.uci(), fenBefore = SCHOLAR, fenAfter = board.fen(), byUser = true), 51.0)
        coach.onEvent(
            MoveAssessment(gameId = "q", ply = 1, uci = move.uci(), expectedLoss = 0.5, severity = Severity.BLUNDER, best = listOf("h5f7"), label = Label.BLUNDER),
            51.5,
        )
        assertEquals(listOf(Trigger.POST_MISTAKE), messages(coach).map { it.trigger })
    }

    /**
     * The user's think times set the pace: in a fast game the word on a move is the template, at
     * once, with no model call; what the player asks still goes to the model; a player who slows
     * down -- in the game, or thinking longer than the pace in the position -- gets the model back
     * (Tai, 2026-09-22; `test_coach.py`).
     */
    @Test
    fun aFastGameGetsTheTemplatesAndSlowingDownBringsTheModelBack() {
        val (coach, provider) = coach(listOf("never used for a move", "[p=1.0] Asked and answered."))
        coach.localVerdicts = false
        val board = playFromStart(coach, listOf("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"), thinkMs = (FAST_PACE_S * 1000 / 3).toInt())
        assertTrue("two quick moves of the user's and the game counts as fast", coach.fastPace)
        assertTrue(coach.userToMove && coach.fastNow(6.0 + 1.0))
        assertFalse("thinking longer than the pace: slowed", coach.fastNow(6.0 + FAST_PACE_S + 1))
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        mateInOnePlayed(coach)
        var out = messages(coach)
        assertTrue(out.size == 1 && out[0].trigger == Trigger.POST_MOVE && out[0].text.isNotEmpty())
        assertTrue("no model call in a fast game", provider.prompts.isEmpty())
        coach.ask("what was that about", 12.0)
        out = messages(coach)
        assertTrue(out.isNotEmpty() && out.last().trigger == Trigger.USER_ASK)
        assertTrue("the ask goes to the model", provider.prompts.isNotEmpty())
        for (ply in listOf(8, 9)) {
            coach.onEvent(
                MovePlayed(gameId = "g", ply = ply, san = "a3", uci = "a2a3", fenBefore = board.fen(), fenAfter = board.fen(), byUser = true, thinkMs = (FAST_PACE_S * 3000).toInt()),
                20.0 + ply,
            )
        }
        assertFalse(coach.fastPace)
    }

    /**
     * A puzzle's last moves come quick -- the line found, played out -- and carried over they
     * made the next puzzle fast from its first move: a find after two minutes' thought was only
     * shown (Tai's puzzles, 2026-09-27: "sometimes it's just completely muted"; desktop 8af8695,
     * now `test_the_pace_is_the_last_think_times_and_the_games_own`).
     */
    @Test
    fun thePaceIsTheGamesOwn() {
        val (coach, _) = coach(emptyList())
        playFromStart(coach, listOf("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"), thinkMs = 1000)
        assertTrue(coach.fastPace)
        coach.onEvent(GameStarted(gameId = "h", userSide = Pieces.Side.WHITE, startFen = Board().fen()), 10.0)
        assertTrue(coach.paceS == null && !coach.fastPace)
        // The next puzzle in the same orientation comes as a re-read of the board: the pace was
        // kept, and its first find was only shown (the game-state review, 2026-10-01).
        val (again, _) = coach(emptyList())
        playFromStart(again, listOf("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"), thinkMs = 1000)
        assertTrue(again.fastPace)
        again.onEvent(Resync("g", SCHOLAR, "position adopted"), 10.0)
        assertTrue(again.paceS == null && !again.fastPace)
    }

    /**
     * No model at all (llm.provider none): the factory's offline provider; a decisive moment says
     * nothing, and stuck, the player gets the idea's template; the player's best move the verdict
     * from the board, a typed question the hint template -- and nothing is ever called
     * (`test_offline_coach.py`, `test_a_stuck_hint_and_a_move_get_the_templates_with_no_call`).
     */
    @Test
    fun offlineTheTemplatesSpeakAndNothingIsCalled() {
        // The factory gives the offline provider, and it reports itself; its ledger, empty, is
        // LlmTest's (`test_the_factory_gives_the_offline_provider_and_it_reports_itself`).
        val none = com.sidecoach.llm.makeProvider(com.sidecoach.llm.LLMSettings(provider = "none"))
        assertTrue(none is OfflineProvider && none.offline)
        val (ok, detail) = none.healthcheck()
        assertTrue(detail, ok && "offline" in detail)
        assertTrue((none as OfflineProvider).withEffort("low") === none)
        val coach = Coach(OfflineProvider(), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
        playToScholar(coach)
        // A decisive moment says nothing; stuck, the idea's template, no model.
        coach.onEvent(mateAnalysis(6), 7.0)
        assertEquals(emptyList<CoachMessage>(), messages(coach))
        var out = stuck(coach)
        assertTrue(out.size == 1 && out[0].trigger == Trigger.STUCK && "?" in out[0].text)
        assertEquals(1, coach.speaker.fallbacks)
        coach.ask("hint", 8.0)
        out = messages(coach)
        assertTrue("a hint from the templates", out.size == 1 && out[0].text.isNotEmpty())
        assertEquals(out[0].fen, Board(out[0].fen).fen())
        mateInOnePlayed(coach)
        out = messages(coach)
        assertTrue(out.size == 1 && out[0].trigger == Trigger.POST_MOVE)
        assertTrue(out[0].text, out[0].text in MATE && "Qxf7#" !in out[0].text)
        assertEquals("nothing was ever called", 0, coach.speaker.failures)
    }

    /**
     * Game 2 answered "how did I do overall?" with game 1's sentences when the model failed
     * (the parity audit, 2026-09-26; desktop c5c488f).
     */
    @Test
    fun aNewGameForgetsTheLastGamesReview() {
        val (coach, _) = coach(emptyList())
        start(coach)
        coach.reviewLines = listOf("You played white and won by checkmate in 4 moves.")
        start(coach)
        assertEquals(emptyList<String>(), coach.reviewLines)
    }

    /**
     * Saving a new rating rewords the system message and the templates alike: the register was
     * kept until the coach was rebuilt (desktop c5c488f). PanelWarm's Save calls [Coach.reword].
     */
    @Test
    fun aSavedRatingChangesTheTemplatesRegister() {
        val coach = Coach(ScriptedProvider(emptyList()), rating = 800 to 1100)
        assertTrue(coach.plainWords && !coach.strongPlayer)
        coach.reword(null, 2000 to 2300)
        assertTrue(!coach.plainWords && coach.strongPlayer)
        assertTrue("2000" in coach.system)
    }

    /** `test_engine_tools.FirstLegal`: the first legal moves by UCI, a line of four, a little ahead. */
    private val firstLegal = EngineLike { board, depth ->
        val replay = board.copy(withStack = false)
        val moves = ArrayList<String>()
        repeat(4) {
            val legal = replay.legalMoves().sortedBy { it.uci() }
            if (legal.isEmpty()) return@repeat
            moves.add(legal[0].uci())
            replay.push(legal[0])
        }
        if (moves.isEmpty()) emptyList() else listOf(Line(moves = moves, depth = depth ?: 14, expectedScore = 0.6))
    }

    /**
     * Asked "ok and after that?" with the line already given, the engine is asked about the
     * line's end before the model answers (Luna at no effort gave the same line back, Tai's
     * puzzle; `test_engine_tools.py`).
     */
    @Test
    fun theLineAskedForAgainBringsTheEngineOnItsEnd() {
        val (coach, provider) = coach(listOf("Qxf7# -- it is mate at once.", "Qxf7# -- it is mate at once."))
        coach.engine = firstLegal
        coach.syncClaims = true // a question with the engine runs on a worker otherwise
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        messages(coach)
        coach.ask("show me the line", 8.0)
        messages(coach)
        assertFalse(LINE_GIVEN in directive(provider))
        coach.ask("ok and after that?", 9.0)
        messages(coach)
        val task = directive(provider)
        assertTrue(task, LINE_GIVEN in task)
        assertTrue(task, "The engine, asked about the line's end:" in task)
        assertTrue(task, "The game is over there: checkmate, White wins." in task)
    }

    /**
     * Luna with a reasoning effort takes no tools on chat completions: the 400 left the coach
     * silent on every question. A refused request is asked again without the tools, and the
     * refusal is on the record; a server that is down is not asked again (`test_engine_tools.py`).
     */
    @Test
    fun aServerThatRefusesTheToolsIsAskedWithoutThem() {
        class Refusing(val error: String) : ToolsProvider {
            override val name = "refusing"
            val seen = ArrayList<Boolean>() // whether each call carried the tools

            override fun healthcheck(): Pair<Boolean, String> = true to "refusing"

            override fun chat(
                system: String,
                messages: List<ChatMessage>,
                maxTokens: Int,
                temperature: Double,
                thinking: Boolean,
            ): Completion {
                synchronized(seen) { seen.add(false) }
                return Completion(text = "The pawn on f7 is defended only by the King.")
            }

            override fun chatTools(
                system: String,
                messages: List<ChatMessage>,
                tools: JSONArray,
                turns: List<ToolRound>,
                maxTokens: Int,
                temperature: Double,
                toolChoice: String?,
                onDelta: StreamCallback?,
            ): Completion {
                synchronized(seen) { seen.add(true) }
                throw RuntimeException(error)
            }
        }
        for ((error, answered) in listOf("400: tools not supported" to true, "503: down" to false)) {
            val provider = Refusing(error)
            val coach = Coach(provider, policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
            coach.engine = firstLegal
            coach.syncClaims = true // a question with the engine runs on a worker otherwise
            playToScholar(coach)
            coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
            messages(coach)
            synchronized(provider.seen) { provider.seen.clear() }
            coach.ask("is f7 weak?", 8.0)
            val said = messages(coach).map { it.text }
            assertEquals(error, answered, "The pawn on f7 is defended only by the King." in said)
            assertEquals(error, answered, synchronized(provider.seen) { false in provider.seen })
            if (answered) {
                // The refusal on the record: asked without the tools, the reply had none of them.
                val stats = coach.spoken.last().calls.last().stats
                assertEquals("400: tools not supported", stats["tools_refused"])
            }
        }
    }

    /**
     * `test_engine_tools.ToolModel`: a model that takes tools. It makes the calls of each round
     * of [rounds] in turn, then answers with [answer]; every call is kept, with the tools, the
     * turns and the tool choice it was given (a plain call with none).
     */
    private class ToolModel(rounds: List<List<ToolCall>>, private val answer: String) : ToolsProvider {
        class Call(val tools: JSONArray?, val turns: List<ToolRound>?, val toolChoice: String?)

        override val name = "tool-model"
        private val script = ArrayDeque(rounds)
        val seen = ArrayList<Call>()

        override fun healthcheck(): Pair<Boolean, String> = true to "tools"

        override fun chat(
            system: String,
            messages: List<ChatMessage>,
            maxTokens: Int,
            temperature: Double,
            thinking: Boolean,
        ): Completion {
            synchronized(seen) { seen.add(Call(null, null, null)) }
            return Completion(text = answer)
        }

        override fun chatTools(
            system: String,
            messages: List<ChatMessage>,
            tools: JSONArray,
            turns: List<ToolRound>,
            maxTokens: Int,
            temperature: Double,
            toolChoice: String?,
            onDelta: StreamCallback?,
        ): Completion {
            synchronized(seen) { seen.add(Call(tools, turns, toolChoice)) }
            if (script.isNotEmpty() && toolChoice != "none") return Completion(text = "", toolCalls = script.removeFirst())
            return Completion(text = answer)
        }
    }

    /** `test_engine_tools._at_scholar`: the coach with [model] and the engine, the player to move at SCHOLAR, analysed. */
    private fun atScholar(model: LLMProvider): Coach {
        val coach = Coach(model, policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
        coach.engine = firstLegal
        coach.syncClaims = true // a question with the engine runs on a worker otherwise
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        messages(coach)
        return coach
    }

    /**
     * The Speaker's rounds of engine calls: the first call carries the tools and no turns; the
     * second, the model's call and the engine's answer to it as the call's result; the Task
     * says what the tools are for, and the calls are on the record for the log
     * (`test_the_model_asks_the_engine_and_answers_from_what_it_returns` in
     * tests/test_engine_tools.py; the parity audit, 2026-09-29).
     */
    @Test
    fun theModelAsksTheEngineAndAnswersFromWhatItReturns() {
        val model = ToolModel(
            listOf(listOf(ToolCall(id = "c1", name = "analyse", arguments = JSONObject().put("moves", JSONArray(listOf("Qxf7#")))))),
            "Qxf7# is mate: the queen takes on f7 and the king has no square.",
        )
        val coach = atScholar(model)
        synchronized(model.seen) { model.seen.clear() }
        coach.ask("what happens if I take on f7 with the queen?", 8.0)
        val out = messages(coach)
        assertTrue(out.map { it.text }.toString(), out.isNotEmpty() && out.last().text.startsWith("Qxf7# is mate"))
        val first = model.seen[0]
        val second = model.seen[1]
        assertTrue("the first call carries the tools and no turns", first.tools!!.length() > 0 && first.turns!!.isEmpty())
        // The engine's answer went back to the model as the call's result.
        val turns = second.turns!!
        assertEquals(turns.toString(), 1, turns.size)
        val turn = turns.single()
        assertEquals(listOf("analyse"), turn.reply.toolCalls.map { it.name })
        val result = turn.results.single()
        assertTrue(result, "The game is over there: checkmate, White wins." in result)
        // The Task says what the tools are for; the call is on the record for the log.
        val logged = coach.spoken.last().calls.last()
        assertTrue(logged.prompt, TOOL_WORDS in logged.prompt)
        assertEquals(listOf("analyse(Qxf7#)"), logged.stats["tools"])
    }

    /**
     * A model that keeps asking is asked [TOOL_ROUNDS] + 1 times, the last with the tools
     * closed ("none"), and its answer then stands
     * (`test_a_model_that_keeps_asking_answers_after_the_last_round` in
     * tests/test_engine_tools.py; the parity audit, 2026-09-29).
     */
    @Test
    fun aModelThatKeepsAskingAnswersAfterTheLastRound() {
        val calls = (0 until 9).map { ToolCall(id = "c$it", name = "square", arguments = JSONObject().put("square", "f7")) }
        val model = ToolModel(calls.map { listOf(it) }, "The pawn on f7 is defended only by the king.")
        val coach = atScholar(model)
        synchronized(model.seen) { model.seen.clear() }
        coach.ask("is f7 weak?", 8.0)
        val out = messages(coach)
        assertTrue(out.map { it.text }.toString(), out.isNotEmpty() && out.last().text == "The pawn on f7 is defended only by the king.")
        assertEquals(TOOL_ROUNDS + 1, model.seen.size)
        assertEquals("none", model.seen.last().toolChoice)
    }

    /**
     * The practice the profile picked, in one line naming the count and the motif, the level and
     * why; a word on a failed puzzle, naming its motif; nothing on a solve (the site says so),
     * and nothing when the coach is kept quiet (the panel's own board, whose bar says it). No
     * model is called (`test_the_coach_says_one_line_on_the_practice_and_a_word_on_a_fail` in
     * tests/test_puzzles.py; the parity audit, 2026-09-29).
     */
    @Test
    fun theCoachSaysOneLineOnThePracticeAndAWordOnAFail() {
        val (coach, provider) = coach(emptyList())
        start(coach)
        coach.onEvent(
            Recommendation(
                gameId = "g",
                motif = "fork",
                motifWords = "fork",
                reason = "fork: found 2 of 5 times over your games",
                level = 1100,
                puzzles = listOf(PuzzleRef(id = "fork1", url = "https://lichess.org/training/fork1", rating = 1000)),
            ),
            1.0,
        )
        val practice = messages(coach).map { it.text }
        assertEquals(practice.toString(), 1, practice.size)
        assertTrue(practice[0], "1 fork puzzle " in practice[0] && "1100" in practice[0])
        assertTrue(practice[0], "fork: found 2 of 5 times over your games" in practice[0])
        coach.onEvent(PuzzleOutcome(gameId = "p", puzzleId = "fork1", motif = "fork", solved = true, level = 1125), 2.0)
        assertEquals("a solve: the site says so", emptyList<CoachMessage>(), messages(coach))
        val failed = PuzzleOutcome(gameId = "p", puzzleId = "fork1", motif = "fork", solved = false, level = 1100)
        coach.onEvent(failed, 3.0)
        val word = messages(coach).map { it.text }
        assertTrue("a word on the fail, naming the motif: $word", word.size == 1 && "fork puzzle" in word[0])
        assertTrue("no model call", provider.prompts.isEmpty())
        assertTrue("Practice recommended" in coach.conversation.render())
        coach.quiet = true
        coach.onEvent(failed, 4.0)
        assertEquals(emptyList<CoachMessage>(), messages(coach))
    }

    /**
     * A clock on the screen scales the policy: its peak bands the time control, and the user's
     * own clock -- Black's for a player of Black -- sets the stuck threshold and the scramble; a
     * new game clears all three until a clock is seen again
     * (`test_clock_scales_the_policy_and_a_new_game_resets_it` in tests/test_coach.py, the
     * remaining time read here through the threshold and the scramble it sets; the parity
     * audit, 2026-09-29).
     */
    @Test
    fun theClockScalesThePolicyAndANewGameResetsIt() {
        val (coach, _) = coach(emptyList())
        start(coach)
        assertEquals("before any clock, untimed", TimeControl.UNTIMED, coach.policy.timeControl)
        coach.onEvent(ClockObservation(whiteMs = 600_000, blackMs = 600_000, running = Pieces.Side.WHITE), 1.0)
        assertEquals(TimeControl.RAPID, coach.policy.timeControl)
        coach.policy.settings = PolicySettings(stuckSeconds = 45.0)
        // White's 50 s, not Black's 300 (which would give 24 s): the floor, under the setting.
        coach.onEvent(ClockObservation(whiteMs = 50_000, blackMs = 300_000, running = Pieces.Side.WHITE), 2.0)
        assertEquals(20.0, coach.policy.stuckThreshold(), 1e-9)
        assertFalse(coach.policy.inScramble)
        // A new game: untimed, the remaining time gone, the setting's threshold again.
        coach.onEvent(GameStarted(gameId = "g2", userSide = Pieces.Side.BLACK, startFen = Board().fen()), 3.0)
        assertEquals(TimeControl.UNTIMED, coach.policy.timeControl)
        assertEquals(45.0, coach.policy.stuckThreshold(), 1e-9)
        // The user plays Black: Black's 25 s is a scramble; about three minutes at most is
        // blitz, not the last game's peak of ten.
        coach.onEvent(ClockObservation(whiteMs = 185_000, blackMs = 25_000, running = Pieces.Side.BLACK), 4.0)
        assertTrue("Black's clock is the player's", coach.policy.inScramble)
        assertEquals(TimeControl.BLITZ, coach.policy.timeControl)
        coach.onEvent(ClockObservation(whiteMs = 25_000, blackMs = 180_000, running = Pieces.Side.BLACK), 5.0)
        assertFalse("White's clock is not the player's", coach.policy.inScramble)
    }

    /**
     * With the local verdicts off, the word on the engine's own move is the model's; when the
     * reply fails, the praise template stands in after one call and one retry
     * (`test_a_good_move_gets_one_sentence_from_the_model_with_the_template_to_fall_back_on` in
     * tests/test_coach.py; the parity audit, 2026-09-29).
     */
    @Test
    fun aSoundMovesWordFallsBackToThePraiseWhenTheModelFails() {
        val (coach, provider) = coach(listOf("", ""))
        coach.localVerdicts = false
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        mateInOnePlayed(coach)
        val out = messages(coach)
        assertTrue(out.map { it.text }.toString(), out.size == 1 && out[0].text in OPENERS.getValue("best").map { "$it." })
        assertEquals("one call, one retry, then the template", 2, provider.prompts.size)
    }

    /**
     * In blitz a sound move gets its two words from the board, with the local verdicts off as
     * well: no model call (`test_in_blitz_a_good_move_earns_two_words_without_the_model` in
     * tests/test_coach.py; the parity audit, 2026-09-29).
     */
    @Test
    fun inBlitzAGoodMoveEarnsTwoWordsWithoutTheModel() {
        val (coach, provider) = coach(listOf("never used"))
        coach.localVerdicts = false
        playToScholar(coach)
        coach.policy.timeControl = TimeControl.BLITZ
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        mateInOnePlayed(coach)
        val out = messages(coach)
        assertTrue(out.map { it.text }.toString(), out.size == 1 && out[0].text in OPENERS.getValue("best").map { "$it." })
        assertTrue("no model call for a good move in blitz", provider.prompts.isEmpty())
        assertTrue(coach.verdictWord("book") in OPENERS.getValue("book").map { "$it." })
    }

    // ---- the coach review, 2026-10-01: the game's lifecycle (desktop tests/test_coach.py) ----

    private fun offlineCoach(): Coach =
        Coach(OfflineProvider(), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))

    /** `_looked`: the engine's look at [board], [ucis] its lines best first, [threats] (UCI) too. */
    private fun looked(
        coach: Coach,
        board: Board,
        ply: Int,
        ucis: List<String>,
        ts: Double,
        threats: List<String> = emptyList(),
    ) {
        coach.onEvent(
            Analysis(
                gameId = "g", fen = board.fen(), ply = ply,
                lines = ucis.mapIndexed { i, u -> Line(moves = listOf(u), depth = 18, expectedScore = 0.5 - i / 10.0) },
                decisive = false, onlyMove = false, threats = threats,
            ),
            ts,
        )
    }

    /** `_judged`: the engine's verdict on the move at [ply]: its own best. */
    private fun judged(coach: Coach, ply: Int, uci: String, ts: Double) {
        coach.onEvent(
            MoveAssessment(
                gameId = "g", ply = ply, uci = uci, expectedLoss = 0.0, severity = Severity.NONE,
                best = listOf(uci), label = Label.BEST,
            ),
            ts,
        )
    }

    /** `_judged_at_scholar`: the user's [san] from SCHOLAR (ply 7) at [ts], its verdict a second later. */
    private fun judgedAtScholar(coach: Coach, san: String, loss: Double, severity: Severity, label: Label, ts: Double) {
        val board = Board(SCHOLAR)
        val move = board.parseSan(san)!!
        board.push(move)
        coach.onEvent(
            MovePlayed(gameId = "g", ply = 7, san = san, uci = move.uci(), fenBefore = SCHOLAR, fenAfter = board.fen(), byUser = true),
            ts,
        )
        coach.onEvent(
            MoveAssessment(
                gameId = "g", ply = 7, uci = move.uci(), expectedLoss = loss, severity = severity,
                best = listOf("h5f7"), label = label,
            ),
            ts + 1,
        )
    }

    /**
     * A reply still being written when the game moved on -- a takeback, a re-read, the next game
     * -- was recorded wherever the game was when it landed: under "Said here" of the position taken
     * back to or of the new game, where it made the player's next words a follow-up, or the answer
     * to its question. And a lesson held for its analysis was said in the next game.
     * `test_a_reply_about_a_moment_the_game_has_left_is_no_turn_of_the_one_now`.
     */
    @Test
    fun aReplyAboutAMomentTheGameHasLeftIsNoTurnOfTheOneNow() {
        for (change in listOf("takeback", "re-read", "new game")) {
            val (coach, _) = coach(listOf("[p=1.0] The queen mates on f7. What did you miss?"))
            playToScholar(coach)
            coach.onEvent(mateAnalysis(6), 7.0)
            judgedAtScholar(coach, "Bxf7+", 0.45, Severity.BLUNDER, Label.MISS, 10.0) // the lesson, asked
            when (change) {
                "takeback" -> coach.onEvent(Takeback("g", 1, SCHOLAR), 12.0)
                "re-read" -> coach.onEvent(Resync("g", SCHOLAR, "re-read"), 12.0)
                else -> start(coach)
            }
            val out = messages(coach) // the reply lands now
            // No reply is dropped (Tai, 2026-10-02): the lesson on a move the board no longer has
            // is shown, in its place, and marked late for the voice -- but it is no turn of the
            // moment now (the deep coach.py review, 2026-10-01).
            assertEquals(change, listOf("The queen mates on f7. What did you miss?"), out.map { it.text })
            assertTrue(change, coach.spoken.last().late)
            assertTrue(change, coach.conversation.entries.none { it.kind == "coach" })
            assertFalse(change, out[0].text in coach.conversation.render())
        }
        // A lesson held for the analysis of the position its move left is no exception: when the
        // moment ends first it is said in its own moment, shown, late for the voice, and never a
        // turn of the moment after. Dropped, the lesson on a mistake taken back at once was never
        // seen (Tai, 2026-10-02: no reply or lesson is dropped because its moment passed).
        val lesson = "The queen mates on f7 at once."
        for (change in listOf("takeback", "re-read", "new game")) {
            val coach = Coach(
                ScriptedProvider(listOf("[p=1.0] $lesson")),
                policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)),
                lessonWaitS = LESSON_WAIT_S,
            )
            playToScholar(coach)
            coach.onEvent(mateAnalysis(6), 7.0)
            judgedAtScholar(coach, "Bxf7+", 0.45, Severity.BLUNDER, Label.MISS, 10.0)
            assertEquals(
                "held for the analysis of the position it left",
                emptyList<CoachMessage>(),
                messages(coach),
            )
            when (change) {
                "takeback" -> coach.onEvent(Takeback("g", 1, SCHOLAR), 12.0)
                "re-read" -> coach.onEvent(Resync("g", SCHOLAR, "re-read"), 12.0)
                else -> start(coach)
            }
            val out = messages(coach)
            assertEquals(change, listOf(lesson), out.map { it.text })
            assertEquals(change, Trigger.POST_MISTAKE, out[0].trigger)
            assertTrue(change, coach.spoken.last().late)
            assertFalse(change, lesson in coach.conversation.render())
            assertEquals(change, emptyList<String>(), coach.saidSinceMove())
            coach.tick(12.0 + LESSON_WAIT_S)
            assertEquals("$change: said once", emptyList<CoachMessage>(), messages(coach))
        }
    }

    /**
     * What was said unasked at the plies taken back is not counted against the positions played
     * to there after: counted by ply, both hints said after 1.e4 left 1.d4, played in its place,
     * with no stuck hint at all.
     * `test_the_hints_said_at_a_withdrawn_ply_do_not_count_against_the_next_position_there`.
     */
    @Test
    fun theHintsSaidAtAWithdrawnPlyDoNotCountAgainstTheNextPositionThere() {
        val coach = offlineCoach()
        start(coach, Pieces.Side.BLACK)
        var board = Board()
        played(coach, board, 1, "e2e4", false, 1.0)
        looked(coach, board, 1, listOf("e7e5", "c7c5"), 1.5)
        assertEquals(
            listOf(DisclosureLevel.IDEA, DisclosureLevel.FOCUS),
            (stuck(coach) + stuck(coach)).map { it.level },
        )
        coach.onEvent(Takeback("g", 1, Board().fen()), 40.0)
        board = Board()
        played(coach, board, 1, "d2d4", false, 41.0)
        looked(coach, board, 1, listOf("d7d5", "g8f6"), 41.5)
        assertEquals(listOf(DisclosureLevel.IDEA), stuck(coach).map { it.level })
    }

    /**
     * The coach's "One moment -- still looking at this position." was recorded as its turn, so the
     * question it held, answered when the analysis landed, read as a follow-up: "what should I
     * have played?" was answered "You have already told them the verdict on it and the better move
     * ... do not repeat either". `test_the_coachs_one_moment_is_no_turn_of_the_conversation`.
     */
    @Test
    fun theCoachsOneMomentIsNoTurnOfTheConversation() {
        val (coach, provider) = coach(listOf("[p=1.0] Qxf7# was mate on the spot."))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        coach.quiet = true // no word after the move
        judgedAtScholar(coach, "Bxf7+", 0.45, Severity.BLUNDER, Label.MISS, 10.0)
        coach.quiet = false
        val board = Board(SCHOLAR)
        board.push(board.parseSan("Bxf7+")!!)
        played(coach, board, 8, "e8e7", false, 12.0)
        assertEquals(emptyList<CoachMessage>(), messages(coach))
        coach.ask("what should I have played?", 12.5)
        assertEquals("one moment", listOf(Trigger.USER_ASK), messages(coach).map { it.trigger })
        assertTrue(coach.conversation.entries.none { it.kind == "coach" })
        looked(coach, board, 8, listOf("h5e5", "d2d3"), 13.0)
        messages(coach)
        val task = directiveOf(given(provider))
        assertTrue(task, "A move they already played: its verdict is in the record" in task)
        assertTrue("# The position they faced, before Bxf7+" in harness(given(provider)))
        assertFalse(task, "do not repeat the verdict or the better move" in task)
    }

    /**
     * Right after the player's move, its position not yet read, "what are they threatening?" was
     * answered from the position before the move: "They threaten Qxf2#", the threat the move had
     * just stopped. It waits for the engine's look at the position on the board, where their
     * threat is their best move. `test_their_threat_after_the_players_move_waits_for_the_position_now`.
     */
    @Test
    fun theirThreatAfterThePlayersMoveWaitsForThePositionNow() {
        val coach = offlineCoach()
        coach.onEvent(GameStarted(gameId = "g", userSide = Pieces.Side.WHITE, startFen = QH4, kind = "position"), 0.0)
        val board = Board(QH4)
        looked(coach, board, 0, listOf("d1e2", "d1f3"), 0.5, threats = listOf("h4f2"))
        assertEquals(listOf("Qxf2#"), coach.positions[0]?.facts?.threats)
        played(coach, board, 1, "d1e2", true, 3.0) // Qe2 guards f2
        judged(coach, 1, "d1e2", 3.2)
        messages(coach)
        coach.ask("what are they threatening?", 3.5)
        assertTrue(messages(coach).none { "Qxf2" in it.text })
        looked(coach, board, 1, listOf("g8f6", "b8c6"), 4.0)
        val said = messages(coach).map { it.text }
        assertTrue(said.toString(), said.isNotEmpty() && said.last().startsWith("It's their move, and their best is"))
    }

    /**
     * Pressed between the player's move and its verdict, "hint" got "Give me a moment; I am still
     * reading this position." with nothing queued -- the promise never kept -- and the rung was
     * climbed at the position the player had moved from: the combination's next stuck hint
     * skipped the idea. `test_a_hint_before_the_verdict_is_answered_and_climbs_no_rung_of_the_position_left`.
     */
    @Test
    fun aHintBeforeTheVerdictIsAnsweredAndClimbsNoRungOfThePositionLeft() {
        val coach = offlineCoach()
        start(coach, Pieces.Side.BLACK)
        val board = Board()
        played(coach, board, 1, "e2e4", false, 1.0)
        looked(coach, board, 1, listOf("e7e5", "c7c5"), 1.5)
        played(coach, board, 2, "e7e5", true, 3.0) // the engine's best
        coach.ask("hint", 3.05)
        assertEquals("a moment", listOf(Trigger.USER_ASK), messages(coach).map { it.trigger })
        judged(coach, 2, "e7e5", 3.1)
        messages(coach)
        looked(coach, board, 2, listOf("g1f3", "b1c3"), 3.6)
        val kept = messages(coach).filter { it.trigger == Trigger.USER_ASK }.map { it.text }
        assertTrue(kept.toString(), kept.size == 1 && "moment" !in kept[0])
        assertEquals(DisclosureLevel.SILENT, coach.positions[1]?.disclosure?.hinted)
        played(coach, board, 3, "g1f3", false, 5.0)
        looked(coach, board, 3, listOf("b8c6", "g8f6"), 5.5)
        assertEquals(listOf(DisclosureLevel.IDEA), stuck(coach).map { it.level })
    }

    /**
     * The coach's "One moment" was a promise it did not keep when the board changed under it: a
     * takeback neither answered nor dropped the question, a re-read and a new game dropped it, and
     * the game's end left it waiting. It is answered at the position the player faces once the
     * coach has read it, and at the game's end at once.
     * `test_a_question_held_for_the_engine_is_answered_whatever_the_board_does`.
     */
    @Test
    fun aQuestionHeldForTheEngineIsAnsweredWhateverTheBoardDoes() {
        val four = Board()
        for (san in listOf("e4", "e5", "Bc4", "Nc6")) four.push(four.parseSan(san)!!)
        val other = Board(QH4)
        for (change in listOf("takeback", "re-read", "new game", "game over")) {
            val coach = offlineCoach()
            playToScholar(coach)
            looked(coach, four, 4, listOf("d1h5", "g1f3"), 6.5)
            coach.ask("hint", 7.0) // the position now not read yet
            assertEquals("$change: a moment", listOf(Trigger.USER_ASK), messages(coach).map { it.trigger })
            when (change) {
                "takeback" -> coach.onEvent(Takeback("g", 2, four.fen()), 8.0)
                "re-read" -> {
                    coach.onEvent(Resync("g", QH4, "re-read"), 8.0)
                    looked(coach, other, 0, listOf("d1e2", "d1f3"), 8.5)
                }
                "new game" -> {
                    coach.onEvent(GameStarted(gameId = "g", userSide = Pieces.Side.WHITE, startFen = QH4, kind = "position"), 8.0)
                    looked(coach, other, 0, listOf("d1e2", "d1f3"), 8.5)
                }
                else -> coach.onEvent(GameEnded(gameId = "g", result = "1-0", reason = "resigned"), 8.0)
            }
            val answered = messages(coach)
            assertEquals(change, listOf(Trigger.USER_ASK), answered.map { it.trigger })
            assertFalse(change, "moment" in answered[0].text)
        }
    }

    /**
     * An inaccuracy got the word for a sound move -- "Fine, but Qxf7# was a little tighter", and
     * the model was told "an inaccuracy: sound, a little behind Qxf7#" -- while asked "was that a
     * mistake?" right after, the coach answered it as the costly move it was.
     * `test_an_inaccuracy_is_never_called_sound`.
     */
    @Test
    fun anInaccuracyIsNeverCalledSound() {
        val fine = FINE_TIGHTER.map { it.substringBefore("{best}") }
        for (offline in listOf(true, false)) {
            val (coach, provider) = if (offline) offlineCoach() to null else coach(listOf("", ""))
            playToScholar(coach)
            coach.onEvent(mateAnalysis(6), 7.0)
            judgedAtScholar(coach, "Bxf7+", 0.07, Severity.INACCURACY, Label.INACCURACY, 10.0)
            val out = messages(coach)
            assertTrue(out.toString(), out.size == 1 && out[0].trigger == Trigger.POST_MOVE)
            assertTrue(out[0].text, "Qxf7" in out[0].text && fine.none { out[0].text.startsWith(it) })
            if (provider != null) {
                val task = directiveOf(given(provider, 0))
                assertTrue(task, "which was an inaccuracy" in task && "sound" !in task)
            }
        }
    }

    /**
     * A hint asked at 15 s was followed by the focus unasked at 16 s: the stuck spell ran from the
     * position, not from the answer. The next hint unasked waits a full spell from the answer, as
     * from one said unasked. `test_a_question_starts_the_stuck_spell_again`.
     */
    @Test
    fun aQuestionStartsTheStuckSpellAgain() {
        val coach = offlineCoach()
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        coach.ask("hint", 15.0)
        assertEquals(listOf(DisclosureLevel.IDEA), messages(coach).map { it.level })
        coach.tick(16.0)
        assertEquals("a second after the answer", emptyList<CoachMessage>(), messages(coach))
        assertEquals(listOf(DisclosureLevel.FOCUS), stuck(coach).map { it.level })
    }

    /**
     * The coach outlives a Stop, and kept the position it last read: the first ticks after the
     * next Start gave stuck hints about the last session's board (the game-state review,
     * 2026-10-01). Paused at the end of a session, it is quiet until the board is read again.
     * `test_after_a_stop_the_coach_says_nothing_unasked_until_the_board_is_read_again`.
     */
    @Test
    fun afterAStopTheCoachSaysNothingUnaskedUntilTheBoardIsReadAgain() {
        val coach = offlineCoach()
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        coach.pause(8.0) // GameWatcher.close
        for (ts in listOf(100.0, 1_000.0, 100_000.0)) coach.tick(ts)
        assertEquals("the last session's position", emptyList<CoachMessage>(), messages(coach))
        playToScholar(coach) // the board read again
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        assertEquals(listOf(Trigger.STUCK), stuck(coach).map { it.trigger })
    }

    /**
     * The analysis of the position the player's move left can land after the opponent's reply:
     * its note went under "Said here" of the position the player now faces, the past position's
     * standing beside the one now's, and without its engine lines, filtered by a title they no
     * longer have. It is the record's, under the move it is about, lines and all.
     * `test_conversation.py::test_a_late_engine_note_is_the_records_and_has_its_lines`.
     */
    @Test
    fun aLateEngineNoteIsTheRecordsAndHasItsLines() {
        val (coach, provider) = coach(emptyList())
        playToScholar(coach)
        coach.onEvent(twoLines(Board(SCHOLAR), 6), 7.0)
        coach.ask("what is my plan?", 8.0) // the model has seen the record: its lines are frozen
        messages(coach)
        val afterQh5 = Board()
        for (san in listOf("e4", "e5", "Bc4", "Nc6", "Qh5")) afterQh5.push(afterQh5.parseSan(san)!!)
        coach.onEvent(twoLines(afterQh5, 5), 9.0) // 3. Qh5's position, read late
        coach.ask("and what should I watch out for?", 10.0)
        messages(coach)
        val text = harness(given(provider))
        val record = text.substringBefore("# Position now\n")
        val now = text.substringAfter("# Position now\n")
        assertTrue(record, "Engine note on the position after 3. Qh5" in record)
        val note = record.substringAfter("Engine note on the position after 3. Qh5")
        assertTrue(note, "Engine lines (the opponent's move" in note && "Standing:" in note)
        assertFalse(now, "Engine note" in now)
    }

    /**
     * The position's message frozen before its analysis came: the whole (provisional) and then
     * the final word with the same lines -- the model read neither, so the final word gets its
     * note, and a second final word with those lines none (the audit of 2026-10-05, round 2:
     * the rule compared against the record's last analysis, not what the model read, and the
     * model kept "not in yet" for the rest of the game).
     * `test_conversation.py::test_lines_the_model_never_read_get_their_note_once`.
     */
    @Test
    fun linesTheModelNeverReadGetTheirNoteOnce() {
        val (coach, provider) = coach(emptyList())
        playToScholar(coach)
        coach.onEvent(twoLines(Board(SCHOLAR), 6), 6.5)
        coach.ask("what is my plan?", 7.0) // 3. Qh5's message frozen with no analysis
        messages(coach)
        val afterQh5 = Board()
        for (san in listOf("e4", "e5", "Bc4", "Nc6", "Qh5")) afterQh5.push(afterQh5.parseSan(san)!!)
        val whole = twoLines(afterQh5, 5)
        val provisional = Analysis(
            gameId = whole.gameId, fen = whole.fen, ply = whole.ply, lines = whole.lines,
            decisive = false, onlyMove = false, provisional = true,
        )
        coach.onEvent(provisional, 8.0)
        coach.onEvent(whole, 9.0)
        coach.onEvent(whole, 9.5)
        coach.ask("and what should I watch out for?", 10.0)
        messages(coach)
        val record = harness(given(provider)).substringBefore("# Position now\n")
        assertEquals(record, 1, Regex("Engine note on the position after 3\\. Qh5").findAll(record).count())
    }

    /**
     * The analyser tells a position whole (provisional), then final once its side searches are
     * in, the lines the same: the model read them whole, and a note repeating them on every move
     * the game has left only grew the prompt (the audit of 2026-10-05).
     * `test_conversation.py::test_a_final_word_that_changes_no_line_the_model_read_adds_no_note`.
     */
    @Test
    fun aFinalWordThatChangesNoLineTheModelReadAddsNoNote() {
        val (coach, provider) = coach(emptyList())
        playToScholar(coach)
        coach.onEvent(twoLines(Board(SCHOLAR), 6), 6.5)
        val afterQh5 = Board()
        for (san in listOf("e4", "e5", "Bc4", "Nc6", "Qh5")) afterQh5.push(afterQh5.parseSan(san)!!)
        val whole = twoLines(afterQh5, 5)
        val provisional = Analysis(
            gameId = whole.gameId, fen = whole.fen, ply = whole.ply, lines = whole.lines,
            decisive = false, onlyMove = false, provisional = true,
        )
        coach.onEvent(provisional, 7.0)
        coach.ask("what is my plan?", 8.0) // the model has seen the record: its lines are frozen
        messages(coach)
        coach.onEvent(whole, 9.0) // final: the same lines
        coach.ask("and what should I watch out for?", 10.0)
        messages(coach)
        val record = harness(given(provider)).substringBefore("# Position now\n")
        assertFalse(record, "Engine note on the position after 3. Qh5" in record)
    }

    // --- the deep prompt review (desktop, 2026-10-01): one test a finding, as the desktop's ---

    /** The scholar's board with the player Black: the opponent to move, Qxf7# on the board. */
    private fun scholarAsBlackAt(replies: List<String>): Pair<Coach, ScriptedProvider> {
        val (coach, provider) = coach(replies)
        start(coach, Pieces.Side.BLACK)
        val board = Board()
        for ((i, san) in listOf("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6").withIndex()) {
            val ply = i + 1
            val move = board.parseSan(san)!!
            val before = board.fen()
            board.push(move)
            coach.onEvent(
                MovePlayed(
                    gameId = "g", ply = ply, san = san, uci = move.uci(), fenBefore = before,
                    fenAfter = board.fen(), byUser = ply % 2 == 0,
                ),
                ply.toDouble(),
            )
        }
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        messages(coach)
        return coach to provider
    }

    /**
     * On the opponent's move the engine's best move and line are the opponent's: "what should I
     * play?" told the model "tell them plainly it is Qxf7#" (finding 1;
     * `test_what_to_play_on_their_turn_never_gives_their_move_as_the_players`).
     */
    @Test
    fun whatToPlayOnTheirTurnNeverGivesTheirMoveAsThePlayers() {
        for (said in listOf("what should I play?", "what is the best move?", "show me the line")) {
            val (coach, provider) = scholarAsBlackAt(listOf("", "", ""))
            for (ts in 0 until 3) coach.ask(said, 9.0 + ts) // climb to the move rung
            val texts = messages(coach).map { it.text }
            for (i in provider.prompts.indices) {
                val task = directiveOf(given(provider, i))
                assertTrue("$said: $task", "it is the opponent's move" in task)
                assertTrue(task, "tell them plainly it is" !in task)
                assertTrue(task, "They asked to see the line: give it" !in task)
            }
            assertTrue(texts.toString(), texts.isNotEmpty() && texts[0].startsWith("It's their move"))
            if (said == "show me the line") {
                // The line asked for is the line from their move on, each move named whose it is
                // (Tai, 2026-10-03: their next move alone was said).
                assertEquals("It's their move, and the line runs their Qxf7#, checkmate.", texts[0])
                assertTrue("the line from here may be said" in directiveOf(given(provider, 0)))
            }
        }
    }

    /**
     * A mistake's verdict landing after the opponent's reply, with the player to move: no question
     * that sends them to the board (finding 7). What the move allowed is still said, as in any
     * lesson: it happened on the board, and nothing of the lesson is held back (Tai, 2026-10-02).
     * `test_a_late_lesson_asks_nothing_while_they_think`.
     */
    @Test
    fun aLateLessonAsksNothingWhileTheyThink() {
        val (coach, provider) = coach(listOf("[p=1.0] Qxf7 was mate at once."))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        messages(coach)
        val board = Board(SCHOLAR)
        val plays = ArrayList<Move>()
        for ((ply, san) in listOf(7 to "d3", 8 to "Nxh5")) {
            val before = board.fen()
            val move = board.parseSan(san)!!
            board.push(move)
            plays.add(move)
            coach.onEvent(
                MovePlayed(
                    gameId = "g", ply = ply, san = san, uci = move.uci(), fenBefore = before,
                    fenAfter = board.fen(), byUser = ply == 7,
                ),
                10.0 + ply,
            )
            if (ply == 7) {
                // The engine's look at the position d3 left: the knight takes the queen.
                val reply = Board(board.fen())
                val line = ArrayList<String>()
                for (step in listOf("Nxh5", "Nf3", "d6")) {
                    val found = reply.parseSan(step)!!
                    line.add(found.uci())
                    reply.push(found)
                }
                coach.onEvent(
                    Analysis(
                        gameId = "g", fen = board.fen(), ply = 7,
                        lines = listOf(Line(moves = line, depth = 12, expectedScore = 0.75)),
                        decisive = false, onlyMove = false,
                    ),
                    17.5,
                )
            }
        }
        coach.onEvent(
            MoveAssessment(
                gameId = "g", ply = 7, uci = plays[0].uci(), expectedLoss = 0.45,
                severity = Severity.BLUNDER, best = listOf("h5f7"), label = Label.MISS,
            ),
            20.0,
        )
        messages(coach)
        val lesson = provider.prompts.indices.map { directiveOf(given(provider, it)) }
            .first { "The player just played d3" in it }
        assertTrue(lesson, "no question" in lesson)
        assertTrue(lesson, "one question that sends them to find" !in lesson)
        assertTrue(lesson, "what their move missed or allowed" in lesson)
        assertTrue(lesson, "What d3 allowed, from the engine's line: after d3, Nxh5 wins your queen" in lesson)
    }

    /**
     * A fourth "hint" was told "The next hint is the answer" after the move was given; castling's
     * second hint says what the king goes after (findings 15 and 9).
     */
    @Test
    fun aHintAfterTheMoveDoesNotPromiseItAndCastlingsSecondHintIsTheKingsSafety() {
        val (coach, provider) = coach(
            listOf("[p=1.0] Look at f7.", "[p=1.0] Your queen and f7.", "[p=1.0] Qxf7# is mate.", ""),
        )
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        messages(coach)
        for (ts in 0 until 4) {
            coach.ask("hint", 8.0 + ts)
            messages(coach)
        }
        val tasks = provider.prompts.indices.map { directiveOf(given(provider, it)) }
        // With the AI on, the ladder says what a hint would be, from what the answers said; once
        // the move was said, the guideline says to give it again, with why it beats worse ones.
        assertTrue(tasks[2], "has been given already" !in tasks[2])
        assertTrue(tasks[3], "The move has been given already, in your turns above: it is Qxf7#" in tasks[3])
        assertTrue(tasks[3], "why it is the move to play instead of something that looks as good" in tasks[3])
        assertTrue(tasks[3], "Bxf7+ passes up the mate" in tasks[3])
        assertEquals(DisclosureLevel.MOVE, coach.positions[6]!!.disclosure.level)

        val (castling, model) = coach(listOf("[p=1.0] One.", "[p=1.0] Two."))
        start(castling)
        val board = Board()
        for ((i, san) in listOf("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5").withIndex()) {
            val ply = i + 1
            val move = board.parseSan(san)!!
            val before = board.fen()
            board.push(move)
            castling.onEvent(
                MovePlayed(
                    gameId = "g", ply = ply, san = san, uci = move.uci(), fenBefore = before,
                    fenAfter = board.fen(), byUser = ply % 2 == 1,
                ),
                ply.toDouble(),
            )
        }
        castling.onEvent(
            Analysis(
                gameId = "g", fen = board.fen(), ply = 6,
                lines = listOf(Line(moves = listOf("e1g1", "g8f6"), depth = 12, expectedScore = 0.55)),
                decisive = false, onlyMove = false,
            ),
            7.0,
        )
        messages(castling)
        castling.ask("hint", 8.0)
        messages(castling)
        castling.ask("hint", 9.0)
        messages(castling)
        val task = directive(model)
        assertTrue(task, "Here the move is castling" in task)
    }

    /**
     * On the opponent's move a named move is never examined: the directive does not say "when the
     * engine does not rate their move, say it is not the strongest" (finding 19).
     */
    @Test
    fun aMoveNamedOnTheirTurnIsNotCalledUnrated() {
        val (coach, provider) = scholarAsBlackAt(listOf("[p=1.0] Fine.", "[p=1.0] Fine."))
        coach.ask("what about Qxf7?", 8.0)
        messages(coach)
        val task = directive(provider)
        assertTrue(task, "say it is not the strongest" !in task)
        assertTrue(task, "rate one only from the engine's word on it" in task)
    }

    /**
     * Both drafts refused, "is Bxf7+ good?" got the mate-in-one hint; the fallback is the engine's
     * word on Bxf7+ (finding 6). The question's engine work is handed to the coach's spawner, and
     * runs when the test runs it: none of it on the caller's thread (finding 13).
     */
    @Test
    fun aRefusedAnswerAboutAMoveFallsBackToTheEnginesWordOnItOffTheCallersThread() {
        val bad = "[p=0.6] Your queen on e8 is in danger here. [p=0.4] Their bishop on h5 is loose."
        val provider = ScriptedProvider(listOf(bad, bad))
        val coach = Coach(
            provider,
            policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)),
            draftCheck = DraftCheck { text, board, after, us ->
                checkDraft(text, board, after, us).map { "${it.text}: ${it.reason}" }
            },
        )
        val searched = ArrayList<String>()
        coach.engine = EngineLike { board, depth ->
            searched.add(board.fen())
            firstLegal.analyse(board, depth)
        }
        val spawned = ArrayList<Pair<() -> Unit, String>>()
        coach.spawn = { work, name -> spawned.add(work to name) }
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        messages(coach)
        searched.clear()
        coach.ask("is Bxf7+ good?", 8.0)
        assertTrue("no search on the caller's thread", searched.isEmpty())
        assertEquals(listOf("sidecoach-ask"), spawned.map { it.second })
        spawned.removeAt(0).first()
        assertTrue("the move named was examined", searched.isNotEmpty())
        val texts = messages(coach).map { it.text }
        assertTrue(texts.toString(), texts.isNotEmpty() && "after 4. Bxf7+" in texts.last())
        assertTrue(texts.toString(), "Qxf7" !in texts.last())
    }

    /**
     * A model calling the engine writes a preamble in that round; the reader sliced the answer
     * round with its spoken length ("ith the queen is mate"). The preamble is not said, and the
     * answer is said from its first word (finding 2).
     */
    @Test
    fun aToolRoundsPreambleIsNotSaidAndTheAnswerIsReadFromItsStart() {
        class StreamingToolModel(private val preamble: String, private val answer: String) :
            ToolsProvider, com.sidecoach.llm.StreamingProvider {
            override val name = "stream-tools"
            private var round = 0
            override fun healthcheck(): Pair<Boolean, String> = true to "x"
            override fun chat(
                system: String, messages: List<ChatMessage>, maxTokens: Int, temperature: Double, thinking: Boolean,
            ): Completion = Completion(text = answer)
            override fun chatStream(
                system: String, messages: List<ChatMessage>, maxTokens: Int, temperature: Double,
                thinking: Boolean, onDelta: StreamCallback,
            ): Completion = go(onDelta, false)
            override fun chatTools(
                system: String, messages: List<ChatMessage>, tools: JSONArray, turns: List<ToolRound>,
                maxTokens: Int, temperature: Double, toolChoice: String?, onDelta: StreamCallback?,
            ): Completion = go(onDelta, true)
            private fun go(onDelta: StreamCallback?, tools: Boolean): Completion {
                round += 1
                val calls = if (round == 1 && tools) {
                    listOf(ToolCall(id = "c1", name = "analyse", arguments = JSONObject().put("moves", JSONArray(listOf("Qxf7#")))))
                } else {
                    emptyList()
                }
                var text = if (calls.isNotEmpty()) preamble else answer
                if (onDelta != null) {
                    val sent = StringBuilder()
                    var i = 0
                    while (i < text.length) {
                        val piece = text.substring(i, minOf(text.length, i + 7))
                        sent.append(piece)
                        if (!onDelta(piece)) break
                        i += 7
                    }
                    text = sent.toString()
                }
                return Completion(text = text, toolCalls = calls)
            }
        }
        val answer = "[p=0.6] Taking on f7 with the queen is mate: the king has no square left. " +
            "[p=0.4] The queen takes on f7 and it is mate at once."
        for (preamble in listOf("Good question. Let me check that with the engine first. ", "Let me ask the engine about Qxf7. ")) {
            val coach = Coach(StreamingToolModel(preamble, answer), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
            coach.engine = firstLegal
            coach.syncClaims = true
            val heard = ArrayList<String>()
            coach.streamSink = { _, _, _, _ -> { s -> heard.add(s); true } }
            playToScholar(coach)
            coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
            messages(coach)
            heard.clear()
            coach.ask("what happens if I take on f7 with the queen?", 8.0)
            val texts = messages(coach).map { it.text }
            assertTrue(heard.toString(), heard.isNotEmpty() && heard[0].startsWith("Taking on f7"))
            assertTrue(heard.toString(), heard.none { "engine" in it })
            assertTrue(texts.toString(), texts.isNotEmpty() && texts.last().startsWith("Taking on f7"))
        }
    }

    /**
     * Before any move, what was said hung under "Game so far" with no position; the past
     * position's free squares are said of that position; "you" as an object takes no verb; the
     * verdict-word check reads an unnamed clause as the move's only where it calls it so; an
     * engine-tool call it cannot read is told so (findings 18, 12, 14, 5, 16).
     */
    @Test
    fun theSmallerWordingFixesHoldAsOnTheDesktop() {
        assertEquals(
            "it gives the player a pawn instead; it hits the player twice",
            Person.thirdPerson("it gives you a pawn instead; it hits you twice"),
        )
        assertEquals(
            "the player answers the opponent instead of playing the player's own moves",
            Person.thirdPerson("they answer you instead of playing their own moves", side = "opponent"),
        )
        assertTrue(MOVE_GIVEN.contains("{move}") && FOCUS_KING.startsWith("Here the move is castling"))
        assertTrue("of the opponent's the sheet names" in IDEA_LEVEL_WORDS)
        assertTrue("which of the player's pieces makes it" in IDEA_LEVEL_WORDS)

        val tools = EngineTools(Board(SCHOLAR), null, firstLegal, true)
        val unread = tools.run("analyse", JSONObject().put("_raw", "moves: Kg2"))
        assertTrue(unread, unread.startsWith("The arguments were not a JSON object ('moves: Kg2')"))
        val start = tools.run("analyse", JSONObject().put("moves", JSONArray(listOf("Bxf7+"))).put("start", "before"))
        assertTrue(start, start.startsWith("No start named 'before'"))
    }

    companion object {
        // White to move; Black's queen and bishop threaten Qxf2#.
        private const val QH4 = "rnb1k1nr/pppp1ppp/8/2b1p3/2B1P2q/2N5/PPPP1PPP/R1BQK1NR w KQkq - 6 4"
        // 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6??: white to move, Qxf7 mates.
        private const val SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"

        // The puzzle of Tai's desktop, 2026-09-24: Black to move, 1...Qd1+ 2.Rxd1 Rxd1+ 3.Qe1 Rxe1#.
        private const val QD1 = "2kr2nr/2p5/2p1p1pp/Q3Pp2/3R4/P3P3/1Pq2PPP/R1B3K1 b - - 1 1"
        private val QD1_LINE = listOf("c2d1", "d4d1", "d8d1", "a5e1", "d1e1")

        /** The leads that announced a rule, gone for good (desktop f983cd0). */
        private val ANNOUNCED = Regex(
            "(keep this one|the lesson to keep|worth remembering|a rule to take with you|" +
                "the thing to remember|remember this):",
            RegexOption.IGNORE_CASE,
        )

        private fun sentenceOf(rule: String): String = rule[0].uppercaseChar() + rule.substring(1).trimEnd('.') + "."
    }
}
