package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.chess.Board
import com.sidecoach.events.Analysis
import com.sidecoach.events.CoachMessage
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.GameStarted
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.Severity
import com.sidecoach.events.Trigger
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.Completion
import com.sidecoach.llm.LLMProvider
import com.sidecoach.vision.Pieces
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The coach's ties to the game, decided in code: the desktop's `tests/test_links.py` (desktop
 * 134b2ea, the systematic round, 2026-09-25). A lesson links only to the earlier moments the
 * coach lists -- the player's moves played with the very same idea -- and names no other; the
 * material is said whole; "why was that bad?" says what the move allowed; and a figure where
 * none was asked for is a fault the directive states.
 */
@RunWith(AndroidJUnit4::class)
class LinksTest {

    private class ScriptedProvider(replies: List<String>) : LLMProvider {
        override val name = "scripted"
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
        return Coach(provider, policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0))) to provider
    }

    private fun messages(coach: Coach): List<CoachMessage> {
        coach.speaker.runPending()
        return coach.poll()
    }

    private fun start(coach: Coach, fen: String = Board().fen()) {
        coach.onEvent(GameStarted(gameId = "g", userSide = Pieces.Side.WHITE, startFen = fen), 0.0)
    }

    private fun given(provider: ScriptedProvider, index: Int = -1): List<ChatMessage> =
        synchronized(provider.prompts) {
            provider.prompts[if (index < 0) provider.prompts.size + index else index].second
        }

    private fun harness(messages: List<ChatMessage>): String {
        assertTrue(messages.size == 1 && messages[0].role == "user")
        return messages[0].content
    }

    private fun directive(messages: List<ChatMessage>): String = harness(messages).substringAfterLast("# Task\n")

    /** `_move`: [san] played on [board] as the event the pipeline sends; the move's UCI. */
    private fun move(coach: Coach, board: Board, ply: Int, san: String, byUser: Boolean, ts: Double): String {
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
                byUser = byUser,
            ),
            ts,
        )
        return move.uci()
    }

    private fun mateAnalysis(ply: Int, decisive: Boolean = true) = Analysis(
        gameId = "g",
        fen = SCHOLAR,
        ply = ply,
        lines = listOf(
            Line(moves = listOf("h5f7"), depth = 12, expectedScore = 1.0, mate = 1),
            Line(moves = listOf("c4f7"), depth = 12, expectedScore = 0.55),
        ),
        decisive = decisive,
        onlyMove = true,
        tags = listOf(com.sidecoach.events.MotifTag(name = "mateIn1", squares = listOf("f7", "e8"), move = "h5f7")),
    )

    /**
     * The scholar's game, with 2. Bc4 judged best from a position read as the same idea (a
     * mating attack) and the coach's word after it.
     */
    private fun scholarWithAnEarlierMoment(coach: Coach) {
        start(coach)
        val board = Board()
        for ((i, san) in listOf("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6").withIndex()) {
            val ply = i + 1
            val uci = move(coach, board, ply, san, byUser = ply % 2 == 1, ts = ply.toDouble())
            if (san == "Bc4") {
                coach.onEvent(
                    MoveAssessment(
                        gameId = "g",
                        ply = ply,
                        uci = uci,
                        expectedLoss = 0.0,
                        severity = Severity.NONE,
                        best = listOf(uci),
                        label = Label.BEST,
                    ),
                    ply.toDouble(),
                )
                coach.conversation.coachSaid("Nice -- Bc4 eyes f7.")
            }
        }
        val faced = coach.conversation.records[1] // after 1... e5: the position 2. Bc4 was played from
        faced.facts = PositionFacts(
            fen = faced.move.fenAfter,
            sideToMove = "white",
            phase = "opening",
            theme = "a mating attack",
        )
        assertEquals(SCHOLAR, coach.board.fen())
    }

    private fun missTheMate(coach: Coach, ts: Double = 10.0) {
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
            ts,
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
            ts + 1,
        )
    }

    @Test
    fun aLessonTiesOnlyToTheMomentsTheCoachLists() {
        val (coach, provider) = coach(
            listOf(
                "Their king on e8 is nearly out of squares.",
                // Names an earlier move the list does not give, with its number.
                "Qxf7# was mate on the spot; as after 3. Qh5, look for the check first. Which " +
                    "check ends it?",
                "Qxf7# was mate on the spot: as with Bc4, the target was f7. Which check ends it?",
            )
        )
        scholarWithAnEarlierMoment(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        messages(coach)
        missTheMate(coach)
        val out = messages(coach)
        assertTrue(out.size == 1 && out[0].trigger == Trigger.POST_MISTAKE)
        val content = harness(given(provider, 1))
        val section = content.substringAfter("# Earlier in this game, the same idea: a mating attack\n")
        // The coach's words on a line of their own, as everywhere in the record.
        assertTrue(
            section,
            section.startsWith(
                "- At Bc4 (the player's move): best -- the player found it.\n" +
                    "    YOU (the coach) said: \"Nice -- Bc4 eyes f7.\""
            ),
        )
        val directive = directive(given(provider, 1))
        assertTrue(directive, "Under 'Earlier in this game'" in directive && TIE_TO_THE_GAME !in directive)
        assertTrue("no example move for the model to copy", "(as with Bg4)" !in directive)
        // The first reply named 3. Qh5, an earlier move not listed: told so, and redone. Its move
        // number is form, and only noted (Tai, 2026-09-28; desktop 3ff672b).
        val retry = given(provider, 2).last().content
        assertTrue(retry, "names Qh5, an earlier moment not listed" in retry)
        assertTrue(retry, "a move with its number" !in retry)
        assertTrue(out[0].text, out[0].text.startsWith("Qxf7# was mate on the spot: as with Bc4"))
        assertEquals(emptyList<String>(), coach.spoken.last().problems)
    }

    @Test
    fun withNoEarlierMomentOfTheSameIdeaTheLessonSpeaksOfNone() {
        val (coach, provider) = coach(
            listOf("Their king on e8 is nearly out of squares.", "Qxf7# was mate. Which check ends it?")
        )
        start(coach)
        val board = Board()
        for ((i, san) in listOf("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6").withIndex()) {
            move(coach, board, i + 1, san, byUser = (i + 1) % 2 == 1, ts = (i + 1).toDouble())
        }
        coach.onEvent(mateAnalysis(6), 7.0)
        messages(coach)
        missTheMate(coach)
        messages(coach)
        assertTrue("# Earlier in this game" !in harness(given(provider, 1)))
        assertTrue(NO_EARLIER in directive(given(provider, 1)))
    }

    /**
     * 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4: every position the player faced read as the same
     * plan, the first move book, 2. Nf3 a mistake with the coach's word after it.
     */
    private fun conversationGame(): Conversation {
        val conversation = Conversation()
        conversation.newGame(GameStarted(gameId = "g", userSide = Pieces.Side.WHITE, startFen = Board().fen()))
        val board = Board()
        val judged = mapOf(
            1 to listOf(Label.BOOK, Severity.NONE, 0.0, null),
            3 to listOf(Label.MISTAKE, Severity.MISTAKE, 0.12, "b1c3"),
            5 to listOf(Label.BEST, Severity.NONE, 0.0, null),
            7 to listOf(Label.GOOD, Severity.NONE, 0.02, "b5c6"),
        )
        for ((i, san) in listOf("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4").withIndex()) {
            val ply = i + 1
            val move = board.parseSan(san)!!
            val before = board.fen()
            board.push(move)
            val record = conversation.move(
                MovePlayed(
                    gameId = "g",
                    ply = ply,
                    san = san,
                    uci = move.uci(),
                    fenBefore = before,
                    fenAfter = board.fen(),
                    byUser = ply % 2 == 1,
                )
            )
            record.facts = PositionFacts(
                fen = board.fen(), sideToMove = "w", phase = "opening", plan = "finishing your development",
            )
            val one = judged[ply]
            if (one != null) {
                conversation.assess(
                    MoveAssessment(
                        gameId = "g",
                        ply = ply,
                        uci = move.uci(),
                        expectedLoss = one[2] as Double,
                        severity = one[1] as Severity,
                        best = listOf((one[3] as String?) ?: move.uci()),
                        label = one[0] as Label,
                    )
                )
            }
            if (san == "Nf3") conversation.coachSaid("That one cost you: Nc3 was better.")
        }
        return conversation
    }

    @Test
    fun theEarlierMomentsAreTheLatestOfTheSameIdeaAndNeverBook() {
        val conversation = conversationGame()
        assertEquals(
            listOf(
                "- At Bb5 (the player's move): best -- the player found it.",
                "- At Ba4 (the player's move): good; Bxc6 was better.",
            ),
            conversation.sameIdea("finishing your development", beforePly = 9),
        )
        assertEquals(
            "1. e4 was book, from no faced position; 4. Ba4 is the move taught",
            listOf(
                "- At Nf3 (the player's move): a mistake; Nc3 was better.\n" +
                    "    YOU (the coach) said: \"That one cost you: Nc3 was better.\"",
                "- At Bb5 (the player's move): best -- the player found it.",
            ),
            conversation.sameIdea("finishing your development", beforePly = 7, limit = 3),
        )
        assertEquals(emptyList<String>(), conversation.sameIdea("a fork", beforePly = 9))
        // The moves a reply may not name: pieces, captures and castling, never a pawn's step.
        assertEquals(setOf("Nf3", "Nc6", "Bb5", "Nc3"), conversation.movesBefore(6))
    }

    @Test
    fun anUnlistedEarlierMoveAndAMoveNumberAreFaults() {
        val unlisted = setOf("Bc4", "Qh5")
        assertEquals(emptyList<String>(), unlistedLinks("Keep your bishop on c4 in mind.", unlisted))
        assertEquals(
            listOf("names Bc4, an earlier moment not listed: tie only to the moments listed"),
            unlistedLinks("As with Bc4, the target was f7.", unlisted),
        )
        assertEquals(
            listOf("a move with its number: name moves bare"),
            unlistedLinks("The same lesson as after 9...a6.", emptySet()),
        )
        assertEquals(emptyList<String>(), unlistedLinks("It cost you 8% of the game. Rc1 was better.", unlisted))
    }

    /** 1. e4 d5 2. Nc3 dxe4: the player, White, to move, a pawn down, with Nxe4 to come. */
    private fun captureGame(coach: Coach) {
        start(coach)
        val board = Board()
        for ((i, san) in listOf("e4", "d5", "Nc3", "dxe4").withIndex()) {
            move(coach, board, i + 1, san, byUser = (i + 1) % 2 == 1, ts = (i + 1).toDouble())
        }
        coach.onEvent(
            Analysis(
                gameId = "g",
                fen = board.fen(),
                ply = 4,
                lines = listOf(Line(moves = listOf("c3e4", "g8f6"), depth = 12, expectedScore = 0.5)),
                decisive = false,
                onlyMove = false,
            ),
            5.0,
        )
    }

    @Test
    fun theMaterialIsSaidWholeNowAndAfterTheRecapture() {
        val (coach, _) = coach(emptyList())
        captureGame(coach)
        val position = coach.positions.getValue(4)
        val (directive, template) = coach.instruction(
            Trigger.USER_ASK, position, DisclosureLevel.IDEA, "am I up material?", emptyList(), null, null,
        )
        // Before the move rung both halves, the recapture unnamed: which piece takes back is the
        // question, and "once you take back" does not answer it -- the material now alone was the
        // vaguer answer (Tai, 2026-09-25: "a vague coach is worse than no coach at all"; desktop
        // ac6b39e).
        assertEquals("Material: you are down a pawn -- and once you take back, it is level.", template)
        assertTrue(directive, "in these terms: you are down a pawn -- and once you take back, it is level." in directive)
        val (_, moveRung) = coach.instruction(
            Trigger.USER_ASK, position, DisclosureLevel.MOVE, "am I up material?", emptyList(), null, null,
        )
        assertEquals("Material: you are down a pawn -- and after your recapture Nxe4, it is level.", moveRung)
    }

    @Test
    fun whyWasThatBadSaysWhatTheMoveAllowed() {
        // 1. Qxd5?? leaves the rook on a1 to Rxa1+; Rxa8+ was the move.
        val opening = "r3k3/5ppp/8/3p4/8/8/5PPP/R2QK3 w - - 0 1"
        val (coach, provider) = coach(emptyList())
        start(coach, opening)
        coach.onEvent(
            Analysis(
                gameId = "g",
                fen = opening,
                ply = 0,
                lines = listOf(Line(moves = listOf("a1a8", "e8e7"), depth = 12, expectedScore = 0.99)),
                decisive = false,
                onlyMove = false,
            ),
            0.5,
        )
        val board = Board(opening)
        val uci = move(coach, board, 1, "Qxd5", byUser = true, ts = 1.0)
        coach.onEvent(
            MoveAssessment(
                gameId = "g",
                ply = 1,
                uci = uci,
                expectedLoss = 0.45,
                severity = Severity.BLUNDER,
                best = listOf("a1a8"),
                label = Label.BLUNDER,
            ),
            1.5,
        )
        val after = board.copy()
        val line = ArrayList<String>()
        for (san in listOf("Rxa1+", "Ke2", "Ra2+", "Kf3")) {
            val m = after.parseSan(san)!!
            line.add(m.uci())
            after.push(m)
        }
        coach.onEvent(
            Analysis(
                gameId = "g",
                fen = board.fen(),
                ply = 1,
                lines = listOf(Line(moves = line, depth = 12, expectedScore = 0.95)),
                decisive = false,
                onlyMove = false,
            ),
            2.0,
        )
        messages(coach)
        coach.ask("why was that bad?", 3.0)
        messages(coach)
        val directive = directive(given(provider))
        // Counted with the pawn Qxd5 took: the coach's board keeps its moves (desktop 2820821,
        // which the phone has; the desktop's test reads so since).
        assertTrue(
            directive,
            "What Qxd5 allowed, from the engine's line: after Qxd5, Rxa1+ wins a rook for a pawn." in directive,
        )
        assertTrue(directive, "say why, concretely: what Rxa8+ does, and what Qxd5 missed or allowed" in directive)
        assertTrue(directive, "Name moves bare" in directive)
    }

    /**
     * The directive asks for no figure; one given anyway is said, and noted for the log (Tai,
     * 2026-09-28: a slip of form is minor, the prompt's to prevent; desktop 3ff672b).
     */
    @Test
    fun aFigureInAHintNobodyAskedAFigureOfIsNoted() {
        val reply = "You are ahead by about 1.2 pawns: their king on e8 has one square. Which check?"
        val (coach, provider) = coach(listOf(reply))
        start(coach)
        val board = Board()
        for ((i, san) in listOf("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6").withIndex()) {
            move(coach, board, i + 1, san, byUser = (i + 1) % 2 == 1, ts = (i + 1).toDouble())
        }
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        coach.ask("give me a hint", 8.0)
        val out = messages(coach)
        assertTrue("Quote no figure" in directive(given(provider, 0)))
        assertTrue(out.map { it.text }.toString(), out.map { it.text } == listOf(reply) && provider.prompts.size == 1)
        val notes = coach.spoken.last().notes
        assertTrue(notes.toString(), "a figure in a remark nobody asked for: say it in words" in notes)
    }

    companion object {
        // 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6??: white to move, Qxf7 mates.
        private const val SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    }
}
