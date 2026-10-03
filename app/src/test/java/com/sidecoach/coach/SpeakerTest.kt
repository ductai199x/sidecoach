package com.sidecoach.coach

import com.sidecoach.chess.Board
import com.sidecoach.events.Analysis
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.GameStarted
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import com.sidecoach.events.Trigger
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.Completion
import com.sidecoach.llm.LLMProvider
import com.sidecoach.llm.StreamCallback
import com.sidecoach.llm.StreamingProvider
import com.sidecoach.vision.Pieces
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The Speaker on its own: the reply streamed to the voice as it is written, the coach handing
 * its sink to the request, the verbalized candidates parsed whole, a verifier's keep-and-repair,
 * a routine request's model, and an ask overtaking what is still waiting. The desktop's
 * `tests/test_streaming.py`, `tests/test_coach_verbalized.py` and the Speaker cases of
 * `tests/test_coach.py`. (The Speaker's rounds of engine calls go through the coach, as the
 * desktop's `tests/test_engine_tools.py` does: CoachTest.)
 */
class SpeakerTest {

    /** `StreamingProvider`: the reply through the callback in small pieces; notes an early stop. */
    private class Streaming(private val reply: String, private val pieces: Int = 7) : StreamingProvider {
        override val name = "streaming"
        var stoppedEarly = false
        var calls = 0
        var streamedCalls = 0

        override fun healthcheck(): Pair<Boolean, String> = true to "streaming"

        override fun chat(
            system: String,
            messages: List<ChatMessage>,
            maxTokens: Int,
            temperature: Double,
            thinking: Boolean,
        ): Completion {
            calls += 1
            return Completion(text = reply)
        }

        override fun chatStream(
            system: String,
            messages: List<ChatMessage>,
            maxTokens: Int,
            temperature: Double,
            thinking: Boolean,
            onDelta: StreamCallback,
        ): Completion {
            calls += 1
            streamedCalls += 1
            val step = maxOf(1, reply.length / pieces)
            val sent = StringBuilder()
            var i = 0
            while (i < reply.length) {
                val piece = reply.substring(i, minOf(reply.length, i + step))
                sent.append(piece)
                if (!onDelta(piece)) {
                    stoppedEarly = true
                    break
                }
                i += step
            }
            return Completion(text = sent.toString(), stats = mapOf("streamed" to true))
        }
    }

    /** `_request`: a word on a move, to [sink] or whole. */
    private fun request(sink: ((String) -> Boolean)?, routine: Boolean = false) = CoachRequest(
        trigger = Trigger.POST_MOVE,
        level = DisclosureLevel.LINE,
        system = "coach",
        messages = listOf(ChatMessage("user", "# Task\nA word on the move, two sentences.")),
        facts = PositionFacts(fen = FEN, sideToMove = "white", phase = "opening", material = "level"),
        board = Board(FEN),
        fallback = "Fine.",
        ply = 6,
        gameId = "g",
        streamTo = sink,
        routine = routine,
    )

    /** A board check that faults the sentences holding [word]. */
    private fun faulting(word: String, fault: String) = DraftCheck { text, _, _, _ ->
        if (word in text) listOf(fault) else emptyList()
    }

    /**
     * The first verbalized candidate goes out sentence by sentence, each checked before it is
     * said, and the stream is closed once that candidate is complete; a sentence that fails
     * stops it with what was said kept, and no template follows a voice that has spoken; a
     * first sentence that fails goes back to the whole-reply path; the sink may stop the stream
     * when the board moves on; a bare reply streams too; without a sink the reply is sampled
     * whole as before. `test_streaming.py`'s six stream cases.
     */
    @Test
    fun theFirstCandidateStreamsSentenceBySentenceChecked() {
        class Case(
            val name: String,
            val reply: String,
            val pieces: Int = 7,
            val check: DraftCheck? = null,
            val sinkTakes: Int? = Int.MAX_VALUE, // null: no sink; else how many sentences it takes
            val heard: List<String>,
            val streamed: Boolean,
            val text: String,
            val problems: List<String>? = null, // null: as the desktop, not asserted
            val fallbacks: Int? = null,
            val stoppedEarly: Boolean? = null,
            val calls: Pair<Int, Int>? = null, // (streamed calls, all calls)
        )
        val cases = listOf(
            Case(
                "first candidate, stream closed",
                "[p=0.4] Nf3 was best. It develops with a threat.\n[p=0.6] Good move, the knight belongs there. Keep going.",
                heard = listOf("Nf3 was best.", "It develops with a threat."),
                streamed = true, text = "Nf3 was best. It develops with a threat.", stoppedEarly = true,
                problems = emptyList(), fallbacks = 0,
            ),
            Case(
                "a failing sentence stops it",
                "[p=0.5] A fine move. Your rook on f7 is strong.\n[p=0.5] Clean reply here.",
                check = faulting("rook on f7", "the rook on f7 is not there"),
                heard = listOf("A fine move."), streamed = true, text = "A fine move.",
                problems = listOf("the rook on f7 is not there"), fallbacks = 0,
            ),
            Case(
                "a failing first sentence: the whole reply",
                "[p=0.5] The secret is Qxf7.\n[p=0.5] Look at the f7 square.",
                check = faulting("secret", "leak"),
                heard = emptyList(), streamed = false, text = "Look at the f7 square.", calls = 1 to 2,
            ),
            Case(
                "the sink stops it",
                "[p=1.0] One. Two. Three.", pieces = 12, sinkTakes = 1,
                heard = listOf("One.", "Two."), streamed = true, text = "One.",
            ),
            Case(
                "a bare reply",
                "Just one plain reply. And a second sentence.",
                heard = listOf("Just one plain reply.", "And a second sentence."), streamed = true,
                text = "Just one plain reply. And a second sentence.",
            ),
            Case(
                "wrapped in quotes: said without them",
                "[p=0.7] \"Solid block. Qe2 was stronger.\"\n[p=0.3] Fine.",
                heard = listOf("Solid block.", "Qe2 was stronger."), streamed = true,
                text = "Solid block. Qe2 was stronger.",
            ),
            Case(
                "no sink", "[p=1.0] Whole reply.", sinkTakes = null,
                heard = emptyList(), streamed = false, text = "Whole reply.", calls = 0 to 1,
            ),
        )
        for (case in cases) {
            val provider = Streaming(case.reply, case.pieces)
            val speaker = Speaker(provider, draftCheck = case.check)
            val heard = ArrayList<String>()
            val sink: ((String) -> Boolean)? = case.sinkTakes?.let { takes ->
                { sentence: String -> heard.add(sentence); heard.size <= takes }
            }
            val spoken = speaker.speakNow(request(sink))
            assertEquals(case.name, case.heard, heard)
            assertEquals(case.name, case.streamed, spoken.streamed)
            assertEquals(case.name, case.text, spoken.message.text)
            case.problems?.let { assertEquals(case.name, it, spoken.problems) }
            case.fallbacks?.let { assertEquals("${case.name}: no template follows", it, speaker.fallbacks) }
            case.stoppedEarly?.let { assertEquals(case.name, it, provider.stoppedEarly) }
            case.calls?.let { assertEquals(case.name, it, provider.streamedCalls to provider.calls) }
        }
    }

    /**
     * The voice declined the first sentence, the stream was closed there, and the cut-off
     * candidate still went out as the text (the deep prompt review, 2026-10-01). Only candidates
     * the stream finished are read; with none, the template stands
     * (`test_a_voice_that_declines_the_first_sentence_leaves_no_cut_off_text`).
     */
    @Test
    fun aVoiceThatDeclinesTheFirstSentenceLeavesNoCutOffText() {
        val reply = "[p=0.6] Taking on f7 wins. It also opens the king for the rook.\n[p=0.4] Other."
        val provider = Streaming(reply, pieces = 30)
        val speaker = Speaker(provider)
        val spoken = speaker.speakNow(request({ false }))
        assertFalse(spoken.streamed)
        assertEquals("Fine.", spoken.message.text)
        assertEquals("no second call about a moment the voice declined", 1, provider.calls)
    }

    /**
     * The stream reader: deltas that split a marker, a stop at the buffer's end ("costing 0." is
     * the start of "0.03"), and a bare move number waiting for its sentence. `test_streaming.py`.
     */
    @Test
    fun theReaderWaitsForWholeSentencesAcrossDeltas() {
        fun read(deltas: List<String>): Pair<List<String>, Boolean> {
            val heard = ArrayList<String>()
            val reader = FirstCandidate({ heard.add(it); true }, { emptyList() })
            for (delta in deltas) if (!reader.feed(delta)) break
            reader.finish()
            return heard to reader.done
        }
        assertEquals(
            listOf("First one.", "Second one.") to true,
            read(listOf("[p", "=0.", "3] Fir", "st one. Sec", "ond one.", " [p=0.7] Other.")),
        )
        assertEquals(
            listOf("6. d3 was good, costing 0.03 of expected score.", "Bxc6+ was better."),
            read(listOf("[p=0.3] 6.", " d3 was good, costing 0.", "03 of expected score.", " Bxc6+ was better.", " [p=0.7] Other.")).first,
        )
        assertEquals(
            listOf("6. d3 was good, costing 0.03; 6. Bxc6+ was better.", "And so on."),
            read(listOf("6. d3 was good, costing 0.03; 6. Bxc6+ was better. And so on.")).first,
        )
    }

    /** A mundane reply takes the routine model; everything else the main one (`test_streaming.py`). */
    @Test
    fun aRoutineRequestTakesTheRoutineProvider() {
        val speaker = Speaker(Streaming("[p=1.0] From the main model."), routineProvider = Streaming("[p=1.0] From the routine model."))
        assertEquals("From the routine model.", speaker.speakNow(request(null, routine = true)).message.text)
        assertEquals("From the main model.", speaker.speakNow(request(null)).message.text)
    }

    /**
     * The coach asks its sink factory for each request it makes, with the trigger and the ply,
     * and hands the sink it gets to the request, which the Speaker then streams to: the
     * sentence reaches that sink (`test_the_coach_hands_its_sink_to_the_request` in
     * tests/test_streaming.py, which reads the queued request's `stream_to`; the phone's queue is
     * the Speaker's own, so what reaches the sink says it; the parity audit, 2026-09-29).
     */
    @Test
    fun theCoachHandsItsSinkToTheRequest() {
        val provider = Streaming("[p=1.0] Their king has one square left.")
        val coach = Coach(provider, policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
        val seen = ArrayList<Pair<Trigger, Int?>>()
        val heard = ArrayList<String>()
        val gone = ArrayList<() -> Boolean>()
        coach.streamSink = { trigger, ply, movedOn, _ ->
            seen.add(trigger to ply)
            gone.add(movedOn)
            val sink: (String) -> Boolean = { sentence -> heard.add(sentence); true }
            sink
        }
        coach.onEvent(GameStarted(gameId = "g", userSide = Pieces.Side.WHITE, startFen = FEN), 0.0)
        coach.onEvent(
            Analysis(
                gameId = "g",
                fen = FEN,
                ply = 0,
                lines = listOf(
                    Line(moves = listOf("h5f7"), depth = 12, expectedScore = 1.0, mate = 1),
                    Line(moves = listOf("c4f7"), depth = 12, expectedScore = 0.55),
                ),
                decisive = false,
                onlyMove = true,
                tags = listOf(MotifTag(name = "mateIn1", squares = listOf("f7", "e8"), move = "h5f7")),
            ),
            1.0,
        )
        coach.tick(12.0) // stuck: the idea, unprompted
        assertEquals(listOf(Trigger.STUCK to 0), seen)
        // The moment it was asked for: over once a new game, a takeback or a re-read comes, or the
        // player moves again -- never because of what the opponent plays.
        assertFalse(gone[0]())
        coach.onEvent(GameStarted(gameId = "h", userSide = Pieces.Side.WHITE, startFen = FEN), 2.0)
        assertTrue(gone[0]())
        coach.speaker.runPending()
        assertEquals("the reply was streamed to the sink", 1, provider.streamedCalls)
        assertEquals(listOf("Their king has one square left."), heard)
    }

    /**
     * The verbalized replies parse into weighted candidates, whole: a wrapped reply kept whole, a
     * plain model's reply one candidate, a think block stripped; a reply the token budget cut
     * mid-marker leaks no marker fragment ("...move?\n\n[p=0.2"), one cut mid-sentence is
     * trimmed to its last full sentence, and a stump ("Given", "The key was using") is no
     * candidate. `test_a_reply_parses_into_weighted_candidates_and_never_a_fragment` and
     * `test_a_reply_cut_off_mid_sentence_keeps_its_last_full_sentence` in
     * tests/test_coach_verbalized.py.
     */
    @Test
    fun verbalizedRepliesParseIntoWholeWeightedCandidates() {
        val cands = parseCandidates("[p=0.5] First reply here.\n[p=0.3] Second reply,\nwrapping on.\n[p=0.2] Third.")
        assertEquals(listOf(0.5, 0.3, 0.2), cands.map { it.first })
        assertEquals("Second reply,\nwrapping on.", cands[1].second)
        assertEquals(listOf(1.0 to "Just one plain reply."), parseCandidates("Just one plain reply."))
        assertEquals(listOf(1.0 to "Clean."), parseCandidates("<think>hidden</think>\n[p=1] Clean."))
        assertEquals("only one", sampleReply(listOf(0.9 to "only one")))
        val cut = "[p=0.5] Placing it on c5 pins the piece. What would that gain?\n\n" +
            "[p=0.3] Consider the long-term implications. How might a pawn move help?\n\n[p=0.2"
        assertEquals(
            listOf(
                "Placing it on c5 pins the piece. What would that gain?",
                "Consider the long-term implications. How might a pawn move help?",
            ),
            parseCandidates(cut).map { it.second },
        )
        assertEquals("Think about the e-file. What does it need?", parseCandidates("Think about the e-file. What does it need?\n\n[p=")[0].second)
        // A reply the model wrapped in quotes is shown and said without them (the tone check of
        // 2026-10-02); a quoted name opening a longer reply stays.
        assertEquals(
            listOf(0.6 to "Solid block, but Qe2 was stronger.", 0.4 to "A fine move."),
            parseCandidates("[p=0.6] \"Solid block, but Qe2 was stronger.\"\n[p=0.4] “A fine move.”"),
        )
        val named = "\"Italian\" setups suit you. Keep playing \"e4.\""
        assertEquals(listOf(1.0 to named), parseCandidates(named))
        for ((text, whole) in listOf(
            "Your idea is spot-on. Given" to "Your idea is spot-on.",
            "Given" to "",
            "What now? Reflect" to "What now?",
            "He said \"go.\" Then" to "He said \"go.\"",
            "Look at e5 (the pawn!)" to "Look at e5 (the pawn!)",
        )) {
            assertEquals(text, whole, completeSentences(text))
        }
        assertEquals(listOf("Fine move. What is the plan?"), parseCandidates("[p=0.5] Fine move. What is the plan?\n[p=0.3] Given").map { it.second })
        assertEquals(emptyList<Pair<Double, String>>(), parseCandidates("[p=0.5] The key was using"))
        assertEquals(emptyList<Pair<Double, String>>(), parseCandidates("The key was using"))
    }

    /**
     * With a verifier, a candidate the second model finds unsupported is not spoken; when every
     * candidate has an unsupported statement, the likeliest is repaired once
     * (`test_coach_verbalized.py`). The verifier answers by what it is shown, as the checks run
     * side by side.
     */
    @Test
    fun aVerifierKeepsTheCleanCandidateAndRepairsWhenNoneIs() {
        val board = Board()
        board.push(board.parseSan("e4")!!)
        board.push(board.parseSan("e5")!!)
        fun ask() = CoachRequest(
            trigger = Trigger.USER_ASK,
            level = DisclosureLevel.LINE,
            system = "You are a coach.",
            messages = listOf(
                ChatMessage("user", "Moves so far: 1. e4 e5 -- White to move."),
                ChatMessage("user", "Answer them."),
            ),
            facts = PositionFacts(fen = board.fen(), sideToMove = "white", userSide = "white", phase = "opening", candidates = listOf("Nf3")),
            board = board,
            fallback = "Think.",
            ply = 2,
            gameId = "g",
            candidates = 2,
        )

        class Scripted(private val reply: String) : LLMProvider {
            override val name = "scripted"
            override fun healthcheck() = true to "scripted"
            override fun chat(system: String, messages: List<ChatMessage>, maxTokens: Int, temperature: Double, thinking: Boolean) =
                Completion(text = reply)
        }

        class Verifier(private val unsupported: Map<String, String>, private val repair: String) : LLMProvider {
            override val name = "scripted-verifier"
            val seen = ArrayList<String>()
            override fun healthcheck() = true to "ok"
            override fun chat(system: String, messages: List<ChatMessage>, maxTokens: Int, temperature: Double, thinking: Boolean): Completion {
                val content = messages.last().content
                synchronized(seen) { seen.add(content) }
                if (system == REPAIR_SYSTEM) return Completion(text = "{\"reply\": \"$repair\"}")
                val reply = content.substringAfter("The coach's reply:\n\"\"\"\n").substringBefore("\n\"\"\"")
                val found = unsupported.entries.firstOrNull { it.key in reply }?.value
                return Completion(text = if (found != null) "{\"unsupported\": [\"$found\"]}" else "{\"unsupported\": []}")
            }
        }

        // The first candidate invents "eyes e5"; the second is clean.
        var speaker = Speaker(
            Scripted("[p=0.6] Nf3 develops and eyes e5.\n[p=0.4] Nf3 develops a piece."),
            verifier = Verifier(mapOf("eyes e5" to "eyes e5"), "unused"),
        )
        var spoken = speaker.speakNow(ask())
        assertEquals("Nf3 develops a piece.", spoken.message.text)
        assertEquals(listOf(listOf("eyes e5"), emptyList()), spoken.verified)
        assertFalse(spoken.repaired)
        // Both unsupported: the likelier one is repaired.
        speaker = Speaker(
            Scripted("[p=0.7] Nf3 develops and eyes e5.\n[p=0.3] Nf3 pins the knight."),
            verifier = Verifier(mapOf("eyes e5" to "eyes e5", "pins the knight" to "pins the knight"), "Nf3 develops a piece."),
        )
        spoken = speaker.speakNow(ask())
        assertTrue(spoken.message.text, spoken.message.text == "Nf3 develops a piece." && spoken.repaired)
    }

    /**
     * An ask goes ahead of a remark still waiting, never instead of it (Tai, 2026-10-02: no reply
     * is dropped); the stuck hint, said first, stays first (`test_coach.py`).
     */
    @Test
    fun anAskOvertakesARemarkStillWaiting() {
        val speaker = Speaker(Streaming("[p=1.0] x"))
        fun ask(trigger: Trigger) = CoachRequest(
            trigger = trigger,
            level = DisclosureLevel.IDEA,
            system = "",
            messages = listOf(ChatMessage("user", "x")),
            facts = PositionFacts(fen = FEN, sideToMove = "white", phase = "opening"),
            board = Board(FEN),
            fallback = "fallback",
            ply = 6,
            gameId = "g",
        )
        speaker.submit(ask(Trigger.STUCK))
        speaker.submit(ask(Trigger.POST_MOVE))
        speaker.submit(ask(Trigger.USER_ASK))
        assertEquals(3, speaker.pending)
        speaker.runPending()
        assertEquals(
            listOf(Trigger.STUCK, Trigger.USER_ASK, Trigger.POST_MOVE),
            speaker.poll().map { it.message.trigger },
        )
    }

    companion object {
        const val FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    }
}
