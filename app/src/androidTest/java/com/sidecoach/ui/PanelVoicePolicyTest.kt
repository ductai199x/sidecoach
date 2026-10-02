package com.sidecoach.ui

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
import com.sidecoach.coach.Coach
import com.sidecoach.events.GameStarted
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.Trigger
import com.sidecoach.llm.OfflineProvider
import com.sidecoach.vision.Pieces
import com.sidecoach.voice.TtsProvider
import com.sidecoach.voice.Voices
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.Collections
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * The voice never talks about a position the board has left behind: `PanelVoice.voiceAction`
 * against `tests/test_panel_voice_policy.py`'s cases and `test_coach.py`'s fast-game one; what
 * speaking does to the voice (a cut-in stops it first), the way from a move to its first sound,
 * a word held back in a fast game and said once the player slows, and the settings page's
 * preview line and voice list (`test_voices_personas.py`).
 */
@RunWith(AndroidJUnit4::class)
class PanelVoicePolicyTest {

    private val long = "Your knight on f5 is loose; the queen check on g6 wins it. What defends it?"
    private val short = "exf5: good."

    /** A voice that writes down what it is asked to do (`FakeTTS`). */
    private class FakeTts(private val ready: Boolean = true) : TtsProvider {
        override val name = "fake"
        val calls: MutableList<String> = Collections.synchronizedList(ArrayList())
        @Volatile override var busy = false
        override fun healthcheck() = ready to (if (ready) "fake" else "fake: no model")
        override fun say(text: String, voice: String?) {
            calls.add("say:" + text + (if (voice != null) " in $voice" else ""))
        }
        override fun stop() {
            calls.add("stop")
        }
    }

    @After
    fun tearDown() {
        PanelVoice.setMuted(false)
        PanelVoice.speaker = null
        PanelVoice.voicing = null
        PanelVoice.coach = null
        PanelEngine.set("ply" to 0, "ptt" to false)
    }

    /**
     * While the push-to-talk key is held the coach is silent, a reply that began under the key
     * stays silent after it, and the key up, the next line is spoken (`test_voice_ptt_quiet.py`).
     */
    @Test
    fun theCoachIsSilentWhileThePushToTalkKeyIsHeld() {
        val fake = FakeTts()
        PanelVoice.speaker = fake
        PanelEngine.set("ply" to 12, "ptt" to true)
        assertEquals("muted", PanelVoice.speak("stuck", long, 12))
        val sink = PanelVoice.streamSink(Trigger.USER_ASK, 12)!!
        assertEquals(true, sink("The knight is loose."))
        PanelEngine.set("ptt" to false)
        assertEquals("the reply stays whole and silent", true, sink("What defends it?"))
        assertEquals(emptyList<String>(), fake.calls.toList())
        assertEquals("speak", PanelVoice.speak("stuck", long, 12))
        assertEquals(listOf("say:$long"), fake.calls.toList())
    }

    private fun act(
        trigger: String,
        text: String,
        ply: Int?,
        plyNow: Int,
        busy: Boolean,
        playing: Pair<String, Int?>? = null,
        fast: Boolean = false,
    ) = PanelVoice.voiceAction(trigger, text, ply, plyNow, busy, playing, fast)

    @Test
    fun aMessageAboutAPositionTheBoardLeftIsNotSpoken() {
        // A verdict on the user's move (made from ply 10) is still the news after the move (11)
        // and the bot's instant reply (12), and stale once the user has moved again (13).
        assertEquals("speak", act("post_mistake", long, 10, 10, false))
        assertEquals("speak", act("post_mistake", long, 10, 12, false))
        assertEquals("speak", act("post_move", short, 10, 12, false))
        assertEquals("stale", act("post_mistake", long, 10, 13, false))
        // A hint about the live position is moot as soon as the user moves.
        assertEquals("speak", act("stuck", long, 10, 10, false))
        assertEquals("stale", act("stuck", long, 10, 11, false))
        assertEquals("stale", act("decisive_question", long, 10, 11, false))
        // The answer to the user's own question is never stale, however far the board has moved.
        assertEquals("supersede", act("user_ask", long, null, 40, true))
        assertEquals("speak", act("user_ask", long, 10, 40, false))
        // Nor is the review: the game is over. After a mate it is written on the last position
        // with facts, two plies behind the board.
        assertEquals("speak", act("review", long, 12, 14, false))
        assertEquals("speak", act("review", long, 12, 14, true, "post_move" to 12))
    }

    @Test
    fun shortVerdictsYieldToABusyVoiceAndSubstanceCutsIn() {
        assertEquals("speak", act("post_move", short, 4, 4, false))
        assertEquals("skip", act("post_move", short, 4, 4, true))
        // Nothing known about what is playing: substance cuts in.
        assertEquals("supersede", act("decisive_question", long, 4, 4, true))
        assertEquals("supersede", act("stuck", long, 4, 4, true))
        // What is playing is about a moment the board has left: cut in.
        assertEquals("supersede", act("post_mistake", long, 6, 7, true, "post_move" to 3))
        // What is playing is still about the board as it stands: wait behind it.
        assertEquals("speak", act("review", long, 88, 88, true, "post_move" to 87))
        assertEquals("speak", act("review", long, null, 88, true, "review" to 88))
        assertEquals("speak", act("stuck", long, 4, 4, true, "post_move" to 3))
        // The user's own question never waits.
        assertEquals("supersede", act("user_ask", long, 4, 4, true, "review" to 4))
    }

    @Test
    fun theVoiceHoldsItsWordInAFastGame() {
        assertEquals("text", act("post_move", "Qxf7#: mate.", 7, 7, false, null, true))
        assertEquals("text", act("post_mistake", "Bb7 was a miss.", 7, 7, false, null, true))
        assertEquals("an ask is answered aloud", "speak", act("user_ask", "Yes.", 7, 7, false, null, true))
        assertEquals("speak", act("review", "You won.", 7, 7, false, null, true))
        assertEquals("speak", act("post_move", "Qxf7#: mate.", 7, 7, false, null, false))
    }

    /** `test_speak_records_the_action_and_stops_before_cutting_in`. */
    @Test
    fun speakRecordsTheActionAndStopsBeforeCuttingIn() {
        val fake = FakeTts()
        PanelVoice.speaker = fake
        PanelEngine.set("ply" to 12)
        assertEquals("speak", PanelVoice.speak("post_mistake", long, 12))
        assertEquals(listOf("say:$long"), fake.calls.toList())
        fake.calls.clear()
        fake.busy = true
        assertEquals("the lesson playing is current", "speak", PanelVoice.speak("stuck", long, 12))
        assertEquals(listOf("say:$long"), fake.calls.toList())
        fake.calls.clear()
        PanelEngine.set("ply" to 16) // the board moved on past what is playing
        assertEquals("supersede", PanelVoice.speak("stuck", long, 16))
        assertEquals(listOf("stop", "say:$long"), fake.calls.toList())
        fake.calls.clear()
        PanelEngine.set("ply" to 12)
        PanelVoice.voicing = "post_move" to 11
        assertEquals("skip", PanelVoice.speak("post_move", short, 12))
        assertEquals("stale", PanelVoice.speak("post_move", short, 8))
        assertEquals(emptyList<String>(), fake.calls.toList())
        PanelVoice.setMuted(true)
        assertEquals("muted", PanelVoice.speak("stuck", long, 12))
    }

    /** `test_the_way_from_a_move_to_its_first_sound_is_timed`. */
    @Test
    fun theWayFromAMoveToItsFirstSoundIsTimed() {
        PanelVoice.speaker = FakeTts()
        PanelEngine.set("ply" to 1)
        val before = PanelVoice.latency()
        val n = before["move_to_speech_n"] as Int
        PanelVoice.noteMove(1)
        assertEquals(n, PanelVoice.latency()["move_to_speech_n"])
        val sink = PanelVoice.streamSink(Trigger.POST_MOVE, 0)
        assertNotNull(sink)
        Thread.sleep(50)
        assertTrue(sink!!("e4: best.")) // the first line of the word on ply 1, queued
        Thread.sleep(50)
        PanelVoice.onAudioStarts("e4: best.") // the voice: playing now
        var lat = PanelVoice.latency()
        assertEquals(n + 1, lat["move_to_speech_n"])
        val last = lat["move_to_speech_last_s"] as Double
        assertTrue("$lat", last in 0.08..2.0)
        if (n == 0) assertEquals(last, lat["move_to_speech_s"])
        // A line nobody waited for, or a stuck hint, is not a move's latency.
        PanelVoice.onAudioStarts("something else")
        val hint = PanelVoice.streamSink(Trigger.STUCK, 1)
        assertTrue(hint != null && hint("Look at the centre."))
        assertEquals(n + 1, PanelVoice.latency()["move_to_speech_n"])
        // The model's first token, amortised the same way.
        val tokens = before["first_token_n"] as Int
        PanelVoice.noteFirstToken(1.0)
        PanelVoice.noteFirstToken(2.0)
        lat = PanelVoice.latency()
        assertEquals(tokens + 2, lat["first_token_n"])
        assertEquals(2.0, lat["first_token_last_s"])
        if (tokens == 0) assertEquals(1.5, lat["first_token_s"])
    }

    /**
     * A mute or a barge-in while a reply is written sentence by sentence: silent from there, the
     * reply's later sentences still taken (the chat keeps the whole reply), none said
     * (`test_voice_hush.py`, voice findings 1 and 2).
     */
    @Test
    fun aMuteOrABargeInMidReplySilencesTheRestAndKeepsTheReplyWhole() {
        for (cut in listOf("mute", "barge")) {
            val fake = FakeTts()
            PanelVoice.speaker = fake
            PanelVoice.voicing = null
            val sink = PanelVoice.streamSink(Trigger.USER_ASK, null)
            assertNotNull(sink)
            assertTrue(sink!!("Your knight on f5 is loose."))
            fake.busy = true
            if (cut == "mute") PanelVoice.setMuted(true) else PanelListen.onSpeechStart()
            assertTrue(cut, sink("That check forks the king and the knight."))
            assertTrue(cut, sink("Then look at Nh4."))
            val said = fake.calls.filter { it.startsWith("say:") }
            assertEquals(cut, listOf("say:Your knight on f5 is loose."), said)
            assertEquals(cut, "stop", fake.calls.last())
            PanelVoice.setMuted(false)
        }
    }

    /**
     * The position moves on: the voice about the old one stops, and a reply that lands after is
     * shown but not spoken (Tai, 2026-10-02: "There is no need to drop any replies, the text
     * message should come out in the same correct order. The only thing that might be good is to
     * interrupt the tts so it stops its long rant"). `tests/test_voice_moved_on.py`: logic on
     * events and on the moved-on flag, never on timing.
     */
    @Test
    fun thePlayersMoveATakebackAReReadAndANewGameStopTheVoice() {
        val board = Board()
        val move = board.parseSan("e4")!!
        val before = board.fen()
        board.push(move)
        val events = listOf(
            "move" to MovePlayed("g", 13, "e4", move.uci(), before, board.fen(), byUser = true),
            "takeback" to com.sidecoach.events.Takeback("g", 1, Board().fen()),
            "re-read" to com.sidecoach.events.Resync("g", Board().fen(), "re-read"),
            "new game" to GameStarted("h", Pieces.Side.WHITE, Board().fen()),
        )
        for ((name, event) in events) {
            val fake = FakeTts()
            PanelVoice.speaker = fake
            PanelVoice.voicing = null
            PanelEngine.set("ply" to 12, "ptt" to false)
            assertEquals(name, "speak", PanelVoice.speak("stuck", long, 12))
            fake.busy = true
            val hushed = PanelVoice.hushed
            PanelEngine.onGameEvent(event)
            assertEquals(name, "stop", fake.calls.last())
            assertEquals(name, hushed + 1, PanelVoice.hushed)
        }
    }

    /** Tai: the opponent's move alone must not stop the voice mid-sentence. */
    @Test
    fun theOpponentsMoveAloneDoesNotStopTheVoice() {
        val fake = FakeTts()
        PanelVoice.speaker = fake
        PanelVoice.voicing = null
        PanelEngine.set("ply" to 12, "ptt" to false)
        assertEquals("speak", PanelVoice.speak("stuck", long, 12))
        fake.busy = true
        val board = Board()
        val move = board.parseSan("e4")!!
        val before = board.fen()
        board.push(move)
        PanelEngine.onGameEvent(MovePlayed("g", 13, "e4", move.uci(), before, board.fen(), byUser = false))
        assertEquals(listOf("say:$long"), fake.calls.toList())
    }

    @Test
    fun aReplyAskedForAMomentThatIsOverIsShownNotSpoken() {
        val fake = FakeTts()
        PanelVoice.speaker = fake
        PanelVoice.voicing = null
        PanelEngine.set("ply" to 12, "ptt" to false)
        var over = true
        var sink = PanelVoice.streamSink(Trigger.USER_ASK, 12) { over }!!
        // Taken, every sentence, so the reply is whole in the record: declined, the coach said a
        // template in its place. Not spoken.
        assertTrue(sink("The knight is loose."))
        assertTrue(sink("What defends it?"))
        assertEquals(emptyList<String>(), fake.calls.toList())
        // The moment still holds: spoken as ever.
        over = false
        sink = PanelVoice.streamSink(Trigger.USER_ASK, 12) { over }!!
        assertTrue(sink("Look at f7."))
        assertEquals(listOf("say:Look at f7."), fake.calls.toList())
        // Not streamed: a reply that is late for the voice is said to be stale.
        assertEquals("stale", PanelVoice.speak("user_ask", "Late word.", 12, movedOn = true))
        assertEquals(listOf("say:Look at f7."), fake.calls.toList())
    }

    /**
     * Before the warm-up has built the voice there is nothing to stream to: no sink, as when
     * muted, so the reply is read whole. The sink declined instead, and the coach said its
     * template in place of every AI reply until the voice was there (the round-3 parity audit).
     */
    @Test
    fun noVoiceYetMeansNoSinkNotADeclinedReply() {
        org.junit.Assume.assumeTrue("the voice is built in this process", PanelVoice.tts == null)
        PanelVoice.speaker = null
        PanelVoice.voicing = null
        PanelEngine.set("ply" to 12, "ptt" to false)
        assertEquals(null, PanelVoice.streamSink(Trigger.USER_ASK, 12) { false })
    }

    @Test
    fun aReplyThatBeganWhenThePositionMovesOnIsCutAndGoesOnSilent() {
        val fake = FakeTts()
        PanelVoice.speaker = fake
        PanelVoice.voicing = null
        PanelEngine.set("ply" to 12, "ptt" to false)
        val sink = PanelVoice.streamSink(Trigger.USER_ASK, 12) { false }!!
        assertTrue(sink("First, the idea."))
        assertEquals(listOf("say:First, the idea."), fake.calls.toList())
        fake.busy = true
        PanelVoice.positionMovedOn("the player moved") // the rest is only text
        assertTrue(sink("Then the second sentence."))
        assertEquals(listOf("say:First, the idea.", "stop"), fake.calls.toList())
    }

    /**
     * A room's words (a review's card, a puzzle's coach): silent when muted, and a room's hush
     * stops only its own words, never the live coach (`test_a_card_is_said_in_the_panels_voice_
     * and_captioned`, voice finding 12).
     */
    @Test
    fun muteSilencesTheRoomsAndARoomStopsOnlyItsOwnWords() {
        val fake = FakeTts()
        PanelVoice.speaker = fake
        PanelVoice.voicing = "post_move" to 10 // the live coach speaking
        PanelVoice.roomStop()
        assertEquals(emptyList<String>(), fake.calls.toList())
        PanelVoice.roomSay("Na5 is a mistake.")
        assertEquals(listOf("say:Na5 is a mistake."), fake.calls.toList())
        PanelVoice.roomStop()
        assertEquals("stop", fake.calls.last())
        PanelVoice.setMuted(true)
        fake.calls.clear()
        PanelVoice.roomSay("Nf3 is best.")
        assertTrue(fake.calls.none { it.startsWith("say:") })
    }

    /**
     * A word the fast game kept on the page is said once the player has been thinking in that
     * position as long as the pace allows (`_say_held_word`); one about a position the board
     * has left, or after the game is over, is dropped
     * (`test_a_word_held_in_a_fast_game_is_said_once_the_player_slows`).
     */
    @Test
    fun aWordHeldInAFastGameIsSaidWhenThePlayerSlows() {
        val fake = FakeTts()
        PanelVoice.speaker = fake
        val coach = Coach(OfflineProvider())
        coach.onEvent(GameStarted("g", Pieces.Side.WHITE, Board().fen()), 0.0)
        val board = Board()
        for ((i, san) in listOf("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6").withIndex()) {
            val ply = i + 1
            val move = board.parseSan(san)!!
            val before = board.fen()
            board.push(move)
            coach.onEvent(MovePlayed("g", ply, san, move.uci(), before, board.fen(), byUser = ply % 2 == 1, thinkMs = 1000), ply.toDouble())
        }
        assertTrue(coach.fastNow(7.0))
        PanelVoice.coach = coach
        PanelVoice.loopTs = 7.0
        PanelEngine.set("ply" to 6)
        assertEquals("text", PanelVoice.speak("post_move", "Nf6: good.", 5))
        assertEquals(emptyList<String>(), fake.calls.toList())
        PanelVoice.sayHeldWord(coach, 7.0) // a second into the position: still fast
        assertEquals(emptyList<String>(), fake.calls.toList())
        PanelVoice.sayHeldWord(coach, 13.0) // thinking longer than the pace: the game slowed
        assertEquals(listOf("say:Nf6: good."), fake.calls.toList())
        PanelVoice.sayHeldWord(coach, 20.0) // said once
        assertEquals(1, fake.calls.size)
        // A word about a position the board has left is dropped, not said late.
        assertEquals("text", PanelVoice.speak("stuck", "Look at f7.", 6))
        PanelEngine.set("ply" to 7)
        PanelVoice.sayHeldWord(coach, 30.0)
        PanelEngine.set("ply" to 6)
        PanelVoice.sayHeldWord(coach, 30.0)
        assertEquals(1, fake.calls.size)
        // Nor after the game: the review speaks, not a late verdict.
        assertEquals("text", PanelVoice.speak("post_move", "Nf6: good.", 5))
        coach.finished = true
        PanelVoice.sayHeldWord(coach, 30.0)
        coach.finished = false
        PanelVoice.sayHeldWord(coach, 30.0)
        assertEquals(1, fake.calls.size)
    }

    /**
     * A streamed reply whose first sentence met a fast game was declined: the model's reply was
     * thrown away and the template shown in its place (Tai, 2026-10-02: kept, never replaced). It is
     * taken sentence by sentence, captioned, not said, and held whole for the pace like a word on
     * the page: said once the player slows.
     * `test_a_streamed_reply_in_a_fast_game_is_kept_and_held_whole_for_the_pace`.
     */
    @Test
    fun aStreamedReplyInAFastGameIsKeptAndHeldWholeForThePace() {
        val fake = FakeTts()
        PanelVoice.speaker = fake
        PanelVoice.voicing = null
        val coach = Coach(OfflineProvider())
        coach.onEvent(GameStarted("g", Pieces.Side.WHITE, Board().fen()), 0.0)
        val board = Board()
        for ((i, san) in listOf("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6").withIndex()) {
            val ply = i + 1
            val move = board.parseSan(san)!!
            val before = board.fen()
            board.push(move)
            coach.onEvent(MovePlayed("g", ply, san, move.uci(), before, board.fen(), byUser = ply % 2 == 1, thinkMs = 1000), ply.toDouble())
        }
        PanelVoice.coach = coach
        PanelVoice.loopTs = 7.0
        PanelEngine.set("ply" to 6, "ptt" to false)
        var sink = PanelVoice.streamSink(Trigger.POST_MISTAKE, 5)!!
        assertTrue("kept, not declined", sink("Nf6 left f7 to the queen."))
        assertTrue(sink("Qxf7 is mate now."))
        assertEquals("a fast game: not said yet", emptyList<String>(), fake.calls.toList())
        PanelVoice.sayHeldWord(coach, 13.0) // the game slowed
        assertEquals(listOf("say:Nf6 left f7 to the queen. Qxf7 is mate now."), fake.calls.toList())
        // Cut off while held (the position left): the rest is captioned and nothing is said later.
        sink = PanelVoice.streamSink(Trigger.POST_MISTAKE, 5)!!
        assertTrue(sink("Look at f7."))
        PanelVoice.positionMovedOn("takeback")
        assertTrue(sink("Their king has no square."))
        PanelVoice.sayHeldWord(coach, 30.0)
        assertEquals(1, fake.calls.count { it.startsWith("say:") })
    }

    /**
     * The settings page's Listen button (`test_the_preview_line_says_whether_the_voice_could_speak`,
     * `test_voice_labels_read_as_accent_sex_and_name`): a voice that cannot speak says so and
     * nothing breaks; one that can says the line in the voice asked for, under its label, and
     * the coach's own voice is untouched.
     */
    @Test
    fun thePreviewSaysWhetherTheVoiceCouldSpeak() {
        val voice = PanelVoice.voice
        PanelVoice.speaker = FakeTts(ready = false)
        assertFalse(PanelVoice.saySample("bf_emma"))
        val fake = FakeTts()
        PanelVoice.speaker = fake
        val cursor = PanelEngine.poll(Int.MAX_VALUE).getInt("cursor")
        assertTrue(PanelVoice.saySample("bf_emma"))
        assertEquals(listOf("stop", "say:${PanelVoice.VOICE_SAMPLE} in bf_emma"), fake.calls.toList())
        val lines = PanelEngine.poll(cursor).getJSONArray("lines")
        val said = (0 until lines.length()).map { lines.getJSONObject(it).getString("text") }
        assertTrue(said.toString(), "preview in British Female, Emma: ${PanelVoice.VOICE_SAMPLE}" in said)
        assertEquals("the coach's own voice is untouched", voice, PanelVoice.voice)
        // The labels: accent, sex and name; outside the scheme, the name as it is.
        assertEquals("American Female, Sarah", Voices.label("af_sarah"))
        assertEquals("British Male, George", Voices.label("bm_george"))
        assertEquals("American Male, Michael", Voices.label("am_michael"))
        assertEquals("jf_alpha", Voices.label("jf_alpha"))
        assertFalse(Voices.isEnglish("jf_alpha"))
        assertTrue(Voices.ENGLISH_VOICES.all { Voices.isEnglish(it) })
        assertEquals(setOf("af", "am", "bf", "bm"), Voices.ENGLISH_VOICES.map { it.take(2) }.toSet())
    }

    /**
     * With no voices file, a missing one, or one that will not open, the picker offers the known
     * names; a voices file that opens offers its own English voices, in the picker's order
     * (`test_the_voice_list_falls_back_to_the_known_names` in tests/test_voices_personas.py; the
     * file that opens is the phone's own case, so the fallback is seen to be one; the parity
     * audit, 2026-09-29).
     */
    @Test
    fun theVoiceListFallsBackToTheKnownNames() {
        assertEquals(Voices.ENGLISH_VOICES, Voices.english(null))
        val dir = File(InstrumentationRegistry.getInstrumentation().targetContext.cacheDir, "voices-fallback")
        dir.deleteRecursively()
        dir.mkdirs()
        try {
            assertEquals("a file that is not there", Voices.ENGLISH_VOICES, Voices.english(File(dir, "missing.bin")))
            val unreadable = File(dir, "voices-v1.0.bin").apply { writeBytes("not a voices bundle".toByteArray()) }
            assertEquals("a file that will not open", Voices.ENGLISH_VOICES, Voices.english(unreadable))
            val bundle = File(dir, "voices-real.bin")
            ZipOutputStream(bundle.outputStream()).use { zip ->
                for (name in listOf("bm_george.npy", "jf_alpha.npy", "af_sarah.npy")) {
                    zip.putNextEntry(ZipEntry(name))
                    zip.write(byteArrayOf(0))
                    zip.closeEntry()
                }
            }
            assertEquals(listOf("af_sarah", "bm_george"), Voices.english(bundle))
        } finally {
            dir.deleteRecursively()
        }
    }
}
