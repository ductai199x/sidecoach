package com.sidecoach.ui

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The voice never talks about a position the board has left behind: `PanelVoice.voiceAction`
 * against `tests/test_panel_voice_policy.py`'s cases and `test_coach.py`'s fast-game one.
 */
@RunWith(AndroidJUnit4::class)
class PanelVoicePolicyTest {

    private val long = "Your knight on f5 is loose; the queen check on g6 wins it. What defends it?"
    private val short = "exf5: good."

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

    /** A voice that writes down what it was told, as the desktop's test's `FakeTTS`. */
    private class Told : PanelVoice.Speaking {
        val calls = ArrayList<String>()
        override var busy = false
        override fun say(text: String) { calls.add("say:$text") }
        override fun stop() { calls.add("stop") }
    }

    /** `test_speak_records_the_action_and_stops_before_cutting_in`. */
    @Test
    fun speakRecordsTheActionAndStopsBeforeCuttingIn() {
        val fake = Told()
        val ply = PanelEngine.plyNow()
        PanelVoice.standIn = fake
        PanelVoice.voicing = null
        try {
            PanelEngine.set("ply" to 12)
            assertEquals("speak", PanelVoice.speak("post_mistake", long, 12))
            assertEquals(listOf("say:$long"), fake.calls)
            fake.calls.clear()
            fake.busy = true
            assertEquals("the lesson playing is current", "speak", PanelVoice.speak("stuck", long, 12))
            assertEquals(listOf("say:$long"), fake.calls)
            fake.calls.clear()
            PanelEngine.set("ply" to 16) // the board moved on past what is playing
            assertEquals("supersede", PanelVoice.speak("stuck", long, 16))
            assertEquals(listOf("stop", "say:$long"), fake.calls)
            fake.calls.clear()
            PanelEngine.set("ply" to 12)
            PanelVoice.voicing = "post_move" to 11
            assertEquals("skip", PanelVoice.speak("post_move", short, 12))
            assertEquals("stale", PanelVoice.speak("post_move", short, 8))
            assertEquals(emptyList<String>(), fake.calls)
            PanelVoice.setMuted(true)
            assertEquals("muted", PanelVoice.speak("stuck", long, 12))
        } finally {
            PanelVoice.setMuted(false)
            PanelVoice.standIn = null
            PanelVoice.voicing = null
            PanelEngine.set("ply" to ply)
        }
    }

    /**
     * The status strip's move-to-speech latency: the move noted when committed, the reply's
     * first line remembered when queued, the time taken when its audio starts; a line nobody
     * waited for, or a hint, is not a move's. The model's first token, amortised the same way.
     * `test_the_way_from_a_move_to_its_first_sound_is_timed`.
     */
    @Test
    fun theWayFromAMoveToItsFirstSoundIsTimed() {
        val fake = Told()
        PanelVoice.standIn = fake
        PanelVoice.voicing = null
        val before = PanelVoice.latency()
        val n = before["move_to_speech_n"] as Int
        val ply = PanelEngine.plyNow()
        PanelEngine.set("ply" to 0)
        try {
            PanelVoice.noteMove(1)
            val sink = PanelVoice.streamSink(com.sidecoach.events.Trigger.POST_MOVE, 0)!!
            Thread.sleep(50)
            assertTrue(sink("e4: best."))
            Thread.sleep(50)
            PanelVoice.onAudioStarts("e4: best.")
            val lat = PanelVoice.latency()
            assertEquals(n + 1, lat["move_to_speech_n"])
            val last = lat["move_to_speech_last_s"] as Double
            assertTrue(lat.toString(), last in 0.08..2.0)
            PanelVoice.onAudioStarts("something else")
            PanelVoice.voicing = null
            assertTrue(PanelVoice.streamSink(com.sidecoach.events.Trigger.STUCK, 1)!!("Look at the centre."))
            assertEquals(n + 1, PanelVoice.latency()["move_to_speech_n"])
            val tokens = PanelVoice.latency()["first_token_n"] as Int
            PanelVoice.noteFirstToken(1.0)
            PanelVoice.noteFirstToken(2.0)
            val after = PanelVoice.latency()
            assertEquals(tokens + 2, after["first_token_n"])
            assertEquals(2.0, after["first_token_last_s"])
        } finally {
            PanelVoice.standIn = null
            PanelVoice.voicing = null
            PanelEngine.set("ply" to ply)
        }
    }

    @Test
    fun theVoiceHoldsItsWordInAFastGame() {
        assertEquals("text", act("post_move", "Qxf7#: mate.", 7, 7, false, null, true))
        assertEquals("text", act("post_mistake", "Bb7 was a miss.", 7, 7, false, null, true))
        assertEquals("an ask is answered aloud", "speak", act("user_ask", "Yes.", 7, 7, false, null, true))
        assertEquals("speak", act("review", "You won.", 7, 7, false, null, true))
        assertEquals("speak", act("post_move", "Qxf7#: mate.", 7, 7, false, null, false))
    }
}
