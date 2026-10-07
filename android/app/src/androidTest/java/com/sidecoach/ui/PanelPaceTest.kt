package com.sidecoach.ui

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.events.Trigger
import com.sidecoach.voice.TtsProvider
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.util.Collections

/**
 * The panel's voice following the player's pace (`voice.pace = "follow"`, the default): the plan
 * decides, the panel captions every sentence the moment it exists, keeps the voice one line ahead
 * (the next is made while one plays: no wait between sentences), and writes the reply into the
 * chat as it comes (Tai, 2026-10-03). The desktop's `tests/test_panel_pace.py`, case for case.
 */
@RunWith(AndroidJUnit4::class)
class PanelPaceTest {

    /** A voice that plays one line and makes the next meanwhile: busy while any line is with it,
     * [waiting] the lines behind the one playing. The test ends a line ([end]). */
    private class PacedTts : TtsProvider {
        override val name = "paced"
        val lines: MutableList<String> = Collections.synchronizedList(ArrayList())
        val calls: MutableList<String> = Collections.synchronizedList(ArrayList())
        override fun healthcheck() = true to "paced"
        override val busy: Boolean get() = lines.isNotEmpty()
        override val waiting: Int get() = maxOf(0, lines.size - 1)
        override fun say(text: String, voice: String?, owner: String) {
            lines.add(text)
            calls.add("say:$text")
        }
        override fun stop(owner: String?) {
            lines.clear()
            calls.add("stop")
        }
        override fun dropWaiting(owner: String) {
            while (lines.size > 1) lines.removeAt(lines.size - 1)
            calls.add("drop")
        }
        override fun prepare(text: String) {
            calls.add("prepare:$text")
        }

        /** The line playing ends; the next starts (its `onPlay`). */
        fun end() {
            lines.removeAt(0)
            if (lines.isNotEmpty()) PanelVoice.onAudioStarts(lines[0])
        }
    }

    private lateinit var tts: PacedTts

    private fun draft(): String = PanelEngine.poll(0).getJSONObject("state").optString("coach_draft")

    @Before
    fun setUp() {
        PanelVoice.setMuted(false)
        PanelVoice.setVoicePace("simple")
        PanelVoice.setVoicePace("follow") // a fresh plan
        tts = PacedTts()
        PanelVoice.speaker = tts
        PanelVoice.positionMovedOn("a new game")
    }

    @After
    fun tearDown() {
        PanelVoice.setMuted(false)
        PanelVoice.speaker = null
        PanelVoice.setVoicePace("simple")
        PanelEngine.set("coach_draft" to "")
    }

    @Test
    fun theVoiceIsKeptOneLineAhead() {
        PanelVoice.positionMovedOn("the player moved", ply = 1, thinkS = 20.0)
        val sink = PanelVoice.streamSink(Trigger.POST_MOVE, 0)!!
        for (sentence in listOf("One.", "Two.", "Three.")) sink(sentence)
        // The first plays, the second is made behind it; the third waits in the plan.
        assertEquals(listOf("One.", "Two."), tts.lines.toList())
        tts.end() // "Two." plays: "Three." goes to the voice now, to be made meanwhile
        assertEquals(listOf("Two.", "Three."), tts.lines.toList())
    }

    @Test
    fun thePlayersMoveFinishesTheSentenceAndDropsWhatWaits() {
        PanelVoice.positionMovedOn("the player moved", ply = 1, thinkS = 20.0)
        val sink = PanelVoice.streamSink(Trigger.POST_MOVE, 0)!!
        sink("One.")
        sink("Two.")
        PanelVoice.positionMovedOn("the player moved", ply = 3, thinkS = 4.0)
        assertFalse("the sentence playing finishes", "stop" in tts.calls)
        assertEquals("the one waiting is no longer the news", listOf("One."), tts.lines.toList())
    }

    @Test
    fun aHintWhosePositionIsGoneIsCut() {
        val sink = PanelVoice.streamSink(Trigger.STUCK, 0)!!
        sink("Only one move holds.")
        assertEquals(listOf("Only one move holds."), tts.lines.toList())
        PanelVoice.positionMovedOn("the player moved", ply = 1, thinkS = 50.0)
        assertEquals("stop", tts.calls.last())
        assertTrue(tts.lines.isEmpty())
    }

    @Test
    fun aReplyStreamsIntoTheChatMutedOrNot() {
        PanelVoice.setMuted(true)
        val sink = PanelVoice.streamSink(Trigger.USER_ASK, 0)
        assertNotNull("muted, the reply is still written sentence by sentence", sink)
        sink!!("Play Qa8+.")
        assertEquals("Play Qa8+.", draft())
        sink("It deflects the rook.")
        assertEquals("Play Qa8+. It deflects the rook.", draft())
        assertFalse("muted: nothing said", tts.calls.any { it.startsWith("say:") })
    }

    @Test
    fun theSettingSwitchesToTheSimpleVoiceAndBack() {
        PanelVoice.setVoicePace("simple")
        assertNull(PanelVoice.pace)
        PanelVoice.setVoicePace("follow")
        assertNotNull(PanelVoice.pace)
    }
}
