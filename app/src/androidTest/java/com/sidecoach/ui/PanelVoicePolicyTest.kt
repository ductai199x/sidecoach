package com.sidecoach.ui

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
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

    @Test
    fun theVoiceHoldsItsWordInAFastGame() {
        assertEquals("text", act("post_move", "Qxf7#: mate.", 7, 7, false, null, true))
        assertEquals("text", act("post_mistake", "Bb7 was a miss.", 7, 7, false, null, true))
        assertEquals("an ask is answered aloud", "speak", act("user_ask", "Yes.", 7, 7, false, null, true))
        assertEquals("speak", act("review", "You won.", 7, 7, false, null, true))
        assertEquals("speak", act("post_move", "Qxf7#: mate.", 7, 7, false, null, false))
    }
}
