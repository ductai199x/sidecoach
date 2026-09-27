package com.sidecoach.ui

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.events.GameStarted
import com.sidecoach.profile.Profiler
import com.sidecoach.profile.Store
import com.sidecoach.vision.Pieces
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The panel's controls, `tests/test_panel_engine.py`'s cases the phone had none for: a correction
 * through the API, the voice and mute controls, the panel starting with the settings' voice, and
 * listening outliving a session.
 */
@RunWith(AndroidJUnit4::class)
class PanelEngineTest {

    private val app = InstrumentationRegistry.getInstrumentation().targetContext

    private fun state(): JSONObject = PanelEngine.poll(Int.MAX_VALUE).getJSONObject("state")

    /** `test_a_correction_moves_through_the_api`. */
    @Test
    fun aCorrectionMovesThroughTheApi() {
        val before = PanelEngine.profiler
        PanelEngine.profiler = null
        try {
            assertFalse("no session, no profile", PanelEngine.correct("g", 0, 3, "time pressure"))
            val store = Store(null)
            val profiler = Profiler(store, sessionId = "s")
            profiler.onEvent(GameStarted("g", Pieces.Side.WHITE, STARTING_FEN), 0.0)
            store.addEvidence(7, "g", 0, 3, "opening", 0.12, "positional misjudgment", null, false)
            PanelEngine.profiler = profiler
            PanelEngine.set(
                "review" to JSONObject().put("game_id", "g").put(
                    "turning_points",
                    JSONArray().put(JSONObject().put("segment", 0).put("ply", 3).put("cause", "positional misjudgment")),
                ),
            )
            assertTrue(PanelEngine.correct("g", 0, 3, "time pressure"))
            assertEquals("time pressure", store.evidence("g")[0].cause)
            val point = state().getJSONObject("review").getJSONArray("turning_points").getJSONObject(0)
            assertEquals("time pressure", point.getString("cause"))
            assertThrows(IllegalArgumentException::class.java) { PanelEngine.correct("g", 0, 3, "bad luck") }
            store.close()
        } finally {
            PanelEngine.profiler = before
            PanelEngine.set("review" to null, "profile" to null)
        }
    }

    /**
     * `test_flip_and_voice_controls_do_not_raise_without_a_game` and
     * `test_the_panel_starts_with_the_settings_voice`: the saved voice is the one the panel
     * speaks with from the start (the field's old default always won, 2026-09-21); the voice
     * and mute controls take effect and are shown.
     */
    @Test
    fun thePanelStartsWithTheSettingsVoiceAndItsControlsTakeEffect() {
        val prefs = app.getSharedPreferences("settings", Context.MODE_PRIVATE)
        val saved = prefs.getString("json", null)
        val voice = PanelVoice.voice
        try {
            val settings = SettingsStore(app)
            val values = settings.values()
            values.getJSONObject("voice").put("voice", "bm_george")
            settings.save(values)
            PanelVoice.build(app, SettingsStore(app))
            assertEquals("bm_george", PanelVoice.voice)
            assertEquals("bm_george", PanelVoice.tts!!.voice)
            assertEquals("bm_george", state().getString("voice"))
            PanelVoice.setVoice("am_adam")
            assertEquals("am_adam", PanelVoice.voice)
            assertEquals("am_adam", PanelVoice.tts!!.voice)
            assertEquals("am_adam", state().getString("voice"))
            PanelVoice.setMuted(true)
            assertTrue(PanelVoice.muted && state().getBoolean("muted"))
            PanelVoice.setMuted(false)
            assertFalse(PanelVoice.muted || state().getBoolean("muted"))
        } finally {
            if (saved == null) prefs.edit().remove("json").commit() else prefs.edit().putString("json", saved).commit()
            PanelVoice.setVoice(voice)
        }
    }

    /**
     * `test_listening_outlives_a_session`: a session that ends leaves listening as it was (it
     * used to turn the listener off on its way out, and push-to-talk did nothing after a Stop
     * while the page still showed it on, 2026-09-23). The phone's own rule rides on it: the
     * microphone is open while the panel is in front or a session holds it.
     */
    @Test
    fun listeningOutlivesASession() {
        InstrumentationRegistry.getInstrumentation().uiAutomation
            .grantRuntimePermission(app.packageName, android.Manifest.permission.RECORD_AUDIO)
        PanelListen.build(app)
        try {
            assertEquals("ptt", PanelListen.setListen("ptt"))
            PanelListen.visible(false)
            PanelListen.sessionHolds(true) // Start
            assertTrue("a session holds the microphone", PanelListen.mic!!.isOpen)
            PanelListen.sessionHolds(false) // Stop
            assertEquals("the session closed the ear", "ptt", PanelListen.listener!!.mode)
            assertEquals("ptt", state().getString("listen"))
            assertFalse("behind another app, with no session, nothing is heard", PanelListen.mic!!.isOpen)
            PanelListen.visible(true)
            assertTrue(PanelListen.mic!!.isOpen)
        } finally {
            PanelListen.visible(true)
            PanelListen.setListen("off")
        }
        assertFalse(PanelListen.mic!!.isOpen)
    }
}
